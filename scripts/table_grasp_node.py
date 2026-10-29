#!/usr/bin/env python3
"""
example_node.py

Basic motions for WUT Velma.
Simple implementation of FSM.
"""

from typing import Optional
from collections.abc import Callable

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from threading import Lock

from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformBroadcaster
from tf2_ros import TransformException # type: ignore

import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Vector3

import PyKDL

from moveit_msgs.action import (MoveGroup, MoveGroup_GetResult_Response)
from moveit_msgs.msg import (Constraints, MoveItErrorCodes, WorkspaceParameters,
    MotionPlanRequest, CollisionObject, LinkPadding)

from shape_msgs.msg import SolidPrimitive

# For gripper control
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient

from moveit_msgs.srv import ApplyPlanningScene, ApplyPlanningScene_Request, ApplyPlanningScene_Response

from wut_velma_tasks.common import (JointStateThread, GazeboPoseThr, GrippersActionsInterface,
                                    poseKDLtoROS, frameKdlToConstraints, tfROStoKDL,
                                    get_gripper_config)


#
#
#
class TaskNode(Node):
    FsmState = Callable[[], Callable|None]

    def __init__(self) -> None:
        super().__init__("task_node")

        self.fsm_group = MutuallyExclusiveCallbackGroup()
        self.action_group = MutuallyExclusiveCallbackGroup()

        self.js_thr = JointStateThread(self)

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        #
        # Grippers actions
        #
        self.grippers_int = GrippersActionsInterface(self)

        self.gz_pose_thr = GazeboPoseThr(self, ['can', 'table'])

        # ApplyPlanningScene service client
        self.planning_scene_cli = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        while not self.planning_scene_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service /apply_planning_scene not available, waiting again...')
    
        #
        # MoveGroup actions
        #
        self.move_group_action = ActionClient( self, MoveGroup,
                                                "/move_action",
                                                 callback_group=self.action_group )

        #
        # Initialize the FSM
        #
        self._FSM_next_state_timer = None

        self.get_logger().info(f'Initialization is done. Starting FSM.')

        # The initial state
        self.FSM_init(self.state_prepare_gripper)


    def FSM_init(self, state_func:FsmState):
        """Set FSM initial state."""
        self._FSM_next(state_func)


    def _FSM_next(self, state_func:FsmState):
        """Internal function for FSM. Do not call it directly."""
        assert self._FSM_next_state_timer is None
        self._FSM_next_state_timer = self.create_timer(0.1,
                                                        lambda: self._FSM_next_timer_cb(state_func),
                                                        callback_group=self.fsm_group)


    def _FSM_next_timer_cb(self, state_func:FsmState):
        """Internal function for FSM. Do not call it directly."""
        assert not self._FSM_next_state_timer is None
        self._FSM_next_state_timer.cancel()
        self._FSM_next_state_timer = None

        # Call the actual state function
        assert not state_func is None
        assert callable(state_func)
        self.get_logger().info(f'FSM transition to: "{state_func.__name__}".')
        next_state_func = state_func()
        if not next_state_func is None:
            # Trigger state transition
            self._FSM_next( next_state_func )


    #
    # User-defined FSM states
    #
    def state_failure(self) -> FsmState:
        return self.state_finish


    def state_finish(self) -> None:
        try:
            rclpy.try_shutdown()
        except:
            pass


    def state_planning_scene(self):
        object_names = ['can', 'table']
        object_primitives = [
            [('cyl', SolidPrimitive(type=SolidPrimitive.CYLINDER, dimensions=[0.2, 0.04]), PyKDL.Frame())],
            # table - one box:
            # [(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[1.5, 0.8, 1.015]),
            #                                         PyKDL.Frame(PyKDL.Vector(0, 0, 1.015/2)))] 
            # table - precise model
            [
                ('top', SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[1.5, 0.8, 0.03]),
                                                    PyKDL.Frame(PyKDL.Vector(0, 0, 1))),
                ('leg1', SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.04, 0.04, 1.0]),
                                                    PyKDL.Frame(PyKDL.Vector(0.68, 0.33, 0.5))),
                ('leg2', SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.04, 0.04, 1.0]),
                                                    PyKDL.Frame(PyKDL.Vector(0.68, -0.33, 0.5))),
                ('leg3', SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.04, 0.04, 1.0]),
                                                    PyKDL.Frame(PyKDL.Vector(-0.68, -0.33, 0.5))),
                ('leg4', SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.04, 0.04, 1.0]),
                                                    PyKDL.Frame(PyKDL.Vector(-0.68, 0.33, 0.5))),
            ]
        ]
        
        req:ApplyPlanningScene_Request = ApplyPlanningScene.Request()
        req.scene.is_diff = True
        req.scene.world.collision_objects = []
        for obj_name, primitives in zip(object_names, object_primitives):
            for prim_name, prim, prim_pose in primitives:
                pose = self.gz_pose_thr.getObjectPose(obj_name)
                if pose is None:
                    self.get_logger().info(f'Could not get pose of object "{obj_name}"')
                    return self.state_planning_scene
                    # continue
                obj = CollisionObject()
                obj.header.frame_id = 'torso_base'
                obj.pose = poseKDLtoROS(pose)
                obj.id = f'{obj_name}.{prim_name}'
                obj.primitives = [prim]
                obj.primitive_poses = [poseKDLtoROS(prim_pose)]
                obj.operation = CollisionObject.ADD
                req.scene.world.collision_objects.append(obj)
        # Add some padding
        padded_links = ['left_HandFingerOneKnuckleThreeLink', 'left_HandFingerTwoKnuckleThreeLink',
                        'left_HandFingerThreeKnuckleThreeLink',
                        'right_HandFingerOneKnuckleThreeLink', 'right_HandFingerTwoKnuckleThreeLink',
                        'right_HandFingerThreeKnuckleThreeLink',]
        req.scene.link_padding = []
        for link_name in padded_links:
            req.scene.link_padding.append( LinkPadding(link_name=link_name, padding=0.02) )

        result:ApplyPlanningScene_Response = self.planning_scene_cli.call(req)
        self.get_logger().info(f'Service call completed (planning scene): {result.success}')

        # This code calculates T_O_E
        # T_B_El = PyKDL.Frame( PyKDL.Rotation.RotZ(math.radians(-90))*PyKDL.Rotation.RotX(math.radians(-45)),
        #                         PyKDL.Vector(0.27, 0.48, 1.05))

        # T_B_Er = PyKDL.Frame( PyKDL.Rotation.RotZ(math.radians(-90))*PyKDL.Rotation.RotX(math.radians(-45)),
        #                         PyKDL.Vector(0.27, -0.48, 1.05))

        # T_B_O = PyKDL.Frame( PyKDL.Rotation.Quaternion(0, 0, -0.7071080798594739, 0.7071054825112361),
        #                         PyKDL.Vector(0.85, 0, 0))

        # T_O_El = T_B_O.Inverse() * T_B_El
        # T_O_Er = T_B_O.Inverse() * T_B_Er

        # self.get_logger().info(f'T_O_El = {frameToStr(T_O_El)}')
        # self.get_logger().info(f'T_O_Er = {frameToStr(T_O_Er)}')

        return self.state_move_to_table


    def state_move_to_table(self) -> FsmState:
        if not self.move_group_action.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("MoveGroup is not available")
            return self.state_failure
        # else:

        obj_name = 'table'
        T_B_O = self.gz_pose_thr.getObjectPose(obj_name)
        if T_B_O is None:
            self.get_logger().info(f'Could not get pose of object "{obj_name}"')
            return self.state_move_to_table

        # Information: where to grasp the table
        T_O_El = PyKDL.Frame(PyKDL.Rotation.Quaternion(-0.3826834323644443, -7.028373685184732e-07,
                                                       1.696799507219915e-06, 0.9238795325097287),
                             PyKDL.Vector(-0.48, -0.6, 1.05))
        T_O_Er = PyKDL.Frame(PyKDL.Rotation.Quaternion(-0.3826834323644443, -7.028373685184732e-07,
                                                       1.696799507219915e-06, 0.9238795325097287),
                             PyKDL.Vector(0.48, -0.6, 1.05))

        T_B_El = T_B_O * T_O_El
        T_B_Er = T_B_O * T_O_Er

        group_name = 'arms_torso'

        self.get_logger().info(
            f'Planning and execution: motion to a given end-effector pose for group "{group_name}"'
        )

        l_pos_constr, l_ori_constr = frameKdlToConstraints(T_B_El, 'left_arm_7_link')
        r_pos_constr, r_ori_constr = frameKdlToConstraints(T_B_Er, 'right_arm_7_link')

        goal = MoveGroup.Goal()
        goal.request = MotionPlanRequest(
            workspace_parameters = WorkspaceParameters(
                                        min_corner=Vector3(x=-2.0, y=-2.0, z=0.0),
                                        max_corner=Vector3(x=-2.0, y=-2.0, z=0.0)),
            goal_constraints = [Constraints(position_constraints=l_pos_constr + r_pos_constr,
                                            orientation_constraints=l_ori_constr + r_ori_constr)],
            group_name = group_name,
            allowed_planning_time = 15.0,
            max_velocity_scaling_factor = 0.3,
            max_acceleration_scaling_factor = 0.3)

        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 15.0

        goal.planning_options.replan_attempts = 10
        goal.planning_options.replan = True

        # Waits until goal is accepted
        response:MoveGroup_GetResult_Response|None = self.move_group_action.send_goal(goal)
        if response is None:
            self.get_logger().error(f'result: None')
            return self.state_failure
        else:
            assert isinstance(response, MoveGroup_GetResult_Response)

            if (response.status == GoalStatus.STATUS_SUCCEEDED
                    and response.result.error_code.val == MoveItErrorCodes.SUCCESS):
                self.get_logger().info(f'succeeded')
                return self.state_close_gripper
            else:
                self.get_logger().error(f'error: status={response.status}, '
                                        f'error_code={response.result.error_code.val}, '
                                        f'error_string={response.result.error_code.message}')
                return self.state_failure


    def state_close_gripper(self) -> FsmState:
        # Information: gripper configuration for table support

        finger_angle = math.radians(90)
        spread_angle = math.radians(180)
        f_add = math.radians(5)
        for side in ['left', 'right']:
            self.get_logger().info( f'Closing gripper {side} to touch an object' )
            # Detect an object.
            # Close fingers with a very small tolerance.
            # If the first motion failed, the object is touched.
            if not self.grippers_int.move_fingers(side, finger_angle, finger_angle, finger_angle, spread_angle, 3.0, math.radians(5)):
                self.get_logger().info(f'The object is touched')
            else:
                self.get_logger().info(f'No object is detected')

            q = self.js_thr.get_joint_state(self.get_clock().now(), 5)
            if q is None:
                self.get_logger().error(f'Could not get joint state')
                return self.state_failure
            # else:
            f1, f2, f3, spread = get_gripper_config(side, q)

            # Close the gripper once again, this time with larger tolerance.
            self.get_logger().info( f'Closing gripper {side} to grasp an object' )
            if not self.grippers_int.move_fingers(side, f1+f_add, f2+f_add, f3+f_add, spread_angle, 3.0, math.radians(45)):
                return self.state_failure
        # Ok
        return self.state_finish


    def state_prepare_gripper(self) -> FsmState:
        self.get_logger().info('Opening gripper')
        if not self.grippers_int.move_fingers('left', 0, 0, 0, math.radians(180), 3.0, math.radians(20)):
            return self.state_failure
        # else:
        if not self.grippers_int.move_fingers('right', 0, 0, 0, math.radians(180), 3.0, math.radians(20)):
            return self.state_failure
        # Ok
        return self.state_planning_scene

    #
    # TF
    #
    def get_frame_position(self, frame_name:str, base_frame:str='torso_base') -> Optional[PyKDL.Frame]:
        try:
            transform = self.tf_buffer.lookup_transform(
                base_frame,
                frame_name,
                Time(),
            )
        except TransformException as ex:
            self.get_logger().debug(
                f"No TF {base_frame} -> {frame_name}: {ex}"
            )
            return None

        # Ok
        return tfROStoKDL(transform.transform)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TaskNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)    
    try:
        executor.spin()
    except KeyboardInterrupt:
        print('User interrupt')
        pass

    executor.shutdown()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
