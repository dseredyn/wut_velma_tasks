#!/usr/bin/env python3
"""
example_node.py

Basic motions for WUT Velma.
Simple implementation of FSM.
"""

import numpy as np
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

from std_msgs.msg import Header
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import JointState

import PyKDL

from moveit_msgs.action import (MoveGroup, MoveGroup_GetResult_Response)
from moveit_msgs.msg import (Constraints, MoveItErrorCodes, WorkspaceParameters,
    MotionPlanRequest, CollisionObject, LinkPadding, PlanningScene, JointConstraint)

from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_msgs.msg import (PositionConstraint, OrientationConstraint, BoundingVolume)

# For gripper control
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient

from moveit_msgs.srv import (ApplyPlanningScene, ApplyPlanningScene_Request,
                ApplyPlanningScene_Response, GetCartesianPath, GetCartesianPath_Request,
                GetCartesianPath_Response)

from wut_velma_tasks.common import (JointStateThread, GazeboPoseThr, GrippersActionsInterface,
                                    ArmsTorsoActionInterface, vectorKDLtoROS, quaternionKDLtoROS,
                                    poseKDLtoROS, frameKdlToConstraints, tfROStoKDL,
                                    get_gripper_config, createConstraints, planningSceneTransforms)

def updateJointState(js_full:JointState, js_part:JointState, group_name:str) -> JointState:
    if group_name == 'right_arm_torso':
        joint_names = ["torso_0_joint",
                   "right_arm_0_joint", "right_arm_1_joint", "right_arm_2_joint",
                   "right_arm_3_joint", "right_arm_4_joint", "right_arm_5_joint",
                   "right_arm_6_joint"]
    elif group_name == 'left_arm':
        joint_names = ["left_arm_0_joint", "left_arm_1_joint", "left_arm_2_joint",
                   "left_arm_3_joint", "left_arm_4_joint", "left_arm_5_joint", "left_arm_6_joint"]
    else:
        raise Exception()
    result = js_full
    for name, pos in zip(js_part.name, js_part.position):
        if name in joint_names:
            for idx, name2 in enumerate(result.name):
                if name2 == name:
                    result.position[idx] = pos
                    break
    return result

def planningSceneRemoveObjects(objects:list[str]) -> PlanningScene:
    scene = PlanningScene()
    scene.is_diff = True
    scene.world.collision_objects = []
    for obj_name in objects:
        obj = CollisionObject()
        obj.id = obj_name
        obj.operation = CollisionObject.REMOVE
        scene.world.collision_objects.append(obj)
    return scene

def planningSceneAddObjects(obj_map:dict, obj_poses_map:dict[str,PyKDL.Frame]) -> PlanningScene:
    scene = PlanningScene()
    scene.is_diff = True
    scene.world.collision_objects = []
    for obj_name, primitives in obj_map.items():
        obj_pose = obj_poses_map[obj_name]
        for prim_name, prim, prim_pose in primitives:
            obj = CollisionObject()
            obj.header.frame_id = 'torso_base'
            obj.pose = poseKDLtoROS(obj_pose)
            obj.id = f'{obj_name}.{prim_name}'
            obj.primitives = [prim]
            obj.primitive_poses = [poseKDLtoROS(prim_pose)]
            obj.operation = CollisionObject.ADD
            scene.world.collision_objects.append(obj)

    # Add some padding
    padded_links = ['left_HandFingerOneKnuckleThreeLink', 'left_HandFingerTwoKnuckleThreeLink',
                    'left_HandFingerThreeKnuckleThreeLink',
                    'right_HandFingerOneKnuckleThreeLink', 'right_HandFingerTwoKnuckleThreeLink',
                    'right_HandFingerThreeKnuckleThreeLink',]
    scene.link_padding = []
    for link_name in padded_links:
        scene.link_padding.append( LinkPadding(link_name=link_name, padding=0.02) )

    return scene

def createPathConstraintsNearJs(js:JointState) -> Constraints:
    joint_names = ["torso_0_joint", "left_arm_0_joint", "left_arm_1_joint", "left_arm_2_joint",
                   "left_arm_3_joint", "left_arm_4_joint", "left_arm_5_joint", "left_arm_6_joint",
                   "right_arm_0_joint", "right_arm_1_joint", "right_arm_2_joint",
                   "right_arm_3_joint", "right_arm_4_joint", "right_arm_5_joint",
                   "right_arm_6_joint"]

    c = Constraints()
    c.joint_constraints = []
    for joint_name, position in zip(js.name, js.position):
        if not joint_name in joint_names:
            continue
        jc = JointConstraint(joint_name=joint_name, position=position,
                             tolerance_below=math.radians(30),
                             tolerance_above=math.radians(30),
                             weight=1.0)
        c.joint_constraints.append(jc)
    return c


def createPathConstraintsNearEE(pt_Er:PyKDL.Vector, pt_El:PyKDL.Vector, pt_r_B:PyKDL.Vector, pt_l_B:PyKDL.Vector) -> Constraints:
    c = Constraints()
    c.position_constraints = []
    pos_tol = 0.4
    c.position_constraints.append( PositionConstraint(header=Header(frame_id='torso_base'),
                                        link_name='right_arm_7_link',
                                        target_point_offset=vectorKDLtoROS(pt_Er),
                                        constraint_region=BoundingVolume(
                                            primitives=[SolidPrimitive(type=SolidPrimitive.BOX,
                                                                        dimensions=[pos_tol, pos_tol, pos_tol])],
                                            primitive_poses=[poseKDLtoROS(PyKDL.Frame(pt_r_B))],
                                        ),
                                        weight=1.0))

    return c


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
        self.arms_torso_int = ArmsTorsoActionInterface(self)


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

        self.cart_path_cli = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        while not self.cart_path_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service /compute_cartesian_path not available, waiting again...')

        # Wait for initialization
        if not self.move_group_action.wait_for_server(timeout_sec=10.0):
            raise Exception('MoveGroup is not available')
        # else:

        #
        # Initialize the FSM
        #
        self._FSM_next_state_timer = None

        self.get_logger().info(f'Initialization is done. Starting FSM.')

        # The initial state
        self.FSM_init(self.state_planning_scene)


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

    def state_prepare_gripper(self) -> FsmState:
        self.get_logger().info('Opening gripper')
        f1 = self.grippers_int.move_fingers_async('left', 0, 0, 0, math.radians(180), 2.0, math.radians(20))
        f2 = self.grippers_int.move_fingers_async('right', 0, 0, 0, math.radians(180), 2.0, math.radians(20))

        if not self.grippers_int.wait_for_result(f1):
            return self.state_failure
        # else:
        if not self.grippers_int.wait_for_result(f2):
            return self.state_failure
        # else:

        f_angle = math.radians(90)
        f1 = self.grippers_int.move_fingers_async('left', f_angle, f_angle, f_angle, math.radians(180), 2.0, math.radians(20))
        f2 = self.grippers_int.move_fingers_async('right', f_angle, f_angle, f_angle, math.radians(180), 2.0, math.radians(20))

        if not self.grippers_int.wait_for_result(f1):
            return self.state_failure
        # else:
        if not self.grippers_int.wait_for_result(f2):
            return self.state_failure
        # else:

        # Ok
        return self.state_move_to_table


    def state_planning_scene(self):
        object_names = ['can', 'table']
        obj_map = {
            'can':[('cyl', SolidPrimitive(type=SolidPrimitive.CYLINDER, dimensions=[0.2, 0.04]), PyKDL.Frame())],
            # table - one box:
            # [(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[1.5, 0.8, 1.015]),
            #                                         PyKDL.Frame(PyKDL.Vector(0, 0, 1.015/2)))] 
            # table - precise model
            'table':[
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
            ]}

        obj_poses_map = self.gz_pose_thr.getAllObjectsPoses()
        for obj_name in object_names:
            if not obj_name in obj_poses_map:
                return self.state_planning_scene

        req:ApplyPlanningScene_Request = ApplyPlanningScene.Request()
        req.scene = planningSceneAddObjects(obj_map, obj_poses_map)
        result:ApplyPlanningScene_Response|None = self.planning_scene_cli.call(req)
        if result is None:
            self.get_logger().error(f'Service call failed')
            return self.state_failure
        # else:
        self.get_logger().info(f'Service call completed (planning scene): {result.success}')

        return self.state_prepare_gripper


    def getWorldState(self):
        # ee_l_link_name = 'left_arm_7_link'
        ee_r_link_name = 'right_arm_7_link'
        f_link_name = 'right_HandFingerOneKnuckleThreeLink'

        T_B_E = self.get_frame_position(ee_r_link_name)
        if T_B_E is None:
            raise Exception(f'Could not get pose of the link "{ee_r_link_name}"')

        T_B_F = self.get_frame_position(f_link_name)
        if T_B_F is None:
            raise Exception(f'Could not get pose of the link "{f_link_name}"')

        T_E_F = T_B_E.Inverse() * T_B_F

        js = self.js_thr.get_JointState(self.get_clock().now(), 5)

        obj_name = 'table'
        T_B_O = self.gz_pose_thr.getObjectPose(obj_name)
        if T_B_O is None:
            raise Exception(f'Could not get pose of object "{obj_name}"')


        return T_E_F, js, T_B_O

    def state_move_to_table(self) -> FsmState:
        # Offsets in B frame
        # v1 = PyKDL.Vector(0.06, 0, 0.24)
        # v2 = PyKDL.Vector(0.25, 0, 0.45)
        # xz 0.5, 1.0
        # 0.56, 1.24
        # 0.75, 1.45

        # Initialize the task
        self.support_task_params = SupportTaskParams()            

        ee_l_link_name = 'left_arm_7_link'
        ee_r_link_name = 'right_arm_7_link'

        T_E_F, js, T_B_O = self.getWorldState()

        # Contact point on the finger tip wrt the end effector
        self.support_task_params.pt_f_contact_E = T_E_F * PyKDL.Vector(0.05, 0, 0)

        # The O frame is the base frame for orientation constraints
        # The free DOF is rotation around X axis in the O frame
        name_O_frame = 'frame_O'
        req:ApplyPlanningScene_Request = ApplyPlanningScene.Request()
        req.scene.is_diff = True
        req.scene = planningSceneTransforms({name_O_frame:T_B_O})
        result:ApplyPlanningScene_Response|None = self.planning_scene_cli.call(req)
        if result is None:
            self.get_logger().error(f'Service call failed')
            return self.state_failure

        # Use constraints for the fingertips
        # We want the fingertip of the middle finger to be at specific pose
        # position: right_HandFingerOneKnuckleThreeLink + (0.05, 0, 0) is at the contact point
        # orientation: right_HandFingerOneKnuckleThreeLink
        #   - central: RotX(-90) * RotZ(-70)
        #   - error tolerance: 5, 5, 30

        pt_contact_r_B = T_B_O * self.support_task_params.pt_contact_r_O
        pt_contact_l_B = T_B_O * self.support_task_params.pt_contact_l_O

        # M_contact_r_B = PyKDL.Rotation.RotZ(math.radians(-90)) * PyKDL.Rotation.RotX(math.radians(-45))
        # M_contact_r_O = T_B_O.Inverse().M * M_contact_r_B

        # M_contact_l_B = PyKDL.Rotation.RotZ(math.radians(-90)) * PyKDL.Rotation.RotX(math.radians(-45))
        # M_contact_l_O = T_B_O.Inverse().M * M_contact_l_B

        # Both end effectors can be rotated around local X by +- 30 deg

        # The medium-level control task:
        # - constrain a given point wrt. the end effector to a specified point wrt the base
        # - constrain orientation of the end effector, but allow some rotations around local X axis

        pos_constr, ori_constr = createConstraints([
            ('constrain local point', ee_r_link_name, self.support_task_params.pt_f_contact_E, pt_contact_r_B),
            ('constrain orientation', ee_r_link_name, name_O_frame, self.support_task_params.M_contact_r_O, math.radians(30), math.radians(1), math.radians(1)),
            ('constrain local point', ee_l_link_name, self.support_task_params.pt_f_contact_E, pt_contact_l_B),
            ('constrain orientation', ee_l_link_name, name_O_frame, self.support_task_params.M_contact_l_O, math.radians(30), math.radians(1), math.radians(1))])
        goal_constraints = [ Constraints(position_constraints=pos_constr,
                                        orientation_constraints=ori_constr) ]

        group_name = 'arms_torso'
        self.get_logger().info(
            f'Planning and execution: motion to a given end-effector pose for group "{group_name}"')
        goal = MoveGroup.Goal()
        goal.request = MotionPlanRequest(
            workspace_parameters = WorkspaceParameters(
                                        min_corner=Vector3(x=-2.0, y=-2.0, z=0.0),
                                        max_corner=Vector3(x=-2.0, y=-2.0, z=0.0)),
            goal_constraints = goal_constraints,
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
                return self.state_support
            else:
                self.get_logger().error(f'error: status={response.status}, '
                                        f'error_code={response.result.error_code.val}, '
                                        f'error_string={response.result.error_code.message}')
                return self.state_failure

    def state_support(self) -> FsmState:
        # "support" task:
        # - table top is not in the planning scene
        # - parameters:
        #   - support point height (common for both arms, depends on table other end height)
        #     - actual value (mean height of "support" points of the table)
        #     - equilibrium value (higher due to gravity)
        #   - left x, y (depends on table pose)
        #   - right x, y (depends on table pose)

        T_E_F, js, T_B_O = self.getWorldState()

        self.support_task_params.height_reference = (T_B_O * self.support_task_params.pt_O).z()

        pt_contact_l_B = T_B_O * self.support_task_params.pt_contact_l_O
        pt_contact_r_B = T_B_O * self.support_task_params.pt_contact_r_O
        pt_contact_mean_B = (pt_contact_l_B + pt_contact_r_B) / 2
        self.support_task_params.height_current = pt_contact_mean_B.z()

        self.support_task_params.pt_l_current_B = pt_contact_l_B
        self.support_task_params.pt_r_current_B = pt_contact_r_B

        if self.support_task_params.height_eq is None:
            # Initialize
            self.support_task_params.height_eq = self.support_task_params.height_reference

        height_tol = 0.01
        max_height_change = 0.02
        max_height_dist = 0.2

        if self.support_task_params.height_current < self.support_task_params.height_reference - height_tol:
            # Slowly raise the equilibrium point, if too low

            if self.support_task_params.height_eq < self.support_task_params.height_reference:
                self.support_task_params.height_eq = self.support_task_params.height_reference
            self.support_task_params.height_eq += max_height_change
            if self.support_task_params.height_eq > self.support_task_params.height_reference + max_height_dist:
                # Limit the value
                self.support_task_params.height_eq = self.support_task_params.height_reference + max_height_dist

        if self.support_task_params.height_current > self.support_task_params.height_reference + height_tol:
            # Slowly lower the equilibrium point, if too high
            self.support_task_params.height_eq -= max_height_change
            if self.support_task_params.height_eq < self.support_task_params.height_reference:
                # Limit the value
                self.support_task_params.height_eq = self.support_task_params.height_reference

        self.get_logger().info(f'height: reference: {self.support_task_params.height_reference:.2f},'
                               f' current: {self.support_task_params.height_current:.2f},'
                               f' equilibrium: {self.support_task_params.height_eq:.2f}')

        # Apply the commands
        ee_l_link_name = 'left_arm_7_link'
        ee_r_link_name = 'right_arm_7_link'

        pt_contact_l_B_cmd = PyKDL.Vector(pt_contact_l_B.x(), pt_contact_l_B.y(), self.support_task_params.height_eq)
        pt_contact_r_B_cmd = PyKDL.Vector(pt_contact_r_B.x(), pt_contact_r_B.y(), self.support_task_params.height_eq)

        # The O frame is the base frame for orientation constraints
        # The free DOF is rotation around X axis in the O frame
        name_O_frame = 'frame_O'
        req:ApplyPlanningScene_Request = ApplyPlanningScene.Request()
        req.scene = planningSceneTransforms({name_O_frame:T_B_O})
        result:ApplyPlanningScene_Response|None = self.planning_scene_cli.call(req)
        if result is None:
            self.get_logger().error(f'Service call failed')
            return self.state_failure

        req:ApplyPlanningScene_Request = ApplyPlanningScene.Request()
        req.scene = planningSceneRemoveObjects(['table.top'])
        result:ApplyPlanningScene_Response|None = self.planning_scene_cli.call(req)
        if result is None:
            self.get_logger().error(f'Service call failed')
            return self.state_failure

        # pos_constr, ori_constr = createConstraints([
        #     ('constrain local point', ee_r_link_name, self.support_task_params.pt_f_contact_E, pt_contact_r_B_cmd),
        #     ('constrain orientation', ee_r_link_name, name_O_frame, self.support_task_params.M_contact_r_O, math.radians(10), math.radians(1), math.radians(1)),
        #     ('constrain local point', ee_l_link_name, self.support_task_params.pt_f_contact_E, pt_contact_l_B_cmd),
        #     ('constrain orientation', ee_l_link_name, name_O_frame, self.support_task_params.M_contact_l_O, math.radians(10), math.radians(1), math.radians(1))])
        # goal_constraints = [ Constraints(position_constraints=pos_constr,
        #                                 orientation_constraints=ori_constr) ]

        # # path_constraints = createPathConstraintsNearJs(js)
        # path_constraints = createPathConstraintsNearEE(self.support_task_params.pt_f_contact_E,
        #                                                self.support_task_params.pt_f_contact_E,
        #                                                pt_contact_r_B_cmd,
        #                                                pt_contact_l_B_cmd)
        # group_name = 'arms_torso'
        # self.get_logger().info(
        #     f'Planning and execution: motion to a given end-effector pose for group "{group_name}"')
        # goal = MoveGroup.Goal()
        # goal.request = MotionPlanRequest(
        #     workspace_parameters = WorkspaceParameters(
        #                                 min_corner=Vector3(x=-2.0, y=-2.0, z=0.0),
        #                                 max_corner=Vector3(x=-2.0, y=-2.0, z=0.0)),
        #     goal_constraints = goal_constraints,
        #     path_constraints = path_constraints,
        #     group_name = group_name,
        #     allowed_planning_time = 15.0,
        #     max_velocity_scaling_factor = 0.3,
        #     max_acceleration_scaling_factor = 0.3)

        # if not self.support_task_params.js_previous is None:
        #     goal.request.start_state.joint_state = self.support_task_params.js_previous
        # goal.request.num_planning_attempts = 10
        # goal.request.allowed_planning_time = 15.0

        # goal.planning_options.replan_attempts = 10
        # goal.planning_options.replan = True

        # Waits until goal is accepted
        # response:MoveGroup_GetResult_Response|None = self.move_group_action.send_goal(goal)
        # if response is None:
        #     self.get_logger().error(f'result: None')
        #     return self.state_failure
        # else:
        #     assert isinstance(response, MoveGroup_GetResult_Response)

        #     if (response.status == GoalStatus.STATUS_SUCCEEDED
        #             and response.result.error_code.val == MoveItErrorCodes.SUCCESS):
        #         self.get_logger().info(f'succeeded')
        #         last_q:JointTrajectoryPoint = response.result.planned_trajectory.joint_trajectory.points[-1]

        #         js = JointState()
        #         js.name = response.result.planned_trajectory.joint_trajectory.joint_names
        #         js.position = last_q.positions
        #         self.support_task_params.js_previous = js
        #         return self.state_support
        #     else:
        #         self.get_logger().error(f'error: status={response.status}, '
        #                                 f'error_code={response.result.error_code.val}, '
        #                                 f'error_string={response.result.error_code.message}')
        #         return self.state_failure

        # The contact frame C wrt E
        T_E_C = T_E_F * PyKDL.Frame(PyKDL.Vector(0.05, 0, 0))

        T_O_Cr = PyKDL.Frame(PyKDL.Rotation.RotY(math.radians(-90)) * PyKDL.Rotation.RotZ(math.radians(30)), self.support_task_params.pt_contact_r_O)
        T_B_Cr = T_B_O * T_O_Cr
        T_B_Cr = PyKDL.Frame(T_B_Cr.M, PyKDL.Vector(T_B_Cr.p.x(), T_B_Cr.p.y(), self.support_task_params.height_eq))
        T_B_Er_d = T_B_Cr * T_E_C.Inverse()

        T_O_Cl = PyKDL.Frame(PyKDL.Rotation.RotY(math.radians(-90)) * PyKDL.Rotation.RotZ(math.radians(30)), self.support_task_params.pt_contact_l_O)
        T_B_Cl = T_B_O * T_O_Cl
        T_B_Cl = PyKDL.Frame(T_B_Cl.M, PyKDL.Vector(T_B_Cl.p.x(), T_B_Cl.p.y(), self.support_task_params.height_eq))
        T_B_El_d = T_B_Cl * T_E_C.Inverse()

        # right
        req:GetCartesianPath_Request = GetCartesianPath.Request()

        if self.support_task_params.js_previous is None:
            self.support_task_params.js_previous = js

        req.start_state.joint_state = self.support_task_params.js_previous
        req.start_state.is_diff = False
        req.group_name = 'right_arm'
        req.link_name = 'right_arm_7_link'
        req.waypoints = [poseKDLtoROS(T_B_Er_d)]
        req.max_step = 0.01
        req.avoid_collisions = True
        req.cartesian_speed_limited_link = req.link_name
        req.max_cartesian_speed = 0.1
        req.max_velocity_scaling_factor = 0.4
        req.max_acceleration_scaling_factor = 0.4

        result:GetCartesianPath_Response = self.cart_path_cli.call(req)
        if result is None:
            self.get_logger().error(f'Service call failed')
            return self.state_failure
        # else:

        self.get_logger().info(f'fraction: {result.fraction:.2f}')

        js = JointState()
        js.name = result.solution.joint_trajectory.joint_names
        last_q:JointTrajectoryPoint = result.solution.joint_trajectory.points[-1]
        js.position = last_q.positions

        self.support_task_params.js_previous = updateJointState(self.support_task_params.js_previous, js, 'right_arm_torso')
        traj_r = result.solution.joint_trajectory

        # left
        req:GetCartesianPath_Request = GetCartesianPath.Request()

        if not self.support_task_params.js_previous is None:
            req.start_state.joint_state = self.support_task_params.js_previous
        else:
            req.start_state.joint_state = js
        req.start_state.is_diff = False
        req.group_name = 'left_arm'
        req.link_name = 'left_arm_7_link'
        req.waypoints = [poseKDLtoROS(T_B_El_d)]
        req.max_step = 0.01
        req.avoid_collisions = True
        req.cartesian_speed_limited_link = req.link_name
        req.max_cartesian_speed = 0.1
        req.max_velocity_scaling_factor = 0.4
        req.max_acceleration_scaling_factor = 0.4

        result:GetCartesianPath_Response = self.cart_path_cli.call(req)
        if result is None:
            self.get_logger().error(f'Service call failed')
            return self.state_failure
        # else:

        self.get_logger().info(f'fraction: {result.fraction:.2f}')

        js = JointState()
        js.name = result.solution.joint_trajectory.joint_names
        last_q:JointTrajectoryPoint = result.solution.joint_trajectory.points[-1]
        js.position = last_q.positions
        self.support_task_params.js_previous = updateJointState(self.support_task_params.js_previous, js, 'left_arm')
        traj_l = result.solution.joint_trajectory

        self.arms_torso_int.move(traj_r, 0.1)
        self.arms_torso_int.move(traj_l, 0.1)


        return self.state_support

    # def state_flip_table2(self) -> FsmState:
    #     # Offsets in B frame
    #     v1 = PyKDL.Vector(0.06, 0, 0.24)
    #     v2 = PyKDL.Vector(0.25, 0, 0.45)
    #     # xz 0.5, 1.0
    #     # 0.56, 1.24
    #     # 0.75, 1.45

    #     link_name = 'right_arm_7_link'
    #     T_B_E = self.get_frame_position(link_name)
    #     if T_B_E is None:
    #         return self.state_failure
    #     js = self.js_thr.get_JointState(self.get_clock().now(), 5)
    #     if js is None:
    #         return self.state_failure

    #     T_B_E1 = PyKDL.Frame(v1) * T_B_E
    #     T_B_E2 = PyKDL.Frame(v2) * T_B_E
    #     req:GetCartesianPath_Request = GetCartesianPath.Request()
    #     req.start_state.joint_state = js
    #     req.start_state.is_diff = False
    #     req.group_name = 'right_arm'
    #     req.link_name = link_name
    #     req.waypoints = [poseKDLtoROS(T_B_E1), poseKDLtoROS(T_B_E2)]
    #     req.max_step = 0.01
    #     req.avoid_collisions = True
    #     req.cartesian_speed_limited_link = link_name
    #     req.max_cartesian_speed = 0.1
    #     req.max_velocity_scaling_factor = 0.4
    #     req.max_acceleration_scaling_factor = 0.4

    #     result:GetCartesianPath_Response = self.cart_path_cli.call(req)
    #     if result is None:
    #         self.get_logger().error(f'Service call failed')
    #         return self.state_failure
    #     # else:

    #     self.arms_torso_int.move(result.solution.joint_trajectory, 0.1)

    #     return self.state_finish

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


class SupportTaskParams:
    def __init__(self):
        # Position of a contact point wrt O
        self.pt_contact_r_O = PyKDL.Vector(0.5, -0.35, 0.95)
        self.pt_contact_l_O = PyKDL.Vector(-0.5, -0.35, 0.95)

        # Orientation of E wrt O
        self.M_contact_r_O = PyKDL.Rotation.RotX(math.radians(-45))
        self.M_contact_l_O = PyKDL.Rotation.RotX(math.radians(-45))

        # Point on the other end of the table - for height reference
        self.pt_O = PyKDL.Vector(0, 0.35, 0.95)

        # Contact point on the robot, expressed in E
        self.pt_f_contact_E = None

        self.height_reference = None
        self.height_current = None
        # self.lift_val = None
        self.height_eq = None
        self.pt_l_current_B = None
        self.pt_r_current_B = None

        # Previous js, used to continue motion
        self.js_previous = None


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
