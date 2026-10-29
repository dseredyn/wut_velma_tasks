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

import time
from sensor_msgs.msg import JointState

from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformBroadcaster
from tf2_ros import TransformException # type: ignore

import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Pose, Transform, TransformStamped, Vector3

import PyKDL

from std_msgs.msg import Header
from moveit_msgs.action import (MoveGroup, MoveGroup_GetResult_Response)
from moveit_msgs.msg import (Constraints, JointConstraint, MoveItErrorCodes, WorkspaceParameters,
    MotionPlanRequest, PositionConstraint, OrientationConstraint, BoundingVolume)

from shape_msgs.msg import SolidPrimitive

# For gripper control
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory, FollowJointTrajectory_GetResult_Response
from rclpy.action import ActionClient
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import JointTolerance

#
# Helper functions
#
def poseKDLtoROS(T:PyKDL.Frame) -> Pose:
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = T.p.x(), T.p.y(), T.p.z()
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = T.M.GetQuaternion()
    return pose


def tfKDLtoROS(T:PyKDL.Frame) -> Transform:
    t = Transform()
    t.translation.x, t.translation.y, t.translation.z = T.p.x(), T.p.y(), T.p.z()
    t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w = T.M.GetQuaternion()
    return t


def tfROStoKDL(t_ros:Transform) -> PyKDL.Frame:
    return PyKDL.Frame( PyKDL.Rotation.Quaternion(  t_ros.rotation.x, t_ros.rotation.y,
                                                    t_ros.rotation.z, t_ros.rotation.w),
                    PyKDL.Vector(t_ros.translation.x, t_ros.translation.y, t_ros.translation.z))


#
#
#
class GraspingNode(Node):
    FsmState = Callable[[], Callable|None]

    def __init__(self) -> None:
        super().__init__("example_node")

        self.fsm_group = MutuallyExclusiveCallbackGroup()
        self.state_group = MutuallyExclusiveCallbackGroup()
        self.action_group = MutuallyExclusiveCallbackGroup()

        #
        # Initialize subscribers
        #
        now = self.get_clock().now()
        self._clock_type = now.clock_type
        self._state_lock = Lock()
        self._current_joint_positions = None
        self.sub_joint_state = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
            callback_group=self.state_group
        )

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        #
        # Grippers actions
        #
        self.hand_left_client = ActionClient( self, FollowJointTrajectory,
                                                "/left_hand_controller/follow_joint_trajectory",
                                                 callback_group=self.action_group )
        self.hand_right_client = ActionClient( self, FollowJointTrajectory,
                                                "/right_hand_controller/follow_joint_trajectory",
                                                 callback_group=self.action_group )

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
        self.FSM_init(self.state_move_high)


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


    def state_move_high(self) -> FsmState:
        if not self.move_group_action.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("MoveGroup is not available")
            return self.state_failure
        # else:

        group_name = 'right_arm_torso'
        js = self.get_joint_state(self.get_clock().now(), 5)
        if js is None:
            self.get_logger().error(f'Could not get joint_states')
            return self.state_failure

        tol = math.radians(2)
        js_goal = js.copy()
        js_goal['right_arm_1_joint'] = 0.0
        joint_constraints = [ JointConstraint(joint_name=joint_name, position=js_goal[joint_name],
                                              tolerance_above=tol, tolerance_below=tol, weight=1.0)
                                                                        for joint_name in js_goal]

        goal = MoveGroup.Goal()
        goal.request = MotionPlanRequest(
            workspace_parameters = WorkspaceParameters(
                                        min_corner=Vector3(x=-2.0, y=-2.0, z=0.0),
                                        max_corner=Vector3(x=-2.0, y=-2.0, z=0.0)),
            goal_constraints = [Constraints(joint_constraints=joint_constraints)],
            group_name = group_name,
            allowed_planning_time = 5.0,
            max_velocity_scaling_factor = 0.3,
            max_acceleration_scaling_factor = 0.3)

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
                return self.state_move_pregrasp
            else:
                self.get_logger().error(f'error: status={response.status}, '
                                        f'error_code={response.result.error_code.val}, '
                                        f'error_string={response.result.error_code.message}')
                return self.state_failure


    def state_move_pregrasp(self) -> FsmState:
        if not self.move_group_action.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("MoveGroup is not available")
            return self.state_failure
        # else:

        group_name = 'right_arm_torso'
        tip_link = 'right_arm_7_link'

        # Object's position is hardcoded
        obj_pt_B = PyKDL.Vector(0.6, -0.3, 1.1150)
        vec_pregrasp_B = PyKDL.Vector(0,0,0.35)

        T_B_Gr_des = PyKDL.Frame(
            PyKDL.Rotation.RotZ(math.radians(-90)) * PyKDL.Rotation.RotY(math.radians(180)),
            obj_pt_B + vec_pregrasp_B)

        # Visualize the calculated pose in TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'torso_base'
        t.child_frame_id = 'T_B_Gr_des'
        t.transform = tfKDLtoROS(T_B_Gr_des)
        self.tf_broadcaster.sendTransform(t)

        self.get_logger().info(
            f'Planning and execution: motion to a given end-effector pose for group "{group_name}"'
        )

        # Get a constant transformation between right hand frame and right end effector frame
        T_Gr_E = self.get_frame_position(tip_link, base_frame='right_HandGripLink')
        if T_Gr_E is None:
            # Retry
            return self.state_move_pregrasp

        # Calculate desired pose of right hand frame
        T_B_Er_des = T_B_Gr_des * T_Gr_E

        pose = poseKDLtoROS(T_B_Er_des)
        position_constraints = [ PositionConstraint(header=Header(frame_id='torso_base'),
                                                    link_name=tip_link,
                                                    constraint_region=BoundingVolume(
                                                        primitives=[SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01])],
                                                        primitive_poses=[pose],
                                                    ),
                                                    weight=1.0) ]

        ori_tol = math.radians(1)
        orientation_constraints = [ OrientationConstraint(header=Header(frame_id='torso_base'),
                                                          link_name=tip_link,
                                                          orientation=pose.orientation,
                                                          absolute_x_axis_tolerance=ori_tol,
                                                          absolute_y_axis_tolerance=ori_tol,
                                                          absolute_z_axis_tolerance=ori_tol,
                                                          parameterization=OrientationConstraint.ROTATION_VECTOR) ]

        goal = MoveGroup.Goal()
        goal.request = MotionPlanRequest(
            workspace_parameters = WorkspaceParameters(
                                        min_corner=Vector3(x=-2.0, y=-2.0, z=0.0),
                                        max_corner=Vector3(x=-2.0, y=-2.0, z=0.0)),
            goal_constraints = [Constraints(position_constraints=position_constraints,
                                            orientation_constraints=orientation_constraints)],
            group_name = group_name,
            allowed_planning_time = 5.0,
            max_velocity_scaling_factor = 0.3,
            max_acceleration_scaling_factor = 0.3)

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
                return self.state_move_grasp
            else:
                self.get_logger().error(f'error: status={response.status}, '
                                        f'error_code={response.result.error_code.val}, '
                                        f'error_string={response.result.error_code.message}')
                return self.state_failure


    def state_move_grasp(self) -> FsmState:
        if not self.move_group_action.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("MoveGroup is not available")
            return self.state_failure
        # else:

        group_name = 'right_arm_torso'
        tip_link = 'right_arm_7_link'

        # Object's position is hardcoded
        obj_pt_B = PyKDL.Vector(0.6, -0.3, 1.1150)
        vec_grasp_B = PyKDL.Vector(0,0, 0.15)

        T_B_Gr_des = PyKDL.Frame(
            PyKDL.Rotation.RotZ(math.radians(-90)) * PyKDL.Rotation.RotY(math.radians(180)),
            obj_pt_B + vec_grasp_B)

        # Visualize the calculated pose in TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'torso_base'
        t.child_frame_id = 'T_B_Gr_des'
        t.transform = tfKDLtoROS(T_B_Gr_des)
        self.tf_broadcaster.sendTransform(t)

        self.get_logger().info(
            f'Planning and execution: motion to a given end-effector pose for group "{group_name}"'
        )

        # Get a constant transformation between right hand frame and right end effector frame
        T_Gr_E = self.get_frame_position(tip_link, base_frame='right_HandGripLink')
        if T_Gr_E is None:
            # Retry
            return self.state_move_grasp

        # Calculate desired pose of right hand frame
        T_B_Er_des = T_B_Gr_des * T_Gr_E

        pose = poseKDLtoROS(T_B_Er_des)
        position_constraints = [ PositionConstraint(header=Header(frame_id='torso_base'),
                                                    link_name=tip_link,
                                                    constraint_region=BoundingVolume(
                                                        primitives=[SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01])],
                                                        primitive_poses=[pose],
                                                    ),
                                                    weight=1.0) ]

        ori_tol = math.radians(1)
        orientation_constraints = [ OrientationConstraint(header=Header(frame_id='torso_base'),
                                                          link_name=tip_link,
                                                          orientation=pose.orientation,
                                                          absolute_x_axis_tolerance=ori_tol,
                                                          absolute_y_axis_tolerance=ori_tol,
                                                          absolute_z_axis_tolerance=ori_tol,
                                                          parameterization=OrientationConstraint.ROTATION_VECTOR) ]

        goal = MoveGroup.Goal()
        goal.request = MotionPlanRequest(
            workspace_parameters = WorkspaceParameters(
                                        min_corner=Vector3(x=-2.0, y=-2.0, z=0.0),
                                        max_corner=Vector3(x=-2.0, y=-2.0, z=0.0)),
            goal_constraints = [Constraints(position_constraints=position_constraints,
                                            orientation_constraints=orientation_constraints)],
            group_name = group_name,
            allowed_planning_time = 5.0,
            max_velocity_scaling_factor = 0.3,
            max_acceleration_scaling_factor = 0.3)

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

            
    def get_gripper_config(self, side:str, q:dict[str,float]) -> tuple[float,float,float,float]:
        spread = (q[f'{side}_HandFingerOneKnuckleOneJoint'] + q[f'{side}_HandFingerTwoKnuckleOneJoint']) / 2
        f1 = (q[f'{side}_HandFingerOneKnuckleTwoJoint']+q[f'{side}_HandFingerOneKnuckleThreeJoint']*3) / 2
        f2 = (q[f'{side}_HandFingerTwoKnuckleTwoJoint']+q[f'{side}_HandFingerTwoKnuckleThreeJoint']*3) / 2
        f3 = (q[f'{side}_HandFingerThreeKnuckleTwoJoint']+q[f'{side}_HandFingerThreeKnuckleThreeJoint']*3) / 2
        return f1, f2, f3, spread


    def state_close_gripper(self) -> FsmState:
        side = 'right'
        self.get_logger().info( 'Closing gripper to touch an object' )
        # Detect an object.
        # Close fingers with a very small tolerance.
        # If the first motion failed, the object is touched.
        if not self.move_fingers(side, math.radians(90), math.radians(90), math.radians(90), 0, 3.0, math.radians(5)):
            self.get_logger().info(f'The object is touched')
        else:
            self.get_logger().info(f'No object is detected')

        q_hand = self.get_joint_state(self.get_clock().now(), 5)
        if q_hand is None:
            return self.state_failure
        # else:
        f1, f2, f3, spread = self.get_gripper_config(side, q_hand)

        # Close the gripper once again, this time with larger tolerance.
        self.get_logger().info( 'Closing gripper to grasp an object' )
        f_add = math.radians(5)
        if not self.move_fingers(side, f1+f_add, f2+f_add, f3+f_add, 0, 3.0, math.radians(45)):
            return self.state_failure
        else:
            # Ok
            return self.state_lift


    def state_lift(self) -> FsmState:
        if not self.move_group_action.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("MoveGroup is not available")
            return self.state_failure
        # else:

        group_name = 'right_arm_torso'
        tip_link = 'right_arm_7_link'

        # Object's position is hardcoded
        obj_pt_B = PyKDL.Vector(0.6, -0.3, 1.1150)
        lift_vector_B = PyKDL.Vector(0,0, 0.3)

        T_B_Gr_des = PyKDL.Frame(
            PyKDL.Rotation.RotZ(math.radians(-90)) * PyKDL.Rotation.RotY(math.radians(180)),
            obj_pt_B + lift_vector_B)

        # Visualize the calculated pose in TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'torso_base'
        t.child_frame_id = 'T_B_Gr_des'
        t.transform = tfKDLtoROS(T_B_Gr_des)
        self.tf_broadcaster.sendTransform(t)

        self.get_logger().info(
            f'Planning and execution: motion to a given end-effector pose for group "{group_name}"'
        )

        # Get a constant transformation between right hand frame and right end effector frame
        T_Gr_E = self.get_frame_position(tip_link, base_frame='right_HandGripLink')
        if T_Gr_E is None:
            # Retry
            return self.state_lift

        # Calculate desired pose of right hand frame
        T_B_Er_des = T_B_Gr_des * T_Gr_E

        pose = poseKDLtoROS(T_B_Er_des)
        position_constraints = [ PositionConstraint(header=Header(frame_id='torso_base'),
                                                    link_name=tip_link,
                                                    constraint_region=BoundingVolume(
                                                        primitives=[SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01])],
                                                        primitive_poses=[pose],
                                                    ),
                                                    weight=1.0) ]

        ori_tol = math.radians(1)
        orientation_constraints = [ OrientationConstraint(header=Header(frame_id='torso_base'),
                                                          link_name=tip_link,
                                                          orientation=pose.orientation,
                                                          absolute_x_axis_tolerance=ori_tol,
                                                          absolute_y_axis_tolerance=ori_tol,
                                                          absolute_z_axis_tolerance=ori_tol,
                                                          parameterization=OrientationConstraint.ROTATION_VECTOR) ]

        goal = MoveGroup.Goal()
        goal.request = MotionPlanRequest(
            workspace_parameters = WorkspaceParameters(
                                        min_corner=Vector3(x=-2.0, y=-2.0, z=0.0),
                                        max_corner=Vector3(x=-2.0, y=-2.0, z=0.0)),
            goal_constraints = [Constraints(position_constraints=position_constraints,
                                            orientation_constraints=orientation_constraints)],
            group_name = group_name,
            allowed_planning_time = 5.0,
            max_velocity_scaling_factor = 0.3,
            max_acceleration_scaling_factor = 0.3)

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
                return self.state_open_gripper
            else:
                self.get_logger().error(f'error: status={response.status}, '
                                        f'error_code={response.result.error_code.val}, '
                                        f'error_string={response.result.error_code.message}')
                return self.state_failure


    def state_open_gripper(self) -> FsmState:
        self.get_logger().info('Opening gripper')
        if not self.move_fingers('right', 0, 0, 0, 0, 3.0, math.radians(20)):
            return self.state_failure
        # else:
        # Ok
        return self.state_finish


    #
    # Current joint_state
    #
    def joint_state_callback(self, msg: JointState) -> None:
        current_joint_positions = {}
        for name, position in zip(msg.name, msg.position):
            current_joint_positions[name] = position

        with self._state_lock:
            self._current_joint_positions = (msg.header.stamp.sec, msg.header.stamp.nanosec, current_joint_positions)


    def get_joint_state(self, stamp:Time, timeout_s:float) -> Optional[dict[str,float]]:
        timeout_wall_time = time.time() + timeout_s
        while time.time() < timeout_wall_time:
            time.sleep(0.1)
            with self._state_lock:
                if self._current_joint_positions is None:
                    continue
                # else:
                sec, nanosec, current_joint_positions = self._current_joint_positions
            js_stamp = Time(seconds=sec, nanoseconds=nanosec, clock_type=self._clock_type)
            if js_stamp > stamp:
                return current_joint_positions
        return None

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


    #
    # Grippers actions
    #

    def move_fingers(self, side:str, f1:float, f2:float, f3:float, spread:float, duration_s:float, path_tolerance:float) -> bool:
        """This is a blocking method. Use it with MultiThreadedExecutor only."""
        if side == 'left':
            hand_client = self.hand_left_client
        elif side == 'right':
            hand_client = self.hand_right_client
        else:
            raise Exception(f'Wrong side: "{side}"')
        
        joint_names = [
            f'{side}_HandFingerOneKnuckleOneJoint',
            f'{side}_HandFingerOneKnuckleTwoJoint',
            f'{side}_HandFingerOneKnuckleThreeJoint',
            f'{side}_HandFingerThreeKnuckleTwoJoint',
            f'{side}_HandFingerThreeKnuckleThreeJoint',
            f'{side}_HandFingerTwoKnuckleOneJoint',
            f'{side}_HandFingerTwoKnuckleTwoJoint',
            f'{side}_HandFingerTwoKnuckleThreeJoint',
        ]
        positions = [spread, f1, f1/3.0, f3, f3/3.0, spread, f2, f2/3.0]

        if not hand_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Gripper controller is not available")
            return False
        # else:

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(joint_names)
        goal.path_tolerance = [
            JointTolerance(name=joint, position=path_tolerance)
            for joint in goal.trajectory.joint_names
        ]
        goal.goal_tolerance = [
            JointTolerance(name=joint, position=path_tolerance)
            for joint in goal.trajectory.joint_names
        ]
        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.time_from_start = Duration(seconds=duration_s).to_msg()
        goal.trajectory.points = [point]

        # Waits until goal is accepted
        response:FollowJointTrajectory_GetResult_Response|None = hand_client.send_goal(goal)
        if response is None:
            self.get_logger().error(f'move_fingers({side}) result: None')
            return False
        else:
            if (response.status == GoalStatus.STATUS_SUCCEEDED
                    and response.result.error_code
                        == FollowJointTrajectory.Result.SUCCESSFUL):
                self.get_logger().info(f'move_fingers({side}) succeeded')
                return True
            else:
                self.get_logger().error(f'move_fingers({side}) error: status={response.status}, '
                                        f'error_code={response.result.error_code}, '
                                        f'error_string={response.result.error_string}')
                return False


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspingNode()
    executor = MultiThreadedExecutor(num_threads=2)
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
