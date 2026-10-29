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

from geometry_msgs.msg import Pose, Transform, TransformStamped, Vector3, Point

import PyKDL

from std_msgs.msg import Header
from moveit_msgs.action import (MoveGroup, MoveGroup_GetResult_Response)
from moveit_msgs.msg import (Constraints, JointConstraint, MoveItErrorCodes, WorkspaceParameters,
    MotionPlanRequest, PositionConstraint, OrientationConstraint, BoundingVolume,
    CollisionObject, LinkPadding)

from shape_msgs.msg import SolidPrimitive

# For gripper control
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory, FollowJointTrajectory_GetResult_Response
from rclpy.action import ActionClient
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import JointTolerance

from gazebo_pose_interfaces.srv import GetObjectPose, GetObjectPose_Request, GetObjectPose_Response
from moveit_msgs.srv import ApplyPlanningScene, ApplyPlanningScene_Request, ApplyPlanningScene_Response

from gazebo_point_springs_interfaces.action import (FollowEquilibriumTrajectory,
                                                    FollowEquilibriumTrajectory_GetResult_Response)
from gazebo_point_springs_interfaces.msg import EquilibriumWaypoint

#
# Helper functions
#
def poseKDLtoROS(T:PyKDL.Frame) -> Pose:
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = T.p.x(), T.p.y(), T.p.z()
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = T.M.GetQuaternion()
    return pose


def poseRosToKDL(p:Pose) -> PyKDL.Frame:
    pos = PyKDL.Vector(p.position.x, p.position.y, p.position.z)
    q = PyKDL.Rotation.Quaternion(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
    return PyKDL.Frame(q, pos)


def tfKDLtoROS(T:PyKDL.Frame) -> Transform:
    t = Transform()
    t.translation.x, t.translation.y, t.translation.z = T.p.x(), T.p.y(), T.p.z()
    t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w = T.M.GetQuaternion()
    return t


def tfROStoKDL(t_ros:Transform) -> PyKDL.Frame:
    return PyKDL.Frame( PyKDL.Rotation.Quaternion(  t_ros.rotation.x, t_ros.rotation.y,
                                                    t_ros.rotation.z, t_ros.rotation.w),
                    PyKDL.Vector(t_ros.translation.x, t_ros.translation.y, t_ros.translation.z))

def rotationToStr(r:PyKDL.Rotation) -> str:
    x, y, z, w = r.GetQuaternion()
    return f'PyKDL.Rotation.Quaternion({x}, {y}, {z}, {w})'

def vectorToStr(p:PyKDL.Vector) -> str:
    return f'PyKDL.Vector({p.x()}, {p.y()}, {p.z()})'

def frameToStr(t:PyKDL.Frame) -> str:
    return f'PyKDL.Frame({rotationToStr(t.M)}, {vectorToStr(t.p)})'

def frameKdlToConstraints(t:PyKDL.Frame, link_name:str) -> tuple[list[PositionConstraint], list[OrientationConstraint]]:
    pose = poseKDLtoROS(t)
    pos_tol = 0.002
    ori_tol = math.radians(1)
    return ([ PositionConstraint(header=Header(frame_id='torso_base'),
                                        link_name=link_name,
                                        constraint_region=BoundingVolume(
                                            primitives=[SolidPrimitive(type=SolidPrimitive.SPHERE,
                                                                        dimensions=[pos_tol])],
                                            primitive_poses=[pose],
                                        ),
                                        weight=1.0) ],
            [ OrientationConstraint(header=Header(frame_id='torso_base'),
                                        link_name=link_name,
                                        orientation=pose.orientation,
                                        absolute_x_axis_tolerance=ori_tol,
                                        absolute_y_axis_tolerance=ori_tol,
                                        absolute_z_axis_tolerance=ori_tol,
                                        parameterization=OrientationConstraint.ROTATION_VECTOR,
                                        weight=1.0) ])

#
#
#
class GraspingNode(Node):
    FsmState = Callable[[], Callable|None]

    def __init__(self) -> None:
        super().__init__("example_node")

        self.fsm_group = MutuallyExclusiveCallbackGroup()
        self.action_group = MutuallyExclusiveCallbackGroup()
        self.pose_group = MutuallyExclusiveCallbackGroup()

        #
        # Initialize subscribers
        #
        now = self.get_clock().now()
        self._clock_type = now.clock_type

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        self.human_client = ActionClient( self, FollowEquilibriumTrajectory,
                                                "/table_human/follow_equilibrium_trajectory",
                                                 callback_group=self.action_group )

        # Pose retrieval service
        self._pose_objects = ['can', 'table']
        self._pose_lock = Lock()
        self._pose_map:dict = {}
        self.pose_cli = self.create_client(GetObjectPose, '/get_gazebo_object_pose')
        while not self.pose_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service /get_gazebo_object_pose not available, waiting again...')

        # Start the pose timer
        self._pose_timer = self.create_timer(1.0, self._pose_timer_callback,
                                                        callback_group=self.pose_group)


        #
        # Initialize the FSM
        #
        self._FSM_next_state_timer = None

        self.get_logger().info(f'Initialization is done. Starting FSM.')

        # The initial state
        self.FSM_init(self.state_lift_table)


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


    def _pose_timer_callback(self):
        for obj_name in self._pose_objects:
            self.req:GetObjectPose_Request = GetObjectPose.Request()
            self.req.name = obj_name
            pose_future = self.pose_cli.call_async(self.req)
            pose_future.add_done_callback(self._pose_callback)

    def _pose_callback(self, future):
        if future.done():
            result:GetObjectPose_Response = future.result()
            if result.success:
                with self._pose_lock:
                    self._pose_map[result.name] = poseRosToKDL(result.pose)

        # Visualize the calculated pose in TF
        for obj_name, obj_pose in self._pose_map.items():
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'torso_base'
            t.child_frame_id = obj_name
            t.transform = tfKDLtoROS(obj_pose)
            self.tf_broadcaster.sendTransform(t)


    def getObjectPose(self, name:str) -> PyKDL.Frame|None:
        with self._pose_lock:
            if name in self._pose_map:
                pose = self._pose_map[name]
            else:
                pose = None
        return pose


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


    def state_lift_table(self) -> FsmState:
        if not self.human_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("human action is not available")
            return self.state_failure
        # else:

        obj_name = 'table'
        T_B_O = self.getObjectPose(obj_name)
        if T_B_O is None:
            self.get_logger().info(f'Could not get pose of object "{obj_name}"')
            return self.state_lift_table

        pl_O = PyKDL.Vector(-0.5, 0.35, 1)
        pr_O = PyKDL.Vector(0.5, 0.35, 1)
    
        pl_B = T_B_O * pl_O + PyKDL.Vector(0, 0, 0.15)
        pr_B = T_B_O * pr_O + PyKDL.Vector(0, 0, 0.15)

        goal = FollowEquilibriumTrajectory.Goal()
        goal.header.frame_id = 'world'
        goal.point_names = ['hand_left', 'hand_right']
        goal.points = [EquilibriumWaypoint(time_from_start=Duration(seconds=5).to_msg(),
                                           positions=[Point(x=pl_B.x(), y=pl_B.y(), z=pl_B.z()),
                                                      Point(x=pr_B.x(), y=pr_B.y(), z=pr_B.z())])]

        # Waits until goal is accepted
        response:FollowEquilibriumTrajectory_GetResult_Response|None = self.human_client.send_goal(goal)
        if response is None:
            self.get_logger().error(f'result: None')
            return self.state_failure
        else:
            assert isinstance(response, FollowEquilibriumTrajectory_GetResult_Response)

            if (response.status == GoalStatus.STATUS_SUCCEEDED
                    and response.result.code == FollowEquilibriumTrajectory.Result.SUCCESS):
                self.get_logger().info(f'succeeded')
                return self.state_finish
            else:
                self.get_logger().error(f'error: status={response.status}, '
                                        f'error_code={response.result.code}, '
                                        f'error_string={response.result.message}')
                return self.state_failure

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
    node = GraspingNode()
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
