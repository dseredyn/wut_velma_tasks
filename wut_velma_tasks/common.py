import time
import math
import PyKDL
from threading import Lock, Event
from typing import Optional

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.task import Future
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.duration import Duration

from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Transform, TransformStamped, Vector3, Quaternion
from moveit_msgs.msg import (PositionConstraint, OrientationConstraint, BoundingVolume,
                             PlanningScene)
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
from gazebo_pose_interfaces.srv import GetObjectPose, GetObjectPose_Request, GetObjectPose_Response
from control_msgs.action import (FollowJointTrajectory, FollowJointTrajectory_GetResult_Response)
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from control_msgs.msg import JointTolerance

# from moveit_msgs.srv import ApplyPlanningScene

#
# Helper functions
#

def planningSceneTransforms(t_map:dict[str, PyKDL.Frame]) -> PlanningScene:
    scene = PlanningScene()
    scene.is_diff = True
    scene.robot_state.is_diff = True
    scene.fixed_frame_transforms = []

    for frame_name, T in t_map.items():
        transform = TransformStamped()

        # Konwencja MoveIt dla fixed_frame_transforms:
        transform.header.frame_id = frame_name
        transform.child_frame_id = 'world'

        transform.transform = tfKDLtoROS(T)
        scene.fixed_frame_transforms.append(transform)

    return scene

def createConstraints(constr_list:list) -> tuple[list[PositionConstraint], list[OrientationConstraint]]:
    pos_constr:list[PositionConstraint] = []
    ori_constr:list[OrientationConstraint] = []

    pos_tol = 0.01
    ori_tol = math.radians(3)

    for constr in constr_list:
            if constr[0] == 'constrain local point':
                ee_link_name, pt_E, pt_B = constr[1:]
                assert ee_link_name in ('left_arm_7_link', 'right_arm_7_link')
                assert isinstance(pt_E, PyKDL.Vector)
                assert isinstance(pt_B, PyKDL.Vector)
                pos_constr.append(
                    PositionConstraint(header=Header(frame_id='torso_base'),
                                                    link_name=ee_link_name,
                                                    target_point_offset=vectorKDLtoROS(pt_E),
                                                    constraint_region=BoundingVolume(
                                                        primitives=[SolidPrimitive(type=SolidPrimitive.SPHERE,
                                                                                    dimensions=[pos_tol])],
                                                        primitive_poses=[poseKDLtoROS(PyKDL.Frame(pt_B))],
                                                    ),
                                                    weight=1.0))
            elif constr[0] == 'constrain orientation':
                ee_link_name, base_frame_name, M_B_E, x_tol, y_tol, z_tol = constr[1:]
                assert ee_link_name in ('left_arm_7_link', 'right_arm_7_link')
                assert isinstance(M_B_E, PyKDL.Rotation)
                ori_constr.append(
                     OrientationConstraint(header=Header(frame_id=base_frame_name),
                                            link_name=ee_link_name,
                                            orientation=quaternionKDLtoROS(M_B_E),
                                            absolute_x_axis_tolerance=x_tol,
                                            absolute_y_axis_tolerance=y_tol,
                                            absolute_z_axis_tolerance=z_tol,
                                            parameterization=OrientationConstraint.ROTATION_VECTOR,
                                            weight=1.0))
            else:
                raise Exception(f'Wrong type of constraint: {constr[0]}')
    return pos_constr, ori_constr



def poseKDLtoROS(T:PyKDL.Frame) -> Pose:
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = T.p.x(), T.p.y(), T.p.z()
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = T.M.GetQuaternion()
    return pose

def vectorKDLtoROS(v:PyKDL.Vector) -> Vector3:
    return Vector3(x=v.x(), y=v.y(), z=v.z())

def poseRosToKDL(p:Pose) -> PyKDL.Frame:
    pos = PyKDL.Vector(p.position.x, p.position.y, p.position.z)
    q = PyKDL.Rotation.Quaternion(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
    return PyKDL.Frame(q, pos)

def quaternionKDLtoROS(M:PyKDL.Rotation) -> Quaternion:
    x, y, z, w = M.GetQuaternion()
    return Quaternion(x=x, y=y, z=z, w=w)

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

def get_gripper_config(side:str, q:dict[str,float]) -> tuple[float,float,float,float]:
    spread = (q[f'{side}_HandFingerOneKnuckleOneJoint'] + q[f'{side}_HandFingerTwoKnuckleOneJoint']) / 2
    f1 = (q[f'{side}_HandFingerOneKnuckleTwoJoint']+q[f'{side}_HandFingerOneKnuckleThreeJoint']*3) / 2
    f2 = (q[f'{side}_HandFingerTwoKnuckleTwoJoint']+q[f'{side}_HandFingerTwoKnuckleThreeJoint']*3) / 2
    f3 = (q[f'{side}_HandFingerThreeKnuckleTwoJoint']+q[f'{side}_HandFingerThreeKnuckleThreeJoint']*3) / 2
    return f1, f2, f3, spread

#
# Helper classes
#
class JointStateThread:
    def __init__(self, node:Node):
        now = node.get_clock().now()
        self._clock_type = now.clock_type
        self._state_lock = Lock()
        #self._current_joint_positions = None
        self._current_js = None
        self.state_group = MutuallyExclusiveCallbackGroup()
        self.sub_joint_state = node.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            10,
            callback_group=self.state_group
        )

    def _joint_state_callback(self, msg: JointState) -> None:
        # current_joint_positions = {}
        # for name, position in zip(msg.name, msg.position):
        #     current_joint_positions[name] = position

        with self._state_lock:
            self._current_js = msg
            # self._current_joint_positions = (msg.header.stamp.sec, msg.header.stamp.nanosec, current_joint_positions)

    def get_JointState(self, stamp:Time, timeout_s:float) -> Optional[JointState]:
        timeout_wall_time = time.time() + timeout_s
        msg = None
        while time.time() < timeout_wall_time:
            time.sleep(0.1)
            with self._state_lock:
                msg = self._current_js
            if not msg is None:
                js_stamp = Time(seconds=msg.header.stamp.sec, nanoseconds=msg.header.stamp.nanosec, clock_type=self._clock_type)
                if js_stamp > stamp:
                    return msg
        return None

    def get_joint_state(self, stamp:Time, timeout_s:float) -> Optional[dict[str,float]]:
        msg = self.get_JointState(stamp, timeout_s=timeout_s)
        if msg is None:
            return None
        # else:
        result:dict[str,float] = {}
        for name, position in zip(msg.name, msg.position):
            result[name] = position
        return result


class GazeboPoseThr:
    def __init__(self, node:Node, pose_objects:list[str]):
        self._pose_objects = pose_objects
        self._pose_lock = Lock()
        self._pose_map:dict = {}
        self.pose_cli = node.create_client(GetObjectPose, '/get_gazebo_object_pose')
        while not self.pose_cli.wait_for_service(timeout_sec=1.0):
            node.get_logger().info('service /get_gazebo_object_pose not available, waiting again...')

        # Start the pose timer
        self.pose_group = MutuallyExclusiveCallbackGroup()
        self._pose_timer = node.create_timer(1.0, self._pose_timer_callback,
                                                        callback_group=self.pose_group)

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
        # for obj_name, obj_pose in self._pose_map.items():
        #     t = TransformStamped()
        #     t.header.stamp = self.get_clock().now().to_msg()
        #     t.header.frame_id = 'torso_base'
        #     t.child_frame_id = obj_name
        #     t.transform = tfKDLtoROS(obj_pose)
        #     self.tf_broadcaster.sendTransform(t)

    def getObjectPose(self, name:str) -> PyKDL.Frame|None:
        with self._pose_lock:
            if name in self._pose_map:
                pose = self._pose_map[name]
            else:
                pose = None
        return pose

    def getAllObjectsPoses(self) -> dict[str,PyKDL.Frame]:
        with self._pose_lock:
            result = self._pose_map.copy()
        return result


class GrippersActionsInterface:
    def __init__(self, node:Node):
        self.node = node
        self.action_group = MutuallyExclusiveCallbackGroup()
        self.hand_left_client = ActionClient( node, FollowJointTrajectory,
                                                "/left_hand_controller/follow_joint_trajectory",
                                                 callback_group=self.action_group )
        self.hand_right_client = ActionClient( node, FollowJointTrajectory,
                                                "/right_hand_controller/follow_joint_trajectory",
                                                 callback_group=self.action_group )

    def _get_client(self, side:str) -> ActionClient:
        if side == 'left':
            return self.hand_left_client
        elif side == 'right':
            return self.hand_right_client
        else:
            raise Exception(f'Wrong side: "{side}"')

    def _prepare_goal(self, side:str, f1:float, f2:float, f3:float, spread:float, duration_s:float,
                      path_tolerance:float) -> FollowJointTrajectory.Goal:
        """This is a blocking method. Use it with MultiThreadedExecutor only."""
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
        return goal

        # Waits until goal is accepted
    def move_fingers(self, side:str, f1:float, f2:float, f3:float, spread:float, duration_s:float, path_tolerance:float) -> bool:
        goal = self._prepare_goal(side, f1, f2, f3, spread, duration_s, path_tolerance)
        hand_client = self._get_client(side)
        if not hand_client.wait_for_server(timeout_sec=10.0):
            self.node.get_logger().error("Gripper controller is not available")
            return False
        # else:
        response:FollowJointTrajectory_GetResult_Response|None = hand_client.send_goal(goal)
        if response is None:
            self.node.get_logger().error(f'move_fingers({side}) result: None')
            return False
        else:
            if (response.status == GoalStatus.STATUS_SUCCEEDED
                    and response.result.error_code
                        == FollowJointTrajectory.Result.SUCCESSFUL):
                self.node.get_logger().info(f'move_fingers({side}) succeeded')
                return True
            else:
                self.node.get_logger().error(f'move_fingers({side}) error: status={response.status}, '
                                        f'error_code={response.result.error_code}, '
                                        f'error_string={response.result.error_string}')
                return False

    def move_fingers_async(self, side:str, f1:float, f2:float, f3:float, spread:float, duration_s:float, path_tolerance:float) -> Future:
        goal = self._prepare_goal(side, f1, f2, f3, spread, duration_s, path_tolerance)
        hand_client = self._get_client(side)
        if not hand_client.wait_for_server(timeout_sec=10.0):
            self.node.get_logger().error("Gripper controller is not available")
            raise Exception("Gripper controller is not available")
        # else:

        # Future obejmujący przyjęcie celu i otrzymanie końcowego wyniku.
        final_future = Future()

        def result_received(future):
            try:
                final_future.set_result(future.result())
            except Exception as exc:
                final_future.set_exception(exc)

        def goal_response_received(future):
            try:
                handle = future.result()
                if handle is None or not handle.accepted:
                    raise RuntimeError(f"Cel dla {side} został odrzucony")

                handle.get_result_async().add_done_callback(result_received)
            except Exception as exc:
                final_future.set_exception(exc)

        hand_client.send_goal_async(goal).add_done_callback(goal_response_received)
        return final_future

    
    def wait_for_result(self, future:Future):
        finished = Event()

        # Callback zostanie wykonany również wtedy, gdy future już się zakończył.
        future.add_done_callback(lambda _: finished.set())
        finished.wait()
        
        if future.done():
            response:FollowJointTrajectory_GetResult_Response|None = future.result()
            if response is None:
                raise RuntimeError("Nie otrzymano wyniku akcji")
            if (response.status == GoalStatus.STATUS_SUCCEEDED
                    and response.result.error_code
                        == FollowJointTrajectory.Result.SUCCESSFUL):
                self.node.get_logger().info(f'move_fingers succeeded')
                return True
            else:
                self.node.get_logger().error(f'move_fingers error: status={response.status}, '
                                        f'error_code={response.result.error_code}, '
                                        f'error_string={response.result.error_string}')
                return False

            
class ArmsTorsoActionInterface:
    def __init__(self, node:Node):
        self.node = node
        self.action_group = MutuallyExclusiveCallbackGroup()

        self.client = ActionClient( node, FollowJointTrajectory,
                                                "/arms_torso_trajectory_controller_jnt/follow_joint_trajectory",
                                                 callback_group=self.action_group )

    def move(self, traj:JointTrajectory, path_tolerance:float) -> bool:
        """This is a blocking method. Use it with MultiThreadedExecutor only."""
        if not self.client.wait_for_server(timeout_sec=10.0):
            self.node.get_logger().error("Controller is not available")
            return False
        # else:

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        # goal.path_tolerance = [
        #     JointTolerance(name=joint, position=path_tolerance)
        #     for joint in goal.trajectory.joint_names
        # ]
        # goal.goal_tolerance = [
        #     JointTolerance(name=joint, position=path_tolerance)
        #     for joint in goal.trajectory.joint_names
        # ]
        # point = JointTrajectoryPoint()
        # point.positions = list(positions)
        # point.time_from_start = Duration(seconds=duration_s).to_msg()
        # goal.trajectory.points = [point]

        # Waits until goal is accepted
        response:FollowJointTrajectory_GetResult_Response|None = self.client.send_goal(goal)
        if response is None:
            self.node.get_logger().error(f'move() result: None')
            return False
        else:
            if (response.status == GoalStatus.STATUS_SUCCEEDED
                    and response.result.error_code
                        == FollowJointTrajectory.Result.SUCCESSFUL):
                self.node.get_logger().info(f'move() succeeded')
                return True
            else:
                self.node.get_logger().error(f'move() error: status={response.status}, '
                                        f'error_code={response.result.error_code}, '
                                        f'error_string={response.result.error_string}')
                return False
