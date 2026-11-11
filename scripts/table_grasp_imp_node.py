#!/usr/bin/env python3
"""
example_node.py

Basic motions for WUT Velma.
Simple implementation of FSM.
"""

import copy
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
from rclpy.time import Time, Duration

from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import JointState

import PyKDL

from moveit_msgs.action import (MoveGroup, MoveGroup_GetResult_Response)
from moveit_msgs.msg import (Constraints, MoveItErrorCodes, WorkspaceParameters,
    MotionPlanRequest, CollisionObject, LinkPadding, PlanningScene, JointConstraint)

from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from moveit_msgs.msg import (PositionConstraint, OrientationConstraint, BoundingVolume)
from visualization_msgs.msg import Marker, MarkerArray

# For gripper control
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient

from moveit_msgs.srv import (ApplyPlanningScene, ApplyPlanningScene_Request,
                ApplyPlanningScene_Response, GetCartesianPath, GetCartesianPath_Request,
                GetCartesianPath_Response)

from wut_velma_tasks.common import (JointStateThread, GazeboPoseThr, GrippersActionsInterface,
                                    ArmsTorsoActionInterface, GetCartesianPathInterface,
                                    vectorKDLtoROS, quaternionKDLtoROS,
                                    poseKDLtoROS, frameKdlToConstraints, tfROStoKDL,
                                    get_gripper_config, createConstraints, planningSceneTransforms)

from bisect import bisect_left

from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from cartesian_impedance_msgs.action import (FollowCartesianImpedanceTrajectory,
                                FollowCartesianImpedanceTrajectory_GetResult_Response)
from cartesian_impedance_msgs.msg import CartesianTrajectoryPoint

from controller_manager_msgs.srv import SwitchController, ListControllers

def secondsToDuration(t:float) -> Duration:
    sec = int(math.floor(t))
    nanosec = int((t - sec)*1_000_000_000)
    return Duration(sec=sec, nanosec=nanosec )

def combineTrajectories(
    t0: JointTrajectory,
    t1: JointTrajectory,
) -> JointTrajectory:
    """Combine trajectories for disjoint joint sets.

    Requires positions and velocities at every point, strictly increasing
    timestamps, and the first point at time zero.

    Copies t0.header. Accelerations and effort are omitted.
    Input trajectories are not modified.
    """
    ns_per_second = 1_000_000_000

    def validate(trajectory: JointTrajectory) -> list[int]:
        n = len(trajectory.joint_names)
        if n == 0 or not trajectory.points:
            raise ValueError("Trajectory must contain joints and points")

        times = []
        for point in trajectory.points:
            if len(point.positions) != n or len(point.velocities) != n:
                raise ValueError(
                    "Every point must provide one position and velocity "
                    "per joint"
                )

            duration = point.time_from_start
            times.append(
                duration.sec * ns_per_second + duration.nanosec
            )

        if times[0] != 0:
            raise ValueError("The first point must be at time zero")
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Timestamps must be strictly increasing")

        return times

    names = list(t0.joint_names) + list(t1.joint_names)
    if len(names) != len(set(names)):
        raise ValueError("Joint names must be unique across both trajectories")

    times0 = validate(t0)
    times1 = validate(t1)

    def sample(trajectory, times, time_ns):
        points = trajectory.points

        if time_ns > times[-1]:
            return (
                list(points[-1].positions),
                [0.0] * len(trajectory.joint_names),
            )

        i = bisect_left(times, time_ns)

        if times[i] == time_ns:
            return list(points[i].positions), list(points[i].velocities)

        before = points[i - 1]
        after = points[i]
        alpha = (time_ns - times[i - 1]) / (times[i] - times[i - 1])

        positions = [
            a + alpha * (b - a)
            for a, b in zip(before.positions, after.positions)
        ]
        velocities = [
            a + alpha * (b - a)
            for a, b in zip(before.velocities, after.velocities)
        ]
        return positions, velocities

    combined = JointTrajectory()
    combined.header = copy.deepcopy(t0.header)
    combined.joint_names = names
    combined.points = []

    for time_ns in sorted(set(times0) | set(times1)):
        positions0, velocities0 = sample(t0, times0, time_ns)
        positions1, velocities1 = sample(t1, times1, time_ns)

        sec, nanosec = divmod(time_ns, ns_per_second)
        combined.points.append(
            JointTrajectoryPoint(
                positions=positions0 + positions1,
                velocities=velocities0 + velocities1,
                time_from_start=Duration(sec=sec, nanosec=nanosec),
            )
        )

    return combined


# def updateJointState(js_full:JointState, js_part:JointState, group_name:str) -> JointState:
#     if group_name == 'right_arm_torso':
#         joint_names = ["torso_0_joint",
#                    "right_arm_0_joint", "right_arm_1_joint", "right_arm_2_joint",
#                    "right_arm_3_joint", "right_arm_4_joint", "right_arm_5_joint",
#                    "right_arm_6_joint"]
#     elif group_name == 'left_arm':
#         joint_names = ["left_arm_0_joint", "left_arm_1_joint", "left_arm_2_joint",
#                    "left_arm_3_joint", "left_arm_4_joint", "left_arm_5_joint", "left_arm_6_joint"]
#     else:
#         raise Exception()
#     result = js_full
#     for name, pos in zip(js_part.name, js_part.position):
#         if name in joint_names:
#             for idx, name2 in enumerate(result.name):
#                 if name2 == name:
#                     result.position[idx] = pos
#                     break
#     return result

def updateJointState(js_full:JointState, js_part:JointState) -> JointState:
    result = copy.deepcopy(js_full)
    for name, pos in zip(js_part.name, js_part.position):
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

from enum import Enum
class ControlMode(Enum):
    JNT_IMP = 1
    CART_IMP = 2
    UNKNOWN = 3

from controller_manager_msgs.msg import ControllerState

class ControllerManagerInterface:
    def __init__(self, node:Node):
        self.node = node
        self.switch_cli = self.node.create_client(SwitchController, '/controller_manager/switch_controller')
        self.list_cli = self.node.create_client(ListControllers, '/controller_manager/list_controllers')
        while not self.switch_cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service /controller_manager/switch_controller not available, waiting again...')

        while not self.list_cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service /controller_manager/switch_controller not available, waiting again...')


    def getCurrentControlMode(self) -> ControlMode:
        response = self.list_cli.call( ListControllers.Request() )
        if response is None:
            raise Exception('SwitchController service call failed')
        controller_list:list[ControllerState] = list(response.controller)
        active_controllers:list[str] = []
        for controller in controller_list:
            if controller.state == 'active':
                active_controllers.append( controller.name )
        if ('arms_torso_effort_jnt' in active_controllers and
                    'arms_torso_trajectory_controller_jnt' in active_controllers and
                    not 'cartesian_controller' in active_controllers):
            return ControlMode.JNT_IMP

        if (not 'arms_torso_effort_jnt' in active_controllers and
                    not 'arms_torso_trajectory_controller_jnt' in active_controllers and
                    'cartesian_controller' in active_controllers):
            return ControlMode.CART_IMP

        return ControlMode.UNKNOWN
        

    def trySwitchTo(self, mode:ControlMode):
        current_mode = self.getCurrentControlMode()
        if current_mode == mode:
            pass
        elif current_mode == ControlMode.CART_IMP and mode == ControlMode.JNT_IMP:
            self.switchTo(mode)
        elif current_mode == ControlMode.JNT_IMP and mode == ControlMode.CART_IMP:
            self.switchTo(mode)
        else:
            raise Exception(f'Switch is not supported: {current_mode} -> {mode}')

    def switchTo(self, mode:ControlMode):
        if mode == ControlMode.JNT_IMP:
            req_cm = SwitchController.Request(activate_controllers=['arms_torso_effort_jnt', 'arms_torso_trajectory_controller_jnt'],
                deactivate_controllers=['cartesian_controller'],
                strictness = SwitchController.Request.STRICT)
        elif mode == ControlMode.CART_IMP:
            req_cm = SwitchController.Request(activate_controllers=['cartesian_controller'],
                deactivate_controllers=['arms_torso_effort_jnt', 'arms_torso_trajectory_controller_jnt'],
                strictness = SwitchController.Request.STRICT)
        else:
            raise Exception(f'Not supported switch goal: {mode}')
        response = self.switch_cli.call(req_cm)
        if response is None:
            raise Exception('SwitchController service call failed')
        if not response.ok:
            raise Exception('SwitchController failed')


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

        self.cart_path_int = GetCartesianPathInterface(self)

        #
        # Cartesian impedance action
        #
        self.move_cart_action = ActionClient( self, FollowCartesianImpedanceTrajectory,
                                                "/cartesian_controller/follow_cartesian_impedance_trajectory",
                                                 callback_group=self.action_group )

        # Wait for initialization
        if not self.move_group_action.wait_for_server(timeout_sec=10.0):
            raise Exception('MoveGroup is not available')
        # else:
        if not self.move_cart_action.wait_for_server(timeout_sec=10.0):
            raise Exception('MoveGroup is not available')
        # else:

        self.cm_int = ControllerManagerInterface(self)

        # Visual marker publisher
        self.publisher_vis = self.create_publisher(MarkerArray, '/task_markers', 10)

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
        ee_l_link_name = 'left_arm_7_link'
        ee_r_link_name = 'right_arm_7_link'
        f_link_name = 'right_HandFingerOneKnuckleThreeLink'

        T_B_Er = self.get_frame_position(ee_r_link_name)
        if T_B_Er is None:
            raise Exception(f'Could not get pose of the link "{ee_r_link_name}"')

        T_B_El = self.get_frame_position(ee_l_link_name)
        if T_B_El is None:
            raise Exception(f'Could not get pose of the link "{ee_l_link_name}"')

        T_B_F = self.get_frame_position(f_link_name)
        if T_B_F is None:
            raise Exception(f'Could not get pose of the link "{f_link_name}"')

        T_E_F = T_B_Er.Inverse() * T_B_F

        js = self.js_thr.get_JointState(self.get_clock().now(), 5)

        obj_name = 'table'
        T_B_O = self.gz_pose_thr.getObjectPose(obj_name)
        if T_B_O is None:
            raise Exception(f'Could not get pose of object "{obj_name}"')


        return T_E_F, js, T_B_O, T_B_El, T_B_Er

    def state_move_to_table(self) -> FsmState:
        self.cm_int.trySwitchTo(ControlMode.JNT_IMP)

        T_E_F, _, T_B_O, _, _ = self.getWorldState()

        # The contact frame C wrt E
        T_E_C = T_E_F * PyKDL.Frame(PyKDL.Vector(0.05, 0, 0))

        # Initialize the task
        self.support_task_params = SupportTaskParams()            
        self.support_task_params.T_El_C = T_E_C
        self.support_task_params.T_Er_C = T_E_C
        pt_ref_B = T_B_O * self.support_task_params.pt_ref_O
        height_reference = pt_ref_B.z()
        self.support_task_params.height_eq = height_reference

        # Lower both E so that planner can find a goal state
        T_B_B_lower = PyKDL.Frame(PyKDL.Vector(0, 0, -0.03))
        T_B_Cr = T_B_O * self.support_task_params.T_O_Cr
        T_B_Cr = PyKDL.Frame(T_B_Cr.M, PyKDL.Vector(T_B_Cr.p.x(), T_B_Cr.p.y(), self.support_task_params.height_eq))
        T_B_Er_d = T_B_B_lower * T_B_Cr * self.support_task_params.T_Er_C.Inverse()

        T_B_Cl = T_B_O * self.support_task_params.T_O_Cl
        T_B_Cl = PyKDL.Frame(T_B_Cl.M, PyKDL.Vector(T_B_Cl.p.x(), T_B_Cl.p.y(), self.support_task_params.height_eq))
        T_B_El_d = T_B_B_lower * T_B_Cl * self.support_task_params.T_El_C.Inverse()

        constr_pos_r, constr_ori_r = frameKdlToConstraints(T_B_Er_d, 'right_arm_7_link')
        constr_pos_l, constr_ori_l = frameKdlToConstraints(T_B_El_d, 'left_arm_7_link')
        goal_constraints = [ Constraints(position_constraints=constr_pos_r + constr_pos_l,
                                        orientation_constraints=constr_ori_r + constr_ori_l) ]

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

        self.cm_int.trySwitchTo(ControlMode.CART_IMP)

        # TODO: set stiffness in the code: linear: 100, rotational: 50
        # TODO: add preferred joint state

        _, _, T_B_O, T_B_El, T_B_Er = self.getWorldState()

        pt_ref_B = T_B_O * self.support_task_params.pt_ref_O
        height_reference = pt_ref_B.z()

        pt_contact_l_B = T_B_O * self.support_task_params.T_O_Cl.p
        pt_contact_r_B = T_B_O * self.support_task_params.T_O_Cr.p
        pt_contact_mean_B = (pt_contact_l_B + pt_contact_r_B) / 2
        height_current = pt_contact_mean_B.z()

        if self.support_task_params.height_eq is None:
            # Initialize
            self.support_task_params.height_eq = height_reference

        height_tol = 0.01
        max_height_change = 0.03
        max_height_dist = 0.2

        if height_current < height_reference - height_tol:
            # Slowly raise the equilibrium point, if too low

            if self.support_task_params.height_eq < height_reference:
                self.support_task_params.height_eq = height_reference
            self.support_task_params.height_eq += max_height_change
            if self.support_task_params.height_eq > height_reference + max_height_dist:
                # Limit the value
                self.support_task_params.height_eq = height_reference + max_height_dist

        if height_current > height_reference + height_tol:
            # Slowly lower the equilibrium point, if too high
            self.support_task_params.height_eq -= max_height_change
            if self.support_task_params.height_eq < height_reference:
                # Limit the value
                self.support_task_params.height_eq = height_reference

        self.get_logger().info(f'height: reference: {height_reference:.2f},'
                               f' current: {height_current:.2f},'
                               f' equilibrium: {self.support_task_params.height_eq:.2f}')

        assert not self.support_task_params.T_Er_C is None and not self.support_task_params.T_El_C is None
        T_B_Cr = T_B_O * self.support_task_params.T_O_Cr
        T_B_Cr_d = PyKDL.Frame(T_B_Cr.M, PyKDL.Vector(T_B_Cr.p.x(), T_B_Cr.p.y(), self.support_task_params.height_eq))
        T_B_Er_d = T_B_Cr_d * self.support_task_params.T_Er_C.Inverse()

        T_B_Cl = T_B_O * self.support_task_params.T_O_Cl
        T_B_Cl_d = PyKDL.Frame(T_B_Cl.M, PyKDL.Vector(T_B_Cl.p.x(), T_B_Cl.p.y(), self.support_task_params.height_eq))
        T_B_El_d = T_B_Cl_d * self.support_task_params.T_El_C.Inverse()

        # Calculate movement time
        # m/s, rad/s
        max_lin_v = 0.1
        max_rot_v = math.radians(30)
        min_time = 0.1
        diff_r = PyKDL.diff(T_B_Er_d, T_B_Er)
        diff_l = PyKDL.diff(T_B_El_d, T_B_El)
        mov_time = max(min_time, diff_r.vel.Norm() / max_lin_v, diff_l.vel.Norm() / max_lin_v,
                                    diff_r.rot.Norm() / max_rot_v, diff_l.rot.Norm() / max_rot_v)

        dur = secondsToDuration(mov_time)

        goal = FollowCartesianImpedanceTrajectory.Goal()
        goal.ee_traj_r.header.frame_id = 'torso_base'
        goal.ee_traj_r.target_frame = 'right_arm_7_link'
        goal.ee_traj_r.points = [CartesianTrajectoryPoint(
                    time_from_start=dur, pose = poseKDLtoROS(T_B_Er_d))]

        goal.ee_traj_l.header.frame_id = 'torso_base'
        goal.ee_traj_l.target_frame = 'left_arm_7_link'
        goal.ee_traj_l.points = [CartesianTrajectoryPoint(
                    time_from_start=dur, pose = poseKDLtoROS(T_B_El_d))]

        # Visualization
        T_B_Cr_act = T_B_Er * self.support_task_params.T_Er_C
        T_B_Cl_act = T_B_El * self.support_task_params.T_El_C

        msg = MarkerArray()
        msg.markers = []
        scale = Vector3(x=0.04, y=0.04, z=0.04)
        colors = {'red':ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
                  'green':ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
                  'blue':ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
                  'yellow':ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
        }
        msg.markers.append( Marker(header=Header(frame_id='world'), ns='height_ref', id=0, type=Marker.SPHERE,
                    action=Marker.ADD, pose=poseKDLtoROS(PyKDL.Frame(pt_ref_B)), scale=scale, color=colors['blue']) )

        msg.markers.append( Marker(header=Header(frame_id='world'), ns='contact_obj', id=0, type=Marker.SPHERE,
                    action=Marker.ADD, pose=poseKDLtoROS(T_B_Cr), scale=scale, color=colors['blue']) )

        msg.markers.append( Marker(header=Header(frame_id='world'), ns='contact_obj', id=1, type=Marker.SPHERE,
                    action=Marker.ADD, pose=poseKDLtoROS(T_B_Cl), scale=scale, color=colors['blue']) )

        msg.markers.append( Marker(header=Header(frame_id='world'), ns='contact_cmd', id=0, type=Marker.SPHERE,
                    action=Marker.ADD, pose=poseKDLtoROS(T_B_Cr_d), scale=scale, color=colors['yellow']) )

        msg.markers.append( Marker(header=Header(frame_id='world'), ns='contact_cmd', id=1, type=Marker.SPHERE,
                    action=Marker.ADD, pose=poseKDLtoROS(T_B_Cl_d), scale=scale, color=colors['yellow']) )

        msg.markers.append( Marker(header=Header(frame_id='world'), ns='contact_ee', id=0, type=Marker.SPHERE,
                    action=Marker.ADD, pose=poseKDLtoROS(T_B_Cr_act), scale=scale, color=colors['green']) )

        msg.markers.append( Marker(header=Header(frame_id='world'), ns='contact_ee', id=1, type=Marker.SPHERE,
                    action=Marker.ADD, pose=poseKDLtoROS(T_B_Cl_act), scale=scale, color=colors['green']) )

        self.publisher_vis.publish(msg)

        # Waits until goal is accepted
        response = self.move_cart_action.send_goal(goal)
        if response is None:
            self.get_logger().error(f'result: None')
            return self.state_failure
        # else:

        assert isinstance(response, FollowCartesianImpedanceTrajectory_GetResult_Response)
        self.get_logger().info(f'error_code: {response.result.error_code}, message: {response.result.error_string}')
        if (response.status == GoalStatus.STATUS_SUCCEEDED
                and response.result.error_code == FollowCartesianImpedanceTrajectory.Result.SUCCESSFUL):
            self.get_logger().info(f'succeeded')
            return self.state_support
        else:
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


class SupportTaskParams:
    def __init__(self):
        # Transformations
        # F - finger 3 knuckle 3
        # C - contact frame

        # Constants:
        self.T_O_Cr = PyKDL.Frame(
            PyKDL.Rotation.RotY(math.radians(-90)) * PyKDL.Rotation.RotZ(math.radians(30)),
            PyKDL.Vector(0.5, -0.35, 0.985))
        self.T_O_Cl = PyKDL.Frame(
            PyKDL.Rotation.RotY(math.radians(-90)) * PyKDL.Rotation.RotZ(math.radians(30)),
            PyKDL.Vector(-0.5, -0.35, 0.985))

        # Point on the other end of the table - for height reference
        self.pt_ref_O = PyKDL.Vector(0, 0.35, 0.985)

        # Grasp-dependent:
        self.T_Er_C = None
        self.T_El_C = None

        # State:
        # Height equilibrium point
        self.height_eq = None


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


# def planWithOrientationConstraintsExample():
#     # The O frame is the base frame for orientation constraints
#     # The free DOF is rotation around X axis in the O frame
#     name_O_frame = 'frame_O'
#     req:ApplyPlanningScene_Request = ApplyPlanningScene.Request()
#     req.scene.is_diff = True
#     req.scene = planningSceneTransforms({name_O_frame:T_B_O})
#     result:ApplyPlanningScene_Response|None = self.planning_scene_cli.call(req)
#     if result is None:
#         self.get_logger().error(f'Service call failed')
#         return self.state_failure

#     ee_l_link_name = 'left_arm_7_link'
#     ee_r_link_name = 'right_arm_7_link'
#     pt_contact_r_B = T_B_O * self.support_task_params.pt_contact_r_O
#     pt_contact_l_B = T_B_O * self.support_task_params.pt_contact_l_O

#     pos_constr, ori_constr = createConstraints([
#         ('constrain local point', ee_r_link_name, self.support_task_params.pt_f_contact_E, pt_contact_r_B),
#         ('constrain orientation', ee_r_link_name, name_O_frame, self.support_task_params.M_contact_r_O, math.radians(30), math.radians(1), math.radians(1)),
#         ('constrain local point', ee_l_link_name, self.support_task_params.pt_f_contact_E, pt_contact_l_B),
#         ('constrain orientation', ee_l_link_name, name_O_frame, self.support_task_params.M_contact_l_O, math.radians(30), math.radians(1), math.radians(1))])
#     goal_constraints = [ Constraints(position_constraints=pos_constr,
#                                     orientation_constraints=ori_constr) ]

