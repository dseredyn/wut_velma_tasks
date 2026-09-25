#!/usr/bin/env python3
"""
circle_trajectory_node.py

ROS 2 Jazzy node:
- co 0.5 s publikuje nową trajektorię JointTrajectory,
- trajektorie są kolejnymi fragmentami tej samej trajektorii okresowej,
- pozycje przegubów wynikają z IK dla położenia TCP po małym okręgu,
- w punktach są zadane pozycje i prędkości.

Wymagania:
  sudo apt install ros-jazzy-moveit-py ros-jazzy-trajectory-msgs ros-jazzy-geometry-msgs
  pip install numpy

Uruchamiaj razem z pełną konfiguracją MoveIt dla robota:
robot_description, robot_description_semantic, kinematics.yaml itd.
"""

import time

from rclpy.task import Future
from sensor_msgs.msg import JointState
from rclpy.impl.rcutils_logger import RcutilsLogger

from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException # type: ignore

import math
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from moveit.planning import MoveItPy, PlanningComponent, PlanRequestParameters, MultiPipelinePlanRequestParameters
from moveit.core.robot_state import RobotState
from moveit.core.robot_model import RobotModel, JointModelGroup
from moveit.core.planning_interface import MotionPlanResponse
from moveit.core.controller_manager import ExecutionStatus
from moveit.core.robot_trajectory import RobotTrajectory

from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory

# States / procedures:
# constructor ContinuousIkCircleTrajectoryPublisher()
#   - initialization
#   - next state: state_move_to_initial_pose
# state_move_to_initial_pose: startup_timer, 0.1s
#   - get current joint state
#   - send action goal: move to the initial pose for trajectory execution
#   - next state: circular_trajectory
# circular_trajectory: publish_timer, 0.5s
#   - send a new 2s trajectory every 0.5s

def plan(
    planning_component:PlanningComponent,
    logger:RcutilsLogger,
    single_plan_parameters:Optional[PlanRequestParameters]=None,
    multi_plan_parameters:Optional[MultiPipelinePlanRequestParameters]=None,
) -> Optional[RobotTrajectory]:
    """Helper function to plan a motion."""
    # plan to goal
    logger.info("Planning trajectory")
    if multi_plan_parameters is not None:
        plan_result:MotionPlanResponse = planning_component.plan(
            multi_plan_parameters=multi_plan_parameters
        )
    elif single_plan_parameters is not None:
        plan_result:MotionPlanResponse = planning_component.plan(
            single_plan_parameters=single_plan_parameters
        )
    else:
        # https://docs.ros.org/en/noetic/api/moveit_core/html/structplanning__interface_1_1MotionPlanResponse.html
        plan_result:MotionPlanResponse = planning_component.plan()

    if plan_result:
        logger.info("Trajectory is ready")
        return plan_result.trajectory
    else:
        logger.error(f'Planning failed: {plan_result.error_code}')
        return None


def execute(
    moveit:MoveItPy,
    trajectory:RobotTrajectory,
    logger:RcutilsLogger
) -> bool:
    """Helper function to execute a motion."""
    logger.info("Executing trajectory")
    # https://moveit.picknik.ai/main/api/html/structmoveit__controller__manager_1_1ExecutionStatus.html
    result:ExecutionStatus = moveit.execute(trajectory, controllers=[])
    logger.info( f'Execute status: "{result.status}"' )
    return bool(result)


# def plan_and_execute(
#     moveit:MoveItPy,
#     planning_component:PlanningComponent,
#     logger:RcutilsLogger,
#     single_plan_parameters:Optional[PlanRequestParameters]=None,
#     multi_plan_parameters:Optional[MultiPipelinePlanRequestParameters]=None,
# ):
#     """Helper function to plan and execute a motion."""
#     # plan to goal
#     logger.info("Planning trajectory")
#     if multi_plan_parameters is not None:
#         plan_result:MotionPlanResponse = planning_component.plan(
#             multi_plan_parameters=multi_plan_parameters
#         )
#     elif single_plan_parameters is not None:
#         plan_result:MotionPlanResponse = planning_component.plan(
#             single_plan_parameters=single_plan_parameters
#         )
#     else:
#         plan_result:MotionPlanResponse = planning_component.plan()

#     # execute the plan
#     if plan_result:
#         logger.info("Executing plan")
#         robot_trajectory = plan_result.trajectory
#         # https://moveit.picknik.ai/main/api/html/structmoveit__controller__manager_1_1ExecutionStatus.html
#         result:ExecutionStatus = moveit.execute(robot_trajectory, controllers=[])
#         logger.info( f'execute status: "{result.status}"' )
#     else:
#         logger.error("Planning failed")



class ContinuousIkCircleTrajectoryPublisher(Node):
    def __init__(self) -> None:
        super().__init__("continuous_ik_circle_trajectory_publisher")

        # Topic: <controller_name>/joint_trajectory
        self.controller_topic = "/arms_torso_trajectory_controller/joint_trajectory"
        self.joint_states_topic = "/joint_states"

        # Names are defined in the SRDF.
        self.group_name = "left_arm"
        self.tip_link = "left_arm_7_link"
        self.base_frame = "torso_base"

        self.center = np.array( [0.55, 0.35, 1.05], dtype=float)
        self.radius = 0.1
        self.period = 8.0
        self.omega = 2.0 * math.pi / self.period

        self.orientation_xyzw = np.array( [0.224, -0.343, 0.911, 0.052], dtype=float)

        self.publish_period = 0.5
        self.horizon = 2.0
        self.waypoint_dt = 0.1
        self.ik_timeout = 0.03

        assert self.horizon > self.publish_period
        assert self.waypoint_dt > 0.0
        assert self.period > 0.0
        assert self.radius > 0.0

        self.startup_timeout = 10.0

        self.initialized_from_current_state = False
        self.current_joint_positions = {}
        self.startup_wall_start_time = time.monotonic()

        self.moveit = MoveItPy(
            node_name="continuous_ik_circle_moveit_py",
        )

        self.robot_model:RobotModel = self.moveit.get_robot_model()
        self.joint_model_group:JointModelGroup = self.robot_model.get_joint_model_group( self.group_name )

        # Get joint_names from MoveIt JointModelGroup.
        self.joint_names = list(self.joint_model_group.active_joint_model_names)

        self.robot_state = RobotState(self.robot_model)
        self.robot_state.set_to_default_values()

        self.seed_for_next_publish: Optional[np.ndarray] = None

        # Joint state subscriber
        self.sub_joint_state = self.create_subscription(
            JointState,
            self.joint_states_topic,
            self.joint_state_callback,
            50,
        )

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Commanded trajectory publisher
        self.pub_traj = self.create_publisher(
            JointTrajectory,
            self.controller_topic,
            10,
        )

        self.start_time:Optional[Time] = None

        self.publish_timer = None

        self.startup_timer = self.create_timer(
            0.1,
            self.state_move_to_initial_pose,
        )

        self.get_logger().info(
            "Publikuję ciągłe trajektorie IK na topicu "
            f"'{self.controller_topic}' dla grupy '{self.group_name}', "
            f"tip_link='{self.tip_link}', joint_names={self.joint_names}"
        )

    def plan_and_execute(self, robot_state:RobotState) -> bool:
        robot_arm:PlanningComponent = self.moveit.get_planning_component(self.group_name)
        robot_arm.set_start_state_to_current_state()
        robot_arm.set_goal_state(robot_state=robot_state)

        # plan to goal
        param = PlanRequestParameters(self.moveit, '')
        param.max_acceleration_scaling_factor = 0.2
        param.max_velocity_scaling_factor = 0.2
        trajectory = plan(robot_arm, self.get_logger(), single_plan_parameters=param)
        if trajectory is None:
            return False
        else:
            return execute(self.moveit, trajectory, self.get_logger())
    
    def action_traj_goal_cb(self, future:Future):
        print('The trajectory is done')
        self.request_clean_stop('ok')

    def joint_state_callback(self, msg: JointState) -> None:
        for name, position in zip(msg.name, msg.position):
            self.current_joint_positions[name] = position

    def get_current_group_positions(self) -> Optional[np.ndarray]:
        q = []

        missing = []
        for joint_name in self.joint_names:
            if joint_name not in self.current_joint_positions:
                missing.append(joint_name)
            else:
                q.append(self.current_joint_positions[joint_name])

        if missing:
            self.get_logger().debug(
                "Czekam na joint_states dla: " + ", ".join(missing)
            )
            return None

        return np.asarray(q, dtype=float)

    def get_current_tcp_position(self) -> Optional[np.ndarray]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tip_link,
                Time(),
            )
        except TransformException as ex:
            self.get_logger().debug(
                f"Czekam na TF {self.base_frame} -> {self.tip_link}: {ex}"
            )
            return None

        t = transform.transform.translation

        return np.array(
            [t.x, t.y, t.z],
            dtype=float,
        )

    def state_move_to_initial_pose(self) -> None:
        self.get_logger().info('state_move_to_initial_pose')
        if self.initialized_from_current_state:
            return

        elapsed = time.monotonic() - self.startup_wall_start_time

        current_q = self.get_current_group_positions()
        current_tcp = self.get_current_tcp_position()

        if current_q is None or current_tcp is None:
            if elapsed > self.startup_timeout:
                self.request_clean_stop(
                    "Could not get the current state from /joint_states "
                    f"and TF {self.base_frame} -> {self.tip_link} is {elapsed:.1f}s > "
                    f"{self.startup_timeout:.1f} s. Aborting."
                )
            return

        start_pose = self.desired_pose(0.0)

        # Calculate IK for the first point on the circle.
        # Use the current configuration as the seed.
        first_q = self.solve_ik(start_pose, current_q)

        if first_q is None:
            self.get_logger().error(
                "IK failed for the current configuration as the seed. Aborting."
            )
            self.request_clean_stop(
                "IK failed for the current configuration as the seed. Aborting."
            )
            return

        self.robot_state.set_joint_group_positions(self.group_name, first_q)
        if self.plan_and_execute(self.robot_state):
            # The start-up procedure is done.
            if self.startup_timer is not None:
                self.startup_timer.cancel()
                self.startup_timer = None

            # IK seed is the current configuration.
            self.seed_for_next_publish = first_q.copy()

            self.initialized_from_current_state = True

            self.publish_timer = self.create_timer(
                self.publish_period,
                self.circular_trajectory,
            )
        else:
            self.get_logger().error(
                            "Could not plan or execute motion. Retrying."
                        )

    def desired_pose(self, t: float) -> Pose:
        """Desired TCP pose fot the given relative time t."""
        phi = self.omega * t

        pose = Pose()
        pose.position.x = float(self.center[0] + self.radius * math.cos(phi))
        pose.position.y = float(self.center[1] + self.radius * math.sin(phi))
        pose.position.z = float(self.center[2])

        pose.orientation.x = float(self.orientation_xyzw[0])
        pose.orientation.y = float(self.orientation_xyzw[1])
        pose.orientation.z = float(self.orientation_xyzw[2])
        pose.orientation.w = float(self.orientation_xyzw[3])
        return pose

    def solve_ik(
        self,
        pose: Pose,
        seed: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Solves IK, using a specific seed."""
        self.robot_state.set_joint_group_positions(
            self.group_name,
            np.asarray(seed, dtype=float),
        )
        self.robot_state.update()

        ok = self.robot_state.set_from_ik(
            self.group_name,
            pose,
            self.tip_link,
            self.ik_timeout,
        )

        if not ok:
            return None

        self.robot_state.update()
        q = np.asarray(
            self.robot_state.get_joint_group_positions(self.group_name),
            dtype=float,
        )

        return q

    @staticmethod
    def duration_msg(seconds: float) -> Duration:
        seconds = max(0.0, float(seconds))
        sec = int(math.floor(seconds))
        nanosec = int(round((seconds - sec) * 1e9))

        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000

        msg = Duration()
        msg.sec = sec
        msg.nanosec = nanosec
        return msg

    def elapsed_trajectory_time(self) -> float:
        if self.start_time is None:
            self.start_time = self.get_clock().now()
        # else:

        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds * 1e-9

        # In case of simulation time reset.
        if elapsed < 0.0:
            self.start_time = now
            return 0.0

        return elapsed

    def circular_trajectory(self) -> None:
        if not self.initialized_from_current_state:
            return
        now = self.get_clock().now()
        t0 = self.elapsed_trajectory_time()

        number_of_points = max(2, int(math.floor(self.horizon / self.waypoint_dt)) + 1)

        qs = []
        assert not self.seed_for_next_publish is None

        seed = self.seed_for_next_publish

        for i in range(number_of_points):
            t_rel = i * self.waypoint_dt
            t_abs = t0 + t_rel

            pose = self.desired_pose(t_abs)
            q = self.solve_ik(pose, seed)

            if q is None:
                self.get_logger().warn(
                    "IK nie znalazło rozwiązania dla jednego z punktów. "
                    "Nie publikuję tej trajektorii."
                )
                return

            qs.append(q)
            seed = q

        q_mat = np.vstack(qs)

        # Calculate velocities.
        qdot_mat = np.gradient(q_mat, self.waypoint_dt, axis=0, edge_order=1)
        qdot_mat[-1, :] = 0.0

        traj = JointTrajectory()
        traj.header.stamp = now.to_msg()
        traj.header.frame_id = self.base_frame
        traj.joint_names = self.joint_names
        traj.points = []

        for i in range(number_of_points):
            point = JointTrajectoryPoint()
            point.positions = [float(x) for x in q_mat[i, :]]
            point.velocities = [float(x) for x in qdot_mat[i, :]]
            point.time_from_start = self.duration_msg(i * self.waypoint_dt)
            traj.points.append(point)

        self.pub_traj.publish(traj)

        # Seed for the next iteration: expected configuration after publish_period.
        next_seed_index = int(round(self.publish_period / self.waypoint_dt))
        next_seed_index = min(max(next_seed_index, 0), number_of_points - 1)
        self.seed_for_next_publish = q_mat[next_seed_index, :].copy()

    def request_clean_stop(self, reason: str) -> None:
        self.get_logger().error(reason)

        if self.publish_timer is not None:
            self.publish_timer.cancel()

        if self.startup_timer is not None:
            self.startup_timer.cancel()

        # Shutdown.
        try:
            rclpy.try_shutdown()
        except:
            pass

def main(args=None) -> None:
    rclpy.init(args=args)
    node = ContinuousIkCircleTrajectoryPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('User interrupt')
        pass

    node.destroy_node()
    rclpy.try_shutdown()

if __name__ == "__main__":
    main()
