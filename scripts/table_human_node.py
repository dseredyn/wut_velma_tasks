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

from geometry_msgs.msg import Pose, Transform, TransformStamped, Vector3, Point

import PyKDL

# For gripper control
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.duration import Duration

from gazebo_point_springs_interfaces.action import (FollowEquilibriumTrajectory,
                                                    FollowEquilibriumTrajectory_GetResult_Response)
from gazebo_point_springs_interfaces.msg import EquilibriumWaypoint

from wut_velma_tasks.common import (GazeboPoseThr, tfROStoKDL)

#
#
#
class HumanNode(Node):
    FsmState = Callable[[], Callable|None]

    def __init__(self) -> None:
        super().__init__("example_node")

        self.fsm_group = MutuallyExclusiveCallbackGroup()
        self.action_group = MutuallyExclusiveCallbackGroup()

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

        self.gz_pose_thr = GazeboPoseThr(self, ['can', 'table'])

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
        T_B_O = self.gz_pose_thr.getObjectPose(obj_name)
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

        # TODO: add rviz controls

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
    node = HumanNode()
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
