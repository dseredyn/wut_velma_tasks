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


from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformBroadcaster
from tf2_ros import TransformException # type: ignore
from rclpy.duration import Duration
import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration

import PyKDL

from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient

from wut_velma_tasks.common import (JointStateThread, GazeboPoseThr,
                                    poseKDLtoROS, tfROStoKDL)

# from builtin_interfaces.msg import Duration
from cartesian_impedance_msgs.action import (FollowCartesianImpedanceTrajectory,
                                FollowCartesianImpedanceTrajectory_GetResult_Response)
from cartesian_impedance_msgs.msg import CartesianTrajectoryPoint

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

        self.gz_pose_thr = GazeboPoseThr(self, ['can', 'table'])

        #
        # Cartesian impedance action
        #
        self.move_cart_action = ActionClient( self, FollowCartesianImpedanceTrajectory,
                                                "/cartesian_controller/follow_cartesian_impedance_trajectory",
                                                 callback_group=self.action_group )
        if not self.move_cart_action.wait_for_server(timeout_sec=10.0):
            raise Exception('MoveGroup is not available')
        # else:

        #
        # Initialize the FSM
        #
        self._FSM_next_state_timer = None

        self.get_logger().info(f'Initialization is done. Starting FSM.')

        # The initial state
        self.FSM_init(self.state_move_to_table)


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


    def state_move_to_table(self) -> FsmState:
        self.get_logger().info(
            f'Execution: motion to a given end-effector pose')

        goal = FollowCartesianImpedanceTrajectory.Goal()
        goal.ee_traj_r.header.frame_id = 'torso_base'
        goal.ee_traj_r.target_frame = 'right_arm_7_link'
        goal.ee_traj_r.points = []
        pt = CartesianTrajectoryPoint()
        pt.time_from_start = Duration(sec=3, nanosec=0)
        
        T_B_Er = PyKDL.Frame( PyKDL.Rotation.RotZ(math.radians(-90)) * PyKDL.Rotation.RotX(math.radians(-135)),
                                PyKDL.Vector(0.3, -0.4, 0.7))
        pt.pose = poseKDLtoROS(T_B_Er)
        goal.ee_traj_r.points.append(pt)

        # Waits until goal is accepted
        response = self.move_cart_action.send_goal(goal, feedback_callback=self.feedback_callback)
        if response is None:
            self.get_logger().error(f'result: None')
            return self.state_failure
        else:
            assert isinstance(response, FollowCartesianImpedanceTrajectory_GetResult_Response)
            self.get_logger().info(f'error_code: {response.result.error_code}, message: {response.result.error_string}')
            if (response.status == GoalStatus.STATUS_SUCCEEDED
                    and response.result.error_code == FollowCartesianImpedanceTrajectory.Result.SUCCESSFUL):
                self.get_logger().info(f'succeeded')
                return self.state_finish
            else:
                return self.state_failure


    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'desired: {feedback.ee_desired_r.pose}')
        self.get_logger().info(f'actual: {feedback.ee_actual_r.pose}')

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
