#!/usr/bin/env python3
"""
example_node.py

Basic motions for WUT Velma.
Simple implementation of FSM.
"""

import copy
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
from geometry_msgs.msg import Pose, Transform, TransformStamped, Vector3, Point
from visualization_msgs.msg import MarkerArray, Marker

import PyKDL

# For gripper control
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.duration import Duration

from gazebo_point_springs_interfaces.action import (FollowEquilibriumTrajectory,
                                                    FollowEquilibriumTrajectory_GetResult_Response)
from gazebo_point_springs_interfaces.msg import (EquilibriumWaypoint, PointSpringsState,
    PointSpringState)

from wut_velma_tasks.common import (GazeboPoseThr, tfROStoKDL, poseKDLtoROS, quaternionKDLtoROS,
                                    pointROStoKDL, poseRosToKDL)

from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
)

from std_msgs.msg import ColorRGBA

class InteractiveMarkerServerInterface:
    def __init__(self, node:Node, T_B_M:PyKDL.Frame, width:float):
        self._lock = Lock()
        self.node = node
        frame_id = 'world'

        self._T_B_M = T_B_M
        self.width = width

        self.server = InteractiveMarkerServer(self.node, 'human_action')

        marker = InteractiveMarker()
        marker.header.frame_id = frame_id
        marker.name = "target"
        marker.description = "Desired pose"
        marker.scale = 0.4

        # Początkowa poza względem frame_id.
        marker.pose = poseKDLtoROS(T_B_M)

        cube = Marker()
        cube.type = Marker.CUBE
        cube.pose = poseKDLtoROS(PyKDL.Frame())
        cube.scale = Vector3(x=width, y=0.01, z=0.01)
        cube.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0)

        visual = InteractiveMarkerControl()
        visual.name = "visual"
        visual.always_visible = True
        visual.interaction_mode = InteractiveMarkerControl.NONE
        visual.markers = [cube]
        marker.controls = [visual]

        # Handle axis is the local X axis.
        orientations = {
            "x": PyKDL.Rotation(),
            "y": PyKDL.Rotation.RotZ(math.radians(90)),
            "z": PyKDL.Rotation.RotY(math.radians(-90)),
        }

        for axis, R in orientations.items():
            for prefix, mode in (("move", InteractiveMarkerControl.MOVE_AXIS),
                                        ("rotate", InteractiveMarkerControl.ROTATE_AXIS)):
                control = InteractiveMarkerControl(name=f"{prefix}_{axis}",
                    interaction_mode=mode,
                    orientation_mode=InteractiveMarkerControl.FIXED,
                    orientation=quaternionKDLtoROS(R))
                marker.controls.append(control)

        self.server.insert(marker, feedback_callback=self.feedback_callback)
        self.server.applyChanges()

        self.node.get_logger().info(
            f"Marker utworzony w układzie '{frame_id}'. "
            "RViz2: dodaj InteractiveMarkers, "
            "Update Topic = /human_action/update"
        )

    def feedback_callback(self, feedback: InteractiveMarkerFeedback):
        if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            with self._lock:
                self._T_B_M = poseRosToKDL(feedback.pose)

    def getPose(self) -> PyKDL.Frame:
        with self._lock:
            result = copy.deepcopy(self._T_B_M)
        return result

    def getPoints(self) -> tuple[PyKDL.Vector, PyKDL.Vector]:
        # left, right
        T_B_M = self.getPose()
        return (T_B_M * PyKDL.Vector(self.width/2, 0, 0), T_B_M * PyKDL.Vector(-self.width/2, 0, 0))


class TableHumanInterface:
    def __init__(self, node:Node):
        self._lock = Lock()
        self._data = {}
        self.node = node
        self.group = MutuallyExclusiveCallbackGroup()
        self.sub = self.node.create_subscription(PointSpringsState, '/table_human/state',
                                                self._callback, 10, callback_group=self.group)

    def _callback(self, msg: PointSpringsState) -> None:
        with self._lock:
            for pt in msg.points:
                assert isinstance(pt, PointSpringState)
                self._data[pt.name] = (pointROStoKDL(pt.position),
                                       pointROStoKDL(pt.equilibrium_position))

    def getPoints(self) -> dict:
        with self._lock:
            result = copy.deepcopy(self._data)
        return result


#
#
#
class HumanNode(Node):
    FsmState = Callable[[], Callable|None]

    def __init__(self) -> None:
        super().__init__("example_node")

        self.fsm_group = MutuallyExclusiveCallbackGroup()
        self.action_group = MutuallyExclusiveCallbackGroup()
        self.timer_group = MutuallyExclusiveCallbackGroup()

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

        self.int_markers = None
        self.gz_pose_thr = GazeboPoseThr(self, ['can', 'table'])
        self.human_int = TableHumanInterface(self)

        self.prev_pts = None

        # Visual marker publisher
        self.publisher_vis = self.create_publisher(MarkerArray, '/task_markers', 10)

        self._pose_timer = self.create_timer(0.1, self._timer_callback,
                                                        callback_group=self.timer_group)

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


    def _timer_callback(self):
        pts = self.human_int.getPoints()
        # self.get_logger().info(f'{pts}')
        if not 'hand_left' in pts or not 'hand_right' in pts:
            return
        # else:

        pt_eq_l:PyKDL.Vector = pts['hand_left'][1]
        pt_eq_r:PyKDL.Vector = pts['hand_right'][1]
        if self.int_markers is None:
            ax = (pt_eq_l-pt_eq_r)
            width = ax.Norm()
            ax.Normalize()
            az = PyKDL.Vector(0, 0, 1)
            ay = az * ax
            az = ax * ay
            ay.Normalize()
            az.Normalize()
            
            # Initialize interactive markers
            self.int_markers = InteractiveMarkerServerInterface(self,
                            PyKDL.Frame(PyKDL.Rotation(ax, ay, az), (pt_eq_l + pt_eq_r)/2), width)
            self.get_logger().info('Added interactive markers')

        msg = MarkerArray()
        msg.markers = []
        scale = Vector3(x=0.04, y=0.04, z=0.04)
        colors = {'red':ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
                  'green':ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
                  'blue':ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
                  'yellow':ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
        }
        msg.markers.append( Marker(header=Header(frame_id='world'), ns='human', id=0, type=Marker.SPHERE,
                    action=Marker.ADD, pose=poseKDLtoROS(PyKDL.Frame(pt_eq_l)), scale=scale, color=colors['blue']) )

        msg.markers.append( Marker(header=Header(frame_id='world'), ns='human', id=1, type=Marker.SPHERE,
                    action=Marker.ADD, pose=poseKDLtoROS(PyKDL.Frame(pt_eq_r)), scale=scale, color=colors['blue']) )

        self.publisher_vis.publish(msg)

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

        if self.int_markers is None:
            # Not ready yet
            return self.state_lift_table
        # else:

        pl_B, pr_B = self.int_markers.getPoints()

        # Calculate movement time
        max_speed = 0.1 # m/s

        if self.prev_pts is None:
            diff_l = PyKDL.Vector()
            diff_r = PyKDL.Vector()
        else:
            diff_l = pl_B - self.prev_pts[0]
            diff_r = pr_B - self.prev_pts[1]
        self.prev_pts = (pl_B, pr_B)

        max_dist = max( diff_l.Norm(), diff_r.Norm() )

        mov_time = max(0.2, max_dist / max_speed)

        obj_name = 'table'
        T_B_O = self.gz_pose_thr.getObjectPose(obj_name)
        if T_B_O is None:
            self.get_logger().info(f'Could not get pose of object "{obj_name}"')
            return self.state_lift_table

        goal = FollowEquilibriumTrajectory.Goal()
        goal.header.frame_id = 'world'
        goal.point_names = ['hand_left', 'hand_right']
        goal.points = [EquilibriumWaypoint(time_from_start=Duration(seconds=mov_time).to_msg(),
                                           positions=[Point(x=pl_B.x(), y=pl_B.y(), z=pl_B.z()),
                                                      Point(x=pr_B.x(), y=pr_B.y(), z=pr_B.z())])]

        # Waits until goal is accepted and result is received
        response:FollowEquilibriumTrajectory_GetResult_Response|None = self.human_client.send_goal(goal)
        if response is None:
            self.get_logger().error(f'result: None')
            return self.state_failure
        else:
            assert isinstance(response, FollowEquilibriumTrajectory_GetResult_Response)

            if (response.status == GoalStatus.STATUS_SUCCEEDED
                    and response.result.code == FollowEquilibriumTrajectory.Result.SUCCESS):
                self.get_logger().info(f'succeeded')
                return self.state_lift_table
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







class MarkerServerNode(Node):
    def __init__(self):
        super().__init__("example_marker_server")

        self.declare_parameter("frame_id", "world")
        frame_id = self.get_parameter("frame_id").value



def main(args=None):
    rclpy.init(args=args)
    node = MarkerServerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()