from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    moveit_config = (
        MoveItConfigsBuilder("velma", package_name="velma_moveit_config")
        .robot_description()
        .robot_description_semantic()
        .robot_description_kinematics()
        .joint_limits()
        .trajectory_execution(
            file_path="config/moveit_controllers.yaml",
            moveit_manage_controllers=False,
        )
        .moveit_cpp(
            file_path="config/moveit_py.yaml",
        )
        .to_moveit_configs()
    )

    circle_node = Node(
        package="wut_velma_tasks",
        executable="circle_trajectory_node.py",
        name="circle_trajectory_node",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                "use_sim_time": use_sim_time,

                # "center_xyz": [0.45, 0.20, 0.45],
                # "radius": 0.03,
                # "period": 8.0,
                # "publish_period": 0.5,
                # "horizon": 2.0,
                # "waypoint_dt": 0.1,
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        circle_node,
    ])
