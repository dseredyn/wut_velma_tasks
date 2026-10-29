cmake_minimum_required(VERSION 3.8)
project(wut_velma_tasks)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

#if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#  add_compile_options(-Wall -Wextra -Wpedantic)
#endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rcutils REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(tf2_ros REQUIRED)

ament_export_dependencies(
  controller_interface
  geometry_msgs
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  realtime_tools
  trajectory_msgs
)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/circle_trajectory_node.py
  scripts/example_node.py
  scripts/grasping_node.py
  scripts/table_grasp_node.py
  scripts/table_human_node.py
  DESTINATION lib/${PROJECT_NAME}
)

#install(DIRECTORY launch
#  DESTINATION share/${PROJECT_NAME}/
#)

ament_package()
