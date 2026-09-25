# wut_velma_tasks

## Install

Create workspace:
```bash
cd
mkdir -p ws_velma/src
cd ~/ws_velma/src
git clone -b ros2 https://github.com/RCPRG-ros-pkg/velma_robot.git
git clone https://github.com/dseredyn/wut_velma_gazebo.git
git clone https://github.com/dseredyn/wut_velma_effort_controller.git
git clone https://github.com/dseredyn/wut_velma_tasks.git
```

Build:
```bash
source /opt/ros/jazzy/setup.bash
cd ~/ws_velma
clear && python3 -m colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

## Run

Simulation + control system:
```bash
ros2 launch wut_velma_gazebo start_system.launch.py
```

Motion generation:
```bash
ros2 launch wut_velma_tasks circle_trajectory.launch.py
```
