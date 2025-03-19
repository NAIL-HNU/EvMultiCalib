Temporal and Rotational Calibration for Event-Centric Multi-Sensor Systems
===========
This repository delivers a multi-sensor temporal and rotational calibration toolbox that facilitates the calibration of event cameras with IMUs, frame-based cameras, and LiDARs.
This is a targetless method.
Departing from conventional calibration approaches that rely on event-to-frame conversion, our method directly estimates normal flow from raw event streams and derives angular velocity via motion field equations.
Concurrently, frame-based camera motion is extracted through the feature-based visual odometry, while LiDAR motion is recovered via Generalized-ICP.
The calibration pipeline employs canonical correlation analysis (CCA) to initialize the time offset and rotation extrinsic by exploiting motion correlations, followed by a joint nonlinear optimization refining the extrinsic parameters on SO(3) spline.

#  1. Installation

We have tested this toolbox on machines with the following configurations
* Ubuntu 20.04 LTS + ROS Noetic + OpenCV 3.4.16 + Eigen 3.3.7 + Ceres 1.14.0

## 1.1 Dependencies Installation

    $ sudo apt install libpcl-dev pcl-tools ros-noetic-pcl-ros ros-noetic-pcl-conversions ros-noetic-tf-conversions

## 1.2 Compile the project

Make sure you have already created a catkin workspace.

    $ cd catkin_ws/src
    $ git clone https://github.com/NAIL-HNU/EvMultiCalib.git
    $ cd ..
    $ catkin_make
    $ source ./devel/setup.bash

# 2. Usage

This project requires a rosbag with **pure rotational motion for at least 30 seconds** as input.
Open a terminal and run the command:

    $ roslaunch evMultiCalib evMultiCalib.launch 

The calibration results will be saved in the following format: 

    $ time offset, quaternion.x, quaternion.y, quaternion.z, quaternion.w
