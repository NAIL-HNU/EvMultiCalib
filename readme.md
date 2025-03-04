EvMultiCalib
===========

#  Instructions

## Requirements:
* [Eigen3](https://eigen.tuxfamily.org/dox/)
* [OpenCV 3](https://opencv.org/releases/)
* [ceres](https://github.com/ceres-solver/ceres-solver) = 1.14.0
* [rpg_dvs_ros](https://github.com/uzh-rpg/rpg_dvs_ros.git)

## Compile the project:


    $ cd catkin_ws/src
    $ git clone https://github.com/NAIL-HNU/EvMultiCalib
    $ cd ..
    $ catkin build
    $ source ./devel/setup.bash
    $ roslaunch evMultiCalib evMultiCalib.launch 