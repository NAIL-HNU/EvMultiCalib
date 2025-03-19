cmake_minimum_required(VERSION 3.15)
project(evMultiCalib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17" )
set(CMAKE_BUILD_TYPE "RELEASE")

find_package(OpenCV 3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rosbag
        image_transport
        cv_bridge
        sensor_msgs)
catkin_package(
        INCLUDE_DIRS
        include
        ${catkin_INCLUDE_DIRS})

find_library(SMALL_GICP_LIB small_gicp PATHS thirdparty/small_gicp/lib)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        thirdparty/small_gicp/include
        thirdparty/basalt-headers/thirdparty/Sophus
        thirdparty/basalt-headers/thirdparty/cereal/include
        thirdparty/basalt-headers/include)

set(HEADERS
        include/calibration.h
        include/spline.h
        include/optimization_cost_functor.h
        include/rgb_rotation_estimator.h
        include/lidar_rotation_estimator.h
        include/event_angular_velocity_estimator.h
        include/utils.h
        include/dvs_msg/Event.h
        include/dvs_msg/EventArray.h)

set(SOURCES
        src/ros_node.cpp
        src/calibration.cpp
        src/rgb_rotation_estimator.cpp
        src/lidar_rotation_estimator.cpp
        src/event_angular_velocity_estimator.cpp
        src/normal_flow.cpp
        src/utils.cpp)

add_library(${PROJECT_NAME}_LIB ${HEADERS} ${SOURCES})

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${SMALL_GICP_LIB} yaml-cpp glog::glog)
