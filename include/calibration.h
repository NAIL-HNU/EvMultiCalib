#pragma once

#include <ros/ros.h>
#include <yaml-cpp/yaml.h>
#include <sensor_msgs/Imu.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <sophus/so3.hpp>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

#include "utils.h"
#include "spline.h"
#include "optimization_cost_functor.h"

#include <glog/logging.h>


namespace ev_calib {
    struct OptionsMethod {
        std::string imu_topic_;
        std::string ev_topic_;
        std::string rgb_topic_;
        std::string lidar_topic_;

        bool calib_imu_;
        bool calib_rgb_;
        bool calib_lidar_;

        // CCA Init
        double td_range_;
        int td_resolution_;

        // Optimization
        double event_weight_;
        double imu_weight_;
        double rgb_weight_;
        double lidar_weight_;
        int time_offset_bound_;
        double dt_knots_;
    };

    enum sensor_type_set { event, imu, rgb, lidar };

    class EventCalibration {
    public:
        EventCalibration(ros::NodeHandle &nh, std::string &rosbag_dir, std::string &params_dir, std::string &result_dir);
        ~EventCalibration();

        //private:
        ros::NodeHandle nh_;

        static void loadVelocityFromTxt(const std::string &motion_dir, std::vector<Eigen::Matrix<double, 1, 4> > &velocity);

        std::vector<std::pair<double, Eigen::Vector3d>> ev_angvel_, imu_angvel_, rgb_angvel_, lidar_angvel_;
        std::vector<RotationalDeltaWithTime> rgb_rot_, lidar_rot_;

        double imu_max_corr_, rgb_max_corr_, lidar_max_corr_; // CCA trace correlation

        double imu_time_offset_, rgb_time_offset_, lidar_time_offset_;
        Eigen::Quaterniond imu_extrin_rot_, rgb_extrin_rot_, lidar_extrin_rot_;
        double gyro_bias_[3];

        void calib_init(std::vector<std::pair<double, Eigen::Vector3d> > &ev_vel,
                        std::vector<std::pair<double, Eigen::Vector3d> > &obj_vel,
                        sensor_type_set sensor_type);

        static double CCA(const std::vector<std::pair<double, Eigen::Vector3d> > &ev_vel,
                          const std::vector<std::pair<double, Eigen::Vector3d> > &obj_vel,
                          Eigen::Matrix3d &R_oe);

        void calib_optim();

        OptionsMethod opts_{};
        std::ofstream ofile_;
    };
}
