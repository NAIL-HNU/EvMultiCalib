#include <ros/ros.h>
#include <rosbag/bag.h>
#include <thread>
#include "calibration.h"
#include "event_angular_velocity_estimator.h"
#include "rgb_rotation_estimator.h"
#include "lidar_rotation_estimator.h"


int main(int argc, char *argv[]) {
    ros::init(argc, argv, "cca_calibration");
    ros::NodeHandle nh;

    std::string rosbag_dir, calib_config_dir, ev_config_dir, rgb_config_dir, lidar_config_dir, result_dir;
    nh.getParam("/evMultiCalib/bag_dir", rosbag_dir);
    nh.getParam("/evMultiCalib/calib_config_file", calib_config_dir);
    nh.getParam("/evMultiCalib/event_config_file", ev_config_dir);
    nh.getParam("/evMultiCalib/rgb_config_file", rgb_config_dir);
    nh.getParam("/evMultiCalib/lidar_config_file", lidar_config_dir);
    nh.getParam("/evMultiCalib/result_dir", result_dir);

    ev_calib::EventCalibration calibration(nh, rosbag_dir, calib_config_dir, result_dir);

    std::vector<std::thread> threads_motion;

    ev_angular_velocity_estimator::EvAngularVelocityEstimator ev_estimate(ev_config_dir);

    threads_motion.emplace_back([&ev_estimate, &rosbag_dir, &calibration]() {ev_estimate.loadData(rosbag_dir, calibration.opts_.ev_topic_); } );

    if (calibration.opts_.calib_imu_) {
        rosbag::Bag bag;
        std::string ros_topic = calibration.opts_.imu_topic_;
        bag.open(rosbag_dir, rosbag::bagmode::Read);
        rosbag::View imu_view(bag, rosbag::TopicQuery(ros_topic));
        for (const auto &m: imu_view) {
            if (m.getTopic() == ros_topic) {
                sensor_msgs::Imu::ConstPtr imu_msg = m.instantiate<sensor_msgs::Imu>();
                if (imu_msg != nullptr) {
                    Eigen::Vector3d vel_i(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
                    std::pair<double, Eigen::Vector3d> imu_vel_i(imu_msg->header.stamp.toSec(), vel_i);
                    calibration.imu_angvel_.push_back(imu_vel_i);
                }
            }
        }
        bag.close();
    }

    rgb_rotation::RgbRotationEstimate rgb_estimate;
    if (calibration.opts_.calib_rgb_) {
        rgb_estimate.params(rgb_config_dir);
        threads_motion.emplace_back([&rgb_estimate, &rosbag_dir, &calibration]() {rgb_estimate.loadData(rosbag_dir, calibration.opts_.rgb_topic_); } );
    }

    lidar_rotation::lidarRotationEstimate lidar_estimate;
    if (calibration.opts_.calib_lidar_) {
        lidar_estimate.params(lidar_config_dir);
        threads_motion.emplace_back([&lidar_estimate, &rosbag_dir, &calibration]() {lidar_estimate.loadData(rosbag_dir, calibration.opts_.lidar_topic_); } );
    }

    for (auto &t: threads_motion) {
        if (t.joinable()) {
            t.join();
        }
    }

    calibration.ev_angvel_ = ev_estimate.est_angvels_;
    calibration.rgb_angvel_ = rgb_estimate.rgb_angvel_;
    calibration.rgb_rot_ = rgb_estimate.rgb_rot_;
    calibration.lidar_angvel_ = lidar_estimate.lidar_angvel_;
    calibration.lidar_rot_ = lidar_estimate.lidar_rot_;


    std::vector<std::thread> threads_CCAinit;

    if (calibration.opts_.calib_imu_) {
        threads_CCAinit.emplace_back(
            [&calibration]() {
                calibration.calib_init(std::ref(calibration.ev_angvel_), std::ref(calibration.imu_angvel_),
                                       ev_calib::imu);
            }
        );
    }
    if (calibration.opts_.calib_rgb_) {
        threads_CCAinit.emplace_back(
            [&calibration]() {
                calibration.calib_init(std::ref(calibration.ev_angvel_), std::ref(calibration.rgb_angvel_),
                                       ev_calib::rgb);
            }
        );
    }
    if (calibration.opts_.calib_lidar_) {
        threads_CCAinit.emplace_back(
            [&calibration]() {
                calibration.calib_init(std::ref(calibration.ev_angvel_), std::ref(calibration.lidar_angvel_),
                                       ev_calib::lidar);
            }
        );
    }
    for (auto &t: threads_CCAinit) {
        if (t.joinable()) {
            t.join();
        }
    }

    calibration.calib_optim();

    if (calibration.opts_.calib_imu_) {
        calibration.ofile_ << "Event-IMU:" << std::endl << std::fixed << std::setprecision(10)
                << "time_offset: " << calibration.imu_time_offset_ << std::endl
                << "extrin_rot: " << calibration.imu_extrin_rot_.x() << ", " << calibration.imu_extrin_rot_.y() << ", "
                << calibration.imu_extrin_rot_.z() << ", " << calibration.imu_extrin_rot_.w() << std::endl
                << "gyro_bias: " << calibration.gyro_bias_[0] << ", " << calibration.gyro_bias_[1] << ", " <<
                calibration.gyro_bias_[2] << std::endl << std::endl;

        LOG(INFO) << "Event-IMU: time_offset = " << calibration.imu_time_offset_
                << ",   extrin_rot = " << calibration.imu_extrin_rot_.x() << ", " << calibration.imu_extrin_rot_.y() <<
                ", " << calibration.imu_extrin_rot_.z() << ", " << calibration.imu_extrin_rot_.w()
                << ",   gyro_bias: " << calibration.gyro_bias_[0] << ", " << calibration.gyro_bias_[1] << ", " <<
                calibration.gyro_bias_[2];
    }
    if (calibration.opts_.calib_rgb_) {
        calibration.ofile_ << "Event-RGB:" << std::endl << std::fixed << std::setprecision(10)
                << "time_offset: " << calibration.rgb_time_offset_ << std::endl
                << "extrin_rot: " << calibration.rgb_extrin_rot_.x() << ", " << calibration.rgb_extrin_rot_.y() << ", "
                << calibration.rgb_extrin_rot_.z() << ", " << calibration.rgb_extrin_rot_.w() << std::endl << std::endl;

        LOG(INFO) << "Event-RGB: time_offset = " << calibration.rgb_time_offset_
                << ",   extrin_rot = " << calibration.rgb_extrin_rot_.x() << ", " << calibration.rgb_extrin_rot_.y() <<
                ", " << calibration.rgb_extrin_rot_.z() << ", " << calibration.rgb_extrin_rot_.w();
    }
    if (calibration.opts_.calib_lidar_) {
        calibration.ofile_ << "Event-LiDAR:" << std::endl << std::fixed << std::setprecision(10)
                << "time_offset: " << calibration.lidar_time_offset_ << std::endl
                << "extrin_rot: " << calibration.lidar_extrin_rot_.x() << ", " << calibration.lidar_extrin_rot_.y() <<
                ", " << calibration.lidar_extrin_rot_.z() << ", " << calibration.lidar_extrin_rot_.w() << std::endl;

        LOG(INFO) << "Event-LiDAR: time_offset = " << calibration.lidar_time_offset_
                << ",   extrin_rot = " << calibration.lidar_extrin_rot_.x() << ", " << calibration.lidar_extrin_rot_.y()
                << ", " << calibration.lidar_extrin_rot_.z() << ", " << calibration.lidar_extrin_rot_.w();
    }

    return 0;
}
