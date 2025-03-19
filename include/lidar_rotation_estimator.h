#pragma once

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <rosbag/message_instance.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <small_gicp/benchmark/read_points.hpp>
#include <small_gicp/registration/registration_helper.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include "utils.h"

#include <glog/logging.h>

namespace lidar_rotation {

struct OptionsMethod {
    small_gicp::RegistrationSetting gicp_setting_;
    bool b_motion_compensation_;
};

class lidarRotationEstimate {
public:
    lidarRotationEstimate(const std::string& params_dir);
    lidarRotationEstimate();
    ~lidarRotationEstimate();
    void params(const std::string &paramsPath);

    std::vector<RotationalDeltaWithTime> lidar_rot_;
    std::vector<std::pair<double, Eigen::Vector3d>> lidar_angvel_;
    std::vector<ros::Time> timestamp1_;
    std::vector<ros::Time> timestamp2_;
    void loadData(const std::string &rosbag_dir, const std::string &ros_topic);

private:
    std::pair<small_gicp::PointCloud::Ptr, std::shared_ptr<small_gicp::KdTree<small_gicp::PointCloud> > > source_;
    std::pair<small_gicp::PointCloud::Ptr, std::shared_ptr<small_gicp::KdTree<small_gicp::PointCloud> > > target_;
    std::vector<ros::Time> point_timestamp_;

    static void AddCloud(const sensor_msgs::PointCloud2 &lidar_msg,
                         std::vector<Eigen::Vector4f> &point_set,
                         std::vector<ros::Time> &point_timestamp);

    OptionsMethod opts_;
};

} // namespace lidar_velocity