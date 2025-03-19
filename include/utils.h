#pragma once

#include <string>
#include <Eigen/Dense>
#include <sophus/so3.hpp>
#include <iostream>

#include <cv_bridge/cv_bridge.h>

#include <tf/tf.h>
#include <tf/tfMessage.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>

#include <dvs_msg/Event.h>
#include <dvs_msg/EventArray.h>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <random>

#include <math.h>

using EventQueue = std::deque<dvs_msgs::Event>;

inline static EventQueue::iterator EventBuffer_lower_bound(EventQueue &eb, ros::Time &t) {
    return std::lower_bound(eb.begin(), eb.end(), t,
                            [](const dvs_msgs::Event &e, const ros::Time &t) { return e.ts.toSec() < t.toSec(); });
}

inline static EventQueue::iterator EventBuffer_upper_bound(EventQueue &eb, ros::Time &t) {
    return std::upper_bound(eb.begin(), eb.end(), t,
                            [](const ros::Time &t, const dvs_msgs::Event &e) { return t.toSec() < e.ts.toSec(); });
}

struct RotationalDeltaWithTime {
    Sophus::SO3d rotation_delta;
    double t_prev;
    double t_cur;
};

class PerspectiveCamera {
public:
    PerspectiveCamera();

    virtual ~PerspectiveCamera();

    using Ptr = std::shared_ptr<PerspectiveCamera>;

    void setIntrinsicParameters(
            size_t width, size_t height,
            std::string &cameraName,
            std::string &distortion_model,
            std::vector<double> &vD,
            std::vector<double> &vK);

public:
    int width_, height_;
    std::string cameraName_;
    std::string distortion_model_;
    cv::Mat D_;
    cv::Mat K_;
    cv::Mat precomputed_undistorted_x_;
    cv::Mat precomputed_undistorted_y_;
};
