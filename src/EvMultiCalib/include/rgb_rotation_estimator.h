#pragma once

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <rosbag/message_instance.h>
#include <sensor_msgs/Image.h>
#include <yaml-cpp/yaml.h>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "utils.h"
#include <glog/logging.h>

# define SAMPLE_SIZE 3

namespace rgb_rotation {

    struct featureObj {
        cv::Mat img_;
        ros::Time timestamp_;
        std::vector<cv::KeyPoint> keypoints_;
        //    std::vector<DescType> descriptors_;
        cv::Mat descriptors_;

        featureObj &operator=(const featureObj &other) {
            if (this != &other) {
                img_ = other.img_;
                timestamp_ = other.timestamp_;
                keypoints_ = other.keypoints_;
                descriptors_ = other.descriptors_;
            }
            return *this;
        }
    };

    typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > VecVector2d;
    typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > VecVector3d;

    inline cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
        return {
            (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
            (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
        };
    }

    struct OptionsMethod {
        double matches_filter_min_dist_;
        double inlier_distance_thres_;
        double inlier_ratio_thres_;
        int inlier_count_thres_;
        int ransac_max_iter_;
        double filter_threshold_;
    };

    class RgbRotationEstimate {
    public:
        RgbRotationEstimate(const std::string& params_dir);
        RgbRotationEstimate();
        ~RgbRotationEstimate();
        void params(const std::string &paramsPath);

        std::vector<RotationalDeltaWithTime> rgb_rot_;
        std::vector<std::pair<double, Eigen::Vector3d>> rgb_angvel_;
        void loadData(const std::string &rosbag_dir, const std::string &ros_topic);

    private:
        featureObj featureObj_firstImg_;
        bool rotationEstimate(const featureObj &featureObj_firstImg, const featureObj &featureObj_secondImg,
                              const std::vector<cv::DMatch> &matches);
        double minReprojectionError(const cv::Mat &K, Sophus::SO3d &pose,
                                    const VecVector3d &points_3d, const VecVector2d &points_2d);

        float scale_;
        OptionsMethod opts_{};
        PerspectiveCamera::Ptr pCam_;

    };
} // namespace rgb_rotation
