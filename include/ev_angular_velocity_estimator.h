#pragma once

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include "utils.h"

#include "glog/logging.h"

namespace ev_angular_velocity_estimator
{
    struct OptionsMethod {
        int PATCH_SIZE;
        int HALF_PATCH_SIZE;
        double PATCH_MEAN_TOLERANCE;
        double NEIGHBOR_CNT_THRESHOLD;
        double TERMINATE_INLIER_RATIO;

        double PLANE_TOLERANCE;
        int CONSTANT_EVENT_NUMBER;

        double REFRACTORY_PERIOD;
    };

    class EvAngularVelocityEstimator
    {
    public:
        EvAngularVelocityEstimator(const std::string& params_dir);
        ~EvAngularVelocityEstimator();
        void params(const std::string &paramsPath);

        void loadData(const std::string& rosbag_dir, const std::string& ev_topic);
        void processData();

        bool calculateNormalFlow();
        bool planeFitting(int pos_x, int pos_y, Eigen::MatrixXd &x_hat, double &cov);
        bool randomSelect();

        bool calculateAngularVelocity();
        bool solveAngularVelocityRansac(const Eigen::MatrixXd &A, const Eigen::VectorXd &b, Eigen::Vector3d &est_angvel);

        OptionsMethod opts_;
        PerspectiveCamera::Ptr pCam_;

        EventQueue events_;
        ros::Time t_end_;
        ros::Time t_begin_;
        double frame_duration_;

        cv::Mat last_timestamp_map_;
        cv::Mat original_last_timestamp_map_;
        cv::Mat undistorted_last_timestamp_map_;

        Eigen::ArrayXXd flow_data_;
        std::vector<std::pair<double, Eigen::Vector3d>> est_angvels_;
    };
}