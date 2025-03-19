#include "event_angular_velocity_estimator.h"

namespace ev_angular_velocity_estimator
{
    EvAngularVelocityEstimator::EvAngularVelocityEstimator(const std::string& params_dir) : pCam_(new PerspectiveCamera()) {
        params(params_dir);
    }

    EvAngularVelocityEstimator::~EvAngularVelocityEstimator() = default;

    void EvAngularVelocityEstimator::params(const std::string &paramsPath) {
        const std::string& cam_params_dir(paramsPath);
        YAML::Node CamParams = YAML::LoadFile(cam_params_dir);

        // load camera intrinsic calibration
        size_t width = CamParams["image_width"].as<int>();
        size_t height = CamParams["image_height"].as<int>();
        auto cameraName = CamParams["camera_name"].as<std::string>();
        auto distortion_model = CamParams["distortion_model"].as<std::string>();
        std::vector<double> vD, vK;
        vD = CamParams["distortion_coefficients"]["data"].as<std::vector<double> >();
        vK = CamParams["camera_matrix"]["data"].as<std::vector<double> >();
        pCam_->setIntrinsicParameters(width, height, cameraName, distortion_model, vD, vK);

        opts_.PATCH_SIZE = CamParams["patch_size"].as<int>();
        opts_.HALF_PATCH_SIZE = opts_.PATCH_SIZE / 2;
        opts_.PATCH_MEAN_TOLERANCE = CamParams["patch_mean_tolerance"].as<double>();
        opts_.NEIGHBOR_CNT_THRESHOLD = CamParams["neighbor_cnt_threshold"].as<double>();
        opts_.TERMINATE_INLIER_RATIO = CamParams["terminate_inlier_ratio"].as<double>();
        opts_.PLANE_TOLERANCE = CamParams["plane_tolerance"].as<double>();
        opts_.CONSTANT_EVENT_NUMBER = CamParams["constant_event_number"].as<int>();
        opts_.REFRACTORY_PERIOD = CamParams["refractory_period"].as<double>();
    }


    bool EvAngularVelocityEstimator::calculateAngularVelocity()
    {

        Eigen::ArrayXXd xp = flow_data_.row(0).transpose() - pCam_->K_.at<double>(0, 2);
        Eigen::ArrayXXd yp = flow_data_.row(1).transpose() - pCam_->K_.at<double>(1, 2);

        Eigen::ArrayXXd nf_x = flow_data_.row(2).transpose();
        Eigen::ArrayXXd nf_y = flow_data_.row(3).transpose();

        Eigen::ArrayXXd cov = flow_data_.row(4).transpose();

        // LOG(INFO) << flow_data_.transpose();

        std::vector<double> cov_vec(cov.data(), cov.data() + cov.size());
        std::sort(cov_vec.begin(), cov_vec.end());

        // Compute the 80th percentile threshold:
        // Here, we choose the element at index floor(0.8 * N) as the threshold.
        int idx80 = static_cast<int>(0.8 * cov_vec.size());
        // Ensure index is within bounds:
        if (idx80 >= cov_vec.size())
            idx80 = cov_vec.size() - 1;
        double threshold = cov_vec[idx80];

        // Now, select indices where cov is below the threshold.
        std::vector<int> selected_indices;
        for (int i = 0; i < cov.size(); i++)
        {
            if (cov(i) < threshold)
            {
                selected_indices.push_back(i);
            }
        }

        int nSelected = selected_indices.size();
        Eigen::ArrayXd xp_filtered(nSelected);
        Eigen::ArrayXd yp_filtered(nSelected);
        Eigen::ArrayXd nf_x_filtered(nSelected);
        Eigen::ArrayXd nf_y_filtered(nSelected);

        for (int i = 0; i < nSelected; i++)
        {
            int idx = selected_indices[i];
            xp_filtered(i) = xp(idx);
            yp_filtered(i) = yp(idx);
            nf_x_filtered(i) = nf_x(idx);
            nf_y_filtered(i) = nf_y(idx);
        }

        Eigen::ArrayXd nf_norm = (nf_x_filtered.pow(2) + nf_y_filtered.pow(2)).sqrt();
        Eigen::ArrayXd nf_dir_x = nf_x_filtered / nf_norm;
        Eigen::ArrayXd nf_dir_y = nf_y_filtered / nf_norm;


        Eigen::MatrixXd flow_matrix_omega(nSelected, 3);

        flow_matrix_omega.col(0) = nf_dir_x * xp_filtered * yp_filtered / pCam_->K_.at<double>(0, 0) + nf_dir_y * (pCam_->K_.at<double>(1, 1) + yp_filtered * yp_filtered / pCam_->K_.at<double>(1, 1));

        flow_matrix_omega.col(1) = -nf_dir_x * (pCam_->K_.at<double>(0, 0) + xp_filtered * xp_filtered / pCam_->K_.at<double>(0, 0)) - nf_dir_y * xp_filtered * yp_filtered / pCam_->K_.at<double>(1, 1);

        flow_matrix_omega.col(2) = nf_dir_x * yp_filtered - nf_dir_y * xp_filtered;

        // LOG(INFO) << flow_matrix_omega;

        Eigen::MatrixXd matrix_b = nf_norm.array();

        Eigen::Vector3d est_angvel;

        bool ransac_res = solveAngularVelocityRansac(flow_matrix_omega, matrix_b, est_angvel);

        double ts = t_begin_.toSec() + frame_duration_ / 2;
        Eigen::Vector3d angvel;
        angvel << est_angvel(0), est_angvel(1), est_angvel(2);
        std::pair<double, Eigen::Vector3d> angvel_ts(ts, angvel);

        // LOG(INFO) << "Angular velocity at " << std::setprecision(15) << angvel_ts;

        if (ransac_res)
        {
            est_angvels_.emplace_back(angvel_ts);
        }

        return ransac_res;
    }

    bool EvAngularVelocityEstimator::solveAngularVelocityRansac(const Eigen::MatrixXd &A, const Eigen::VectorXd &b, Eigen::Vector3d &est_angvel)
    {
        // RANSAC to remove outliers
        int flow_num = A.rows();
        int max_iter = 100;
        int max_inlier_num = 0;
        std::vector<int> best_inlier_idx;
        best_inlier_idx.reserve(flow_num);
        Eigen::Vector3d best_velocity;
        srand(0);
        int sample_num = 3;
        Eigen::MatrixXd A_sample(sample_num, 3);
        Eigen::VectorXd b_sample(sample_num);

        double inlier_threshold = 20;
        for (int i = 0; i < max_iter; i++)
        {
            // each time we sample 5 points to estimate the velocity
            std::vector<int> sample_idx;
            // LOG(INFO) << "flow_num : " << flow_num;
            for (int j = 0; j < sample_num; j++)
            {
                int idx = rand() % flow_num;
                sample_idx.push_back(idx);
            }

            for (int j = 0; j < sample_num; j++)
            {
                A_sample.row(j) = A.row(sample_idx[j]);
                b_sample(j) = b(sample_idx[j]);
            }

            Eigen::Vector3d velocity = A_sample.colPivHouseholderQr().solve(b_sample);
            Eigen::VectorXd error = (A * velocity - b).array().abs();

            int inlier_num = 0;
            std::vector<int> inlier_idx;
            for (int j = 0; j < flow_num; j++)
            {
                if (error(j) < inlier_threshold)
                {
                    // LOG(INFO) << "inlier_num: " << inlier_num;
                    inlier_idx.emplace_back(j);
                    inlier_num++;
                }
            }
            if (inlier_num > max_inlier_num)
            {
                max_inlier_num = inlier_num;
                best_inlier_idx = inlier_idx;
                best_velocity = velocity;
            }
        }

        // re-estimate the velocity using all inliers
        Eigen::MatrixXd A_inlier(max_inlier_num, 3);
        Eigen::VectorXd b_inlier(max_inlier_num);
        for (int i = 0; i < max_inlier_num; i++)
        {
            A_inlier.row(i) = A.row(best_inlier_idx[i]);
            b_inlier(i) = b(best_inlier_idx[i]);
        }

        est_angvel = A_inlier.colPivHouseholderQr().solve(b_inlier);

        return true;
    }


    void EvAngularVelocityEstimator::loadData(const std::string& rosbag_dir, const std::string& ev_topic)
    {
        rosbag::Bag bag;
        try
        {
            bag.open(rosbag_dir, rosbag::bagmode::Read);
        }
        catch (rosbag::BagException &e)
        {
            ROS_ERROR("failed: %s", e.what());
        }

        // std::vector<std::string> topics;
        // topics.push_back(ev_topic);
        rosbag::View ev_view(bag, rosbag::TopicQuery(ev_topic));

        for (rosbag::MessageInstance const m : ev_view)
        {
            dvs_msgs::EventArray::ConstPtr msg = m.instantiate<dvs_msgs::EventArray>();
            if (msg != nullptr)
            {
                for (const dvs_msgs::Event &e : msg->events)
                    events_.push_back(e);
            }

            while (events_.size() > opts_.CONSTANT_EVENT_NUMBER)
            {
                processData();
            }
        }

        bag.close();
        LOG(INFO) << "Event camera ego-motion estimation completed. Recovered angular velocity count = " << est_angvels_.size();
    }

    void EvAngularVelocityEstimator::processData()
    {
        EventQueue::iterator it_end = events_.begin();
        auto it_first = events_.begin();
        t_begin_ = it_first->ts;

        std::advance(it_end, opts_.CONSTANT_EVENT_NUMBER);
        t_end_ = it_end->ts;

        double t_end = t_end_.toSec();
        double t_begin = t_begin_.toSec();
        double t_duration = t_end - t_begin;
        frame_duration_ = t_duration;

        // LOG(ERROR) << "t_begin: " << std::setprecision(15) << t_begin;
        // LOG(ERROR) << "t_end: " << std::setprecision(15) << t_end;

        last_timestamp_map_ = cv::Mat::zeros(pCam_->height_, pCam_->width_, CV_32F);
        original_last_timestamp_map_ = cv::Mat::zeros(pCam_->height_, pCam_->width_, CV_64F);

        cv::Mat prev_p = cv::Mat::zeros(pCam_->height_, pCam_->width_, CV_8U) + 2;

        // LOG(INFO) << "event count: " << std::distance(events_.begin(), it_end);

        for (auto it = events_.begin(); it != it_end; it++)
        {
            double normalized_ts = (it->ts.toSec() - t_begin) / t_duration;
            // LOG(INFO) << it->y << " " << it->x << " " << int(it->polarity) << " " << normalized_ts << " " << (it->ts.toSec() - original_last_timestamp_map_.at<double>(it->y, it->x) > REFRACTORY_PERIOD)  << " " << (prev_p.at<uchar>(it->y, it->x) != it->polarity);
            if ((it->ts.toSec() - original_last_timestamp_map_.at<double>(it->y, it->x) > opts_.REFRACTORY_PERIOD) || prev_p.at<uchar>(it->y, it->x) != it->polarity)
            {
                original_last_timestamp_map_.at<double>(it->y, it->x) = it->ts.toSec();
                prev_p.at<uchar>(it->y, it->x) = it->polarity;

                last_timestamp_map_.at<float>(it->y, it->x) = normalized_ts;

            }
        }

        cv::remap(last_timestamp_map_, undistorted_last_timestamp_map_, pCam_->precomputed_undistorted_x_, pCam_->precomputed_undistorted_y_, cv::INTER_LINEAR);

        // cv::Mat ev_visual = 255.0 * undistorted_last_timestamp_map_;
        // ev_visual.convertTo(ev_visual, CV_8U);
        // cv::imshow("ev_visual", ev_visual);
        // cv::waitKey(1);

        events_.erase(events_.begin(), it_end);

        bool nf_res = calculateNormalFlow();
        if (nf_res)
        {
            // LOG(INFO) << "Normal flow calculation is successful.";
            bool angular_velocity_res = calculateAngularVelocity();

        }
        // else
        // {
        //     LOG(INFO) << "Normal flow calculation is failed.";
        // }


    }
}