#include "rgb_rotation_estimator.h"

namespace rgb_rotation {
    RgbRotationEstimate::RgbRotationEstimate(const std::string& params_dir) : pCam_(new PerspectiveCamera()), scale_(1.0) {
        params(params_dir);
    }

    RgbRotationEstimate::RgbRotationEstimate() : pCam_(new PerspectiveCamera()), scale_(1.0) {}

    RgbRotationEstimate::~RgbRotationEstimate() = default;

    void RgbRotationEstimate::params(const std::string &paramsPath) {
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

        opts_.matches_filter_min_dist_ = CamParams["matches_filter_min_dist"].as<double>();
        opts_.inlier_distance_thres_ = CamParams["inlier_distance_thres"].as<double>();
        opts_.inlier_ratio_thres_ = CamParams["inlier_ratio_thres"].as<double>();
        opts_.inlier_count_thres_ = CamParams["inlier_count_thres"].as<int>();
        opts_.filter_threshold_ = CamParams["filter_threshold"].as<double>();
        opts_.ransac_max_iter_ = CamParams["ransac_max_iter"].as<int>();

        // Init
        scale_ = std::min(1.0, std::max(0.25, 400.0 / pCam_->height_));
        featureObj_firstImg_.img_.create(pCam_->height_, pCam_->width_, CV_8UC3);
        featureObj_firstImg_.timestamp_ = ros::Time(0);
    }


    bool RgbRotationEstimate::rotationEstimate(const featureObj &featureObj_firstImg,
                                               const featureObj &featureObj_secondImg,
                                               const std::vector<cv::DMatch> &matches) {
        cv::Mat K = pCam_->K_ * scale_;
        K.at<double>(2, 2) = 1.0;

        VecVector3d pts_3d;
        VecVector2d pts_2d;
        for (auto &m: matches) {
            cv::Point2d p1 = pixel2cam(featureObj_firstImg.keypoints_[m.queryIdx].pt, K);
            pts_3d.push_back(Eigen::Vector3d(p1.x, p1.y, 1));
            pts_2d.push_back(Eigen::Vector2d(featureObj_secondImg.keypoints_[m.trainIdx].pt.x,
                                             featureObj_secondImg.keypoints_[m.trainIdx].pt.y));
        }

        // Sophus::SO3d pose;
        // double cost = poseEstimation_PnP(pCam_->K_, pose, pts_3d, pts_2d);
        // R_ = pose.matrix();

        // RANSAC
        int iter = 0;
        Eigen::Matrix3d R;
        double ransac_inlier_ratio(0.0);
        int ransac_inlier_count(0);
        for (; iter < opts_.ransac_max_iter_; ++iter) {
            std::vector<cv::DMatch> matches_fit;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, matches.size() - 1);
            VecVector3d pts_3d_fit;
            VecVector2d pts_2d_fit;
            for (int i = 0; i < SAMPLE_SIZE; i++) {
                int random_num = dis(gen);
                pts_3d_fit.push_back(pts_3d[random_num]);
                pts_2d_fit.push_back(pts_2d[random_num]);
            }

            Sophus::SO3d pose;
            minReprojectionError(K, pose, pts_3d_fit, pts_2d_fit);
            int inlier_count(0);
            for (int i = 0; i < matches.size(); i++) {
                Eigen::Vector3d pc = pose * pts_3d[i];
                Eigen::Vector2d proj(K.at<double>(0, 0) * pc[0] / pc[2] + K.at<double>(0, 2),
                                     K.at<double>(1, 1) * pc[1] / pc[2] + K.at<double>(1, 2));
                double error_i = (pts_2d[i] - proj).norm();
                if (error_i < opts_.inlier_distance_thres_) inlier_count++;
                // lastCost += error_i;
            }
            double inlier_ratio = static_cast<double>(inlier_count) / matches.size();

            if (inlier_ratio > ransac_inlier_ratio) {
                ransac_inlier_ratio = inlier_ratio;
                ransac_inlier_count = inlier_count;
                R = pose.matrix();
            }
            if (ransac_inlier_ratio >= 0.5 && ransac_inlier_count >= opts_.inlier_count_thres_) {
                break;
            }
        }

        // LOG(INFO) << "RANSAC iter = " << iter << ",\transac_inlier_ratio = " << ransac_inlier_ratio;

        if (ransac_inlier_ratio >= opts_.inlier_ratio_thres_ || ransac_inlier_count >= opts_.inlier_count_thres_) {
            double angle = std::acos((R.trace() - 1) / 2);
            if (abs(angle) < opts_.filter_threshold_) {
                RotationalDeltaWithTime rgb_rot_i;
                rgb_rot_i.rotation_delta = Sophus::SO3d(R);
                rgb_rot_i.t_prev = featureObj_firstImg.timestamp_.toSec();
                rgb_rot_i.t_cur = featureObj_secondImg.timestamp_.toSec();
                rgb_rot_.push_back(rgb_rot_i);

                double dt = rgb_rot_i.t_cur - rgb_rot_i.t_prev;
                double t = (rgb_rot_i.t_prev + rgb_rot_i.t_cur) / 2;
                std::pair<double, Eigen::Vector3d> rgb_angvel_i(t, Eigen::Vector3d(-rgb_rot_i.rotation_delta.log() / dt));
                rgb_angvel_.push_back(rgb_angvel_i);
                return true;
            }
        }
        return false;
    }

    double RgbRotationEstimate::minReprojectionError(const cv::Mat &K, Sophus::SO3d &pose,
                                                     const VecVector3d &points_3d,
                                                     const VecVector2d &points_2d) {
        const int iterations = 50;
        double cost(0.0), lastCost(0.0);
        double fx = K.at<double>(0, 0);
        double fy = K.at<double>(1, 1);
        double cx = K.at<double>(0, 2);
        double cy = K.at<double>(1, 2);

        for (int iter = 0; iter < iterations; iter++) {
            Eigen::Matrix<double, 3, 3> H = Eigen::Matrix<double, 3, 3>::Zero();
            Eigen::Matrix<double, 3, 1> b = Eigen::Matrix<double, 3, 1>::Zero();

            cost = 0;
            // compute cost
            for (int i = 0; i < points_3d.size(); i++) {
                Eigen::Vector3d pc = pose * points_3d[i];
                double inv_z = 1.0 / pc[2];
                double inv_z2 = inv_z * inv_z;
                Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);

                Eigen::Vector2d e = points_2d[i] - proj;

                cost += e.squaredNorm();
                Eigen::Matrix<double, 2, 3> J;
                J << fx * pc[0] * pc[1] * inv_z2,
                        -fx - fx * pc[0] * pc[0] * inv_z2,
                        fx * pc[1] * inv_z,
                        fy + fy * pc[1] * pc[1] * inv_z2,
                        -fy * pc[0] * pc[1] * inv_z2,
                        -fy * pc[0] * inv_z;

                H += J.transpose() * J;
                b += -J.transpose() * e;
            }

            Eigen::Matrix<double, 3, 1> dx;
            dx = H.ldlt().solve(b);

            if (isnan(dx[0])) {
                // LOG(INFO) << "result is nan!";
                break;
            }

            if (iter > 0 && cost >= lastCost) {
                // cost increase, update is not good
                break;
            }

            // update estimation
            pose = Sophus::SO3d::exp(dx) * pose;
            lastCost = cost;

            if (dx.norm() < 1e-10) {
                // converge
                break;
            }
        }
        return lastCost;
    }

    void RgbRotationEstimate::loadData(const std::string &rosbag_dir, const std::string &ros_topic) {
        rosbag::Bag bag;
        try
        {
            bag.open(rosbag_dir, rosbag::bagmode::Read);
        }
        catch (rosbag::BagException &e)
        {
            ROS_ERROR("failed: %s", e.what());
        }

        rosbag::View img_view(bag, rosbag::TopicQuery(ros_topic));

        bool b_first_img = true;

        for (const auto &m: img_view) {
            if (m.getTopic() == ros_topic) {
                sensor_msgs::Image::ConstPtr img_msg = m.instantiate<sensor_msgs::Image>();
                if (img_msg != nullptr) {
                    cv_bridge::CvImageConstPtr cv_img_ptr = cv_bridge::toCvCopy(
                        img_msg, sensor_msgs::image_encodings::BGR8);

                    cv::Mat image = cv::Mat(pCam_->height_, pCam_->width_, CV_8UC1);

                    cv::remap(cv_img_ptr->image, image, pCam_->precomputed_undistorted_x_, pCam_->precomputed_undistorted_y_, cv::INTER_LINEAR);

                    // processing image edges
                    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0,
                                                               -1, 5, -1,
                                                               0, -1, 0);
                    filter2D(image, image, -1, kernel);
                    // downsampling
                    resize(image, image, cv::Size(), scale_, scale_, cv::INTER_AREA);

                    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
                    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
                    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
                    if (b_first_img) {
                        featureObj_firstImg_.img_ = image;
                        featureObj_firstImg_.timestamp_ = img_msg->header.stamp;
                        detector->detect(featureObj_firstImg_.img_, featureObj_firstImg_.keypoints_);
                        if (featureObj_firstImg_.keypoints_.empty()) {
                            // return;
                            continue;
                        }
                        descriptor->compute(featureObj_firstImg_.img_, featureObj_firstImg_.keypoints_,
                                            featureObj_firstImg_.descriptors_);
                        b_first_img = false;
                    } else {
                        rgb_rotation::featureObj featureObj_secondImg;
                        featureObj_secondImg.img_ = /*cv_img_ptr->*/image;
                        featureObj_secondImg.timestamp_ = img_msg->header.stamp;
                        detector->detect(featureObj_secondImg.img_, featureObj_secondImg.keypoints_);
                        if (featureObj_secondImg.keypoints_.empty()) {
                            // return;
                            continue;
                        }
                        // LOG(INFO) << "A total of " << featureObj_secondImg.keypoints_.size() << " key points were found";
                        descriptor->compute(featureObj_secondImg.img_, featureObj_secondImg.keypoints_,
                                            featureObj_secondImg.descriptors_);

                        std::vector<cv::DMatch> matches_rough, matches_filt;
                        matcher->match(featureObj_firstImg_.descriptors_, featureObj_secondImg.descriptors_,
                                       matches_rough);
                        double min_dist = 10000, max_dist = 0;
                        for (int i = 0; i < featureObj_firstImg_.descriptors_.rows; i++) {
                            double dist = matches_rough[i].distance;
                            if (dist < min_dist) min_dist = dist;
                            if (dist > max_dist) max_dist = dist;
                        }
                        // LOG(INFO) << "min_dist: " << min_dist;
                        for (int i = 0; i < featureObj_firstImg_.descriptors_.rows; i++) {
                            if (matches_rough[i].distance <= std::max(2.0 * min_dist,
                                                                      opts_.matches_filter_min_dist_)) {
                                matches_filt.push_back(matches_rough[i]);
                            }
                        }

                        if (matches_filt.size() > opts_.inlier_count_thres_) {
                            rotationEstimate(featureObj_firstImg_, featureObj_secondImg, matches_filt);
                        }
                        featureObj_firstImg_ = featureObj_secondImg;
                    }
                }
            }
        }
        bag.close();
        LOG(INFO) << "Frame-based camera ego-motion estimation completed. Recovered rotation count = " << rgb_rot_.size();
    }
} // namespace rgb_rotation
