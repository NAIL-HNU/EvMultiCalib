#include "lidar_rotation_estimator.h"

namespace lidar_rotation {
    lidarRotationEstimate::lidarRotationEstimate(const std::string& params_dir) {
        params(params_dir);
    }
    lidarRotationEstimate::lidarRotationEstimate() = default;
    lidarRotationEstimate::~lidarRotationEstimate() = default;

    void lidarRotationEstimate::params(const std::string &paramsPath) {
        const std::string params_dir(paramsPath);
        YAML::Node LidarParams = YAML::LoadFile(params_dir);
        opts_.gicp_setting_.num_threads = LidarParams["num_threads"].as<int>();
        opts_.gicp_setting_.downsampling_resolution = LidarParams["downsampling_resolution"].as<double>();
        opts_.gicp_setting_.max_correspondence_distance = LidarParams["max_correspondence_distance"].as<double>();
        opts_.b_motion_compensation_ = LidarParams["b_motion_compensation"].as<bool>();
    }


    void lidarRotationEstimate::AddCloud(const sensor_msgs::PointCloud2 &lidar_msg,
                                               std::vector<Eigen::Vector4f> &point_set,
                                               std::vector<ros::Time> &point_timestamp) {
        size_t point_set_size = lidar_msg.width * lidar_msg.height;
        point_set.reserve(point_set_size);

        point_timestamp.clear();
        point_timestamp.reserve(point_set_size);

        const int x_idx = getPointCloud2FieldIndex(lidar_msg, "x");
        const int y_idx = getPointCloud2FieldIndex(lidar_msg, "y");
        const int z_idx = getPointCloud2FieldIndex(lidar_msg, "z");
        const int t_idx = getPointCloud2FieldIndex(lidar_msg, "timestamp");
        if (x_idx == -1 || y_idx == -1 || z_idx == -1 || t_idx == -1) {
            std::cout << "x/y/z coordinates or timestamp not found." << std::endl;
            return;
        }
        unsigned int x_offset = lidar_msg.fields[x_idx].offset;
        unsigned int y_offset = lidar_msg.fields[y_idx].offset;
        unsigned int z_offset = lidar_msg.fields[z_idx].offset;
        unsigned int t_offset = lidar_msg.fields[t_idx].offset;
        uint8_t x_datatype = lidar_msg.fields[x_idx].datatype;
        uint8_t y_datatype = lidar_msg.fields[y_idx].datatype;
        uint8_t z_datatype = lidar_msg.fields[z_idx].datatype;
        uint8_t t_datatype = lidar_msg.fields[t_idx].datatype;

        for (size_t cp = 0; cp < point_set_size; ++cp) {
            Eigen::Vector4f p;
            p[0] = sensor_msgs::readPointCloud2BufferValue<float>(
                &lidar_msg.data[cp * lidar_msg.point_step + x_offset], x_datatype);
            p[1] = sensor_msgs::readPointCloud2BufferValue<float>(
                &lidar_msg.data[cp * lidar_msg.point_step + y_offset], y_datatype);
            p[2] = sensor_msgs::readPointCloud2BufferValue<float>(
                &lidar_msg.data[cp * lidar_msg.point_step + z_offset], z_datatype);
            p[3] = 1.0;
            point_set.push_back(p);

            double t = sensor_msgs::readPointCloud2BufferValue<double>(
                &lidar_msg.data[cp * lidar_msg.point_step + t_offset], t_datatype);
            t *= 1e-9;
            // LOG(INFO) << std::fixed << std::setprecision(20) << t;
            point_timestamp.push_back(ros::Time(t));
        }
        // LOG(INFO) << std::fixed << std::setprecision(20) << "point_timestamp.front() = " << point_timestamp.front().toSec();
        // LOG(INFO) << std::fixed << std::setprecision(20) << "point_timestamp.back() = " << point_timestamp.back().toSec();
        // LOG(INFO) << std::fixed << std::setprecision(20) << "radar_msg.header.stamp = " << radar_msg.header.stamp.toSec();
        // LOG(INFO) << "Loaded " << point_set.size() << " points into this scan.";
    }

    void lidarRotationEstimate::loadData(const std::string &rosbag_dir, const std::string &ros_topic) {
        rosbag::Bag bag;
        try
        {
            bag.open(rosbag_dir, rosbag::bagmode::Read);
        }
        catch (rosbag::BagException &e)
        {
            ROS_ERROR("failed: %s", e.what());
        }

        rosbag::View lidar_view(bag, rosbag::TopicQuery(ros_topic));

        bool b_first_frame = true;

        for (const auto &m: lidar_view) {
            if (m.getTopic() == ros_topic) {
                sensor_msgs::PointCloud2::ConstPtr lidar_msg = m.instantiate<sensor_msgs::PointCloud2>();
                if (lidar_msg != nullptr) {
                    if (b_first_frame) {
                        std::vector<Eigen::Vector4f> target_points;
                        AddCloud(*lidar_msg, target_points, point_timestamp_);
                        target_ = small_gicp::preprocess_points(*std::make_shared<small_gicp::PointCloud>(target_points), opts_.gicp_setting_.downsampling_resolution, 10, opts_.gicp_setting_.num_threads);
                        b_first_frame = false;
                    } else {
                        source_ = target_;
                        std::vector<Eigen::Vector4f> target_points;
                        AddCloud(*lidar_msg, target_points, point_timestamp_);
                        // std::cout << "raw point num = " << target_points.size() << std::endl;

                        // Motion compensation
                        if (opts_.b_motion_compensation_ && !timestamp2_.empty()) {
                            for (int i = 0; i < target_points.size(); ++i) {
                                float dt = static_cast<float>((point_timestamp_[i] - point_timestamp_.back()).toSec());
                                Eigen::AngleAxisf omega(dt, lidar_angvel_.back().second.cast<float>());

                                Eigen::Matrix4f R = Eigen::Matrix4f::Identity();
                                R.block(0, 0, 3, 3) = omega.matrix();
                                target_points[i] = R * (target_points[i]);
                            }
                        }

                        target_ = small_gicp::preprocess_points(*std::make_shared<small_gicp::PointCloud>(target_points), opts_.gicp_setting_.downsampling_resolution, 10, opts_.gicp_setting_.num_threads);

                        Eigen::Isometry3d init_T_target_source = Eigen::Isometry3d::Identity();
                        small_gicp::RegistrationResult result = small_gicp::align(*target_.first, *source_.first, *target_.second, init_T_target_source, opts_.gicp_setting_);

                        RotationalDeltaWithTime lidar_rot_i;
                        lidar_rot_i.t_prev = lidar_msg->header.stamp.toSec();
                        lidar_rot_i.t_cur = point_timestamp_.back().toSec();
                        lidar_rot_i.rotation_delta = Sophus::SO3d(result.T_target_source.rotation());
                        lidar_rot_.push_back(lidar_rot_i);

                        double dt = lidar_rot_i.t_cur - lidar_rot_i.t_prev;
                        double t = (lidar_rot_i.t_prev + lidar_rot_i.t_cur) / 2;
                        std::pair<double, Eigen::Vector3d> lidar_angvel_i(t, Eigen::Vector3d(-lidar_rot_i.rotation_delta.log() / dt));
                        lidar_angvel_.push_back(lidar_angvel_i);

                    }
                }
            }
        }
        bag.close();
        LOG(INFO) << "LiDAR ego-motion estimation completed. Recovered rotation count = " << lidar_rot_.size();
    }

} // namespace lidar_velocity
