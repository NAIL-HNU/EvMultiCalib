#include "calibration.h"

namespace ev_calib {
    OptionsMethod params(const std::string &paramsPath) {
        OptionsMethod opts;
        const std::string params_dir(paramsPath);
        YAML::Node CalibParams = YAML::LoadFile(params_dir);
        opts.imu_topic_ = CalibParams["imu_topic"].as<std::string>();
        opts.ev_topic_ = CalibParams["event_topic"].as<std::string>();
        opts.rgb_topic_ = CalibParams["rgb_topic"].as<std::string>();
        opts.lidar_topic_ = CalibParams["lidar_topic"].as<std::string>();

        opts.calib_imu_ = CalibParams["calib_imu"].as<bool>();
        opts.calib_rgb_ = CalibParams["calib_rgb"].as<bool>();
        opts.calib_lidar_ = CalibParams["calib_lidar"].as<bool>();

        opts.td_range_ = CalibParams["td_range"].as<double>();
        opts.td_resolution_ = CalibParams["td_resolution"].as<int>();

        opts.event_weight_ = CalibParams["event_weight"].as<double>();
        opts.imu_weight_ = CalibParams["imu_weight"].as<double>();
        opts.rgb_weight_ = CalibParams["rgb_weight"].as<double>();
        opts.lidar_weight_ = CalibParams["lidar_weight"].as<double>();
        opts.time_offset_bound_ = CalibParams["time_offset_bound"].as<int>();
        opts.dt_knots_ = CalibParams["dt_knots"].as<double>();

        return opts;
    }


    EventCalibration::EventCalibration(ros::NodeHandle &nh, std::string &rosbag_dir, std::string &params_dir,
                                       std::string &result_dir)
        : nh_(nh),
          imu_max_corr_(0.0), rgb_max_corr_(0.0), lidar_max_corr_(0.0),
          imu_time_offset_(0.0), rgb_time_offset_(0.0), lidar_time_offset_(0.0),
          imu_extrin_rot_(Eigen::Quaterniond::Identity()),
          rgb_extrin_rot_(Eigen::Quaterniond::Identity()),
          lidar_extrin_rot_(Eigen::Quaterniond::Identity()),
          gyro_bias_{0, 0, 0} {
        // Initialize
        opts_ = params(params_dir);
        if (!opts_.calib_imu_ && !opts_.calib_rgb_ && !opts_.calib_lidar_) {
            LOG(INFO) << "Select sensors to calibrate.";
            exit(-1);
        }
        ofile_.open(result_dir + "/extrinsicCalib.txt");
        if (!ofile_.is_open()) {
            LOG(INFO) << "Fail to open " + result_dir + "/extrinsicCalib.txt";
            exit(-1);
        }
    }

    EventCalibration::~EventCalibration() = default;

    void EventCalibration::loadVelocityFromTxt(const std::string &motion_dir,
                                               std::vector<Eigen::Matrix<double, 1, 4> > &velocity) {
        std::ifstream velocity_file(motion_dir);
        if (!velocity_file.is_open()) {
            std::cout << "Velocity file is not opened!" << std::endl;
            exit(-1);
        }
        std::string velocity_message_line;
        while (std::getline(velocity_file, velocity_message_line)) {
            std::stringstream ss(velocity_message_line);
            double vel_x, vel_y, vel_z, timestamp;
            ss >> std::fixed >> vel_x >> vel_y >> vel_z >> timestamp;
            velocity.emplace_back(vel_x, vel_y, vel_z, timestamp);
        }
    }


    void EventCalibration::calib_init(std::vector<std::pair<double, Eigen::Vector3d> > &ev_vel,
                                      std::vector<std::pair<double, Eigen::Vector3d> > &obj_vel,
                                      sensor_type_set sensor_type) {
        std::vector<double> x, y, z, t;
        for (auto &i: obj_vel) {
            x.emplace_back(i.second(0));
            y.emplace_back(i.second(1));
            z.emplace_back(i.second(2));
            t.emplace_back(i.first);
        }

        tk::spline spline_x(t, x, tk::spline::spline_type::cspline_hermite);
        tk::spline spline_y(t, y, tk::spline::spline_type::cspline_hermite);
        tk::spline spline_z(t, z, tk::spline::spline_type::cspline_hermite);

        double max_corr(-1);
        double delta_t = 1.0 / opts_.td_resolution_;
        double td_sec = -opts_.td_range_;
        int step = opts_.td_range_ * opts_.td_resolution_ * 2;
        for (int i = 0; i < step; ++i) {
            std::vector<std::pair<double, Eigen::Vector3d>> vel_shift;
            for (int i = 0; i < ev_vel.size(); i++) {
                Eigen::Vector3d vel_i(spline_x(ev_vel[i].first - td_sec),
                                      spline_y(ev_vel[i].first - td_sec),
                                      spline_z(ev_vel[i].first - td_sec));
                std::pair<double, Eigen::Vector3d> vel_shift_i(ev_vel[i].first, vel_i);
                vel_shift.push_back(vel_shift_i);
            }

            Eigen::Matrix3d R_oe;
            double corr = CCA(ev_vel, vel_shift, R_oe);
            if (corr > max_corr) {
                max_corr = corr;
                if (sensor_type == imu) {
                    imu_max_corr_ = corr;
                    imu_time_offset_ = td_sec;
                    imu_extrin_rot_ = Eigen::Quaterniond(R_oe);
                } else if (sensor_type == rgb) {
                    rgb_max_corr_ = corr;
                    rgb_time_offset_ = td_sec;
                    rgb_extrin_rot_ = Eigen::Quaterniond(R_oe);
                } else if (sensor_type == lidar) {
                    lidar_max_corr_ = corr;
                    lidar_time_offset_ = td_sec;
                    lidar_extrin_rot_ = Eigen::Quaterniond(R_oe);
                }
            }
            td_sec += delta_t;
        }
    }


    double EventCalibration::CCA(const std::vector<std::pair<double, Eigen::Vector3d> > &ev_vel,
                                 const std::vector<std::pair<double, Eigen::Vector3d> > &obj_vel,
                                 Eigen::Matrix3d &R_oe) {
        double corr;
        const size_t num_vel = ev_vel.size();
        Eigen::MatrixXd vel(num_vel, 6);
        for (int i = 0; i < num_vel; i++) {
            vel.row(i) << ev_vel[i].second.transpose(), obj_vel[i].second.transpose();
        }
        Eigen::Matrix<double, 1, 6> vel_mean;
        vel_mean = vel.colwise().mean();
        vel.rowwise() -= vel_mean;

        Eigen::MatrixXd cov = (vel.transpose() * vel) / double(num_vel - 1);
        Eigen::Matrix3d S_xx, S_yy, S_xy, M;
        S_xx = cov.block(0, 0, 3, 3);
        S_yy = cov.block(3, 3, 3, 3);
        S_xy = cov.block(0, 3, 3, 3);
        M = S_xx.inverse() * S_xy * S_yy.inverse() * S_xy.transpose();
        corr = std::sqrt(M.trace() / 3);

        // rotation
        Eigen::Matrix3d H;
        H = S_yy.inverse() * S_xy.transpose();
        Eigen::JacobiSVD<Eigen::MatrixXd> H_svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Matrix3d H_U, H_V;
        H_U = H_svd.matrixU();
        H_V = H_svd.matrixV();
        Eigen::Matrix3d R_normalize;
        R_normalize << 1, 0, 0,
                       0, 1, 0,
                       0, 0, (H_V * H_U.transpose()).determinant();
        R_oe = H_V * R_normalize * H_U.transpose();

        return corr;
    }


    void EventCalibration::calib_optim() {
        double start_time = ev_angvel_.front().first;
        double end_time = ev_angvel_.back().first;
        if (opts_.calib_imu_) {
            if (imu_angvel_.front().first < start_time) {
                start_time = imu_angvel_.front().first;
            }
            if (imu_angvel_.back().first > end_time) {
                end_time = imu_angvel_.back().first;
            }
        }
        if (opts_.calib_rgb_) {
            if (rgb_rot_.front().t_prev < start_time) {
                start_time = rgb_rot_.front().t_prev;
            }
            if (rgb_rot_.back().t_prev > end_time) {
                end_time = rgb_rot_.back().t_prev;
            }
        }
        if (opts_.calib_lidar_) {
            if (lidar_rot_.front().t_prev < start_time) {
                start_time = lidar_rot_.front().t_prev;
            }
            if (lidar_rot_.back().t_prev > end_time) {
                end_time = lidar_rot_.back().t_prev;
            }
        }
        double duration = end_time - start_time;

        const int desired_cp_num = static_cast<int>(std::ceil(duration / opts_.dt_knots_)) + SPLINE_DEGREE;

        double control_points[desired_cp_num][4];
        for (int i = 0; i < desired_cp_num; i++) {
            control_points[i][0] = 0;
            control_points[i][1] = 0;
            control_points[i][2] = 0;
            control_points[i][3] = 1;
        }

        double time_offsets[3][1];

        time_offsets[0][0] = imu_time_offset_;
        time_offsets[1][0] = rgb_time_offset_;
        time_offsets[2][0] = lidar_time_offset_;

        double extrinsic_rotation[3][4];

        extrinsic_rotation[0][0] = imu_extrin_rot_.x();
        extrinsic_rotation[0][1] = imu_extrin_rot_.y();
        extrinsic_rotation[0][2] = imu_extrin_rot_.z();
        extrinsic_rotation[0][3] = imu_extrin_rot_.w();

        extrinsic_rotation[1][0] = rgb_extrin_rot_.x();
        extrinsic_rotation[1][1] = rgb_extrin_rot_.y();
        extrinsic_rotation[1][2] = rgb_extrin_rot_.z();
        extrinsic_rotation[1][3] = rgb_extrin_rot_.w();

        extrinsic_rotation[2][0] = lidar_extrin_rot_.x();
        extrinsic_rotation[2][1] = lidar_extrin_rot_.y();
        extrinsic_rotation[2][2] = lidar_extrin_rot_.z();
        extrinsic_rotation[2][3] = lidar_extrin_rot_.w();

        double gyro_bias[1][3];

        gyro_bias[0][0] = 0;
        gyro_bias[0][1] = 0;
        gyro_bias[0][2] = 0;

        ceres::Problem problem;
        ceres::LossFunction *loss_function;
        loss_function = new ceres::HuberLoss(0.2);
        // loss_function = new ceres::CauchyLoss(0.1);

        ceres::LocalParameterization *quat_param = new ceres::EigenQuaternionParameterization();

        problem.AddParameterBlock(time_offsets[0], 1);
        problem.SetParameterLowerBound(time_offsets[0], 0, -opts_.time_offset_bound_ * opts_.dt_knots_);
        problem.SetParameterUpperBound(time_offsets[0], 0, opts_.time_offset_bound_ * opts_.dt_knots_);

        problem.AddParameterBlock(time_offsets[1], 1);
        problem.SetParameterLowerBound(time_offsets[1], 0, -opts_.time_offset_bound_ * opts_.dt_knots_);
        problem.SetParameterUpperBound(time_offsets[1], 0, opts_.time_offset_bound_ * opts_.dt_knots_);

        problem.AddParameterBlock(time_offsets[2], 1);
        problem.SetParameterLowerBound(time_offsets[2], 0, -opts_.time_offset_bound_ * opts_.dt_knots_);
        problem.SetParameterUpperBound(time_offsets[2], 0, opts_.time_offset_bound_ * opts_.dt_knots_);

        problem.AddParameterBlock(extrinsic_rotation[0], 4, quat_param);
        problem.AddParameterBlock(extrinsic_rotation[1], 4, quat_param);
        problem.AddParameterBlock(extrinsic_rotation[2], 4, quat_param);

        problem.AddParameterBlock(gyro_bias[0], 3);

        for (int i = 0; i < desired_cp_num; i++) {
            problem.AddParameterBlock(control_points[i], 4, quat_param);
        }

        int cur_segment_index = 0;
        // event vel constrains
        for (auto & ev_angvel_i: ev_angvel_) {
            // break;
            double cur_time = ev_angvel_i.first;

            if (cur_time > end_time) {
                break;
            }

            while (cur_time > start_time + opts_.dt_knots_ * (cur_segment_index + 1)) {
                cur_segment_index++;
            }

            double time_offset = cur_time - (start_time + opts_.dt_knots_ * cur_segment_index);
            double normalized_time = time_offset / opts_.dt_knots_;

            auto *event_cost_function = new angVelCostFunctorWoBiasTimeOffset(normalized_time, ev_angvel_i.second, opts_.event_weight_, opts_.dt_knots_);
            auto *event_factor = new ceres::DynamicAutoDiffCostFunction<angVelCostFunctorWoBiasTimeOffset>(event_cost_function);

            for (int j = 0; j < SPLINE_DEGREE + 1; j++) {
                event_factor->AddParameterBlock(4);
            }

            event_factor->SetNumResiduals(3);

            std::vector<double *> paramBlockVec;
            for (int j = 0; j < SPLINE_DEGREE + 1; j++) {
                paramBlockVec.push_back(control_points[cur_segment_index + j]);
            }

            // self_data : huber loss
            // ECD_data : null
            // problem.AddResidualBlock(event_factor, nullptr, paramBlockVec);
            problem.AddResidualBlock(event_factor, loss_function, paramBlockVec);
        }

        // imu vel constains
        if(opts_.calib_imu_) {
            cur_segment_index = 0;
            for (auto & imu_angvel_i : imu_angvel_) {
                // break;

                double cur_time = imu_angvel_i.first;

                while (cur_time > start_time + opts_.dt_knots_ * (cur_segment_index + 1)) {
                    cur_segment_index++;
                }

                if (cur_segment_index < opts_.time_offset_bound_) {
                    continue;
                }

                if (cur_segment_index >= desired_cp_num - opts_.time_offset_bound_ - SPLINE_DEGREE) {
                    break;
                }
                // LOG(INFO) << cur_time - start_time << " " << imu_bspline_vels[i].transpose();

                double time_offset = cur_time - (start_time + opts_.dt_knots_ * cur_segment_index);

                double normalized_time = time_offset / opts_.dt_knots_;
                // LOG(INFO) << "normalized_time: " << normalized_time;

                auto *imu_cost_function = new angVelCostFunctor(normalized_time, imu_angvel_i.second, opts_.imu_weight_, opts_.dt_knots_, opts_.time_offset_bound_);
                auto *imu_factor = new ceres::DynamicAutoDiffCostFunction<angVelCostFunctor>(imu_cost_function);

                imu_factor->AddParameterBlock(1);
                imu_factor->AddParameterBlock(4);
                imu_factor->AddParameterBlock(3);
                for (int j = 0; j < SPLINE_DEGREE + 1 + 2 * opts_.time_offset_bound_; j++) {
                    imu_factor->AddParameterBlock(4);
                }

                imu_factor->SetNumResiduals(3);

                std::vector<double *> paramBlockVec;
                paramBlockVec.push_back(time_offsets[0]);
                paramBlockVec.push_back(extrinsic_rotation[0]);
                paramBlockVec.push_back(gyro_bias[0]);
                // LOG(INFO) << "extrinsic_rotation: " << paramBlockVec[1][0] << ", " << paramBlockVec[1][1] << ", " << paramBlockVec[1][2] << ", " << paramBlockVec[1][3];

                for (int j = 0; j < SPLINE_DEGREE + 1 + 2 * opts_.time_offset_bound_; j++) {
                    paramBlockVec.push_back(control_points[cur_segment_index + j - 2]);

                    // LOG(INFO) << "control_points: " << i << " " << control_points[cur_segment_index + j - 2][0] << ", " << control_points[cur_segment_index + j - 2][1] << ", " << control_points[cur_segment_index + j - 2][2] << ", " << control_points[cur_segment_index + j - 2][3];
                }

                problem.AddResidualBlock(imu_factor, nullptr, paramBlockVec);

                // break;
            }
        }

        // rgb rotation delta constains
        if(opts_.calib_rgb_) {
            int prev_segment_index = 0;
            cur_segment_index = 0;
            for (auto & rgb_rot_i : rgb_rot_) {
                // break;
                double cur_time = rgb_rot_i.t_cur;
                double prev_time = rgb_rot_i.t_prev;

                while (cur_time > start_time + opts_.dt_knots_ * (cur_segment_index + 1)) {
                    cur_segment_index++;
                }

                while (prev_time > start_time + opts_.dt_knots_ * (prev_segment_index + 1)) {
                    prev_segment_index++;
                }

                if (cur_segment_index < opts_.time_offset_bound_ || prev_segment_index < opts_.time_offset_bound_) {
                    continue;
                }

                if (cur_segment_index >= desired_cp_num - opts_.time_offset_bound_ - SPLINE_DEGREE) {
                    break;
                }

                double time_offset_cur = cur_time - (start_time + opts_.dt_knots_ * cur_segment_index);
                double normalized_time_cur = time_offset_cur / opts_.dt_knots_;

                double time_offset_prev = prev_time - (start_time + opts_.dt_knots_ * prev_segment_index);
                double normalized_time_prev = time_offset_prev / opts_.dt_knots_;

                int segment_index_involved = cur_segment_index - prev_segment_index;

                auto *rotation_cost_function = new rotationCostFunctor(normalized_time_prev, normalized_time_cur, rgb_rot_i.rotation_delta, opts_.rgb_weight_, opts_.dt_knots_, segment_index_involved, opts_.time_offset_bound_);
                auto *rotation_factor = new ceres::DynamicAutoDiffCostFunction<rotationCostFunctor>(rotation_cost_function);

                rotation_factor->AddParameterBlock(1);
                rotation_factor->AddParameterBlock(4);

                for (int j = 0; j < SPLINE_DEGREE + 1 + 2 * opts_.time_offset_bound_ + segment_index_involved; j++) {
                    rotation_factor->AddParameterBlock(4);
                }

                rotation_factor->SetNumResiduals(3);

                std::vector<double *> paramBlockVec;
                paramBlockVec.push_back(time_offsets[1]);
                paramBlockVec.push_back(extrinsic_rotation[1]);

                for (int j = 0; j < SPLINE_DEGREE + 1 + 2 * opts_.time_offset_bound_ + segment_index_involved; j++) {
                    paramBlockVec.push_back(control_points[prev_segment_index + j - opts_.time_offset_bound_]);

                    // LOG(INFO) << "control_points: " << i << " " << control_points[cur_segment_index + j - 2][0] << ", " << control_points[cur_segment_index + j - 2][1] << ", " << control_points[cur_segment_index + j - 2][2] << ", " << control_points[cur_segment_index + j - 2][3];
                }

                problem.AddResidualBlock(rotation_factor, nullptr, paramBlockVec);
            }
        }

        // lidar constains
        if (opts_.calib_lidar_) {
            int prev_segment_index = 0;
            cur_segment_index = 0;
            for (auto & lidar_rot_i : lidar_rot_) {
                // break;
                double cur_time = lidar_rot_i.t_cur;
                double prev_time = lidar_rot_i.t_prev;

                while (cur_time > start_time + opts_.dt_knots_ * (cur_segment_index + 1)) {
                    cur_segment_index++;
                }

                while (prev_time > start_time + opts_.dt_knots_ * (prev_segment_index + 1)) {
                    prev_segment_index++;
                }

                if (cur_segment_index < opts_.time_offset_bound_ || prev_segment_index < opts_.time_offset_bound_) {
                    continue;
                }

                if (cur_segment_index >= desired_cp_num - opts_.time_offset_bound_ - SPLINE_DEGREE) {
                    break;
                }

                double time_offset_cur = cur_time - (start_time + opts_.dt_knots_ * cur_segment_index);
                double normalized_time_cur = time_offset_cur / opts_.dt_knots_;

                double time_offset_prev = prev_time - (start_time + opts_.dt_knots_ * prev_segment_index);
                double normalized_time_prev = time_offset_prev / opts_.dt_knots_;

                int segment_index_involved = cur_segment_index - prev_segment_index;

                auto *rotation_cost_function = new rotationCostFunctor(normalized_time_prev, normalized_time_cur, lidar_rot_i.rotation_delta, opts_.lidar_weight_, opts_.dt_knots_, segment_index_involved, opts_.time_offset_bound_);
                auto *rotation_factor = new ceres::DynamicAutoDiffCostFunction<rotationCostFunctor>(rotation_cost_function);

                rotation_factor->AddParameterBlock(1);
                rotation_factor->AddParameterBlock(4);

                for (int j = 0; j < SPLINE_DEGREE + 1 + 2 * opts_.time_offset_bound_ + segment_index_involved; j++) {
                    rotation_factor->AddParameterBlock(4);
                }

                rotation_factor->SetNumResiduals(3);

                std::vector<double *> paramBlockVec;
                paramBlockVec.push_back(time_offsets[2]);
                paramBlockVec.push_back(extrinsic_rotation[2]);

                for (int j = 0; j < SPLINE_DEGREE + 1 + 2 * opts_.time_offset_bound_ + segment_index_involved; j++) {
                    paramBlockVec.push_back(control_points[prev_segment_index + j - opts_.time_offset_bound_]);

                    // LOG(INFO) << "control_points: " << i << " " << control_points[cur_segment_index + j - 2][0] << ", " << control_points[cur_segment_index + j - 2][1] << ", " << control_points[cur_segment_index + j - 2][2] << ", " << control_points[cur_segment_index + j - 2][3];
                }

                problem.AddResidualBlock(rotation_factor, nullptr, paramBlockVec);

                // LOG(INFO) << "adding a lidar measurement";
            }
        }

        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = true;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        // options.max_num_iterations = 20;
        options.num_threads = 8;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        // LOG(INFO) << summary.FullReport();

        if(opts_.calib_imu_) {
            imu_time_offset_ = time_offsets[0][0];
            imu_extrin_rot_.x() = extrinsic_rotation[0][0];
            imu_extrin_rot_.y() = extrinsic_rotation[0][1];
            imu_extrin_rot_.z() = extrinsic_rotation[0][2];
            imu_extrin_rot_.w() = extrinsic_rotation[0][3];
            gyro_bias_[0] = gyro_bias[0][0];
            gyro_bias_[1] = gyro_bias[0][1];
            gyro_bias_[2] = gyro_bias[0][2];
        }

        if(opts_.calib_rgb_) {
            rgb_time_offset_ = time_offsets[1][0];
            rgb_extrin_rot_.x() = extrinsic_rotation[1][0];
            rgb_extrin_rot_.y() = extrinsic_rotation[1][1];
            rgb_extrin_rot_.z() = extrinsic_rotation[1][2];
            rgb_extrin_rot_.w() = extrinsic_rotation[1][3];
        }

        if(opts_.calib_lidar_) {
            lidar_time_offset_ = time_offsets[2][0];
            lidar_extrin_rot_.x() = extrinsic_rotation[2][0];
            lidar_extrin_rot_.y() = extrinsic_rotation[2][1];
            lidar_extrin_rot_.z() = extrinsic_rotation[2][2];
            lidar_extrin_rot_.w() = extrinsic_rotation[2][3];
        }
    }
}
