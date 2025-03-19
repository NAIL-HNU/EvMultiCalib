#include "event_angular_velocity_estimator.h"

namespace ev_angular_velocity_estimator
{
    size_t computeOrientationBin(double vx, double vy)
    {
        int orient = 0;
        if (fabs(vx) < 1e-10f)
        {
            orient = 2;
            if (vy < 0)
                orient = 6;
            return orient;
        }

        float tantheta = vy / vx;

        if (tantheta < -0.4142f)
        {
            if (tantheta < -2.4142f)
            {
                orient = 2;
                if (vy < 0)
                    orient = 6;
            }
            else
            {
                orient = 3;
                if (vy < 0)
                    orient = 7;
            }
        }
        else
        {
            if (tantheta > 0.4142f)
            {
                if (tantheta > 2.4142f)
                {
                    orient = 2;
                    if (vy < 0)
                        orient = 6;
                }
                else
                {
                    orient = 1;
                    if (vy < 0)
                        orient = 5;
                }
            }
        }

        if (orient == 0 && vx < 0.0f)
            orient = 4;

        return orient;
    }

    bool EvAngularVelocityEstimator::calculateNormalFlow()
    {
        cv::Mat threshold_mat;

        cv::threshold(undistorted_last_timestamp_map_, threshold_mat, 0, 1, cv::THRESH_BINARY);

        cv::blur(threshold_mat, threshold_mat, cv::Size(opts_.PATCH_SIZE, opts_.PATCH_SIZE));

        cv::Mat neighbor_counting = threshold_mat * opts_.PATCH_SIZE * opts_.PATCH_SIZE;

        double threshold = opts_.NEIGHBOR_CNT_THRESHOLD;
        cv::threshold(threshold_mat, threshold_mat, threshold, 1, cv::THRESH_BINARY);


        // only use the middle part of the interval
        double normalized_t_upper = 0.65;
        double normalized_t_lower = 0.35;
        cv::Mat lower_threshold_mat;

        cv::threshold(undistorted_last_timestamp_map_, lower_threshold_mat, normalized_t_lower, 1, cv::THRESH_BINARY);

        cv::Mat upper_threshold_mat;

        cv::threshold(undistorted_last_timestamp_map_, upper_threshold_mat, normalized_t_upper, 1, cv::THRESH_BINARY_INV);

        // {
        //     int nonzero_count = cv::countNonZero(lower_threshold_mat);
        //     LOG(ERROR) << "lower_threshold_mat nonzero_count : " << nonzero_count;
        // }
        // {
        //     int nonzero_count = cv::countNonZero(upper_threshold_mat);
        //     LOG(ERROR) << " upper_threshold_matnonzero_count : " << nonzero_count;
        // }

        cv::Mat interval_threshold_mat = upper_threshold_mat.mul(lower_threshold_mat);

        threshold_mat = threshold_mat.mul(interval_threshold_mat);

        double mean_threshold = opts_.PATCH_MEAN_TOLERANCE;

        // Create a kernel for averaging non-zero values in the neighborhood
        cv::Mat kernel = cv::Mat::ones(opts_.PATCH_SIZE, opts_.PATCH_SIZE, CV_32F);

        // Apply the filter2D operation to compute the neighborhood average
        cv::Mat result;
        cv::filter2D(undistorted_last_timestamp_map_, result, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
        // Divide the result by the count of non-zero pixels in the neighborhood
        cv::divide(result, neighbor_counting, result);
        cv::Mat diff_mat = cv::abs(result - undistorted_last_timestamp_map_);
        cv::Mat diff_mat_threshold;
        cv::threshold(diff_mat, diff_mat_threshold, mean_threshold, 1, cv::THRESH_BINARY_INV);

        cv::Mat final_threshold_res = threshold_mat.mul(diff_mat_threshold);
        // {
        //     int nonzero_count = cv::countNonZero(final_threshold_res);
        //     LOG(ERROR) << "nonzero_count : " << nonzero_count;
        // }

        // remove pixel near boundaries
        cv::Mat boundary_mask = cv::Mat::zeros(pCam_->height_, pCam_->width_, CV_32FC1);

        cv::Mat center_area = boundary_mask(cv::Rect(opts_.HALF_PATCH_SIZE, opts_.HALF_PATCH_SIZE, pCam_->width_ - 2 * opts_.HALF_PATCH_SIZE, pCam_->height_ - 2 * opts_.HALF_PATCH_SIZE));

        center_area.setTo(1);
        final_threshold_res = final_threshold_res.mul(boundary_mask);

        int nonzero_count = cv::countNonZero(final_threshold_res);
        // LOG(WARNING) << "nonzero_count : " << nonzero_count;

        std::vector<cv::Point> no_zero_locations;
        final_threshold_res.convertTo(final_threshold_res, CV_8UC1);
        cv::findNonZero(final_threshold_res, no_zero_locations);

        // stores flow data, x ,y , fx, fy, cov
        flow_data_ = Eigen::ArrayXXd::Constant(5, nonzero_count, 0.0);

        int valid_nf_count = 0;

        for (int i = 0; i < no_zero_locations.size(); ++i)
        {
            int pos_x = no_zero_locations[i].x;
            int pos_y = no_zero_locations[i].y;

            cv::Rect surrounding_rect(pos_x - opts_.HALF_PATCH_SIZE, pos_y - opts_.HALF_PATCH_SIZE, opts_.PATCH_SIZE, opts_.PATCH_SIZE);

            cv::Mat surrounding_mat = undistorted_last_timestamp_map_(surrounding_rect);

            Eigen::MatrixXd x_hat;
            double cov = 0;
            if (planeFitting(pos_x, pos_y, x_hat, cov))
            {
                flow_data_(0, valid_nf_count) = no_zero_locations[i].x;
                flow_data_(1, valid_nf_count) = no_zero_locations[i].y;

                double time_gradient_norm_square = x_hat(0, 0) * x_hat(0, 0) + x_hat(1, 0) * x_hat(1, 0);

                // LOG(INFO) << "frame_duration_: " << frame_duration_;
                flow_data_(2, valid_nf_count) = x_hat(0, 0) / time_gradient_norm_square / frame_duration_;
                flow_data_(3, valid_nf_count) = x_hat(1, 0) / time_gradient_norm_square / frame_duration_;

                flow_data_(4, valid_nf_count) = cov;

                if (time_gradient_norm_square < 1e-4)
                {

                    continue;
                }

                valid_nf_count++;
            }
        }

        Eigen::ArrayXXd flow_data_temp = flow_data_.leftCols(valid_nf_count);
        flow_data_ = flow_data_temp;

        // LOG(INFO) << "valid_nf_count : " << valid_nf_count;

        return randomSelect();
    }

    bool EvAngularVelocityEstimator::planeFitting(int pos_x, int pos_y, Eigen::MatrixXd &x_hat, double &cov)
    {
        int patch_square = opts_.PATCH_SIZE * opts_.PATCH_SIZE;
        Eigen::MatrixXd A(patch_square, 3);
        Eigen::VectorXd b(patch_square);

        int valid_cout = 0;
        for (int m = 0; m < opts_.PATCH_SIZE; m++)
        {
            for (int n = 0; n < opts_.PATCH_SIZE; n++)
            {

                if (undistorted_last_timestamp_map_.at<float>(pos_y + n - opts_.HALF_PATCH_SIZE, pos_x + m - opts_.HALF_PATCH_SIZE) < 0.01)
                {
                    continue;
                }

                A.row(valid_cout) << m - opts_.HALF_PATCH_SIZE, n - opts_.HALF_PATCH_SIZE, 1;

                b(valid_cout) = undistorted_last_timestamp_map_.at<float>(pos_y + n - opts_.HALF_PATCH_SIZE, pos_x + m - opts_.HALF_PATCH_SIZE) - undistorted_last_timestamp_map_.at<float>(pos_y, pos_x);
                valid_cout++;
            }
        }

        Eigen::MatrixXd A_temp = A.topRows(valid_cout);
        A = A_temp;

        Eigen::VectorXd b_temp = b.head(valid_cout);
        b = b_temp;

        Eigen::ArrayXXi res_threshold(valid_cout, 1);
        res_threshold.setOnes();

        x_hat = A.colPivHouseholderQr().solve(b);

        int iteration_max_num = 10;
        Eigen::ArrayXXd res;
        Eigen::ArrayXXi prev_threshold(valid_cout, 1);
        prev_threshold.setOnes();
        int inlier_count;
        int i = 0;

        Eigen::MatrixXd inlier_A, inlier_b;
        inlier_A = A;
        inlier_b = b;
        for (; i < iteration_max_num; i++)
        {
            // std::cout << "i: " << i << std::endl;
            res = b - A * x_hat;
            res = res.abs();
            res_threshold = (res < opts_.PLANE_TOLERANCE).cast<int>();

            if (((res_threshold - prev_threshold) == 0).all())
            {
                break;
            }

            inlier_count = res_threshold.sum();

            if (inlier_count < valid_cout * opts_.TERMINATE_INLIER_RATIO)
            {
                x_hat.setZero();

                return false;
            }

            prev_threshold = res_threshold;

            Eigen::MatrixXd new_A(inlier_count, 3);
            Eigen::VectorXd new_b(inlier_count);

            int valid_index = 0;
            for (int k = 0; k < valid_cout; k++)
            {
                if (res_threshold(k, 0) == 1)
                {
                    new_A.row(valid_index) = A.row(k);
                    new_b.row(valid_index) = b.row(k);
                    valid_index++;
                }
            }

            x_hat = new_A.colPivHouseholderQr().solve(new_b);
            inlier_A = new_A;
            inlier_b = new_b;
        }

        Eigen::VectorXd plane_residue = A * x_hat - b;
        double sigma = plane_residue.dot(plane_residue);
        Eigen::Matrix3d plane_cov = sigma / (plane_residue.rows() - 3) * (A.transpose() * A).inverse();
        double A2_B2 = x_hat(0, 0) * x_hat(0, 0) + x_hat(1, 0) * x_hat(1, 0);
        double A2_B2_32 = A2_B2 * std::sqrt(A2_B2);
        Eigen::Vector3d dn_norm_dx_hat(-x_hat(0, 0) / A2_B2_32, -x_hat(1, 0) / A2_B2_32, 0);
        double n_norm_var = dn_norm_dx_hat.transpose() * plane_cov * dn_norm_dx_hat;

        cov = n_norm_var / frame_duration_ / frame_duration_;

        return true;
    }

    bool EvAngularVelocityEstimator::randomSelect()
    {
        int valid_nf_count = flow_data_.cols();
        if (valid_nf_count < 30)
        {
            return false;
        }
        std::vector<std::vector<Eigen::ArrayXXd>> flow_data_all_orientations;

        flow_data_all_orientations.resize(8);
        for (size_t i = 0; i < 8; i++)
        {
            flow_data_all_orientations[i].reserve(valid_nf_count);
        }
        for (size_t i = 0; i < valid_nf_count; i++)
        {
            size_t orient = computeOrientationBin(flow_data_(2, i), flow_data_(3, i));
            flow_data_all_orientations[orient].push_back(flow_data_.col(i));
        }

        for (size_t i = 0; i < 8; i++)
        {
            unsigned seed = i;
            std::default_random_engine e1(seed);
            std::shuffle(flow_data_all_orientations[i].begin(), flow_data_all_orientations[i].end(), e1);
        }

        size_t maxNumCorrespondenceInEachOrientation = 100;
        Eigen::ArrayXXd selected_flow_data_(5, valid_nf_count);
        int selected_count = 0;
        for (size_t i = 0; i < 8; i++)
        {
            size_t num = flow_data_all_orientations[i].size();
            if (num > maxNumCorrespondenceInEachOrientation)
                num = maxNumCorrespondenceInEachOrientation;
            for (size_t j = 0; j < num; j++)
            {
                selected_flow_data_.col(selected_count) = flow_data_all_orientations[i][j];
                selected_count++;
            }
        }
        // LOG(INFO) << "selected_count: " << selected_count;

        // flow_data_ = selected_flow_data_.leftCols(selected_count);

        return true;
    }
}