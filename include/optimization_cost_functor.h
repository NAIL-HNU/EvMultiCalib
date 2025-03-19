#pragma once

#include <vector>

#include <sophus/so3.hpp>
#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include "basalt/spline/ceres_local_param.hpp"
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/spline/ceres_spline_helper_jet.h"

#include "calibration.h"

#define SPLINE_DEGREE 3

namespace ev_calib {
    struct angVelCostFunctorWoBiasTimeOffset {
        angVelCostFunctorWoBiasTimeOffset() = delete;
        angVelCostFunctorWoBiasTimeOffset(double timestamp, Eigen::Vector3d ang_vel, double weight, double dt_knots) : timestamp_(timestamp), ang_vel_(std::move(ang_vel)), weight_(weight), dt_knots_(dt_knots) {};

        Eigen::Vector3d ang_vel_;
        double timestamp_;
        double weight_;
        double dt_knots_;

        template<typename T>
        bool operator()(T const *const *data, T *residuals) {
            T u = T(timestamp_);

            Sophus::SO3<T> spline_rotation;
            Eigen::Matrix<T, 3, 1> spline_ang_vel;
            ns_ctraj::CeresSplineHelperJet<T, SPLINE_DEGREE + 1>::template EvaluateLie(data, u, 1.0 / dt_knots_, &spline_rotation, &spline_ang_vel, nullptr, nullptr);

            Eigen::Matrix<T, 3, 1> ang_vel_res = ang_vel_ - spline_ang_vel;

            Eigen::Map<Eigen::Matrix<T, 3, 1> > residual(residuals);

            residual = ang_vel_res * weight_;

            return true;
        }
    };

    struct angVelCostFunctor {
        angVelCostFunctor() = delete;
        angVelCostFunctor(double timestamp, Eigen::Vector3d ang_vel, double weight, double dt_knots, int time_offset_bound) : timestamp_(timestamp), ang_vel_(std::move(ang_vel)), weight_(weight), dt_knots_(dt_knots), time_offset_bound_(time_offset_bound) {};

        Eigen::Vector3d ang_vel_;
        double timestamp_;
        double weight_;
        double dt_knots_;
        int time_offset_bound_;

        template<typename T>
        bool operator()(T const *const *data, T *residuals) {
            T time_offset = data[0][0] / dt_knots_;
            T u = T(timestamp_) + time_offset;

            Eigen::Quaternion<T> quat(data[1][3], data[1][0], data[1][1], data[1][2]);
            Sophus::SO3<T> extrinsic = Sophus::SO3<T>(quat);

            Eigen::Matrix<T, 3, 1> gyro_bias(data[2][0], data[2][1], data[2][2]);

            int control_point_offset = 0;

            for (int i = -time_offset_bound_; i < time_offset_bound_ + 1; i++) {
                if (u < T(i) || u > T(i + 1))
                    continue;

                u = u - T(i);

                control_point_offset = i + 3 + time_offset_bound_;
                break;
            }

            Sophus::SO3<T> spline_rotation;
            Eigen::Matrix<T, 3, 1> spline_ang_vel;
            ns_ctraj::CeresSplineHelperJet<T, SPLINE_DEGREE + 1>::template EvaluateLie(data + control_point_offset, u, 1.0 / dt_knots_, &spline_rotation, &spline_ang_vel, nullptr, nullptr);

            Eigen::Matrix<T, 3, 1> ang_vel_res = extrinsic.matrix() * (ang_vel_ - gyro_bias) - spline_ang_vel;

            Eigen::Map<Eigen::Matrix<T, 3, 1> > residual(residuals);

            residual = ang_vel_res * weight_;

            return true;
        }
    };

    struct rotationCostFunctor {
    public:
        rotationCostFunctor() = delete;

        rotationCostFunctor(double timestamp_prev, double timestamp_cur, Sophus::SO3d rotation_delta, double weight, double dt_knots, int cp_number_involved, int time_offset_bound) : timestamp_prev_(timestamp_prev), timestamp_cur_(timestamp_cur), rotation_delta_(rotation_delta), weight_(weight), dt_knots_(dt_knots), cp_number_involved_(cp_number_involved), time_offset_bound_(time_offset_bound) {};

        Sophus::SO3d rotation_delta_;
        double timestamp_prev_, timestamp_cur_;
        double weight_;
        double dt_knots_;
        int cp_number_involved_;
        int time_offset_bound_;

        template<typename T>
        bool operator()(T const *const *data, T *residuals) {
            T time_offset = data[0][0] / dt_knots_;
            T u_prev = T(timestamp_prev_) + time_offset;
            T u_cur = T(timestamp_cur_) + time_offset;

            Eigen::Quaternion<T> quat(data[1][3], data[1][0], data[1][1], data[1][2]);
            Sophus::SO3<T> extrinsic = Sophus::SO3<T>(quat);

            // LOG(INFO) << "data[0][0]: " << data[0][0];
            // LOG(INFO) << "quat: " << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w();

            int control_point_offset_prev = 0;

            for (int i = -time_offset_bound_; i < time_offset_bound_ + 1; i++) {
                if (u_prev < T(i) || u_prev > T(i + 1))
                    continue;

                u_prev = u_prev - T(i);

                control_point_offset_prev = i + 2 + time_offset_bound_;
                break;
            }

            Sophus::SO3<T> spline_rotation_prev;
            ns_ctraj::CeresSplineHelperJet<T, SPLINE_DEGREE + 1>::template EvaluateLie(data + control_point_offset_prev, u_prev, 1.0 / dt_knots_, &spline_rotation_prev, nullptr, nullptr, nullptr);

            int control_point_offset_cur = 0;

            for (int i = -time_offset_bound_; i < time_offset_bound_ + 1; i++) {
                if (u_cur < T(i) || u_cur > T(i + 1))
                    continue;

                u_cur = u_cur - T(i);

                control_point_offset_cur = i + 2 + time_offset_bound_ + cp_number_involved_;
                break;
            }

            Sophus::SO3<T> spline_rotation_cur;
            ns_ctraj::CeresSplineHelperJet<T, SPLINE_DEGREE + 1>::template EvaluateLie(data + control_point_offset_cur, u_cur, 1.0 / dt_knots_, &spline_rotation_cur, nullptr, nullptr, nullptr);

            // Sophus::SO3<T> error = extrinsic * rotation_.cast<T>() * spline_rotation.inverse();
            Sophus::SO3<T> error = extrinsic.inverse() * spline_rotation_prev.inverse() * spline_rotation_cur * extrinsic * rotation_delta_.cast<T>();

            Eigen::Map<Eigen::Matrix<T, 3, 1> > residual(residuals);
            residual = error.log() * T(weight_);

            return true;
        }
    };

}
