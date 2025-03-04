#include "utils.h"


PerspectiveCamera::PerspectiveCamera() {}

PerspectiveCamera::~PerspectiveCamera() {}

void PerspectiveCamera::setIntrinsicParameters(
        size_t width,
        size_t height,
        std::string &cameraName,
        std::string &distortion_model,
        std::vector<double> &vD,
        std::vector<double> &vK) {
    width_ = width;
    height_ = height;
    cameraName_ = cameraName;
    distortion_model_ = distortion_model;
    D_ = (cv::Mat_<double>(1,5) << vD[0], vD[1], vD[2], vD[3], vD[4]);
    K_ = (cv::Mat_<double>(3,3) << vK[0], vK[1], vK[2],
                                              vK[3], vK[4], vK[5],
                                              vK[6], vK[7], vK[8]);
//    preComputeUndistortedCoordinate();
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    cv::initUndistortRectifyMap(K_, D_, R, K_,
                                cv::Size(width_, height_), CV_32FC1,
                                precomputed_undistorted_x_, precomputed_undistorted_y_);
}