#include "Projector.hpp"

#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>

#include <cmath>
#include <limits>
#include <cstring>
#include <iostream>

Projector::Projector() : overlay_updated_(false) {}

bool Projector::init(
    double min_range, double max_range,
    double min_ang_fov, double max_ang_fov,
    const std::vector<double>& cam,
    const std::vector<double>& d,
    const std::vector<double>& rlc,
    const std::vector<double>& tlc,
    const std::vector<SemanticClass>& classes)
{
    minRange_ = min_range;
    maxRange_ = max_range;
    minAngFOV_ = min_ang_fov;
    maxAngFOV_ = max_ang_fov;

    classes_ = classes;
    bgr_by_id_.clear();
    id_by_rgb_.clear();

    if (classes_.empty())
    {
        std::cerr << "[Projector::init] No classes provided" << std::endl;
        return false;
    }

    // Build LUTs from YAML
    for (const auto& c : classes_)
    {
        bgr_by_id_[c.id] = cv::Vec3b(c.b, c.g, c.r); // OpenCV BGR
        id_by_rgb_[packRGB(c.r, c.g, c.b)] = c.id;   // semantic image assumed RGB
    }

    if (cam.size() != 9 || d.size() != 5 || rlc.size() != 9 || tlc.size() != 3)
    {
        std::cerr << "[Projector::init] Invalid calibration parameter sizes" << std::endl;
        return false;
    }

    // Camera matrix 3x3
    cameraMatrix_ = cv::Mat::eye(3, 3, CV_64F);
    std::memcpy(cameraMatrix_.data, cam.data(), sizeof(double) * 9);

    // Distortion coefficients (5)
    distCoeffs_ = cv::Mat(d).clone().reshape(1, 5);

    // Rotation matrix (3x3) + translation (3x1)
    cv::Mat RlcMat(3, 3, CV_64F, const_cast<double*>(rlc.data()));
    cv::Rodrigues(RlcMat, rvec_);
    cv::Rodrigues(rvec_, R_);
    tvec_ = cv::Mat(3, 1, CV_64F, const_cast<double*>(tlc.data())).clone();
    R_inv_ = R_.t();

    // Scale camera matrix to label image size
    K_ = cameraMatrix_.clone();
    const double sx = static_cast<double>(LABEL_W) / CALIB_W;
    const double sy = static_cast<double>(LABEL_H) / CALIB_H;
    K_.at<double>(0, 0) *= sx; K_.at<double>(0, 2) *= sx;
    K_.at<double>(1, 1) *= sy; K_.at<double>(1, 2) *= sy;

    return true;
}

bool Projector::project_cloud_onto_image(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg,
                                         const sensor_msgs::msg::Image::ConstSharedPtr& image_msg)
{
    if (!cloud_msg || !image_msg)
    {
        std::cout << "[Projector::project_cloud_onto_image] Received null cloud or image message" << std::endl;
        return false;
    }

    // 1) Read semantic image WITHOUT converting whole frame to IDs (Optimization #2)
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(image_msg);
    } catch (const cv_bridge::Exception& e) {
        std::cout << "[Projector::project_cloud_onto_image] cv_bridge error: " << e.what() << std::endl;
        return false;
    }

    const std::string enc = image_msg->encoding;

    semantic_is_color_ = false;
    semantic_rgb_.release();
    labels_.release();

    if (enc == "mono8" || cv_ptr->image.type() == CV_8UC1)
    {
        // IDs directos (CV_8UC1)
        labels_ = cv_ptr->image;
        semantic_is_color_ = false;
    }
    else if (enc == "rgb8" || enc == "bgr8" || enc == "rgba8" || enc == "bgra8")
    {
        // Guardar como RGB CV_8UC3 y decodificar label_id SOLO cuando haga falta
        semantic_is_color_ = true;

        cv::Mat img = cv_ptr->image;
        const bool has_alpha = (enc == "rgba8" || enc == "bgra8");
        const bool is_bgr    = (enc == "bgr8"  || enc == "bgra8");

        if (has_alpha)
        {
            if (img.type() != CV_8UC4) img.convertTo(img, CV_8UC4);
            cv::cvtColor(img, semantic_rgb_, is_bgr ? cv::COLOR_BGRA2RGB : cv::COLOR_RGBA2RGB);
        }
        else
        {
            if (img.type() != CV_8UC3) img.convertTo(img, CV_8UC3);

            if (is_bgr)
                cv::cvtColor(img, semantic_rgb_, cv::COLOR_BGR2RGB);
            else
                semantic_rgb_ = img; // already RGB
        }
    }
    else
    {
        std::cout << "[Projector] Unsupported semantic image encoding: " << enc
                  << " (expected mono8/rgb8/bgr8/rgba8/bgra8)" << std::endl;
        return false;
    }

    // 2) Convert cloud and filter
    pcl::PointCloud<pcl::PointXYZ> cloud_in;
    pcl::fromROSMsg(*cloud_msg, cloud_in);
    filterPointCloud(cloud_in);

    // 3) Project to 2D
    proj2d_.clear();
    cv::projectPoints(pts3d_, rvec_, tvec_, K_, distCoeffs_, proj2d_);
    std::cout << proj2d_.size() << std::endl;

    // 4) z-buffer
    createDepthBuffers();

    overlay_updated_ = false;
    return true;
}

void Projector::filterPointCloud(const pcl::PointCloud<pcl::PointXYZ>& cloud_in)
{
    pts3d_.clear();

    const double minAngRad = minAngFOV_ * M_PI / 180.0;
    const double maxAngRad = maxAngFOV_ * M_PI / 180.0;

    for (const auto& pt : cloud_in.points)
    {
        //if (pt.x <= 0.0) continue;

        const double range = std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
        //if (range < minRange_ || range > maxRange_) continue;

        // Keep your original behavior (cone angle, always >= 0)
        const double angle = std::atan2(std::sqrt(pt.y*pt.y + pt.z*pt.z), pt.x);
        //if (angle < minAngRad || angle > maxAngRad) continue;

        pts3d_.emplace_back(-pt.x, -pt.y, pt.z);
    }
}

void Projector::createDepthBuffers()
{
    depth_buf_.create(cv::Size(LABEL_W, LABEL_H), CV_32F);
    idx_buf_.create(cv::Size(LABEL_W, LABEL_H), CV_32S);
    depth_buf_.setTo(std::numeric_limits<float>::infinity());
    idx_buf_.setTo(-1);

    const int H = depth_buf_.rows;
    const int W = depth_buf_.cols;

    for (size_t k = 0; k < proj2d_.size(); ++k)
    {
        const int u = cvRound(proj2d_[k].x);
        const int v = cvRound(proj2d_[k].y);

        if (u < 0 || u >= W || v < 0 || v >= H)
            continue;

        const auto& p = pts3d_[k];

        const double Zc =
            R_.at<double>(2,0)*p.x +
            R_.at<double>(2,1)*p.y +
            R_.at<double>(2,2)*p.z +
            tvec_.at<double>(2);

        if (Zc <= 0) continue;

        float& depthValue = depth_buf_.at<float>(v, u);
        if (Zc < depthValue)
        {
            depthValue = static_cast<float>(Zc);
            idx_buf_.at<int>(v, u) = static_cast<int>(k);
        }
    }
}

const cv::Mat& Projector::getOverlay(const sensor_msgs::msg::Image::ConstSharedPtr& raw_img_msg)
{
    if (overlay_updated_)
        return overlay_cache_;

    const int DS = 4;
    const int outW = std::max(1, LABEL_W / DS);
    const int outH = std::max(1, LABEL_H / DS);

    cv::Mat base_small(outH, outW, CV_8UC3, cv::Scalar(0, 0, 0));

    if (raw_img_msg)
    {
        cv_bridge::CvImageConstPtr raw_ptr;
        try {
            raw_ptr = cv_bridge::toCvShare(raw_img_msg);
        } catch (const cv_bridge::Exception& e) {
            std::cout << "[Projector::getOverlay] cv_bridge error (raw): " << e.what() << std::endl;
            raw_ptr.reset();
        }

        if (raw_ptr)
        {
            cv::Mat raw_bgr;

            if (raw_ptr->image.type() == CV_8UC3) {
                raw_bgr = raw_ptr->image;
            } else if (raw_ptr->image.type() == CV_8UC1) {
                cv::cvtColor(raw_ptr->image, raw_bgr, cv::COLOR_GRAY2BGR);
            } else {
                raw_ptr->image.convertTo(raw_bgr, CV_8UC3);
            }

            cv::Mat raw_label;
            if (raw_bgr.cols != LABEL_W || raw_bgr.rows != LABEL_H) {
                cv::resize(raw_bgr, raw_label, cv::Size(LABEL_W, LABEL_H), 0, 0, cv::INTER_NEAREST);
            } else {
                raw_label = raw_bgr;
            }

            cv::resize(raw_label, base_small, cv::Size(outW, outH), 0, 0, cv::INTER_AREA);
        }
    }

    overlay_cache_ = base_small.clone();

    const float r_min = 0.1f;
    const float r_max = 2.0f;
    const float k_rad = 10.0f;

    const int H = idx_buf_.rows;
    const int W = idx_buf_.cols;

    // O(Npoints): iterate projected points, keep winners
    for (size_t k = 0; k < proj2d_.size(); ++k)
    {
        const int u = cvRound(proj2d_[k].x);
        const int v = cvRound(proj2d_[k].y);

        if (u < 0 || u >= W || v < 0 || v >= H)
            continue;

        if (idx_buf_.at<int>(v, u) != static_cast<int>(k))
            continue;

        const float z = depth_buf_.at<float>(v, u);
        if (!std::isfinite(z) || z <= 0.0f)
            continue;

        const int us = u / DS;
        const int vs = v / DS;
        if (us < 0 || us >= outW || vs < 0 || vs >= outH)
            continue;

        const int label_id = labelIdAt(v, u);

        auto it = bgr_by_id_.find(label_id);
        const cv::Vec3b bgr = (it != bgr_by_id_.end()) ? it->second : cv::Vec3b(0,0,255);

        float r = k_rad / z;
        if (r < r_min) r = r_min;
        if (r > r_max) r = r_max;

        cv::circle(
            overlay_cache_,
            cv::Point(us, vs),
            static_cast<int>(std::round(r)),
            cv::Scalar(bgr[0], bgr[1], bgr[2]),
            -1,
            cv::LINE_AA
        );
    }

    overlay_updated_ = true;
    return overlay_cache_;
}

void Projector::getSemanticClouds(
    pcl::PointCloud<pcl::PointXYZRGB>& semanticCloud,
    std::unordered_map<int, pcl::PointCloud<pcl::PointXYZRGB>>& cloudsByClass) const
{
    // Optimization #1: O(Npoints), no full image scan
    semanticCloud.clear();
    cloudsByClass.clear();

    for (const auto& c : classes_)
        cloudsByClass.emplace(c.id, pcl::PointCloud<pcl::PointXYZRGB>{});

    const int H = idx_buf_.rows;
    const int W = idx_buf_.cols;

    // Iterate only projected points and keep z-buffer winners
    for (size_t k = 0; k < proj2d_.size(); ++k)
    {
        const int u = cvRound(proj2d_[k].x);
        const int v = cvRound(proj2d_[k].y);

        if (u < 0 || u >= W || v < 0 || v >= H)
            continue;

        if (idx_buf_.at<int>(v, u) != static_cast<int>(k))
            continue;

        const int label_id = labelIdAt(v, u);

        const auto& p = pts3d_[k];
        pcl::PointXYZRGB pt_rgb;
        pt_rgb.x = p.x;
        pt_rgb.y = p.y;
        pt_rgb.z = p.z;

        auto it = bgr_by_id_.find(label_id);
        const cv::Vec3b bgr = (it != bgr_by_id_.end()) ? it->second : cv::Vec3b(0,0,255);

        pt_rgb.r = bgr[2];
        pt_rgb.g = bgr[1];
        pt_rgb.b = bgr[0];

        semanticCloud.push_back(pt_rgb);

        auto itCloud = cloudsByClass.find(label_id);
        if (itCloud != cloudsByClass.end())
            itCloud->second.push_back(pt_rgb);
    }
}

const cv::Mat& Projector::getDepthMap()
{
    static cv::Mat depth_color;

    if (depth_buf_.empty())
        return depth_color;

    cv::Mat depth_norm, depth_u8;
    cv::Mat validMask = depth_buf_ < std::numeric_limits<float>::infinity();

    cv::Mat depth_clipped;
    depth_buf_.copyTo(depth_clipped);
    depth_clipped.setTo(maxRange_, depth_clipped > maxRange_);
    depth_clipped.setTo(minRange_, depth_clipped < minRange_);

    cv::normalize(depth_clipped, depth_norm, 255, 0,
                  cv::NORM_MINMAX, CV_32F, validMask);

    depth_norm.convertTo(depth_u8, CV_8U);
    cv::applyColorMap(depth_u8, depth_color, cv::COLORMAP_TURBO);
    depth_color.setTo(cv::Scalar::all(0), ~validMask);

    return depth_color;
}