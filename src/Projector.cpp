#include "Projector.hpp"
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <cmath>
#include <limits>

// ** Public methods **

Projector::Projector() : overlay_updated_(false)
{}

bool Projector::init(
    double min_range, double max_range,
    double min_ang_fov, double max_ang_fov,
    const std::vector<double>& cam,
    const std::vector<double>& d,
    const std::vector<double>& rlc,
    const std::vector<double>& tlc)
{
    minRange_ = min_range;
    maxRange_ = max_range;
    minAngFOV_ = min_ang_fov;
    maxAngFOV_ = max_ang_fov;

    if (cam.size() != 9 || d.size() != 5 || rlc.size() != 9 || tlc.size() != 3) {
        std::cerr << "[Projector::init] Invalid calibration parameter sizes" << std::endl;
        return false;
    }

    // Camera matrix 3x3
    cameraMatrix_ = cv::Mat::eye(3, 3, CV_64F);
    std::memcpy(cameraMatrix_.data, cam.data(), sizeof(double) * 9);

    // Distortion coefficients as column vector 5x1
    distCoeffs_ = cv::Mat(d).clone().reshape(1, 5);

    // Rotation vector and matrix
    cv::Mat RlcMat(3, 3, CV_64F, const_cast<double*>(rlc.data()));
    cv::Rodrigues(RlcMat, rvec_);
    cv::Rodrigues(rvec_, R_);
    tvec_ = cv::Mat(3, 1, CV_64F, const_cast<double*>(tlc.data())).clone();
    R_inv_ = R_.t();

    // Semantic LUT (label index to BGR color)
    semantic_lut_ = cv::Mat(1, 3, CV_8UC3);
    semantic_lut_.at<cv::Vec3b>(0, 0) = {0,   0, 255};  // Dynamic – red
    semantic_lut_.at<cv::Vec3b>(0, 1) = {0, 255,   0};  // Obstacle – green
    semantic_lut_.at<cv::Vec3b>(0, 2) = {255, 0,   0};  // Traversable – blue

    // Scale camera matrix to label image size
    K_ = cameraMatrix_.clone();
    double sx = static_cast<double>(LABEL_W) / CALIB_W;
    double sy = static_cast<double>(LABEL_H) / CALIB_H;
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

    // 1. Convert input image to label matrix
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
    cv_ptr = cv_bridge::toCvShare(image_msg);
    } catch (const cv_bridge::Exception& e) {
        std::cout << "[Projector::project_cloud_onto_image] cv_bridge error: " << e.what() << std::endl;
        return false;
    }
    cv_ptr->image.convertTo(labels_, CV_8UC1);

    // cv::medianBlur(labels_, labels_, 5);

    // int clusters[3] = {0, 0, 0};

    // for (int class_id = 0; class_id < 3; ++class_id)
    // {
    //     // Binary mask for this class (255 where pixel == class_id)
    //     cv::Mat mask = (labels_ == class_id);

    //     cv::Mat cc_labels, stats, centroids;
    //     int n = cv::connectedComponentsWithStats(
    //                 mask, cc_labels, stats, centroids,
    //                 8,                 // 4- or 8-connectivity
    //                 CV_32S);

    //     clusters[class_id] = n - 1;      // row 0 is background
    // }

    // ROS_INFO_STREAM("Clusters  =  class Dynamic: " << clusters[0]
    //                       << "   class Static: " << clusters[1]
    //                       << "   class Traversable: " << clusters[2]);

    // 2. Convert cloud message to pcl and filter points
    pcl::PointCloud<pcl::PointXYZ> cloud_in;
    pcl::fromROSMsg(*cloud_msg, cloud_in);
    this->filterPointCloud(cloud_in);

    // 3. Project filtered points into 2D image space
    proj2d_.clear();
    cv::projectPoints(pts3d_, rvec_, tvec_, K_, distCoeffs_, proj2d_);
    
    // 4. Create depth and index buffers for projected points
    this->createDepthBuffers();

    overlay_updated_ = false;  // Mark overlay cache as dirty
		
		return true;
}

// ** Private methods **

void Projector::filterPointCloud(const pcl::PointCloud<pcl::PointXYZ>& cloud_in)
{
  pts3d_.clear();

  const double minAngRad = minAngFOV_ * M_PI / 180.0;
  const double maxAngRad = maxAngFOV_ * M_PI / 180.0;

  for (const auto& pt : cloud_in.points)
  {
    if (pt.x <= 0.0) continue;

    const double range = std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
    if (range < minRange_ || range > maxRange_) continue;

    const double angle = std::atan2(std::sqrt(pt.y*pt.y + pt.z*pt.z), pt.x);
    if (angle < minAngRad || angle > maxAngRad) continue;

    pts3d_.emplace_back(pt.x, pt.y, pt.z);
  }
}

void Projector::createDepthBuffers()
{
    // Initialize depth and index buffers
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

      const double Zc = R_.at<double>(2,0)*p.x + 
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

    // -----------------------------
    // 1) Parámetros "low quality"
    // -----------------------------
    // Downscale factor: 2, 3, 4... (más grande => menos resolución => menos red)
    const int DS = 4;

    const int outW = std::max(1, LABEL_W / DS);
    const int outH = std::max(1, LABEL_H / DS);

    // -----------------------------
    // 2) Base: raw image -> BGR -> resize a (outW,outH)
    // -----------------------------
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

            // Intenta mantenerlo simple: si ya es 8UC3 lo usamos, si es mono lo pasamos a BGR
            if (raw_ptr->image.type() == CV_8UC3) {
                raw_bgr = raw_ptr->image;
            } else if (raw_ptr->image.type() == CV_8UC1) {
                cv::cvtColor(raw_ptr->image, raw_bgr, cv::COLOR_GRAY2BGR);
            } else {
                // Fallback: intenta convertir a 8UC3 (puede que no funcione para todos los encodings)
                raw_ptr->image.convertTo(raw_bgr, CV_8UC3);
            }

            // Primero lo llevamos a tamaño label, luego downsample para que quede pixelado/ligero
            cv::Mat raw_label;
            if (raw_bgr.cols != LABEL_W || raw_bgr.rows != LABEL_H) {
                cv::resize(raw_bgr, raw_label, cv::Size(LABEL_W, LABEL_H), 0, 0, cv::INTER_NEAREST);
            } else {
                raw_label = raw_bgr;
            }

            // Downsample: INTER_AREA suele ser buen compromiso
            cv::resize(raw_label, base_small, cv::Size(outW, outH), 0, 0, cv::INTER_AREA);
        }
    }

    // overlay_cache_ será la imagen final (pequeña)
    overlay_cache_ = base_small.clone();

    // -----------------------------
    // 3) Dibujar círculos con radio ~ 1/depth
    // -----------------------------
    // Ajusta estos 3 valores a gusto:
    const float r_min = 0.1f;     // radio mínimo en píxeles (en la imagen pequeña)
    const float r_max = 2.0f;     // radio máximo en píxeles (en la imagen pequeña)
    const float k_rad = 10.0f;    // ganancia del 1/depth  (más alto => círculos más grandes)

    // Nota: depth_buf_ está en tamaño LABEL_W/H con Zc (float). idx_buf_ también.
    // Vamos a recorrer la imagen "grande", pero dibujar en la pequeña.
    for (int v = 0; v < LABEL_H; ++v)
    {
        for (int u = 0; u < LABEL_W; ++u)
        {
            const int idx = idx_buf_.at<int>(v, u);
            if (idx < 0) continue;

            // Coordenadas en imagen pequeña
            const int us = u / DS;
            const int vs = v / DS;
            if (us < 0 || us >= outW || vs < 0 || vs >= outH)
                continue;

            // Profundidad para radio
            const float z = depth_buf_.at<float>(v, u);
            if (!std::isfinite(z) || z <= 0.0f)
                continue;

            // label -> color
            int label_idx = static_cast<int>(labels_.at<uchar>(v, u));
            if (label_idx < 0 || label_idx >= semantic_lut_.cols)
                label_idx = 0;

            const cv::Vec3b& bgr = semantic_lut_.at<cv::Vec3b>(0, label_idx);

            // Radio inversamente proporcional a z
            float r = k_rad / z;
            if (r < r_min) r = r_min;
            if (r > r_max) r = r_max;

            // Dibujar círculo relleno (thickness = -1)
            cv::circle(
                overlay_cache_,
                cv::Point(us, vs),
                static_cast<int>(std::round(r)),
                cv::Scalar(bgr[0], bgr[1], bgr[2]),
                -1,
                cv::LINE_AA
            );
        }
    }

    overlay_updated_ = true;
    return overlay_cache_;
}


void Projector::getSemanticClouds(
    pcl::PointCloud<pcl::PointXYZRGB>& semanticCloud,
    pcl::PointCloud<pcl::PointXYZRGB>& travCloud,
    pcl::PointCloud<pcl::PointXYZRGB>& obstacleCloud) const
{
    semanticCloud.clear();
    travCloud.clear();
    obstacleCloud.clear();

    const int H = idx_buf_.rows;
    const int W = idx_buf_.cols;

    // 1) Populate semanticCloud, travCloud, obstacleCloud exactly as before
    for (int v = 0; v < H; ++v)
    {
        for (int u = 0; u < W; ++u)
        {
            const int idx = idx_buf_.at<int>(v, u);
            if (idx < 0)
                continue;

            const uchar label = labels_.at<uchar>(v, u);
            const auto& p = pts3d_[idx];

            // fill a colored PointXYZRGB
            pcl::PointXYZRGB pt_rgb;
            pt_rgb.x = p.x;
            pt_rgb.y = p.y;
            pt_rgb.z = p.z;

            // guard against out‐of‐range label
            if (label < 0 || label >= semantic_lut_.cols)
                continue;

            const cv::Vec3b& color = semantic_lut_.at<cv::Vec3b>(0, label);
            pt_rgb.r = color[2];
            pt_rgb.g = color[1];
            pt_rgb.b = color[0];

            semanticCloud.push_back(pt_rgb);

            if (label == 2)
            {
                travCloud.push_back(pt_rgb);
            }
            else if (label == 1 || label == 0)
            {
                obstacleCloud.push_back(pt_rgb);
            }
        }
    }



    // 2) If travCloud is non‐empty, run RANSAC plane segmentation to keep only the largest planar inliers
    if (!travCloud.empty())
    {
        // (a) Set up the segmentation object
        pcl::SACSegmentation<pcl::PointXYZRGB> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.1f);

        // (b) Provide the input cloud
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
        seg.setInputCloud(travCloud.makeShared());
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty())
        {
            // No plane found—keep travCloud as is (or optionally clear it)
            // travCloud remains unchanged or you can choose to clear if you only want planar points.
        }
        else
        {
            // (c) Extract only the planar inliers
            pcl::ExtractIndices<pcl::PointXYZRGB> extract;
            extract.setInputCloud(travCloud.makeShared());
            extract.setIndices(inliers);
            extract.setNegative(false); // false = keep inliers, true = remove inliers

            pcl::PointCloud<pcl::PointXYZRGB> filteredTrav, obstacleTrav;
            extract.filter(filteredTrav);

            extract.setNegative(false); // false = keep inliers, true = remove inliers
            extract.filter(obstacleTrav);
            travCloud = filteredTrav;  // Assign the filtered points back to travCloud
            for(const auto& pt : obstacleTrav.points)
            {
                // Add the points that were not part of the plane to obstacleCloud
                pcl::PointXYZRGB pt_rgb;
                pt_rgb.x = pt.x;
                pt_rgb.y = pt.y;
                pt_rgb.z = pt.z;
                pt_rgb.r = 255;
                pt_rgb.g = 255;
                pt_rgb.b = 0; 
                obstacleCloud.push_back(pt_rgb);
            }
        }
    }
}


// void Projector::getSemanticClouds(pcl::PointCloud<pcl::PointXYZRGB>& semanticCloud,
//                                  pcl::PointCloud<pcl::PointXYZRGB>& travCloud,
//                                  pcl::PointCloud<pcl::PointXYZRGB>& obstacleCloud) const
// {
//     semanticCloud.clear();
//     travCloud.clear();
//     obstacleCloud.clear();

//     const int H = idx_buf_.rows;
//     const int W = idx_buf_.cols;

//     for (int v = 0; v < H; ++v)
//     {
//         for (int u = 0; u < W; ++u)
//         {
//             const int idx = idx_buf_.at<int>(v, u);
//             if (idx < 0) continue;

//             const uchar label = labels_.at<uchar>(v, u);

//             const auto& p = pts3d_[idx];

//             pcl::PointXYZRGB pt_rgb;
//             pt_rgb.x = p.x;
//             pt_rgb.y = p.y;
//             pt_rgb.z = p.z;

//             if (label < 0 || label >= semantic_lut_.cols)
//                 continue;

//             const cv::Vec3b& color = semantic_lut_.at<cv::Vec3b>(0, label);
//             pt_rgb.r = color[2];
//             pt_rgb.g = color[1];
//             pt_rgb.b = color[0];

//             semanticCloud.push_back(pt_rgb);

//             if (label == 2)
//                 travCloud.push_back(pt_rgb);
//             else if (label == 1 || label == 0)
//                 obstacleCloud.push_back(pt_rgb);
//         }
//     }
// }

const cv::Mat& Projector::getDepthMap()
{
    static cv::Mat depth_color;

    if (depth_buf_.empty())            // aún no hay datos
        return depth_color;

    // 1) Normaliza solo los píxeles válidos [minRange_, maxRange_] → [0,255]
    cv::Mat depth_norm, depth_u8;
    cv::Mat validMask = depth_buf_ < std::numeric_limits<float>::infinity();

    // Saturamos fuera de rango para que no escale mal el histograma
    cv::Mat depth_clipped;
    depth_buf_.copyTo(depth_clipped);
    depth_clipped.setTo(maxRange_, depth_clipped > maxRange_);
    depth_clipped.setTo(minRange_, depth_clipped < minRange_);

    cv::normalize(depth_clipped, depth_norm, 255, 0,
                  cv::NORM_MINMAX, CV_32F, validMask);

    // 2) A 8 bit
    depth_norm.convertTo(depth_u8, CV_8U);

    // 3) Colormap TURBO (OpenCV 4; cambia a JET si tu OpenCV es 3.x)
    cv::applyColorMap(depth_u8, depth_color, cv::COLORMAP_TURBO);

    // 4) Píxeles sin dato a negro
    depth_color.setTo(cv::Scalar::all(0), ~validMask);

    return depth_color;
}
