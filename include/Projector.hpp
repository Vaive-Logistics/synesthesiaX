#pragma once
/**
 * @file   Projector.hpp
 * @brief  Projects 3D point cloud onto semantic image; provides semantic colored clouds and overlays.
 */

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

class Projector
{
public:
  struct SemanticClass
  {
    int id = 0;
    std::string name;
    uint8_t r = 0; // RGB
    uint8_t g = 0;
    uint8_t b = 0;
  };

  Projector();

  bool init(
      double min_range, double max_range,
      double min_ang_fov, double max_ang_fov,
      const std::vector<double>& camera_matrix,
      const std::vector<double>& dist_coeffs,
      const std::vector<double>& rlc,
      const std::vector<double>& tlc,
      const std::vector<SemanticClass>& classes
  );

  bool project_cloud_onto_image(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg,
                                const sensor_msgs::msg::Image::ConstSharedPtr& image_msg);

  /**
   * @brief Full semantic cloud + per-class clouds (by label id).
   *        Optimized: O(Npoints), iterates projected points and keeps z-buffer winners.
   */
  void getSemanticClouds(
      pcl::PointCloud<pcl::PointXYZRGB>& semanticCloud,
      std::unordered_map<int, pcl::PointCloud<pcl::PointXYZRGB>>& cloudsByClass) const;

  /**
   * @brief Get semantic overlay image (low quality). Uses z-buffer winners.
   */
  const cv::Mat& getOverlay(const sensor_msgs::msg::Image::ConstSharedPtr& raw_image_msg);

  const cv::Mat& getDepthMap();

private:
  void filterPointCloud(const pcl::PointCloud<pcl::PointXYZ>& cloud_in);
  void createDepthBuffers();

  // Pack RGB to 24-bit key for fast hash lookup
  static inline uint32_t packRGB(uint8_t r, uint8_t g, uint8_t b)
  {
    return (static_cast<uint32_t>(r) << 16) |
           (static_cast<uint32_t>(g) << 8)  |
           (static_cast<uint32_t>(b));
  }

  // Decode label id at pixel (v,u) (handles both mono-id and RGB semantic images)
  inline int labelIdAt(int v, int u) const
  {
    if (!semantic_is_color_)
    {
      // labels_ is normalized to CV_32SC1, so semantic masks such as
      // mono8, 16UC1, 16SC1, 32SC1, etc. are read correctly.
      return labels_.at<int32_t>(v, u);
    }

    const cv::Vec3b rgb = semantic_rgb_.at<cv::Vec3b>(v, u); // RGB
    const uint32_t key = packRGB(rgb[0], rgb[1], rgb[2]);
    auto it = id_by_rgb_.find(key);
    return (it != id_by_rgb_.end()) ? it->second : 255; // 255 => unknown
  }
  // Calibration
  cv::Mat cameraMatrix_, distCoeffs_, rvec_, tvec_, R_, R_inv_, K_;

  // Classes
  std::vector<SemanticClass> classes_;
  std::unordered_map<int, cv::Vec3b> bgr_by_id_;      // id -> BGR (OpenCV)
  std::unordered_map<uint32_t, int> id_by_rgb_;       // packed RGB -> id

	// Semantic image storage:
	// - If single-channel mask: labels_ is normalized to CV_32SC1 with class IDs
	//   Supports encodings such as mono8, 8UC1, 16UC1, 16SC1, 32SC1 and 32FC1.
	// - If rgb/bgr: semantic_rgb_ is CV_8UC3 in RGB, labels_ may be empty.
	cv::Mat labels_;        // CV_32SC1 (single-channel semantic ID mask)
	cv::Mat semantic_rgb_;  // CV_8UC3 RGB (only used when color semantic)
	bool semantic_is_color_ = false;

	// Actual semantic image size for the current frame.
	// The projection matrix and depth buffers are updated per frame using this size.
	int label_width_ = LABEL_W;
	int label_height_ = LABEL_H;

  // Filtering params
  double minRange_ = 0.5, maxRange_ = 30.0;
  double minAngFOV_ = -45.0, maxAngFOV_ = 45.0;

  static constexpr int CALIB_W = 1440;
  static constexpr int CALIB_H = 1080;
  static constexpr int LABEL_W = 1440;
  static constexpr int LABEL_H = 1080;

  // Projected data
  std::vector<cv::Point3f> pts3d_;
  std::vector<cv::Point2f> proj2d_;

  // z-buffer / winner index
  cv::Mat depth_buf_;
  cv::Mat idx_buf_;

  // Overlay cache
  cv::Mat overlay_cache_;
  bool overlay_updated_ = false;

  // Debug
  mutable bool warned_unknown_colors_ = false;
};
