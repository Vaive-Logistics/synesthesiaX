#include "Projector.hpp"
#include "Utils.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/header.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

class SynesthesiaxNode : public rclcpp::Node
{
public:
    SynesthesiaxNode()
    : Node("synesthesiax")
    {
        // -------------------------
        // Topics and input transport
        // -------------------------
        this->declare_parameter<std::string>("cloud_topic", "/lidar/points");
        this->declare_parameter<std::string>("labels_img_topic", "/camera/labels");
        this->declare_parameter<std::string>("labels_transport", "raw"); // "raw" or "compressed"
        this->declare_parameter<std::string>("raw_img_topic", "/camera/raw");
        this->declare_parameter<bool>("debug_mode", false);
        this->declare_parameter<int>("sync_queue_size", 10);
        this->declare_parameter<int>("raw_buffer_size", 100);

        cloud_topic_ = this->get_parameter("cloud_topic").as_string();
        labels_img_topic_ = this->get_parameter("labels_img_topic").as_string();
        labels_transport_ = this->get_parameter("labels_transport").as_string();
        raw_img_topic_ = this->get_parameter("raw_img_topic").as_string();
        debug_mode_ = this->get_parameter("debug_mode").as_bool();
        sync_queue_size_ = this->get_parameter("sync_queue_size").as_int();
        raw_buffer_size_ = static_cast<size_t>(this->get_parameter("raw_buffer_size").as_int());

        if (sync_queue_size_ <= 0)
        {
            RCLCPP_WARN(this->get_logger(), "sync_queue_size <= 0. Forcing sync_queue_size=10.");
            sync_queue_size_ = 10;
        }

        if (labels_transport_ != "raw" && labels_transport_ != "compressed")
        {
            RCLCPP_FATAL(this->get_logger(),
                         "Invalid labels_transport='%s'. Expected 'raw' or 'compressed'.",
                         labels_transport_.c_str());
            throw std::runtime_error("Invalid labels_transport");
        }

        // -------------------------
        // Semantic classes config
        // -------------------------
        this->declare_parameter<std::string>("classes_config", "");
        this->declare_parameter<std::string>("class_cloud_topic_prefix", "/synesthesiax/class");

        const std::string classes_cfg_path = this->get_parameter("classes_config").as_string();
        const std::string class_topic_prefix = this->get_parameter("class_cloud_topic_prefix").as_string();

        classes_ = synesthesiax::loadClassesFromYaml(classes_cfg_path);

        if (classes_.empty())
        {
            RCLCPP_FATAL(this->get_logger(),
                         "No classes loaded. Please set parameter 'classes_config' to a valid YAML file.");
            throw std::runtime_error("No classes loaded");
        }

        // -------------------------
        // Projector params
        // -------------------------
        this->declare_parameter("min_range", 0.5);
        this->declare_parameter("max_range", 30.0);
        this->declare_parameter("min_ang_fov", -45.0);
        this->declare_parameter("max_ang_fov", 45.0);
        this->declare_parameter("enable_range_filter", true);
        this->declare_parameter("enable_fov_filter", true);
        this->declare_parameter("require_positive_x", true);
        this->declare_parameter("camera_matrix", std::vector<double>());
        this->declare_parameter("d", std::vector<double>());
        this->declare_parameter("rlc", std::vector<double>());
        this->declare_parameter("tlc", std::vector<double>());

        auto cam = this->get_parameter("camera_matrix").as_double_array();
        auto d   = this->get_parameter("d").as_double_array();
        auto rlc = this->get_parameter("rlc").as_double_array();
        auto tlc = this->get_parameter("tlc").as_double_array();

        bool ok = projector_.init(
            this->get_parameter("min_range").as_double(),
            this->get_parameter("max_range").as_double(),
            this->get_parameter("min_ang_fov").as_double(),
            this->get_parameter("max_ang_fov").as_double(),
            this->get_parameter("enable_range_filter").as_bool(),
            this->get_parameter("enable_fov_filter").as_bool(),
            this->get_parameter("require_positive_x").as_bool(),
            std::vector<double>(cam.begin(), cam.end()),
            std::vector<double>(d.begin(), d.end()),
            std::vector<double>(rlc.begin(), rlc.end()),
            std::vector<double>(tlc.begin(), tlc.end()),
            classes_
        );

        if (!ok)
        {
            RCLCPP_FATAL(this->get_logger(), "Projector init failed. Check calibration and class config.");
            throw std::runtime_error("Projector init failed");
        }

        // -------------------------
        // Publishers
        // -------------------------
        this->declare_parameter<std::string>("semantic_cloud_topic", "/synesthesiax/semantic_cloud");
        this->declare_parameter<std::string>("overlay_topic", "/synesthesiax/cloud_onto_img");

        const std::string semantic_cloud_topic = this->get_parameter("semantic_cloud_topic").as_string();
        const std::string overlay_topic = this->get_parameter("overlay_topic").as_string();

        auto sensor_qos = rclcpp::SensorDataQoS();
        sensor_qos.keep_last(1);
        sensor_qos.best_effort();

        if (debug_mode_)
        {
            pc_on_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(overlay_topic, sensor_qos);
        }

        pc_color_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(semantic_cloud_topic, sensor_qos);

        for (const auto& c : classes_)
        {
            const std::string topic = class_topic_prefix + "/" + c.name;
            class_cloud_pubs_[c.id] = this->create_publisher<sensor_msgs::msg::PointCloud2>(topic, sensor_qos);

            RCLCPP_INFO(this->get_logger(),
                        "Class publisher: id=%d name=%s topic=%s color_rgb=[%d,%d,%d]",
                        c.id, c.name.c_str(), topic.c_str(), c.r, c.g, c.b);
        }

        // -------------------------
        // Subscriptions + sync
        // -------------------------
        const auto sensor_qos_profile = sensor_qos.get_rmw_qos_profile();

        if (labels_transport_ == "compressed")
        {
            pc_sub_compressed_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(
                this, cloud_topic_, sensor_qos_profile);

            lab_sub_compressed_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::CompressedImage>>(
                this, labels_img_topic_, sensor_qos_profile);

            sync_compressed_ = std::make_shared<message_filters::Synchronizer<CompressedSyncPolicy>>(
                CompressedSyncPolicy(static_cast<uint32_t>(sync_queue_size_)),
                *pc_sub_compressed_, *lab_sub_compressed_);

            sync_compressed_->registerCallback(
                std::bind(&SynesthesiaxNode::compressedCallback,
                          this, std::placeholders::_1, std::placeholders::_2));
        }
        else
        {
            pc_sub_raw_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(
                this, cloud_topic_, sensor_qos_profile);

            lab_sub_raw_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
                this, labels_img_topic_, sensor_qos_profile);

            sync_raw_ = std::make_shared<message_filters::Synchronizer<RawSyncPolicy>>(
                RawSyncPolicy(static_cast<uint32_t>(sync_queue_size_)),
                *pc_sub_raw_, *lab_sub_raw_);

            sync_raw_->registerCallback(
                std::bind(&SynesthesiaxNode::rawCallback,
                          this, std::placeholders::_1, std::placeholders::_2));
        }

        if (debug_mode_)
        {
            raw_img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                raw_img_topic_, sensor_qos,
                std::bind(&SynesthesiaxNode::rawImgCallback, this, std::placeholders::_1));
        }

        RCLCPP_INFO(this->get_logger(),
                    "Synesthesiax node started. cloud='%s', labels='%s', labels_transport='%s'",
                    cloud_topic_.c_str(), labels_img_topic_.c_str(), labels_transport_.c_str());
    }

private:
    using RawSyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::PointCloud2,
        sensor_msgs::msg::Image>;

    using CompressedSyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::PointCloud2,
        sensor_msgs::msg::CompressedImage>;

    static std::string encodingFromCvMat(const cv::Mat& img)
    {
        if (img.channels() == 1)
        {
            switch (img.depth())
            {
                case CV_8U:  return "mono8";
                case CV_16U: return "16UC1";
                case CV_16S: return "16SC1";
                case CV_32S: return "32SC1";
                case CV_32F: return "32FC1";
                default:     return "mono8";
            }
        }

        if (img.channels() == 3 && img.depth() == CV_8U)
            return "bgr8";   // OpenCV imdecode returns BGR.

        if (img.channels() == 4 && img.depth() == CV_8U)
            return "bgra8";  // OpenCV imdecode returns BGRA.

        return "passthrough";
    }

    sensor_msgs::msg::Image::SharedPtr compressedToImageMsg(
        const sensor_msgs::msg::CompressedImage::ConstSharedPtr& labels_msg) const
    {
        const cv::Mat labels_mat = cv::imdecode(labels_msg->data, cv::IMREAD_UNCHANGED);
        if (labels_mat.empty())
        {
            RCLCPP_WARN(this->get_logger(), "Failed to decode compressed semantic image.");
            return nullptr;
        }

        const std::string encoding = encodingFromCvMat(labels_mat);
        return cv_bridge::CvImage(labels_msg->header, encoding, labels_mat).toImageMsg();
    }

    void compressedCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg,
                            const sensor_msgs::msg::CompressedImage::ConstSharedPtr labels_msg)
    {
        auto labels_img_msg = compressedToImageMsg(labels_msg);
        if (!labels_img_msg)
            return;

        processFrame(cloud_msg, labels_img_msg);
    }

    void rawCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg,
                     const sensor_msgs::msg::Image::ConstSharedPtr labels_msg)
    {
        processFrame(cloud_msg, labels_msg);
    }

    void processFrame(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg,
                      const sensor_msgs::msg::Image::ConstSharedPtr& labels_msg)
    {
        const auto start = std::chrono::high_resolution_clock::now();

        if (!projector_.project_cloud_onto_image(cloud_msg, labels_msg))
        {
            RCLCPP_WARN(this->get_logger(),
                        "Projector called without image/cloud or conversion failed.");
            return;
        }

        if (debug_mode_)
        {
            const rclcpp::Time t_lab(labels_msg->header.stamp);
            auto raw_opt = synesthesiax::getNearestRawImg(raw_img_buffer_, raw_mtx_, t_lab);
            if (raw_opt.has_value())
            {
                const auto& raw_msg = raw_opt.value();
                const cv::Mat& overlay = projector_.getOverlay(raw_msg);

                auto img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", overlay).toImageMsg();
                img_msg->header = cloud_msg->header;
                pc_on_img_pub_->publish(*img_msg);
            }
        }

        pcl::PointCloud<pcl::PointXYZRGB> semanticCloud;
        std::unordered_map<int, pcl::PointCloud<pcl::PointXYZRGB>> clouds_by_class;
        projector_.getSemanticClouds(semanticCloud, clouds_by_class);

        sensor_msgs::msg::PointCloud2 pc_color_msg;
        pcl::toROSMsg(semanticCloud, pc_color_msg);
        pc_color_msg.header = cloud_msg->header;
        pc_color_pub_->publish(pc_color_msg);

        for (auto& kv : clouds_by_class)
        {
            const int class_id = kv.first;
            auto it_pub = class_cloud_pubs_.find(class_id);
            if (it_pub == class_cloud_pubs_.end())
                continue;

            sensor_msgs::msg::PointCloud2 msg;
            pcl::toROSMsg(kv.second, msg);
            msg.header = cloud_msg->header;
            it_pub->second->publish(msg);
        }

        const auto end = std::chrono::high_resolution_clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        RCLCPP_DEBUG(this->get_logger(), "Processing time %ld ms", ms);
    }

    void rawImgCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        {
            std::lock_guard<std::mutex> lk(raw_mtx_);
            raw_img_buffer_.push_back(msg);

            while (raw_img_buffer_.size() > raw_buffer_size_)
                raw_img_buffer_.pop_front();
        }

        RCLCPP_INFO_ONCE(this->get_logger(), "Receiving raw images...");
    }

    // Parameters
    std::string cloud_topic_;
    std::string labels_img_topic_;
    std::string labels_transport_;
    std::string raw_img_topic_;
    bool debug_mode_ = false;
    int sync_queue_size_ = 10;

    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pc_on_img_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_color_pub_;
    std::unordered_map<int, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> class_cloud_pubs_;

    // Projector
    Projector projector_;
    std::vector<Projector::SemanticClass> classes_;

    // Raw semantic image sync
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> pc_sub_raw_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> lab_sub_raw_;
    std::shared_ptr<message_filters::Synchronizer<RawSyncPolicy>> sync_raw_;

    // Compressed semantic image sync
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> pc_sub_compressed_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CompressedImage>> lab_sub_compressed_;
    std::shared_ptr<message_filters::Synchronizer<CompressedSyncPolicy>> sync_compressed_;

    // Raw image buffer, only used for debug overlay
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr raw_img_sub_;
    std::deque<sensor_msgs::msg::Image::ConstSharedPtr> raw_img_buffer_;
    size_t raw_buffer_size_ = 100;
    std::mutex raw_mtx_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SynesthesiaxNode>());
    rclcpp::shutdown();
    return 0;
}
