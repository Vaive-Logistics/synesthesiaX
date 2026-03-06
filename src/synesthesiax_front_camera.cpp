#include "Projector.hpp"
#include "Utils.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class SynesthesiaxNode : public rclcpp::Node
{
public:
    SynesthesiaxNode()
    : Node("synesthesiax_front_camera")
    {
        // -------------------------
        // Topics
        // -------------------------
        this->declare_parameter<std::string>("cloud_topic", "/lidar/points");
        this->declare_parameter<std::string>("labels_img_topic", "/camera/labels");
        this->declare_parameter<std::string>("raw_img_topic", "/camera/raw");

        const std::string labels_img_topic = this->get_parameter("labels_img_topic").as_string();
        const std::string cloud_topic      = this->get_parameter("cloud_topic").as_string();
        const std::string raw_img_topic    = this->get_parameter("raw_img_topic").as_string();

        // -------------------------
        // Semantic classes config
        // -------------------------
        this->declare_parameter<std::string>("classes_config", ""); // path to yaml
        this->declare_parameter<std::string>("class_cloud_topic_prefix", "/synesthesiax/class");

        const std::string classes_cfg_path  = this->get_parameter("classes_config").as_string();
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
            std::vector<double>(cam.begin(), cam.end()),
            std::vector<double>(d.begin(), d.end()),
            std::vector<double>(rlc.begin(), rlc.end()),
            std::vector<double>(tlc.begin(), tlc.end()),
            classes_
        );

        if (!ok)
        {
            RCLCPP_FATAL(this->get_logger(), "Projector init failed (check calibration + class config).");
            throw std::runtime_error("Projector init failed");
        }

        // -------------------------
        // Publishers
        // -------------------------
        pc_on_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/synesthesiax/frontside_cloud_onto_img", 1);

        pc_color_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/synesthesiax/semantic_cloud", 1);

        // Dynamic publishers per class
        for (const auto& c : classes_)
        {
            const std::string topic = class_topic_prefix + "/" + c.name;
            class_cloud_pubs_[c.id] =
                this->create_publisher<sensor_msgs::msg::PointCloud2>(topic, 1);

            RCLCPP_INFO(this->get_logger(),
                        "Class publisher: id=%d name=%s topic=%s color_rgb=[%d,%d,%d]",
                        c.id, c.name.c_str(), topic.c_str(), c.r, c.g, c.b);
        }

        // -------------------------
        // Subscriptions + sync
        // -------------------------
        pc_sub_  = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(this, cloud_topic);
        lab_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, labels_img_topic);

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), *pc_sub_, *lab_sub_);
        sync_->registerCallback(
            std::bind(&SynesthesiaxNode::callback, this, std::placeholders::_1, std::placeholders::_2));

        raw_img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            raw_img_topic, 1,
            std::bind(&SynesthesiaxNode::raw_img_callback, this, std::placeholders::_1));
    }

private:
    void callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg,
                  const sensor_msgs::msg::Image::ConstSharedPtr labels_msg)
    {
        auto start = std::chrono::high_resolution_clock::now();

        if (!projector_.project_cloud_onto_image(cloud_msg, labels_msg))
        {
            RCLCPP_WARN(this->get_logger(),
                        "Projector called without image or cloud (or conversion failed).");
            return;
        }

        const rclcpp::Time t_lab(labels_msg->header.stamp);
        auto raw_opt = synesthesiax::getNearestRawImg(raw_img_buffer_, raw_mtx_, t_lab);
        if (raw_opt.has_value()) {

            const auto& raw_msg = raw_opt.value();
            const cv::Mat& overlay = projector_.getOverlay(raw_msg);
            
            auto img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", overlay).toImageMsg();
            img_msg->header = cloud_msg->header;
            pc_on_img_pub_->publish(*img_msg);
            
        }
        
        pcl::PointCloud<pcl::PointXYZRGB> semanticCloud;
        std::unordered_map<int, pcl::PointCloud<pcl::PointXYZRGB>> clouds_by_class;
        projector_.getSemanticClouds(semanticCloud, clouds_by_class);

        sensor_msgs::msg::PointCloud2 pc_color_msg;
        pcl::toROSMsg(semanticCloud, pc_color_msg);
        pc_color_msg.header = cloud_msg->header;
        pc_color_pub_->publish(pc_color_msg);

        // Publish per-class clouds
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

        auto end = std::chrono::high_resolution_clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        RCLCPP_INFO(this->get_logger(), "Processing time %ld ms", ms);
    }

    void raw_img_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        {
            std::lock_guard<std::mutex> lk(raw_mtx_);
            raw_img_buffer_.push_back(msg);

            while (raw_img_buffer_.size() > raw_buffer_size_)
                raw_img_buffer_.pop_front();
        }

        RCLCPP_INFO_ONCE(this->get_logger(), "Receiving raw images...");
    }

    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pc_on_img_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_color_pub_;
    std::unordered_map<int, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> class_cloud_pubs_;

    // Projector
    Projector projector_;
    std::vector<Projector::SemanticClass> classes_;

    // Sync
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::PointCloud2, sensor_msgs::msg::Image>;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> pc_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> lab_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // Raw buffer
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