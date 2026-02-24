#include "Projector.hpp"

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

#include <yaml-cpp/yaml.h>

#include <unordered_map>
#include <string>
#include <vector>
#include <chrono>

class SynesthesiaxNode : public rclcpp::Node
{
public:
    SynesthesiaxNode()
    : Node("synesthesiax")
    {
        // -------------------------
        // Topics
        // -------------------------
        this->declare_parameter<std::string>("cloud_topic", "/lidar/points");
        this->declare_parameter<std::string>("img_topic", "/camera/labels");
        this->declare_parameter<std::string>("raw_img_topic", "/camera/raw");

        const std::string cloud_topic   = this->get_parameter("cloud_topic").as_string();
        const std::string img_topic     = this->get_parameter("img_topic").as_string();
        const std::string raw_img_topic = this->get_parameter("raw_img_topic").as_string();

        // -------------------------
        // Semantic classes config
        // -------------------------
        this->declare_parameter<std::string>("classes_config", ""); // path to yaml
        this->declare_parameter<std::string>("class_cloud_topic_prefix", "/synesthesiax/class");

        const std::string classes_cfg_path = this->get_parameter("classes_config").as_string();
        const std::string class_topic_prefix = this->get_parameter("class_cloud_topic_prefix").as_string();

        classes_ = loadClassesFromYaml(classes_cfg_path);

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
            classes_ // <-- NEW
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
            "/synesthesiax/cloud_onto_img", 1);

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
        img_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, img_topic);

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), *pc_sub_, *img_sub_);
        sync_->registerCallback(std::bind(&SynesthesiaxNode::callback, this, std::placeholders::_1, std::placeholders::_2));

        raw_img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            raw_img_topic, rclcpp::SensorDataQoS(),
            std::bind(&SynesthesiaxNode::raw_img_callback, this, std::placeholders::_1)
        );
    }

private:
    static std::vector<Projector::SemanticClass> loadClassesFromYaml(const std::string& path)
    {
        std::vector<Projector::SemanticClass> out;

        if (path.empty())
        {
            // No file provided -> empty => fatal in ctor
            return out;
        }

        YAML::Node root;
        try {
            root = YAML::LoadFile(path);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to load YAML file: ") + path + " : " + e.what());
        }

        if (!root["classes"] || !root["classes"].IsSequence())
        {
            throw std::runtime_error("YAML must contain a top-level 'classes:' sequence");
        }

        for (const auto& n : root["classes"])
        {
            if (!n["id"] || !n["name"] || !n["color_rgb"])
                throw std::runtime_error("Each class must contain {id, name, color_rgb}");

            const int id = n["id"].as<int>();
            const std::string name = n["name"].as<std::string>();

            const auto rgb = n["color_rgb"];
            if (!rgb.IsSequence() || rgb.size() != 3)
                throw std::runtime_error("color_rgb must be a list of 3 integers: [R,G,B]");

            int r = rgb[0].as<int>();
            int g = rgb[1].as<int>();
            int b = rgb[2].as<int>();

            auto clamp255 = [](int x) { return std::max(0, std::min(255, x)); };
            r = clamp255(r); g = clamp255(g); b = clamp255(b);

            Projector::SemanticClass c;
            c.id = id;
            c.name = name;
            c.r = static_cast<uint8_t>(r);
            c.g = static_cast<uint8_t>(g);
            c.b = static_cast<uint8_t>(b);

            out.push_back(c);
        }

        return out;
    }

    void callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg,
                  const sensor_msgs::msg::Image::ConstSharedPtr image_msg)
    {
        auto start = std::chrono::high_resolution_clock::now();

        if (!projector_.project_cloud_onto_image(cloud_msg, image_msg))
        {
            RCLCPP_WARN(this->get_logger(), "Projector called without image or cloud (or conversion failed).");
            return;
        }

        // Get overlay + clouds
        const cv::Mat& overlay = projector_.getOverlay(last_raw_img_);

        pcl::PointCloud<pcl::PointXYZRGB> semanticCloud;
        std::unordered_map<int, pcl::PointCloud<pcl::PointXYZRGB>> clouds_by_class;
        projector_.getSemanticClouds(semanticCloud, clouds_by_class);

        // Publish overlay image
        auto img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", overlay).toImageMsg();
        img_msg->header = cloud_msg->header;
        pc_on_img_pub_->publish(*img_msg);

        // Publish full semantic cloud
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
        RCLCPP_INFO(this->get_logger(), "Processing time: %ld ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }

    void raw_img_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        RCLCPP_INFO_ONCE(this->get_logger(), "Receiving raw images...");
        last_raw_img_ = msg;
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
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> img_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // Raw image (not synced)
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr raw_img_sub_;
    sensor_msgs::msg::Image::ConstSharedPtr last_raw_img_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SynesthesiaxNode>());
    rclcpp::shutdown();
    return 0;
}