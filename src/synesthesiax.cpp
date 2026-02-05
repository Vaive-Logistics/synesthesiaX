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

class SynesthesiaxNode : public rclcpp::Node
{
public:
    SynesthesiaxNode()
    : Node("synesthesiax")
    {
        // Declare and read parameters
        this->declare_parameter<std::string>("cloud_topic", "/lidar/points");
        this->declare_parameter<std::string>("img_topic", "/camera/labels");
        this->declare_parameter<std::string>("raw_img_topic", "/camera/raw");

        std::string cloud_topic = this->get_parameter("cloud_topic").as_string();
        std::string img_topic = this->get_parameter("img_topic").as_string();
        std::string raw_img_topic = this->get_parameter("raw_img_topic").as_string();
        
        // Initialize projector with node parameters
        this->declare_parameter("min_range", 0.5);
        this->declare_parameter("max_range", 30.0);
        this->declare_parameter("min_ang_fov", -45.0);
        this->declare_parameter("max_ang_fov", 45.0);
        this->declare_parameter("camera_matrix", std::vector<double>());
        this->declare_parameter("d", std::vector<double>());
        this->declare_parameter("rlc", std::vector<double>());
        this->declare_parameter("tlc", std::vector<double>());

        // Load parameters from ROS2
        auto cam  = this->get_parameter("camera_matrix").as_double_array();
        auto d    = this->get_parameter("d").as_double_array();
        auto rlc  = this->get_parameter("rlc").as_double_array();
        auto tlc  = this->get_parameter("tlc").as_double_array();

        bool ok = projector_.init(
            this->get_parameter("min_range").as_double(),
            this->get_parameter("max_range").as_double(),
            this->get_parameter("min_ang_fov").as_double(),
            this->get_parameter("max_ang_fov").as_double(),
            std::vector<double>(cam.begin(), cam.end()),
            std::vector<double>(d.begin(), d.end()),
            std::vector<double>(rlc.begin(), rlc.end()),
            std::vector<double>(tlc.begin(), tlc.end())
        );

        // Create publishers
        pc_on_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/synesthesiax/cloud_onto_img", 1);
        pc_color_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/synesthesiax/semantic_cloud", 1);
        pub_obstacles_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/synesthesiax/obstacles", 1);
        pub_traversable_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/synesthesiax/traversable", 1);
        pub_depth_ = this->create_publisher<sensor_msgs::msg::Image>("/synesthesiax/depth_map", 1);

        // Set up message filters and synchronization
        pc_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(this, cloud_topic);
        img_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, img_topic);

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), *pc_sub_, *img_sub_);
        sync_->registerCallback(std::bind(&SynesthesiaxNode::callback, this, std::placeholders::_1, std::placeholders::_2));

        raw_img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
		     raw_img_topic,
		     1,
		     std::bind(&SynesthesiaxNode::raw_img_callback, this, std::placeholders::_1)
		 );
    }

private:
    void callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg,
                  const sensor_msgs::msg::Image::ConstSharedPtr image_msg)
    {
        auto start = std::chrono::high_resolution_clock::now();

        if(!projector_.project_cloud_onto_image(cloud_msg, image_msg)){
          RCLCPP_INFO(this->get_logger(), "Projector called without image or cloud");
          return;
        }

        pcl::PointCloud<pcl::PointXYZRGB> semanticCloud, travCloud, obstacleCloud;
        projector_.getSemanticClouds(semanticCloud, travCloud, obstacleCloud);
        const cv::Mat& overlay = projector_.getOverlay(last_raw_img_);

        // Publish overlay image
        auto img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", overlay).toImageMsg();
        img_msg->header = cloud_msg->header;
        pc_on_img_pub_->publish(*img_msg);

        // Publish semantic clouds
        sensor_msgs::msg::PointCloud2 pc_color_msg, pc_obstacles_msg, pc_traversable_msg;
        pcl::toROSMsg(semanticCloud, pc_color_msg);
        pcl::toROSMsg(obstacleCloud, pc_obstacles_msg);
        pcl::toROSMsg(travCloud, pc_traversable_msg);

        pc_color_msg.header = pc_obstacles_msg.header = pc_traversable_msg.header = cloud_msg->header;

        pc_color_pub_->publish(pc_color_msg);
        pub_obstacles_->publish(pc_obstacles_msg);
        pub_traversable_->publish(pc_traversable_msg);

        auto end = std::chrono::high_resolution_clock::now();
        RCLCPP_INFO(this->get_logger(), "Processing time: %ld ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }


    void raw_img_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        RCLCPP_INFO_ONCE(this->get_logger(), "Receiving raw images...");
        last_raw_img_ = msg;
    }

    // ROS2 publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pc_on_img_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_color_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_obstacles_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_traversable_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_depth_;

    // Projector logic
    Projector projector_;

    // Message filters
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::PointCloud2, sensor_msgs::msg::Image>;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> pc_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> img_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
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
