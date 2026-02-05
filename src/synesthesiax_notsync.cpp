#include "Projector.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class SynesthesiaxNotSyncedNode : public rclcpp::Node
{
public:
    SynesthesiaxNotSyncedNode()
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

        // Set up subscribers
        pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                    cloud_topic,
                    10,
                    std::bind(&SynesthesiaxNotSyncedNode::cloud_callback, this, std::placeholders::_1)
                );
        img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                    img_topic,
                    1,
                    std::bind(&SynesthesiaxNotSyncedNode::img_callback, this, std::placeholders::_1)
                );
        raw_img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
		     raw_img_topic,
		     1,
		     std::bind(&SynesthesiaxNotSyncedNode::raw_img_callback, this, std::placeholders::_1)
		 );
    }

private:
    void cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg)
    {
        auto start = std::chrono::high_resolution_clock::now();

        if(!projector_.project_cloud_onto_image(cloud_msg, last_image_)){
          RCLCPP_INFO(this->get_logger(), "Projector called without image");
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

    void img_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        RCLCPP_INFO_ONCE(this->get_logger(), "Receiving images...");
        last_image_ = msg;
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

    // Projector logic
    Projector projector_;

    // Last image msg
    sensor_msgs::msg::Image::ConstSharedPtr last_image_, last_raw_img_;

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_, raw_img_sub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SynesthesiaxNotSyncedNode>());
    rclcpp::shutdown();
    return 0;
}
