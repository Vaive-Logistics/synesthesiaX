cmake_minimum_required(VERSION 3.5)
project(synesthesiax)

# Compile as C++14 (you can upgrade to C++17/20 if needed)
add_compile_options(-std=c++17)

# Find required packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

# Declare the executable
add_executable(synesthesiax_node
  src/synesthesiax.cpp
  src/Projector.cpp
)

# Link libraries
ament_target_dependencies(synesthesiax_node
  rclcpp
  sensor_msgs
  cv_bridge
  pcl_conversions
  pcl_ros
  message_filters
)

target_link_libraries(synesthesiax_node
  ${OpenCV_LIBRARIES}
)

# Install the binary
install(TARGETS synesthesiax_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch and config files
install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Export package
ament_package()
