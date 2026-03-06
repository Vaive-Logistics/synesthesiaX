cmake_minimum_required(VERSION 3.5)
project(synesthesiax)

# -----------------------------
# Default build type: Release
# -----------------------------
# (Only applies to single-config generators; colcon typically uses this path)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# -----------------------------
# C++ standard
# -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------
# Dependencies
# -----------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# -----------------------------
# Target
# -----------------------------

# synesthesia_node for back camera
add_executable(synesthesiax_node
  src/synesthesiax.cpp
  src/Projector.cpp
)

# synesthesia_node for front camera
add_executable(synesthesiax_front_camera_node
  src/synesthesiax_front_camera.cpp
  src/Projector_front_camera.cpp
)

# back camera
target_include_directories(synesthesiax_node PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
)

# front camera
target_include_directories(synesthesiax_front_camera_node PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
)

# back camera
ament_target_dependencies(synesthesiax_node
  rclcpp
  sensor_msgs
  cv_bridge
  pcl_conversions
  pcl_ros
  message_filters
)


# front camera
ament_target_dependencies(synesthesiax_front_camera_node
  rclcpp
  sensor_msgs
  cv_bridge
  pcl_conversions
  pcl_ros
  message_filters
)


# back camera
target_link_libraries(synesthesiax_node
  ${OpenCV_LIBRARIES}
  yaml-cpp::yaml-cpp
)

# front camera
target_link_libraries(synesthesiax_front_camera_node
  ${OpenCV_LIBRARIES}
  yaml-cpp::yaml-cpp
)



# -----------------------------
# Install
# -----------------------------
install(TARGETS synesthesiax_node

  synesthesiax_front_camera_node

  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()