"""
ROS2 launch file for synesthesiax
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('synesthesiax')

    # Calibration YAML 
    calib_file = os.path.join(pkg_share, 'config', 'pinhole_model_back_camera.yaml')

    # Classes config YAML
    classes_file = os.path.join(pkg_share, 'config', 'classes.yaml')

    return LaunchDescription([
        Node(
            package='synesthesiax',
            executable='synesthesiax_node',
            name='synesthesiax_node',
            output='screen',
            parameters=[
                calib_file,
                {
                    # --- node topics ---
                    'cloud_topic': '/ona2/sensors/pandar_back/cloud',
                    'labels_img_topic': '/semantic_inference_front/semantic_color/image_raw',
                    'raw_img_topic': '/ona2/sensors/flir_camera_back/image_raw',   

                    # --- projector params ---
                    'max_range': 20.0,
                    'min_range': 1.0,
                    'max_ang_fov':180.0,
                    'min_ang_fov': -180.0,

                    'classes_config': classes_file,
                    'class_cloud_topic_prefix': '/synesthesiax/class',
                }
            ]
        )
    ])