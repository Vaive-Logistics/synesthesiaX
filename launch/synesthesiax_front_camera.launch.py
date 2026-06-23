"""
Launch only the front LiDAR/camera synesthesiax_node instance.
The executable is the same generic node used by the back configuration.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('synesthesiax')

    calib_file = os.path.join(pkg_share, 'config', 'pinhole_model.yaml')
    classes_file = os.path.join(pkg_share, 'config', 'classes.yaml')
    debug_mode = LaunchConfiguration('debug_mode')

    return LaunchDescription([
        DeclareLaunchArgument(
            'debug_mode',
            default_value='true',
            description='Enable raw-image debug overlay subscription and publisher',
        ),
        Node(
            package='synesthesiax',
            executable='synesthesiax_node',
            name='synesthesiax_front',
            output='screen',
            parameters=[
                calib_file,
                {
                    'cloud_topic': '/ona2/sensors/pandar_front/cloud',
                    'labels_img_topic': '/semantic_inference_front/semantic/image_raw/compressed',
                    'labels_transport': 'compressed',
                    'raw_img_topic': '/ona2/sensors/flir_camera_front/image_raw',
                    'debug_mode': ParameterValue(debug_mode, value_type=bool),
                    'sync_queue_size': 10,

                    'semantic_cloud_topic': '/synesthesiax/frontside_semantic_cloud',
                    'overlay_topic': '/synesthesiax/frontside_cloud_onto_img',
                    'class_cloud_topic_prefix': '/synesthesiax/front/class',

                    'max_range': 20.0,
                    'min_range': 1.0,
                    'max_ang_fov': 60.0,
                    'min_ang_fov': -60.0,
                    'enable_range_filter': True,
                    'enable_fov_filter': True,
                    'require_positive_x': True,

                    'classes_config': classes_file,
                }
            ],
        ),
    ])
