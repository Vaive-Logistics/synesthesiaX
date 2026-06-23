"""
Launch the same synesthesiax_node executable twice:
- one instance for the front LiDAR/camera pair
- one instance for the back LiDAR/camera pair
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

    front_calib_file = os.path.join(pkg_share, 'config', 'pinhole_model.yaml')
    back_calib_file = os.path.join(pkg_share, 'config', 'pinhole_model_back_camera.yaml')
    classes_file = os.path.join(pkg_share, 'config', 'classes.yaml')

    front_debug_mode = LaunchConfiguration('front_debug_mode')
    back_debug_mode = LaunchConfiguration('back_debug_mode')

    return LaunchDescription([
        DeclareLaunchArgument(
            'front_debug_mode',
            default_value='true',
            description='Enable front raw-image debug overlay subscription and publisher',
        ),
        DeclareLaunchArgument(
            'back_debug_mode',
            default_value='false',
            description='Enable back raw-image debug overlay subscription and publisher',
        ),

        Node(
            package='synesthesiax',
            executable='synesthesiax_node',
            name='synesthesiax_front',
            output='screen',
            parameters=[
                front_calib_file,
                {
                    # --- input topics ---
                    'cloud_topic': '/ona2/sensors/pandar_front/cloud',
                    'labels_img_topic': '/semantic_inference_front/semantic/image_raw/compressed',
                    'labels_transport': 'compressed',
                    'raw_img_topic': '/ona2/sensors/flir_camera_front/image_raw',
                    'debug_mode': ParameterValue(front_debug_mode, value_type=bool),
                    'sync_queue_size': 10,

                    # --- output topics ---
                    'semantic_cloud_topic': '/synesthesiax/frontside_semantic_cloud',
                    'overlay_topic': '/synesthesiax/frontside_cloud_onto_img',
                    'class_cloud_topic_prefix': '/synesthesiax/front/class',

                    # --- projector params ---
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

        Node(
            package='synesthesiax',
            executable='synesthesiax_node',
            name='synesthesiax_back',
            output='screen',
            parameters=[
                back_calib_file,
                {
                    # --- input topics ---
                    'cloud_topic': '/ona2/sensors/pandar_back/cloud',
                    'labels_img_topic': '/semantic_inference_back/semantic_color/image_raw',
                    'labels_transport': 'raw',
                    'raw_img_topic': '/ona2/sensors/flir_camera_back/image_raw',
                    'debug_mode': ParameterValue(back_debug_mode, value_type=bool),
                    'sync_queue_size': 2,

                    # --- output topics ---
                    'semantic_cloud_topic': '/synesthesiax/backside_semantic_cloud',
                    'overlay_topic': '/synesthesiax/backside_cloud_onto_img',
                    'class_cloud_topic_prefix': '/synesthesiax/back/class',

                    # --- projector params ---
                    # Back keeps the original no-filter behavior. Enable these once validated.
                    'max_range': 20.0,
                    'min_range': 1.0,
                    'max_ang_fov': 180.0,
                    'min_ang_fov': -180.0,
                    'enable_range_filter': False,
                    'enable_fov_filter': False,
                    'require_positive_x': False,

                    'classes_config': classes_file,
                }
            ],
        ),
    ])
