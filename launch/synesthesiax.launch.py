"""
ROS2 launch file for synesthesiax
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('synesthesiax')
    # Path to the pinhole_model.yaml configuration
    config_file = os.path.join(pkg_share, 'config', 'pinhole_model.yaml')

    return LaunchDescription([
        Node(
            package='synesthesiax',
            executable='synesthesiax_node',
            name='synesthesiax_node',
            output='screen',
            parameters=[
                # Load calibration parameters from YAML
                config_file,
                # Node-specific parameters
                {
                    'max_range': 20.0,
                    'min_range': 0.5,
                    'max_ang_fov': 60.0,
                    'min_ang_fov': -60.0,
                    'cloud_topic': '/ona2/sensors/pandar_front/cloud',
                    'img_topic': '/semantic_inference/semantic/image_raw'
                }
            ]
        )
    ])
