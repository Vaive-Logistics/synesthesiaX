# SynesthesiaX

C++/OpenCV module for multimodal LiDAR-camera fusion, segmenting point clouds semantically based on camera labels using EfficientViT-SAM.
`‍Synesthesiax` projects LiDAR point clouds onto semantic camera images and publishes colored clouds + overlays.


## 📦 Important Dependencies

* ROS 2 Humble (rclcpp, sensor\_msgs, cv\_bridge)
* PCL (pcl\_ros, pcl\_conversions)
* OpenCV (>=4.0)
* message\_filters

---

## 🛠️ Installation

```bash
# 1. Clone into your ROS 2 workspace
cd ~/ros2_ws/src
git clone https://github.com/Vaive-Logistics/synesthesiaX

# 2. Build the package
cd ~/ros2_ws
colcon build --packages-select synesthesiax
```

---

## ▶️ Usage

```bash
# Source your workspace
cd ~/ros2_ws
source /install/setup.bash

# Launch the node with default parameters
ros2 launch synesthesiax synesthesiax.launch.py
```

📁 Calibration file is loaded from
`install/share/synesthesiax/config/pinhole_model.yaml`

---

👍 Enjoy! If you hit any issues, feel free to open an issue on the repo.
