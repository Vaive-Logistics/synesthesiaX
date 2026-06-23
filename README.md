# synesthesiax

ROS 2 package that projects LiDAR point clouds onto semantic camera images and publishes colored semantic point clouds.

## Refactored node structure

The package now builds a single reusable executable:

```bash
synesthesiax_node
```

The same executable is launched twice with different parameters:

- `synesthesiax_front`
- `synesthesiax_back`

Use:

```bash
ros2 launch synesthesiax synesthesiax.launch.py
```

Single-side launch files are also available:

```bash
ros2 launch synesthesiax synesthesiax_front_camera.launch.py
ros2 launch synesthesiax synesthesiax_back_camera.launch.py
```

## Important parameters

- `labels_transport`: `raw` or `compressed`.
- `semantic_cloud_topic`: output topic for the full semantic cloud.
- `overlay_topic`: output topic for the projected debug image.
- `class_cloud_topic_prefix`: prefix for per-class semantic clouds.
- `enable_range_filter`: enable/disable range filtering.
- `enable_fov_filter`: enable/disable FOV filtering.
- `require_positive_x`: discard points with `x <= 0` when enabled.
- `sync_queue_size`: approximate-time synchronizer queue size.

## Coordinate convention

The old back-camera implementation manually projected points as:

```cpp
(-x, -y, z)
```

That convention is now absorbed in `config/pinhole_model_back_camera.yaml`:

```text
R_new = R_old * diag(-1, -1, 1)
```

Therefore, the C++ projector is now frame-agnostic and always uses the original cloud coordinates directly.
