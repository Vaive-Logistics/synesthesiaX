#pragma once

#include "Projector.hpp"

#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

namespace synesthesiax
{

inline std::vector<Projector::SemanticClass>
loadClassesFromYaml(const std::string& path)
{
    std::vector<Projector::SemanticClass> out;

    if (path.empty())
    {
        return out;
    }

    YAML::Node root;
    try
    {
        root = YAML::LoadFile(path);
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error(std::string("Failed to load YAML file: ") + path + " : " + e.what());
    }

    if (!root["classes"] || !root["classes"].IsSequence())
    {
        throw std::runtime_error("YAML must contain a top-level 'classes:' sequence");
    }

    auto clamp255 = [](int x) { return std::max(0, std::min(255, x)); };

    for (const auto& n : root["classes"])
    {
        if (!n["id"] || !n["name"] || !n["color_rgb"])
        {
            throw std::runtime_error("Each class must contain {id, name, color_rgb}");
        }

        const int id = n["id"].as<int>();
        const std::string name = n["name"].as<std::string>();

        const auto rgb = n["color_rgb"];
        if (!rgb.IsSequence() || rgb.size() != 3)
        {
            throw std::runtime_error("color_rgb must be a list of 3 integers: [R,G,B]");
        }

        int r = clamp255(rgb[0].as<int>());
        int g = clamp255(rgb[1].as<int>());
        int b = clamp255(rgb[2].as<int>());

        Projector::SemanticClass c;
        c.id = id;
        c.name = name;
        c.r = static_cast<uint8_t>(r);
        c.g = static_cast<uint8_t>(g);
        c.b = static_cast<uint8_t>(b);

        out.push_back(c);
    }

    return out;
}

inline std::optional<sensor_msgs::msg::Image::ConstSharedPtr>
getNearestRawImg(const std::deque<sensor_msgs::msg::Image::ConstSharedPtr>& raw_img_buffer,
                 std::mutex& raw_mtx,
                 const rclcpp::Time& t_ref,
                 int64_t max_dt_ns = 250LL * 1000LL * 1000LL)
{
    std::lock_guard<std::mutex> lk(raw_mtx);

    if (raw_img_buffer.empty())
    {
        std::cout << "[synesthesiax] raw_img_buffer is empty\n";
        return std::nullopt;
    }

    const rclcpp::Time t_front(raw_img_buffer.front()->header.stamp);
    const rclcpp::Time t_back (raw_img_buffer.back()->header.stamp);

    size_t best_i = 0;
    rclcpp::Time best_t(raw_img_buffer[0]->header.stamp);
    int64_t best_dt_ns =
        std::llabs((best_t - t_ref).nanoseconds());

    for (size_t i = 1; i < raw_img_buffer.size(); ++i)
    {
        const rclcpp::Time ti(raw_img_buffer[i]->header.stamp);
        const int64_t dt_ns = std::llabs((ti - t_ref).nanoseconds());

        if (dt_ns < best_dt_ns)
        {
            best_dt_ns = dt_ns;
            best_i = i;
            best_t = ti;
        }
    }

    const double best_dt_ms = static_cast<double>(best_dt_ns) * 1e-6;
    const double max_dt_ms  = static_cast<double>(max_dt_ns)  * 1e-6;

    return raw_img_buffer[best_i];
}

}  // namespace synesthesiax