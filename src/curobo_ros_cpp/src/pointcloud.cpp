// SPDX-License-Identifier: Apache-2.0
// 解码和裁剪点云，过滤机器人本体并生成碰撞网格或距离场。
#include "curobo_ros_cpp/pointcloud.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <sensor_msgs/msg/point_field.hpp>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace curobo_ros_cpp {
namespace {
uint64_t unsigned_value(const uint8_t *data, size_t width, bool big_endian) {
  uint64_t value = 0;
  for (size_t i = 0; i < width; ++i)
    value |= static_cast<uint64_t>(data[big_endian ? width - 1 - i : i]) << (8 * i);
  return value;
}
float float_value(const uint8_t *data, bool big_endian) {
  const auto bits = static_cast<uint32_t>(unsigned_value(data, 4, big_endian));
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
struct GridHash {
  size_t operator()(const std::array<int64_t, 3> &key) const {
    size_t value = 0;
    for (const auto axis : key)
      value ^= std::hash<int64_t>{}(axis) + 0x9e3779b9 + (value << 6) + (value >> 2);
    return value;
  }
};
using Grid = std::unordered_set<std::array<int64_t, 3>, GridHash>;
Eigen::Vector3d voxel_origin(const std::vector<Point> &points, double pitch) {
  Eigen::Vector3d origin = points.front().cast<double>();
  for (const auto &point : points) origin = origin.cwiseMin(point.cast<double>());
  return origin.array() - pitch;
}
Grid occupied_voxels(const std::vector<Point> &points, const Eigen::Vector3d &origin, double pitch) {
  Grid voxels;
  voxels.reserve(points.size());
  for (const auto &point : points) {
    std::array<int64_t, 3> key;
    for (int axis = 0; axis < 3; ++axis)
      key[axis] = static_cast<int64_t>(std::floor((point[axis] - origin[axis]) / pitch));
    voxels.insert(key);
  }
  return voxels;
}
bool protected_label(const uint8_t *data, const sensor_msgs::msg::PointField *field,
                     bool big_endian) {
  if (!field) return false;
  const auto type = field->datatype;
  const size_t width = (type <= 2) ? 1 : (type <= 4) ? 2 : 4;
  const uint64_t value = unsigned_value(data + field->offset, width, big_endian);
  if (type == sensor_msgs::msg::PointField::INT8) return static_cast<int8_t>(value) > 0;
  if (type == sensor_msgs::msg::PointField::INT16) return static_cast<int16_t>(value) > 0;
  if (type == sensor_msgs::msg::PointField::INT32) return static_cast<int32_t>(value) > 0;
  return value > 0;
}
}  // namespace
// 裁剪并判断本体，再为每个体素选择需要保留的代表点。
PreparedCloud prepare_cloud(const sensor_msgs::msg::PointCloud2 &message, const Transform &to_base,
                            const PointCloudSettings &settings, const RobotModel &robot,
                            const JointMap &joints, const std::string &group) {
  const auto started = Clock::now();
  PreparedCloud output;
  output.raw_count = static_cast<size_t>(message.width) * message.height;
  std::map<std::string, const sensor_msgs::msg::PointField *> fields;
  for (const auto &field : message.fields) fields[field.name] = &field;
  for (const auto &name : {"x", "y", "z"}) {
    if (!fields.count(name) || fields.at(name)->datatype != sensor_msgs::msg::PointField::FLOAT32 ||
        static_cast<size_t>(fields.at(name)->offset) + 4 > message.point_step)
      throw std::runtime_error("PointCloud2 requires float32 xyz");
  }
  const auto label = fields.count("label") ? fields.at("label") : nullptr;
  if (label && (label->count != 1 || label->datatype < 1 || label->datatype > 6 ||
                static_cast<size_t>(label->offset) + (label->datatype <= 2   ? 1
                                                      : label->datatype <= 4 ? 2
                                                                             : 4) >
                    message.point_step))
    throw std::runtime_error("invalid point-cloud label field");
  if (message.row_step < static_cast<size_t>(message.width) * message.point_step ||
      message.data.size() < static_cast<size_t>(message.row_step) * message.height)
    throw std::runtime_error("truncated PointCloud2 data");
  std::vector<Point> sensor_points;
  std::vector<bool> labels;
  for (uint32_t row = 0; row < message.height; ++row)
    for (uint32_t column = 0; column < message.width; ++column) {
      const auto *data = message.data.data() + static_cast<size_t>(row) * message.row_step +
                         column * message.point_step;
      Point point(float_value(data + fields.at("x")->offset, message.is_bigendian),
                  float_value(data + fields.at("y")->offset, message.is_bigendian),
                  float_value(data + fields.at("z")->offset, message.is_bigendian));
      if (!point.allFinite()) continue;
      if (settings.filter &&
          (point.norm() <= settings.min_range || point.norm() >= settings.max_range))
        continue;
      sensor_points.push_back(point);
      labels.push_back(protected_label(data, label, message.is_bigendian));
    }
  auto inside = [&](const Point &point) {
    if ((point.cast<double>().array() < settings.lower.array()).any() ||
        (point.cast<double>().array() > settings.upper.array()).any())
      return false;
    for (const auto &box : settings.exclude_boxes)
      if (point.x() >= box[0] && point.y() >= box[1] && point.z() >= box[2] &&
          point.x() <= box[3] && point.y() <= box[4] && point.z() <= box[5])
        return false;
    return true;
  };
  // 同体素可能同时包含本体和环境，选代表点前先判断所有裁剪后的点。
  std::vector<Point> cropped;
  std::vector<bool> cropped_labels;
  for (size_t i = 0; i < sensor_points.size(); ++i) {
    const Point point = (to_base * sensor_points[i].cast<double>()).cast<float>();
    if (!inside(point)) continue;
    cropped.push_back(point);
    cropped_labels.push_back(labels[i]);
  }
  output.cropped_count = cropped.size();
  output.crop_ms = elapsed_ms(started);
  std::vector<bool> mesh_remove, sphere_remove;
  if (settings.self_filter_mode != "off") {
    if (!settings.self_filter_use_spheres || settings.self_filter_sphere_compare) {
      const auto mesh_started = Clock::now();
      mesh_remove = robot.self_mask(cropped, joints, group, settings.self_filter_padding,
                                    settings.self_filter_budget_ms, output.mesh_filter_timed_out);
      output.mesh_filter_ms = elapsed_ms(mesh_started);
    }
    if (settings.self_filter_use_spheres || settings.self_filter_sphere_compare) {
      const auto sphere_started = Clock::now();
      sphere_remove = robot.self_sphere_mask(cropped, joints, group);
      output.sphere_filter_ms = elapsed_ms(sphere_started);
    }
    if (settings.self_filter_sphere_compare) {
      for (size_t i = 0; i < cropped.size(); ++i) {
        if (cropped_labels[i]) continue;
        if (mesh_remove[i] && sphere_remove[i]) ++output.both_removed;
        else if (mesh_remove[i]) output.mesh_only.push_back(cropped[i]);
        else if (sphere_remove[i]) output.sphere_only.push_back(cropped[i]);
      }
    }
  }
  output.self_filter_ms =
      settings.self_filter_use_spheres ? output.sphere_filter_ms : output.mesh_filter_ms;
  const auto &remove = settings.self_filter_use_spheres ? sphere_remove : mesh_remove;
  const auto is_robot_point = [&](size_t index) {
    return !remove.empty() && remove[index];
  };
  const auto voxel_started = Clock::now();
  struct Slot {
    int plain = -1;
    int marked = -1;
  };
  std::unordered_map<std::array<int64_t, 3>, Slot, GridHash> occupied;
  std::vector<size_t> selected;
  for (size_t i = 0; i < cropped.size(); ++i) {
    std::array<int64_t, 3> key;
    for (int axis = 0; axis < 3; ++axis)
      key[axis] = static_cast<int64_t>(std::floor(cropped[i][axis] / settings.pitch));
    auto &slot = occupied[key];
    int &chosen = cropped_labels[i] ? slot.marked : slot.plain;
    if (chosen < 0) {
      chosen = static_cast<int>(selected.size());
      selected.push_back(i);
    } else if (is_robot_point(selected[chosen]) && !is_robot_point(i)) {
      // 有任意环境点就保留该体素。超时未检查的点也按环境保留。
      selected[chosen] = i;
    }
  }
  if (selected.size() > settings.max_points) {
    std::vector<size_t> limited;
    limited.reserve(settings.max_points);
    for (size_t i = 0; i < settings.max_points; ++i) {
      const size_t index = settings.max_points == 1
                               ? 0
                               : i * (selected.size() - 1) / (settings.max_points - 1);
      limited.push_back(selected[index]);
    }
    selected = std::move(limited);
  }
  for (const auto index : selected) {
    output.input.push_back(cropped[index]);
    if (is_robot_point(index) && !cropped_labels[index])
      output.removed.push_back(cropped[index]);
    else
      output.environment.push_back(cropped[index]);
  }
  // 超时只是没滤完。已经判定的本体点删掉，没来得及看的点留下来，不再整帧退回未过滤点云。
  output.points = settings.self_filter_mode == "observe" ? output.input : output.environment;
  output.voxel_ms = elapsed_ms(voxel_started);
  return output;
}

// 每个保留点生成一个立方体，拼成环境网格。
curobo::robot::Mesh voxel_mesh(const std::vector<Point> &points, double pitch) {
  curobo::robot::Mesh mesh;
  mesh.name = "environment";
  if (points.empty()) return mesh;
  const auto origin = voxel_origin(points, pitch);
  const auto voxels = occupied_voxels(points, origin, pitch);
  // Faces are emitted only at occupied/empty interfaces, matching Mesh.from_pointcloud.
  const std::array<std::array<int, 3>, 8> corners{
      {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}};
  const std::array<std::array<int, 4>, 6> faces{
      {{0, 3, 7, 4}, {1, 5, 6, 2}, {0, 4, 5, 1}, {3, 2, 6, 7}, {0, 1, 2, 3}, {4, 7, 6, 5}}};
  const std::array<std::array<int, 3>, 6> neighbors{
      {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}}};
  std::unordered_map<std::array<int64_t, 3>, int32_t, GridHash> vertex_map;
  std::vector<std::array<int64_t, 3>> ordered(voxels.begin(), voxels.end());
  std::sort(ordered.begin(), ordered.end());
  for (const auto &voxel : ordered)
    for (size_t face = 0; face < 6; ++face) {
      auto neighbor = voxel;
      for (int axis = 0; axis < 3; ++axis) neighbor[axis] += neighbors[face][axis];
      if (voxels.count(neighbor)) continue;
      std::array<int32_t, 4> indices;
      for (int corner = 0; corner < 4; ++corner) {
        auto key = voxel;
        for (int axis = 0; axis < 3; ++axis) key[axis] += corners[faces[face][corner]][axis];
        const auto found = vertex_map.emplace(key, static_cast<int32_t>(mesh.vertices.size()));
        indices[corner] = found.first->second;
        if (found.second)
          mesh.vertices.push_back({static_cast<float>(origin.x() + key[0] * pitch),
                                   static_cast<float>(origin.y() + key[1] * pitch),
                                   static_cast<float>(origin.z() + key[2] * pitch)});
      }
      mesh.triangles.push_back({indices[0], indices[1], indices[2]});
      mesh.triangles.push_back({indices[0], indices[2], indices[3]});
    }
  return mesh;
}

std::vector<Eigen::Vector3d> voxel_centers(const std::vector<Point> &points, double pitch) {
  std::vector<Eigen::Vector3d> centers;
  if (points.empty()) return centers;
  const auto origin = voxel_origin(points, pitch);
  const auto voxels = occupied_voxels(points, origin, pitch);
  centers.reserve(voxels.size());
  for (const auto &key : voxels)
    centers.emplace_back(origin.x() + (static_cast<double>(key[0]) + .5) * pitch,
                         origin.y() + (static_cast<double>(key[1]) + .5) * pitch,
                         origin.z() + (static_cast<double>(key[2]) + .5) * pitch);
  return centers;
}

bool same_voxel_occupancy(const std::vector<Point> &first, const std::vector<Point> &second,
                          double pitch) {
  if (first.empty() || second.empty()) return first.empty() && second.empty();
  const auto origin = voxel_origin(first, pitch);
  return origin == voxel_origin(second, pitch) &&
      occupied_voxels(first, origin, pitch) == occupied_voxels(second, origin, pitch);
}

// 去掉工件包围盒里的点，避免把手上或已放下的物体再当成障碍。
std::vector<Point> exclude_box(const std::vector<Point> &points, const Transform &at,
                               const Eigen::Vector3d &size) {
  std::vector<Point> result;
  result.reserve(points.size());
  const auto inverse = at.inverse();
  for (const auto &point : points)
    if (((inverse * point.cast<double>()).array().abs() > size.array() * .5).any())
      result.push_back(point);
  return result;
}

// 体素模式使用的距离场。网格模式不走这里。
std::vector<float> point_esdf(const std::vector<Point> &points, const Eigen::Vector3d &center,
                              const Eigen::Vector3d &dimensions, double pitch) {
  const Eigen::Array3d extent = (dimensions.array() / pitch - 1e-9).ceil();
  if (!center.allFinite() || !std::isfinite(pitch) || pitch <= 0. ||
      !extent.allFinite() || (extent <= 0.).any())
    throw std::invalid_argument("ESDF dimensions must be finite and positive");
  if (extent.prod() > 4000000) throw std::runtime_error("invalid ESDF capacity");
  const Eigen::Array3i shape = extent.cast<int>();
  const size_t count = static_cast<size_t>(shape.x()) * shape.y() * shape.z();
  const Eigen::Vector3d lower = center - shape.cast<double>().matrix() * pitch * .5;
  std::vector<float> distance(count, 1e10f);
  std::vector<bool> inside(count, false);
  const auto offset = [&](int x, int y, int z) {
    return (static_cast<size_t>(x) * shape.y() + y) * shape.z() + z;
  };
  for (const auto &point : points) {
    const Eigen::Array3d cell = ((point.cast<double>() - lower).array() / pitch).floor();
    if ((cell >= 0.).all() && (cell < extent).all()) {
      const Eigen::Array3i index = cell.cast<int>();
      distance[offset(index.x(), index.y(), index.z())] = 0.;
      inside[offset(index.x(), index.y(), index.z())] = true;
    }
  }
  if (std::none_of(inside.begin(), inside.end(), [](bool value) { return value; }))
    return std::vector<float>(count, 5.F);
  if (std::all_of(inside.begin(), inside.end(), [](bool value) { return value; })) {
    for (int x = 0; x < shape.x(); ++x)
      for (int y = 0; y < shape.y(); ++y)
        for (int z = 0; z < shape.z(); ++z) {
          const float steps = std::min({x + 1, shape.x() - x, y + 1, shape.y() - y, z + 1,
                                        shape.z() - z});
          distance[offset(x, y, z)] = -steps * static_cast<float>(pitch);
        }
    return distance;
  }
  auto edt = [&](std::vector<float> distance) {
    // Exact separable squared Euclidean distance transform.
    for (int axis = 0; axis < 3; ++axis) {
      const int a = (axis + 1) % 3, b = (axis + 2) % 3, n = shape[axis];
      std::vector<float> source(n), transformed(n), boundaries(n + 1);
      std::vector<int> sites(n);
      for (int ia = 0; ia < shape[a]; ++ia)
        for (int ib = 0; ib < shape[b]; ++ib) {
          Eigen::Array3i index;
          index[a] = ia;
          index[b] = ib;
          for (int q = 0; q < n; ++q) {
            index[axis] = q;
            source[q] = distance[offset(index.x(), index.y(), index.z())];
          }
          int k = 0;
          sites[0] = 0;
          boundaries[0] = -std::numeric_limits<float>::infinity();
          boundaries[1] = std::numeric_limits<float>::infinity();
          for (int q = 1; q < n; ++q) {
            float intersection;
            do {
              const int site = sites[k];
              intersection =
                  ((source[q] + q * q) - (source[site] + site * site)) / (2.f * (q - site));
              if (intersection <= boundaries[k])
                --k;
              else
                break;
            } while (k >= 0);
            ++k;
            sites[k] = q;
            boundaries[k] = intersection;
            boundaries[k + 1] = std::numeric_limits<float>::infinity();
          }
          k = 0;
          for (int q = 0; q < n; ++q) {
            while (boundaries[k + 1] < q) ++k;
            const auto delta = q - sites[k];
            transformed[q] = delta * delta + source[sites[k]];
          }
          for (int q = 0; q < n; ++q) {
            index[axis] = q;
            distance[offset(index.x(), index.y(), index.z())] = transformed[q];
          }
        }
    }
    return distance;
  };
  auto outside = edt(distance);
  for (size_t i = 0; i < count; ++i) distance[i] = inside[i] ? 1e10F : 0.F;
  auto inward = edt(distance);
  for (size_t i = 0; i < count; ++i)
    distance[i] = static_cast<float>((std::sqrt(outside[i]) - std::sqrt(inward[i])) * pitch);
  return distance;
}
}  // namespace curobo_ros_cpp
