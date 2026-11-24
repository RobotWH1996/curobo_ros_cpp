// SPDX-License-Identifier: Apache-2.0
// 构建碰撞场景，处理点云、静态障碍、放置物及携物状态。
#include "curobo_ros_cpp/detail/native_planner_group.hpp"
#include <algorithm>
#include <cmath>

namespace curobo_ros_cpp {
namespace {
Transform native_pose(const curobo::robot::Pose &pose) {
  Transform at = Transform::Identity();
  at.translation() = Eigen::Vector3d(pose.position[0], pose.position[1], pose.position[2]);
  at.linear() = Eigen::Quaterniond(pose.quaternion[0], pose.quaternion[1], pose.quaternion[2],
                                   pose.quaternion[3])
                    .normalized()
                    .toRotationMatrix();
  return at;
}
curobo::robot::Pose native_pose(const Transform &pose) {
  curobo::robot::Pose result;
  for (int i = 0; i < 3; ++i) result.position[i] = pose.translation()[i];
  Eigen::Quaterniond q(pose.linear());
  result.quaternion = {static_cast<float>(q.w()), static_cast<float>(q.x()),
                       static_cast<float>(q.y()), static_cast<float>(q.z())};
  return result;
}
curobo::robot::Mesh box_mesh(const curobo::robot::Cuboid &box) {
  curobo::robot::Mesh mesh;
  mesh.name = box.name;
  mesh.pose = box.pose;
  for (int x : {-1, 1})
    for (int y : {-1, 1})
      for (int z : {-1, 1})
        mesh.vertices.push_back({x * box.dimensions[0] * .5F, y * box.dimensions[1] * .5F,
                                 z * box.dimensions[2] * .5F});
  mesh.triangles = {{{0, 4, 6}}, {{0, 6, 2}}, {{1, 3, 7}}, {{1, 7, 5}}, {{0, 1, 5}}, {{0, 5, 4}},
                    {{2, 6, 7}}, {{2, 7, 3}}, {{0, 2, 3}}, {{0, 3, 1}}, {{4, 5, 7}}, {{4, 7, 6}}};
  return mesh;
}
}  // namespace
// 静态盒子和网格只记录下来，下次重建场景时一并放进 GPU。
void NativePlanner::load_static_scene(const std::string &file) {
  static_boxes_.clear();
  static_meshes_.clear();
  if (file.empty()) {
    rebuild_world();
    return;
  }
  const auto config = YAML::LoadFile(file);
  for (const auto &entry : config["cuboid"]) {
    curobo::robot::Cuboid box;
    box.name = entry.first.as<std::string>();
    const auto pose = entry.second["pose"].as<std::vector<float>>();
    if (pose.size() != 7) throw std::invalid_argument("static obstacle pose requires 7 values");
    for (int i = 0; i < 3; ++i) {
      box.dimensions[i] = entry.second["dims"][i].as<float>();
      box.pose.position[i] = pose[i];
    }
    for (int i = 0; i < 4; ++i) box.pose.quaternion[i] = pose[3 + i];
    static_boxes_.push_back(box);
  }
  for (const auto &entry : config["mesh"]) {
    curobo::robot::Mesh mesh;
    mesh.name = entry.first.as<std::string>();
    const auto value = entry.second;
    if (!value["vertices"] || !value["faces"])
      throw std::invalid_argument("static mesh requires offline scene export");
    for (const auto &row : value["vertices"])
      mesh.vertices.push_back(row.as<std::array<float, 3>>());
    for (const auto &row : value["faces"])
      mesh.triangles.push_back(row.as<std::array<int32_t, 3>>());
    const auto pose = value["pose"].as<std::vector<float>>();
    for (int i = 0; i < 3; ++i) mesh.pose.position[i] = pose.at(i);
    for (int i = 0; i < 4; ++i) mesh.pose.quaternion[i] = pose.at(i + 3);
    static_meshes_.push_back(std::move(mesh));
  }
  for (const auto &kind : {"sphere", "capsule", "cylinder", "voxel"})
    if (config[kind] && config[kind].size())
      throw std::invalid_argument("static primitive or voxel scene requires offline scene export");
  rebuild_world();
}
// 放下的物体留在场景里，直到本任务结束被恢复。
void NativePlanner::add_placed_object(const Transform &pose, const Eigen::Vector3d &size) {
  curobo::robot::Cuboid box;
  box.name = "placed_" + std::to_string(placed_boxes_.size());
  box.pose = native_pose(pose);
  for (int i = 0; i < 3; ++i) box.dimensions[i] = size[i];
  if (placed_boxes_.size() >= max_placed_objects || static_boxes_.size() + placed_boxes_.size() >= 32)
    throw std::runtime_error("placed object capacity exceeded");
  placed_boxes_.push_back(box);
  rebuild_world();
}
// 拷贝当前障碍和已放置物体，供任务结束后恢复。
WorldState NativePlanner::world_state() const {
  return {world_points_, world_pitch_,      world_mode_,
          world_center_, world_dimensions_, placed_boxes_};
}
// 抓放结束或失败后回到任务开始时的障碍。
void NativePlanner::restore_world(const WorldState &state) {
  placed_boxes_ = state.placed;
  world_points_ = state.points;
  world_pitch_ = state.pitch;
  world_mode_ = state.mode;
  world_center_ = state.center;
  world_dimensions_ = state.dimensions;
  rebuild_world();
}
// 去掉已放置物体并重建场景。
void NativePlanner::clear_placed_objects() {
  if (placed_boxes_.empty()) return;
  placed_boxes_.clear();
  rebuild_world();
}
// 清空点云障碍，静态场景仍然保留。
void NativePlanner::clear_world() {
  world_points_.clear();
  rebuild_world();
}
// 点云、静态物体、已放置物体和携物合成当前 GPU 场景。
void NativePlanner::rebuild_world() {
  ++world_generation_;
  scene_->clear();
  world_.reset();
  std::vector<Eigen::Vector3d> vertices;
  std::vector<fcl::Triangle> faces;
  auto add_geometry = [&](const curobo::robot::Mesh &mesh) {
    const auto transform = native_pose(mesh.pose);
    const int offset = vertices.size();
    for (const auto &p : mesh.vertices)
      vertices.push_back(transform * Eigen::Vector3d(p[0], p[1], p[2]));
    for (const auto &face : mesh.triangles)
      faces.emplace_back(face[0] + offset, face[1] + offset, face[2] + offset);
  };
  for (const auto &mesh : static_meshes_) {
    scene_->add_mesh(mesh);
    add_geometry(mesh);
  }
  for (const auto &box : static_boxes_) {
    scene_->add_cuboid(box);
    add_geometry(box_mesh(box));
  }
  for (const auto &box : placed_boxes_) {
    scene_->add_cuboid(box);
    add_geometry(box_mesh(box));
  }
  if (!world_points_.empty()) {
    curobo::robot::Mesh mesh;
    if (world_mode_ == "voxel") {
      const Eigen::Array3i shape =
          (world_dimensions_.array() / world_pitch_ - 1e-9).ceil().cast<int>();
      const Eigen::Vector3d lower =
          world_center_ - shape.cast<double>().matrix() * world_pitch_ * .5;
      std::vector<Point> occupied;
      for (const auto &point : world_points_) {
        const Eigen::Array3i index =
            ((point.cast<double>() - lower).array() / world_pitch_).floor().cast<int>();
        if ((index >= 0).all() && (index < shape).all())
          occupied.push_back(index.cast<float>().matrix());
      }
      mesh = voxel_mesh(occupied, 1.);
      for (auto &vertex : mesh.vertices)
        for (int axis = 0; axis < 3; ++axis)
          vertex[axis] = lower[axis] + vertex[axis] * world_pitch_;
    } else
      mesh = voxel_mesh(world_points_, world_pitch_);
    add_geometry(mesh);
    if (world_mode_ == "voxel") {
      curobo::robot::VoxelGrid grid;
      grid.name = "environment";
      for (int i = 0; i < 3; ++i) {
        grid.pose.position[i] = world_center_[i];
        grid.dimensions[i] = std::ceil(world_dimensions_[i] / world_pitch_ - 1e-9) * world_pitch_;
      }
      grid.voxel_size = world_pitch_;
      grid.esdf = point_esdf(world_points_, world_center_, world_dimensions_, world_pitch_);
      world_ = scene_->add_voxel_grid(grid);
    } else
      world_ = scene_->add_mesh(mesh);
  }
  if (faces.empty())
    robot_.clear_environment();
  else
    robot_.set_environment(std::move(vertices), std::move(faces));
}
// 换一帧点云。mesh 和 voxel 的差别在重建时处理。
void NativePlanner::set_world(const std::vector<Point> &points, double pitch,
                              const std::string &mode, const Eigen::Vector3d &center,
                              const Eigen::Vector3d &dimensions) {
  if (pitch == world_pitch_ && mode == world_mode_ && center == world_center_ &&
      dimensions == world_dimensions_ && points.size() == world_points_.size() &&
      std::equal(points.begin(), points.end(), world_points_.begin(),
                 [](const Point &a, const Point &b) { return a == b; }))
    return;
  world_points_ = points;
  world_pitch_ = pitch;
  world_mode_ = mode;
  world_center_ = center;
  world_dimensions_ = dimensions;
  rebuild_world();
}
// 更新手上的工件，并让碰撞世界跟着变。
void NativePlanner::set_payload(const std::optional<Payload> &payload) {
  payload_ = payload;
  auto apply = [&](auto &groups) {
    for (auto &entry : groups) {
      auto &group = *entry.second;
      for (const auto &side : {"left_arm", "right_arm"})
        group.planner->detach_object(robot_.profile.payload_link(side));
      if (payload) {
        curobo::robot::SphereAttachment attachment;
        attachment.link_name = robot_.profile.payload_link(payload->group);
        const auto &value = payload->in_tool;
        const Eigen::Quaterniond q(value.linear());
        attachment.object_in_link.position = {static_cast<float>(value.translation().x()),
                                              static_cast<float>(value.translation().y()),
                                              static_cast<float>(value.translation().z())};
        attachment.object_in_link.quaternion = {
            static_cast<float>(q.w()), static_cast<float>(q.x()), static_cast<float>(q.y()),
            static_cast<float>(q.z())};
        const auto cell = payload->size / 3.;
        const float radius = cell.norm() * .5 + .001;
        for (int x = 0; x < 3; ++x)
          for (int y = 0; y < 3; ++y)
            for (int z = 0; z < 3; ++z)
              attachment.spheres.push_back(
                  {{static_cast<float>((x - 1) * cell.x()), static_cast<float>((y - 1) * cell.y()),
                    static_cast<float>((z - 1) * cell.z())},
                   radius});
        group.planner->attach_object(attachment);
      }
    }
  };
  apply(groups_);
  apply(mpc_groups_);
}
}  // namespace curobo_ros_cpp
