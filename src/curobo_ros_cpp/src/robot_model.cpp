// SPDX-License-Identifier: Apache-2.0
// 解析机器人配置和 URDF，计算 CPU 正运动学并执行 FCL 碰撞检查。
#include "curobo_ros_cpp/robot_model.hpp"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <future>
#include <queue>
#include <sstream>
#include <thread>
#include <unordered_map>

namespace curobo_ros_cpp {
namespace {
std::string substitute(std::string value, const std::string &key, const std::string &replacement) {
  const auto at = value.find(key);
  if (at != std::string::npos) value.replace(at, key.size(), replacement);
  return value;
}
Transform urdf_origin(const urdf::Pose &pose) {
  Transform result = Transform::Identity();
  result.linear() =
      Eigen::Quaterniond(pose.rotation.w, pose.rotation.x, pose.rotation.y, pose.rotation.z)
          .toRotationMatrix();
  result.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  return result;
}
std::shared_ptr<fcl::BVHModel<fcl::OBBRSSd>> make_model(
    const std::vector<Eigen::Vector3d> &vertices, const std::vector<fcl::Triangle> &faces) {
  if (vertices.empty() || faces.empty()) throw std::invalid_argument("empty collision mesh");
  auto model = std::make_shared<fcl::BVHModel<fcl::OBBRSSd>>();
  model->beginModel(static_cast<int>(faces.size()), static_cast<int>(vertices.size()));
  model->addSubModel(vertices, faces);
  model->endModel();
  return model;
}
bool intersects(fcl::CollisionObjectd &first, fcl::CollisionObjectd &second) {
  fcl::CollisionResultd result;
  return fcl::collide(&first, &second, fcl::CollisionRequestd(), result) > 0;
}
bool inside_mesh(const Eigen::Vector3d &point, const MeshGeometry &mesh) {
  const Eigen::Vector3d direction =
      Eigen::Vector3d(1., .1732050807569, .2718281828459).normalized();
  std::vector<double> hits;
  for (const auto &face : mesh.faces) {
    const auto &a = mesh.vertices.at(face[0]);
    const auto edge1 = mesh.vertices.at(face[1]) - a;
    const auto edge2 = mesh.vertices.at(face[2]) - a;
    const Eigen::Vector3d h = direction.cross(edge2);
    const double determinant = edge1.dot(h);
    if (std::abs(determinant) < 1e-12) continue;
    const auto s = point - a;
    const double u = s.dot(h) / determinant;
    if (u < -1e-10 || u > 1. + 1e-10) continue;
    const Eigen::Vector3d q = s.cross(edge1);
    const double v = direction.dot(q) / determinant;
    if (v < -1e-10 || u + v > 1. + 1e-10) continue;
    const double t = edge2.dot(q) / determinant;
    if (t > 1e-10) hits.push_back(t);
  }
  std::sort(hits.begin(), hits.end());
  hits.erase(std::unique(hits.begin(), hits.end(),
                         [](double a, double b) { return std::abs(a - b) < 1e-8; }),
             hits.end());
  return hits.size() % 2 == 1;
}
}  // namespace

// 读取机器人名字表。关节顺序要和控制器一致。
RobotProfile::RobotProfile(const std::string &file) {
  const auto config = YAML::LoadFile(file);
  id = config["robot_id"].as<std::string>();
  groups["left_arm"] = strings(config["left_joints"]);
  groups["right_arm"] = strings(config["right_joints"]);
  groups["both_arms"] = groups.at("left_arm");
  groups["both_arms"].insert(groups["both_arms"].end(), groups["right_arm"].begin(),
                             groups["right_arm"].end());
  for (const auto &entry : config["ee_links"])
    tools[entry.first.as<std::string>()] = entry.second.as<std::string>();
  for (const auto &entry : config["gripper_joints"])
    grippers[entry.first.as<std::string>()] = entry.second.as<std::string>();
  for (const auto &entry : config["gripper_controllers"])
    gripper_controllers[entry.first.as<std::string>()] = entry.second.as<std::string>();
  const std::map<std::string, std::string> default_topics{
      {"left_arm", "/left_arm_controller/follow_joint_trajectory"},
      {"right_arm", "/right_arm_controller/follow_joint_trajectory"},
      {"both_arms", "/both_arms_controller/follow_joint_trajectory"}};
  const std::map<std::string, std::string> default_names{
      {"left_arm", "left_arm_controller"},
      {"right_arm", "right_arm_controller"},
      {"both_arms", "both_arms_controller"}};
  arm_controllers = default_topics;
  arm_controller_names = default_names;
  if (config["arm_controllers"])
    for (const auto &entry : config["arm_controllers"])
      arm_controllers[entry.first.as<std::string>()] = entry.second.as<std::string>();
  if (config["arm_controller_names"])
    for (const auto &entry : config["arm_controller_names"])
      arm_controller_names[entry.first.as<std::string>()] = entry.second.as<std::string>();
  if (config["posture_goals"])
    for (const auto &named : config["posture_goals"]) {
      JointMap goal;
      for (const auto &entry : named.second)
        goal[entry.first.as<std::string>()] = entry.second.as<double>();
      postures[named.first.as<std::string>()] = std::move(goal);
    }
  if (postures.count("l")) posture = postures.at("l");
  camera_frame = config["camera_optical_frame"].as<std::string>();
  camera_parent = config["camera_parent_frame"].as<std::string>();
  policy_file = config["fcl_self_collision"].as<std::string>();
  payload_pattern = config["payload_link"].as<std::string>();
  jaw_patterns = strings(config["jaw_links"]);
  gripper_base_pattern = config["gripper_base_link"].as<std::string>();
}
// 按规划组展开携物连杆名。
std::string RobotProfile::payload_link(const std::string &group) const {
  return substitute(payload_pattern, "{group}", group);
}
// 接触段需要放行的夹爪连杆。
std::vector<std::string> RobotProfile::contact_links(const std::string &group) const {
  const auto side = group.substr(0, group.find('_'));
  std::vector<std::string> result;
  for (const auto &pattern : jaw_patterns) result.push_back(substitute(pattern, "{side}", side));
  return result;
}
std::string Collision::message() const { return kind + " " + first + " / " + second; }

// 读 URDF、默认姿态和碰撞网格，并建好允许忽略的连杆对。
RobotModel::RobotModel(const std::string &urdf, const std::string &robot_yaml,
                       const std::string &profile_file, const std::string &config_directory,
                       const std::string &geometry_directory)
    : profile(profile_file) {
  if (!urdf_.initFile(urdf)) throw std::runtime_error("cannot load URDF " + urdf);
  const auto kin = YAML::LoadFile(robot_yaml)["robot_cfg"]["kinematics"];
  base_ = kin["base_link"].as<std::string>();
  const auto names = strings(kin["cspace"]["joint_names"]);
  const auto positions = kin["cspace"]["default_joint_position"];
  for (size_t i = 0; i < names.size(); ++i) defaults[names[i]] = positions[i].as<double>();
  for (const auto &item : kin["lock_joints"])
    defaults[item.first.as<std::string>()] = item.second.as<double>();
  std::queue<urdf::LinkConstSharedPtr> pending;
  pending.push(urdf_.getLink(base_));
  while (!pending.empty()) {
    auto link = pending.front();
    pending.pop();
    if (!link) throw std::runtime_error("URDF base frame missing");
    for (const auto &child : link->child_links) {
      auto joint = child->parent_joint;
      joints_.push_back(joint);
      pending.push(child);
      if (joint->type != urdf::Joint::FIXED && !joint->mimic) {
        full_joint_names.push_back(joint->name);
        if (!defaults.count(joint->name)) defaults[joint->name] = 0.;
        if (joint->limits) {
          limits[joint->name] = {joint->limits->lower, joint->limits->upper};
          velocity_limits[joint->name] = joint->limits->velocity;
        }
      }
    }
  }
  const auto cache =
      YAML::LoadFile((std::filesystem::path(geometry_directory) / "geometry.yaml").string());
  for (const auto &entry : cache["meshes"]) {
    MeshGeometry mesh;
    mesh.link = entry["link"].as<std::string>();
    mesh.watertight = entry["watertight"].as<bool>();
    const auto vertices = entry["vertex_count"].as<size_t>();
    const auto faces = entry["face_count"].as<size_t>();
    std::vector<double> packed_vertices(vertices * 3);
    std::vector<int32_t> packed_faces(faces * 3);
    std::ifstream vertex_file(
        std::filesystem::path(geometry_directory) / entry["vertices"].as<std::string>(),
        std::ios::binary);
    std::ifstream face_file(
        std::filesystem::path(geometry_directory) / entry["faces"].as<std::string>(),
        std::ios::binary);
    if (!vertex_file.read(reinterpret_cast<char *>(packed_vertices.data()),
                          packed_vertices.size() * sizeof(double)) ||
        !face_file.read(reinterpret_cast<char *>(packed_faces.data()),
                        packed_faces.size() * sizeof(int32_t)))
      throw std::runtime_error("invalid exported robot geometry");
    for (size_t i = 0; i < vertices; ++i)
      mesh.vertices.emplace_back(packed_vertices[i * 3], packed_vertices[i * 3 + 1],
                                 packed_vertices[i * 3 + 2]);
    for (size_t i = 0; i < faces; ++i)
      mesh.faces.emplace_back(packed_faces[i * 3], packed_faces[i * 3 + 1],
                              packed_faces[i * 3 + 2]);
    if (mesh.vertices.empty()) throw std::invalid_argument("empty collision mesh");
    mesh.lower = mesh.vertices.front();
    mesh.upper = mesh.lower;
    for (const auto &point : mesh.vertices) {
      mesh.lower = mesh.lower.cwiseMin(point);
      mesh.upper = mesh.upper.cwiseMax(point);
    }
    if (mesh.faces.empty()) throw std::invalid_argument("empty collision mesh");
    meshes_.push_back(std::move(mesh));
  }
  // Each link owns its BVH. Build independent trees in parallel during startup.
  std::atomic_size_t next_mesh{0};
  const auto build_meshes = [&] {
    for (size_t i = next_mesh.fetch_add(1); i < meshes_.size(); i = next_mesh.fetch_add(1)) {
      auto &mesh = meshes_[i];
      mesh.model = make_model(mesh.vertices, mesh.faces);
      mesh.object = std::make_unique<fcl::CollisionObjectd>(mesh.model);
    }
  };
  std::vector<std::future<void>> mesh_builders;
  const auto builders = std::min(4u, std::max(1u, std::thread::hardware_concurrency()));
  for (unsigned i = 1; i < builders; ++i)
    mesh_builders.push_back(std::async(std::launch::async, build_meshes));
  build_meshes();
  for (auto &builder : mesh_builders) builder.get();
  for (const auto &mesh : meshes_) {
    auto link = urdf_.getLink(mesh.link);
    while (link && link->parent_joint) {
      urdf::JointConstSharedPtr joint = link->parent_joint;
      if (joint->type != urdf::Joint::FIXED) {
        std::set<std::string> visited;
        while (joint->mimic) {
          if (!visited.insert(joint->name).second)
            throw std::runtime_error("cyclic mimic feedback dependency");
          joint = urdf_.getJoint(joint->mimic->joint_name);
          if (!joint) throw std::runtime_error("missing mimic source joint");
        }
        required_joint_names.insert(joint->name);
      }
      link = urdf_.getLink(link->parent_joint->parent_link_name);
    }
  }
  auto policy =
      YAML::LoadFile((std::filesystem::path(config_directory) / profile.policy_file).string());
  std::set<std::pair<std::string, std::string>> ignored;
  for (const auto &entry : policy["self_collision_ignore"]) {
    const auto first = entry.first.as<std::string>();
    for (const auto &second : strings(entry.second)) ignored.insert(std::minmax(first, second));
  }
  for (size_t i = 0; i < meshes_.size(); ++i)
    for (size_t j = i + 1; j < meshes_.size(); ++j)
      if (!ignored.count(std::minmax(meshes_[i].link, meshes_[j].link))) pairs_.emplace_back(i, j);
}

// 解析 mimic 关节。visited 用来发现循环引用。
double RobotModel::joint_value(const std::string &name, const JointMap &values,
                               std::set<std::string> &visited) const {
  if (!visited.insert(name).second) throw std::runtime_error("cyclic mimic joint");
  const auto joint = urdf_.getJoint(name);
  if (joint->mimic)
    return joint_value(joint->mimic->joint_name, values, visited) * joint->mimic->multiplier +
           joint->mimic->offset;
  auto found = values.find(name);
  const double value = found == values.end() ? defaults.at(name) : found->second;
  if (!std::isfinite(value)) throw std::runtime_error("nonfinite joint " + name);
  return value;
}
// 沿 URDF 树计算每个连杆的基座系位姿。
std::map<std::string, Transform> RobotModel::forward(const JointMap &state) const {
  std::map<std::string, Transform> result;
  result[base_] = Transform::Identity();
  for (const auto &joint : joints_) {
    auto value = urdf_origin(joint->parent_to_joint_origin_transform);
    if (joint->type != urdf::Joint::FIXED) {
      std::set<std::string> visited;
      const auto q = joint_value(joint->name, state, visited);
      const Eigen::Vector3d axis(joint->axis.x, joint->axis.y, joint->axis.z);
      if (joint->type == urdf::Joint::PRISMATIC)
        value.translate(axis * q);
      else
        value.rotate(Eigen::AngleAxisd(q, axis.normalized()));
    }
    result[joint->child_link_name] = result.at(joint->parent_link_name) * value;
  }
  return result;
}
// 该组关节驱动到的连杆。过滤本体点云时只用这一部分。
std::set<std::string> RobotModel::moving_links(const std::string &group) const {
  const auto &names = profile.groups.at(group);
  std::set<std::string> active(names.begin(), names.end()), result;
  for (const auto &joint : joints_)
    if (active.count(joint->name) || result.count(joint->parent_link_name))
      result.insert(joint->child_link_name);
  return result;
}
// 该组工具连杆的基座系位姿。
Transform RobotModel::tool(const JointMap &state, const std::string &group) const {
  return forward(state).at(profile.tools.at(group));
}
// 先摆好网格，再查自碰撞、环境和携物。
std::optional<Collision> RobotModel::check(const JointMap &state, const std::string &group,
                                           bool environment, bool contact,
                                           const std::optional<Payload> &payload) const {
  std::lock_guard<std::mutex> guard(query_mutex_);
  const auto transforms = forward(state);
  const auto moving = moving_links(group);
  const auto contact_names = profile.contact_links(group);
  const std::set<std::string> fingers(contact_names.begin(), contact_names.end());
  for (const auto &mesh : meshes_) {
    mesh.object->setTransform(transforms.at(mesh.link));
    mesh.object->computeAABB();
  }
  for (const auto &pair : pairs_) {
    const auto &first = meshes_[pair.first];
    const auto &second = meshes_[pair.second];
    if (!(moving.count(first.link) || moving.count(second.link))) continue;
    if (first.object->getAABB().overlap(second.object->getAABB()) &&
        intersects(*first.object, *second.object))
      return Collision{"self", first.link, second.link};
  }
  if (environment && environment_)
    for (const auto &mesh : meshes_)
      if (moving.count(mesh.link) && !(contact && fingers.count(mesh.link)) &&
          mesh.object->getAABB().overlap(environment_->getAABB()) &&
          intersects(*mesh.object, *environment_))
        return Collision{"environment", mesh.link, "environment"};
  if (payload) {
    const auto side = payload->group.substr(0, payload->group.find('_'));
    const auto &tool_name = profile.tools.at(payload->group);
    std::set<std::string> allowed{tool_name,
                                  substitute(profile.gripper_base_pattern, "{side}", side)};
    for (const auto &name : profile.contact_links(payload->group)) allowed.insert(name);
    auto geometry =
        std::make_shared<fcl::Boxd>(payload->size.x(), payload->size.y(), payload->size.z());
    fcl::CollisionObjectd object(geometry, transforms.at(tool_name) * payload->in_tool);
    const bool payload_moves = moving.count(tool_name);
    for (const auto &mesh : meshes_)
      if (!allowed.count(mesh.link) && (payload_moves || moving.count(mesh.link)) &&
          object.getAABB().overlap(mesh.object->getAABB()) && intersects(object, *mesh.object))
        return Collision{"payload_self", "payload", mesh.link};
    if (payload_moves && environment && environment_ && intersects(object, *environment_))
      return Collision{"payload_environment", "payload", "environment"};
  }
  return {};
}
// 判断点是否落在运动连杆里。预算用完就停，剩下的点不算本体。
std::vector<bool> RobotModel::self_mask(const std::vector<Point> &points, const JointMap &state,
                                        const std::string &group, double padding, double budget_ms,
                                        bool &timed_out) const {
  std::lock_guard<std::mutex> guard(query_mutex_);
  const auto started = Clock::now();
  const auto transforms = forward(state);
  const auto moving = moving_links(group);
  std::vector<bool> remove(points.size(), false);
  timed_out = false;
  struct Box {
    Eigen::Vector3d lower, upper;
    Transform inverse;
    const MeshGeometry *mesh;
  };
  std::vector<Box> boxes;
  for (const auto &mesh : meshes_) {
    if (!moving.count(mesh.link)) continue;
    const auto &pose = transforms.at(mesh.link);
    Eigen::Vector3d lower = Eigen::Vector3d::Constant(1e9);
    Eigen::Vector3d upper = Eigen::Vector3d::Constant(-1e9);
    for (int corner = 0; corner < 8; ++corner) {
      const Eigen::Vector3d local((corner & 1) ? mesh.upper.x() : mesh.lower.x(),
                                  (corner & 2) ? mesh.upper.y() : mesh.lower.y(),
                                  (corner & 4) ? mesh.upper.z() : mesh.lower.z());
      const Eigen::Vector3d world = pose * local;
      lower = lower.cwiseMin(world);
      upper = upper.cwiseMax(world);
    }
    lower.array() -= padding;
    upper.array() += padding;
    boxes.push_back({lower, upper, pose.inverse(), &mesh});
  }
  std::vector<fcl::CollisionObjectd> local_meshes;
  local_meshes.reserve(boxes.size());
  for (const auto &box : boxes) local_meshes.emplace_back(box.mesh->model);
  auto sphere = std::make_shared<fcl::Sphered>(std::max(padding, 1e-9));
  fcl::CollisionObjectd probe(sphere);
  for (size_t i = 0; i < points.size(); ++i) {
    if (budget_ms > 0. && elapsed_ms(started) > budget_ms) {
      timed_out = true;
      break;
    }
    const Eigen::Vector3d world = points[i].cast<double>();
    for (size_t box_index = 0; box_index < boxes.size(); ++box_index) {
      const auto &box = boxes[box_index];
      if ((world.array() < box.lower.array()).any() || (world.array() > box.upper.array()).any())
        continue;
      const Eigen::Vector3d point = box.inverse * world;
      const auto &mesh = *box.mesh;
      if ((point.array() < (mesh.lower.array() - padding)).any() ||
          (point.array() > (mesh.upper.array() + padding)).any())
        continue;
      Transform at = Transform::Identity();
      at.translation() = point;
      probe.setTransform(at);
      probe.computeAABB();
      if (intersects(probe, local_meshes[box_index]) ||
          (mesh.watertight && inside_mesh(point, mesh))) {
        remove[i] = true;
        break;
      }
    }
  }
  return remove;
}
// 替换 FCL 环境网格。GPU 场景由规划器另行更新。
void RobotModel::set_environment(const std::vector<Eigen::Vector3d> &vertices,
                                 const std::vector<fcl::Triangle> &faces) {
  std::lock_guard<std::mutex> guard(query_mutex_);
  auto geometry = make_model(vertices, faces);
  auto object = std::make_unique<fcl::CollisionObjectd>(geometry);
  environment_geometry_ = std::move(geometry);
  environment_ = std::move(object);
}
// 去掉环境障碍，自碰撞检查仍然保留。
void RobotModel::clear_environment() {
  std::lock_guard<std::mutex> guard(query_mutex_);
  environment_.reset();
  environment_geometry_.reset();
}
// 用关节速度上限把夹爪行程换成秒。
double RobotModel::gripper_duration(const std::string &joint, double start, double target) const {
  const double delta = std::abs(target - start);
  if (delta <= .0005) return 0.;
  const auto bounds = limits.at(joint);
  const auto speed = velocity_limits.at(joint);
  if (speed <= 0 || bounds.second <= bounds.first)
    throw std::runtime_error("invalid gripper limits");
  return std::max(
      {.02, .55 * delta / (bounds.second - bounds.first), 1.875 * delta / speed * 1.01});
}
}  // namespace curobo_ros_cpp
