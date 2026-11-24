// SPDX-License-Identifier: Apache-2.0
// 初始化 GPU 场景，加载分组产物并同步锁定关节。
#include "curobo_ros_cpp/detail/native_planner_group.hpp"
#include <filesystem>
#include <fstream>

namespace curobo_ros_cpp {
namespace {
std::vector<float> tensor_file(const std::filesystem::path &root, const YAML::Node &value) {
  std::vector<float> result(value["bytes"].as<size_t>() / sizeof(float));
  std::ifstream file(root / value["file"].as<std::string>(), std::ios::binary);
  if (!file.read(reinterpret_cast<char *>(result.data()), result.size() * sizeof(float)))
    throw std::runtime_error("truncated model tensor");
  return result;
}
Eigen::VectorXd eigen(const std::vector<float> &values) {
  Eigen::VectorXd result(values.size());
  for (size_t i = 0; i < values.size(); ++i) result[i] = values[i];
  return result;
}
}  // namespace

// 只创建空的 GPU 场景。规划组等到真正用到时再加载。
NativePlanner::NativePlanner(RobotModel &robot, std::string artifacts,
                             PlannerSettings configuration)
    : settings(std::move(configuration)), robot_(robot), artifacts_(std::move(artifacts)) {
  if (settings.device.rfind("cuda:", 0) == 0) {
    const int device = std::stoi(settings.device.substr(5));
    check_cuda(cudaSetDevice(device));
  }
  curobo::robot::SceneConfig scene;
  scene.max_cuboids = 32;
  scene.max_capsules = 0;
  scene.max_cylinders = 0;
  scene.max_meshes = 8;
  scene.max_voxel_grids = 1;
  scene.max_voxels_per_grid = 4000000;
  scene_ = std::make_unique<curobo::robot::Scene>(scene);
  if (!scene_->mesh_support_available())
    throw std::runtime_error("native runtime requires the Warp mesh backend");
}
NativePlanner::~NativePlanner() = default;
// 从导出目录读运动学、轨迹优化和逆解，并按控制点数分配缓冲。
std::unique_ptr<NativePlanner::Group> NativePlanner::create_group(const std::string &name,
                                                                  int knots, bool mpc_mode) {
  auto group = std::make_unique<Group>();
  group->name = name;
  group->knots = knots;
  const auto role_root =
      std::filesystem::path(artifacts_) / name / (mpc_mode ? "mpc" : "opt") / std::to_string(knots);
  const auto root = std::filesystem::exists(role_root)
                        ? role_root
                        : std::filesystem::path(artifacts_) / name / std::to_string(knots);
  group->metadata = YAML::LoadFile((root / "deployment.json").string());
  const auto kin = YAML::LoadFile((root / "robot_kinematics_manifest.json").string());
  const auto trajopt = YAML::LoadFile((root / "robot_trajopt_manifest.json").string());
  group->interpolation_steps = trajopt["spline_layout"]["interpolation_steps"].as<int>();
  if (group->interpolation_steps < 1)
    throw std::runtime_error("artifact interpolation density is invalid");
  if (!mpc_mode && group->interpolation_steps != settings.interpolation_steps)
    throw std::runtime_error(
        "artifact interpolation density differs from configuration, export matching artifacts");
  const auto ik_manifest = YAML::LoadFile((root / "robot_ik_manifest.json").string());
  group->names = strings(kin["joint_names"]);
  group->tool_names = strings(kin["tool_frames"]);
  group->link_names = strings(kin["link_names"]);
  if (group->names != robot_.profile.groups.at(name))
    throw std::runtime_error("exported joint order differs from robot profile");
  group->dof = group->names.size();
  group->tools = group->tool_names.size();
  group->seeds = settings.seeds;
  group->ik_seeds = settings.ik_seeds;
  group->nominal_fixed = group->metadata["fixed_transforms"].as<std::vector<float>>();
  group->lower = group->metadata["position_lower"].as<std::vector<float>>();
  group->upper = group->metadata["position_upper"].as<std::vector<float>>();
  group->default_position = tensor_file(root, ik_manifest["tensors"]["default_joint_position"]);
  for (const auto &joint : group->metadata["dynamic_joints"]) {
    Group::FixedJoint parsed;
    parsed.source = joint["source"].as<std::string>();
    parsed.kind = joint["kind"].as<std::string>();
    parsed.xyz = vector3(joint["xyz"]);
    parsed.rpy = vector3(joint["rpy"]);
    parsed.axis = vector3(joint["axis"]).normalized();
    parsed.multiplier = joint["multiplier"].as<double>();
    parsed.offset = joint["offset"].as<double>();
    parsed.row = joint["row"].as<size_t>();
    parsed.fallback = group->metadata["default_positions"][parsed.source].as<double>();
    group->fixed_joints.push_back(std::move(parsed));
  }
  const auto limits = trajopt["tensors"];
  auto limit = [&](const std::string &field) {
    auto values = tensor_file(root, limits[field]);
    return eigen(std::vector<float>(values.begin() + group->dof, values.end())).cwiseAbs().eval();
  };
  group->limits = {limit("velocity_bounds"), limit("acceleration_bounds"), limit("jerk_bounds")};
  curobo::robot::MotionPlanningConfig configuration;
  auto &opt = configuration.trajectory_optimization;
  opt.num_seeds = settings.seeds;
  opt.num_knots = knots;
  opt.max_iterations = settings.iterations;
  opt.initial_check_interval = settings.iterations;
  opt.finetune_attempts = settings.finetune;
  opt.position_tolerance = settings.position_tolerance;
  opt.orientation_tolerance = settings.orientation_tolerance;
  opt.execution_speed_scale = settings.velocity_scale;
  opt.interpolation_dt = settings.dt;
  configuration.use_cuda_graph = settings.cuda_graph;
  configuration.inverse_kinematics.num_seeds = settings.ik_seeds;
  configuration.inverse_kinematics.lm_num_seeds = settings.ik_seeds;
  configuration.inverse_kinematics.max_goalset = 1;
  configuration.inverse_kinematics.position_tolerance = settings.position_tolerance;
  configuration.inverse_kinematics.orientation_tolerance = settings.orientation_tolerance;
  configuration.rrt.max_plans = settings.seeds;
  group->planner =
      std::make_unique<curobo::robot::MotionPlanner>(root.string(), configuration, scene_.get());
  group->kinematics = std::make_unique<curobo::robot::Kinematics>(root.string());
  auto ik_config = configuration.inverse_kinematics;
  group->ik = group->planner->create_inverse_kinematics(ik_config);
  if (!mpc_mode) {
    ik_config.num_seeds = 1;
    ik_config.lm_num_seeds = 1;
    ik_config.check_self_collision = false;
    group->contact_ik =
        std::make_unique<curobo::robot::InverseKinematics>(root.string(), ik_config, nullptr);
    ik_config.max_batch_size = Group::kIkBatchCapacity;
    group->batch_ik =
        std::make_unique<curobo::robot::InverseKinematics>(root.string(), ik_config, nullptr);
    ik_config.max_batch_size = 1;
    ik_config.num_seeds = 4;
    ik_config.lm_num_seeds = 4;
    ik_config.exit_early = false;
    group->refine_ik =
        std::make_unique<curobo::robot::InverseKinematics>(root.string(), ik_config, nullptr);
  }
  group->horizon = group->planner->info().trajectory_steps;
  group->capacity = trajopt["interpolation_buffer_size"].as<int>();
  // World masks preserve self-collision spheres, including the stationary arm and torso.
  std::set<int32_t> sphere_links;
  {
    const auto tensor = ik_manifest["tensors"]["link_sphere_map"];
    std::vector<int16_t> indices(tensor["bytes"].as<size_t>() / sizeof(int16_t));
    std::ifstream file(root / tensor["file"].as<std::string>(), std::ios::binary);
    if (!file.read(reinterpret_cast<char *>(indices.data()), indices.size() * sizeof(int16_t)))
      throw std::runtime_error("invalid link sphere map");
    sphere_links.insert(indices.begin(), indices.end());
  }
  const auto moving_names = group->metadata["moving_links"].as<std::vector<std::string>>();
  for (const auto &link : group->link_names) {
    if (sphere_links.count(std::find(group->link_names.begin(), group->link_names.end(), link) -
                           group->link_names.begin()) &&
        std::find(moving_names.begin(), moving_names.end(), link) == moving_names.end())
      group->planner->set_link_collision_mask(link, 0U);
  }
  group->start.resize(group->dof);
  group->goal.resize(group->dof);
  group->seed_goals.resize(group->seeds * group->dof);
  group->ik_goals.resize(Group::kIkBatchCapacity * group->tools * 3);
  group->ik_quaternions.resize(Group::kIkBatchCapacity * group->tools * 4);
  group->ik_current.resize(group->dof);
  group->ik_seeds_buffer.resize(Group::kIkBatchCapacity * group->dof);
  group->ik_solution.resize(Group::kIkBatchCapacity * group->dof);
  group->ik_success.resize(Group::kIkBatchCapacity);
  group->ik_position_error.resize(Group::kIkBatchCapacity);
  group->ik_orientation_error.resize(Group::kIkBatchCapacity);
  group->ik_collision.resize(Group::kIkBatchCapacity);
  group->seed_valid.resize(group->seeds);
  group->success.resize(1);
  group->used_rrt.resize(1);
  group->rrt_iterations.resize(1);
  group->cost.resize(1);
  group->collision.resize(1);
  group->trajectory_dt.resize(1);
  group->knots_buffer.resize(knots * group->dof);
  const size_t outputs = static_cast<size_t>(group->capacity) * group->dof;
  group->trajectory_position.resize(outputs);
  group->trajectory_velocity.resize(outputs);
  group->trajectory_acceleration.resize(outputs);
  group->trajectory_jerk.resize(outputs);
  group->fk_position.resize(group->tools * 3);
  group->fk_quaternion.resize(group->tools * 4);
  group->fk_jacobian.resize(group->tools * 6 * group->dof);
  group->validate_position.resize(group->seeds * outputs);
  if (!mpc_mode) group->configuration_feasible.resize(group->capacity);
  group->validate_velocity.resize(group->seeds * outputs);
  group->validate_acceleration.resize(group->seeds * outputs);
  group->validate_jerk.resize(group->seeds * outputs);
  group->mpc_velocity.resize(group->dof);
  group->mpc_acceleration.resize(group->dof);
  group->mpc_jerk.resize(group->dof);
  if (mpc_mode) {
    curobo::robot::MpcConfig mpc;
    mpc.iterations = settings.mpc_iterations;
    mpc.trajectory_dt = settings.mpc_optimization_dt / group->interpolation_steps;
    mpc.collision_activation_distance = settings.mpc_collision_activation_distance;
    group->mpc = std::make_unique<curobo::robot::MpcController>(*group->planner, mpc);
  }
  return group;
}
// OPT 和 MPC 分开缓存，避免两套控制点数互相覆盖。
NativePlanner::Group &NativePlanner::get(const std::string &name, bool mpc) {
  if (!robot_.profile.groups.count(name))
    throw std::invalid_argument("unknown planning group " + name);
  auto &groups = mpc ? mpc_groups_ : groups_;
  auto found = groups.find(name);
  if (found == groups.end()) {
    auto group = create_group(name, mpc ? settings.mpc_knots : settings.knots, mpc);
    found = groups.emplace(name, std::move(group)).first;
    if (payload_) set_payload(payload_);
  }
  return *found->second;
}
// 触发该组加载。预热和第一次规划都会走到这里。
void NativePlanner::initialize(const std::string &group, bool mpc) { get(group, mpc); }
// 按求解器关节顺序取位置，缺的用默认值。
std::vector<float> NativePlanner::positions(const Group &group, const JointMap &state) const {
  std::vector<float> result;
  for (const auto &name : group.names) {
    auto found = state.find(name);
    if (found == state.end() || !std::isfinite(found->second))
      throw std::runtime_error("missing or nonfinite joint " + name);
    result.push_back(found->second);
  }
  return result;
}
// 把当前关节写进锁定关节和活动关节，保证模型和机器人一致。
void NativePlanner::synchronize(const std::string &name, const JointMap &state, bool mpc) {
  auto &group = get(name, mpc);
  auto transforms = group.nominal_fixed;
  for (const auto &joint : group.fixed_joints) {
    auto value = state.find(joint.source);
    const double q = value == state.end() ? joint.fallback : value->second;
    if (!std::isfinite(q)) throw std::runtime_error("invalid fixed joint " + joint.source);
    auto at = origin(joint.xyz, joint.rpy);
    const double angle = q * joint.multiplier + joint.offset;
    if (joint.kind == "prismatic")
      at.translate(joint.axis * angle);
    else
      at.rotate(Eigen::AngleAxisd(angle, joint.axis));
    const size_t row = joint.row;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 4; ++j)
        transforms.at(row * 12 + i * 4 + j) = static_cast<float>(at.matrix()(i, j));
  }
  if (transforms != group.last_fixed) {
    group.planner->update_fixed_transforms(transforms);
    group.kinematics->update_fixed_transforms(transforms);
    if (group.contact_ik) group.contact_ik->update_fixed_transforms(transforms);
    if (group.refine_ik) group.refine_ik->update_fixed_transforms(transforms);
    if (group.batch_ik) group.batch_ik->update_fixed_transforms(transforms);
    if (group.mpc) group.mpc->update_fixed_transforms(transforms);
    group.last_fixed = std::move(transforms);
  }
}
}  // namespace curobo_ros_cpp
