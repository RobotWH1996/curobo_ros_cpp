// SPDX-License-Identifier: Apache-2.0
// 执行轨迹优化、图搜索回退、预热与轨迹验收。
#include "curobo_ros_cpp/detail/native_planner_group.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <random>
#include <rclcpp/rclcpp.hpp>
#include <utility>

namespace curobo_ros_cpp {
namespace {
std::string rrt_failure(const curobo::robot::RrtDiagnostics &stats) {
  if (stats.requested_pairs == 0) return "RRT 没有可检查的起点和目标，未开始搜索。";
  if (stats.valid_pairs == 0) {
    const bool start_rejected = stats.invalid_starts == stats.requested_pairs;
    const bool goal_rejected = stats.invalid_goals == stats.requested_pairs;
    if (start_rejected && goal_rejected)
      return "RRT 起点和目标检查均失败，未开始搜索。";
    if (start_rejected) return "RRT 起点检查失败，未开始搜索。";
    if (goal_rejected) return "RRT 目标检查失败，未开始搜索。";
    return "RRT 没有找到一组同时通过检查的起点和目标，未开始搜索。";
  }
  if (stats.paths_found > 0) return "RRT 找到了路径，但后续优化未能生成可执行轨迹。";
  const auto iterations = "RRT 搜索了 " + std::to_string(stats.iterations) + " 轮";
  return iterations + (stats.trees_connected ? "，找到了连接，但路径未通过检查。"
                                             : "，没能找到从起点到目标的路径。");
}
}  // namespace

// 把产物里的原始上限乘上速度比例。速度、加速度、加加速度分别乘一次、两次、三次。
DynamicLimits NativePlanner::dynamic_limits(const std::string &group) {
  auto limits = get(group).limits;
  const double scale = settings.velocity_scale;
  limits.velocity *= scale;
  limits.acceleration *= scale * scale;
  limits.jerk *= scale * scale * scale;
  return limits;
}
// 从 GPU 读回轨迹。需要降速时只拉长时间，不改路径形状。
Trajectory NativePlanner::output(Group &group, bool scaled) {
  const auto optimized = group.planner->timeopt_trajectory();
  int count = group.horizon;
  float dt = 0.;
  const float *p = group.trajectory_position.get(), *v = group.trajectory_velocity.get(),
              *a = group.trajectory_acceleration.get();
  if (optimized.sample_count) {
    check_cuda(cudaMemcpy(&count, optimized.sample_count, sizeof(count), cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(&dt, optimized.dt, sizeof(dt), cudaMemcpyDeviceToHost));
    p = optimized.position;
    v = optimized.velocity;
    a = optimized.acceleration;
  } else
    dt = group.trajectory_dt.host(1).front();
  if (count < 2 || count > group.capacity || !std::isfinite(dt) || dt <= 0)
    throw std::runtime_error("native solver returned invalid trajectory timing");
  std::vector<float> position(count * group.dof), velocity(position.size()),
      acceleration(position.size());
  check_cuda(
      cudaMemcpy(position.data(), p, position.size() * sizeof(float), cudaMemcpyDeviceToHost));
  check_cuda(
      cudaMemcpy(velocity.data(), v, velocity.size() * sizeof(float), cudaMemcpyDeviceToHost));
  check_cuda(cudaMemcpy(acceleration.data(), a, acceleration.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  const double scale = scaled ? settings.velocity_scale : 1.;
  Trajectory result;
  result.joint_names = group.names;
  for (int i = 0; i < count; ++i) {
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.time_from_start = duration(i * dt / scale);
    for (int j = 0; j < group.dof; ++j) {
      const auto at = i * group.dof + j;
      if (!std::isfinite(position[at]) || !std::isfinite(velocity[at]) ||
          !std::isfinite(acceleration[at]))
        throw std::runtime_error("native solver returned nonfinite trajectory");
      point.positions.push_back(position[at]);
      point.velocities.push_back(velocity[at] * scale);
      point.accelerations.push_back(acceleration[at] * scale * scale);
    }
    result.points.push_back(std::move(point));
  }
  return result;
}

// 先尝试轨迹优化。允许图搜索时，RRT 失败后还可以用 PRM。
PlanResult NativePlanner::solve(Group &group, const JointMap &state,
                                const std::vector<std::vector<float>> &goals, int attempts,
                                bool pose, bool allow_graph) {
  PlanResult result;
  const auto current = positions(group, state);
  group.start.upload(current);
  std::vector<float> prm_goal;
  int total_rrt_calls = 0;
  const auto logger = rclcpp::get_logger("curobo_motion_planning_node");
  std::string last_failure = "未能生成可执行轨迹。";
  for (int attempt = 0; attempt < attempts; ++attempt) {
    auto reject = [&](std::string reason) {
      last_failure = std::move(reason);
      RCLCPP_DEBUG(logger, "【OPT 规划】%s的%s规划第 %d 次尝试失败，共允许尝试 %d 次。%s",
                   group.name.c_str(), pose ? "位姿" : "关节", attempt + 1, attempts,
                   last_failure.c_str());
    };
    result.timing.attempts = attempt + 1;
    std::vector<std::vector<float>> valid_goals = goals;
    if (pose) {
      const auto before = Clock::now();
      group.ik_current.upload(current);
      curobo::robot::IkSolveRequest ik;
      ik.goal_positions = group.ik_goals.get();
      ik.goal_quaternions = group.ik_quaternions.get();
      ik.seed_positions = nullptr;
      ik.current_positions = group.ik_current.get();
      ik.solution = group.ik_solution.get();
      ik.success = group.ik_success.get();
      ik.position_error = group.ik_position_error.get();
      ik.orientation_error = group.ik_orientation_error.get();
      ik.collision_cost = group.ik_collision.get();
      ik.num_solutions = group.ik_seeds;
      ik.seed_offset = attempt * group.ik_seeds;
      group.ik->solve(ik);
      const auto flags = group.ik_success.host(group.ik_seeds);
      const auto solutions = group.ik_solution.host(group.ik_seeds * group.dof);
      result.timing.ik_ms += elapsed_ms(before);
      valid_goals.clear();
      int ik_passed = 0;
      std::string first_collision;
      const auto check_started = Clock::now();
      for (int seed = 0; seed < group.ik_seeds; ++seed)
        if (flags[seed]) {
          ++ik_passed;
          JointMap full = state;
          std::vector<float> q;
          for (int j = 0; j < group.dof; ++j) {
            q.push_back(solutions[seed * group.dof + j]);
            full[group.names[j]] = q.back();
          }
          const auto collision = robot_.check(full, group.name, true, false, payload_);
          if (!collision)
            valid_goals.push_back(std::move(q));
          else if (first_collision.empty())
            first_collision = collision->first + " 与 " + collision->second;
        }
      result.timing.check_ms += elapsed_ms(check_started);
      RCLCPP_DEBUG(logger, "【OPT 逆解】%s尝试了 %d 个逆解初值，求得 %d 个候选，其中 %zu 个通过了终点碰撞检查。",
                   group.name.c_str(), group.ik_seeds, ik_passed, valid_goals.size());
      if (valid_goals.empty()) {
        if (ik_passed) {
          reject("已算出目标关节角，但所有候选都在目标处发生碰撞。第一处碰撞为 " +
                 first_collision + "。");
        } else {
          const auto position = group.ik_position_error.host(group.ik_seeds);
          const auto orientation = group.ik_orientation_error.host(group.ik_seeds);
          const auto collision = group.ik_collision.host(group.ik_seeds);
          int best = 0;
          double score = std::numeric_limits<double>::infinity();
          for (int seed = 0; seed < group.ik_seeds; ++seed) {
            const double candidate = position[seed] / settings.position_tolerance +
                                     orientation[seed] / settings.orientation_tolerance;
            if (std::isfinite(candidate) && candidate < score) {
              score = candidate;
              best = seed;
            }
          }
          RCLCPP_DEBUG(logger,
                       "【OPT 逆解】最佳候选的位置误差为 %.6f 米，允许 %.6f 米。姿态误差为 %.6f 弧度，"
                       "允许 %.6f 弧度。碰撞代价为 %.6f。",
                       position[best], settings.position_tolerance, orientation[best],
                       settings.orientation_tolerance, collision[best]);
          reject("逆运动学未找到满足目标位姿和约束的关节角。");
        }
        continue;
      }
    }
    if (valid_goals.empty()) continue;
    std::vector<float> packed;
    for (int seed = 0; seed < group.seeds; ++seed) {
      const auto &q = valid_goals[seed % valid_goals.size()];
      packed.insert(packed.end(), q.begin(), q.end());
    }
    group.seed_goals.upload(packed);
    group.goal.upload(valid_goals.front());
    group.seed_valid.upload(std::vector<uint8_t>(group.seeds, 1));
    curobo::robot::MotionPlanningRequest request;
    request.start_position = group.start.get();
    request.goal_position = group.goal.get();
    request.seed_goal_positions = group.seed_goals.get();
    request.seed_valid = group.seed_valid.get();
    request.optimized_knots = group.knots_buffer.get();
    request.trajectory_position = group.trajectory_position.get();
    request.trajectory_velocity = group.trajectory_velocity.get();
    request.trajectory_acceleration = group.trajectory_acceleration.get();
    request.trajectory_jerk = group.trajectory_jerk.get();
    request.success = group.success.get();
    request.used_rrt = group.used_rrt.get();
    request.rrt_iterations = group.rrt_iterations.get();
    request.cost = group.cost.get();
    request.collision_cost = group.collision.get();
    request.trajectory_dt = group.trajectory_dt.get();
    request.finetune_attempts = settings.finetune;
    request.allow_rrt = allow_graph && settings.graph == "rrt" && attempt > 0;
    const auto before = Clock::now();
    group.planner->plan(request);
    const bool success = group.success.host(1).front() != 0;
    const double total = elapsed_ms(before);
    const int rrt_calls = group.planner->last_rrt_calls();
    total_rrt_calls += rrt_calls;
    const double rrt_ms = group.planner->last_rrt_time() * 1000.;
    result.timing.graph_ms += rrt_ms;
    result.timing.trajopt_ms += std::max(0., total - rrt_ms);
    if (!success) {
      if (rrt_calls > 0)
        reject(rrt_failure(group.planner->last_rrt_diagnostics()));
      else
        reject("直接优化未能生成可执行轨迹，求解器未提供失败原因。");
      RCLCPP_DEBUG(logger, "【OPT 规划】第 %d 次求解共耗时 %.1f 毫秒，其中 RRT 用时 %.1f 毫秒。",
                   attempt + 1, total, rrt_ms);
      if (allow_graph && prm_goal.empty()) prm_goal = valid_goals.front();
      continue;
    }
    result.trajectory = output(group, true);
    JointMap endpoint = state;
    for (int j = 0; j < group.dof; ++j)
      endpoint[group.names[j]] = result.trajectory.points.back().positions[j];
    const auto check = robot_.check(endpoint, group.name, true, false, payload_);
    if (check) {
      reject(std::string(rrt_calls > 0 ? "RRT 找到了轨迹" : "直接优化找到了轨迹") +
             "，但在终点发现 " + check->first + " 与 " + check->second + " 发生碰撞。");
      continue;
    }
    result.success = true;
    result.message = "planning success";
    return result;
  }
  if (!allow_graph) {
    result.message = last_failure;
    return result;
  }
  result.message = "轨迹规划共尝试 " + std::to_string(result.timing.attempts) + " 次。"
                   "最后一次尝试的结果是" + last_failure;
  if (prm_goal.empty() ||
      (settings.graph != "prm" && total_rrt_calls == 0))
    return result;

  RCLCPP_INFO(logger, "【OPT · 回退】%s ｜ 改用 PRM 搜索路径 ｜ %s",
              pose ? "位姿规划" : "关节规划", last_failure.c_str());
  const auto before_prm = Clock::now();
  PlanResult fallback;
  try {
    fallback = plan_prm(group, state, prm_goal);
  } catch (const std::runtime_error &error) {
    fallback.message = std::string("PRM 规划遇到异常。") + error.what();
  }
  const double fallback_ms = elapsed_ms(before_prm);
  result.timing.graph_ms += fallback_ms;
  fallback.timing = result.timing;
  if (fallback.success)
    RCLCPP_INFO(logger, "【OPT · PRM】通过 ｜ PRM 用时 %.1f ms ｜ %s",
                fallback_ms, fallback.message.c_str());
  else
    RCLCPP_WARN(logger, "【OPT · PRM】失败 ｜ PRM 用时 %.1f ms ｜ %s",
                fallback_ms, fallback.message.c_str());
  return fallback;
}
// 关节目标直接优化，不经过逆解。
PlanResult NativePlanner::plan_joint(const std::string &name, const JointMap &state,
                                     const std::vector<double> &target, int attempts) {
  synchronize(name, state);
  auto &group = get(name);
  if (target.size() != group.names.size()) throw std::invalid_argument("joint goal size mismatch");
  JointMap final = state;
  std::vector<float> goal;
  for (size_t i = 0; i < target.size(); ++i) {
    if (!std::isfinite(target[i])) throw std::invalid_argument("nonfinite joint goal");
    final[group.names[i]] = target[i];
    goal.push_back(target[i]);
  }
  PlanResult result;
  const auto check = robot_.check(final, name, true, false, payload_);
  if (check) {
    result.message = "到达目标时，" + check->first + " 会与 " + check->second + " 发生碰撞。";
    return result;
  }
  std::map<std::string, Transform> target_poses;
  for (const auto &side : {"left_arm", "right_arm"})
    if (name == "both_arms" || name == side)
      target_poses[robot_.profile.tools.at(side)] = robot_.tool(final, side);
  fill_targets(group, state, target_poses);
  return solve(group, state, {goal}, attempts, false);
}
// 位姿先逆解出关节目标，再做轨迹优化。
PlanResult NativePlanner::plan_pose(const std::string &name, const JointMap &state,
                                    const std::map<std::string, Transform> &targets, int attempts) {
  for (const auto &[tool, target] : targets) {
    const Eigen::Quaterniond q(target.linear());
    const auto &p = target.translation();
    RCLCPP_DEBUG(rclcpp::get_logger("curobo_motion_planning_node"),
                 "【OPT 目标】%s的末端 %s 相对基座的目标坐标为 %.6f、%.6f、%.6f 米。"
                 "目标姿态的四元数按 x、y、z、w 顺序为 %.6f、%.6f、%.6f、%.6f。",
                 name.c_str(), tool.c_str(), p.x(), p.y(), p.z(), q.x(), q.y(), q.z(), q.w());
  }
  synchronize(name, state);
  auto &group = get(name);
  fill_targets(group, state, targets);
  return solve(group, state, {}, attempts, true);
}
// 多个位姿路点一次求解，供逐段拼接失败时回退。
PlanResult NativePlanner::plan_pose_waypoints(
    const std::string &name, const JointMap &state,
    const std::vector<std::map<std::string, Transform>> &targets) {
  const auto started = Clock::now();
  if (targets.size() < 2) throw std::invalid_argument("pose route needs multiple waypoints");
  synchronize(name, state);
  auto &group = get(name);
  const auto fixed = robot_.forward(state);
  std::vector<float> positions, quaternions;
  JointMap seed_state = state;
  std::vector<double> endpoint;
  for (const auto &waypoint : targets) {
    auto ik = inverse_kinematics(name, seed_state, waypoint, true);
    if (!ik.success) return ik;
    endpoint = ik.trajectory.points.back().positions;
    for (int j = 0; j < group.dof; ++j) seed_state[group.names[j]] = endpoint[j];
    for (const auto &tool : group.tool_names) {
      const auto found = waypoint.find(tool);
      const auto at = found == waypoint.end() ? fixed.at(tool) : found->second;
      Eigen::Quaterniond q(at.linear());
      for (int axis = 0; axis < 3; ++axis) positions.push_back(at.translation()[axis]);
      quaternions.insert(quaternions.end(), {static_cast<float>(q.w()), static_cast<float>(q.x()),
                                             static_cast<float>(q.y()), static_cast<float>(q.z())});
    }
  }
  DeviceArray<float> goal_positions(positions.size()), goal_quaternions(quaternions.size());
  goal_positions.upload(positions);
  goal_quaternions.upload(quaternions);
  group.start.upload(this->positions(group, state));
  std::vector<float> seeds;
  for (int seed = 0; seed < group.ik_seeds; ++seed)
    for (double value : endpoint) seeds.push_back(value);
  group.ik_seeds_buffer.upload(seeds);
  curobo::robot::WaypointMotionPlanningRequest request;
  request.positions = goal_positions.get();
  request.quaternions = goal_quaternions.get();
  request.num_waypoints = targets.size();
  request.path_position_tolerance = settings.position_tolerance;
  request.path_orientation_tolerance = settings.orientation_tolerance;
  request.final_position_tolerance = settings.position_tolerance;
  request.final_orientation_tolerance = settings.orientation_tolerance;
  auto &motion = request.motion;
  motion.goal_positions = group.ik_goals.get();
  motion.goal_quaternions = group.ik_quaternions.get();
  motion.current_position = group.start.get();
  motion.ik_seed_positions = group.ik_seeds_buffer.get();
  motion.optimized_knots = group.knots_buffer.get();
  motion.trajectory_position = group.trajectory_position.get();
  motion.trajectory_velocity = group.trajectory_velocity.get();
  motion.trajectory_acceleration = group.trajectory_acceleration.get();
  motion.trajectory_jerk = group.trajectory_jerk.get();
  motion.success = group.success.get();
  motion.cost = group.cost.get();
  motion.collision_cost = group.collision.get();
  motion.used_rrt = group.used_rrt.get();
  motion.rrt_iterations = group.rrt_iterations.get();
  motion.trajectory_dt = group.trajectory_dt.get();
  motion.max_attempts = 3;
  motion.finetune_attempts = settings.finetune;
  group.planner->plan_pose_waypoints(request);
  PlanResult result;
  result.timing.trajopt_ms = elapsed_ms(started);
  result.timing.attempts = 1;
  if (!group.success.host(1).front()) {
    result.message = "unified Cartesian waypoint optimization failed";
    return result;
  }
  result.trajectory = output(group, true);
  validate(name, state, result.trajectory);
  const double total = seconds(result.trajectory.points.back().time_from_start);
  const int count = std::max(2, static_cast<int>(std::ceil(total / .005)) + 1);
  int previous = 0;
  for (const auto &waypoint : targets) {
    bool reached = false;
    for (int i = previous; i < count; ++i) {
      const auto sample = sample_trajectory(result.trajectory, total * i / (count - 1));
      JointMap values = state;
      for (int j = 0; j < group.dof; ++j) values[group.names[j]] = sample.position[j];
      const auto actual = robot_.forward(values);
      bool close = true;
      for (const auto &tool : waypoint)
        close = close &&
                (actual.at(tool.first).translation() - tool.second.translation()).norm() <=
                    settings.position_tolerance &&
                Eigen::Quaterniond(actual.at(tool.first).linear())
                        .angularDistance(Eigen::Quaterniond(tool.second.linear())) <=
                    settings.orientation_tolerance;
      if (close && !robot_.check(values, name, true, false, payload_)) {
        reached = true;
        previous = i;
        break;
      }
    }
    if (!reached) {
      result.message = "Cartesian waypoint FCL or pose tolerance check failed";
      return result;
    }
  }
  result.success = true;
  result.message = "unified Cartesian waypoint planning success";
  return result;
}
// 一批关节点的 GPU 碰撞检查，有一个失败就返回假。
bool NativePlanner::gpu_valid(Group &group, const std::vector<std::vector<float>> &samples) {
  if (samples.empty()) return false;
  if (samples.size() > static_cast<size_t>(group.capacity))
    throw std::runtime_error("validation sample capacity exceeded");
  const size_t batch = group.capacity;
  std::vector<float> positions, zero(static_cast<size_t>(group.seeds) * batch * group.dof, 0.F);
  positions.reserve(zero.size());
  for (int seed = 0; seed < group.seeds; ++seed)
    for (size_t i = 0; i < batch; ++i)
      positions.insert(positions.end(), samples[std::min(i, samples.size() - 1)].begin(),
                       samples[std::min(i, samples.size() - 1)].end());
  group.validate_position.upload(positions);
  group.validate_velocity.upload(zero);
  group.validate_acceleration.upload(zero);
  group.validate_jerk.upload(zero);
  curobo::robot::TrajectoryValidationRequest request;
  request.position = group.validate_position.get();
  request.velocity = group.validate_velocity.get();
  request.acceleration = group.validate_acceleration.get();
  request.jerk = group.validate_jerk.get();
  request.num_samples = batch;
  const auto flags = group.planner->validate(request);
  std::vector<int32_t> values(group.seeds);
  check_cuda(
      cudaMemcpy(values.data(), flags, values.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
  return std::all_of(values.begin(), values.end(), [](int32_t v) { return v == 0; });
}
// 合并后的轨迹再查一遍限位和碰撞，接触段可以跳过环境。
void NativePlanner::validate(const std::string &name, const JointMap &state,
                             const Trajectory &trajectory, bool collision) {
  synchronize(name, state);
  auto &group = get(name);
  const auto &arm = robot_.profile.groups.at(name);
  validate_dynamics(trajectory, arm, dynamic_limits(name));
  std::vector<size_t> columns;
  for (const auto &joint : group.names) {
    auto found = std::find(trajectory.joint_names.begin(), trajectory.joint_names.end(), joint);
    if (found == trajectory.joint_names.end()) throw std::runtime_error("validation joint missing");
    columns.push_back(found - trajectory.joint_names.begin());
  }
  std::vector<std::vector<float>> samples;
  const double end = seconds(trajectory.points.back().time_from_start);
  const int count = std::max(2, static_cast<int>(std::ceil(end / .005)) + 1);
  for (int i = 0; i < count; ++i) {
    const auto value = sample_trajectory(trajectory, end * i / (count - 1));
    std::vector<float> row;
    for (size_t j = 0; j < columns.size(); ++j) {
      const auto position = value.position[columns[j]];
      const float lower = group.lower[j];
      const float upper = group.upper[j];
      if (!std::isfinite(position) || position < lower - 1e-6 || position > upper + 1e-6)
        throw std::runtime_error("merged joint position limit violation");
      row.push_back(position);
    }
    samples.push_back(std::move(row));
    if (samples.size() == static_cast<size_t>(group.capacity) || i == count - 1) {
      if (collision && !gpu_valid(group, samples))
        throw std::runtime_error("merged GPU collision or joint-limit validation failed");
      samples.clear();
    }
  }
  if (collision)
    for (const auto &point : {trajectory.points.front(), trajectory.points.back()}) {
      JointMap values = state;
      for (size_t i = 0; i < columns.size(); ++i)
        values[group.names[i]] = point.positions.at(columns[i]);
      const auto hit = robot_.check(values, name, true, false, payload_);
      if (hit) throw std::runtime_error("merged FCL keypoint collision " + hit->message());
    }
}
// 重复一次小位移规划。第一次捕获 CUDA graph，之后只是重放。
void NativePlanner::warmup(const std::string &name, int iterations) {
  auto &group = get(name);
  JointMap state = robot_.defaults;
  synchronize(name, state);
  for (int i = 0; i < iterations; ++i) {
    auto goal = positions(group, state);
    goal[0] += .01F;
    group.start.upload(positions(group, state));
    group.goal.upload(goal);
    curobo::robot::MotionPlanningRequest request;
    request.start_position = group.start.get();
    request.goal_position = group.goal.get();
    request.optimized_knots = group.knots_buffer.get();
    request.trajectory_position = group.trajectory_position.get();
    request.trajectory_velocity = group.trajectory_velocity.get();
    request.trajectory_acceleration = group.trajectory_acceleration.get();
    request.trajectory_jerk = group.trajectory_jerk.get();
    request.success = group.success.get();
    request.used_rrt = group.used_rrt.get();
    request.rrt_iterations = group.rrt_iterations.get();
    request.cost = group.cost.get();
    request.collision_cost = group.collision.get();
    request.trajectory_dt = group.trajectory_dt.get();
    request.allow_rrt = false;
    request.finetune_attempts = settings.finetune;
    group.planner->plan(request);
    check_cuda(cudaDeviceSynchronize());
  }
}

// 优化失败后的 PRM 关节路径。RRT 走求解器内部，不进这个函数。
PlanResult NativePlanner::plan_prm(Group &group, const JointMap &state,
                                   const std::vector<float> &goal) {
  // 起点与终点先验收，端点无效时无需生成路网。
  const auto start = positions(group, state);
  PlanResult result;
  const auto endpoint_failure = [&](const std::vector<float> &position) -> std::string {
    // Keep the GPU acceptance criterion, then explain a rejection using the exported bounds.
    if (gpu_valid(group, {position})) return {};
    const float tolerance = group.planner->config().trajectory_optimization.constraint_tolerance;
    for (int j = 0; j < group.dof; ++j) {
      const float value = position[j];
      const auto &joint = group.names[j];
      if (!std::isfinite(value)) return "关节 " + joint + " 角度无效";
      if (value < group.lower[j] - tolerance || value > group.upper[j] + tolerance) {
        return "关节 " + joint + " 超限，实际角度 " + std::to_string(value) +
               " 弧度，允许范围为 " + std::to_string(group.lower[j]) + " 至 " +
               std::to_string(group.upper[j]) + " 弧度";
      }
    }
    return "关节角都在允许范围内，但 GPU 检查失败。可能发生了碰撞，也可能是数值异常";
  };
  for (const auto &[label, position] : {std::pair{"起点", &start}, std::pair{"目标", &goal}}) {
    const auto failure = endpoint_failure(*position);
    if (!failure.empty()) {
      result.message = std::string("PRM ") + label + "检查失败。" + failure + "。";
      return result;
    }
  }
  constexpr int random_samples = 180;
  std::vector<std::vector<float>> nodes{start, goal};
  std::mt19937 random(137);
  for (int i = 0; i < random_samples; ++i) {
    std::vector<float> q;
    for (const auto &name : group.names) {
      const auto bounds = robot_.limits.at(name);
      q.push_back(std::uniform_real_distribution<float>(bounds.first, bounds.second)(random));
    }
    nodes.push_back(std::move(q));
  }
  std::vector<std::vector<float>> valid{start, goal};
  for (size_t i = 2; i < nodes.size(); ++i)
    if (gpu_valid(group, {nodes[i]})) valid.push_back(std::move(nodes[i]));
  nodes = std::move(valid);
  const auto distance = [](const auto &a, const auto &b) {
    double value = 0.;
    for (size_t i = 0; i < a.size(); ++i) value += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(value);
  };
  std::vector<double> costs(nodes.size(), std::numeric_limits<double>::infinity());
  std::vector<int> parents(nodes.size(), -1);
  std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<>>
      queue;
  costs[0] = 0.;
  queue.emplace(0., 0);
  size_t checked_edges = 0;
  size_t blocked_edges = 0;
  while (!queue.empty()) {
    auto [cost, node] = queue.top();
    queue.pop();
    if (cost != costs[node]) continue;
    if (node == 1) break;
    std::vector<std::pair<double, int>> nearest;
    for (size_t i = 0; i < nodes.size(); ++i)
      if (i != static_cast<size_t>(node)) nearest.emplace_back(distance(nodes[node], nodes[i]), i);
    std::sort(nearest.begin(), nearest.end());
    for (size_t k = 0; k < std::min<size_t>(16, nearest.size()); ++k) {
      const auto [length, next] = nearest[k];
      if (cost + length >= costs[next]) continue;
      const int count = std::max(2, static_cast<int>(std::ceil(length / .025)) + 1);
      std::vector<std::vector<float>> edge;
      for (int i = 0; i < count; ++i) {
        std::vector<float> q;
        for (int j = 0; j < group.dof; ++j)
          q.push_back(nodes[node][j] +
                      static_cast<float>(i) / (count - 1) * (nodes[next][j] - nodes[node][j]));
        edge.push_back(std::move(q));
      }
      ++checked_edges;
      if (!gpu_valid(group, edge)) {
        ++blocked_edges;
        continue;
      }
      costs[next] = cost + length;
      parents[next] = node;
      queue.emplace(costs[next], next);
    }
  }
  if (parents[1] < 0) {
    const auto logger = rclcpp::get_logger("curobo_motion_planning_node");
    RCLCPP_DEBUG(logger, "【OPT 回退】PRM 采样了 %d 个关节姿态，其中 %zu 个通过检查。"
                         "尝试连接了 %zu 次，其中 %zu 次未通过检查。",
                 random_samples, nodes.size() - 2, checked_edges, blocked_edges);
    result.message = "PRM 没能找到从起点到目标的可行路径。";
    return result;
  }
  std::vector<std::vector<float>> path;
  for (int node = 1; node >= 0; node = parents[node]) path.insert(path.begin(), nodes[node]);
  std::vector<Trajectory> segments;
  JointMap current = state;
  for (size_t i = 1; i < path.size(); ++i) {
    auto part = solve(group, current, {path[i]}, 1, false, false);
    if (!part.success) {
      result.message = "PRM 找到了由 " + std::to_string(path.size() - 1) +
                       " 段组成的路径，但第 " + std::to_string(i) +
                       " 段优化失败。" + part.message;
      return result;
    }
    for (int j = 0; j < group.dof; ++j)
      current[group.names[j]] = part.trajectory.points.back().positions[j];
    segments.push_back(std::move(part.trajectory));
  }
  try {
    result.trajectory = segments.size() == 1
                            ? segments.front()
                            : blend_trajectories(segments, group.names, dynamic_limits(group.name),
                                                 std::vector<bool>(segments.size(), false))
                                  .trajectory;
  } catch (const std::runtime_error &error) {
    result.message = std::string("PRM 找到了路径，但没能把各段轨迹拼接起来。") + error.what();
    return result;
  }
  try {
    validate(group.name, state, result.trajectory);
  } catch (const std::runtime_error &error) {
    result.message = std::string("PRM 拼接后的整条轨迹未通过检查。") + error.what();
    result.trajectory = Trajectory{};
    return result;
  }
  result.success = true;
  result.message = "PRM 回退成功，轨迹已通过验收";
  return result;
}
}  // namespace curobo_ros_cpp
