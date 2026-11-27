// SPDX-License-Identifier: Apache-2.0
// 处理跟随 Action，检查输入新鲜度并协调规划周期和轨迹交接。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <Eigen/Cholesky>
#include "curobo_ros_cpp/mpc_detour.hpp"
#include "curobo_ros_cpp/node.hpp"
namespace curobo_ros_cpp {
namespace {
constexpr double kTargetPositionJump = .08;  // m
constexpr double kTargetRotationJump = .30;  // rad
constexpr double kSwitchLead = .16;          // s
constexpr double kMaxJointAdvance = .45;     // rad per prediction horizon
constexpr double kRecoveryAfterMs = 200.;
constexpr double kRecoveryAbortAfterMs = 600.;
constexpr double kRecoveryLead = .12;     // s
constexpr double kRecoveryHold = 2.;        // s
constexpr double kRestartReserve = .40;     // s, cold solve while stationary
constexpr unsigned kDetourAfterBrakes = 2;
constexpr double kObstacleLookahead = .45;
constexpr double kObstacleBrakeLead = .09;
constexpr double kMpcSubmitLead = .05;

std::vector<double> route_goal(const Trajectory &route, size_t &cursor,
                               const Eigen::VectorXd &current) {
  const auto gap = [&](size_t i) {
    const auto &q = route.points[i].positions;
    return (Eigen::Map<const Eigen::VectorXd>(q.data(), q.size()) - current)
        .cwiseAbs().maxCoeff();
  };
  const size_t last = route.points.size() - 1;
  while (cursor < last && gap(cursor) < .045) ++cursor;
  if (cursor == last && gap(last) < .03) return {};
  size_t lookahead = cursor;
  while (lookahead < last && gap(lookahead + 1) < .12) ++lookahead;
  return route.points[lookahead].positions;
}

double rotation_distance(const Transform &a, const Transform &b) {
  return Eigen::Quaterniond(a.linear()).angularDistance(Eigen::Quaterniond(b.linear()));
}

bool target_changed(const geometry_msgs::msg::PoseStamped &a,
                    const geometry_msgs::msg::PoseStamped &b) {
  if (a.header.frame_id != b.header.frame_id) return true;
  try {
    const auto first = transform(a.pose);
    const auto second = transform(b.pose);
    return (first.translation() - second.translation()).norm() >= kTargetPositionJump ||
           rotation_distance(first, second) >= kTargetRotationJump;
  } catch (const std::exception &) {
    return true;
  }
}

trajectory_msgs::msg::JointTrajectoryPoint packed_sample(const TrajectorySample &sample,
                                                        double time) {
  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.positions.assign(sample.position.data(), sample.position.data() + sample.position.size());
  point.velocities.assign(sample.velocity.data(), sample.velocity.data() + sample.velocity.size());
  point.accelerations.assign(sample.acceleration.data(),
                             sample.acceleration.data() + sample.acceleration.size());
  point.time_from_start = duration(time);
  return point;
}

Trajectory stop_trajectory(const std::vector<std::string> &names,
                           const TrajectorySample &start, double stop_time,
                           const Eigen::VectorXd &shift = {}) {
  // 积分三次速度曲线，保持初始位置、速度和加速度，末端速度与加速度均为零。
  Trajectory trajectory;
  trajectory.joint_names = names;
  Eigen::VectorXd end = start.position + start.velocity * (stop_time / 2.) +
                        start.acceleration * (stop_time * stop_time / 12.);
  if (shift.size()) end += shift;
  trajectory.points.resize(2);
  trajectory.points.front() = packed_sample(start, 0.);
  auto &last = trajectory.points.back();
  last.positions.assign(end.data(), end.data() + end.size());
  last.velocities.assign(names.size(), 0.);
  last.accelerations.assign(names.size(), 0.);
  last.time_from_start = duration(stop_time);
  return trajectory;
}

Eigen::Vector3d inward(const Eigen::Vector3d &start, const Eigen::Vector3d &goal) {
  const Eigen::Vector3d lateral = Eigen::Vector3d::UnitZ().cross(goal - start);
  if (lateral.norm() < 1e-6) return Eigen::Vector3d::Zero();
  return (lateral.dot(start) > 0. ? -1. : 1.) * lateral.normalized();
}

PlanResult compact_detour(NativePlanner &planner, const RobotModel &robot,
                          const std::string &group, const JointMap &start_state,
                          const std::map<std::string, Transform> &targets, double pitch,
                          double execution_scale, Clock::time_point deadline,
                          bool allow_fixed_offset) {
  PlanResult result;
  if (targets.size() != 1) return result;
  const auto &[tool, goal] = *targets.begin();
  const auto poses = robot.forward(start_state);
  const auto &start = poses.at(tool);
  const Eigen::Vector3d delta = goal.translation() - start.translation();
  if (delta.head<2>().norm() < 4. * pitch || std::abs(delta.z()) > 2. * pitch) return result;
  const Eigen::Vector3d offset = 2. * pitch * inward(start.translation(), goal.translation()) +
                                 4. * pitch * Eigen::Vector3d::UnitZ();
  const std::array<Eigen::Vector3d, 2> fixed_points{start.translation() + offset,
                                                   goal.translation() + offset};
  const double dynamics_scale = std::min(planner.settings.velocity_scale, execution_scale);
  const auto plan_route = [&](const std::array<Eigen::Vector3d, 2> &points) {
    PlanResult route;
    auto first = start;
    auto second = goal;
    first.translation() = points[0];
    second.translation() = points[1];
    auto current = start_state;
    std::vector<Trajectory> parts;
    for (const auto &pose : {first, second, goal}) {
      if (Clock::now() >= deadline) {
        route.message = "MPC compact detour ran out of stationary planning time";
        return route;
      }
      auto part = planner.plan_pose(group, current, {{tool, pose}}, 1);
      if (!part.success) return part;
      for (size_t i = 0; i < part.trajectory.joint_names.size(); ++i)
        current[part.trajectory.joint_names[i]] = part.trajectory.points.back().positions[i];
      parts.push_back(std::move(part.trajectory));
    }
    try {
      route.trajectory = blend_trajectories(parts, parts.front().joint_names,
          planner.dynamic_limits(group, dynamics_scale), {false, false, false}).trajectory;
      planner.validate(group, start_state, route.trajectory, true, dynamics_scale);
      route.success = true;
    } catch (const std::runtime_error &error) {
      route.message = error.what();
    }
    return route;
  };
  if (planner.payloads().empty() && rotation_distance(start, goal) <= .10) {
    const auto points = suggest_mpc_detour(start.translation(), goal.translation(),
        robot.gripper_spheres(group, poses), planner.world_points(), pitch,
        planner.settings.mpc_collision_activation_distance);
    if (points) {
      const bool fixed = (*points)[0].isApprox(fixed_points[0]) &&
                         (*points)[1].isApprox(fixed_points[1]);
      const auto started = Clock::now();
      result = plan_route(*points);
      if (result.success) {
        result.message = fixed ? "fixed-offset compact detour" : "scene-guided compact detour";
        return result;
      }
      if (fixed || !allow_fixed_offset || Clock::now() + (Clock::now() - started) >= deadline)
        return result;
    }
  }
  if (!allow_fixed_offset) return result;
  auto fixed = plan_route(fixed_points);
  if (fixed.success) fixed.message = "fixed-offset compact detour";
  return fixed;
}

std::map<std::string, Transform> intermediate_targets(
    const std::map<std::string, Transform> &starts,
    const std::map<std::string, Transform> &targets, double scale) {
  auto waypoints = targets;
  for (auto &[tool, pose] : waypoints) {
    const auto &start = starts.at(tool);
    const double distance = (pose.translation() - start.translation()).norm();
    const double angle = rotation_distance(start, pose);
    const double fraction =
        scale / std::max({1., distance / .24, angle / .65});
    pose.translation() = start.translation() + fraction * (pose.translation() - start.translation());
    pose.linear() = Eigen::Quaterniond(start.linear())
                        .slerp(fraction, Eigen::Quaterniond(pose.linear())).toRotationMatrix();
  }
  return waypoints;
}

bool moving_away(const RobotModel &robot, JointMap state, const Trajectory &trajectory,
                 double handoff_time, const std::map<std::string, Transform> &targets) {
  const auto start = robot.forward(state);
  const double end = seconds(trajectory.points.back().time_from_start);
  const auto next = sample_trajectory(trajectory, std::min(end, handoff_time + .18));
  for (size_t j = 0; j < trajectory.joint_names.size(); ++j)
    state[trajectory.joint_names[j]] = next.position[j];
  const auto poses = robot.forward(state);
  for (const auto &[tool, goal] : targets) {
    const double position_change = (poses.at(tool).translation() - goal.translation()).norm() -
                                   (start.at(tool).translation() - goal.translation()).norm();
    const double rotation_change = rotation_distance(poses.at(tool), goal) -
                                   rotation_distance(start.at(tool), goal);
    if (position_change > .005 || (position_change >= -.003 && rotation_change > .045))
      return true;
  }
  return false;
}

bool advances_toward_target(const RobotModel &robot, JointMap state, const Trajectory &trajectory,
                            const std::map<std::string, Transform> &targets) {
  const auto start = robot.forward(state);
  constexpr double kAllowedBacktrack = .023;  // m
  constexpr double kRotationWeight = .08;      // m/rad
  const auto end = seconds(trajectory.points.back().time_from_start);
  const double lookahead = std::min(end, .85);
  bool advances = false;
  for (int step = 1; step <= 5; ++step) {
    const auto sample = sample_trajectory(trajectory, lookahead * step / 5.);
    for (size_t j = 0; j < trajectory.joint_names.size(); ++j)
      state[trajectory.joint_names[j]] = sample.position[j];
    const auto poses = robot.forward(state);
    for (const auto &[tool, goal] : targets) {
      const double initial_distance = (start.at(tool).translation() - goal.translation()).norm();
      const double distance = (poses.at(tool).translation() - goal.translation()).norm();
      if (distance > initial_distance + kAllowedBacktrack) return false;
      const double gain = initial_distance - distance +
          kRotationWeight * (rotation_distance(start.at(tool), goal) -
                             rotation_distance(poses.at(tool), goal));
      if (step == 5 && gain > .001) advances = true;
    }
  }
  return advances;
}
}  // namespace

// 按控制周期读取目标并推进 MPC。取消或超时后停止下发。
std::shared_ptr<interfaces::action::FollowMpc::Result> PlanningNode::follow_mpc(
    std::shared_ptr<rclcpp_action::ServerGoalHandle<interfaces::action::FollowMpc>> handle,
    Clock::time_point started) {
  struct RestoreMode {
    std::string &mode;
    std::string previous;
    ~RestoreMode() { mode.swap(previous); }
  } restore{cloud_mode_, cloud_mode_};
  cloud_mode_ = "mesh";
  auto result = std::make_shared<interfaces::action::FollowMpc::Result>();
  const auto goal = handle->get_goal();
  bool reset = true;
  std::string group;
  Trajectory accepted;
  rclcpp::Time accepted_origin(0, 0, get_clock()->get_clock_type());
  struct QueuedPrefix {
    rclcpp::Time origin;
    Trajectory path;
  };
  std::deque<QueuedPrefix> queued;
  std::optional<rclcpp::Time> reviewed_prefix_end;
  sensor_msgs::msg::PointCloud2::SharedPtr processed_cloud;
  auto accepted_at = started;
  double knot_dt = .14;
  int consumed_knots = 0;
  Clock::time_point missing_warned{};
  auto cancelled = [&] { return shutdown_ || !rclcpp::ok() || handle->is_canceling(); };
  const auto timed_out = [&] { return goal->timeout > 0. && elapsed_ms(started) >= goal->timeout * 1000.; };
  const double target_max_age = parameter<double>("mpc_target_max_age", 1.);
  const double joint_max_age = parameter<double>("mpc_joint_state_max_age", 1.);
  const double cloud_max_age_ms = parameter<double>("pick_place_cloud_max_age", 2.) * 1000.;
  bool scene_ready = false;
  if (!std::isfinite(target_max_age) || target_max_age <= 0. ||
      !std::isfinite(joint_max_age) || joint_max_age <= 0.)
    throw std::invalid_argument("MPC input max ages must be positive and finite");
  auto target_group = [](const auto &target) -> std::string {
    if (target->has_left && target->has_right) return "both_arms";
    if (target->has_left) return "left_arm";
    if (target->has_right) return "right_arm";
    return {};
  };
  auto joint_failure = [&](const std::string &name) -> const char * {
    const auto received = joint_received_.find(name);
    if (received == joint_received_.end() || !std::isfinite(joints_.at(name)))
      return "MPC joint feedback missing or invalid";
    if (elapsed_ms(received->second) >= joint_max_age * 1000.)
      return "MPC joint feedback expired";
    return nullptr;
  };
  auto stop_reason = [&]() -> const char * {
    if (cancelled()) return "MPC cancelled";
    if (timed_out()) return "MPC timeout reached";
    std::lock_guard<std::mutex> guard(input_mutex_);
    if (!target_ || elapsed_ms(target_time_) >= target_max_age * 1000.)
      return "MPC target expired";
    if (!target_->has_left && !target_->has_right) return "MPC target cleared";
    if (scene_ready && pointcloud_mode_ != PointCloudMode::Off && !cloud_is_fresh(cloud_max_age_ms))
      return "MPC pointcloud expired or unavailable";
    for (const auto &name : planning_joints_)
      if (const auto failure = joint_failure(name)) return failure;
    return nullptr;
  };
  const auto stopped = [&] { return stop_reason() != nullptr; };
  const auto check = [&] {
    if (const auto failure = stop_reason()) throw std::runtime_error(failure);
  };
  bool owns_execution = false;
  uint64_t cycles = 0, submitted = 0, rejected = 0;
  std::map<std::string, Transform> logged_targets;
  auto last_summary = started;
  bool needs_goal_seed = false;
  bool reseed_after_failure = false;
  std::vector<double> goal_joints;
  std::optional<rclcpp::Time> brake_end;
  bool brake_pending = false;
  interfaces::msg::MpcTarget::SharedPtr accepted_target;
  bool checking_direction = false;
  bool initialize_next_seed = false;
  double guide_scale = 0.;
  double joint_advance = kMaxJointAdvance;
  unsigned recovery_brakes = 0;
  bool obstacle_replan = false;
  bool detour_attempted = false;
  Trajectory detour;
  std::optional<rclcpp::Time> compact_origin;
  size_t detour_cursor = 0;
  auto targets_differ = [&](const auto &first, const auto &second) {
    if (!first || !second || target_group(first) != target_group(second)) return true;
    return (first->has_left && target_changed(first->left, second->left)) ||
           (first->has_right && target_changed(first->right, second->right));
  };
  auto target_superseded = [&](const auto &target) {
    std::lock_guard<std::mutex> guard(input_mutex_);
    return targets_differ(target, target_);
  };
  try {
    if (execution->active()) throw std::runtime_error("execution busy");
    execution->clear();
    owns_execution = true;
    planner->reset_mpc();
    const double velocity_scale = parameter<double>("mpc_velocity_scale", 1.);
    const double follow_scale = parameter<double>("mpc_follow_velocity_scale", 1.);
    if (!std::isfinite(follow_scale) || follow_scale <= 0. || follow_scale > 1.)
      throw std::invalid_argument("mpc_follow_velocity_scale must be in (0, 1]");
    // Keep time scaling fixed across handoffs so derivatives stay consistent.
    const double execution_scale = velocity_scale * follow_scale;
    const bool track_joint_goal = parameter<bool>("mpc_follow_joint_goal", false);
    RCLCPP_INFO(get_logger(),
                "【MPC 跟随 · 开始】Action /follow_mpc ｜ 最短周期 %.0f ms ｜ 速度倍率 %.3f ｜ 模式 %s",
                parameter<double>("mpc_control_period", .1) * 1000.,
                execution_scale, execution->enabled() ? "控制器执行" : "仅预览");
    while (!cancelled()) {
      const auto tick = Clock::now();
      if (timed_out()) throw std::runtime_error("MPC timeout reached");
      interfaces::msg::MpcTarget::SharedPtr target;
      sensor_msgs::msg::PointCloud2::SharedPtr cloud;
      {
        std::lock_guard<std::mutex> guard(input_mutex_);
        if (target_ && (elapsed_ms(target_time_) < target_max_age * 1000. || !group.empty()))
          target = target_;
        cloud = cloud_;
      }
      if (!target) {
        if (elapsed_ms(started) >= target_max_age * 1000.)
          throw std::runtime_error("MPC fresh target missing at startup");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      const auto desired_group = target_group(target);
      if (desired_group.empty()) throw std::runtime_error("MPC target cleared");
      if (group != desired_group) {
        group = desired_group;
        logged_targets.clear();
        needs_goal_seed = false;
        reseed_after_failure = false;
        goal_joints.clear();
        brake_end.reset();
        brake_pending = false;
        accepted_target.reset();
        checking_direction = false;
        initialize_next_seed = false;
        guide_scale = 0.;
        joint_advance = kMaxJointAdvance;
        recovery_brakes = 0;
        obstacle_replan = false;
        detour_attempted = false;
        detour = Trajectory{};
        compact_origin.reset();
        detour_cursor = 0;
        reset = true;
        accepted = Trajectory{};
        queued.clear();
        reviewed_prefix_end.reset();
        consumed_knots = 0;
      }
      check();
      if (reset && execution->active()) {
        execution->cancel();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      // 普通求解等最新轨迹开始，新点云可提前检查已排队的区间。
      const auto now_before_scene = now();
      if (now_before_scene >= accepted_origin) queued.clear();
      while (queued.size() > 1 && queued[1].origin <= now_before_scene) queued.pop_front();
      const bool waiting = !reset && now_before_scene < accepted_origin &&
                           !targets_differ(accepted_target, target);
      if (waiting && cloud == processed_cloud) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
      }
      const auto cycle = ++cycles;
      if (target->has_left != target->has_right) {
        const double period = parameter<double>("mpc_missing_target_warn_period", 2.);
        if (period > 0. && elapsed_ms(missing_warned) >= period * 1000.) {
          missing_warned = Clock::now();
          RCLCPP_WARN(get_logger(), "【跟随】只有一侧目标，缺的那侧保持当前末端");
        }
      }
      execution->prepare(group, stopped);
      const auto feedback_started = Clock::now();
      JointMap joints = robot->defaults;
      JointMap velocities;
      {
        std::lock_guard<std::mutex> guard(input_mutex_);
        for (const auto &name : planning_joints_)
          if (const auto failure = joint_failure(name)) throw std::runtime_error(failure);
        for (const auto &joint : joints_) joints[joint.first] = joint.second;
        velocities = joint_velocity_;
      }
      const double feedback_ms = elapsed_ms(feedback_started);
      const auto target_started = Clock::now();
      double tf_ms = 0., scene_ms = 0., preview_ms = 0., submit_ms = 0., lead_ms = 0.;
      int seed_shift = 0;
      const char *start_source = "关节反馈";
      const char *dispatch = "未提交";
      size_t points = 0;
      std::map<std::string, Transform> targets;
      bool valid = true;
      bool scene_changed = false;
      std::string error;
      try {
        if (target->has_left) targets[robot->profile.tools.at("left_arm")] = in_base(target->left);
        if (target->has_right)
          targets[robot->profile.tools.at("right_arm")] = in_base(target->right);
        tf_ms = elapsed_ms(target_started);
        for (const auto &[tool, pose] : targets) {
          const Eigen::Quaterniond rotation(pose.linear());
          const auto last = logged_targets.find(tool);
          if (last != logged_targets.end() &&
              (pose.translation() - last->second.translation()).norm() < .005 &&
              rotation.angularDistance(Eigen::Quaterniond(last->second.linear())) < .02)
            continue;
          if (last != logged_targets.end() &&
              ((pose.translation() - last->second.translation()).norm() >= kTargetPositionJump ||
               rotation_distance(pose, last->second) >= kTargetRotationJump)) {
            brake_pending = !reset && !accepted.points.empty() && !brake_end;
            checking_direction = true;
            initialize_next_seed = true;
            guide_scale = 0.;
            joint_advance = kMaxJointAdvance;
            recovery_brakes = 0;
            obstacle_replan = false;
            detour_attempted = false;
            detour = Trajectory{};
            compact_origin.reset();
            detour_cursor = 0;
            RCLCPP_INFO(get_logger(), "【MPC 跟随 · 切换】重新选择交接和运动方向");
          }
          logged_targets[tool] = pose;
          needs_goal_seed = track_joint_goal;
          if (checking_direction) goal_joints.clear();
          RCLCPP_INFO(get_logger(),
                      "【MPC 跟随 · 目标】周期 %llu ｜ 组 %s ｜ %s @ %s ｜ xyz=(%.3f, %.3f, %.3f) m ｜ "
                      "xyzw=(%.3f, %.3f, %.3f, %.3f)",
                      static_cast<unsigned long long>(cycle), group.c_str(), tool.c_str(), base.c_str(),
                      pose.translation().x(), pose.translation().y(), pose.translation().z(),
                      rotation.x(), rotation.y(), rotation.z(), rotation.w());
        }
        const auto scene_started = Clock::now();
        const auto generation = planner->world_generation();
        update_world(group, CloudFreshness::Latest, /*require_cloud=*/true);
        scene_ready = true;
        processed_cloud = cloud;
        scene_changed = planner->world_generation() != generation;
        scene_ms = elapsed_ms(scene_started);
      } catch (const std::exception &exception) {
        if (!reset && !accepted.points.empty()) throw;
        valid = false;
        error = exception.what();
      }
      interfaces::msg::MpcPlan message;
      bool published = false;
      message.header.frame_id = base;
      message.header.stamp = now();
      message.left.target_valid = target->has_left;
      message.right.target_valid = target->has_right;
      if (target->has_left) message.left.used_target = target->left;
      if (target->has_right) message.right.used_target = target->right;
      if (valid) {
        TrajectorySample handoff;
        const auto &names = robot->profile.groups.at(group);
        handoff.position = Eigen::VectorXd(names.size());
        handoff.velocity = Eigen::VectorXd::Zero(names.size());
        handoff.acceleration = Eigen::VectorXd::Zero(names.size());
        handoff.jerk = Eigen::VectorXd::Zero(names.size());
        for (size_t j = 0; j < names.size(); ++j) {
          handoff.position[j] = joints.at(names[j]);
          const auto value = velocities.find(names[j]);
          if (value != velocities.end() && std::isfinite(value->second))
            handoff.velocity[j] = value->second;
        }
        int shift = 0;
        bool continuing = false;
        double handoff_time = 0.;
        constexpr double kReserve = .25;
        if (!reset && !accepted.points.empty()) {
          const double elapsed = (now() - accepted_origin).seconds();
          if (brake_pending) {
            handoff_time = std::max(.001, elapsed + kSwitchLead);
          } else if (brake_end) {
            handoff_time =
                std::max(elapsed + kRestartReserve, (*brake_end - accepted_origin).seconds());
          } else {
            shift = std::max(
                {1, consumed_knots, static_cast<int>(std::ceil((elapsed + kReserve) / knot_dt))});
            handoff_time = shift * knot_dt;
          }
          if (handoff_time >= seconds(accepted.points.back().time_from_start) ||
              shift > parameter<int>("mpc_num_control_points", 6))
            throw std::runtime_error("MPC no safe continuation window; stopping stream");
          handoff = sample_trajectory(accepted, handoff_time);
          continuing = true;
        }
        JointMap handoff_joints = joints;
        for (size_t j = 0; j < names.size(); ++j)
          handoff_joints[names[j]] = handoff.position[j];
        planner->settings.mpc_posture_weight = parameter<double>("mpc_posture_weight", 5.);
        planner->settings.mpc_posture = parameter<std::string>("mpc_posture", "l");
        planner->settings.mpc_run_ik = parameter<bool>("mpc_run_ik", false);
        const auto errors = [&](const std::string &side, auto &output) {
          const auto found = targets.find(robot->profile.tools.at(side));
          if (found == targets.end()) return;
          const auto current = robot->tool(joints, side);
          output.position_error = (found->second.translation() - current.translation()).norm();
          output.orientation_error = Eigen::Quaterniond(found->second.linear())
                                         .angularDistance(Eigen::Quaterniond(current.linear()));
        };
        errors("left_arm", message.left);
        errors("right_arm", message.right);
        check();
        MpcProposal proposal;
        bool braking = false;
        std::optional<rclcpp::Time> obstacle_origin;
        std::optional<rclcpp::Time> reference_origin;
        bool brake_rebased = false;
        double obstacle_brake_ms = 0.;
        const auto prefix_safe = [&](const rclcpp::Time &stamp, const rclcpp::Time &until) {
          const auto safe_interval = [&](const Trajectory &path, const rclcpp::Time &origin,
                                         const rclcpp::Time &end) {
            const double first = std::max(0., (stamp - origin).seconds());
            const double last = std::min((end - origin).seconds(),
                                         seconds(path.points.back().time_from_start));
            return last <= first ||
                planner->validate_mpc_continuation(group, path, first, last, execution_scale);
          };
          for (size_t i = 0; i < queued.size(); ++i) {
            const auto &end = i + 1 < queued.size() ? queued[i + 1].origin : accepted_origin;
            if (!safe_interval(queued[i].path, queued[i].origin, std::min(until, end)))
              return false;
          }
          return safe_interval(accepted, accepted_origin, until);
        };
        if (continuing && scene_changed) {
          planner->synchronize(group, joints, true);
          const auto stamp = now();
          const auto review_end = std::max(stamp + rclcpp::Duration(duration(kObstacleLookahead)),
                                          accepted_origin + rclcpp::Duration(duration(handoff_time)));
          if (prefix_safe(stamp, review_end)) {
            reviewed_prefix_end = review_end;
          } else {
            const auto brake_started = Clock::now();
            const auto first_origin = queued.empty() || execution->enabled()
                ? accepted_origin : queued.front().origin;
            const auto splice_after = first_origin + rclcpp::Duration(duration(.01));
            const auto brake_lead = rclcpp::Duration(duration(kObstacleBrakeLead));
            const auto submit_lead = rclcpp::Duration(duration(kMpcSubmitLead));
            auto start_at = std::max(stamp + brake_lead, splice_after);
            for (int attempt = 0; attempt < 2; ++attempt) {
              check();
              if (attempt == 1) {
                start_at = std::max(now() + brake_lead, splice_after);
                // A shifted timestamp alone would invalidate the seam and the unchecked prefix.
                if (start_at >= review_end)
                  throw std::runtime_error("MPC braking splice is outside the reviewed queue window");
                brake_rebased = true;
              }
              const Trajectory *source = &accepted;
              auto source_origin = accepted_origin;
              if (start_at < accepted_origin) {
                for (auto it = queued.rbegin(); it != queued.rend(); ++it) {
                  if (it->origin <= start_at) {
                    source = &it->path;
                    source_origin = it->origin;
                    break;
                  }
                }
              }
              const double seam = (start_at - source_origin).seconds();
              if (seam < 0. || seam >= seconds(source->points.back().time_from_start))
                throw std::runtime_error("MPC queued trajectory ends before the earliest braking splice");
              if (!prefix_safe(stamp, start_at))
                throw std::runtime_error("MPC queued prefix failed safety validation before the earliest braking splice");
              if (start_at < now() + submit_lead) continue;
              const auto start = sample_trajectory(*source, seam);
              const auto find_brake = [&](const Eigen::VectorXd &offset) -> Trajectory {
                for (const double stop_time : {.25, .32, .40, .50, .60}) {
                  auto stop = stop_trajectory(names, start, stop_time, offset);
                  if (!planner->validate_mpc_continuation(group, stop, 0., stop_time, execution_scale))
                    continue;
                  append_hold(stop, kRecoveryHold);
                  return stop;
                }
                return Trajectory{};
              };
              auto stop = find_brake({});
              if (stop.points.empty() && targets.size() == 1) {
                const auto &[tool, goal] = *targets.begin();
                JointMap before = joints;
                for (size_t j = 0; j < names.size(); ++j) before[names[j]] = start.position[j];
                const auto pose = robot->forward(before).at(tool);
                const auto jac = planner->jacobian(group, before, {tool});
                Eigen::VectorXd twist = Eigen::VectorXd::Zero(6);
                const double pitch = cloud_settings_.pitch;
                twist.head<3>() = pitch * inward(pose.translation(), goal.translation()) +
                                  3. * pitch * Eigen::Vector3d::UnitZ();
                const Eigen::MatrixXd regularized = jac * jac.transpose() +
                    .0025 * Eigen::MatrixXd::Identity(6, 6);
                const Eigen::VectorXd lifted = jac.transpose() * regularized.ldlt().solve(twist);
                for (double fraction : {1., .5}) {
                  stop = find_brake(fraction * lifted);
                  if (!stop.points.empty()) break;
                }
              }
              if (stop.points.empty())
                throw std::runtime_error("MPC no collision-free brake after obstacle update");
              if (start_at < now() + submit_lead) continue;
              proposal.trajectory = std::move(stop);
              proposal.message = "MPC recovery braking";
              proposal.success = true;
              braking = true;
              obstacle_origin = start_at;
              reviewed_prefix_end = start_at;
              start_source = "新障碍制动";
              break;
            }
            if (!obstacle_origin)
              throw std::runtime_error("MPC obstacle brake missed the splice deadline; stopping stream");
            obstacle_brake_ms = elapsed_ms(brake_started);
          }
        }
        if (waiting && !obstacle_origin) continue;
        if (obstacle_origin) {
          proposal.elapsed_ms = obstacle_brake_ms;
        } else if (brake_pending && continuing &&
            (handoff.velocity.cwiseAbs().maxCoeff() > .025 ||
             handoff.acceleration.cwiseAbs().maxCoeff() > .12) &&
            moving_away(*robot, handoff_joints, accepted, handoff_time, targets)) {
          // 优先下发通过当前动态和碰撞检查的最短制动段。
          for (const double stop_time : {.30, .36, .40, .45, .50, .60, .70, .95}) {
            auto trajectory = stop_trajectory(names, handoff, stop_time);
            try {
              planner->validate(group, joints, trajectory, true, execution_scale);
            } catch (const std::runtime_error &error) {
              proposal.message = error.what();
              continue;
            }
            append_hold(trajectory, 1.25);
            proposal.success = true;
            proposal.message = "MPC target switch braking";
            proposal.trajectory = std::move(trajectory);
            braking = true;
            break;
          }
          if (!braking)
            throw std::runtime_error("MPC target switch could not brake: " + proposal.message);
        } else {
          if (brake_pending && continuing) {
            // 本拍未制动，预留正常求解时间，并用新交接状态规划。
            handoff_time += kReserve - kSwitchLead;
            if (handoff_time >= seconds(accepted.points.back().time_from_start))
              throw std::runtime_error("MPC no safe retarget window; stopping stream");
            handoff = sample_trajectory(accepted, handoff_time);
            for (size_t j = 0; j < names.size(); ++j)
              handoff_joints[names[j]] = handoff.position[j];
          }
          const bool force_seed = initialize_next_seed || brake_end || brake_pending;
          brake_pending = false;
          seed_shift = continuing && !force_seed ? shift - consumed_knots : 0;
          start_source = continuing ? "轨迹交接" : "关节反馈";
          const bool can_search = brake_end ||
              (!continuing && handoff.velocity.cwiseAbs().maxCoeff() <= .025);
          if (can_search && (obstacle_replan || recovery_brakes >= kDetourAfterBrakes) &&
              !detour_attempted) {
            detour_attempted = true;
            const auto graph_start = Clock::now();
            PlanResult route;
            // A blocked future horizon can trigger recovery braking before the obstacle enters the queued prefix.
            if (brake_end) {
              const double remaining = seconds(accepted.points.back().time_from_start) -
                                       (now() - accepted_origin).seconds() - kRestartReserve;
              const auto deadline = graph_start + std::chrono::duration_cast<Clock::duration>(
                  std::chrono::duration<double>(remaining));
              route = compact_detour(*planner, *robot, group, handoff_joints, targets,
                                     cloud_settings_.pitch, execution_scale, deadline, obstacle_replan);
            }
            const bool compact = route.success;
            if (!compact && !route.message.empty())
              RCLCPP_INFO(get_logger(), "【MPC 跟随 · 绕行】紧凑路线未通过 ｜ %s", route.message.c_str());
            if (!compact && (!brake_end || (now() - accepted_origin).seconds() + kRestartReserve <
                                          seconds(accepted.points.back().time_from_start)))
              route = planner->plan_pose(group, handoff_joints, targets, 1, true);
            check();
            if (target_superseded(target)) continue;
            if (brake_end) {
              handoff_time = std::max((now() - accepted_origin).seconds() + kRestartReserve,
                                      (*brake_end - accepted_origin).seconds());
              if (handoff_time >= seconds(accepted.points.back().time_from_start))
                throw std::runtime_error("MPC detour search exhausted the stationary handoff window");
              handoff = sample_trajectory(accepted, handoff_time);
            }
            if (route.success) {
              if (!compact) planner->validate(group, handoff_joints, route.trajectory);
              detour = std::move(route.trajectory);
              if (compact) {
                compact_origin = continuing ? accepted_origin + rclcpp::Duration(duration(handoff_time))
                                            : now() + rclcpp::Duration(duration(.25));
                append_hold(detour, .45);
              }
              RCLCPP_INFO(get_logger(), "【MPC 跟随 · 绕行】%s ｜ %zu 点 ｜ %.1f ms",
                          compact ? route.message.c_str() : "图搜索", detour.points.size(),
                          elapsed_ms(graph_start));
            } else {
              RCLCPP_WARN(get_logger(), "【MPC 跟随 · 绕行】搜索未通过 ｜ %s", route.message.c_str());
            }
          }
          if (compact_origin) {
            const auto lead = now() + rclcpp::Duration(duration(.16));
            const auto available = continuing
                ? std::max(lead, accepted_origin + rclcpp::Duration(duration(.001))) : lead;
            const auto from = std::lower_bound(detour.points.begin(), detour.points.end(),
                (available - *compact_origin).seconds(),
                [](const auto &point, double t) { return seconds(point.time_from_start) < t; });
            if (from != detour.points.end() && from + 1 != detour.points.end()) {
              const double offset = seconds(from->time_from_start);
              const auto until = std::upper_bound(from + 2, detour.points.end(), offset + 1.0,
                  [](double t, const auto &point) { return t < seconds(point.time_from_start); });
              const auto origin = *compact_origin + rclcpp::Duration(from->time_from_start);
              if (!continuing || origin < accepted_origin + rclcpp::Duration(accepted.points.back().time_from_start)) {
                auto &piece = proposal.trajectory;
                piece.joint_names = detour.joint_names;
                piece.points.assign(from, until);
                for (auto &point : piece.points)
                  point.time_from_start = duration(seconds(point.time_from_start) - offset);
                if (planner->validate_mpc_continuation(
                        group, piece, 0., seconds(piece.points.back().time_from_start), execution_scale)) {
                  reference_origin = origin;
                  proposal.success = true;
                  proposal.message = "MPC compact bypass";
                  start_source = "紧凑绕行";
                }
              }
            }
            if (!reference_origin) compact_origin.reset();
          }
          if (!reference_origin) {
            auto route_joints = detour.points.empty() ? std::vector<double>{}
                : route_goal(detour, detour_cursor, handoff.position);
            const bool following_detour = !route_joints.empty();
            if (!following_detour && !detour.points.empty()) {
              detour = Trajectory{};
              goal_joints.clear();
              needs_goal_seed = track_joint_goal;
            }
            const auto handoff_poses = checking_direction ? robot->forward(handoff_joints)
                                                         : std::map<std::string, Transform>{};
            if (checking_direction &&
                std::all_of(targets.begin(), targets.end(), [&](const auto &item) {
                  const auto &pose = handoff_poses.at(item.first);
                  return (pose.translation() - item.second.translation()).norm() < .035 &&
                         rotation_distance(pose, item.second) < .18;
                })) {
              checking_direction = false;
              guide_scale = 0.;
              goal_joints.clear();
              needs_goal_seed = track_joint_goal;
            }

            const bool guided = checking_direction && guide_scale > 0.;
            const auto solve_targets = guided ? intermediate_targets(handoff_poses, targets, guide_scale)
                                              : targets;
            if (!following_detour && track_joint_goal && (needs_goal_seed || guided ||
                                     (goal_joints.empty() && reseed_after_failure))) {
              const auto ik = planner->inverse_kinematics(group, handoff_joints, solve_targets, true,
                                                          checking_direction);
              goal_joints = ik.success ? ik.trajectory.points.back().positions : std::vector<double>{};
            }
            std::vector<double> seed_goal;
            const auto &guide_joints = following_detour ? route_joints : goal_joints;
            if (!guide_joints.empty()) {
              const Eigen::Map<const Eigen::VectorXd> goal_q(guide_joints.data(), guide_joints.size());
              const Eigen::VectorXd delta = goal_q - handoff.position;
              const double fraction =
                  joint_advance / std::max(joint_advance, delta.cwiseAbs().maxCoeff());
              const Eigen::VectorXd endpoint = handoff.position + fraction * delta;
              seed_goal.assign(endpoint.data(), endpoint.data() + endpoint.size());
            }
            const bool joint_goal = !seed_goal.empty();
            const bool initialize_seed = reset || force_seed;
            proposal = planner->mpc_step(group, joints, solve_targets, handoff,
                                         initialize_seed ? 0 : seed_shift,
                                         initialize_seed, execution_scale, seed_goal, joint_goal);
            initialize_next_seed = false;
            needs_goal_seed = false;
            reseed_after_failure = track_joint_goal && !proposal.success;
            consumed_knots = shift;
            if (following_detour && proposal.success)
              proposal.message = "MPC graph-guided";
            const bool wrong_direction = !following_detour && checking_direction && proposal.success &&
                !advances_toward_target(*robot, handoff_joints, proposal.trajectory, targets);
            if (wrong_direction) {
              proposal.success = false;
              proposal.message = "MPC retarget: predicted motion moves away or makes no progress";
            }
            if (checking_direction && !proposal.success) {
              if (wrong_direction || guided) {
                guide_scale = guided && guide_scale > .26 ? guide_scale * .5 : 1.;
                if (guided && guide_scale == 1.) guide_scale = 0.;
                needs_goal_seed = track_joint_goal;
              }
              joint_advance = joint_advance > .19 ? joint_advance * .67 : kMaxJointAdvance;
              initialize_next_seed = true;
            }
          }
        }
        if (!proposal.success && !continuing) ++recovery_brakes;
        // Fill an imminent stream gap with a validated stop on the old time
        // axis. The prefix still queued before handoff must also be clear.
        if (!proposal.success && continuing && elapsed_ms(accepted_at) >= kRecoveryAfterMs) {
          const double elapsed = std::max(0., (now() - accepted_origin).seconds());
          const double recovery_time = elapsed + kRecoveryLead;
          if (recovery_time < seconds(accepted.points.back().time_from_start)) {
            if (!prefix_safe(now(), accepted_origin + rclcpp::Duration(duration(recovery_time))))
              throw std::runtime_error("MPC queued prefix failed safety validation; stopping stream; "
                                       "new prediction: " + proposal.message);
            const auto start = sample_trajectory(accepted, recovery_time);
            for (const auto stop_time : {.30, .45, .55}) {
              auto stop = stop_trajectory(names, start, stop_time);
              if (!planner->validate_mpc_continuation(
                      group, stop, 0., seconds(stop.points.back().time_from_start), execution_scale))
                continue;
              append_hold(stop, kRecoveryHold);
              proposal.trajectory = std::move(stop);
              proposal.message = "MPC recovery braking";
              proposal.success = true;
              handoff_time = recovery_time;
              start_source = "断档制动";
              braking = true;
              break;
            }
          }
          if (!braking && elapsed_ms(accepted_at) >= kRecoveryAbortAfterMs)
            throw std::runtime_error("MPC could not validate a braking segment; stopping stream");
        }
        check();
        if (!obstacle_origin && target_superseded(target)) {
          initialize_next_seed = true;
          RCLCPP_DEBUG(get_logger(), "【MPC 跟随 · 切换】丢弃已变更目标的求解结果");
          continue;
        }
        message.message = proposal.message;
        message.solve_time = proposal.elapsed_ms / 1000.;
        if (proposal.success) {
          points = proposal.trajectory.points.size();
          const auto preview_started = Clock::now();
          if (!obstacle_origin) publish_mpc_preview(group, joints, proposal.trajectory, targets);
          preview_ms = elapsed_ms(preview_started);
          auto trajectory = proposal.trajectory;
          const auto origin = obstacle_origin ? *obstacle_origin
              : reference_origin ? *reference_origin
              : continuing ? accepted_origin + rclcpp::Duration(duration(handoff_time))
                           : now() + rclcpp::Duration(duration(.2));
          const auto submit_started = Clock::now();
          if (!obstacle_origin && target_superseded(target)) {
            initialize_next_seed = true;
            continue;
          }
          // A later splice can outlive the last obstacle review, even on a tick with no new cloud.
          if (continuing && reviewed_prefix_end && origin > *reviewed_prefix_end) {
            planner->synchronize(group, joints, true);
            if (!prefix_safe(now(), origin))
              throw std::runtime_error("MPC queued prefix failed safety validation up to the new handoff");
            reviewed_prefix_end = origin;
          }
          lead_ms = (origin - now()).seconds() * 1000.;
          const bool within_window = origin >= now() + rclcpp::Duration(duration(kMpcSubmitLead));
          if (!within_window || !execution->submit_mpc(group, trajectory, origin, stopped, continuing)) {
            if (obstacle_origin)
              throw std::runtime_error("MPC obstacle brake missed the splice deadline; stopping stream");
            reset = !continuing;
            message.message = "MPC handoff missed or misaligned";
            dispatch = within_window ? "续接未通过" : "提交窗口不足";
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                                 "【MPC 跟随 · 续接】%s ｜ 提前量 %.1f ms",
                                 dispatch, lead_ms);
          } else {
            dispatch = execution->enabled() ? "控制器已接收" : "仅预览";
            check();
            trajectory.header.stamp = origin;
            message.header.stamp = origin;
            message.success = true;
            if (braking) {
              brake_end = origin + rclcpp::Duration(trajectory.points[1].time_from_start);
              if (proposal.message == "MPC recovery braking") ++recovery_brakes;
              if (obstacle_origin) {
                obstacle_replan = true;
                detour_attempted = false;
                detour = Trajectory{};
                compact_origin.reset();
                detour_cursor = 0;
              }
              brake_pending = false;
              initialize_next_seed = true;
              if (!obstacle_origin)
                RCLCPP_INFO(get_logger(), "【MPC 跟随 · 切换】已提交减速段 ｜ %.0f ms",
                            seconds(trajectory.points[1].time_from_start) * 1000.);
            } else {
              if (!reference_origin) planner->accept_mpc(group);
              brake_end.reset();
              recovery_brakes = 0;
              obstacle_replan = false;
              if (reference_origin)
                initialize_next_seed = true;
              else
                knot_dt = proposal.knot_dt;
              if (checking_direction) {
                // 新方向已通过检查。下一拍回到最终目标，避免每拍都重算局部路点。
                if (guide_scale > 0.) needs_goal_seed = track_joint_goal;
                checking_direction = false;
                guide_scale = 0.;
                joint_advance = kMaxJointAdvance;
              }
            }
            while (!queued.empty() && queued.back().origin >= origin) queued.pop_back();
            if (!accepted.points.empty() && accepted_origin < origin)
              queued.push_back({accepted_origin, std::move(accepted)});
            accepted = trajectory;
            accepted_target = target;
            accepted_origin = origin;
            reviewed_prefix_end.reset();
            accepted_at = Clock::now();
            consumed_knots = 0;
            reset = false;
            if (target->has_left)
              message.left.trajectory =
                  select_joints(trajectory, robot->profile.groups.at("left_arm"));
            if (target->has_right)
              message.right.trajectory =
                  select_joints(trajectory, robot->profile.groups.at("right_arm"));
            if (obstacle_origin) {
              lead_ms = (origin - now()).seconds() * 1000.;
              if (!execution->enabled() && lead_ms < kMpcSubmitLead * 1000.)
                throw std::runtime_error("MPC obstacle brake missed the publish deadline; stopping stream");
              if (!execution->enabled()) {
                mpc_pub_->publish(message);
                published = true;
              }
              RCLCPP_INFO(get_logger(),
                          "【MPC 跟随 · 新障碍】已发布制动 ｜ 求解 %.1f ms ｜ 提前量 %.1f ms ｜ 重选接续 %s",
                          obstacle_brake_ms, lead_ms, brake_rebased ? "是" : "否");
              const auto preview = Clock::now();
              publish_mpc_preview(group, joints, trajectory, targets);
              preview_ms = elapsed_ms(preview);
            }
            const auto position = goal->position_tolerance, orientation = goal->orientation_tolerance;
            bool reached = !braking && position > 0;
            for (const auto *side : {&message.left, &message.right})
              if (side->target_valid)
                reached = reached && side->position_error <= position &&
                          (orientation <= 0 || side->orientation_error <= orientation);
            if (reached) {
              result->success = true;
              result->message = "MPC reached target";
            }
          }
          submit_ms = elapsed_ms(submit_started);
        }
      } else {
        message.success = false;
        message.message = error;
      }
      if (message.success)
        ++submitted;
      else
        ++rejected;
      RCLCPP_DEBUG(get_logger(),
                   "【MPC 跟随 · 逐拍】周期 %llu ｜ %s ｜ 起点 %s ｜ 种子推进 %d ｜ "
                   "关节 %.1f ms ｜ TF %.1f ms ｜ 场景 %.1f ms ｜ 求解 %.1f ms ｜ "
                   "预览 %.1f ms ｜ 交接与提交 %.1f ms ｜ 本拍 %.1f ms ｜ 轨迹 %zu 点 ｜ "
                   "交接提前 %.1f ms ｜ %s ｜ %s",
                   static_cast<unsigned long long>(cycle), group.c_str(), start_source, seed_shift,
                   feedback_ms, tf_ms, scene_ms, message.solve_time * 1000., preview_ms, submit_ms,
                   elapsed_ms(tick), points, lead_ms, dispatch, message.message.c_str());
      if (cycle == 1 || elapsed_ms(last_summary) >= 1000.) {
        last_summary = Clock::now();
        RCLCPP_INFO(get_logger(),
                    "【MPC 跟随 · 进度】周期 %llu ｜ %s ｜ 累计可用 %llu / 未通过 %llu ｜ "
                    "位置误差左/右 %.3f/%.3f m ｜ 求解 %.1f ms ｜ %s ｜ %s",
                    static_cast<unsigned long long>(cycle), group.c_str(),
                    static_cast<unsigned long long>(submitted), static_cast<unsigned long long>(rejected),
                    message.left.position_error, message.right.position_error, message.solve_time * 1000.,
                    dispatch, message.message.c_str());
      }
      if (!published && !execution->enabled()) mpc_pub_->publish(message);
      if (result->success) break;
      auto feedback = std::make_shared<interfaces::action::FollowMpc::Feedback>();
      feedback->left_target_valid = message.left.target_valid;
      feedback->right_target_valid = message.right.target_valid;
      feedback->left_position_error = message.left.position_error;
      feedback->right_position_error = message.right.position_error;
      feedback->left_orientation_error = message.left.orientation_error;
      feedback->right_orientation_error = message.right.orientation_error;
      feedback->solve_time = message.solve_time;
      feedback->plan_success = message.success;
      feedback->message = message.message;
      handle->publish_feedback(feedback);
      const auto remaining = parameter<double>("mpc_control_period", .1) * 1000. - elapsed_ms(tick);
      const auto next_tick = Clock::now() + std::chrono::duration<double, std::milli>(std::max(0., remaining));
      while (Clock::now() < next_tick && !stopped())
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  } catch (const std::exception &error) {
    result->message = error.what();
    if (!group.empty())
      if (const auto failure = stop_reason()) result->message = failure;
  }
  if (cancelled()) {
    result->success = false;
    result->message = "MPC cancelled";
  }
  if (!result->success && owns_execution) {
    execution->cancel();
    if (rclcpp::ok() && !execution->enabled()) {
      interfaces::msg::MpcPlan stopped_plan;
      stopped_plan.header.stamp = now();
      stopped_plan.header.frame_id = base;
      stopped_plan.message = result->message;
      mpc_pub_->publish(stopped_plan);
    }
  }
  RCLCPP_INFO(get_logger(),
              "【MPC 跟随 · 结束】组 %s ｜ 可用 %llu / 未通过 %llu ｜ 总耗时 %.1f ms ｜ %s",
              group.empty() ? "-" : group.c_str(), static_cast<unsigned long long>(submitted),
              static_cast<unsigned long long>(rejected), elapsed_ms(started), result->message.c_str());
  if (!rclcpp::ok()) return nullptr;
  return result;
}
}  // namespace curobo_ros_cpp
