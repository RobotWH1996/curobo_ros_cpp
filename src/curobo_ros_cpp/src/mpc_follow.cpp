// SPDX-License-Identifier: Apache-2.0
// 处理跟随 Action，检查输入新鲜度并协调规划周期和轨迹交接。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include "curobo_ros_cpp/node.hpp"
namespace curobo_ros_cpp {
// 按控制周期读取目标并推进 MPC。取消或超时后停止下发。
void PlanningNode::follow_mpc(
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
  double knot_dt = .14;
  int consumed_knots = 0;
  Clock::time_point missing_warned{};
  auto cancelled = [&] { return shutdown_ || !rclcpp::ok() || handle->is_canceling(); };
  const auto timed_out = [&] { return goal->timeout > 0. && elapsed_ms(started) >= goal->timeout * 1000.; };
  const double target_max_age = parameter<double>("mpc_target_max_age", 1.);
  const double joint_max_age = parameter<double>("mpc_joint_state_max_age", 1.);
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
    for (const auto &name : robot->profile.groups.at(group))
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
  std::vector<double> last_joint_target;
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
    constexpr double kMaxJointAdvance = .45;  // rad per prediction horizon
    RCLCPP_INFO(get_logger(),
                "【MPC 跟随 · 开始】Action /follow_mpc ｜ 最短周期 %.0f ms ｜ 速度倍率 %.3f ｜ 模式 %s",
                parameter<double>("mpc_control_period", .1) * 1000.,
                execution_scale, execution->enabled() ? "控制器执行" : "仅预览");
    while (!cancelled()) {
      const auto tick = Clock::now();
      if (timed_out()) throw std::runtime_error("MPC timeout reached");
      interfaces::msg::MpcTarget::SharedPtr target;
      {
        std::lock_guard<std::mutex> guard(input_mutex_);
        if (target_ && (elapsed_ms(target_time_) < target_max_age * 1000. || !group.empty()))
          target = target_;
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
        last_joint_target.clear();
        reset = true;
        accepted = Trajectory{};
        consumed_knots = 0;
      }
      check();
      if (reset && execution->active()) {
        execution->cancel();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      // Wait for the accepted segment to start before scheduling another handoff.
      // Otherwise every solve advances the origin even while the arm is still on the old prefix.
      if (!reset && now() < accepted_origin) {
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
        for (const auto &name : robot->profile.groups.at(group))
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
          logged_targets[tool] = pose;
          needs_goal_seed = track_joint_goal;
          RCLCPP_INFO(get_logger(),
                      "【MPC 跟随 · 目标】周期 %llu ｜ 组 %s ｜ %s @ %s ｜ xyz=(%.3f, %.3f, %.3f) m ｜ "
                      "xyzw=(%.3f, %.3f, %.3f, %.3f)",
                      static_cast<unsigned long long>(cycle), group.c_str(), tool.c_str(), base.c_str(),
                      pose.translation().x(), pose.translation().y(), pose.translation().z(),
                      rotation.x(), rotation.y(), rotation.z(), rotation.w());
        }
        const auto scene_started = Clock::now();
        update_world(group, joints);
        scene_ms = elapsed_ms(scene_started);
      } catch (const std::exception &exception) {
        valid = false;
        error = exception.what();
      }
      interfaces::msg::MpcPlan message;
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
          const int knot = std::max(
              {1, consumed_knots, static_cast<int>(std::ceil((elapsed + kReserve) / knot_dt))});
          handoff_time = knot * knot_dt;
          if (handoff_time >= seconds(accepted.points.back().time_from_start) ||
              knot > parameter<int>("mpc_num_control_points", 6)) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                                 "【MPC 跟随 · 续接】旧轨迹已无交接窗口，下一拍从关节反馈冷启动");
            reset = true;
            continue;
          } else {
            handoff = sample_trajectory(accepted, handoff_time);
            shift = knot;
            continuing = true;
          }
        }
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
        // The native solver shifts its seed even when it rejects the proposal.
        seed_shift = continuing ? shift - consumed_knots : 0;
        start_source = continuing ? "轨迹交接" : "关节反馈";
        std::vector<double> seed_goal;
        if (track_joint_goal && (needs_goal_seed || (goal_joints.empty() && reseed_after_failure))) {
          auto handoff_joints = joints;
          for (size_t j = 0; j < names.size(); ++j)
            handoff_joints[names[j]] = handoff.position[j];
          const auto ik = planner->inverse_kinematics(group, handoff_joints, targets, true);
          goal_joints = ik.success ? ik.trajectory.points.back().positions : std::vector<double>{};
        }
        // Give the joint-space horizon a reachable endpoint along the IK direction.
        // The full MPC collision and dynamic-limit checks still apply to each candidate.
        if (!goal_joints.empty()) {
          const Eigen::Map<const Eigen::VectorXd> goal_q(goal_joints.data(), goal_joints.size());
          const Eigen::VectorXd delta = goal_q - handoff.position;
          const double fraction =
              kMaxJointAdvance / std::max(kMaxJointAdvance, delta.cwiseAbs().maxCoeff());
          const Eigen::VectorXd endpoint = handoff.position + fraction * delta;
          seed_goal.assign(endpoint.data(), endpoint.data() + endpoint.size());
        }
        const bool joint_goal = !seed_goal.empty();
        const bool change_mode_or_goal = last_joint_target != seed_goal;
        const bool initialize_seed = reset || change_mode_or_goal;
        const auto proposal = planner->mpc_step(group, joints, targets, handoff,
                                                initialize_seed ? 0 : seed_shift,
                                                initialize_seed, execution_scale, seed_goal, joint_goal);
        last_joint_target = std::move(seed_goal);
        needs_goal_seed = false;
        reseed_after_failure = track_joint_goal && !proposal.success;
        consumed_knots = shift;
        check();
        {
          std::lock_guard<std::mutex> guard(input_mutex_);
          if (target_group(target_) != group) continue;
        }
        message.message = proposal.message;
        message.solve_time = proposal.elapsed_ms / 1000.;
        if (proposal.success) {
          points = proposal.trajectory.points.size();
          const auto preview_started = Clock::now();
          publish_mpc_preview(group, joints, proposal.trajectory, targets);
          preview_ms = elapsed_ms(preview_started);
          auto trajectory = proposal.trajectory;
          const auto origin = continuing ? accepted_origin + rclcpp::Duration(duration(handoff_time))
                                         : now() + rclcpp::Duration(duration(.2));
          const auto submit_started = Clock::now();
          lead_ms = (origin - now()).seconds() * 1000.;
          const bool within_window = origin >= now() + rclcpp::Duration(duration(.05));
          if (!within_window || !execution->submit_mpc(group, trajectory, origin, stopped, continuing)) {
            reset = true;
            message.message = "MPC handoff missed or misaligned";
            dispatch = within_window ? "续接未通过" : "提交窗口不足";
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                                 "【MPC 跟随 · 续接】%s ｜ 提前量 %.1f ms ｜ 下一拍重新对齐",
                                 dispatch, lead_ms);
          } else {
            dispatch = execution->enabled() ? "控制器已接收" : "仅预览";
            check();
            trajectory.header.stamp = origin;
            message.header.stamp = origin;
            message.success = true;
            planner->accept_mpc(group);
            accepted = trajectory;
            accepted_origin = origin;
            consumed_knots = 0;
            knot_dt = proposal.knot_dt;
            reset = false;
            if (target->has_left)
              message.left.trajectory =
                  select_joints(trajectory, robot->profile.groups.at("left_arm"));
            if (target->has_right)
              message.right.trajectory =
                  select_joints(trajectory, robot->profile.groups.at("right_arm"));
            const auto position = goal->position_tolerance, orientation = goal->orientation_tolerance;
            bool reached = position > 0;
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
      mpc_pub_->publish(message);
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
    if (rclcpp::ok()) {
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
  if (!rclcpp::ok()) return;
  if (handle->is_canceling())
    handle->canceled(result);
  else if (result->success)
    handle->succeed(result);
  else
    handle->abort(result);
}
}  // namespace curobo_ros_cpp
