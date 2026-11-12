// SPDX-License-Identifier: Apache-2.0
// 处理跟随 Action，检查输入新鲜度并协调规划周期和轨迹交接。
#include <algorithm>
#include <cmath>
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
  try {
    if (execution->active()) throw std::runtime_error("execution busy");
    execution->clear();
    owns_execution = true;
    planner->reset_mpc();
    while (!cancelled()) {
      const auto tick = Clock::now();
      if (timed_out()) throw std::runtime_error("MPC timeout reached");
      interfaces::msg::MpcTarget::SharedPtr target;
      {
        std::lock_guard<std::mutex> guard(input_mutex_);
        target = target_;
      }
      if (!target) {
        if (elapsed_ms(started) >= target_max_age * 1000.)
          throw std::runtime_error("MPC target missing");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      const auto desired_group = target_group(target);
      if (desired_group.empty()) throw std::runtime_error("MPC target cleared");
      if (group != desired_group) {
        group = desired_group;
        reset = true;
        accepted = Trajectory{};
      }
      check();
      if (reset && execution->active()) {
        execution->cancel();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      if (target->has_left != target->has_right) {
        const double period = parameter<double>("mpc_missing_target_warn_period", 2.);
        if (period > 0. && elapsed_ms(missing_warned) >= period * 1000.) {
          missing_warned = Clock::now();
          RCLCPP_WARN(get_logger(), "【跟随】只有一侧目标，缺的那侧保持当前末端");
        }
      }
      execution->prepare(group, stopped);
      JointMap joints = robot->defaults;
      JointMap velocities;
      {
        std::lock_guard<std::mutex> guard(input_mutex_);
        for (const auto &name : robot->profile.groups.at(group))
          if (const auto failure = joint_failure(name)) throw std::runtime_error(failure);
        for (const auto &joint : joints_) joints[joint.first] = joint.second;
        velocities = joint_velocity_;
      }
      std::map<std::string, Transform> targets;
      bool valid = true;
      std::string error;
      try {
        if (target->has_left) targets[robot->profile.tools.at("left_arm")] = in_base(target->left);
        if (target->has_right)
          targets[robot->profile.tools.at("right_arm")] = in_base(target->right);
        update_world(group, joints);
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
          const int knot =
              std::max(1, static_cast<int>(std::ceil((elapsed + kReserve) / knot_dt)));
          handoff_time = knot * knot_dt;
          if (handoff_time >= seconds(accepted.points.back().time_from_start) ||
              knot > parameter<int>("mpc_num_control_points", 6)) {
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
        check();
        const auto proposal = planner->mpc_step(group, joints, targets, handoff, shift, reset,
                                                parameter<double>("mpc_velocity_scale", 1.));
        check();
        {
          std::lock_guard<std::mutex> guard(input_mutex_);
          if (target_group(target_) != group) continue;
        }
        message.message = proposal.message;
        message.solve_time = proposal.elapsed_ms / 1000.;
        if (proposal.success) {
          auto trajectory = proposal.trajectory;
          const auto origin = continuing ? accepted_origin + rclcpp::Duration(duration(handoff_time))
                                         : now() + rclcpp::Duration(duration(.2));
          const bool within_window = origin >= now() + rclcpp::Duration(duration(.05));
          if (!within_window || !execution->submit_mpc(group, trajectory, origin, stopped, continuing)) {
            reset = true;
            message.message = "MPC handoff missed or misaligned";
            RCLCPP_WARN(get_logger(), "%s, replanning from feedback", message.message.c_str());
          } else {
            check();
            trajectory.header.stamp = origin;
            message.header.stamp = origin;
            message.success = true;
            planner->accept_mpc(group);
            accepted = trajectory;
            accepted_origin = origin;
            knot_dt = proposal.knot_dt;
            reset = false;
            if (target->has_left)
              message.left.trajectory =
                  select_joints(trajectory, robot->profile.groups.at("left_arm"));
            if (target->has_right)
              message.right.trajectory =
                  select_joints(trajectory, robot->profile.groups.at("right_arm"));
            auto errors = [&](const std::string &side, auto &output) {
              auto found = targets.find(robot->profile.tools.at(side));
              if (found == targets.end()) return;
              const auto current = robot->tool(joints, side);
              output.position_error = (found->second.translation() - current.translation()).norm();
              output.orientation_error = Eigen::Quaterniond(found->second.linear())
                                             .angularDistance(Eigen::Quaterniond(current.linear()));
            };
            errors("left_arm", message.left);
            errors("right_arm", message.right);
            const auto position = goal->position_tolerance, orientation = goal->orientation_tolerance;
            bool reached = position > 0;
            for (const auto *side : {&message.left, &message.right})
              if (side->target_valid)
                reached = reached && side->position_error <= position &&
                          (orientation <= 0 || side->orientation_error <= orientation);
            if (reached) {
              result->success = true;
              result->message = "MPC reached target";
              mpc_pub_->publish(message);
              break;
            }
          }
        }
      } else {
        message.success = false;
        message.message = error;
      }
      mpc_pub_->publish(message);
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
  if (!rclcpp::ok()) return;
  if (handle->is_canceling())
    handle->canceled(result);
  else if (result->success)
    handle->succeed(result);
  else
    handle->abort(result);
}
}  // namespace curobo_ros_cpp
