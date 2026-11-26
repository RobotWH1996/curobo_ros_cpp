// SPDX-License-Identifier: Apache-2.0
// 编排抓放阶段，合并并提交轨迹块，汇总规划和执行结果。
#include <algorithm>
#include "curobo_ros_cpp/node.hpp"
namespace curobo_ros_cpp {
namespace {
// 切出 [start, end] 上的轨迹，时间改成从 0 开始。
Trajectory interval(const Trajectory &trajectory, double start, double end) {
  Trajectory result;
  result.joint_names = trajectory.joint_names;
  TrajectorySampler sampler(trajectory);
  auto add = [&](double time) {
    const auto &value = sampler.sample(time);
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.time_from_start = duration(time - start);
    point.positions = {value.position.data(), value.position.data() + value.position.size()};
    point.velocities = {value.velocity.data(), value.velocity.data() + value.velocity.size()};
    point.accelerations = {value.acceleration.data(),
                           value.acceleration.data() + value.acceleration.size()};
    result.points.push_back(std::move(point));
  };
  add(start);
  for (const auto &point : trajectory.points) {
    double time = seconds(point.time_from_start);
    if (time > start + 1e-8 && time < end - 1e-8) add(time);
  }
  add(end);
  return result;
}
Trajectory without_wheel_joints(const Trajectory &trajectory) {
  std::vector<std::string> names;
  for (const auto &name : trajectory.joint_names)
    if (name.rfind("Wheel", 0) != 0) names.push_back(name);
  return names.size() == trajectory.joint_names.size() ? trajectory :
                                                         select_joints(trajectory, names);
}
struct ContactPath {
  Transform start, end;
};
struct RestoreWorld {
  NativePlanner *planner;
  Payloads payloads;
  WorldState world;
  rclcpp::Logger logger;
  ~RestoreWorld() {
    try {
      planner->set_payloads(payloads);
      planner->restore_world(world);
    } catch (const std::exception &error) {
      RCLCPP_ERROR(logger, "Failed to restore planning scene: %s", error.what());
    } catch (...) {
      RCLCPP_ERROR(logger, "Unknown error restoring planning scene");
    }
  }
};
}  // namespace
// 抓取和放置都先规划非接触段，接触段用直线插补，再按夹爪停顿点分块提交。
TaskResult PlanningNode::manipulate(
    const interfaces::msg::PickTask *pick, const interfaces::msg::PlaceTask *place, double timeout,
    const std::function<bool()> &cancelled,
    const std::function<void(const std::string &, const std::string &)> &feedback,
    Clock::time_point started, const std::string &task_id, bool stop_at_pick) {
  TaskResult result;
  result.execution_requested = execution->enabled();
  const double task_timeout = timeout > 0. ? timeout : parameter<double>("pick_place_timeout", 180.);
  const auto timed_out = [&] { return elapsed_ms(started) >= task_timeout * 1000.; };
  const auto stopped = [&] { return cancelled() || timed_out(); };
  const char *task_name = pick && place ? "抓放" : pick ? "抓取" : "放置";
  std::string arm = "未知";
  bool opened = false;
  auto open_task = [&] {
    if (opened) return;
    RCLCPP_INFO(get_logger(), "──────── 任务开始 · %s · %s ────────", task_name, arm.c_str());
    opened = true;
  };
  auto close_task = [&](const char *status) {
    open_task();
    RCLCPP_INFO(get_logger(), "──────── 任务结束 · %s · %s ────────", task_name, status);
  };
  RestoreWorld restore{planner.get(), planner->payloads(), planner->world_state(), get_logger()};
  bool submitted = false;
  const bool external_stream = !execution->enabled() && !task_id.empty();
  uint32_t block_revision = 0;
  builtin_interfaces::msg::Time stream_stamp;
  auto send_block = [&](bool final, bool cancel) {
    if (!external_stream) return;
    if (block_revision == 0) stream_stamp = now();
    interfaces::msg::TrajectoryBlock message;
    message.task_id = task_id;
    message.block_index = ++block_revision;
    message.final = final;
    message.cancel = cancel;
    result.trajectory.header.frame_id = base;
    result.trajectory.header.stamp = stream_stamp;
    message.trajectory = result.trajectory;
    block_pub_->publish(message);
  };
  auto check = [&] {
    if (cancelled()) throw std::runtime_error("cancelled");
    if (timed_out()) throw std::runtime_error("Action timeout");
  };
  try {
    check();
    if (cloud_mode_ != "mesh")
      throw std::invalid_argument("pick/place requires pointcloud_collision=mesh");
    if (execution->active()) throw std::runtime_error("execution busy");
    execution->clear();
    const std::string group = pick ? pick->group_name : place->group_name;
    const auto held = std::find_if(held_payloads_.begin(), held_payloads_.end(),
                                   [&](const Payload &object) { return object.group == group; });
    if (!held_payloads_.empty() && (pick || held == held_payloads_.end()))
      throw std::invalid_argument("release the confirmed held object before changing the pick arm");
    arm = arm_text(group);
    open_task();
    if (group != "left_arm" && group != "right_arm")
      throw std::invalid_argument("pick/place requires left_arm or right_arm");
    if (pick && place && place->group_name != group)
      throw std::invalid_argument("pick/place groups differ");
    const auto gripper = robot->profile.grippers.at(group), tool = robot->profile.tools.at(group);
    const auto limits = robot->limits.at(gripper);
    auto validate_gripper = [&](double value) {
      if (!std::isfinite(value) || value < limits.first - 1e-8 || value > limits.second + 1e-8)
        throw std::invalid_argument("gripper target outside joint limits");
    };
    if (pick) {
      validate_gripper(pick->open_position);
      validate_gripper(pick->close_position);
      if (std::abs(pick->close_position - pick->open_position) < 1e-6)
        throw std::invalid_argument("pick open and close targets coincide");
    }
    if (place) validate_gripper(place->open_position);
    const auto names = robot->full_joint_names;
    Payload object;
    object.group = group;
    auto size = [&](const geometry_msgs::msg::Vector3 &input) {
      Eigen::Vector3d output(input.x, input.y, input.z);
      if (output.isZero()) {
        auto values =
            parameter<std::vector<double>>("pick_place_default_object_size", {0., 0., 0.});
        if (values.size() != 3)
          throw std::invalid_argument("default object size needs three values");
        output = Eigen::Vector3d(values[0], values[1], values[2]);
      }
      if (!output.allFinite() || (output.array() <= 0).any())
        throw std::invalid_argument("object size requires finite positive values");
      return output;
    };

    auto validate_waypoint = [&](const interfaces::msg::MotionWaypoint &waypoint, bool optional) {
      if (waypoint.type.empty() && optional) {
        if (!waypoint.joint_positions.empty() || !waypoint.pose.header.frame_id.empty())
          throw std::invalid_argument("end_pose target requires an explicit type");
        return;
      }
      if (waypoint.type == "joint") {
        const auto &joints = robot->profile.groups.at(group);
        if (waypoint.joint_positions.size() != joints.size())
          throw std::invalid_argument("joint waypoint size mismatch");
        for (size_t j = 0; j < joints.size(); ++j) {
          const auto bounds = robot->limits.at(joints[j]);
          const double value = waypoint.joint_positions[j];
          if (!std::isfinite(value) || value < bounds.first || value > bounds.second)
            throw std::invalid_argument("joint waypoint outside limits");
        }
        if (!waypoint.pose.header.frame_id.empty())
          throw std::invalid_argument("joint waypoint also contains a pose");
      } else if (waypoint.type == "pose") {
        if (!waypoint.joint_positions.empty())
          throw std::invalid_argument("pose waypoint also contains joint positions");
        in_base(waypoint.pose);
      } else
        throw std::invalid_argument("waypoint.type must be joint or pose");
    };
    auto separation = [](const Transform &a, const Transform &b) {
      const double length = (a.translation() - b.translation()).norm();
      if (length < .001 || length > .5)
        throw std::invalid_argument("contact separation must be between 1mm and 0.5m");
    };
    if (pick) {
      const auto reference = in_base(pick->tool_pose),
                 contact = compose(reference, pick->pick_pose, pick->in_base_frame);
      separation(contact, compose(reference, pick->prepick_in_target, pick->in_base_frame));
      separation(contact, compose(reference, pick->lift_in_target, pick->in_base_frame));
      for (const auto &waypoint : pick->waypoints) validate_waypoint(waypoint, false);
      validate_waypoint(pick->end_pose, true);
    }
    if (place) {
      const auto reference = in_base(place->tool_pose),
                 contact = compose(reference, place->place_pose, place->in_base_frame);
      separation(contact, compose(reference, place->preplace_in_target, place->in_base_frame));
      separation(contact, compose(reference, place->retreat_in_target, place->in_base_frame));
      for (const auto &waypoint : place->waypoints) validate_waypoint(waypoint, false);
      validate_waypoint(place->end_pose, true);
    }
    const auto cloud = select_cloud(CloudFreshness::WaitForPickPlace, true);
    check();
    const auto initial = state(true);
    auto current = initial;
    if (place && std::abs(place->open_position - (pick ? pick->close_position : initial.at(gripper))) < 1e-6)
      throw std::invalid_argument("place release equals initial gripping position");
    planner->synchronize(group, current);
    const auto dynamic = planner->dynamic_limits(group);
    if (pick) {
      const auto reference = in_base(pick->tool_pose);
      const auto grasp = compose(reference, pick->pick_pose, pick->in_base_frame);
      object.size = size(pick->object_size);
      object.in_tool = grasp.inverse() * reference;
      object.source_pose = reference;
    } else if (held != held_payloads_.end()) {
      object = *held;
    } else {
      const auto reference = in_base(place->tool_pose);
      const auto goal = compose(reference, place->place_pose, place->in_base_frame);
      object.size = size(place->object_size);
      object.in_tool = goal.inverse() * reference;
      object.source_pose = robot->tool(current, group) * object.in_tool;
      planner->set_payloads({object});
    }
    if (pick && !stop_at_pick) current[gripper] = pick->open_position;
    std::vector<std::pair<Transform, Eigen::Vector3d>> source_exclusions;
    const auto &frozen_cloud = prepare_world(group, cloud);
    // 抬升或撤退后，用冻结点云去掉已处理物体并刷新碰撞场景。
    auto refresh_world = [&]() {
      auto points = frozen_cloud.points;
      for (const auto &exclusion : source_exclusions)
        points = exclude_box(points, exclusion.first, exclusion.second);
      for (const auto &payload : planner->payloads())
        if (held_payloads_.empty()) points = exclude_box(points, payload.source_pose, payload.size);
      planner->set_world(points, cloud_settings_.pitch, cloud_mode_, voxel_center_,
                         voxel_dimensions_);
    };
    std::vector<Trajectory> pending, completed;
    std::vector<bool> contacts;
    std::vector<ContactPath> contact_paths;
    std::vector<JointMap> stage_states;
    std::vector<std::string> stages;
    double open_from = initial.at(gripper), open_to = pick ? pick->open_position : open_from;
    bool opening = pick && !stop_at_pick;
    auto set_stage = [&](const std::string &stage) {
      check();
      result.stage = stage;
      feedback(stage, "planning");
    };
    // 规划一段非接触 OPT 或接触直线，先攒着，到夹爪动作再合并。
    auto move = [&](const interfaces::msg::MotionWaypoint &waypoint, const std::string &stage,
                    bool contact, const std::optional<Transform> &target) {
      set_stage(stage);
      const auto before = Clock::now();
      PlanResult part;
      const Transform start_pose = contact ? robot->tool(current, group) : Transform::Identity();
      const auto before_state = current;
      if (target) publish_goals(stage, {{tool, *target}});
      if (target)
        part = contact ? planner->interpolate(group, current, start_pose, *target)
                       : planner->plan_pose(group, current, {{tool, *target}}, 3,
                                            /*rrt_on_first_attempt=*/stage == "prepick");
      else if (waypoint.type == "joint")
        part = planner->plan_joint(group, current, waypoint.joint_positions);
      else if (waypoint.type == "pose")
        part = planner->plan_pose(group, current, {{tool, in_base(waypoint.pose)}});
      else
        throw std::invalid_argument("waypoint.type must be joint or pose");
      if (!part.success) {
        RCLCPP_ERROR(get_logger(),
                     "【阶段失败】group=%s stage=%s ｜ 耗时 %.1f 毫秒 ｜ ik %.1f ｜ graph %.1f ｜ "
                     "trajopt %.1f ｜ fcl %.1f ｜ attempts %d ｜ %s",
                     group.c_str(), stage.c_str(), elapsed_ms(before), part.timing.ik_ms,
                     part.timing.graph_ms, part.timing.trajopt_ms, part.timing.check_ms,
                     part.timing.attempts, part.message.c_str());
        throw std::runtime_error(part.message);
      }
      check();
      auto full = expand_trajectory(part.trajectory, names, current);
      for (size_t i = 0; i < names.size(); ++i) current[names[i]] = full.points.back().positions[i];
      pending.push_back(std::move(full));
      contacts.push_back(contact);
      contact_paths.push_back({start_pose, contact ? *target : Transform::Identity()});
      stage_states.push_back(before_state);
      stages.push_back(stage);
      const auto label = stage_text(stage);
      RCLCPP_INFO(get_logger(),
                  "【阶段 · %s】耗时 %.1f 毫秒 ｜ ik %.1f 毫秒 ｜ graph %.1f 毫秒 ｜ trajopt %.1f 毫秒 ｜ "
                  "fcl %.1f 毫秒 ｜ attempts %d ｜ points %zu",
                  label.c_str(), elapsed_ms(before), part.timing.ik_ms, part.timing.graph_ms,
                  part.timing.trajopt_ms, part.timing.check_ms, part.timing.attempts,
                  part.trajectory.points.size());
    };
    // 合并已经规划的手臂段，并检查接触段仍沿直线。
    auto arm_block = [&]() {
      if (pending.empty()) return Trajectory{};
      const auto before = Clock::now();
      auto merged = blend_trajectories(pending, robot->profile.groups.at(group), dynamic, contacts);
      const double blend_ms = elapsed_ms(before);
      const size_t deadline_index =
          std::find(contacts.begin(), contacts.end(), true) - contacts.begin();
      if (opening) {
        const auto time = robot->gripper_duration(gripper, open_from, open_to);
        if (time > 0) {
          if (time > merged.boundaries[deadline_index]) {
            const auto extra = time - merged.boundaries[deadline_index];
            insert_hold(merged.trajectory, 0., extra);
            for (size_t i = 1; i < merged.boundaries.size(); ++i) merged.boundaries[i] += extra;
          }
          overlay_gripper(merged.trajectory, gripper, open_from, open_to, time);
        }
        opening = false;
      }
      double validation_ms = 0., contact_fk_ms = 0.;
      for (size_t i = 0; i < pending.size(); ++i) {
        const auto before_validation = Clock::now();
        auto slice = interval(merged.trajectory, merged.boundaries[i], merged.boundaries[i + 1]);
        planner->validate(group, stage_states[i], slice, !contacts[i]);
        validation_ms += elapsed_ms(before_validation);
        if (contacts[i]) {
          const auto before_fk = Clock::now();
          const auto &path = contact_paths[i];
          const Eigen::Vector3d delta = path.end.translation() - path.start.translation();
          const double length = delta.norm();
          if (length < 1e-6) throw std::runtime_error("contact segment too short");
          const auto direction = delta / length;
          const Eigen::Quaterniond a(path.start.linear()), b(path.end.linear());
          double previous = -1.;
          for (const auto &point : slice.points) {
            JointMap joints = stage_states[i];
            for (size_t j = 0; j < names.size(); ++j) joints[names[j]] = point.positions[j];
            const auto at = robot->tool(joints, group);
            const Eigen::Vector3d offset = at.translation() - path.start.translation();
            const double along = offset.dot(direction);
            const double lateral = planner->settings.path_tolerance > 0.
                                       ? planner->settings.path_tolerance
                                       : planner->settings.position_tolerance;
            if ((offset - along * direction).norm() > lateral ||
                along < -.005 || along > length + .005 || along < previous - .005 ||
                Eigen::Quaterniond(at.linear())
                        .angularDistance(a.slerp(std::clamp(along / length, 0., 1.), b)) >
                    planner->settings.orientation_tolerance + .005)
              throw std::runtime_error("merged contact path deviates from line");
            previous = along;
          }
          contact_fk_ms += elapsed_ms(before_fk);
        }
      }
      result.trajectories.push_back(without_wheel_joints(merged.trajectory));
      std::string labels, shown;
      for (const auto &stage : stages) {
        if (!labels.empty()) labels += "+";
        labels += stage;
        if (!shown.empty()) shown += " + ";
        shown += stage_text(stage);
      }
      result.trajectory_stages.push_back(labels);
      RCLCPP_INFO(get_logger(), "【合并】%s ｜ points %zu ｜ 耗时 %.1f 毫秒 ｜ 拼接重定时 %.1f "
                  "毫秒 ｜ 分段验收 %.1f 毫秒 ｜ 接触直线验收 %.1f 毫秒",
                  shown.c_str(), merged.trajectory.points.size(), elapsed_ms(before),
                  blend_ms, validation_ms, contact_fk_ms);
      pending.clear();
      contacts.clear();
      contact_paths.clear();
      stage_states.clear();
      stages.clear();
      return merged.trajectory;
    };
    // 把本块接到已提交轨迹末尾。控制器收到的是从任务起点开始的整段。
    auto commit = [&](Trajectory chunk) {
      if (chunk.points.empty()) return;
      check();
      completed.push_back(without_wheel_joints(chunk));
      result.trajectory = concatenate(completed);
      publish(group, initial, result.trajectory, !external_stream);
      if (external_stream) send_block(false, false);
      if (completed.size() == 1 && !stop_at_pick)
        pending_picks_.erase(std::remove_if(pending_picks_.begin(), pending_picks_.end(),
                                           [&](const Payload &item) { return item.group == group; }),
                             pending_picks_.end());
      execution->submit(group, result.trajectory, stopped);
      submitted = execution->enabled();
      RCLCPP_INFO(get_logger(), "【提交】第 %zu 块 ｜ 已等待 %.1f 毫秒", completed.size(),
                  elapsed_ms(started));
    };
    // 先提交手臂，再接上夹爪闭合或释放以及停留。
    auto close = [&](double target, const std::string &stage) {
      set_stage(stage);
      auto block = arm_block();
      const double time = robot->gripper_duration(gripper, current.at(gripper), target);
      Trajectory gripper_path;
      if (time > 0)
        gripper_path =
            gripper_trajectory(names, current, gripper, target, time, planner->settings.dt);
      else {
        gripper_path = gripper_trajectory(names, current, gripper, target, planner->settings.dt,
                                          planner->settings.dt);
      }
      append_hold(gripper_path, parameter<double>("gripper_wait_time", 1.));
      current[gripper] = target;
      result.trajectories.push_back(without_wheel_joints(gripper_path));
      result.trajectory_stages.push_back(stage);
      commit(block.points.empty() ? gripper_path : concatenate({block, gripper_path}));
    };
    interfaces::msg::MotionWaypoint empty;
    if (pick) {
      const auto reference = in_base(pick->tool_pose);
      const auto grasp = compose(reference, pick->pick_pose, pick->in_base_frame);
      const auto prepick = compose(reference, pick->prepick_in_target, pick->in_base_frame);
      const auto lift = compose(reference, pick->lift_in_target, pick->in_base_frame);
      if ((grasp.translation() - prepick.translation()).norm() < 1e-6 ||
          (grasp.translation() - lift.translation()).norm() < 1e-6)
        throw std::invalid_argument("pick approach/lift separation required");
      for (size_t i = 0; i < pick->waypoints.size(); ++i)
        move(pick->waypoints[i], "pick_via_" + std::to_string(i), false, {});
      move(empty, "prepick", false, prepick);
      move(empty, "pick", true, grasp);
      if (!stop_at_pick) {
        close(pick->close_position, "close");
        object.source_pose = robot->tool(current, group) * object.in_tool;
        planner->set_payloads({object});
        move(empty, "lift", true, lift);
        refresh_world();
        if (!pick->end_pose.type.empty()) move(pick->end_pose, "end", false, {});
      }
    }
    if (place) {
      const auto reference = in_base(place->tool_pose);
      const auto goal = compose(reference, place->place_pose, place->in_base_frame);
      const auto preplace = compose(reference, place->preplace_in_target, place->in_base_frame);
      const auto retreat = compose(reference, place->retreat_in_target, place->in_base_frame);
      if ((goal.translation() - preplace.translation()).norm() < 1e-6 ||
          (goal.translation() - retreat.translation()).norm() < 1e-6)
        throw std::invalid_argument("place approach/retreat separation required");
      for (size_t i = 0; i < place->waypoints.size(); ++i)
        move(place->waypoints[i], "place_via_" + std::to_string(i), false, {});
      move(empty, "preplace", false, preplace);
      move(empty, "place", true, goal);
      close(place->open_position, "release");
      if (pick) source_exclusions.emplace_back(object.source_pose, object.size);
      auto remaining = planner->payloads();
      remaining.erase(std::remove_if(remaining.begin(), remaining.end(),
                                     [&](const Payload &payload) { return payload.group == group; }),
                      remaining.end());
      planner->set_payloads(remaining);
      planner->add_placed_object(goal * object.in_tool, object.size);
      move(empty, "retreat", true, retreat);
      refresh_world();
      if (!place->end_pose.type.empty()) move(place->end_pose, "end", false, {});
    }
    auto remainder = arm_block();
    if (!remainder.points.empty()) commit(std::move(remainder));
    check();
    if (external_stream) send_block(true, false);
    const double planning_ms = elapsed_ms(started);
    if (submitted) {
      execution->wait(stopped);
      check();
      result.execution_completed = true;
    }
    result.success = true;
    if (pick && !place && !stop_at_pick) pending_picks_ = {object};

    result.message = "planning success";
    RCLCPP_INFO(get_logger(),
                "【任务】成功 ｜ 规划 %.1f 毫秒 ｜ 总耗时 %.1f 毫秒 ｜ points %zu ｜ 时长 %.3f 秒 ｜ 分块 %zu",
                planning_ms, elapsed_ms(started), result.trajectory.points.size(),
                seconds(result.trajectory.points.back().time_from_start), completed.size());
    close_task("成功");
  } catch (const std::exception &error) {
    result.message = cancelled() ? "Action cancelled" : timed_out() ? "Action timeout" : error.what();
    if (block_revision && external_stream) send_block(true, true);
    if (submitted) execution->cancel();
    const auto label = stage_text(result.stage);
    RCLCPP_ERROR(get_logger(), "【任务】失败 ｜ 阶段 %s ｜ 总耗时 %.1f 毫秒 ｜ 已提交 %s ｜ 错误：%s",
                 label.c_str(), elapsed_ms(started), submitted ? "是" : "否", result.message.c_str());
    close_task("失败");
  }
  return result;
}

interfaces::action::DualPick::Result PlanningNode::dual_pick(
    const interfaces::action::DualPick::Goal &goal, const std::string &task_id,
    const std::function<bool()> &stopped,
    const std::function<void(const std::string &, const std::string &, const std::string &)> &feedback) {
  interfaces::action::DualPick::Result result;
  const std::string group = "both_arms";
  const std::array<std::string, 2> sides{"left_arm", "right_arm"};
  const std::array<const interfaces::msg::PickTask *, 2> tasks{&goal.left, &goal.right};
  std::array<std::string, 2> grippers;
  std::array<double, 2> open{}, close{};
  std::map<std::string, Transform> prepicks, grasps, lifts;
  Payloads objects;
  auto check = [&] {
    if (stopped()) throw std::runtime_error("dual pick cancelled or timed out");
  };
  auto stage = [&](const std::string &name, const std::string &side) {
    result.stage = name;
    result.failed_group = side;
    check();
    feedback(name, side, "planning");
  };

  try {
    stage("prepare", group);
    if (cloud_mode_ != "mesh") throw std::invalid_argument("dual pick requires pointcloud_collision=mesh");
    if (execution->active() || (goal.publish_trajectory && execution->enabled()))
      throw std::invalid_argument("dual pick publishing requires an idle SDK bridge mode");
    if (!held_payloads_.empty()) throw std::invalid_argument("release carried objects before dual pick");
    if (goal.publish_trajectory && !pending_picks_.empty())
      throw std::invalid_argument("resolve pending held-object confirmation before publishing another dual pick");
    for (size_t i = 0; i < sides.size(); ++i) {
      const auto &side = sides[i];
      const auto &task = *tasks[i];
      result.failed_group = side;
      if (task.group_name != side || !task.waypoints.empty() || !task.end_pose.type.empty() ||
          !task.end_pose.joint_positions.empty() || !task.end_pose.pose.header.frame_id.empty())
        throw std::invalid_argument("dual pick requires the fixed arm and empty waypoints/end_pose");
      grippers[i] = robot->profile.grippers.at(side);
      open[i] = task.open_position;
      close[i] = task.close_position;
      const auto [lower, upper] = robot->limits.at(grippers[i]);
      if (!(open[i] >= lower && open[i] <= upper && close[i] >= lower && close[i] < open[i]))
        throw std::invalid_argument("dual pick gripper targets must close within URDF limits");
      const auto &tool = robot->profile.tools.at(side);
      const auto reference = in_base(task.tool_pose);
      const auto grasp = compose(reference, task.pick_pose, task.in_base_frame);
      const auto prepick = compose(reference, task.prepick_in_target, task.in_base_frame);
      const auto lift = compose(reference, task.lift_in_target, task.in_base_frame);
      const double descent = (prepick.translation() - grasp.translation()).norm();
      const double rise = (lift.translation() - grasp.translation()).norm();
      if (descent < .001 || descent > .5 || rise < .001 || rise > .5)
        throw std::invalid_argument("dual pick contact paths must be between 1mm and 0.5m");
      const Eigen::Vector3d size(task.object_size.x, task.object_size.y, task.object_size.z);
      if (!size.allFinite() || (size.array() <= 0.).any())
        throw std::invalid_argument("dual pick requires three positive object dimensions per arm");
      prepicks.emplace(tool, prepick);
      grasps.emplace(tool, grasp);
      lifts.emplace(tool, lift);
      objects.push_back({side, size, grasp.inverse() * reference, reference});
    }
    if ((objects[0].source_pose.translation() - objects[1].source_pose.translation()).norm() < .001)
      throw std::invalid_argument("dual pick requires two distinct objects");

    stage("scene", group);
    RestoreWorld restore{planner.get(), planner->payloads(), planner->world_state(), get_logger()};
    const auto cloud = select_cloud(CloudFreshness::WaitForPickPlace, true);
    check();
    const auto initial = state(true);
    auto current = initial;
    auto points = prepare_world(group, cloud).points;
    const auto &arm = robot->profile.groups.at(group);
    auto names = robot->full_joint_names;
    names.erase(std::remove_if(names.begin(), names.end(),
                               [](const std::string &name) { return name.rfind("Wheel", 0) == 0; }),
                names.end());
    const auto dynamics = planner->dynamic_limits(group);

    auto accept = [&](Trajectory trajectory, const Payloads &targets = {}) {
      retime_dynamics(trajectory, arm, dynamics);
      planner->validate(group, current, trajectory, true, std::nullopt, true, targets);
      const auto &end = trajectory.points.back().positions;
      for (size_t i = 0; i < names.size(); ++i) current[names[i]] = end[i];
      result.trajectory_stages.push_back(result.stage);
      result.trajectories.push_back(std::move(trajectory));
    };
    auto accept_arm = [&](PlanResult part, const Payloads &targets = {}) {
      if (!part.success) throw std::runtime_error(part.message);
      auto path = expand_trajectory(part.trajectory, names, current);
      for (auto *point : {&path.points.front(), &path.points.back()}) {
        point->velocities.assign(names.size(), 0.);
        point->accelerations.assign(names.size(), 0.);
      }
      accept(std::move(path), targets);
    };
    auto move_grippers = [&](const std::array<double, 2> &targets, const Payloads &contact) {
      double time = 0.;
      for (size_t i = 0; i < grippers.size(); ++i)
        if (std::abs(targets[i] - current.at(grippers[i])) > 1e-8) {
          const auto [lower, upper] = robot->limits.at(grippers[i]);
          time = std::max(time, robot->gripper_duration(grippers[i], upper, lower));
        }
      if (time == 0.) return;
      auto path = gripper_trajectory(names, current, grippers[0], targets[0], time, planner->settings.dt);
      overlay_gripper(path, grippers[1], current.at(grippers[1]), targets[1], time);
      if (result.stage == "close") append_hold(path, parameter<double>("gripper_wait_time", 1.));
      accept(std::move(path), contact);
    };

    stage("open", group);
    move_grippers(open, {});
    stage("prepick", group);
    publish_goals("dual_prepick", prepicks);
    accept_arm(planner->plan_pose(group, current, prepicks));

    stage("descend", group);
    // Only the intended jaw/object pair may touch; keep each target as a separate FCL box.
    for (const auto &object : objects) points = exclude_box(points, object.source_pose, object.size);
    planner->set_world(points, cloud_settings_.pitch, cloud_mode_, voxel_center_, voxel_dimensions_);
    publish_goals("dual_pick", grasps);
    accept_arm(planner->interpolate_both(current, grasps), objects);
    stage("close", group);
    move_grippers(close, objects);

    stage("lift", group);
    const auto grasp_state = robot->forward(current);
    for (auto &object : objects)
      object.in_tool = grasp_state.at(robot->profile.tools.at(object.group)).inverse() * object.source_pose;
    planner->set_payloads(objects);
    publish_goals("dual_lift", lifts);
    accept_arm(planner->interpolate_both(current, lifts));

    stage("validate", group);
    result.trajectory = concatenate(result.trajectories);
    validate_dynamics(result.trajectory, arm, dynamics);
    if (result.trajectory.points.size() > 5000)
      throw std::runtime_error("dual pick trajectory exceeds the bridge point capacity");
    result.trajectory.header.frame_id = base;
    result.trajectory.header.stamp = now();
    publish(group, initial, result.trajectory, false);
    check();
    if (goal.publish_trajectory) {
      const auto latest = state(true);
      for (size_t i = 0; i < sides.size(); ++i)
        if (std::abs(latest.at(grippers[i]) - initial.at(grippers[i])) > .0005)
          throw std::runtime_error("gripper opening changed before publishing dual pick trajectory");
      check();
      interfaces::msg::TrajectoryBlock block;
      block.task_id = task_id;
      block.block_index = 1;
      block.final = true;
      block.trajectory = result.trajectory;
      block_pub_->publish(block);
      pending_picks_ = std::move(objects);
      result.trajectory_published = true;
    }
    result.stage = "ready";
    result.failed_group.clear();
    result.success = true;
    result.message = goal.publish_trajectory ? "dual pick submitted to bridge" : "dual pick plan ready";
  } catch (const std::exception &error) {
    result.message = error.what();
    RCLCPP_ERROR(get_logger(), "【双臂抓取】失败 ｜ %s ｜ %s ｜ %s", result.stage.c_str(),
                 result.failed_group.c_str(), result.message.c_str());
  }
  return result;
}

TaskResult PlanningNode::move_lever(
    const interfaces::action::CloseLever::Goal &goal, const std::string &task_id,
    const std::function<bool()> &stopped,
    const std::function<void(const std::string &, const std::string &)> &feedback) {
  TaskResult result;
  const bool opening = goal.sweep_rad < 0.;
  const std::string action = opening ? "open_lever" : "close_lever";
  auto stage = [&](const std::string &name) {
    if (stopped()) throw std::runtime_error("lever motion cancelled or timed out");
    result.stage = name;
    feedback(name, "planning");
  };
  try {
    stage("prepare_" + action);
    const std::string group = "left_arm";
    if (execution->active() || (goal.publish_trajectory && execution->enabled()))
      throw std::invalid_argument("lever motion requires an idle SDK bridge");
    if (!held_payloads_.empty())
      throw std::invalid_argument("release and clear the attached payload before moving the lever");
    if (goal.contact_start.header.frame_id != base)
      throw std::invalid_argument("lever contact and pivot must use the planning base frame");
    const auto contact = in_base(goal.contact_start);
    const Eigen::Vector3d finger(goal.contact_in_ee.x, goal.contact_in_ee.y, goal.contact_in_ee.z);
    const Eigen::Vector2d pivot(goal.pivot.x, goal.pivot.y);
    const double radius = (contact.translation().head<2>() - pivot).norm();
    if (!pivot.allFinite() || !finger.allFinite() || !(radius >= .03 && radius <= .12) ||
        !(std::abs(goal.sweep_rad) <= std::acos(-1.)) ||
        !(goal.gripper_position >= -.042 && goal.gripper_position <= 0.))
      throw std::invalid_argument("invalid lever geometry or gripper position");

    update_world(group, CloudFreshness::WaitForPickPlace, true);
    const auto initial = state(true);
    const auto gripper = robot->profile.grippers.at(group);
    const double gripper_start = initial.at(gripper);
    const bool move_gripper = gripper_start != goal.gripper_position;
    const auto [closed, open] = robot->limits.at(gripper);
    const double gripper_time = robot->gripper_duration(gripper, open, closed);
    const auto &arm_joints = robot->profile.groups.at(group);
    const auto limits = planner->dynamic_limits(group);
    auto names = arm_joints;
    names.push_back(gripper);
    const Eigen::Vector3d shift = contact.linear() * finger;
    Transform start = contact;
    start.translation() -= shift;
    Transform above = start;
    above.translation().z() += .08;
    const Eigen::Vector2d tcp_pivot = pivot - shift.head<2>();
    const auto &tool = robot->profile.tools.at(group);

    auto current = initial;
    auto append = [&](PlanResult part, bool stop_at_ends = false) {
      if (!part.success) throw std::runtime_error(part.message);
      auto trajectory = expand_trajectory(part.trajectory, names, current);
      if (stop_at_ends) {
        for (auto *stop : {&trajectory.points.front(), &trajectory.points.back()}) {
          stop->velocities.assign(names.size(), 0.);
          stop->accelerations.assign(names.size(), 0.);
        }
      }
      const auto before = seconds(trajectory.points.back().time_from_start);
      if (retime_dynamics(trajectory, arm_joints, limits) > 1.)
        RCLCPP_INFO(get_logger(), "【拨杆 · 分段定时】%s %.3f s → %.3f s", result.stage.c_str(),
                    before, seconds(trajectory.points.back().time_from_start));
      const auto &endpoint = trajectory.points.back().positions;
      for (size_t i = 0; i < names.size(); ++i) current[names[i]] = endpoint[i];
      result.trajectory_stages.push_back(result.stage);
      result.trajectories.push_back(std::move(trajectory));
    };

    stage("approach_above_lever");
    const auto current_tool = robot->tool(initial, group);
    if ((current_tool.translation() - above.translation()).norm() > .001 ||
        Eigen::Quaterniond(current_tool.linear()).angularDistance(Eigen::Quaterniond(above.linear())) > .01) {
      append(planner->plan_pose(group, initial, {{tool, above}}), true);
      auto &approach = result.trajectories.back();
      if (move_gripper) {
        const double scale = std::max(1., gripper_time / seconds(approach.points.back().time_from_start));
        if (scale > 1.) stretch_trajectory(approach, scale);
        // Close at the stationary approach endpoint, where the bridge waits for feedback.
        overlay_gripper(approach, gripper, gripper_start, goal.gripper_position,
                        seconds(approach.points.back().time_from_start));
      }
    } else if (move_gripper) {
      result.trajectory_stages.push_back(result.stage);
      result.trajectories.push_back(
          gripper_trajectory(names, initial, gripper, goal.gripper_position, gripper_time,
                             planner->settings.dt));
    }
    if (!result.trajectories.empty())
      planner->validate(group, initial, result.trajectories.back(), /*collision=*/true,
                        std::nullopt, /*mesh_along_path=*/true);
    current[gripper] = goal.gripper_position;
    stage("descend_to_lever");
    append(planner->interpolate(group, current, above, start));

    stage("sweep_" + action);
    append(planner->interpolate_arc(group, current, start, tcp_pivot, goal.sweep_rad));

    stage("lift_after_lever");
    const auto lift_start = robot->tool(current, group);
    auto lift_end = lift_start;
    lift_end.translation().z() += .10;
    append(planner->interpolate(group, current, lift_start, lift_end));

    stage("validate_lever_trajectory");
    result.trajectory = concatenate(result.trajectories);
    // The approach is collision-checked above; preserve the accepted stage timings here.
    planner->validate(group, initial, result.trajectory, /*collision=*/false);

    stage(action + "_ready");
    const auto latest = state(true);
    if (std::abs(latest.at(gripper) - gripper_start) > .0005)
      throw std::runtime_error("gripper opening changed before publishing lever trajectory");
    publish(group, initial, result.trajectory, false);
    if (stopped()) throw std::runtime_error("lever motion cancelled or timed out");
    if (goal.publish_trajectory) {
      interfaces::msg::TrajectoryBlock block;
      block.task_id = task_id;
      block.block_index = 1;
      block.final = true;
      result.trajectory.header.frame_id = base;
      result.trajectory.header.stamp = now();
      block.trajectory = result.trajectory;
      block_pub_->publish(block);
    }
    result.success = true;
    result.message = goal.publish_trajectory ? "lever trajectory published" : "lever plan ready";
  } catch (const std::exception &error) {
    result.message = error.what();
    RCLCPP_ERROR(get_logger(), "【%s拨杆】失败 ｜ %s ｜ %s", opening ? "开" : "关",
                 result.stage.c_str(), error.what());
  }
  return result;
}
}  // namespace curobo_ros_cpp
