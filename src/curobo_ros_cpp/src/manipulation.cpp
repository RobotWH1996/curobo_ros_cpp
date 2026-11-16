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
  auto add = [&](double time) {
    auto value = sample_trajectory(trajectory, time);
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
  struct RestoreWorld {
    NativePlanner *planner;
    std::optional<Payload> payload;
    WorldState world;
    rclcpp::Logger logger;
    ~RestoreWorld() {
      try {
        planner->set_payload(payload);
        planner->restore_world(world);
      } catch (const std::exception &error) {
        RCLCPP_ERROR(logger, "Failed to restore planning scene: %s", error.what());
      } catch (...) {
        RCLCPP_ERROR(logger, "Unknown error restoring planning scene");
      }
    }
  } restore{planner.get(), planner->payload(), planner->world_state(), get_logger()};
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
    auto initial = state(true), current = initial;
    const auto names = robot->full_joint_names;
    planner->synchronize(group, current);
    const auto dynamic = planner->dynamic_limits(group);
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
      if (std::abs(place->open_position - (pick ? pick->close_position : current.at(gripper))) <
          1e-6)
        throw std::invalid_argument("place release equals initial gripping position");
    }
    if (pick) {
      const auto reference = in_base(pick->tool_pose);
      const auto grasp = compose(reference, pick->pick_pose, pick->in_base_frame);
      object.size = size(pick->object_size);
      object.in_tool = grasp.inverse() * reference;
      object.source_pose = reference;
    } else {
      const auto reference = in_base(place->tool_pose);
      const auto goal = compose(reference, place->place_pose, place->in_base_frame);
      object.size = size(place->object_size);
      object.in_tool = goal.inverse() * reference;
      object.source_pose = robot->tool(current, group) * object.in_tool;
      planner->set_payload(object);
    }
    if (pick && !stop_at_pick) current[gripper] = pick->open_position;
    std::vector<std::pair<Transform, Eigen::Vector3d>> source_exclusions;
    const auto frozen_cloud = update_world(group, initial, /*check_receive_age=*/true);
    // 抬升或撤退后，用冻结点云去掉已处理物体并刷新碰撞场景。
    auto refresh_world = [&]() {
      auto points = frozen_cloud.points;
      for (const auto &exclusion : source_exclusions)
        points = exclude_box(points, exclusion.first, exclusion.second);
      const auto payload = planner->payload();
      if (payload) points = exclude_box(points, payload->source_pose, payload->size);
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
      const auto start_pose = robot->tool(current, group);
      const auto before_state = current;
      if (target) publish_goals(stage, {{tool, *target}});
      if (target)
        part = contact ? planner->interpolate(group, current, start_pose, *target)
                       : planner->plan_pose(group, current, {{tool, *target}});
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
      contact_paths.push_back({start_pose, target.value_or(robot->tool(current, group))});
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
      for (size_t i = 0; i < pending.size(); ++i) {
        auto slice = interval(merged.trajectory, merged.boundaries[i], merged.boundaries[i + 1]);
        planner->validate(group, stage_states[i], slice, !contacts[i]);
        if (contacts[i]) {
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
      RCLCPP_INFO(get_logger(), "【合并】%s ｜ points %zu ｜ 耗时 %.1f 毫秒", shown.c_str(),
                  merged.trajectory.points.size(), elapsed_ms(before));
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
        planner->set_payload(object);
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
      planner->set_payload({});
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
}  // namespace curobo_ros_cpp
