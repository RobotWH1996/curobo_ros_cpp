// SPDX-License-Identifier: Apache-2.0
// 切换控制器，拆分并发送轨迹，处理确认、结果、续接和取消。
#include "curobo_ros_cpp/executor.hpp"
#include <thread>
#include "curobo_ros_cpp/trajectory.hpp"

namespace curobo_ros_cpp {
namespace {
constexpr double kCommandLead = .25;   // 新轨迹起点留出的提前量，秒
constexpr double kAcceptSlack = .02;   // 必须在起点前这么多秒拿到控制器确认
constexpr double kHandoffPosition = 1e-3;
constexpr double kHandoffVelocity = 1e-2;
constexpr double kHandoffAcceleration = 1e-1;
void check_stopped(const std::function<bool()> &stopped) {
  if (stopped()) throw std::runtime_error("execution interrupted");
}

template <class Wait>
void wait_ready(Wait wait, double timeout, const std::function<bool()> &stopped,
                const char *failure) {
  const auto started = Clock::now();
  check_stopped(stopped);
  while (!wait(std::chrono::milliseconds(10))) {
    check_stopped(stopped);
    if (elapsed_ms(started) >= timeout * 1000.) throw std::runtime_error(failure);
  }
  check_stopped(stopped);
}
}  // namespace
std::string Executor::arm_topic(const std::string &group) const {
  const auto found = profile_.arm_controllers.find(group);
  if (found == profile_.arm_controllers.end())
    throw std::invalid_argument("arm controller is not configured for " + group);
  return found->second;
}
// 只创建控制器切换客户端。轨迹客户端在第一次提交时再建。
Executor::Executor(rclcpp::Node &node, const RobotProfile &profile, bool enabled, bool switching,
                   double timeout)
    : node_(node), profile_(profile), enabled_(enabled), switch_(switching), timeout_(timeout) {
  switch_client_ = node.create_client<controller_manager_msgs::srv::SwitchController>(
      "/controller_manager/switch_controller");
}
rclcpp_action::Client<Executor::Follow>::SharedPtr Executor::client(const std::string &topic) {
  auto &value = clients_[topic];
  if (!value) value = rclcpp_action::create_client<Follow>(&node_, topic);
  return value;
}
// 切换到该手臂的控制器，并停掉另一套。等待期间不占着锁。
void Executor::prepare_group(std::unique_lock<std::mutex> &guard, const std::string &group,
                             const std::function<bool()> &stopped) {
  check_stopped(stopped);
  if (group_ == group) return;
  const auto topic = arm_topic(group);
  auto transport = client(topic);
  std::vector<rclcpp_action::Client<Follow>::SharedPtr> grippers;
  auto add_gripper = [&](const std::string &joint) {
    const auto controller = profile_.gripper_controllers.find(joint);
    if (controller != profile_.gripper_controllers.end())
      grippers.push_back(client(controller->second));
  };
  if (group == "both_arms") {
    for (const auto &item : profile_.grippers) add_gripper(item.second);
  } else if (const auto gripper = profile_.grippers.find(group);
             gripper != profile_.grippers.end()) {
    add_gripper(gripper->second);
  }
  guard.unlock();
  if (switch_) {
    wait_ready([&](auto interval) { return switch_client_->wait_for_service(interval); },
               2., stopped, "controller switch service unavailable");
    auto request = std::make_shared<controller_manager_msgs::srv::SwitchController::Request>();
    for (const auto &item : profile_.arm_controller_names) {
      if ((group == "both_arms") == (item.first == "both_arms"))
        request->activate_controllers.push_back(item.second);
      else
        request->deactivate_controllers.push_back(item.second);
    }
    request->strictness = 1;
    request->activate_asap = true;
    request->timeout = duration(2.);
    auto future = switch_client_->async_send_request(request);
    wait_ready([&](auto interval) { return future.wait_for(interval) == std::future_status::ready; },
               3., stopped, "controller switch timeout");
    if (!future.get()->ok) throw std::runtime_error("controller switch failed");
  }
  wait_ready([&](auto interval) { return transport->wait_for_action_server(interval); },
             2., stopped, "arm controller unavailable");
  for (const auto &grip : grippers)
    wait_ready([&](auto interval) { return grip->wait_for_action_server(interval); },
               2., stopped, "gripper controller unavailable");
  guard.lock();
  group_ = group;
}
// 对外的预切换入口。未启用执行时什么都不做。
void Executor::prepare(const std::string &group, const std::function<bool()> &stopped) {
  if (!enabled_) return;
  std::unique_lock<std::mutex> guard(mutex_);
  prepare_group(guard, group, stopped);
}
// 还有未完成的控制器目标时，新的规划不能开始。
bool Executor::active() const {
  std::lock_guard<std::mutex> guard(mutex_);
  for (const auto &goal : goals_)
    if (goal.result.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return true;
  return false;
}
// 新任务开始前清掉上一段轨迹的时间原点。
void Executor::clear() {
  std::lock_guard<std::mutex> guard(mutex_);
  goals_.clear();
  previous_ = Trajectory{};
  origin_ = rclcpp::Time(0, 0, node_.get_clock()->get_clock_type());
  trajectory_offset_ = 0.;
}
// 把手臂和夹爪轨迹分别发给对应控制器。
std::vector<Executor::Goal> Executor::send(
    const std::vector<std::pair<std::string, Trajectory>> &trajectories, rclcpp::Time origin,
    rclcpp::Time deadline, const std::function<bool()> &cancelled) {
  struct Pending {
    std::string topic;
    std::shared_future<Handle::SharedPtr> acknowledgement;
  };
  auto abandoned = std::make_shared<std::atomic_bool>(false);
  std::vector<Pending> pending;
  std::vector<Goal> admitted;
  try {
    for (const auto &item : trajectories) {
      check_stopped(cancelled);
      if (node_.now() >= deadline) throw std::runtime_error("controller acknowledgement deadline exceeded");
      auto transport = client(item.first);
      Follow::Goal goal;
      goal.trajectory = item.second;
      goal.trajectory.header.stamp = origin;
      rclcpp_action::Client<Follow>::SendGoalOptions options;
      options.goal_response_callback = [abandoned, weak = std::weak_ptr(transport), deadline,
                                        clock = node_.get_clock(), logger = node_.get_logger()]
          (Handle::SharedPtr handle) {
        try {
          if (handle && (abandoned->load() || clock->now() >= deadline))
            if (auto client = weak.lock()) client->async_cancel_goal(handle);
        } catch (const std::exception &error) {
          RCLCPP_WARN(logger, "Failed to cancel late controller goal: %s", error.what());
        } catch (...) {
          RCLCPP_WARN(logger, "Failed to cancel late controller goal");
        }
      };
      pending.push_back({item.first, transport->async_send_goal(goal, options)});
    }
    for (auto &item : pending) {
      while (item.acknowledgement.wait_for(std::chrono::milliseconds(1)) !=
             std::future_status::ready) {
        if (cancelled() || node_.now() >= deadline)
          throw std::runtime_error("controller acknowledgement deadline exceeded");
      }
      check_stopped(cancelled);
      auto handle = item.acknowledgement.get();
      if (!handle || node_.now() >= deadline)
        throw std::runtime_error("controller rejected or missed continuation window");
      admitted.push_back({item.topic, handle, client(item.topic)->async_get_result(handle)});
    }
  } catch (...) {
    abandoned->store(true);
    for (auto &item : pending) {
      try {
        if (item.acknowledgement.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
          auto handle = item.acknowledgement.get();
          if (handle) client(item.topic)->async_cancel_goal(handle);
        }
      } catch (...) {
        RCLCPP_WARN(node_.get_logger(), "Failed to cancel controller goal on %s", item.topic.c_str());
      }
    }
    throw;
  }
  return admitted;
}
// 用从任务起点开始的整段轨迹替换当前目标。
void Executor::submit(const std::string &group, const Trajectory &cumulative,
                      const std::function<bool()> &cancelled) {
  if (!enabled_) return;
  if (cumulative.points.size() < 2)
    throw std::runtime_error("execution requires at least two points");
  std::unique_lock<std::mutex> guard(mutex_);
  prepare_group(guard, group, cancelled);
  for (const auto &entry : profile_.gripper_controllers) {
    if (std::find(cumulative.joint_names.begin(), cumulative.joint_names.end(), entry.first) ==
        cumulative.joint_names.end())
      continue;
    auto transport = client(entry.second);
    guard.unlock();
    wait_ready([&](auto interval) { return transport->wait_for_action_server(interval); },
               2., cancelled, "gripper controller unavailable");
    guard.lock();
  }
  const bool continuation = !previous_.points.empty();
  bool restarted = false;
  double offset = trajectory_offset_;
  auto origin = origin_;
  Trajectory trajectory = cumulative;
  if (continuation) {
    if (previous_.joint_names != cumulative.joint_names ||
        cumulative.points.size() < previous_.points.size())
      throw std::runtime_error("stream prefix changed");
    for (size_t i = 0; i < previous_.points.size(); ++i)
      if (previous_.points[i] != cumulative.points[i])
        throw std::runtime_error("stream prefix changed");
    const auto boundary =
        origin_ + rclcpp::Duration(duration(
                      std::max(0., seconds(previous_.points.back().time_from_start) - .05)));
    if (node_.now() >= boundary) {
      const auto preceding = goals_;
      guard.unlock();
      const auto started = Clock::now();
      for (const auto &goal : preceding) {
        wait_ready([&](auto interval) { return goal.result.wait_for(interval) == std::future_status::ready; },
                   timeout_ - elapsed_ms(started) / 1000., cancelled,
                   "previous execution result timeout");
        const auto result = goal.result.get();
        if (result.code != rclcpp_action::ResultCode::SUCCEEDED || !result.result ||
            result.result->error_code != 0)
          throw std::runtime_error("previous execution failed");
      }
      guard.lock();
      offset = seconds(previous_.points.back().time_from_start);
      origin = node_.now() + rclcpp::Duration(duration(kCommandLead)) -
               rclcpp::Duration(duration(offset));
      restarted = true;
    }
  } else
    origin = node_.now() + rclcpp::Duration(duration(kCommandLead));
  // 后续每次替换都沿用已截掉的前缀，累计时间与控制器时间始终相差 offset。
  if (offset > 0.) {
    trajectory.points.clear();
    for (auto point : cumulative.points)
      if (seconds(point.time_from_start) >= offset - 1e-8) {
        point.time_from_start = duration(seconds(point.time_from_start) - offset);
        trajectory.points.push_back(std::move(point));
      }
  }
  const auto stamp = origin + rclcpp::Duration(duration(offset));
  std::vector<std::pair<std::string, Trajectory>> goals{
      {arm_topic(group), select_joints(trajectory, profile_.groups.at(group))}};
  for (const auto &entry : profile_.gripper_controllers)
    if (std::find(trajectory.joint_names.begin(), trajectory.joint_names.end(), entry.first) !=
        trajectory.joint_names.end())
      goals.emplace_back(entry.second, select_joints(trajectory, {entry.first}));
  auto deadline = stamp - rclcpp::Duration(duration(kAcceptSlack));
  const auto earliest = node_.now() + rclcpp::Duration(duration(.1));
  if (deadline < earliest) deadline = earliest;
  if (continuation && !restarted) {
    deadline = node_.now() + rclcpp::Duration(duration(.5));
    const auto boundary =
        origin_ + rclcpp::Duration(duration(
                      std::max(0., seconds(previous_.points.back().time_from_start) - .05)));
    if (boundary < deadline) deadline = boundary;
  }
  goals_ = send(goals, stamp, deadline, cancelled);
  origin_ = origin;
  trajectory_offset_ = offset;
  previous_ = cumulative;
}
// 等到控制器结束，或调用方取消。
void Executor::wait(const std::function<bool()> &cancelled) {
  std::vector<Goal> goals;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    goals = goals_;
  }
  const auto start = Clock::now();
  try {
    for (const auto &goal : goals) {
      wait_ready([&](auto interval) { return goal.result.wait_for(interval) == std::future_status::ready; },
                 timeout_ - elapsed_ms(start) / 1000., cancelled, "execution result timeout");
      const auto result = goal.result.get();
      if (result.code != rclcpp_action::ResultCode::SUCCEEDED || !result.result ||
          result.result->error_code != 0)
        throw std::runtime_error("controller execution failed");
    }
  } catch (...) {
    cancel();
    throw;
  }
}
// 取消尚未完成的控制器目标。
void Executor::cancel() noexcept {
  try {
    if (!rclcpp::ok()) return;
    std::lock_guard<std::mutex> guard(mutex_);
    for (auto &goal : goals_) {
      try {
        if (goal.result.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
          client(goal.controller)->async_cancel_goal(goal.handle);
      } catch (...) {
        RCLCPP_WARN(node_.get_logger(), "Failed to cancel controller goal on %s", goal.controller.c_str());
      }
    }
  } catch (...) {
    RCLCPP_ERROR(node_.get_logger(), "Failed to cancel execution");
  }
}
// MPC 每拍替换目标，时间零点由调用方保持不变。
bool Executor::submit_mpc(const std::string &group, const Trajectory &trajectory,
                          rclcpp::Time origin, const std::function<bool()> &cancelled,
                          bool continuing) {
  if (!enabled_) return true;
  std::unique_lock<std::mutex> guard(mutex_);
  prepare_group(guard, group, cancelled);
  const auto deadline = origin - rclcpp::Duration(duration(kAcceptSlack));
  if (node_.now() >= deadline) return false;
  if (trajectory.points.size() < 2) throw std::invalid_argument("MPC trajectory requires two points");
  Trajectory command = trajectory;
  rclcpp::Time stamp = origin;
  if (continuing && !previous_.points.empty()) {
    const double handoff = (origin - origin_).seconds();
    const double end = seconds(previous_.points.back().time_from_start);
    if (handoff <= 0. || handoff >= end || previous_.joint_names != trajectory.joint_names)
      return false;
    const auto expected = sample_trajectory(previous_, handoff);
    const auto &start = trajectory.points.front();
    if (start.positions.size() != static_cast<size_t>(expected.position.size())) return false;
    for (size_t j = 0; j < start.positions.size(); ++j)
      if (std::abs(start.positions[j] - expected.position[j]) > kHandoffPosition ||
          std::abs(start.velocities.at(j) - expected.velocity[j]) > kHandoffVelocity ||
          std::abs(start.accelerations.at(j) - expected.acceleration[j]) > kHandoffAcceleration)
        return false;

    Trajectory prefix;
    prefix.joint_names = trajectory.joint_names;
    const double history = std::max(0., (node_.now() - origin_).seconds() - .1);
    double offset = 0.;
    for (const auto &point : previous_.points)
      if (seconds(point.time_from_start) <= history) offset = seconds(point.time_from_start);
    for (auto point : previous_.points) {
      const double time = seconds(point.time_from_start);
      if (time >= offset && time < handoff - 1e-8) {
        point.time_from_start = duration(time - offset);
        prefix.points.push_back(std::move(point));
      }
    }
    auto boundary = trajectory.points.front();
    boundary.time_from_start = duration(handoff - offset);
    prefix.points.push_back(std::move(boundary));
    command = concatenate({prefix, trajectory});
    stamp = origin_ + rclcpp::Duration(duration(offset));
  }
  if (node_.now() >= deadline) return false;
  goals_ = send({{arm_topic(group), command}}, stamp, deadline, cancelled);
  previous_ = std::move(command);
  origin_ = stamp;
  return true;
}
}  // namespace curobo_ros_cpp
