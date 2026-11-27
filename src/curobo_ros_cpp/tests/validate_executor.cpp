// SPDX-License-Identifier: Apache-2.0
// 启动假控制器，验证轨迹提交、续接、超时和取消。
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <iostream>
#include <thread>
#include "curobo_ros_cpp/executor.hpp"
#include "curobo_ros_cpp/trajectory.hpp"
// 用假的 FollowJointTrajectory 服务器检查执行器的提交、替换和取消。
using namespace curobo_ros_cpp;
class MockController {
 public:
  using Follow = Executor::Follow;
  using GoalHandle = rclcpp_action::ServerGoalHandle<Follow>;
  MockController(rclcpp::Node::SharedPtr node, std::string name) : node_(std::move(node)) {
    server_ = rclcpp_action::create_server<Follow>(
        node_, name,
        [this](const auto &, auto) {
          if (delay_ms.load()) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
          return reject ? rclcpp_action::GoalResponse::REJECT
                        : rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [this](auto) {
          ++cancelled;
          return rclcpp_action::CancelResponse::ACCEPT;
        },
        [this](std::shared_ptr<GoalHandle> handle) {
          const int generation = ++generation_;
          {
            std::lock_guard<std::mutex> guard(mutex_);
            commands.push_back(handle->get_goal()->trajectory);
          }
          workers_.emplace_back([this, handle, generation] {
            auto result = std::make_shared<Follow::Result>();
            if (fail_execution) {
              result->error_code = Follow::Result::PATH_TOLERANCE_VIOLATED;
              handle->abort(result);
              ++failed;
              return;
            }
            const auto &trajectory = handle->get_goal()->trajectory;
            const auto end = rclcpp::Time(trajectory.header.stamp) +
                             rclcpp::Duration(trajectory.points.back().time_from_start);
            while (rclcpp::ok() && node_->now() < end && generation == generation_ &&
                   !handle->is_canceling())
              std::this_thread::sleep_for(std::chrono::milliseconds(2));
            if (handle->is_canceling())
              handle->canceled(result);
            else if (generation == generation_)
              handle->succeed(result);
            else {
              result->error_code = -4;
              handle->abort(result);
            }
          });
        });
  }
  ~MockController() {
    ++generation_;
    for (auto &worker : workers_)
      if (worker.joinable()) worker.join();
  }
  std::vector<Trajectory> snapshot() {
    std::lock_guard<std::mutex> guard(mutex_);
    return commands;
  }
  std::atomic_int delay_ms{0}, cancelled{0}, failed{0};
  std::atomic_bool reject{false}, fail_execution{false};

 private:
  rclcpp::Node::SharedPtr node_;
  rclcpp_action::Server<Follow>::SharedPtr server_;
  std::atomic_int generation_{0};
  std::mutex mutex_;
  std::vector<Trajectory> commands;
  std::vector<std::thread> workers_;
};
// 启动假控制器，检查替换目标、取消和超时。
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  int status = 0;
  {
    auto transport = std::make_shared<rclcpp::Node>("executor_validation");
    auto controller_node = std::make_shared<rclcpp::Node>("mock_controller");
    RobotProfile profile(ament_index_cpp::get_package_share_directory("curobo_ros_cpp") +
                         "/config/robots/lycus.yaml");
    MockController arm(controller_node, "/left_arm_controller/follow_joint_trajectory");
    MockController gripper(controller_node, "/left_gripper_controller/follow_joint_trajectory");
    rclcpp::executors::MultiThreadedExecutor spin(rclcpp::ExecutorOptions(), 6);
    spin.add_node(transport);
    spin.add_node(controller_node);
    std::thread thread([&] { spin.spin(); });
    try {
      Executor executor(*transport, profile, true, false, 5.);
      Trajectory prefix;
      prefix.joint_names = profile.groups.at("left_arm");
      prefix.joint_names.push_back("left_gripper");
      trajectory_msgs::msg::JointTrajectoryPoint start;
      start.positions.resize(prefix.joint_names.size(), 0.);
      start.velocities = start.positions;
      start.accelerations = start.positions;
      prefix.points.push_back(start);
      start.positions[0] = .1;
      start.time_from_start = duration(.4);
      prefix.points.push_back(start);
      auto suffix = prefix;
      suffix.points.front() = prefix.points.back();
      suffix.points.front().time_from_start = duration(0.);
      suffix.points.back().positions[0] = .2;
      suffix.points.back().time_from_start = duration(.3);
      const auto cumulative = concatenate({prefix, suffix});
      executor.submit("left_arm", prefix, [] { return false; });
      executor.submit("left_arm", cumulative, [] { return false; });
      executor.wait([] { return false; });
      auto arms = arm.snapshot(), grippers = gripper.snapshot();
      if (arms.size() != 2 || grippers.size() != 2 ||
          arms[0].header.stamp != grippers[0].header.stamp ||
          arms[1].header.stamp != arms[0].header.stamp)
        throw std::runtime_error("parallel start or cumulative continuation mismatch");
      std::cout << "PARALLEL_CONTINUATION_PASS" << std::endl;
      executor.clear();
      executor.submit("left_arm", prefix, [] { return false; });
      executor.wait([] { return false; });
      executor.submit("left_arm", cumulative, [] { return false; });
      executor.wait([] { return false; });
      arms = arm.snapshot();
      const auto &restarted = arms.back();
      if (seconds(restarted.points.front().time_from_start) != 0. ||
          std::abs(restarted.points.front().positions[0] - .1) > 1e-8 ||
          std::abs(seconds(restarted.points.back().time_from_start) - .3) > 1e-8)
        throw std::runtime_error("late continuation did not restart from the stopped boundary");
      std::cout << "LATE_RESTART_PASS" << std::endl;
      // 第二块迟到重启，第三块及时续接，第四块再次迟到，第五块及时续接。
      executor.clear();
      executor.submit("left_arm", prefix, [] { return false; });
      executor.wait([] { return false; });
      auto extended = cumulative;
      executor.submit("left_arm", extended, [] { return false; });
      const auto second_stamp = arm.snapshot().back().header.stamp;
      auto third = suffix;
      third.points.front() = suffix.points.back();
      third.points.front().time_from_start = duration(0.);
      third.points.back().positions[0] = .3;
      extended = concatenate({extended, third});
      executor.submit("left_arm", extended, [] { return false; });
      arms = arm.snapshot();
      grippers = gripper.snapshot();
      if (arms.back().header.stamp != second_stamp ||
          grippers.back().header.stamp != second_stamp ||
          std::abs(arms.back().points.front().positions[0] - .1) > 1e-8 ||
          std::abs(seconds(arms.back().points.back().time_from_start) - .6) > 1e-8 ||
          std::abs(seconds(grippers.back().points.back().time_from_start) - .6) > 1e-8)
        throw std::runtime_error("third block replayed the retired prefix");
      executor.wait([] { return false; });
      auto fourth = third;
      fourth.points.front() = third.points.back();
      fourth.points.front().time_from_start = duration(0.);
      fourth.points.back().positions[0] = .4;
      extended = concatenate({extended, fourth});
      executor.submit("left_arm", extended, [] { return false; });
      const auto fourth_stamp = arm.snapshot().back().header.stamp;
      if (rclcpp::Time(fourth_stamp) <= rclcpp::Time(second_stamp))
        throw std::runtime_error("repeated late continuation kept the old origin");
      auto fifth = fourth;
      fifth.points.front() = fourth.points.back();
      fifth.points.front().time_from_start = duration(0.);
      fifth.points.back().positions[0] = .5;
      extended = concatenate({extended, fifth});
      executor.submit("left_arm", extended, [] { return false; });
      arms = arm.snapshot();
      if (arms.back().header.stamp != fourth_stamp ||
          std::abs(arms.back().points.front().positions[0] - .3) > 1e-8 ||
          std::abs(seconds(arms.back().points.back().time_from_start) - .6) > 1e-8)
        throw std::runtime_error("fifth block replayed the retired prefix");
      executor.wait([] { return false; });
      std::cout << "MULTI_BLOCK_LATE_CONTINUATION_PASS" << std::endl;
      executor.clear();
      gripper.reject = true;
      bool rejected = false;
      try {
        executor.submit("left_arm", prefix, [] { return false; });
      } catch (const std::exception &) {
        rejected = true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      if (!rejected || arm.cancelled.load() == 0)
        throw std::runtime_error("partial admission was not cancelled");
      std::cout << "PARTIAL_ADMISSION_CANCEL_PASS" << std::endl;
      executor.clear();
      gripper.reject = false;
      gripper.fail_execution = true;
      auto long_arm_motion = prefix;
      long_arm_motion.points.back().time_from_start = duration(2.);
      const int previous_arm_cancels = arm.cancelled;
      executor.submit("left_arm", long_arm_motion, [] { return false; });
      const auto failure_started = Clock::now();
      bool gripper_failed = false;
      try {
        executor.wait([] { return false; });
      } catch (const std::runtime_error &error) {
        gripper_failed = std::string(error.what()).find("left_gripper_controller") != std::string::npos;
      }
      while (arm.cancelled == previous_arm_cancels && elapsed_ms(failure_started) < 600.)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      if (!gripper_failed || arm.cancelled == previous_arm_cancels || elapsed_ms(failure_started) >= 600.)
        throw std::runtime_error("gripper execution failure did not promptly stop the arm");
      std::cout << "PARALLEL_FAILURE_CANCEL_PASS" << std::endl;
      executor.clear();
      const int preceding_arm_cancels = arm.cancelled;
      const int preceding_failures = gripper.failed;
      executor.submit("left_arm", long_arm_motion, [] { return false; });
      const auto preceding_started = Clock::now();
      while (gripper.failed == preceding_failures && elapsed_ms(preceding_started) < 500.)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      if (gripper.failed == preceding_failures)
        throw std::runtime_error("mock gripper did not return the preceding failure");
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      const auto arm_goals_before = arm.snapshot().size();
      const auto gripper_goals_before = gripper.snapshot().size();
      bool continuation_failed = false;
      try {
        executor.submit("left_arm", concatenate({long_arm_motion, suffix}), [] { return false; });
      } catch (const std::runtime_error &error) {
        continuation_failed =
            std::string(error.what()).find("left_gripper_controller") != std::string::npos;
      }
      while (arm.cancelled == preceding_arm_cancels && elapsed_ms(preceding_started) < 600.)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      if (!continuation_failed || arm.cancelled == preceding_arm_cancels ||
          arm.snapshot().size() != arm_goals_before || gripper.snapshot().size() != gripper_goals_before)
        throw std::runtime_error("continuation replaced a failed gripper goal instead of stopping the arm");
      std::cout << "FAILED_PRECEDING_CONTINUATION_CANCEL_PASS" << std::endl;
      executor.clear();
      gripper.fail_execution = false;
      executor.submit("left_arm", prefix, [] { return false; });
      arms = arm.snapshot();
      if (arms.back().points.front().positions[0] != 0. ||
          std::abs(seconds(arms.back().points.back().time_from_start) - .4) > 1e-8)
        throw std::runtime_error("new task inherited a retired prefix");
      executor.wait([] { return false; });
      executor.clear();
      std::cout << "CONTINUATION_CLEAR_PASS" << std::endl;
      const auto arm_prefix = select_joints(prefix, profile.groups.at("left_arm"));
      const auto initial_origin = transport->now() + rclcpp::Duration(duration(.2));
      executor.submit_mpc("left_arm", arm_prefix, initial_origin, [] { return false; });
      const double split = .14;
      const auto handoff = sample_trajectory(arm_prefix, split);
      auto next = arm_prefix;
      for (size_t j = 0; j < next.joint_names.size(); ++j) {
        next.points.front().positions[j] = handoff.position[j];
        next.points.front().velocities[j] = handoff.velocity[j];
        next.points.front().accelerations[j] = handoff.acceleration[j];
      }
      next.points.back().positions[0] = .2;
      next.points.back().time_from_start = duration(.3);
      executor.submit_mpc("left_arm", next, initial_origin + rclcpp::Duration(duration(split)),
                          [] { return false; });
      executor.wait([] { return false; });
      arms = arm.snapshot();
      const auto &mpc_command = arms.back();
      if (mpc_command.header.stamp != initial_origin ||
          std::abs(sample_trajectory(mpc_command, .07).position[0] -
                   sample_trajectory(arm_prefix, .07).position[0]) > 1e-8)
        throw std::runtime_error("MPC replacement changed the executing prefix");
      std::cout << "MPC_PREFIX_HANDOFF_PASS" << std::endl;
      executor.clear();
      const int arm_cancels = arm.cancelled;
      const int gripper_cancels = gripper.cancelled;
      executor.submit("left_arm", prefix, [] { return false; });
      const auto task_start = Clock::now();
      bool interrupted = false;
      try {
        executor.wait([&] { return elapsed_ms(task_start) >= 30.; });
      } catch (const std::runtime_error &) {
        interrupted = true;
      }
      while (executor.active() && elapsed_ms(task_start) < 500.)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      if (!interrupted || executor.active() || arm.cancelled <= arm_cancels ||
          gripper.cancelled <= gripper_cancels)
        throw std::runtime_error("task deadline did not cancel both controllers");
      std::cout << "EXECUTION_DEADLINE_CANCEL_PASS" << std::endl;

      executor.clear();
      const auto late_origin = transport->now() + rclcpp::Duration(duration(.04));
      arm.delay_ms = 60;
      const int late_cancels = arm.cancelled;
      bool late_rejected = false;
      try {
        late_rejected = !executor.submit_mpc("left_arm", arm_prefix, late_origin,
                                             [] { return false; });
      } catch (const std::runtime_error &) {
        late_rejected = true;
      }
      const auto late_started = Clock::now();
      while (arm.cancelled == late_cancels && elapsed_ms(late_started) < 500.)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      arm.delay_ms = 0;
      if (!late_rejected || arm.cancelled == late_cancels)
        throw std::runtime_error("late MPC acknowledgement was not rejected and cancelled");
      std::cout << "MPC_LATE_ACK_CANCEL_PASS" << std::endl;

    } catch (const std::exception &error) {
      std::cerr << "EXECUTOR_VALIDATION_ERROR " << error.what() << std::endl;
      status = 1;
    }
    spin.cancel();
    thread.join();
  }
  rclcpp::shutdown();
  return status;
}
