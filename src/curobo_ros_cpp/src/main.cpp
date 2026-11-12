// SPDX-License-Identifier: Apache-2.0
// 进程入口，初始化 ROS 2，启动规划节点及执行器。
#include "curobo_ros_cpp/node.hpp"
// 启动规划节点并阻塞到退出。初始化失败时异常信息已经由节点写成中文日志。
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  int result = 0;
  try {
    auto node = std::make_shared<curobo_ros_cpp::PlanningNode>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 6);
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception &error) {
    RCLCPP_FATAL(rclcpp::get_logger("curobo_ros_cpp"), "%s", error.what());
    result = 1;
  }
  rclcpp::shutdown();
  return result;
}
