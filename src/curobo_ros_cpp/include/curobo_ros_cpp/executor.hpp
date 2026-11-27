// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <controller_manager_msgs/srv/switch_controller.hpp>
#include <future>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include "curobo_ros_cpp/robot_model.hpp"

// 控制器侧执行适配，负责控制器切换、轨迹提交和结果收尾。
namespace curobo_ros_cpp {
// 将规划轨迹拆给 ros2_control 的手臂和夹爪控制器。一个实例保存当前任务的
// 目标句柄和续接基准。调用方按任务串行提交，取消可以与结果等待并发。
class Executor {
 public:
  // 控制器提供的 ROS Action 类型。
  using Follow = control_msgs::action::FollowJointTrajectory;
  // 单个已受理轨迹目标的客户端句柄。
  using Handle = rclcpp_action::ClientGoalHandle<Follow>;

  // enabled 控制是否实际下发。switch_controllers 决定规划组切换时是否调用控制器管理器。
  // timeout 是执行结果等待上限，单位秒。node 和 profile 由调用方持有。
  Executor(rclcpp::Node &node, const RobotProfile &profile, bool enabled, bool switch_controllers,
           double timeout);
  // 当前模式是否真的向控制器下发。deploy 为假，debug 为真。
  bool enabled() const { return enabled_; }
  // 提前准备该组控制器。stopped 返回真时结束服务或 Action server 等待。
  // 控制器不可用或切换失败时抛异常。
  void prepare(const std::string &group, const std::function<bool()> &stopped);
  // 下发从任务起点计时的累计轨迹。续接时校验已提交前缀，迟到时从停止边界重启。
  // 已退休前缀通过偏移量截除。取消、确认失败或控制器拒绝时抛异常。
  void submit(const std::string &group, const Trajectory &cumulative,
              const std::function<bool()> &cancelled);
  // 按 ROS 时间 origin 下发 MPC。continuing 为真时尝试拼接尚在执行的前缀。
  // 交接不匹配或发送窗口已过时返回 false。控制器拒绝或确认超时会抛异常。
  bool submit_mpc(const std::string &group, const Trajectory &trajectory, rclcpp::Time origin,
                  const std::function<bool()> &cancelled, bool continuing = true);
  // 等当前全部目标完成。失败、超时或 cancelled 返回真时请求取消并抛异常。
  void wait(const std::function<bool()> &cancelled);
  // 向尚未完成的控制器目标请求异步取消，取消请求异常写入日志。
  void cancel() noexcept;
  // 清空目标记录和续接时钟，供任务切换使用。调用前由调用方确保上一任务已收尾。
  void clear();
  // 是否还有控制器目标未结束。
  bool active() const;

 private:
  // 一个已经发出的 FollowJointTrajectory 目标。手臂和夹爪各有一份。
  struct Goal {
    std::string controller;                              // 控制器名，用于日志和取消
    Handle::SharedPtr handle;                           // 动作句柄
    std::shared_future<Handle::WrappedResult> result;   // 控制器最终结果
  };
  // 需要切换控制器时才访问服务。已经是该组则直接返回。
  // guard 在等待服务期间会放开，避免把其他规划组堵住。
  void prepare_group(std::unique_lock<std::mutex> &guard, const std::string &group,
                     const std::function<bool()> &stopped);
  // 从 RobotProfile 取规划组对应的控制器话题，未配置时抛异常。
  std::string arm_topic(const std::string &group) const;
  // 给已拆好的轨迹写入统一起点并发给各控制器。deadline 为最晚确认的 ROS 时间。
  // 部分控制器确认失败时，撤销已受理的目标，并尝试撤销随后到达的迟到目标。
  std::vector<Goal> send(const std::vector<std::pair<std::string, Trajectory>> &trajectories,
                         rclcpp::Time origin, rclcpp::Time deadline,
                         const std::function<bool()> &cancelled);
  static bool results_complete(const std::vector<Goal> &goals);
  void wait_results(const std::vector<Goal> &goals, const std::function<bool()> &cancelled,
                    const char *timeout_message);
  // 按话题名复用动作客户端，避免每次提交都重新创建。
  rclcpp_action::Client<Follow>::SharedPtr client(const std::string &topic);

  rclcpp::Node &node_;              // 用来创建客户端的节点，不拥有
  const RobotProfile &profile_;     // 夹爪关节和控制器名
  bool enabled_;                    // 假则所有下发函数立即返回
  bool switch_;                     // 是否在组变化时切换控制器
  double timeout_;                  // 等待执行完成的秒数
  std::map<std::string, rclcpp_action::Client<Follow>::SharedPtr> clients_;  // 按话题缓存的 Action 客户端
  rclcpp::Client<controller_manager_msgs::srv::SwitchController>::SharedPtr switch_client_;  // 控制器切换服务
  std::vector<Goal> goals_;         // 最近一次提交的各控制器目标及其结果
  Trajectory previous_;             // 上次接受的累计抓放轨迹，或实际提交的 MPC 指令
  rclcpp::Time origin_{0, 0, RCL_ROS_TIME};  // previous_ 的 ROS 时间原点
  double trajectory_offset_ = 0.;   // 当前控制器轨迹在累计轨迹中的起始时间
  std::string group_;               // 已经准备好的规划组
  mutable std::mutex mutex_;        // 保护客户端缓存、目标记录与续接状态
};
}  // namespace curobo_ros_cpp
