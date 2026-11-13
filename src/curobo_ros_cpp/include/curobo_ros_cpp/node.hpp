// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <curobo_ros_interfaces/action/follow_mpc.hpp>
#include <curobo_ros_interfaces/action/pick.hpp>
#include <curobo_ros_interfaces/action/pick_place.hpp>
#include <curobo_ros_interfaces/action/place.hpp>
#include <curobo_ros_interfaces/msg/mpc_plan.hpp>
#include <curobo_ros_interfaces/msg/trajectory_block.hpp>
#include <curobo_ros_interfaces/msg/mpc_target.hpp>
#include <curobo_ros_interfaces/srv/compute_ik.hpp>
#include <curobo_ros_interfaces/srv/plan_joint_goal.hpp>
#include <curobo_ros_interfaces/srv/plan_pose_goal.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <visualization_msgs/msg/marker_array.hpp>
#include "curobo_ros_cpp/executor.hpp"
#include "curobo_ros_cpp/native_planner.hpp"

// ROS 接口层。服务和动作在这里排队，真正的求解交给 NativePlanner。
namespace curobo_ros_cpp {
namespace interfaces = curobo_ros_interfaces;

// 规划组的中文名，只用于日志。未知名字原样返回。
inline std::string arm_text(const std::string &group) {
  if (group == "left_arm") return "左臂";
  if (group == "right_arm") return "右臂";
  if (group == "both_arms") return "双臂";
  return group;
}
// 抓放阶段的中文名。pick_via_N 和 place_via_N 显示为“途经 N”。
inline std::string stage_text(const std::string &stage) {
  if (stage == "prepick") return "预抓取";
  if (stage == "pick") return "抓取";
  if (stage == "lift") return "抬升";
  if (stage == "end") return "终点";
  if (stage == "close") return "闭合";
  if (stage == "preplace") return "预放置";
  if (stage == "place") return "放置";
  if (stage == "release") return "释放";
  if (stage == "retreat") return "撤退";
  if (stage.compare(0, 9, "pick_via_") == 0) return "途经 " + stage.substr(9);
  if (stage.compare(0, 10, "place_via_") == 0) return "途经 " + stage.substr(10);
  return stage.empty() ? "准备" : stage;
}

// 一次抓取、放置或抓放的结果，字段与对应 Action 的 Result 对齐。
struct TaskResult {
  bool success = false;               // 规划完成，且启用执行时控制器也成功完成
  bool execution_requested = false;   // 本模式是否会把轨迹交给控制器
  bool execution_completed = false;   // 手臂及夹爪目标全部执行成功后置真
  std::string message;                // 成功或失败原因
  std::string stage;                  // 失败时所在阶段；成功时是最后阶段
  Trajectory trajectory;              // 时间对齐的全身轨迹，含夹爪
  std::vector<Trajectory> trajectories;          // 按阶段拆开的轨迹，时间从各段起点算
  std::vector<std::string> trajectory_stages;    // 与 trajectories 一一对应的阶段名
};

// ROS 规划节点。订阅关节、点云和跟随目标，对外提供 IK、规划服务及抓放和 MPC Action。
// operation 串行化求解器访问，订阅回调可并发更新输入。Action 工作线程处理整任务生命周期。
class PlanningNode : public rclcpp::Node {
 public:
  // 读取参数，装载机器人与求解器，注册 ROS 接口并按配置预热。
  explicit PlanningNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
  // 通知工作线程停止，取消当前控制器目标，然后等待线程退出。
  ~PlanningNode();
  // 读取 ROS 参数。声明加了锁，工作线程里重复调用不会互相抢声明。
  // YAML 里写成整数的浮点参数会转成 double，避免类型不匹配直接抛异常。
  template <class Value>
  Value parameter(const std::string &name, const Value &fallback) {
    std::lock_guard<std::mutex> guard(parameter_mutex_);
    if (!has_parameter(name)) declare_parameter<Value>(name, fallback);
    const auto value = get_parameter(name);
    if constexpr (std::is_same_v<Value, double>) {
      if (value.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
        return static_cast<double>(value.as_int());
    }
    return value.get_value<Value>();
  }
  // 返回基于默认状态和最新关节反馈合并的全身快照。
  // fresh 启用抓放的反馈接收年龄和必需关节校验。没有任何反馈时抛异常。
  JointMap state(bool fresh = false);
  // 将带坐标系的位姿转换到规划基座系。空 frame_id 按基座系解释。
  // 对其他帧查询最新 TF，超时或位姿无效时抛异常。
  Transform in_base(const geometry_msgs::msg::PoseStamped &pose);
  // 读取最新点云，变换到基座系后按关节快照过滤，并更新规划场景。
  // required 启用抓放的点云年龄和可用性检查。可选点云缺失时清除点云障碍，
  // 更新异常时按配置向上报告或记录告警。相同输入可复用已处理的点云。
  PreparedCloud update_world(const std::string &group, const JointMap &state,
                             bool required = false);
  // 把本阶段的工具目标画到 /curobo/goal_waypoints。
  void publish_goals(const std::string &stage, const std::map<std::string, Transform> &goals);
  // 发布规划轨迹，按配置生成 RViz 显示路径和末端折线。start 提供未规划关节的起始值。
  void publish(const std::string &group, const JointMap &start, const Trajectory &trajectory,
               bool command = true);
  // 执行抓取、放置或组合任务。至少提供 pick 或 place 其中一个。
  // timeout 是整任务预算，单位秒。started 使用单调时钟，并可包含进入求解器前的排队时间。
  // cancelled 查询取消状态，feedback 接收阶段与状态信息。启用执行时等待控制器结果。
  TaskResult manipulate(
      const interfaces::msg::PickTask *pick, const interfaces::msg::PlaceTask *place,
      double timeout, const std::function<bool()> &cancelled,
      const std::function<void(const std::string &, const std::string &)> &feedback,
      Clock::time_point started = Clock::now(), const std::string &task_id = {},
      bool stop_at_pick = false);

  std::unique_ptr<RobotModel> robot;       // 名义模型和 FCL
  std::unique_ptr<NativePlanner> planner;  // GPU 求解器
  std::unique_ptr<Executor> execution;     // 控制器下发。deploy 模式下不真正执行
  std::string base;                        // 规划基座坐标系
  std::string robot_model_id_;             // RViz 模型名。空则用机器人配置里的短名
  std::mutex operation;                    // 保证规划、抓放和 MPC 不同时改求解器
  std::mutex parameter_mutex_;             // 保护参数的延迟声明

 private:
  // 处理关节路点规划服务。根据执行配置发布或提交验收后的轨迹。
  void plan_joint(const std::shared_ptr<interfaces::srv::PlanJointGoal::Request> request,
                  std::shared_ptr<interfaces::srv::PlanJointGoal::Response> response);
  // 处理工具位姿路点规划服务，将目标转换到基座系并按请求规划。
  void plan_pose(const std::shared_ptr<interfaces::srv::PlanPoseGoal::Request> request,
                 std::shared_ptr<interfaces::srv::PlanPoseGoal::Response> response);
  // 处理逆解服务，返回满足目标的关节解。
  void compute_ik(const std::shared_ptr<interfaces::srv::ComputeIK::Request> request,
                  std::shared_ptr<interfaces::srv::ComputeIK::Response> response);
  // 注册 Action server。接受目标后由 worker_ 按整任务预算执行并统一捕获异常。
  template <class Action>
  void add_action(const std::string &name);
  // 运行抓取、放置或组合任务，把 TaskResult 转换成对应 Action 的终态与结果。
  template <class Action>
  void run_action(std::shared_ptr<rclcpp_action::ServerGoalHandle<Action>> handle,
                  double timeout, Clock::time_point started);
  // 推进跟随循环，校验输入年龄、求解并交接轨迹，结束时设置 Action 终态。
  void follow_mpc(
      std::shared_ptr<rclcpp_action::ServerGoalHandle<interfaces::action::FollowMpc>> handle,
      Clock::time_point started);

  tf2_ros::Buffer tf_;  // 点云与工具目标的 TF 查询缓存
  tf2_ros::TransformListener listener_;  // 接收动态和静态 TF，填充 tf_
  std::mutex input_mutex_;                 // 保护关节、点云、MPC 目标及其接收时间
  JointMap joints_;                        // 最新关节位置
  JointMap joint_velocity_;                // 最新或差分得到的关节速度
  std::map<std::string, Clock::time_point> joint_received_;  // 每个关节的反馈接收时间
  double joint_stamp_ = 0.;                // 上一帧关节消息的 ROS 时间，秒
  sensor_msgs::msg::PointCloud2::SharedPtr cloud_;  // 最新点云，可为空
  Clock::time_point joint_time_;           // 最近一次关节消息的单调接收时间，供抓放使用
  Clock::time_point cloud_time_;           // 收到点云的本机时间
  interfaces::msg::MpcTarget::SharedPtr target_;    // 最新 MPC 目标
  Clock::time_point target_time_;          // 收到 MPC 目标的本机时间
  // 本体过滤的输入、被删点和环境点，仅在观察或启用过滤时创建。
  std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr>
      self_filter_pubs_;
  PointCloudSettings cloud_settings_;    // 裁剪范围、过滤模式、采样密度及点数上限
  PreparedCloud cached_cloud_;             // 上一帧已经写进求解器的点云
  std::string cached_group_;               // 生成 cached_cloud_ 时的规划组
  const void *cached_message_ = nullptr;   // 对应的点云消息地址，用来判断是不是同一帧
  JointMap cached_joints_;                 // 生成缓存时的关节，移动超过余量就重算
  bool cached_payload_ = false;            // 缓存时手上是否有工件
  uint64_t cached_generation_ = 0;         // 写入缓存时的场景代数
  std::string cloud_mode_;                 // mesh 或 voxel
  std::string cloud_frame_;                // 点云坐标系。auto 时用机器人配置里的相机帧
  Eigen::Vector3d voxel_center_;           // 距离场栅格中心
  Eigen::Vector3d voxel_dimensions_;       // 距离场栅格尺寸
  std::vector<rclcpp::ServiceBase::SharedPtr> services_;  // 持有各 ROS 服务实例
  std::vector<std::shared_ptr<void>> actions_;  // 类型擦除后的动作服务器
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;  // 关节反馈
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;  // 场景点云
  rclcpp::Subscription<interfaces::msg::MpcTarget>::SharedPtr target_sub_;  // 跟随目标
  rclcpp::Publisher<interfaces::msg::TrajectoryBlock>::SharedPtr block_pub_;
  rclcpp::Subscription<interfaces::msg::TrajectoryBlock>::SharedPtr block_cancel_sub_;
  std::mutex block_cancel_mutex_;
  std::string block_cancel_task_id_;
  rclcpp::Publisher<Trajectory>::SharedPtr trajectory_pub_;  // 规划出的关节轨迹
  rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr display_pub_;  // RViz 轨迹预览
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;  // 末端折线
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr world_pub_;   // 碰撞世界
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr goal_pub_;    // 目标路点
  rclcpp::Publisher<interfaces::msg::MpcPlan>::SharedPtr mpc_pub_;
  rclcpp::CallbackGroup::SharedPtr callbacks_;  // 可重入，让反馈订阅不被规划堵住
  std::atomic_bool busy_{false};                // 已有动作在执行
  std::atomic_bool shutdown_{false};            // 析构开始，工作线程应当退出
  std::mutex workers_mutex_;                    // 保护 worker_ 的创建和汇合
  std::thread worker_;                          // 当前动作的执行线程
};
}  // namespace curobo_ros_cpp
