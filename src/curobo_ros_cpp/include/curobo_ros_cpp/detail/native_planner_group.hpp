// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "curobo_ros_cpp/native_planner.hpp"

namespace curobo_ros_cpp {

// 一个规划组的一套 GPU 运行状态。OPT 和 MPC 分别持有实例，避免共享控制点数、
// 求解器状态和显存缓冲。该定义由规划器实现文件共享，只参与包内构建。
struct NativePlanner::Group {
  // 批量接触 IK 可接收的样本数，同时决定输入和结果显存的预分配容量。
  static constexpr int kIkBatchCapacity = 513;

  // 从导出元数据解析的锁定关节。同步时根据反馈重新计算对应的固定变换。
  struct FixedJoint {
    std::string source;  // 提供反馈的关节名，mimic 关节也使用其源关节
    std::string kind;    // 关节类型，prismatic 沿轴平移，旋转关节绕轴旋转
    Eigen::Vector3d xyz = Eigen::Vector3d::Zero();  // 关节原点平移，米
    Eigen::Vector3d rpy = Eigen::Vector3d::Zero();  // 关节原点固定轴 RPY，弧度
    Eigen::Vector3d axis = Eigen::Vector3d::UnitX();  // 关节局部坐标系中的单位轴
    double multiplier = 1.;  // 反馈转换比例，与 offset 一起计算关节实际位移
    double offset = 0.;      // 反馈转换偏置，单位与关节类型一致
    double fallback = 0.;    // 源关节反馈缺失时使用的导出默认值
    size_t row = 0;           // GPU 固定变换数组中的连杆行，每行 12 个浮点数
  };
  std::string name;  // 规划组名，对应导出目录和机器人配置中的组
  // knots 为样条控制点数，dof 为活动关节数，tools 为工具帧数，horizon 为求解采样长度。
  // seeds 和 ik_seeds 分别是轨迹优化及 IK 种子数，capacity 是轨迹碰撞验收缓冲的样本上限。
  int knots = 16, dof = 0, tools = 0, horizon = 0, seeds = 2, ik_seeds = 32, capacity = 4000;
  int interpolation_steps = 1;  // 相邻控制点之间的采样步数，来自导出 manifest
  bool mpc_initialized = false;  // MPC 是否已初始化，可据此决定是否需要冷启动
  bool mpc_joint_goal = false;  // 上一次 reset 选择的目标类型。切换类型时重新初始化。
  YAML::Node metadata;  // 导出产物的 deployment.json，用于初始化关节、工具和固定变换
  std::vector<FixedJoint> fixed_joints;  // 同步阶段遍历的锁定关节配置
  // names、tool_names、link_names 分别按导出产物的活动关节、工具帧和连杆顺序排列。
  std::vector<std::string> names, tool_names, link_names;
  // nominal_fixed 为原始固定变换，last_fixed 为上次下发的变换。
  // default_position、lower、upper 是活动关节的名义位置和位置下、上限。
  std::vector<float> nominal_fixed, last_fixed, default_position, lower, upper;
  DynamicLimits limits;  // 导出时的原始动态限位，执行比例在调用处应用
  std::unique_ptr<curobo::robot::MotionPlanner> planner;  // 轨迹优化、GPU 碰撞验收和相关 CUDA graph
  std::unique_ptr<curobo::robot::Kinematics> kinematics;  // 当前组的 GPU 正运动学及雅可比
  std::unique_ptr<curobo::robot::InverseKinematics> ik;  // 带场景的多种子逆解
  std::unique_ptr<curobo::robot::InverseKinematics> contact_ik;  // 接触路径使用的单种子逆解
  std::unique_ptr<curobo::robot::InverseKinematics> refine_ik;  // 单种子接触逆解失败时的多种子重试
  std::unique_ptr<curobo::robot::InverseKinematics> batch_ik;  // 同批次求解多个直线路点的逆解
  std::unique_ptr<curobo::robot::MpcController> mpc;  // MPC 组才创建的笛卡尔跟踪控制器
  // start、goal、seed_goals 保存起点、目标关节与候选关节种子。
  // ik_goals、ik_quaternions 是工具位置与姿态，ik_current 和 ik_seeds_buffer 是逆解输入。
  // ik_solution、ik_position_error、ik_orientation_error、ik_collision 是逆解结果及误差代价。
  DeviceArray<float> start, goal, seed_goals, ik_goals, ik_quaternions, ik_current, ik_seeds_buffer,
      ik_solution, ik_position_error, ik_orientation_error, ik_collision;
  // IK 成功标志、轨迹种子有效标志、OPT 成功标志，以及是否实际使用 RRT。
  DeviceArray<uint8_t> ik_success, seed_valid, success, used_rrt;
  DeviceArray<uint8_t> configuration_feasible;
  DeviceArray<int32_t> rrt_iterations;  // 求解器写回的 RRT 搜索迭代数
  // 控制点、离散位置和各阶导数、轨迹时间步长，以及优化和碰撞代价输出。
  DeviceArray<float> knots_buffer, trajectory_position, trajectory_velocity,
      trajectory_acceleration, trajectory_jerk, trajectory_dt, cost, collision;
  // fk_* 保存工具位置、姿态与雅可比，validate_* 保存待验收的轨迹状态与导数。
  DeviceArray<float> fk_position, fk_quaternion, fk_jacobian, validate_position, validate_velocity,
      validate_acceleration, validate_jerk;
  // MPC 交接状态的速度、加速度和 jerk，求解前按 MPC 时间尺度转换后上传。
  DeviceArray<float> mpc_velocity, mpc_acceleration, mpc_jerk;
};

}  // namespace curobo_ros_cpp
