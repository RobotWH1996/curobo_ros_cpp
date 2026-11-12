// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include "curobo_ros_cpp/types.hpp"

// 轨迹采样、拼接、夹爪叠加和速度限位检查。时间单位是秒。
namespace curobo_ros_cpp {
// 某一时刻的关节运动状态。向量顺序与调用方传入的关节名一致。
struct TrajectorySample {
  Eigen::VectorXd position;      // 位置，弧度或米
  Eigen::VectorXd velocity;      // 速度，弧度/秒或米/秒
  Eigen::VectorXd acceleration;  // 加速度，位置单位/秒²
  Eigen::VectorXd jerk;          // 加加速度，位置单位/秒³
};

// 一组关节的动态上限，顺序与 arm_joints 相同。数值来自求解器，再乘速度比例。
struct DynamicLimits {
  Eigen::VectorXd velocity;      // 各关节速度上限的绝对值
  Eigen::VectorXd acceleration;  // 各关节加速度上限的绝对值
  Eigen::VectorXd jerk;          // 各关节加加速度上限的绝对值
};

// 在严格递增的 time_from_start 上二分定位区间，按端点导数选五次、三次或线性多项式。
// 返回位置与前三阶导数。调用方应传入轨迹区间内的秒数，并预先校验完整轨迹。
TrajectorySample sample_trajectory(const Trajectory &trajectory, double time);
// 保持输入关节顺序，将每段归一到从零开始后顺次拼接，并去重连接点。
// 关节布局不同、段内时间非递增或连接位置不连续时抛异常。
Trajectory concatenate(const std::vector<Trajectory> &segments);
// 从轨迹里抽出指定关节，保持原来的时间戳。缺关节则失败。
Trajectory select_joints(const Trajectory &trajectory, const std::vector<std::string> &names);
// 把求解器输出的部分关节补成全身轨迹。未出现的关节保持 held 里的值，速度为零。
Trajectory expand_trajectory(const Trajectory &trajectory, const std::vector<std::string> &names,
                             const JointMap &held);
// 用五次多项式在 time 秒内移动指定夹爪，起止速度和加速度为零。
// dt 控制输出采样密度，其余关节保持 initial，输出顺序与 names 一致。
Trajectory gripper_trajectory(const std::vector<std::string> &names, const JointMap &initial,
                              const std::string &joint, double target, double time, double dt);
// 将夹爪五次插补叠加到轨迹开头，在 until 秒到达 target。
// 保留其他关节列，并在需要时补入夹爪动作结束的轨迹点。
void overlay_gripper(Trajectory &trajectory, const std::string &joint, double start, double target,
                     double until);
// 在轨迹末尾追加一段原地停留，用于闭合或释放后的稳定时间。
void append_hold(Trajectory &trajectory, double time);
// 在距轨迹起点 at 秒处插入 time 秒停留，其后各点的相对时间顺延。
void insert_hold(Trajectory &trajectory, double at, double time);

// 多段轨迹合并后的结果。boundaries 长度是段数加一，记录每段在总轨迹上的起止时间。
struct BlendedTrajectory {
  Trajectory trajectory;           // 已合并并重定时的轨迹
  std::vector<double> boundaries;  // 原各段映射到合并轨迹后的边界，单位秒
};
// 合并并重定时连续路段。contact 与 segments 一一对应，标记需要保留接触路径的段。
// arm_joints 决定限位向量的顺序，返回映射后的段边界供调用方按阶段验收。
BlendedTrajectory blend_trajectories(const std::vector<Trajectory> &segments,
                                     const std::vector<std::string> &arm_joints,
                                     const DynamicLimits &limits, const std::vector<bool> &contact);
// 按 step 秒采样并检查 arm_joints 的动态上限。轨迹布局、时间或任一动态限位违规时抛异常。
void validate_dynamics(const Trajectory &trajectory, const std::vector<std::string> &arm_joints,
                       const DynamicLimits &limits, double step = .005);
}  // namespace curobo_ros_cpp
