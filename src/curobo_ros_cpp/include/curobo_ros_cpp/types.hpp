// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <yaml-cpp/yaml.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <builtin_interfaces/msg/duration.hpp>
#include <chrono>
#include <cmath>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <map>
#include <stdexcept>
#include <string>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <vector>

// 规划节点内部共用的类型、时间和位姿换算。距离单位是米，关节角是弧度。
namespace curobo_ros_cpp {
// 关节名到位置的映射。URDF 里转动关节用弧度，移动关节用米。
using JointMap = std::map<std::string, double>;
// 刚体位姿。平移是米，旋转是 3×3 矩阵。
using Transform = Eigen::Isometry3d;
// ROS 关节轨迹。时间从本段起点算起。
using Trajectory = trajectory_msgs::msg::JointTrajectory;
// 点云里的一个点，坐标在规划基座系，单位米。用 float 是为了直接送进 GPU。
using Point = Eigen::Vector3f;
// 节点内部计时用的单调时钟，不随 ROS 仿真时间变化。
using Clock = std::chrono::steady_clock;

// 将 ROS Duration 的整秒和纳秒合为 double 秒数，用于轨迹计算。
inline double seconds(const builtin_interfaces::msg::Duration &duration) {
  return duration.sec + duration.nanosec * 1e-9;
}
// 将非负有限的秒数四舍五入到纳秒并填入 ROS Duration。负数或非有限值会抛异常。
inline builtin_interfaces::msg::Duration duration(double value) {
  if (!std::isfinite(value) || value < 0.) throw std::invalid_argument("invalid trajectory time");
  const auto nanoseconds = static_cast<int64_t>(std::llround(value * 1e9));
  builtin_interfaces::msg::Duration result;
  result.sec = static_cast<int32_t>(nanoseconds / 1000000000);
  result.nanosec = static_cast<uint32_t>(nanoseconds % 1000000000);
  return result;
}
// 用单调时钟计算从 start 到当前的毫秒数，供阶段计时和超时判断。
inline double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
// ROS 位姿换成刚体变换。四元数是 xyzw，内部会归一化。
// zero_is_identity 为真时，全零四元数当成单位旋转，用来表示“相对位姿没写姿态”。
inline Transform transform(const geometry_msgs::msg::Pose &pose, bool zero_is_identity = false) {
  const Eigen::Vector3d position(pose.position.x, pose.position.y, pose.position.z);
  Eigen::Quaterniond rotation(pose.orientation.w, pose.orientation.x, pose.orientation.y,
                              pose.orientation.z);
  if (!position.allFinite() || !rotation.coeffs().allFinite())
    throw std::invalid_argument("nonfinite pose");
  if (rotation.norm() < 1e-8) {
    if (!zero_is_identity) throw std::invalid_argument("zero pose quaternion");
    rotation = Eigen::Quaterniond::Identity();
  }
  Transform result = Transform::Identity();
  result.linear() = rotation.normalized().toRotationMatrix();
  result.translation() = position;
  return result;
}
// 刚体变换写回 ROS 位姿，四元数仍是 xyzw。
inline geometry_msgs::msg::Pose pose_msg(const Transform &value) {
  geometry_msgs::msg::Pose result;
  result.position.x = value.translation().x();
  result.position.y = value.translation().y();
  result.position.z = value.translation().z();
  const Eigen::Quaterniond rotation(value.linear());
  result.orientation.w = rotation.w();
  result.orientation.x = rotation.x();
  result.orientation.y = rotation.y();
  result.orientation.z = rotation.z();
  return result;
}
// 由平移 xyz（米）和固定轴 rpy（弧度，绕 X、Y、Z）组成位姿。URDF 原点用这个顺序。
inline Transform origin(const Eigen::Vector3d &xyz, const Eigen::Vector3d &rpy) {
  Transform result = Transform::Identity();
  result.linear() = (Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()) *
                     Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()) *
                     Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX()))
                        .toRotationMatrix();
  result.translation() = xyz;
  return result;
}
// 把相对位姿叠到参考位姿上。base 为假时相对量在工具坐标系，T = reference * offset；
// 为真时平移在基座系相加，旋转仍左乘。相对平移超过 0.5 米视为写错。
inline Transform compose(const Transform &reference, const geometry_msgs::msg::Pose &relative,
                         bool base) {
  const auto offset = transform(relative, true);
  if (offset.translation().norm() > .5) throw std::invalid_argument("relative offset exceeds 0.5m");
  if (!base) return reference * offset;
  Transform result = Transform::Identity();
  result.translation() = reference.translation() + offset.translation();
  result.linear() = offset.linear() * reference.linear();
  return result;
}
// 读取 YAML 字符串数组。节点不存在时返回空数组，而不是抛异常。
inline std::vector<std::string> strings(const YAML::Node &node) {
  return node ? node.as<std::vector<std::string>>() : std::vector<std::string>{};
}
// 依次读取 YAML 序列的前三项作为 x、y、z，元素缺失或不可转换时抛异常。
inline Eigen::Vector3d vector3(const YAML::Node &value) {
  return Eigen::Vector3d(value[0].as<double>(), value[1].as<double>(), value[2].as<double>());
}

// 抓取后挂在工具上的工件包围盒，同时给 GPU 球和 FCL 盒子使用。
struct Payload {
  std::string group;                                 // 携带工件的规划组，left_arm 或 right_arm
  Eigen::Vector3d size = Eigen::Vector3d::Zero();    // 包围盒三边长度，米
  Transform in_tool = Transform::Identity();         // 盒子中心相对工具的位姿
  Transform source_pose = Transform::Identity();     // 抓取前工件在基座系的位姿，用于从点云里挖掉原位置
};
using Payloads = std::vector<Payload>;

// 一次求解各阶段的墙钟耗时，毫秒。日志里的 ik、graph、trajopt、fcl 都从这里来。
struct Timing {
  double ik_ms = 0.;        // 逆解
  double graph_ms = 0.;     // RRT 或 PRM
  double trajopt_ms = 0.;   // 轨迹优化
  double check_ms = 0.;     // FCL 或 GPU 碰撞验收
  int attempts = 0;         // 实际求解次数。接触插补为 0
};

// 一次规划的结果。失败时 trajectory 可能为空，原因写在 message。
struct PlanResult {
  bool success = false;     // 求解与该入口启用的验收均通过
  std::string message;      // 成功状态或失败诊断
  Trajectory trajectory;    // 规划轨迹。仅求 IK 时含一个关节点
  Timing timing;            // 求解器填充的阶段耗时和尝试次数
};
}  // namespace curobo_ros_cpp
