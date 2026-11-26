// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <fcl/fcl.h>
#include <urdf/model.h>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include "curobo_ros_cpp/types.hpp"

// 机器人名义模型、关节限位、CPU 正运动学和 FCL 网格碰撞。
namespace curobo_ros_cpp {
// 从 config/robots 加载机器人特定的命名、控制器映射与构型参考。
// 活动关节顺序须与 AOT 产物及控制器一致，双臂组按左臂、右臂依次拼接。
struct RobotProfile {
  std::string id;  // 机器人短名，例如 lycus
  // 规划组到关节名。包含 left_arm、right_arm、both_arms。
  std::map<std::string, std::vector<std::string>> groups;
  std::map<std::string, std::string> tools;                 // 规划组到工具连杆
  std::map<std::string, std::string> grippers;              // 规划组到夹爪关节
  std::map<std::string, std::string> gripper_controllers;   // 夹爪关节到夹爪控制器
  std::map<std::string, std::string> arm_controllers;       // 规划组到手臂 FollowJointTrajectory 话题
  std::map<std::string, std::string> arm_controller_names;  // 规划组到 controller_manager 里的控制器名
  JointMap posture;                                         // posture_goals 里名为 l 的参考姿态
  std::map<std::string, JointMap> postures;                 // YAML 中所有命名构型，含 l
  std::string camera_frame;                                 // 点云所在坐标系
  std::string camera_parent;                                // 配置记录的相机父连杆
  std::string policy_file;                                  // 允许碰撞对的 YAML
  std::string payload_pattern;                              // 携物连杆名模板，{group} 会被替换
  std::vector<std::string> jaw_patterns;                    // 夹爪指尖连杆名模板
  std::string gripper_base_pattern;                         // 夹爪基座连杆名模板
  // 把 payload_pattern 里的 {group} 换成实际规划组。
  std::string payload_link(const std::string &group) const;
  // 该组接触时需要单独放行的夹爪连杆。
  std::vector<std::string> contact_links(const std::string &group) const;
  // 从 YAML 读取映射，缺少手臂控制器配置时采用内置默认值。
  explicit RobotProfile(const std::string &file);
};

// 一次 FCL 碰撞。kind 区分自碰撞、环境和携物。
struct Collision {
  std::string kind;    // self、environment、payload_self 或 payload_environment
  std::string first;   // 第一个连杆或障碍名
  std::string second;  // 第二个连杆或障碍名
  // 将碰撞类别和两侧名称格式化为诊断文本。
  std::string message() const;
};

// 一个连杆的碰撞网格及其包围盒。model 和 object 在加载时建好，查询时只更新位姿。
struct MeshGeometry {
  std::string link;  // 该碰撞网格所属的 URDF 连杆
  std::vector<Eigen::Vector3d> vertices;  // 连杆局部坐标系顶点，单位米
  std::vector<fcl::Triangle> faces;  // 引用 vertices 中三个顶点的三角面
  std::shared_ptr<fcl::BVHModel<fcl::OBBRSSd>> model;  // 局部几何及包围盒层次结构
  std::unique_ptr<fcl::CollisionObjectd> object;  // 更新到当前机器人位姿的 FCL 对象
  Eigen::Vector3d lower;   // 网格局部包围盒最小角
  Eigen::Vector3d upper;   // 网格局部包围盒最大角
  bool watertight = false; // 封闭网格才允许用射线判断点是否在内部
};

// 持有 URDF、关节默认状态、连杆碰撞网格及 FCL 环境对象。
// 本体与环境查询通过 query_mutex_ 串行化，规划器另行维护 GPU 场景。
class RobotModel {
 public:
  // urdf 和 robot_yaml 来自导出的几何目录。profile_file 是 config/robots 下的名字表。
  // geometry_directory 里放碰撞网格文件。
  RobotModel(const std::string &urdf, const std::string &robot_yaml,
             const std::string &profile_file, const std::string &config_directory,
             const std::string &geometry_directory,
             std::optional<double> self_filter_sphere_padding = std::nullopt);
  const RobotProfile profile;  // 加载后只读的机器人配置
  JointMap defaults;                              // 求解和预热使用的默认关节位置
  std::vector<std::string> full_joint_names;      // 全身独立关节，含双臂和夹爪
  std::set<std::string> required_joint_names;     // 抓放开始前必须收到的反馈关节
  std::map<std::string, std::pair<double, double>> limits;  // 位置下限和上限
  std::map<std::string, double> velocity_limits;  // 速度上限，用来估算夹爪动作时长
  // 计算每个连杆相对基座的位姿。缺失的关节用默认值。
  std::map<std::string, Transform> forward(const JointMap &state) const;
  // 该规划组会运动的连杆。另一条手臂和躯干不算在内。
  std::set<std::string> moving_links(const std::string &group) const;
  // Sources needed to place the moving collision geometry, including mimic sources.
  std::set<std::string> self_filter_joints(const std::string &group, bool spheres,
                                           bool compare) const;
  std::vector<Eigen::Vector4d> gripper_spheres(
      const std::string &group, const std::map<std::string, Transform> &poses) const;
  // 摆放当前连杆和携物，按规划组检查自碰撞、环境碰撞与携物碰撞。
  // contact 只影响夹爪接触连杆的环境碰撞检查，environment 控制是否查询环境网格。
  // 返回第一个碰撞对，无碰撞时返回空。
  std::optional<Collision> check(const JointMap &state, const std::string &group,
                                 bool environment = true, bool contact = false,
                                 const Payloads &payloads = {}, const Payloads &grasp_targets = {}) const;
  // 返回与输入点一一对应的本体掩码，真表示落入运动连杆的膨胀几何。
  // padding 为膨胀距离，budget_ms 为处理预算。超时时保留已算掩码，未检查项为假。
  std::vector<bool> self_mask(const std::vector<Point> &points, const JointMap &state,
                              const std::string &group, double padding, double budget_ms,
                              bool &timed_out) const;
  std::vector<bool> self_sphere_mask(const std::vector<Point> &points, const JointMap &state,
                                     const std::string &group) const;
  // 保存基座系三角网格，第一次 CPU 环境碰撞查询时构建 FCL 碰撞树。
  void set_environment(std::vector<Eigen::Vector3d> vertices,
                       std::vector<fcl::Triangle> faces);
  // 同时设置静态三角网格与点云体素中心（体素使用 FCL 动态 AABB 盒子树，建树快数十倍）。
  void set_environment(std::vector<Eigen::Vector3d> vertices,
                       std::vector<fcl::Triangle> faces,
                       std::vector<Eigen::Vector3d> voxel_centers, double voxel_pitch);
  // 释放 FCL 环境障碍，保留已加载的机器人连杆网格。
  void clear_environment();
  // 该规划组工具连杆在基座系的位姿。
  Transform tool(const JointMap &state, const std::string &group) const;
  std::set<std::string> tool_feedback_joints(const std::string &group) const;
  // 用行程、关节范围和速度上限估算五次夹爪插补时长，单位秒。
  // 位置几乎相同时返回零，查不到关节或限位非法时抛异常。
  double gripper_duration(const std::string &joint, double start, double target) const;
  // 返回连杆碰撞网格集合的借用引用，供场景检查和诊断。
  const std::vector<MeshGeometry> &meshes() const { return meshes_; }

 private:
  urdf::Model urdf_;  // URDF 关节树与 mimic 关系
  std::string base_;                                      // 运动学根连杆
  std::vector<urdf::JointConstSharedPtr> joints_;         // URDF 关节，含固定关节
  std::map<std::string, std::vector<urdf::JointConstSharedPtr>> tool_chains_;  // 各组根到工具的关节链
  std::vector<MeshGeometry> meshes_;                      // 全部连杆碰撞网格
  struct Sphere {
    Eigen::Vector3d center;
    double radius_squared;
    double physical_radius;
  };
  std::map<std::string, std::vector<Sphere>> self_filter_spheres_;
  std::vector<std::pair<size_t, size_t>> pairs_;          // 需要互查的网格下标对
  struct EnvironmentMesh {
    std::vector<Eigen::Vector3d> vertices;
    std::vector<fcl::Triangle> faces;
  };
  struct EnvironmentVoxels {
    std::vector<Eigen::Vector3d> centers;
    double pitch = 0.;
  };
  mutable std::optional<EnvironmentMesh> pending_environment_;
  mutable std::unique_ptr<fcl::CollisionObjectd> environment_;
  mutable std::optional<EnvironmentVoxels> pending_voxels_;
  mutable std::vector<std::unique_ptr<fcl::CollisionObjectd>> voxel_objects_;
  mutable std::unique_ptr<fcl::DynamicAABBTreeCollisionManagerd> voxel_manager_;
  std::set<std::string> feedback_joints(const std::set<std::string> &links) const;
  // check 会改写网格位姿。查询和环境替换共用这把锁。
  mutable std::mutex query_mutex_;
  // 读取关节位置。mimic 关节按 URDF 公式换算，并用 visited 防止循环引用。
  double joint_value(const std::string &name, const JointMap &values,
                     std::set<std::string> &visited) const;
};
}  // namespace curobo_ros_cpp
