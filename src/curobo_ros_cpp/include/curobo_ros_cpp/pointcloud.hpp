// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <curobo_robot_runtime/scene.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "curobo_ros_cpp/robot_model.hpp"

// 点云裁剪、本体过滤，以及把点变成 GPU 碰撞网格或 ESDF。
namespace curobo_ros_cpp {
// 点云进入碰撞场景前的处理配置。空间参数使用米，过滤预算使用毫秒。
struct PointCloudSettings {
  bool filter = false;                         // 是否启用相机系距离过滤，基座系裁剪始终生效
  double min_range = .2;                       // 距离相机原点的最小保留距离
  double max_range = 3.;                       // 距离相机原点的最大保留距离
  double pitch = .02;                          // 体素边长，也是网格立方体边长
  size_t max_points = 40000;                   // 体素化之后最多保留多少点
  Eigen::Vector3d lower{0., -1.15, .2};        // 基座系包围盒的最小角
  Eigen::Vector3d upper{1.05, 1.15, 1.9};      // 基座系包围盒的最大角
  std::string self_filter_mode = "apply";      // off 不过滤，observe 只统计，apply 真正删掉本体点
  double self_filter_padding = .05;            // 本体网格向外膨胀的距离
  double self_filter_budget_ms = 300.;         // 本体过滤的时间上限，超时则保留未滤完的结果
  bool self_filter_sphere_compare = false;
  bool self_filter_use_spheres = false;
  std::vector<std::array<double, 6>> exclude_boxes;  // 基座系排除盒，每项是 xmin ymin zmin xmax ymax zmax
};

// prepare_cloud 的输出。点都在基座系。
struct PreparedCloud {
  std::vector<Point> points;          // 送进碰撞场景的点
  std::vector<Point> removed;         // 被判定为本体的点，仅用于观察
  std::vector<Point> input;           // 用于诊断的体素代表点，包含选中的本体点
  std::vector<Point> environment;     // 明确保留的环境点
  size_t raw_count = 0;               // 消息里的原始点数，裁剪之前
  size_t cropped_count = 0;
  double crop_ms = 0.;
  double self_filter_ms = 0.;
  double voxel_ms = 0.;
  bool mesh_filter_timed_out = false;
  double mesh_filter_ms = 0.;
  double sphere_filter_ms = 0.;
  size_t both_removed = 0;
  std::vector<Point> mesh_only;
  std::vector<Point> sphere_only;
};

// 解码 PointCloud2 的 float32 x/y/z，可选整数 label 用于保护标注点。to_base 为相机到基座变换。
// 裁剪后用 joints 和 group 判断运动连杆，再为同体素选代表点，优先保留环境点。
// 按 max_points 限制输出规模。字段无效或消息截断时抛异常。
PreparedCloud prepare_cloud(const sensor_msgs::msg::PointCloud2 &message, const Transform &to_base,
                            const PointCloudSettings &settings, const RobotModel &robot,
                            const JointMap &joints, const std::string &group);
// 将基座系点离散为 pitch 大小的占用体素，输出占用与空体素交界处的三角面。
// 相邻占用格共享顶点。无点时返回空网格。
curobo::robot::Mesh voxel_mesh(const std::vector<Point> &points, double pitch);
// 返回与 voxel_mesh 对应的去重占用体素中心，用于 FCL 盒子树碰撞检查。
std::vector<Eigen::Vector3d> voxel_centers(const std::vector<Point> &points, double pitch);
// 在 center 和 dimensions 指定的基座系范围构建带符号的欧氏距离场。
// pitch 是分辨率，输出按 x、y、z 顺序展平，z 最快。占用体素为负，空场返回 5 米。
// 栅格尺寸无效或容量超限时抛异常。
std::vector<float> point_esdf(const std::vector<Point> &points, const Eigen::Vector3d &center,
                              const Eigen::Vector3d &dimensions, double pitch);
// 将各点变换到 at 的局部坐标系，去除盒内和边界点并返回剩余点。
// size 是局部三轴的完整边长，单位米。可用于排除抓取前的工件占位。
std::vector<Point> exclude_box(const std::vector<Point> &points, const Transform &at,
                               const Eigen::Vector3d &size);
}  // namespace curobo_ros_cpp
