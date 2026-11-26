// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
#include <curobo_robot_runtime/kinematics.h>
#include <curobo_robot_runtime/motion_planner.h>
#include <curobo_robot_runtime/mpc.h>
#include <functional>
#include <memory>
#include <optional>
#include "curobo_ros_cpp/pointcloud.hpp"
#include "curobo_ros_cpp/trajectory.hpp"

// 原生 CUDA 求解器的封装。逆解、轨迹优化、图搜索、接触插补和 MPC 都从这里进入。
// 每个规划组第一次使用时才加载对应的导出产物。
namespace curobo_ros_cpp {
// 检查 CUDA Runtime 返回值。成功时直接返回，失败时用 CUDA 错误文本构造 runtime_error。
inline void check_cuda(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

// 按元素计数的显存缓冲，独占 CUDA 分配。容量足够时复用地址，供 CUDA graph 重放。
// 扩容会重新分配缓冲，内容由调用方重新上传。
template <class Value>
class DeviceArray {
 public:
  // 构造空缓冲。后续 resize 分配容量。
  DeviceArray() = default;
  // 预分配 count 个 Value 元素。
  explicit DeviceArray(size_t count) { resize(count); }
  // 析构时释放持有的显存。
  ~DeviceArray() {
    if (pointer_) cudaFree(pointer_);
  }
  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;
  // 转移显存所有权，原缓冲置空。
  DeviceArray(DeviceArray &&other) noexcept : pointer_(other.pointer_), count_(other.count_) {
    other.pointer_ = nullptr;
    other.count_ = 0;
  }
  // 容量不足时重新分配。已有容量够用则保持原地址。
  void resize(size_t count) {
    if (count <= count_) return;
    if (pointer_) check_cuda(cudaFree(pointer_));
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&pointer_), count * sizeof(Value)));
    count_ = count;
  }
  // 把主机数据拷到已分配的显存。超出容量则失败，避免悄悄换地址。
  void upload(const std::vector<Value> &values) {
    if (values.size() > count_) throw std::runtime_error("GPU buffer capacity exceeded");
    check_cuda(
        cudaMemcpy(pointer_, values.data(), values.size() * sizeof(Value), cudaMemcpyHostToDevice));
  }
  // 同步拷回前 count 个元素并返回主机数组。count 超过容量时抛异常。
  std::vector<Value> host(size_t count) const {
    if (count > count_) throw std::runtime_error("GPU download capacity exceeded");
    std::vector<Value> values(count);
    check_cuda(cudaMemcpy(values.data(), pointer_, count * sizeof(Value), cudaMemcpyDeviceToHost));
    return values;
  }
  // 将已分配范围内的所有字节置零。
  void zero() { check_cuda(cudaMemset(pointer_, 0, count_ * sizeof(Value))); }
  // 显存首地址。传给运行库时必须保持稳定。
  Value *get() const { return pointer_; }

 private:
  Value *pointer_ = nullptr;  // cudaMalloc 得到的地址，空表示尚未分配
  size_t count_ = 0;          // 当前容量，元素个数
};

// ROS 参数映射到求解器的配置。控制点数与内部插值密度须匹配已导出的产物。
// dt 为对外轨迹的采样步长，mpc_optimization_dt 为相邻 MPC 控制点的时间间隔。
struct PlannerSettings {
  int seeds = 2;                      // 轨迹优化并行种子数
  int ik_seeds = 32;                  // 逆解种子数
  int iterations = 50;                // 每轮 L-BFGS 步数，须为 25 的倍数
  int initial_check_interval = 25;    // 初始迭代中每隔多少步做一次早停校验（0 表示等于 iterations）
  int finetune = 0;                   // 求出可行解后再压缩时间的轮数
  int knots = 16;                     // OPT 控制点数，对应产物目录
  int interpolation_steps = 2;        // OPT 内部样条采样密度
  int interpolation_buffer_size = 0;   // OPT 稠密验证缓冲大小，0 使用产物默认容量
  int mpc_knots = 6;                  // MPC 控制点数
  int mpc_iterations = 50;            // MPC 热启动迭代
  int mpc_cold_iterations = 100;      // MPC 冷启动迭代
  double mpc_optimization_dt = .14;   // MPC 相邻控制点的时间间隔，秒
  float mpc_collision_activation_distance = .005F; // MPC 环境碰撞代价的激活距离，米
  double mpc_posture_weight = 5.;     // 构型参考的权重
  std::string mpc_posture = "l";      // 机器人配置中的命名姿态，zero 表示零关节位
  bool mpc_run_ik = false;            // MPC 每拍是否先跑逆解
  bool cuda_graph = true;             // 捕获 CUDA graph。关闭后每次都重新发射内核
  bool vertical = true;               // 工具竖直偏好
  std::string graph = "rrt";          // 图搜索后端：rrt 或 prm
  double dt = .02;                    // 输出轨迹的时间间隔，秒
  double velocity_scale = .5;         // 速度上限的使用比例，范围 (0, 1]
  double position_tolerance = .005;   // 位置成功阈值，米
  double orientation_tolerance = .05; // 姿态成功阈值，弧度
  double path_tolerance = .01;        // 接触直线允许的横向偏差，米
  std::string device = "cuda:0";      // 求解使用的 CUDA 设备
};

// 一拍 MPC 的输出。时间单位是秒，elapsed_ms 是本拍求解耗时。
struct MpcProposal {
  bool success = false;     // 该拍预测轨迹是否可行
  std::string message;     // 求解器返回的可行性说明
  Trajectory trajectory;     // 本拍预测的关节轨迹
  double elapsed_ms = 0.;    // 求解墙钟，毫秒
  double sample_dt = 0.;     // 输出点间隔，秒
  double knot_dt = 0.;       // 控制点间隔，秒
};

// 动态碰撞场景快照，记录点云和已放置物体。
// 抓放退出时用它恢复场景。静态场景和工具上的工件由规划器分别持有。
struct WorldState {
  std::vector<Point> points;  // 当时的障碍点
  double pitch = 0.;          // 点云体素边长，米
  std::string mode;           // mesh 或 voxel
  Eigen::Vector3d center = Eigen::Vector3d::Zero();      // 体素栅格中心
  Eigen::Vector3d dimensions = Eigen::Vector3d::Zero();  // 体素栅格尺寸
  std::vector<curobo::robot::Cuboid> placed;             // 已经放下的物体
};

// 封装 cuRobo 原生求解器、GPU 场景和分组缓冲。OPT 与 MPC 分别缓存各组运行状态。
// 规划、同步和场景更新都会修改共享状态，调用方需串行使用同一个实例。
class NativePlanner {
 public:
  // artifacts 是导出目录。构造时只建空场景，规划组在 initialize 或首次求解时加载。
  NativePlanner(RobotModel &robot, std::string artifacts, PlannerSettings settings);
  // 释放规划组和其共享的 GPU 场景。
  ~NativePlanner();
  PlannerSettings settings;  // 实际求解配置，含跟随周期中读取的 MPC 权重与构型选项
  size_t max_placed_objects = 16;  // 一次任务里允许记下的已放置物体个数
  // 返回场景重建代数，供节点判定点云缓存是否可复用。
  uint64_t world_generation() const { return world_generation_; }
  // 加载规划组。mpc 为真时加载 MPC 产物，否则加载 OPT 产物。
  void initialize(const std::string &group, bool mpc = false);
  // 从 state 重算锁定关节对应的固定变换，变换变化时同步到该组的 GPU 求解器。
  // 活动关节在具体求解时单独上传，mpc 决定更新哪一套规划组实例。
  void synchronize(const std::string &group, const JointMap &state, bool mpc = false);
  // 用名义状态和小位移目标重复求解，提前初始化 OPT 和 CUDA graph。
  void warmup(const std::string &group, int iterations);
  // 按给定批次数各跑一次接触插补逆解，避免第一次抓放时才初始化。
  void warmup_interpolation(const std::string &group, const std::vector<int> &samples);
  // 用基座系点云替换环境障碍。mesh 生成占用体素的表面，voxel 生成距离场。
  // pitch 为体素边长。center 和 dimensions 分别指定距离场的中心和范围，单位为米。
  void set_world(const std::vector<Point> &points, double pitch, const std::string &mode,
                 const Eigen::Vector3d &center, const Eigen::Vector3d &dimensions);
  // 清除点云障碍并重建场景，保留静态障碍和当前的已放置物体。
  void clear_world();
  // 读入烘焙好的静态场景 YAML，盒子和网格会在每次重建世界时保留。
  void load_static_scene(const std::string &file);
  // 按基座系 pose 与三边长度 size 添加盒状障碍。超出已放置物体或场景容量时抛异常。
  void add_placed_object(const Transform &pose, const Eigen::Vector3d &size);
  // 清除已放置物体并更新碰撞场景。
  void clear_placed_objects();
  // 复制当前动态场景，供抓放操作结束后恢复。
  WorldState world_state() const;
  const std::vector<Point> &world_points() const { return world_points_; }
  // 恢复快照里的点云和已放置物体，并重建 GPU 场景。
  void restore_world(const WorldState &state);
  // 按夹持臂设置携带工件。空数组表示已释放。
  void set_payloads(const Payloads &payloads);
  // 返回当前携物配置的借用引用，生命周期随规划器。
  const Payloads &payloads() const { return payloads_; }
  // 规划到关节目标。target 按该组的活动关节顺序排列，state 提供全身起始位置。
  // attempts 是总尝试上限，结果包含轨迹、成功标志和阶段计时。
  PlanResult plan_joint(const std::string &group, const JointMap &state,
                        const std::vector<double> &target, int attempts = 3,
                        bool mesh_along_path = false);
  // 规划到一个或多个工具位姿。targets 的键是导出的工具帧，值为基座系目标。
  // 未指定的工具用求解器当前 FK 补齐。attempts 是总尝试上限。
  PlanResult plan_pose(const std::string &group, const JointMap &state,
                       const std::map<std::string, Transform> &targets, int attempts = 3,
                       bool rrt_on_first_attempt = false);
  // 按顺序将多个笛卡尔路点交给一次规划，每个路点按工具帧指定基座系位姿。
  PlanResult plan_pose_waypoints(const std::string &group, const JointMap &state,
                                 const std::vector<std::map<std::string, Transform>> &targets);
  // 求一个工具目标集合的逆解。成功时结果只含一个关节点，供后续规划使用。
  // collision 启用带场景的逆解和 FCL 验收，关闭时使用接触逆解。
  // prefer_nearby 在带碰撞验收时按距起点的关节距离尝试候选。
  PlanResult inverse_kinematics(const std::string &group, const JointMap &state,
                                const std::map<std::string, Transform> &targets,
                                bool collision = true, bool prefer_nearby = false,
                                std::vector<std::vector<double>> *all_candidates = nullptr);
  struct PickApproach {
    PlanResult prepick, descent;
    double prepick_ms = 0., descent_ms = 0.;
  };
  // 先验收从实际预抓取终点到抓取位的连续下降，必要时选择另一套关节构型。
  PickApproach plan_pick_approach(const std::string &group, const JointMap &state,
                                  const Transform &prepick, const Transform &grasp,
                                  const std::function<void()> &check);
  // 按基座系 start 和 end 生成接触段。沿直线与姿态插值求 IK，再重定时。
  // 验收位置限位、关节连续性、动态限位及末端路径偏差，接触段跳过碰撞检查。
  PlanResult interpolate(const std::string &group, const JointMap &state, const Transform &start,
                         const Transform &end);
  PlanResult interpolate_both(const JointMap &state, const std::map<std::string, Transform> &targets);
  PlanResult interpolate_arc(const std::string &group, const JointMap &state,
                             const Transform &start, const Eigen::Vector2d &center,
                             double sweep);
  // 采样检查限位和 GPU 碰撞。FCL 默认检查端点，mesh_along_path 检查所有采样。
  // 接触段传 false 时跳过碰撞，任一启用的检查失败即抛异常。
  void validate(const std::string &group, const JointMap &state, const Trajectory &trajectory,
                bool collision = true, std::optional<double> velocity_scale = std::nullopt,
                bool mesh_along_path = false, const Payloads &grasp_targets = {});
  // 返回组内关节的执行限位。速度、加速度和 jerk 分别乘速度比例的一、二、三次方。
  DynamicLimits dynamic_limits(const std::string &group,
                               std::optional<double> velocity_scale = std::nullopt);
  // 计算给定关节状态下的 GPU FK，按导出的工具帧顺序返回基座系位姿。
  std::vector<Transform> forward(const std::string &group, const JointMap &state);
  // 按 tools 顺序堆叠每个工具的六行雅可比，列顺序对应该组的活动关节。
  Eigen::MatrixXd jacobian(const std::string &group, const JointMap &state,
                           const std::vector<std::string> &tools);
  // 计算一拍 MPC。state 用于同步机器人几何，handoff 提供本拍起点的关节状态。
  // executed_knots 指相对上次接受轨迹推进的控制点数，reset 请求冷启动。
  // velocity_scale 用于转换求解和执行时间尺度，可行结果仍需由调用方接受。
  // joint_goal 固定预测末端的关节目标，由 cold_seed_goal 提供；同类目标更新保留已接受的种子。
  MpcProposal mpc_step(const std::string &group, const JointMap &state,
                       const std::map<std::string, Transform> &targets,
                       const TrajectorySample &handoff, int executed_knots, bool reset,
                       double velocity_scale = 1.,
                       const std::vector<double> &cold_seed_goal = {}, bool joint_goal = false);
  // 在当前场景下检查原轨迹的指定区间，from 和 until 使用原轨迹的时间轴。
  bool validate_mpc_continuation(const std::string &group, const Trajectory &trajectory,
                                 double from, double until, double velocity_scale);
  // 接受本拍 MPC 结果，下一拍从这里热启动。
  void accept_mpc(const std::string &group);
  // 丢弃所有组的 MPC 热启动状态。
  void reset_mpc();

 private:
  struct ContactPath {
    std::string tool;
    double length, tolerance;
    int samples;
    std::function<Transform(double)> pose_at;
    std::function<double(const Transform &)> progress;
  };
  ContactPath linear_contact(const std::string &tool, const Transform &start, const Transform &end) const;
  PlanResult interpolate_path(const std::string &name, const JointMap &state,
                              const std::vector<ContactPath> &paths);
  // 一个规划组的运行时。定义在私有头文件里，供规划器各实现文件共享。
  struct Group;
  // 从 artifacts/<group>/opt 或 mpc/<knots> 加载求解器、逆解和缓冲。
  std::unique_ptr<Group> create_group(const std::string &group, int knots, bool mpc);
  // 取出已加载的组。没有则创建。mpc 选择 OPT 缓存还是 MPC 缓存。
  Group &get(const std::string &group, bool mpc = false);
  // 按该组活动关节顺序提取位置。关节缺失或数值非有限时抛异常。
  std::vector<float> positions(const Group &group, const JointMap &state) const;
  // 上传工具位置和姿态，用 GPU FK 补齐未指定的工具，并更新工具竖直偏好。
  void fill_targets(Group &group, const JointMap &state,
                    const std::map<std::string, Transform> &targets);
  // 对一组候选关节目标做轨迹优化。pose 为真时目标来自笛卡尔逆解。
  // allow_graph 控制是否允许 RRT/PRM 回退，PRM 路径细化时关闭。
  PlanResult solve(Group &group, const JointMap &state,
                   const std::vector<std::vector<float>> &goals, int attempts, bool pose,
                   bool allow_graph = true, bool rrt_on_first_attempt = false,
                   bool mesh_along_path = false);
  // 失败时读取 IK 候选误差及碰撞代价。各项最小值独立统计。
  std::string ik_failure_detail(const Group &group, size_t candidates,
                                bool collision_checked) const;
  // 从 GPU 轨迹缓冲读回 ROS 轨迹。scaled 为真时按 velocity_scale 拉长时间。
  Trajectory output(Group &group, bool scaled);
  // 一批关节样本的 GPU 碰撞和限位检查，全部通过才返回真。
  bool gpu_valid(Group &group, const std::vector<std::vector<float>> &samples);
  // 批量检查 PRM 静止状态，返回与输入顺序相同的违规标志，0 表示通过。
  std::vector<uint8_t> configuration_violations(
      Group &group, const std::vector<std::vector<float>> &samples);
  // 轨迹优化失败时用 PRM 找一条关节路径，再交给优化器。
  PlanResult plan_prm(Group &group, const JointMap &state, const std::vector<float> &target,
                      bool mesh_along_path);

  RobotModel &robot_;    // 名义模型和 FCL，不拥有
  std::string artifacts_;  // AOT 产物根目录，按规划组、求解角色和控制点数索引
  std::unique_ptr<curobo::robot::Scene> scene_;  // GPU 碰撞场景，各组共享
  std::map<std::string, std::unique_ptr<Group>> groups_;      // OPT 组
  std::map<std::string, std::unique_ptr<Group>> mpc_groups_;  // MPC 组
  Payloads payloads_;
  std::optional<curobo::robot::ObstacleHandle> world_;  // 当前点云障碍在场景中的句柄
  std::vector<Point> world_points_;  // 写入当前碰撞场景的基座系障碍点
  double world_pitch_ = 0.;  // 上次写入点云场景的体素边长，米
  std::string world_mode_;  // 当前点云场景表示，mesh 或 voxel
  Eigen::Vector3d world_center_ = Eigen::Vector3d::Zero();  // ESDF 中心，基座系米
  Eigen::Vector3d world_dimensions_ = Eigen::Vector3d::Zero();  // ESDF 各轴范围，米
  std::vector<curobo::robot::Cuboid> static_boxes_;  // 静态场景里的盒子
  std::vector<curobo::robot::Cuboid> placed_boxes_;  // 本次任务已放下的盒子
  std::vector<curobo::robot::Mesh> static_meshes_;   // 静态场景里的网格
  uint64_t world_generation_ = 0;                    // 每次重建场景加一，用来判断点云缓存是否还有效
  // 按当前点云、静态物体、已放置物体和携物重建 GPU 场景。
  void rebuild_world();
};
}  // namespace curobo_ros_cpp
