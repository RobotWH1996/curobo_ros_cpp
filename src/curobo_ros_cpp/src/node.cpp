// SPDX-License-Identifier: Apache-2.0
// 注册 ROS 2 接口，维护反馈与场景缓存，处理规划服务和 Action 工作线程。
#include "curobo_ros_cpp/node.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <tf2_eigen/tf2_eigen.hpp>
#include <type_traits>
namespace curobo_ros_cpp {
// 按参数加载机器人和求解器。预热只覆盖配置里列出的规划组。
PlanningNode::PlanningNode(const rclcpp::NodeOptions &options)
    : Node("curobo_motion_planning_node",
           rclcpp::NodeOptions(options).automatically_declare_parameters_from_overrides(true)),
      tf_(get_clock()),
      listener_(tf_) {
  const auto started = Clock::now();
  std::string step = "启动";
  std::string step_group;
  try {
  RCLCPP_INFO(get_logger(), "【初始化】开始加载机器人与规划器");
  step = "碰撞模型";
  const auto share = ament_index_cpp::get_package_share_directory("curobo_ros_cpp");
  const auto artifacts =
      parameter<std::string>("artifacts_directory", share + "/artifacts");
  const auto robot_file = parameter<std::string>("robot", "lycus.yml");
  const auto id = std::filesystem::path(robot_file).stem().string();
  const auto geometry = std::filesystem::path(artifacts) / "geometry" / id;
  const auto fcl_start = Clock::now();
  robot = std::make_unique<RobotModel>(
      (geometry / "robot.urdf").string(), (geometry / "robot.yml").string(),
      share + "/config/robots/" + id + ".yaml", share + "/config", geometry.string());
  RCLCPP_INFO(get_logger(), "【初始化 · 碰撞模型】完成 ｜ 网格 %zu ｜ 耗时 %.1f 毫秒",
              robot->meshes().size(), elapsed_ms(fcl_start));
  step = "规划参数";
  PlannerSettings settings;
  settings.seeds = parameter<int>("trajopt_num_seeds", 2);
  settings.ik_seeds = parameter<int>("ik_num_seeds", 32);
  settings.iterations = parameter<int>("trajopt_num_iters", 50);
  settings.finetune = parameter<int>("trajopt_finetune_attempts", 0);
  settings.knots = parameter<int>("trajopt_num_control_points", 16);
  settings.interpolation_steps = parameter<int>("trajopt_interpolation_steps", 2);
  settings.mpc_knots = parameter<int>("mpc_num_control_points", 6);
  settings.mpc_iterations = parameter<int>("mpc_warm_start_iters", 50);
  settings.mpc_cold_iterations = parameter<int>("mpc_cold_start_iters", 100);
  settings.mpc_optimization_dt = parameter<double>("mpc_optimization_dt", .14);
  settings.graph = parameter<std::string>("trajopt_graph_planner", "rrt");
  settings.cuda_graph = parameter<bool>("use_cuda_graph", true);
  settings.vertical = parameter<bool>("keep_tool_z_vertical", true);
  settings.dt = parameter<double>("interpolation_dt", .02);
  settings.velocity_scale = parameter<double>("trajopt_velocity_scale", .5);
  settings.position_tolerance = parameter<double>("position_tolerance", .005);
  settings.orientation_tolerance = parameter<double>("orientation_tolerance", .05);
  settings.path_tolerance = parameter<double>("pick_place_path_tolerance", .01);
  settings.joint_tolerance = parameter<double>("waypoint_joint_tolerance", .1);
  settings.device = parameter<std::string>("mpc_device", "cuda:0");
  if (settings.dt <= 0 || settings.velocity_scale <= 0 || settings.velocity_scale > 1.)
    throw std::invalid_argument("invalid timing or velocity scale");
  const auto warp = std::filesystem::path(share).parent_path().parent_path() / "lib" / "warp.so";
  if (!std::getenv("CUROBO_WARP_LIBRARY")) setenv("CUROBO_WARP_LIBRARY", warp.c_str(), 0);
  step = "规划器";
  planner = std::make_unique<NativePlanner>(*robot, artifacts, settings);
  planner->max_placed_objects =
      static_cast<size_t>(std::max(0, parameter<int>("pick_place_max_placed_objects", 16)));
  step = "静态场景";
  const auto scene = parameter<std::string>("scene_model", "");
  if (!scene.empty()) {
    const auto exported =
        std::filesystem::path(artifacts) / "scenes" / std::filesystem::path(scene).filename();
    planner->load_static_scene(std::filesystem::exists(exported) ? exported.string() : scene);
  }
  step = "点云参数";
  base = parameter<std::string>("base_frame", "base_link");
  robot_model_id_ = parameter<std::string>("robot_model_id", "");
  const auto cloud_parent = parameter<std::string>("pointcloud_parent_frame", base);
  if (!cloud_parent.empty() && cloud_parent != base)
    throw std::invalid_argument("pointcloud_parent_frame must match base_frame");
  cloud_settings_.filter = parameter<bool>("filter_pointcloud", false);
  cloud_settings_.pitch = parameter<double>("pointcloud_pitch", .02);
  cloud_settings_.min_range = parameter<double>("pointcloud_min_range", .2);
  cloud_settings_.max_range = parameter<double>("pointcloud_max_range", 3.);
  cloud_settings_.max_points = parameter<int>("pointcloud_max_points", 40000);
  cloud_settings_.self_filter_mode = parameter<std::string>("self_filter_mode", "apply");
  cloud_settings_.self_filter_padding = parameter<double>("self_filter_padding", .05);
  cloud_settings_.self_filter_budget_ms = parameter<double>("self_filter_max_processing_ms", 300.);
  auto vector_parameter = [&](const std::string &name, const std::vector<double> &fallback) {
    auto value = parameter<std::vector<double>>(name, fallback);
    if (value.size() != 3) throw std::invalid_argument(name + " requires three values");
    return Eigen::Vector3d(value[0], value[1], value[2]);
  };
  cloud_settings_.lower = vector_parameter("pointcloud_bounds_min", {0., -1.15, .2});
  cloud_settings_.upper = vector_parameter("pointcloud_bounds_max", {1.05, 1.15, 1.9});
  voxel_center_ = vector_parameter("pointcloud_voxel_center", {.2, 0., 1.});
  voxel_dimensions_ = vector_parameter("pointcloud_voxel_dims", {2., 2., 2.});
  cloud_mode_ = parameter<std::string>("pointcloud_collision", "mesh");
  cloud_frame_ = parameter<std::string>("pointcloud_child_frame", "auto");
  if (cloud_frame_ == "auto") cloud_frame_ = robot->profile.camera_frame;
  if (cloud_settings_.pitch <= 0 || cloud_settings_.max_points == 0 ||
      cloud_settings_.self_filter_padding < 0)
    throw std::invalid_argument("invalid pointcloud settings");
  if (!std::isfinite(cloud_settings_.pitch) || !cloud_settings_.lower.allFinite() ||
      !cloud_settings_.upper.allFinite() ||
      (cloud_settings_.lower.array() >= cloud_settings_.upper.array()).any() ||
      !std::isfinite(cloud_settings_.self_filter_padding) ||
      !std::isfinite(cloud_settings_.self_filter_budget_ms) ||
      cloud_settings_.self_filter_budget_ms <= 0.)
    throw std::invalid_argument("invalid pointcloud geometry or filtering budget");
  if (cloud_mode_ != "mesh" && cloud_mode_ != "voxel")
    throw std::invalid_argument("pointcloud_collision must be mesh or voxel");
  if (cloud_settings_.self_filter_mode != "off" && cloud_settings_.self_filter_mode != "observe" &&
      cloud_settings_.self_filter_mode != "apply")
    throw std::invalid_argument("invalid self_filter_mode");
  auto boxes = parameter<std::string>("pointcloud_exclude_boxes", "");
  for (char &character : boxes)
    if (character == ',' || character == ';') character = ' ';
  std::istringstream box_stream(boxes);
  double coordinate;
  std::array<double, 6> box;
  size_t index = 0;
  while (box_stream >> coordinate) {
    box[index++] = coordinate;
    if (index == 6) {
      cloud_settings_.exclude_boxes.push_back(box);
      index = 0;
    }
  }
  if (index) throw std::invalid_argument("pointcloud_exclude_boxes requires groups of six values");
  step = "接口";
  step_group.clear();
  callbacks_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  rclcpp::SubscriptionOptions subscription_options;
  subscription_options.callback_group = callbacks_;
  joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      parameter<std::string>("joint_states_topic", "/joint_states"), 10,
      [this](sensor_msgs::msg::JointState::SharedPtr message) {
        std::lock_guard<std::mutex> guard(input_mutex_);
        const double stamp = rclcpp::Time(message->header.stamp).seconds();
        const double dt = stamp - joint_stamp_;
        const bool reported = message->velocity.size() == message->name.size() &&
                              std::any_of(message->velocity.begin(), message->velocity.end(),
                                          [](double value) { return std::abs(value) > 1e-4; });
        const auto received = Clock::now();
        for (size_t i = 0; i < std::min(message->name.size(), message->position.size()); ++i) {
          const auto &name = message->name[i];
          const double position = message->position[i];
          auto old = joints_.find(name);
          if (reported)
            joint_velocity_[name] = message->velocity[i];
          else if (old != joints_.end() && dt > 1e-4 && dt < .5) {
            const double delta = position - old->second;
            joint_velocity_[name] = std::abs(delta) < 1e-5 ? 0. : std::clamp(delta / dt, -10., 10.);
          } else if (dt >= .5 || old == joints_.end())
            joint_velocity_[name] = 0.;
          joints_[name] = position;
          joint_received_[name] = received;
        }
        joint_stamp_ = stamp;
        joint_time_ = received;
      },
      subscription_options);
  if (parameter<bool>("use_pointcloud", true))
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        parameter<std::string>("pointcloud_topic", "/all_points"), rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::PointCloud2::SharedPtr message) {
          std::lock_guard<std::mutex> guard(input_mutex_);
          cloud_ = message;
          cloud_time_ = Clock::now();
        },
        subscription_options);
  if (cloud_settings_.self_filter_mode != "off")
    for (const auto &name : {"input", "removed", "environment"})
      self_filter_pubs_[name] = create_publisher<sensor_msgs::msg::PointCloud2>(
          std::string("/curobo/self_filter/") + name, 1);
  target_sub_ = create_subscription<interfaces::msg::MpcTarget>(
      parameter<std::string>("mpc_target_topic", "/curobo/target"), 10,
      [this](interfaces::msg::MpcTarget::SharedPtr message) {
        std::lock_guard<std::mutex> guard(input_mutex_);
        target_ = message;
        target_time_ = Clock::now();
      },
      subscription_options);
  trajectory_pub_ = create_publisher<Trajectory>(
      parameter<std::string>("planned_trajectory_topic", "/curobo/planned_trajectory"), 1);
  const auto block_topic =
      parameter<std::string>("planned_trajectory_topic", "/curobo/planned_trajectory") + "/blocks";
  block_pub_ = create_publisher<interfaces::msg::TrajectoryBlock>(block_topic, 16);
  block_cancel_sub_ = create_subscription<interfaces::msg::TrajectoryBlock>(
      block_topic, 16, [this](interfaces::msg::TrajectoryBlock::ConstSharedPtr message) {
        if (!message->cancel || message->task_id.empty()) return;
        std::lock_guard<std::mutex> guard(block_cancel_mutex_);
        block_cancel_task_id_ = message->task_id;
      }, subscription_options);
  display_pub_ = create_publisher<moveit_msgs::msg::DisplayTrajectory>("/display_planned_path", 1);
  const auto marker = parameter<std::string>("trajectory_marker_topic", "");
  marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      marker.empty() ? "/" + id + "_motion_trajectory" : marker, 1);
  world_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/curobo/collision_world", 1);
  goal_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/curobo/goal_waypoints", 1);
  mpc_pub_ = create_publisher<interfaces::msg::MpcPlan>(
      parameter<std::string>("mpc_plan_topic", "/curobo/mpc_plan"), 1);
  execution = std::make_unique<Executor>(
      *this, robot->profile, parameter<bool>("execute_trajectory", false),
      parameter<bool>("switch_controllers", true), parameter<double>("execute_timeout", 60.));
  step = "轨迹优化";
  const auto warm = parameter<std::vector<std::string>>("trajopt_prewarm_groups", {"left_arm"});
  for (const auto &group : warm) {
    step_group = group;
    const auto stage = Clock::now();
    planner->initialize(group);
    planner->warmup(group, parameter<int>("warmup_iterations", 3));
    const auto label = arm_text(group);
    RCLCPP_INFO(get_logger(), "【初始化 · 轨迹优化】完成 ｜ 规划组 %s ｜ 耗时 %.1f 毫秒", label.c_str(),
                elapsed_ms(stage));
  }
  if (parameter<bool>("pick_place_ik_prewarm", true)) {
    step = "接触插补";
    for (const auto &group :
         parameter<std::vector<std::string>>("pick_place_ik_prewarm_modes", {"left_arm"})) {
      step_group = group;
      const auto stage = Clock::now();
      std::vector<int> samples;
      for (auto count : parameter<std::vector<int64_t>>("pick_place_ik_prewarm_samples", {33}))
        samples.push_back(count);
      planner->warmup_interpolation(group, samples);
      const auto label = arm_text(group);
      RCLCPP_INFO(get_logger(), "【初始化 · 接触插补】完成 ｜ 规划组 %s ｜ 耗时 %.1f 毫秒",
                  label.c_str(), elapsed_ms(stage));
    }
  }
  if (parameter<bool>("mpc_prewarm", false)) {
    step = "模型预测";
    const auto stage = Clock::now();
    for (const auto &group : parameter<std::vector<std::string>>(
             "mpc_prewarm_modes", {"left_arm", "right_arm", "both_arms"})) {
      step_group = group;
      planner->initialize(group, true);
      std::map<std::string, Transform> targets;
      for (const auto &tool : robot->profile.tools)
        if (group == "both_arms" || tool.first == group)
          targets[tool.second] = robot->tool(robot->defaults, tool.first);
      planner->mpc_step(group, robot->defaults, targets, TrajectorySample{}, 0, true);
      planner->mpc_step(group, robot->defaults, targets, TrajectorySample{}, 0, true);
    }
    planner->reset_mpc();
    step_group.clear();
    RCLCPP_INFO(get_logger(), "【初始化 · 模型预测】完成 ｜ 耗时 %.1f 毫秒", elapsed_ms(stage));
  }
  step = "服务";
  step_group.clear();
  services_.push_back(create_service<interfaces::srv::PlanJointGoal>(
      "plan_joint_goal",
      std::bind(&PlanningNode::plan_joint, this, std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default, callbacks_));
  services_.push_back(create_service<interfaces::srv::PlanPoseGoal>(
      "plan_pose_goal",
      std::bind(&PlanningNode::plan_pose, this, std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default, callbacks_));
  services_.push_back(create_service<interfaces::srv::ComputeIK>(
      "compute_ik",
      std::bind(&PlanningNode::compute_ik, this, std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default, callbacks_));
  add_action<interfaces::action::Pick>("pick");
  add_action<interfaces::action::Place>("place");
  add_action<interfaces::action::PickPlace>("pick_place");
  add_action<interfaces::action::FollowMpc>("follow_mpc");
  RCLCPP_INFO(get_logger(), "【初始化】完成 ｜ 总耗时 %.1f 毫秒 ｜ 可以接收任务", elapsed_ms(started));
  } catch (const std::exception &error) {
    if (step_group.empty())
      RCLCPP_FATAL(get_logger(), "【初始化 · %s】未完成 ｜ 错误：%s", step.c_str(), error.what());
    else {
      const auto label = arm_text(step_group);
      RCLCPP_FATAL(get_logger(), "【初始化 · %s】未完成 ｜ 规划组 %s ｜ 错误：%s", step.c_str(),
                   label.c_str(), error.what());
    }
    throw std::runtime_error("【初始化】未完成");
  }
}
// 停止下发并等动作线程结束，避免析构时还在访问求解器。
PlanningNode::~PlanningNode() {
  shutdown_ = true;
  execution->cancel();
  if (worker_.joinable()) worker_.join();
}
// 用最新关节反馈覆盖默认值。fresh 时检查年龄和任务关节是否齐全。
JointMap PlanningNode::state(bool fresh) {
  std::lock_guard<std::mutex> guard(input_mutex_);
  if (joints_.empty()) throw std::runtime_error("no joint_states received");
  if (fresh &&
      elapsed_ms(joint_time_) > parameter<double>("pick_place_joint_state_max_age", 1.) * 1000.)
    throw std::runtime_error("joint_states expired");
  if (fresh)
    for (const auto &joint : robot->required_joint_names)
      if (!joints_.count(joint) || !std::isfinite(joints_.at(joint)))
        throw std::runtime_error("missing or invalid task joint feedback " + joint);
  auto result = robot->defaults;
  for (const auto &joint : joints_) result[joint.first] = joint.second;
  return result;
}
// 感知位姿留在原坐标系，这里再用 TF 变到基座系。
Transform PlanningNode::in_base(const geometry_msgs::msg::PoseStamped &pose) {
  const auto value = transform(pose.pose);
  if (pose.header.frame_id.empty() || pose.header.frame_id == base) return value;
  return tf2::transformToEigen(tf_.lookupTransform(base, pose.header.frame_id, tf2::TimePointZero,
                                                   tf2::durationFromSec(1.))) *
         value;
}
// 处理一帧点云并写入碰撞场景。点云缺失时可以选择沿用上一帧。
PreparedCloud PlanningNode::update_world(const std::string &group, const JointMap &joints,
                                         bool required) {
  try {
    sensor_msgs::msg::PointCloud2::SharedPtr cloud;
    {
      std::lock_guard<std::mutex> guard(input_mutex_);
      cloud = cloud_;
      if (required && cloud &&
          elapsed_ms(cloud_time_) > parameter<double>("pick_place_cloud_max_age", 2.) * 1000.)
        throw std::runtime_error("pointcloud expired");
    }
    if (!cloud) {
      if (required || parameter<bool>("require_pointcloud", false))
        throw std::runtime_error("no pointcloud received");
      planner->clear_world();
      cached_message_ = nullptr;
      return {};
    }
    const bool same_cloud = cached_message_ == cloud.get() && cached_group_ == group &&
                            cached_payload_ == planner->payload().has_value() &&
                            cached_generation_ == planner->world_generation();
    bool joints_close = same_cloud && cached_joints_.size() == joints.size();
    if (joints_close)
      for (const auto &joint : joints) {
        const auto previous = cached_joints_.find(joint.first);
        if (previous == cached_joints_.end() || std::abs(previous->second - joint.second) > .02) {
          joints_close = false;
          break;
        }
      }
    if (joints_close) return cached_cloud_;
    const auto started = Clock::now();
    Transform at = Transform::Identity();
    const auto frame = cloud_frame_.empty() ? cloud->header.frame_id : cloud_frame_;
    if (frame != base)
      at = tf2::transformToEigen(
          tf_.lookupTransform(base, frame, tf2::TimePointZero, tf2::durationFromSec(1.)));
    auto prepared = prepare_cloud(*cloud, at, cloud_settings_, *robot, joints, group);
    if (required && prepared.points.empty() && prepared.removed.empty())
      throw std::runtime_error("required pointcloud has no usable points inside planning bounds");
    for (const auto &publisher : self_filter_pubs_)
      if (publisher.second->get_subscription_count()) {
        const auto &points = publisher.first == "input"     ? prepared.input
                             : publisher.first == "removed" ? prepared.removed
                                                            : prepared.environment;
        sensor_msgs::msg::PointCloud2 message;
        message.header = cloud->header;
        message.header.frame_id = base;
        message.height = 1;
        message.width = points.size();
        message.point_step = 12;
        message.row_step = 12 * points.size();
        message.is_dense = true;
        message.data.resize(message.row_step);
        for (int axis = 0; axis < 3; ++axis) {
          sensor_msgs::msg::PointField field;
          field.name = axis == 0 ? "x" : axis == 1 ? "y" : "z";
          field.offset = axis * 4;
          field.datatype = field.FLOAT32;
          field.count = 1;
          message.fields.push_back(field);
        }
        for (size_t i = 0; i < points.size(); ++i)
          for (int axis = 0; axis < 3; ++axis) {
            const float value = points[i][axis];
            std::memcpy(message.data.data() + i * 12 + axis * 4, &value, 4);
          }
        publisher.second->publish(message);
      }
    auto points = prepared.points;
    const auto payload = planner->payload();
    if (payload) points = exclude_box(points, payload->source_pose, payload->size);
    planner->set_world(points, cloud_settings_.pitch, cloud_mode_, voxel_center_,
                       voxel_dimensions_);
    if (parameter<bool>("visualize_mesh_obstacles", false)) {
      visualization_msgs::msg::MarkerArray markers;
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = base;
      marker.header.stamp = now();
      marker.ns = "pointcloud";
      marker.id = 0;
      marker.type = marker.CUBE_LIST;
      marker.action = marker.ADD;
      marker.pose.orientation.w = 1.;
      marker.scale.x = marker.scale.y = marker.scale.z = cloud_settings_.pitch;
      marker.color.r = .7;
      marker.color.g = .3;
      marker.color.b = .2;
      marker.color.a = .35;
      for (const auto &point : points) {
        geometry_msgs::msg::Point p;
        p.x = point.x();
        p.y = point.y();
        p.z = point.z();
        marker.points.push_back(p);
      }
      markers.markers.push_back(marker);
      world_pub_->publish(markers);
    }
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                         "【场景】原始 %zu ｜ 保留 %zu ｜ 滤除本体 %zu ｜ 超时 %s ｜ 耗时 %.1f 毫秒",
                         prepared.raw_count, points.size(), prepared.removed.size(),
                         prepared.self_filter_timed_out ? "是" : "否", elapsed_ms(started));
    cached_cloud_ = prepared;
    cached_group_ = group;
    cached_message_ = cloud.get();
    cached_joints_ = joints;
    cached_payload_ = planner->payload().has_value();
    cached_generation_ = planner->world_generation();
    return prepared;
  } catch (const std::exception &error) {
    if (required || parameter<bool>("require_pointcloud", false)) throw;
    RCLCPP_WARN(get_logger(), "【场景】更新失败，沿用上一帧 ｜ 错误：%s", error.what());
    return {};
  }
}
// 发布本阶段目标坐标系，便于在 RViz 里核对抓取和放置位置。
void PlanningNode::publish_goals(const std::string &stage,
                                 const std::map<std::string, Transform> &goals) {
  if (!parameter<bool>("visualize_goal_waypoints", true)) return;
  visualization_msgs::msg::MarkerArray markers;
  int id = 0;
  for (const auto &goal : goals) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = base;
    marker.header.stamp = now();
    marker.ns = "goal_" + stage + "_" + goal.first;
    marker.id = id++;
    marker.type = marker.ARROW;
    marker.action = marker.ADD;
    marker.pose = pose_msg(goal.second);
    marker.scale.x = .08;
    marker.scale.y = .012;
    marker.scale.z = .012;
    marker.color.r = 1.;
    marker.color.g = .5;
    marker.color.a = 1.;
    markers.markers.push_back(marker);
  }
  goal_pub_->publish(markers);
}
// 同时发轨迹话题、MoveIt 显示和末端折线。
void PlanningNode::publish(const std::string &group, const JointMap &start,
                           const Trajectory &trajectory, bool command) {
  auto output = trajectory;
  output.header.frame_id = base;
  output.header.stamp = now();
  if (command) trajectory_pub_->publish(output);
  if (!parameter<bool>("visualize_trajectory", true)) return;
  if (output.joint_names.size() == robot->profile.groups.at(group).size() && group != "both_arms")
    output = expand_trajectory(output, robot->profile.groups.at("both_arms"), start);
  moveit_msgs::msg::DisplayTrajectory display;
  display.model_id = robot_model_id_.empty() ? robot->profile.id : robot_model_id_;
  for (const auto &joint : start) {
    display.trajectory_start.joint_state.name.push_back(joint.first);
    display.trajectory_start.joint_state.position.push_back(joint.second);
  }
  moveit_msgs::msg::RobotTrajectory motion;
  motion.joint_trajectory = output;
  display.trajectory.push_back(motion);
  display_pub_->publish(display);
  if (parameter<bool>("compute_ee_paths", true)) {
    visualization_msgs::msg::MarkerArray markers;
    for (const auto &tool : robot->profile.tools) {
      if (group != "both_arms" && tool.first != group) continue;
      visualization_msgs::msg::Marker marker;
      marker.header = output.header;
      marker.ns = tool.first;
      marker.id = 0;
      marker.type = marker.LINE_STRIP;
      marker.action = marker.ADD;
      marker.pose.orientation.w = 1.;
      marker.scale.x = .005;
      marker.color.g = 1.;
      marker.color.a = 1.;
      for (size_t i = 0; i < trajectory.points.size(); i += 2) {
        auto state = start;
        for (size_t j = 0; j < trajectory.joint_names.size(); ++j)
          state[trajectory.joint_names[j]] = trajectory.points[i].positions[j];
        const auto at = robot->tool(state, tool.first);
        geometry_msgs::msg::Point p;
        p.x = at.translation().x();
        p.y = at.translation().y();
        p.z = at.translation().z();
        marker.points.push_back(p);
      }
      markers.markers.push_back(marker);
    }
    marker_pub_->publish(markers);
  }
}
// 逐个关节路点规划，再拼成一条轨迹。忙时直接拒绝。
void PlanningNode::plan_joint(
    const std::shared_ptr<interfaces::srv::PlanJointGoal::Request> request,
    std::shared_ptr<interfaces::srv::PlanJointGoal::Response> response) {
  std::unique_lock<std::mutex> lock(operation, std::try_to_lock);
  if (!lock.owns_lock() || busy_) {
    response->message = "planner busy";
    RCLCPP_WARN(get_logger(), "【任务 · 关节规划】未开始 ｜ 错误：planner busy");
    return;
  }
  const auto started = Clock::now();
  const auto arm = arm_text(request->group_name);
  RCLCPP_INFO(get_logger(), "──────── 任务开始 · 关节规划 · %s ────────", arm.c_str());
  try {
    if (execution->active()) throw std::runtime_error("execution busy");
    execution->clear();
    auto initial = state(), current = initial;
    update_world(request->group_name, current);
    std::vector<Trajectory> parts;
    for (const auto &waypoint : request->waypoints) {
      auto target = current;
      const auto &group_joints = robot->profile.groups.at(request->group_name);
      if (waypoint.positions.size() != group_joints.size())
        throw std::invalid_argument("joint waypoint count mismatch");
      for (size_t j = 0; j < group_joints.size(); ++j)
        target[group_joints[j]] = waypoint.positions[j];
      std::map<std::string, Transform> goals;
      for (const auto &tool : robot->profile.tools)
        if (request->group_name == "both_arms" || tool.first == request->group_name)
          goals[tool.second] = robot->tool(target, tool.first);
      publish_goals("joint", goals);
      auto result = planner->plan_joint(request->group_name, current, waypoint.positions);
      if (!result.success) throw std::runtime_error(result.message);
      for (size_t j = 0; j < result.trajectory.joint_names.size(); ++j)
        current[result.trajectory.joint_names[j]] = result.trajectory.points.back().positions[j];
      parts.push_back(std::move(result.trajectory));
    }
    if (parts.empty()) throw std::runtime_error("empty waypoints");
    auto trajectory =
        blend_trajectories(parts, robot->profile.groups.at(request->group_name),
                           planner->dynamic_limits(request->group_name),
                           std::vector<bool>(parts.size(), false))
            .trajectory;
    planner->validate(request->group_name, initial, trajectory);
    publish(request->group_name, initial, trajectory);
    execution->submit(request->group_name, trajectory, [] { return false; });
    response->success = true;
    response->message = "planning success";
    RCLCPP_INFO(get_logger(), "【任务】成功 ｜ 耗时 %.1f 毫秒", elapsed_ms(started));
    RCLCPP_INFO(get_logger(), "──────── 任务结束 · 关节规划 · 成功 ────────");
  } catch (const std::exception &error) {
    response->message = error.what();
    RCLCPP_ERROR(get_logger(), "【任务】失败 ｜ 耗时 %.1f 毫秒 ｜ 错误：%s", elapsed_ms(started),
                 error.what());
    RCLCPP_INFO(get_logger(), "──────── 任务结束 · 关节规划 · 失败 ────────");
  }
}
// 逐个笛卡尔路点规划。段与段无法连续通过时改成一次多路点求解。
void PlanningNode::plan_pose(const std::shared_ptr<interfaces::srv::PlanPoseGoal::Request> request,
                             std::shared_ptr<interfaces::srv::PlanPoseGoal::Response> response) {
  std::unique_lock<std::mutex> lock(operation, std::try_to_lock);
  if (!lock.owns_lock() || busy_) {
    response->message = "planner busy";
    RCLCPP_WARN(get_logger(), "【任务 · 位姿规划】未开始 ｜ 错误：planner busy");
    return;
  }
  const auto started = Clock::now();
  const auto arm = arm_text(request->group_name);
  RCLCPP_INFO(get_logger(), "──────── 任务开始 · 位姿规划 · %s ────────", arm.c_str());
  try {
    if (execution->active()) throw std::runtime_error("execution busy");
    execution->clear();
    auto initial = state(), current = initial;
    update_world(request->group_name, current);
    if (request->waypoints.empty() ||
        (!request->other_waypoints.empty() &&
         request->other_waypoints.size() != request->waypoints.size()))
      throw std::runtime_error("invalid pose waypoint counts");
    std::string tool = request->ee_link;
    if (tool.empty())
      tool = robot->profile.tools.at(request->group_name == "both_arms" ? "left_arm"
                                                                        : request->group_name);
    bool known_tool = false;
    for (const auto &entry : robot->profile.tools)
      if (entry.second == tool &&
          (request->group_name == "both_arms" || request->group_name == entry.first))
        known_tool = true;
    if (!known_tool)
      throw std::invalid_argument("tool frame does not belong to the requested group");
    if (!request->other_waypoints.empty() && request->group_name != "both_arms")
      throw std::invalid_argument("paired Cartesian waypoints require both_arms");
    const std::string opposite = tool == robot->profile.tools.at("left_arm")
                                     ? robot->profile.tools.at("right_arm")
                                     : robot->profile.tools.at("left_arm");
    const std::string other_tool =
        request->other_ee_link.empty() ? opposite : request->other_ee_link;
    if (!request->other_waypoints.empty() && other_tool != opposite)
      throw std::invalid_argument("secondary tool must be the opposite arm");
    std::vector<Trajectory> parts;
    std::vector<std::map<std::string, Transform>> route;
    for (size_t i = 0; i < request->waypoints.size(); ++i) {
      std::map<std::string, Transform> targets{{tool, in_base(request->waypoints[i])}};
      if (!request->other_waypoints.empty())
        targets[other_tool] = in_base(request->other_waypoints[i]);
      route.push_back(targets);
      publish_goals("pose", targets);
      auto result = planner->plan_pose(request->group_name, current, targets);
      if (!result.success) throw std::runtime_error(result.message);
      for (size_t j = 0; j < result.trajectory.joint_names.size(); ++j)
        current[result.trajectory.joint_names[j]] = result.trajectory.points.back().positions[j];
      parts.push_back(std::move(result.trajectory));
    }
    Trajectory trajectory;
    if (parts.size() == 1)
      trajectory = parts.front();
    else {
      try {
        trajectory = blend_trajectories(parts, robot->profile.groups.at(request->group_name),
                                        planner->dynamic_limits(request->group_name),
                                        std::vector<bool>(parts.size(), false))
                         .trajectory;
      } catch (const std::runtime_error &error) {
        if (std::string(error.what()) != "via waypoint cannot be passed continuously") throw;
        auto unified = planner->plan_pose_waypoints(request->group_name, initial, route);
        if (!unified.success) throw std::runtime_error(unified.message);
        trajectory = std::move(unified.trajectory);
      }
    }
    planner->validate(request->group_name, initial, trajectory);
    publish(request->group_name, initial, trajectory);
    execution->submit(request->group_name, trajectory, [] { return false; });
    response->success = true;
    response->message = "planning success";
    RCLCPP_INFO(get_logger(), "【任务】成功 ｜ 耗时 %.1f 毫秒", elapsed_ms(started));
    RCLCPP_INFO(get_logger(), "──────── 任务结束 · 位姿规划 · 成功 ────────");
  } catch (const std::exception &error) {
    response->message = error.what();
    RCLCPP_ERROR(get_logger(), "【任务】失败 ｜ 耗时 %.1f 毫秒 ｜ 错误：%s", elapsed_ms(started),
                 error.what());
    RCLCPP_INFO(get_logger(), "──────── 任务结束 · 位姿规划 · 失败 ────────");
  }
}
// 只返回逆解关节角，不生成轨迹，也不下发控制器。
void PlanningNode::compute_ik(const std::shared_ptr<interfaces::srv::ComputeIK::Request> request,
                              std::shared_ptr<interfaces::srv::ComputeIK::Response> response) {
  std::unique_lock<std::mutex> lock(operation, std::try_to_lock);
  if (!lock.owns_lock() || busy_) {
    response->message = "planner busy";
    return;
  }
  try {
    auto joints = state();
    update_world(request->group_name, joints);
    auto result = planner->inverse_kinematics(
        request->group_name, joints,
        {{robot->profile.tools.at(request->group_name), in_base(request->pose)}});
    response->success = result.success;
    response->message = result.message;
    if (result.success) response->joint_angles = result.trajectory.points.back().positions;
  } catch (const std::exception &error) {
    response->message = error.what();
  }
}
template <class Action>
// 同一时刻只接受一个动作。新目标到来前先收回上一个工作线程。
void PlanningNode::add_action(const std::string &name) {
  actions_.push_back(rclcpp_action::create_server<Action>(
      this, name,
      [this](const auto &, auto goal) {
        if (!std::isfinite(goal->timeout)) return rclcpp_action::GoalResponse::REJECT;
        if constexpr (!std::is_same_v<Action, interfaces::action::FollowMpc>) {
          if (goal->timeout < 0) return rclcpp_action::GoalResponse::REJECT;
          try {
            const auto wait = parameter<double>("gripper_wait_time", 1.);
            if (!std::isfinite(wait) || wait < 0) return rclcpp_action::GoalResponse::REJECT;
          } catch (...) {
            RCLCPP_ERROR(get_logger(), "Invalid gripper_wait_time parameter");
            return rclcpp_action::GoalResponse::REJECT;
          }
        }
        if (busy_.exchange(true)) return rclcpp_action::GoalResponse::REJECT;
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
      },
      [](auto) { return rclcpp_action::CancelResponse::ACCEPT; },
      [this](auto handle) {
        const auto started = Clock::now();
        auto fail = [this, handle](const char *message) noexcept {
          RCLCPP_ERROR(get_logger(), "Action worker failed: %s", message);
          try {
            if (!rclcpp::ok() || !handle->is_active()) return;
            auto result = std::make_shared<typename Action::Result>();
            result->success = false;
            result->message = message;
            if (handle->is_canceling())
              handle->canceled(result);
            else
              handle->abort(result);
          } catch (const std::exception &error) {
            RCLCPP_ERROR(get_logger(), "Failed to finish Action: %s", error.what());
          } catch (...) {
            RCLCPP_ERROR(get_logger(), "Unknown error finishing Action");
          }
        };
        try {
          std::lock_guard<std::mutex> guard(workers_mutex_);
          if (worker_.joinable()) worker_.join();
          worker_ = std::thread([this, handle, started, fail] {
            bool can_cancel_execution = false;
            try {
              double timeout = handle->get_goal()->timeout;
              if constexpr (!std::is_same_v<Action, interfaces::action::FollowMpc>) {
                if (timeout == 0.) timeout = parameter<double>("pick_place_timeout", 180.);
                if (!std::isfinite(timeout) || timeout <= 0.)
                  throw std::invalid_argument("pick_place_timeout must be positive and finite");
              }
              auto check = [&] {
                if (shutdown_ || !rclcpp::ok() || handle->is_canceling())
                  throw std::runtime_error("Action cancelled");
                if (timeout > 0. && elapsed_ms(started) >= timeout * 1000.)
                  throw std::runtime_error("Action timeout");
              };
              std::unique_lock<std::mutex> lock(operation, std::defer_lock);
              while (!lock.try_lock()) {
                check();
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
              }
              check();
              can_cancel_execution = !execution->active();
              if constexpr (std::is_same_v<Action, interfaces::action::FollowMpc>)
                follow_mpc(handle, started);
              else
                run_action<Action>(handle, timeout, started);
            } catch (const std::exception &error) {
              if (can_cancel_execution) execution->cancel();
              fail(error.what());
            } catch (...) {
              if (can_cancel_execution) execution->cancel();
              fail("Unknown Action exception");
            }
            busy_ = false;
          });
        } catch (const std::exception &error) {
          fail(error.what());
          busy_ = false;
        } catch (...) {
          fail("Failed to start Action worker");
          busy_ = false;
        }
      },
      rcl_action_server_get_default_options(), callbacks_));
}
template <class Action>
// 把抓取、放置或抓放目标交给 manipulate，再按结果结束动作。
void PlanningNode::run_action(std::shared_ptr<rclcpp_action::ServerGoalHandle<Action>> handle,
                              double timeout, Clock::time_point started) {
  auto goal = handle->get_goal();
  std::string task_id;
  constexpr char hex[] = "0123456789abcdef";
  for (auto byte : handle->get_goal_id()) {
    task_id += hex[byte >> 4];
    task_id += hex[byte & 15];
  }
  auto cancelled = [&] {
    if (shutdown_ || !rclcpp::ok() || handle->is_canceling()) return true;
    std::lock_guard<std::mutex> guard(block_cancel_mutex_);
    return block_cancel_task_id_ == task_id;
  };
  auto feedback = [&](const std::string &stage, const std::string &message) {
    auto value = std::make_shared<typename Action::Feedback>();
    value->stage = stage;
    value->message = message;
    handle->publish_feedback(value);
  };
  TaskResult outcome;
  if constexpr (std::is_same_v<Action, interfaces::action::Pick>)
    outcome = manipulate(&goal->task, nullptr, timeout, cancelled, feedback, started, task_id,
                         goal->stop_at_pick);
  else if constexpr (std::is_same_v<Action, interfaces::action::Place>)
    outcome = manipulate(nullptr, &goal->task, timeout, cancelled, feedback, started, task_id);
  else
    outcome = manipulate(&goal->pick, &goal->place, timeout, cancelled, feedback, started, task_id);
  auto result = std::make_shared<typename Action::Result>();
  result->success = outcome.success;
  result->execution_requested = outcome.execution_requested;
  result->execution_completed = outcome.execution_completed;
  result->stage = outcome.stage;
  result->message = outcome.message;
  result->trajectory = std::move(outcome.trajectory);
  result->trajectories = std::move(outcome.trajectories);
  result->trajectory_stages = std::move(outcome.trajectory_stages);
  if (!rclcpp::ok()) return;
  if (handle->is_canceling())
    handle->canceled(result);
  else if (result->success)
    handle->succeed(result);
  else
    handle->abort(result);
}
}  // namespace curobo_ros_cpp
