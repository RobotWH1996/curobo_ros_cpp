// SPDX-License-Identifier: Apache-2.0
// 回放记录的关节、TF 和点云，验证规划、抓放和 MPC 接口。
#include <tf2_ros/static_transform_broadcaster.h>
#include <std_srvs/srv/set_bool.hpp>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include "curobo_ros_cpp/node.hpp"
// 回放一份抓放快照，确认规划结果和当时记录的轨迹一致。不下发控制器。
using namespace curobo_ros_cpp;
// 从快照 YAML 读一个位姿。缺字段时保持零。
geometry_msgs::msg::Pose read_pose(const YAML::Node &value) {
  geometry_msgs::msg::Pose p;
  if (value["position"]) {
    auto v = vector3(YAML::Load("[" + value["position"]["x"].as<std::string>() + "," +
                                value["position"]["y"].as<std::string>() + "," +
                                value["position"]["z"].as<std::string>() + "]"));
    p.position.x = v.x();
    p.position.y = v.y();
    p.position.z = v.z();
  }
  if (value["orientation"]) {
    p.orientation.x = value["orientation"]["x"].as<double>();
    p.orientation.y = value["orientation"]["y"].as<double>();
    p.orientation.z = value["orientation"]["z"].as<double>();
    p.orientation.w = value["orientation"]["w"].as<double>();
  }
  return p;
}
// 带坐标系的位姿。时间戳不参与回放比较。
geometry_msgs::msg::PoseStamped read_stamped(const YAML::Node &value) {
  geometry_msgs::msg::PoseStamped p;
  p.header.frame_id = value["header"]["frame_id"].as<std::string>();
  p.pose = read_pose(value["pose"]);
  return p;
}
// 关节路点或位姿路点。空节点表示没有终点。
interfaces::msg::MotionWaypoint read_waypoint(const YAML::Node &value) {
  interfaces::msg::MotionWaypoint waypoint;
  if (!value) return waypoint;
  waypoint.type = value["type"].as<std::string>("");
  if (value["joint_positions"])
    waypoint.joint_positions = value["joint_positions"].as<std::vector<double>>();
  if (value["pose"]) waypoint.pose = read_stamped(value["pose"]);
  return waypoint;
}
template <class Task>
// 抓取和放置消息里相同的字段。
void read_common(Task &task, const YAML::Node &value) {
  task.group_name = value["group_name"].as<std::string>();
  task.tool_pose = read_stamped(value["tool_pose"]);
  task.in_base_frame = value["in_base_frame"].as<bool>();
  task.open_position = value["open_position"].as<double>();
  task.object_size.x = value["object_size"]["x"].as<double>();
  task.object_size.y = value["object_size"]["y"].as<double>();
  task.object_size.z = value["object_size"]["z"].as<double>();
  task.end_pose = read_waypoint(value["end_pose"]);
  for (auto waypoint : value["waypoints"]) task.waypoints.push_back(read_waypoint(waypoint));
}
// 依次回放关节规划、位姿规划、逆解和抓放，打印与快照的差异。
int main(int argc, char **argv) {
  const auto process = Clock::now();
  rclcpp::init(argc, argv);
  int status = 0;
  try {
    const char *directory = std::getenv("CUROBO_SNAPSHOT_DIR");
    if (!directory || !*directory)
      throw std::runtime_error("set CUROBO_SNAPSHOT_DIR to the case.yaml and cloud.bin directory");
    const std::filesystem::path fixture_directory(directory);
    const auto fixture = YAML::LoadFile((fixture_directory / "case.yaml").string());
    rclcpp::NodeOptions options;
    options.append_parameter_override("execute_trajectory", false);
    options.append_parameter_override("pointcloud_prewarm", false);
    auto node = std::make_shared<PlanningNode>(options);
    if (node->execution->enabled())
      throw std::runtime_error("validation requires execute_trajectory=false");
    std::cout << "READY_WALL_MS " << elapsed_ms(process) << std::endl;
    auto replay = std::make_shared<rclcpp::Node>("snapshot_replay");
    auto js_pub = replay->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
    auto cloud_pub = replay->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/all_points", rclcpp::SensorDataQoS());
    tf2_ros::StaticTransformBroadcaster broadcaster(replay);
    auto captured = fixture["capture"];
    sensor_msgs::msg::JointState joints;
    joints.name = strings(captured["joints"]["name"]);
    joints.position = captured["joints"]["position"].as<std::vector<double>>();
    sensor_msgs::msg::PointCloud2 cloud;
    auto c = captured["cloud"];
    cloud.header.frame_id = c["header"]["frame_id"].as<std::string>();
    cloud.width = c["width"].as<uint32_t>();
    cloud.height = c["height"].as<uint32_t>();
    cloud.point_step = c["point_step"].as<uint32_t>();
    cloud.row_step = c["row_step"].as<uint32_t>();
    cloud.is_bigendian = c["is_bigendian"].as<bool>();
    cloud.is_dense = c["is_dense"].as<bool>();
    for (const auto &field : c["fields"]) {
      sensor_msgs::msg::PointField f;
      f.name = field["name"].as<std::string>();
      f.offset = field["offset"].as<uint32_t>();
      f.datatype = static_cast<uint8_t>(field["datatype"].as<unsigned>());
      f.count = field["count"].as<uint32_t>();
      cloud.fields.push_back(f);
    }
    std::ifstream bytes(fixture_directory / "cloud.bin", std::ios::binary);
    if (!bytes) throw std::runtime_error("snapshot cloud.bin cannot be opened");
    cloud.data = {std::istreambuf_iterator<char>(bytes), std::istreambuf_iterator<char>()};
    geometry_msgs::msg::TransformStamped tf;
    auto t = captured["cloud_tf"];
    tf.header.frame_id = t["header"]["frame_id"].as<std::string>();
    tf.child_frame_id = t["child_frame_id"].as<std::string>();
    auto v = t["transform"]["translation"], q = t["transform"]["rotation"];
    tf.transform.translation.x = v["x"].as<double>();
    tf.transform.translation.y = v["y"].as<double>();
    tf.transform.translation.z = v["z"].as<double>();
    tf.transform.rotation.w = q["w"].as<double>();
    tf.transform.rotation.x = q["x"].as<double>();
    tf.transform.rotation.y = q["y"].as<double>();
    tf.transform.rotation.z = q["z"].as<double>();
    broadcaster.sendTransform(tf);
    std::atomic_bool publish_joints{true};
    std::atomic_bool inject_old_cloud{false};
    auto timer = replay->create_wall_timer(std::chrono::milliseconds(100), [&] {
      joints.header.stamp = replay->now();
      cloud.header.stamp = replay->now();
      if (publish_joints) js_pub->publish(joints);
      cloud_pub->publish(cloud);
      if (inject_old_cloud) {
        auto old = cloud;
        old.header.stamp = rclcpp::Time(cloud.header.stamp) - rclcpp::Duration::from_seconds(.18);
        old.width = old.height = old.row_step = 0;
        old.data.clear();
        cloud_pub->publish(old);
      }
    });
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 6);
    executor.add_node(node);
    executor.add_node(replay);
    std::thread thread([&] { executor.spin(); });
    try {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      inject_old_cloud = true;
      std::this_thread::sleep_for(std::chrono::milliseconds(350));
      if (node->update_world("left_arm").raw_count != static_cast<size_t>(c["width"].as<uint32_t>()) *
                                                       c["height"].as<uint32_t>())
        throw std::runtime_error("late capture replaced a newer pointcloud");
      inject_old_cloud = false;
      interfaces::msg::PickTask pick;
      read_common(pick, fixture["pick"]);
      pick.close_position = fixture["pick"]["close_position"].as<double>();
      pick.pick_pose = read_pose(fixture["pick"]["pick_pose"]);
      pick.prepick_in_target = read_pose(fixture["pick"]["prepick_in_target"]);
      pick.lift_in_target = read_pose(fixture["pick"]["lift_in_target"]);
      interfaces::msg::PlaceTask place;
      read_common(place, fixture["place"]);
      place.place_pose = read_pose(fixture["place"]["place_pose"]);
      place.preplace_in_target = read_pose(fixture["place"]["preplace_in_target"]);
      place.retreat_in_target = read_pose(fixture["place"]["retreat_in_target"]);
      auto joint_map = node->state();
      auto gpu_fk = node->planner->forward("left_arm", joint_map);
      const std::filesystem::path artifacts(node->get_parameter("artifacts_directory").as_string());
      const auto knots = std::to_string(node->get_parameter("trajopt_num_control_points").as_int());
      const auto group = artifacts / "left_arm";
      const auto role = group / "opt" / knots;
      const auto path = std::filesystem::exists(role) ? role : group / knots;
      const auto artifact = YAML::LoadFile((path / "robot_kinematics_manifest.json").string());
      const auto tools = strings(artifact["tool_frames"]);
      const auto cpu_fk = node->robot->forward(joint_map);
      for (size_t i = 0; i < tools.size(); ++i) {
        const auto error = (gpu_fk[i].translation() - cpu_fk.at(tools[i]).translation()).norm();
        std::cout << "FK_ERROR " << tools[i] << " " << error << std::endl;
        if (error > 1e-5) throw std::runtime_error("native fixed transform FK mismatch");
      }
      for (int run = 0; run < 3; ++run) {
        const auto start = Clock::now();
        auto result = node->manipulate(
            &pick, nullptr, 180., [] { return false; },
            [](const auto &stage, const auto &) { std::cout << "STAGE " << stage << std::endl; });
        std::cout << "PICK_RUN " << run << " success=" << result.success
                  << " wall_ms=" << elapsed_ms(start) << " process_ms=" << elapsed_ms(process)
                  << " stage=" << result.stage << " message=" << result.message
                  << " points=" << result.trajectory.points.size() << std::endl;
        if (!result.success) {
          status = 1;
          break;
        }
        std::vector<std::string> expected_names;
        for (const auto &name : node->robot->full_joint_names)
          if (name.rfind("Wheel", 0) != 0) expected_names.push_back(name);
        if (expected_names.size() == node->robot->full_joint_names.size())
          throw std::runtime_error("wheel output regression requires a model with wheels");
        auto check_output = [&](const Trajectory &trajectory) {
          if (trajectory.joint_names != expected_names)
            throw std::runtime_error("task output lost body/gripper joints or retained wheels");
          for (const auto &point : trajectory.points)
            if (point.positions.size() != expected_names.size() ||
                point.velocities.size() != expected_names.size() ||
                point.accelerations.size() != expected_names.size())
              throw std::runtime_error("task output columns differ after wheel projection");
        };
        check_output(result.trajectory);
        for (const auto &trajectory : result.trajectories) check_output(trajectory);
        std::cout << "TASK_OUTPUT_NO_WHEELS_PASS " << expected_names.size() << std::endl;
      }

      auto invalid_pick = pick;
      invalid_pick.close_position = invalid_pick.open_position;
      const auto bad_pick_result = node->manipulate(&invalid_pick, nullptr, 180., [] { return false; },
                                                    [](const auto &, const auto &) {});
      if (bad_pick_result.success) throw std::runtime_error("invalid pick goal was accepted");
      auto confirm = replay->create_client<std_srvs::srv::SetBool>("/confirm_held_object");
      for (const bool holding : {true, false}) {
        auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
        request->data = holding;
        auto reply = confirm->async_send_request(request);
        if (reply.wait_for(std::chrono::seconds(2)) != std::future_status::ready || !reply.get()->success)
          throw std::runtime_error("invalid request lost the preceding successful pick");
      }
      const auto both_start = node->robot->check(joint_map, "both_arms", false);
      std::cout << "BOTH_START_FCL " << (both_start ? both_start->message() : "clear") << std::endl;
      Trajectory both_hold;
      both_hold.joint_names = node->robot->profile.groups.at("both_arms");
      trajectory_msgs::msg::JointTrajectoryPoint hold_point;
      for (const auto &name : both_hold.joint_names)
        hold_point.positions.push_back(joint_map.at(name));
      hold_point.velocities.resize(14, 0.);
      hold_point.accelerations.resize(14, 0.);
      both_hold.points.push_back(hold_point);
      hold_point.time_from_start = duration(.2);
      both_hold.points.push_back(hold_point);
      node->planner->clear_world();
      try {
        node->planner->validate("both_arms", joint_map, both_hold);
        std::cout << "BOTH_START_GPU clear" << std::endl;
      } catch (const std::exception &error) {
        std::cout << "BOTH_START_GPU " << error.what() << std::endl;
      }
      const auto transform_map = node->robot->forward(joint_map);
      const auto moving = node->robot->moving_links("left_arm");
      std::vector<Point> surface_points;
      std::vector<std::string> surface_links;
      for (const auto &mesh : node->robot->meshes())
        if (moving.count(mesh.link)) {
          surface_points.push_back(
              (transform_map.at(mesh.link) * mesh.vertices.front()).cast<float>());
          surface_links.push_back(mesh.link);
        }
      bool timed_out = false;
      auto mask =
          node->robot->self_mask(surface_points, joint_map, "left_arm", .005, 300., timed_out);
      if (timed_out || std::any_of(mask.begin(), mask.end(), [](bool value) { return !value; }))
        throw std::runtime_error("moving-arm surface filtering failed");
      std::cout << "SELF_FILTER_SURFACE_PASS " << mask.size() << std::endl;
      sensor_msgs::msg::PointCloud2 labeled;
      labeled.height = 1;
      labeled.width = 2;
      labeled.point_step = 16;
      labeled.row_step = 32;
      labeled.data.resize(32);
      for (int axis = 0; axis < 4; ++axis) {
        sensor_msgs::msg::PointField field;
        field.name = axis == 0 ? "x" : axis == 1 ? "y" : axis == 2 ? "z" : "label";
        field.offset = axis * 4;
        field.count = 1;
        field.datatype = axis == 3 ? field.UINT32 : field.FLOAT32;
        labeled.fields.push_back(field);
      }
      for (int i = 0; i < 2; ++i) {
        for (int axis = 0; axis < 3; ++axis) {
          const float value = surface_points.front()[axis];
          std::memcpy(labeled.data.data() + i * 16 + axis * 4, &value, 4);
        }
        const uint32_t label = i == 0 ? 0 : 5;
        std::memcpy(labeled.data.data() + i * 16 + 12, &label, 4);
      }
      PointCloudSettings protected_settings;
      protected_settings.filter = true;
      protected_settings.min_range = 0.;
      protected_settings.max_range = 10.;
      protected_settings.lower = Eigen::Vector3d::Constant(-10.);
      protected_settings.upper = Eigen::Vector3d::Constant(10.);
      protected_settings.self_filter_padding = .005;
      auto protected_cloud = prepare_cloud(labeled, Transform::Identity(), protected_settings,
                                           *node->robot, joint_map, "left_arm");
      if (protected_cloud.points.size() != 1 || protected_cloud.removed.size() != 1)
        throw std::runtime_error("labeled target lost during body filtering or voxel sampling");
      std::cout << "LABEL_PROTECTION_PASS" << std::endl;
      // 在实际机器人表面附近找同体素内的一对本体点和环境点。
      Point body_point, obstacle_point;
      bool mixed_voxel_found = false;
      for (const auto &surface : surface_points) {
        const float pitch = protected_settings.pitch;
        const Point corner = (surface.array() / pitch).floor().matrix() * pitch;
        std::vector<Point> candidates;
        for (int x = 1; x <= 9; ++x)
          for (int y = 1; y <= 9; ++y)
            for (int z = 1; z <= 9; ++z)
              candidates.push_back(corner + Point(x, y, z) * (pitch / 10.F));
        bool exhausted = false;
        const auto classification = node->robot->self_mask(
            candidates, joint_map, "left_arm", protected_settings.self_filter_padding, 0.,
            exhausted);
        const auto body = std::find(classification.begin(), classification.end(), true);
        const auto obstacle = std::find(classification.begin(), classification.end(), false);
        if (body == classification.end() || obstacle == classification.end()) continue;
        body_point = candidates[body - classification.begin()];
        obstacle_point = candidates[obstacle - classification.begin()];
        mixed_voxel_found = true;
        break;
      }
      if (!mixed_voxel_found) throw std::runtime_error("mixed voxel fixture not found");
      auto mixed_cloud = labeled;
      for (const bool filtering : {false, true}) {
        auto settings = protected_settings;
        settings.filter = filtering;
        for (const bool body_first : {false, true}) {
          const std::array<Point, 2> points = body_first
                                                ? std::array<Point, 2>{body_point, obstacle_point}
                                                : std::array<Point, 2>{obstacle_point, body_point};
          for (size_t i = 0; i < points.size(); ++i) {
            std::memcpy(mixed_cloud.data.data() + i * 16, points[i].data(), 12);
            const uint32_t label = 0;
            std::memcpy(mixed_cloud.data.data() + i * 16 + 12, &label, 4);
          }
          const auto prepared = prepare_cloud(mixed_cloud, Transform::Identity(), settings,
                                                *node->robot, joint_map, "left_arm");
          if (prepared.points.size() != 1 ||
              (prepared.points.front() - obstacle_point).norm() > 1e-7 ||
              voxel_mesh(prepared.points, settings.pitch).triangles.empty())
            throw std::runtime_error("mixed voxel lost its obstacle after body filtering");
        }
      }
      for (const std::string mode : {"off", "observe", "apply"}) {
        auto settings = protected_settings;
        settings.self_filter_mode = mode;
        const auto prepared = prepare_cloud(labeled, Transform::Identity(), settings,
                                              *node->robot, joint_map, "left_arm");
        const size_t expected = mode == "apply" ? 1 : 2;
        if (prepared.points.size() != expected)
          throw std::runtime_error("label protection or self-filter mode changed");
      }
      std::cout << "MIXED_VOXEL_ORDER_PASS" << std::endl;

      auto modified = joint_map;
      modified["Waist_joint"] += .08;
      modified["Head_joint"] -= .05;
      modified["right_gripper"] = -.01;
      auto updated = node->planner->forward("left_arm", modified);
      const auto expected = node->robot->forward(modified);
      for (size_t i = 0; i < tools.size(); ++i)
        if ((updated[i].translation() - expected.at(tools[i]).translation()).norm() > 1e-5)
          throw std::runtime_error("feedback-only joint transform update failed");
      std::cout << "FIXED_FEEDBACK_PASS" << std::endl;

      Trajectory held;
      held.joint_names = node->robot->profile.groups.at("left_arm");
      trajectory_msgs::msg::JointTrajectoryPoint fixed;
      for (const auto &name : held.joint_names) fixed.positions.push_back(joint_map.at(name));
      fixed.velocities.resize(held.joint_names.size(), 0.);
      fixed.accelerations.resize(held.joint_names.size(), 0.);
      held.points.push_back(fixed);
      fixed.time_from_start = duration(.2);
      held.points.push_back(fixed);
      const auto all_transforms = node->robot->forward(joint_map);
      const auto right_mesh = std::find_if(node->robot->meshes().begin(),
                                           node->robot->meshes().end(), [](const auto &mesh) {
                                             return mesh.link.find("08R-") != std::string::npos &&
                                                    mesh.link.find("link7") != std::string::npos;
                                           });
      const auto left_mesh = std::find_if(node->robot->meshes().begin(),
                                          node->robot->meshes().end(), [](const auto &mesh) {
                                            return mesh.link.find("08L-") != std::string::npos &&
                                                   mesh.link.find("link5") != std::string::npos;
                                          });
      if (right_mesh == node->robot->meshes().end() || left_mesh == node->robot->meshes().end())
        throw std::runtime_error("collision fixture link missing");
      Point right_point =
          (all_transforms.at(right_mesh->link) * (.5 * (right_mesh->lower + right_mesh->upper)))
              .cast<float>();
      node->planner->set_world({right_point}, .02, "mesh", Eigen::Vector3d(.2, 0, 1),
                               Eigen::Vector3d(2, 2, 2));
      node->planner->validate("left_arm", joint_map, held);
      std::cout << "STATIC_WORLD_MASK_PASS" << std::endl;
      Point left_point =
          (all_transforms.at(left_mesh->link) * (.5 * (left_mesh->lower + left_mesh->upper)))
              .cast<float>();
      node->planner->set_world({left_point}, .02, "mesh", Eigen::Vector3d(.2, 0, 1),
                               Eigen::Vector3d(2, 2, 2));
      bool rejected = false;
      try {
        node->planner->validate("left_arm", joint_map, held);
      } catch (const std::exception &error) {
        rejected = true;
        std::cout << "MOVING_WORLD_REJECT " << error.what() << std::endl;
      }
      if (!rejected) throw std::runtime_error("moving-arm obstacle was missed");
      node->planner->set_world({left_point}, .02, "voxel", left_point.cast<double>(),
                               Eigen::Vector3d::Constant(.4));
      bool voxel_rejected = false;
      try {
        node->planner->validate("left_arm", joint_map, held);
      } catch (const std::exception &) {
        voxel_rejected = true;
      }
      if (!voxel_rejected) throw std::runtime_error("voxel obstacle was missed");
      std::cout << "VOXEL_MOVING_REJECT_PASS" << std::endl;
      node->planner->set_world({left_point}, .03, "voxel", Eigen::Vector3d::Constant(3.),
                               Eigen::Vector3d::Constant(.31));
      node->planner->validate("left_arm", joint_map, held);
      std::cout << "VOXEL_BOUNDS_PASS" << std::endl;

      node->planner->clear_world();
      auto pick_client = rclcpp_action::create_client<interfaces::action::Pick>(replay, "/pick");
      if (!pick_client->wait_for_action_server(std::chrono::seconds(2)))
        throw std::runtime_error("pick action unavailable");
      interfaces::action::Pick::Goal pick_goal;
      pick_goal.task = pick;
      pick_goal.timeout = 60.;
      auto pick_ack = pick_client->async_send_goal(pick_goal);
      if (pick_ack.wait_for(std::chrono::seconds(2)) != std::future_status::ready ||
          !pick_ack.get())
        throw std::runtime_error("pick action rejected");
      auto pick_result = pick_client->async_get_result(pick_ack.get());
      if (pick_result.wait_for(std::chrono::seconds(60)) != std::future_status::ready)
        throw std::runtime_error("pick action timeout");
      std::cout << "PICK_ACTION " << pick_result.get().result->success << std::endl;
      if (!pick_result.get().result->success) status = 1;
      {
        std::unique_lock<std::mutex> held_operation(node->operation);
        auto queued_pick = [&](double timeout) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          auto queued = pick_goal;
          queued.timeout = timeout;
          auto ack = pick_client->async_send_goal(queued);
          if (ack.wait_for(std::chrono::seconds(2)) != std::future_status::ready || !ack.get())
            throw std::runtime_error("queued pick rejected");
          return ack.get();
        };
        auto timed = pick_client->async_get_result(queued_pick(.05));
        if (timed.wait_for(std::chrono::seconds(2)) != std::future_status::ready ||
            timed.get().code != rclcpp_action::ResultCode::ABORTED ||
            timed.get().result->message != "Action timeout")
          throw std::runtime_error("queued pick exceeded whole-task timeout");
        auto goal_handle = queued_pick(5.);
        auto cancelled = pick_client->async_get_result(goal_handle);
        auto cancel_ack = pick_client->async_cancel_goal(goal_handle);
        if (cancel_ack.wait_for(std::chrono::seconds(2)) != std::future_status::ready ||
            cancelled.wait_for(std::chrono::seconds(2)) != std::future_status::ready ||
            cancelled.get().code != rclcpp_action::ResultCode::CANCELED)
          throw std::runtime_error("queued pick could not be cancelled");
        std::cout << "ACTION_TIMEOUT_AND_CANCEL_PASS" << std::endl;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      auto service = replay->create_client<interfaces::srv::PlanJointGoal>("/plan_joint_goal");
      auto request = std::make_shared<interfaces::srv::PlanJointGoal::Request>();
      request->group_name = "left_arm";
      interfaces::msg::JointWaypoint a, b;
      for (const auto &name : node->robot->profile.groups.at("left_arm"))
        a.positions.push_back(joint_map.at(name));
      b = a;
      a.positions[0] += .08;
      b.positions[0] += .16;
      request->waypoints = {a, b};
      auto service_future = service->async_send_request(request);
      if (service_future.wait_for(std::chrono::seconds(30)) != std::future_status::ready)
        throw std::runtime_error("joint service timeout");
      auto service_result = service_future.get();
      std::cout << "JOINT_WAYPOINT_SERVICE " << service_result->success << " "
                << service_result->message << std::endl;
      if (!service_result->success) status = 1;
      const auto tool = node->robot->profile.tools.at("left_arm");
      std::vector<geometry_msgs::msg::PoseStamped> pose_waypoints;
      for (const auto &angles : {a.positions, b.positions}) {
        auto target_joints = joint_map;
        const auto &names = node->robot->profile.groups.at("left_arm");
        for (size_t j = 0; j < names.size(); ++j) target_joints[names[j]] = angles[j];
        geometry_msgs::msg::PoseStamped pose;
        pose.header.frame_id = "base_link";
        pose.pose = pose_msg(node->robot->tool(target_joints, "left_arm"));
        pose_waypoints.push_back(pose);
      }
      auto ik_client = replay->create_client<interfaces::srv::ComputeIK>("/compute_ik");
      auto ik_request = std::make_shared<interfaces::srv::ComputeIK::Request>();
      ik_request->group_name = "left_arm";
      ik_request->pose = pose_waypoints.front();
      auto ik_future = ik_client->async_send_request(ik_request);
      if (ik_future.wait_for(std::chrono::seconds(20)) != std::future_status::ready)
        throw std::runtime_error("compute_ik timeout");
      const auto ik_reply = ik_future.get();
      std::cout << "IK_SERVICE " << ik_reply->success << std::endl;
      if (!ik_reply->success) status = 1;
      auto pose_client = replay->create_client<interfaces::srv::PlanPoseGoal>("/plan_pose_goal");
      auto pose_request = std::make_shared<interfaces::srv::PlanPoseGoal::Request>();
      pose_request->group_name = "left_arm";
      pose_request->waypoints = pose_waypoints;
      auto pose_future = pose_client->async_send_request(pose_request);
      if (pose_future.wait_for(std::chrono::seconds(30)) != std::future_status::ready)
        throw std::runtime_error("Cartesian waypoints timeout");
      const auto pose_reply = pose_future.get();
      std::cout << "POSE_WAYPOINT_SERVICE " << pose_reply->success << " " << pose_reply->message
                << std::endl;
      if (!pose_reply->success) status = 1;
      pose_request = std::make_shared<interfaces::srv::PlanPoseGoal::Request>();
      pose_request->group_name = "both_arms";
      pose_request->waypoints = pose_waypoints;
      geometry_msgs::msg::PoseStamped other;
      other.header.frame_id = "base_link";
      other.pose = pose_msg(node->robot->tool(joint_map, "right_arm"));
      pose_request->other_waypoints = {other, other};
      auto both_future = pose_client->async_send_request(pose_request);
      if (both_future.wait_for(std::chrono::seconds(30)) != std::future_status::ready)
        throw std::runtime_error("dual Cartesian waypoints timeout");
      const auto both_reply = both_future.get();
      std::cout << "DUAL_POSE_SERVICE " << both_reply->success << " " << both_reply->message
                << std::endl;
      if (!both_reply->success) status = 1;

      const auto reference = node->in_base(pick.tool_pose);
      auto placement = reference;
      placement.translation().y() += .12;
      place.tool_pose.header.frame_id = "base_link";
      place.tool_pose.pose = pose_msg(placement);
      place.place_pose = pick.pick_pose;
      place.preplace_in_target = pick.prepick_in_target;
      place.retreat_in_target = pick.lift_in_target;
      auto place_client = rclcpp_action::create_client<interfaces::action::Place>(replay, "/place");
      if (!place_client->wait_for_action_server(std::chrono::seconds(2)))
        throw std::runtime_error("place action unavailable");
      interfaces::action::Place::Goal place_goal;
      place_goal.task = place;
      place_goal.timeout = 60.;
      auto place_ack = place_client->async_send_goal(place_goal);
      if (place_ack.wait_for(std::chrono::seconds(2)) != std::future_status::ready ||
          !place_ack.get())
        throw std::runtime_error("place action rejected");
      auto place_result = place_client->async_get_result(place_ack.get());
      if (place_result.wait_for(std::chrono::seconds(60)) != std::future_status::ready)
        throw std::runtime_error("place action timeout");
      auto placed = place_result.get().result;
      std::cout << "PLACE_ACTION " << placed->success << " " << placed->stage << " "
                << placed->message << std::endl;
      if (!placed->success) status = 1;
      auto pair_client =
          rclcpp_action::create_client<interfaces::action::PickPlace>(replay, "/pick_place");
      if (!pair_client->wait_for_action_server(std::chrono::seconds(2)))
        throw std::runtime_error("pick_place action unavailable");
      interfaces::action::PickPlace::Goal pair_goal;
      pair_goal.pick = pick;
      pair_goal.place = place;
      pair_goal.timeout = 60.;
      auto pair_ack = pair_client->async_send_goal(pair_goal);
      if (pair_ack.wait_for(std::chrono::seconds(2)) != std::future_status::ready ||
          !pair_ack.get())
        throw std::runtime_error("pick_place action rejected");
      auto pair_result = pair_client->async_get_result(pair_ack.get());
      if (pair_result.wait_for(std::chrono::seconds(60)) != std::future_status::ready)
        throw std::runtime_error("pick_place action timeout");
      auto paired = pair_result.get().result;
      std::cout << "PICK_PLACE_ACTION " << paired->success << " " << paired->stage << " "
                << paired->message << std::endl;
      if (!paired->success) status = 1;
      node->planner->clear_world();
      auto target = node->robot->tool(joint_map, "left_arm");
      target.translation().z() += .02;
      auto proposal = node->planner->mpc_step("left_arm", joint_map,
                                              {{node->robot->profile.tools.at("left_arm"), target}},
                                              TrajectorySample{}, 0, true);
      std::cout << "MPC_NATIVE " << proposal.success << " " << proposal.message
                << " elapsed_ms=" << proposal.elapsed_ms << std::endl;
      if (proposal.success)
        node->planner->accept_mpc("left_arm");
      else
        status = 1;

      if (proposal.success) {
        const auto handoff = sample_trajectory(proposal.trajectory, proposal.knot_dt);
        auto next = node->planner->mpc_step("left_arm", joint_map,
                                            {{node->robot->profile.tools.at("left_arm"), target}},
                                            handoff, 1, false);
        std::cout << "MPC_HANDOFF " << next.success << " elapsed_ms=" << next.elapsed_ms
                  << std::endl;
        if (next.success)
          node->planner->accept_mpc("left_arm");
        else
          status = 1;
      }

      auto target_pub = replay->create_publisher<interfaces::msg::MpcTarget>("/curobo/target", 10);
      std::atomic_int mpc_messages{0};
      std::atomic_bool stamped_plans{true};
      auto mpc_subscription = replay->create_subscription<interfaces::msg::MpcPlan>(
          "/curobo/mpc_plan", 10, [&](interfaces::msg::MpcPlan::SharedPtr message) {
            if (!message->success) return;
            const auto &trajectory = message->left.trajectory;
            if (trajectory.points.empty() || rclcpp::Time(message->header.stamp).nanoseconds() == 0 ||
                rclcpp::Time(message->header.stamp) != rclcpp::Time(trajectory.header.stamp))
              stamped_plans = false;
            ++mpc_messages;
          });
      interfaces::msg::MpcTarget tracking;
      tracking.has_left = true;
      tracking.left.header.frame_id = "base_link";
      tracking.left.pose = pose_msg(target);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      target_pub->publish(tracking);
      auto tracking_client =
          rclcpp_action::create_client<interfaces::action::FollowMpc>(replay, "/follow_mpc");
      auto track = [&](double timeout, const std::string &expected) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        interfaces::action::FollowMpc::Goal tracking_goal;
        tracking_goal.timeout = timeout;
        auto ack = tracking_client->async_send_goal(tracking_goal);
        if (ack.wait_for(std::chrono::seconds(2)) != std::future_status::ready || !ack.get())
          throw std::runtime_error("follow_mpc rejected");
        auto result = tracking_client->async_get_result(ack.get());
        if (result.wait_for(std::chrono::seconds(10)) != std::future_status::ready)
          throw std::runtime_error("follow_mpc did not finish");
        const auto outcome = result.get();
        if (outcome.code != rclcpp_action::ResultCode::ABORTED || outcome.result->success ||
            outcome.result->message != expected)
          throw std::runtime_error("unexpected MPC result: " + outcome.result->message);
      };
      track(.45, "MPC timeout reached");
      if (mpc_messages == 0 || !stamped_plans)
        throw std::runtime_error("MPC published no valid stamped trajectories");
      std::cout << "MPC_TIMEOUT_AND_STAMP_PASS " << mpc_messages << std::endl;
      target_pub->publish(tracking);
      track(3., "MPC target expired");
      std::cout << "MPC_STALE_TARGET_PASS" << std::endl;

      auto target_timer = replay->create_wall_timer(std::chrono::milliseconds(50),
                                                     [&] { target_pub->publish(tracking); });
      target_pub->publish(tracking);
      publish_joints = false;
      track(3., "MPC joint feedback expired");
      publish_joints = true;
      target_timer->cancel();
      std::cout << "MPC_STALE_JOINTS_PASS" << std::endl;
      const auto feedback_resumed = Clock::now();
      for (;;) {
        try {
          node->state(true);
          break;
        } catch (const std::runtime_error &) {
          if (elapsed_ms(feedback_resumed) > 1000.)
            throw std::runtime_error("joint feedback did not resume");
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }

      if (!node->set_parameter(rclcpp::Parameter("mpc_target_max_age", -1.)).successful)
        throw std::runtime_error("could not set test MPC parameter");
      track(.45, "MPC input max ages must be positive and finite");
      if (!node->set_parameter(rclcpp::Parameter("mpc_target_max_age", 1.)).successful)
        throw std::runtime_error("could not restore test MPC parameter");
      target_pub->publish(tracking);
      track(.45, "MPC timeout reached");
      std::cout << "ACTION_WORKER_EXCEPTION_RECOVERY_PASS" << std::endl;
    } catch (...) {
      executor.cancel();
      thread.join();
      throw;
    }
    executor.cancel();
    thread.join();
  } catch (const std::exception &error) {
    std::cerr << "VALIDATION_ERROR " << error.what() << std::endl;
    status = 1;
  }
  rclcpp::shutdown();
  return status;
}
