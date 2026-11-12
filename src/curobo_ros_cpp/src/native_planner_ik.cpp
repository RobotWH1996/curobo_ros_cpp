// SPDX-License-Identifier: Apache-2.0
// 计算工具位姿与雅可比，构造工具目标并求逆解。
#include "curobo_ros_cpp/detail/native_planner_group.hpp"
#include <algorithm>

namespace curobo_ros_cpp {
// 用该组运动学求工具位姿，不经过 FCL。
std::vector<Transform> NativePlanner::forward(const std::string &name, const JointMap &state) {
  synchronize(name, state);
  auto &group = get(name);
  group.start.upload(positions(group, state));
  group.kinematics->forward(group.start.get(), group.fk_position.get(), group.fk_quaternion.get(),
                            group.fk_jacobian.get(), 1, 1);
  const auto p = group.fk_position.host(group.tools * 3),
             q = group.fk_quaternion.host(group.tools * 4);
  std::vector<Transform> result;
  for (int i = 0; i < group.tools; ++i) {
    Transform value = Transform::Identity();
    value.translation() = Eigen::Vector3d(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
    value.linear() = Eigen::Quaterniond(q[i * 4], q[i * 4 + 1], q[i * 4 + 2], q[i * 4 + 3])
                         .normalized()
                         .toRotationMatrix();
    result.push_back(value);
  }
  return result;
}
// 读取工具位置雅可比，供外部分析，规划主路径不依赖它。
Eigen::MatrixXd NativePlanner::jacobian(const std::string &name, const JointMap &state,
                                        const std::vector<std::string> &tools) {
  forward(name, state);
  auto &group = get(name);
  const auto data = group.fk_jacobian.host(group.tools * 6 * group.dof);
  Eigen::MatrixXd result(tools.size() * 6, group.dof);
  for (size_t frame = 0; frame < tools.size(); ++frame) {
    auto found = std::find(group.tool_names.begin(), group.tool_names.end(), tools[frame]);
    if (found == group.tool_names.end()) throw std::runtime_error("unknown tool frame");
    const auto offset = (found - group.tool_names.begin()) * 6 * group.dof;
    for (int row = 0; row < 6; ++row)
      for (int column = 0; column < group.dof; ++column)
        result(frame * 6 + row, column) = data[offset + row * group.dof + column];
  }
  return result;
}
// 把工具目标位置和四元数写进逆解缓冲。没给目标的工具用本求解器的正运动学，避免和 URDF 差一截。
void NativePlanner::fill_targets(Group &group, const JointMap &state,
                                 const std::map<std::string, Transform> &targets) {
  for (const auto &target : targets)
    if (std::find(group.tool_names.begin(), group.tool_names.end(), target.first) ==
        group.tool_names.end())
      throw std::invalid_argument("unknown target tool frame " + target.first);
  group.start.upload(positions(group, state));
  group.kinematics->forward(group.start.get(), group.fk_position.get(), group.fk_quaternion.get(),
                            group.fk_jacobian.get(), 1, 1);
  const auto current_position = group.fk_position.host(group.tools * 3);
  const auto current_quaternion = group.fk_quaternion.host(group.tools * 4);
  std::vector<Transform> current(group.tools, Transform::Identity());
  for (int i = 0; i < group.tools; ++i) {
    current[i].translation() =
        Eigen::Vector3d(current_position[i * 3], current_position[i * 3 + 1], current_position[i * 3 + 2]);
    current[i].linear() = Eigen::Quaterniond(current_quaternion[i * 4], current_quaternion[i * 4 + 1],
                                             current_quaternion[i * 4 + 2], current_quaternion[i * 4 + 3])
                              .normalized()
                              .toRotationMatrix();
  }
  std::vector<float> p, q, weights(group.tools, 0.F), signs(group.tools, 1.F);
  for (int i = 0; i < group.tools; ++i) {
    const auto &name = group.tool_names[i];
    const auto found = targets.find(name);
    const auto &value = found == targets.end() ? current[i] : found->second;
    const Eigen::Quaterniond rotation(value.linear());
    p.insert(p.end(), {static_cast<float>(value.translation().x()),
                       static_cast<float>(value.translation().y()),
                       static_cast<float>(value.translation().z())});
    q.insert(q.end(), {static_cast<float>(rotation.w()), static_cast<float>(rotation.x()),
                       static_cast<float>(rotation.y()), static_cast<float>(rotation.z())});
    const bool moving_tool = found != targets.end() &&
                             ((found->second.translation() - current[i].translation()).norm() >
                                  settings.position_tolerance ||
                              Eigen::Quaterniond(found->second.linear())
                                      .angularDistance(Eigen::Quaterniond(current[i].linear())) >
                                  settings.orientation_tolerance);
    if (settings.vertical && moving_tool) {
      weights[i] = 10000.F;
      signs[i] = value.linear()(2, 2) >= 0 ? 1.F : -1.F;
    }
  }
  group.ik_goals.upload(p);
  group.ik_quaternions.upload(q);
  group.planner->set_tool_upright(weights, signs);
}

// 求一帧逆解。碰撞开关只影响这一次调用。
PlanResult NativePlanner::inverse_kinematics(const std::string &name, const JointMap &state,
                                             const std::map<std::string, Transform> &targets,
                                             bool collision) {
  synchronize(name, state);
  auto &group = get(name);
  fill_targets(group, state, targets);
  const auto start = positions(group, state);
  group.ik_current.upload(start);
  std::vector<float> seeds;
  for (int i = 0; i < group.ik_seeds; ++i) seeds.insert(seeds.end(), start.begin(), start.end());
  group.ik_seeds_buffer.upload(seeds);
  curobo::robot::IkSolveRequest request;
  request.goal_positions = group.ik_goals.get();
  request.goal_quaternions = group.ik_quaternions.get();
  request.seed_positions = collision ? nullptr : group.ik_seeds_buffer.get();
  request.current_positions = group.ik_current.get();
  request.solution = group.ik_solution.get();
  request.success = group.ik_success.get();
  request.position_error = group.ik_position_error.get();
  request.orientation_error = group.ik_orientation_error.get();
  request.collision_cost = group.ik_collision.get();
  request.num_solutions = collision ? group.ik_seeds : 1;
  const auto before = Clock::now();
  if (collision)
    group.ik->solve(request);
  else
    group.contact_ik->solve(request);
  auto success = group.ik_success.host(request.num_solutions);
  auto solutions = group.ik_solution.host(request.num_solutions * group.dof);
  if (!collision &&
      std::none_of(success.begin(), success.end(), [](auto value) { return value != 0; })) {
    request.num_solutions = 4;
    group.refine_ik->solve(request);
    success = group.ik_success.host(4);
    solutions = group.ik_solution.host(4 * group.dof);
    std::vector<int> order{0, 1, 2, 3};
    auto distance = [&](int row) {
      double value = 0.;
      for (int j = 0; j < group.dof; ++j)
        value += std::pow(solutions[row * group.dof + j] - start[j], 2);
      return value;
    };
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return distance(a) < distance(b); });
    auto flags = success;
    auto rows = solutions;
    for (int i = 0; i < 4; ++i) {
      success[i] = flags[order[i]];
      std::copy_n(rows.begin() + order[i] * group.dof, group.dof,
                  solutions.begin() + i * group.dof);
    }
  }

  PlanResult result;
  result.timing.ik_ms = elapsed_ms(before);
  result.trajectory.joint_names = group.names;
  for (int seed = 0; seed < request.num_solutions; ++seed)
    if (success[seed]) {
      JointMap full = state;
      for (int j = 0; j < group.dof; ++j) full[group.names[j]] = solutions[seed * group.dof + j];
      if (collision && robot_.check(full, name, true, false, payload_)) continue;
      trajectory_msgs::msg::JointTrajectoryPoint point;
      for (int j = 0; j < group.dof; ++j)
        point.positions.push_back(solutions[seed * group.dof + j]);
      result.trajectory.points.push_back(std::move(point));
      result.success = true;
      result.message = "IK success";
      break;
    }
  if (!result.success) result.message = "IK failed reachability, collision or pose tolerance";
  return result;
}
}  // namespace curobo_ros_cpp
