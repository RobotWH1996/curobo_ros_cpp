// SPDX-License-Identifier: Apache-2.0
// 生成每拍 MPC 预测轨迹，管理冷启动与已接受的热启动状态。
#include "curobo_ros_cpp/detail/native_planner_group.hpp"
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>

namespace curobo_ros_cpp {
// 推进一拍 MPC。冷启动和热启动用的迭代次数不同。
MpcProposal NativePlanner::mpc_step(const std::string &name, const JointMap &state,
                                    const std::map<std::string, Transform> &targets,
                                    const TrajectorySample &physical_handoff, int executed_knots,
                                    bool reset, double velocity_scale,
                                    const std::vector<double> &cold_seed_goal, bool joint_goal) {
  if (!std::isfinite(velocity_scale) || velocity_scale <= 0. || velocity_scale > 1.)
    throw std::invalid_argument("invalid MPC velocity scale");
  auto handoff = physical_handoff;
  handoff.velocity /= velocity_scale;
  handoff.acceleration /= velocity_scale * velocity_scale;
  handoff.jerk /= velocity_scale * velocity_scale * velocity_scale;
  const auto started = Clock::now();
  synchronize(name, state, true);
  auto &group = get(name, true);
  if (!joint_goal) fill_targets(group, state, targets);
  auto q = positions(group, state);
  if (handoff.position.size() == group.dof)
    for (int j = 0; j < group.dof; ++j) q[j] = handoff.position[j];
  auto upload = [&](DeviceArray<float> &device, const Eigen::VectorXd &value) {
    std::vector<float> values(group.dof, 0.F);
    if (value.size() == group.dof)
      for (int j = 0; j < group.dof; ++j) values[j] = value[j];
    device.upload(values);
  };
  group.start.upload(q);
  upload(group.mpc_velocity, handoff.velocity);
  upload(group.mpc_acceleration, handoff.acceleration);
  upload(group.mpc_jerk, handoff.jerk);
  if ((joint_goal || !cold_seed_goal.empty()) && cold_seed_goal.size() != q.size())
    throw std::invalid_argument("invalid MPC seed or joint goal count");
  reset = reset || !group.mpc_initialized || group.mpc_joint_goal != joint_goal;
  group.mpc->set_iterations(reset ? settings.mpc_cold_iterations : settings.mpc_iterations);
  if (joint_goal)
    group.goal.upload(std::vector<float>(cold_seed_goal.begin(), cold_seed_goal.end()));
  if (reset) {
    std::vector<float> seed;
    for (int i = 0; i < group.knots; ++i) {
      const double fraction = static_cast<double>(i + 1) / group.knots;
      for (size_t j = 0; j < q.size(); ++j) {
        const double endpoint = cold_seed_goal.empty() ? q[j] : cold_seed_goal[j];
        seed.push_back(q[j] + fraction * (endpoint - q[j]));
      }
    }
    group.knots_buffer.upload(seed);
    if (joint_goal) {
      group.mpc->reset(group.goal.get(), group.knots_buffer.get());
    } else {
      curobo::robot::MpcCartesianConfig configuration;
      configuration.constrain_path = false;
      group.mpc->reset_cartesian(group.ik_goals.get(), group.ik_quaternions.get(),
                                 group.knots_buffer.get(), configuration);
    }
    group.mpc_initialized = true;
    group.mpc_joint_goal = joint_goal;
    executed_knots = 0;
  } else if (joint_goal) {
    group.mpc->update_joint_goal(group.goal.get());
  } else {
    group.mpc->update_cartesian_target(group.ik_goals.get(), group.ik_quaternions.get());
  }
  std::vector<float> reference = q, dof_weights(group.dof, 0.F);
  float posture_weight = 0.F;
  if (!joint_goal && settings.mpc_posture_weight > 0.) {
    group.start.upload(q);
    group.kinematics->forward(group.start.get(), group.fk_position.get(), group.fk_quaternion.get(),
                              group.fk_jacobian.get(), 1, 1);
    const auto data = group.fk_jacobian.host(group.tools * 6 * group.dof);
    std::vector<int> requested;
    for (int i = 0; i < group.tools; ++i)
      if (targets.count(group.tool_names[i])) requested.push_back(i);
    Eigen::MatrixXd jac(requested.size() * 6, group.dof);
    for (size_t i = 0; i < requested.size(); ++i)
      for (int row = 0; row < 6; ++row)
        for (int column = 0; column < group.dof; ++column)
          jac(i * 6 + row, column) = data[(requested[i] * 6 + row) * group.dof + column];
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(jac, Eigen::ComputeFullV);
    const auto singular = svd.singularValues();
    int rank = 0;
    if (singular.size() && singular[0] > 1e-8) {
      for (int i = 0; i < singular.size(); ++i)
        if (singular[i] > singular[0] * 1e-3) ++rank;
      if (rank < group.dof) {
        const Eigen::MatrixXd basis = svd.matrixV().rightCols(group.dof - rank);
        Eigen::VectorXd desired(group.dof);
        const JointMap *posture = nullptr;
        if (settings.mpc_posture != "zero") {
          const auto table = robot_.profile.postures.find(settings.mpc_posture);
          if (table == robot_.profile.postures.end())
            throw std::invalid_argument("unknown MPC posture " + settings.mpc_posture);
          posture = &table->second;
        }
        for (int j = 0; j < group.dof; ++j) {
          double reference_position = 0.;
          if (posture) {
            const auto joint = posture->find(group.names[j]);
            if (joint == posture->end())
              throw std::invalid_argument("MPC posture is missing " + group.names[j]);
            reference_position = joint->second;
          }
          desired[j] = reference_position - q[j];
        }
        Eigen::VectorXd delta = basis * (basis.transpose() * desired);
        delta *= 1. / std::max(1., delta.norm());
        const Eigen::VectorXd weights = basis.cwiseAbs().rowwise().sum();
        for (int j = 0; j < group.dof; ++j) {
          const double lower = group.lower[j];
          const double upper = group.upper[j];
          reference[j] = std::clamp(q[j] + delta[j], lower, upper);
          dof_weights[j] = weights[j] / std::max(1e-6, weights.maxCoeff());
        }
        posture_weight = settings.mpc_posture_weight;
      }
    }
  }
  if (!joint_goal && settings.mpc_run_ik) {
    auto ik = inverse_kinematics(name, state, targets, true);
    if (ik.success) {
      for (int j = 0; j < group.dof; ++j) reference[j] = ik.trajectory.points.back().positions[j];
      std::fill(dof_weights.begin(), dof_weights.end(), 1.F);
      posture_weight = std::max(1.F, posture_weight);
    }
  }
  if (!joint_goal) group.mpc->set_joint_tracking(reference, dof_weights, posture_weight);
  const auto output =
      group.mpc->step(group.start.get(), group.mpc_velocity.get(), group.mpc_acceleration.get(),
                      group.mpc_jerk.get(), executed_knots);
  MpcProposal result;
  result.elapsed_ms = elapsed_ms(started);
  result.success = output.feasible;
  result.sample_dt = output.trajectory_dt / velocity_scale;
  result.knot_dt = result.sample_dt * group.interpolation_steps;
  result.message = result.success ? "MPC feasible" : "MPC infeasible (candidate=" +
      std::to_string(output.selected_candidate) + ", cost=" + std::to_string(output.cost) +
      ", collision_cost=" + std::to_string(output.collision_cost) + ")";
  if (!result.success) return result;
  result.trajectory.joint_names = group.names;
  const auto count = static_cast<size_t>(group.horizon) * group.dof;
  std::vector<float> p(count), v(count), a(count);
  check_cuda(cudaMemcpy(p.data(), output.position, count * sizeof(float), cudaMemcpyDeviceToHost));
  check_cuda(cudaMemcpy(v.data(), output.velocity, count * sizeof(float), cudaMemcpyDeviceToHost));
  check_cuda(
      cudaMemcpy(a.data(), output.acceleration, count * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < group.horizon; ++i) {
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.time_from_start = duration(i * result.sample_dt);
    for (int j = 0; j < group.dof; ++j) {
      const auto index = i * group.dof + j;
      point.positions.push_back(p[index]);
      point.velocities.push_back(v[index] * velocity_scale);
      point.accelerations.push_back(a[index] * velocity_scale * velocity_scale);
    }
    result.trajectory.points.push_back(std::move(point));
  }
  return result;
}
bool NativePlanner::validate_mpc_continuation(const std::string &name,
                                              const Trajectory &trajectory,
                                              double from, double until,
                                              double velocity_scale) {
  require_valid_world();
  auto &group = get(name, true);
  const auto ordered = select_joints(trajectory, group.names);
  const double length = until - from;
  constexpr double kMaxSampleDt = .004;
  const int intervals = group.horizon - 1;
  const int chunks = std::max(1, static_cast<int>(std::ceil(length / (intervals * kMaxSampleDt))));
  const double dt = length / (chunks * intervals);
  const double scale2 = velocity_scale * velocity_scale;
  const double scale3 = scale2 * velocity_scale;
  TrajectorySampler sampler(ordered);
  std::vector<float> q(group.horizon * group.dof), v(q.size()), a(q.size()), j(q.size());
  for (int chunk = 0; chunk < chunks; ++chunk) {
    for (int sample = 0; sample < group.horizon; ++sample) {
      const auto &point = sampler.sample(from + dt * (chunk * intervals + sample));
      for (int joint = 0; joint < group.dof; ++joint) {
        const int index = sample * group.dof + joint;
        q[index] = point.position[joint];
        v[index] = point.velocity[joint] / velocity_scale;
        a[index] = point.acceleration[joint] / scale2;
        j[index] = point.jerk[joint] / scale3;
      }
    }
    group.validate_position.upload(q);
    group.validate_velocity.upload(v);
    group.validate_acceleration.upload(a);
    group.validate_jerk.upload(j);
    if (!group.mpc->validate_trajectory(group.validate_position.get(), group.validate_velocity.get(),
                                        group.validate_acceleration.get(), group.validate_jerk.get(),
                                        dt * velocity_scale))
      return false;
  }
  return true;
}
// 本拍结果可用，下一拍从这里继续。
void NativePlanner::accept_mpc(const std::string &group) { get(group, true).mpc->accept(); }
// 新的跟随开始前丢掉旧的热启动。
void NativePlanner::reset_mpc() {
  for (auto &entry : mpc_groups_) entry.second->mpc_initialized = false;
}
}  // namespace curobo_ros_cpp
