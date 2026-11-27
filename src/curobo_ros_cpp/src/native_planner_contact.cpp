// SPDX-License-Identifier: Apache-2.0
// 求解和验收接近、抬升、撤退等接触段的笛卡尔直线插补。
#include "curobo_ros_cpp/detail/native_planner_group.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <rclcpp/rclcpp.hpp>

namespace curobo_ros_cpp {
namespace {
struct ContactPathFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};
}  // namespace
// 每种批次数跑一次接触逆解，把对应的图先捕获好。
void NativePlanner::warmup_interpolation(const std::string &name, const std::vector<int> &samples) {
  auto &group = get(name);
  const auto state = robot_.defaults;
  synchronize(name, state);
  std::map<std::string, Transform> targets;
  const auto transforms = robot_.forward(state);
  for (const auto &tool : group.tool_names) targets[tool] = transforms.at(tool);
  auto endpoint = inverse_kinematics(name, state, targets, false);
  if (!endpoint.success) throw std::runtime_error("contact IK prewarm failed");
  const auto first = positions(group, state);
  for (int count : samples) {
    if (count < 1 || count > Group::kIkBatchCapacity)
      throw std::invalid_argument("invalid interpolation prewarm count");
    std::vector<float> p, q, seeds;
    for (int i = 0; i < count; ++i) {
      for (const auto &tool : group.tool_names) {
        const auto &at = transforms.at(tool);
        Eigen::Quaterniond rotation(at.linear());
        for (int axis = 0; axis < 3; ++axis) p.push_back(at.translation()[axis]);
        q.insert(q.end(), {static_cast<float>(rotation.w()), static_cast<float>(rotation.x()),
                           static_cast<float>(rotation.y()), static_cast<float>(rotation.z())});
      }
      seeds.insert(seeds.end(), first.begin(), first.end());
    }
    group.ik_goals.upload(p);
    group.ik_quaternions.upload(q);
    group.ik_seeds_buffer.upload(seeds);
    curobo::robot::IkSolveRequest request;
    request.goal_positions = group.ik_goals.get();
    request.goal_quaternions = group.ik_quaternions.get();
    request.seed_positions = group.ik_seeds_buffer.get();
    request.solution = group.ik_solution.get();
    request.success = group.ik_success.get();
    request.position_error = group.ik_position_error.get();
    request.orientation_error = group.ik_orientation_error.get();
    request.collision_cost = group.ik_collision.get();
    request.batch_size = count;
    group.batch_ik->solve(request);
    const auto flags = group.ik_success.host(count);
    if (std::any_of(flags.begin(), flags.end(), [](auto flag) { return flag == 0; }))
      throw std::runtime_error("batch contact IK prewarm failed");
  }
}
// 沿工具直线采样逆解。这一段不查环境碰撞。
NativePlanner::ContactPath NativePlanner::linear_contact(const std::string &tool,
    const Transform &start, const Transform &end) const {
  const Eigen::Vector3d delta = end.translation() - start.translation();
  const double length = delta.norm();
  const Eigen::Quaterniond qa(start.linear()), qb(end.linear());
  const int required = std::max({3, static_cast<int>(std::ceil(length / .005)) + 1,
                                 static_cast<int>(std::ceil(qa.angularDistance(qb) / .03)) + 1});
  const auto pose_at = [start, delta, qa, qb](double alpha) {
    Transform pose = Transform::Identity();
    pose.translation() = start.translation() + alpha * delta;
    pose.linear() = qa.slerp(alpha, qb).toRotationMatrix();
    return pose;
  };
  const auto progress = [start, delta, length](const Transform &pose) {
    return (pose.translation() - start.translation()).dot(delta) / length;
  };
  const double tolerance =
      settings.path_tolerance > 0. ? settings.path_tolerance : settings.position_tolerance;
  return {tool, length, tolerance, required, pose_at, progress};
}

PlanResult NativePlanner::interpolate(const std::string &name, const JointMap &state,
                                      const Transform &start, const Transform &end) {
  return interpolate_path(name, state, {linear_contact(robot_.profile.tools.at(name), start, end)});
}

PlanResult NativePlanner::interpolate_both(const JointMap &state,
                                           const std::map<std::string, Transform> &targets) {
  const auto transforms = robot_.forward(state);
  std::vector<ContactPath> paths;
  for (const auto &side : {"left_arm", "right_arm"}) {
    const auto &tool = robot_.profile.tools.at(side);
    paths.push_back(linear_contact(tool, transforms.at(tool), targets.at(tool)));
  }
  return interpolate_path("both_arms", state, paths);
}

NativePlanner::PickApproach NativePlanner::plan_pick_approach(
    const std::string &name, const JointMap &state, const Transform &prepick, const Transform &grasp,
    const std::function<void()> &check) {
  const auto tool = robot_.profile.tools.at(name);
  const auto &joints = robot_.profile.groups.at(name);
  const std::map<std::string, Transform> targets{{tool, prepick}};
  PickApproach result;
  const auto prepick_started = Clock::now();
  result.prepick = plan_pose(name, state, targets, 3, /*rrt_on_first_attempt=*/true);
  result.prepick_ms = elapsed_ms(prepick_started);
  if (!result.prepick.success) return result;

  auto descend_from = [&](const std::vector<double> &candidate) {
    check();
    auto at = state;
    for (size_t i = 0; i < joints.size(); ++i) at[joints[i]] = candidate[i];
    const auto started = Clock::now();
    PlanResult descent;
    try {
      descent = interpolate(name, at, robot_.tool(at, name), grasp);
    } catch (const ContactPathFailure &error) {
      descent.message = error.what();
    }
    result.descent_ms = elapsed_ms(started);
    return descent;
  };

  const auto &original = result.prepick.trajectory.points.back().positions;
  result.descent = descend_from(original);
  if (result.descent.success) return result;
  const double original_descent_ms = result.descent_ms;
  const auto logger = rclcpp::get_logger("curobo_motion_planning_node");
  RCLCPP_WARN(logger, "【抓取 · 构型选择】group=%s ｜ 原预抓取构型无法连续下降 ｜ %s",
              name.c_str(), result.descent.message.c_str());

  const auto search_started = Clock::now();
  std::vector<std::vector<double>> candidates;
  inverse_kinematics(name, state, targets, /*collision=*/true, /*prefer_nearby=*/true, &candidates);
  std::vector<std::vector<double>> tried{original};
  size_t checked = 0;
  size_t connected = 0;
  std::string last_approach_failure;
  for (const auto &candidate : candidates) {
    const bool duplicate = std::any_of(tried.begin(), tried.end(), [&](const auto &previous) {
      for (size_t i = 0; i < candidate.size(); ++i)
        if (std::abs(candidate[i] - previous[i]) > 1e-3) return false;
      return true;
    });
    if (duplicate) continue;
    tried.push_back(candidate);
    ++checked;
    auto descent = descend_from(candidate);
    if (!descent.success) continue;
    ++connected;
    check();
    auto approach = plan_joint(name, state, candidate);
    if (!approach.success) {
      last_approach_failure = approach.message;
      continue;
    }
    // Use the cached descent only when joint planning reaches its exact start.
    const auto &endpoint = approach.trajectory.points.back().positions;
    if (endpoint != candidate) {
      descent = descend_from(endpoint);
      if (!descent.success) continue;
    }
    result.prepick_ms += elapsed_ms(search_started) - result.descent_ms;
    result.prepick = std::move(approach);
    result.descent = std::move(descent);
    RCLCPP_INFO(logger, "【抓取 · 构型选择】group=%s ｜ 已选择可连续下降的预抓取构型 ｜ 已检查 %zu 个",
                name.c_str(), checked);
    return result;
  }
  result.prepick_ms += elapsed_ms(search_started);
  result.descent_ms = original_descent_ms;
  result.descent.message += " | 备选预抓取构型检查=" + std::to_string(checked) +
                            " | 通过连续下降=" + std::to_string(connected);
  if (!last_approach_failure.empty()) result.descent.message += " | 接近规划：" + last_approach_failure;
  return result;
}

PlanResult NativePlanner::interpolate_arc(const std::string &name, const JointMap &state,
                                           const Transform &start, const Eigen::Vector2d &center,
                                           double sweep) {
  const Eigen::Vector2d radial = start.translation().head<2>() - center;
  const double radius = radial.norm(), length = radius * std::abs(sweep);
  const auto pose_at = [&](double alpha) {
    Transform pose = start;
    pose.translation().head<2>() = center + Eigen::Rotation2Dd(alpha * sweep) * radial;
    return pose;
  };
  const Eigen::Vector2d middle = Eigen::Rotation2Dd(sweep / 2.) * radial;
  const auto progress = [&](const Transform &pose) {
    const Eigen::Vector2d at = pose.translation().head<2>() - center;
    const double angle = std::atan2(middle.x() * at.y() - middle.y() * at.x(), middle.dot(at));
    return (sweep / 2. + angle) * std::copysign(radius, sweep);
  };
  const int required = std::max(3, static_cast<int>(std::ceil(length / .005)) + 1);
  return interpolate_path(name, state, {{robot_.profile.tools.at(name), length, .003, required,
                                         pose_at, progress}});
}

PlanResult NativePlanner::interpolate_path(
    const std::string &name, const JointMap &state, const std::vector<ContactPath> &paths) {
  const auto before = Clock::now();
  synchronize(name, state);
  auto &group = get(name);
  int required = 0;
  for (const auto &path : paths) {
    required = std::max(required, path.samples);
    if (path.length < 1e-6) throw std::invalid_argument("contact distance outside interpolation capacity");
  }
  if (required > Group::kIkBatchCapacity)
    throw std::invalid_argument("contact distance outside interpolation capacity");
  auto targets_at = [&](double progress) {
    std::map<std::string, Transform> targets;
    for (const auto &path : paths) targets.emplace(path.tool, path.pose_at(progress));
    return targets;
  };
  int count = 33;
  while (count < required) count = 2 * count - 1;
  auto endpoint = inverse_kinematics(name, state, targets_at(1.), false);
  if (!endpoint.success) return endpoint;
  const auto first = positions(group, state);
  const auto last = endpoint.trajectory.points.back().positions;
  const auto fixed = robot_.forward(state);
  std::vector<float> p, q, seeds;
  for (int i = 0; i < count; ++i) {
    const double alpha = static_cast<double>(i) / (count - 1);
    const auto desired = targets_at(alpha);
    for (const auto &frame : group.tool_names) {
      const auto found = desired.find(frame);
      const auto &at = found == desired.end() ? fixed.at(frame) : found->second;
      Eigen::Quaterniond rotation(at.linear());
      for (int axis = 0; axis < 3; ++axis) p.push_back(at.translation()[axis]);
      q.insert(q.end(), {static_cast<float>(rotation.w()), static_cast<float>(rotation.x()),
                         static_cast<float>(rotation.y()), static_cast<float>(rotation.z())});
    }
    for (int j = 0; j < group.dof; ++j) seeds.push_back(first[j] + alpha * (last[j] - first[j]));
  }
  group.ik_goals.upload(p);
  group.ik_quaternions.upload(q);
  group.ik_seeds_buffer.upload(seeds);
  curobo::robot::IkSolveRequest request;
  request.goal_positions = group.ik_goals.get();
  request.goal_quaternions = group.ik_quaternions.get();
  request.seed_positions = group.ik_seeds_buffer.get();
  request.solution = group.ik_solution.get();
  request.success = group.ik_success.get();
  request.position_error = group.ik_position_error.get();
  request.orientation_error = group.ik_orientation_error.get();
  request.collision_cost = group.ik_collision.get();
  request.batch_size = count;
  group.batch_ik->solve(request);
  auto flags = group.ik_success.host(count);
  auto rows = group.ik_solution.host(count * group.dof);
  for (int j = 0; j < group.dof; ++j) rows[j] = first[j];
  bool batch_valid = std::all_of(flags.begin(), flags.end(), [](auto value) { return value != 0; });
  for (int i = 1; i < count; ++i)
    for (int j = 0; j < group.dof; ++j)
      batch_valid =
          batch_valid && std::abs(rows[i * group.dof + j] - rows[(i - 1) * group.dof + j]) <= .15;
  if (!batch_valid) {
    JointMap previous = state;
    std::vector<float> continuation = first;
    std::deque<double> pending;
    for (int i = 1; i < required; ++i) pending.push_back(static_cast<double>(i) / (required - 1));
    double previous_alpha = 0.;
    int calls = 0;
    while (!pending.empty()) {
      const double alpha = pending.front();
      pending.pop_front();
      auto step = inverse_kinematics(name, previous, targets_at(alpha), false);
      ++calls;
      bool valid = step.success;
      if (valid)
        for (int j = 0; j < group.dof; ++j)
          valid = valid && std::abs(step.trajectory.points.back().positions[j] -
                                    previous.at(group.names[j])) <= .15;
      if (!valid) {
        if (alpha - previous_alpha <= 1e-4 || calls >= 2000)
          throw ContactPathFailure("contact IK continuation failed or discontinuous");
        pending.push_front(alpha);
        pending.push_front(.5 * (alpha + previous_alpha));
        continue;
      }
      for (int j = 0; j < group.dof; ++j) {
        const auto value = step.trajectory.points.back().positions[j];
        continuation.push_back(value);
        previous[group.names[j]] = value;
      }
      previous_alpha = alpha;
    }
    rows = std::move(continuation);
    count = rows.size() / group.dof;
  }

  PlanResult result;
  result.trajectory.joint_names = group.names;
  const int input_count = count;
  const double dt = settings.dt;
  const double speed = settings.velocity_scale;
  std::vector<Eigen::VectorXd> original(input_count, Eigen::VectorXd(group.dof));
  for (int i = 0; i < input_count; ++i)
    for (int j = 0; j < group.dof; ++j) original[i][j] = rows[i * group.dof + j];

  // Smooth high-frequency independent batch-IK jitter while pinning endpoints and
  // verifying every interior point stays well within half the Cartesian path tolerance.
  if (input_count >= 5) {
    auto smoothed = original;
    for (int pass = 0; pass < 2; ++pass) {
      auto next = smoothed;
      for (int i = 1; i + 1 < input_count; ++i)
        next[i] = 0.25 * smoothed[i - 1] + 0.5 * smoothed[i] + 0.25 * smoothed[i + 1];
      smoothed = std::move(next);
    }
    bool smooth_ok = true;
    JointMap check_state = state;
    for (int i = 1; i + 1 < input_count && smooth_ok; ++i) {
      for (int j = 0; j < group.dof; ++j) {
        const double qj = smoothed[i][j];
        if (qj < group.lower[j] || qj > group.upper[j]) {
          smooth_ok = false;
          break;
        }
        check_state[group.names[j]] = qj;
      }
      if (!smooth_ok) break;
      const auto actual = robot_.forward(check_state);
      for (const auto &path : paths) {
        const auto &pose = actual.at(path.tool);
        const double along = path.progress(pose), alpha = along / path.length;
        const auto projected = path.pose_at(alpha);
        const Eigen::Quaterniond expected(path.pose_at(std::clamp(alpha, 0., 1.)).linear());
        if ((pose.translation() - projected.translation()).norm() > 0.5 * path.tolerance ||
            along < -.0025 || along > path.length + .0025 ||
            Eigen::Quaterniond(pose.linear()).angularDistance(expected) >
                0.5 * settings.orientation_tolerance + .0025) {
          smooth_ok = false;
          break;
        }
      }
    }
    if (smooth_ok) original = std::move(smoothed);
  }

  auto gradient = [](const std::vector<Eigen::VectorXd> &values, double step) {
    const size_t n = values.size();
    std::vector<Eigen::VectorXd> out(n);
    out.front() = (-3 * values[0] + 4 * values[1] - values[2]) / (2 * step);
    out.back() = (3 * values[n - 1] - 4 * values[n - 2] + values[n - 3]) / (2 * step);
    for (size_t i = 1; i + 1 < n; ++i) out[i] = (values[i + 1] - values[i - 1]) / (2 * step);
    return out;
  };
  auto input_tangent = gradient(original, 1.);

  // Initialize unscaled duration from actual joint travel & dynamic limits rather than
  // tying it to the spatial batch-IK sample bucket (33/65/...).
  double base_duration = 0.04;
  for (int j = 0; j < group.dof; ++j) {
    double travel = 0.;
    for (int i = 1; i < input_count; ++i)
      travel += std::abs(original[i][j] - original[i - 1][j]);
    if (travel <= 1e-9) continue;
    base_duration = std::max({
        base_duration,
        1.875 * travel / group.limits.velocity[j],
        std::sqrt(5.7735026919 * travel / group.limits.acceleration[j]),
        std::cbrt(60. * travel / group.limits.jerk[j]),
    });
  }

  std::vector<Eigen::VectorXd> positions, velocities, accelerations;
  double sample_dt = dt;
  bool retimed = false;
  std::string bottleneck = "none";
  int bottleneck_joint = -1;
  for (int iteration = 0; iteration < 10; ++iteration) {
    count = std::max(17, static_cast<int>(std::ceil(base_duration / dt)) + 1);
    if (count > 10000) throw ContactPathFailure("contact retiming capacity exceeded");
    sample_dt = base_duration / (count - 1);
    positions.clear();
    positions.reserve(count);
    for (int i = 0; i < count; ++i) {
      const double u = static_cast<double>(i) / (count - 1),
                   warp = 10 * u * u * u - 15 * std::pow(u, 4) + 6 * std::pow(u, 5);
      const double index = warp * (input_count - 1);
      const int left = std::clamp(static_cast<int>(std::floor(index)), 0, input_count - 2);
      const double v = index - left;
      positions.push_back((2 * v * v * v - 3 * v * v + 1) * original[left] +
                          (v * v * v - 2 * v * v + v) * input_tangent[left] +
                          (-2 * v * v * v + 3 * v * v) * original[left + 1] +
                          (v * v * v - v * v) * input_tangent[left + 1]);
    }
    velocities = gradient(positions, sample_dt);
    velocities.front().setZero();
    velocities.back().setZero();
    accelerations = gradient(velocities, sample_dt);
    accelerations.front().setZero();
    accelerations.back().setZero();
    const auto jerk = gradient(accelerations, sample_dt);
    double peak = 0.;
    bottleneck = "none";
    bottleneck_joint = -1;
    for (int i = 0; i < count; ++i)
      for (int j = 0; j < group.dof; ++j) {
        const double v_ratio = std::abs(velocities[i][j]) / group.limits.velocity[j];
        const double a_ratio =
            std::sqrt(std::abs(accelerations[i][j]) / group.limits.acceleration[j]);
        const double j_ratio = std::cbrt(std::abs(jerk[i][j]) / group.limits.jerk[j]);
        if (v_ratio > peak) {
          peak = v_ratio;
          bottleneck = "vel";
          bottleneck_joint = j;
        }
        if (a_ratio > peak) {
          peak = a_ratio;
          bottleneck = "acc";
          bottleneck_joint = j;
        }
        if (j_ratio > peak) {
          peak = j_ratio;
          bottleneck = "jerk";
          bottleneck_joint = j;
        }
      }
    if (peak <= 1.001) {
      if (iteration == 0 && peak > 1e-4 && peak < 0.97 && base_duration * peak > 0.04) {
        base_duration = std::max(0.04, base_duration * peak * 1.02);
        continue;
      }
      retimed = true;
      break;
    }
    base_duration = base_duration * peak * 1.015;
  }
  if (!retimed) throw ContactPathFailure("contact retiming failed dynamic limits");
  for (int i = 0; i < count; ++i) {
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.time_from_start = duration(i * sample_dt / speed);
    for (int j = 0; j < group.dof; ++j) {
      point.positions.push_back(static_cast<float>(positions[i][j]));
      point.velocities.push_back(static_cast<float>(velocities[i][j]) * speed);
      point.accelerations.push_back(static_cast<float>(accelerations[i][j]) * speed * speed);
    }
    result.trajectory.points.push_back(std::move(point));
  }
  std::vector<double> previous_along(paths.size(), -1.);
  const double total = seconds(result.trajectory.points.back().time_from_start);
  const int samples = static_cast<int>(std::ceil(total / .02));
  TrajectorySampler sampler(result.trajectory);
  JointMap full = state;
  const auto tool = robot_.profile.tools.find(name);
  const bool single_tool = paths.size() == 1 && tool != robot_.profile.tools.end() &&
                           paths.front().tool == tool->second;
  for (int i = 0; i <= samples; ++i) {
    const auto &at = sampler.sample(total * i / samples);
    for (int j = 0; j < group.dof; ++j) {
      full[group.names[j]] = at.position[j];
      const float lower = group.lower[j];
      const float upper = group.upper[j];
      if (at.position[j] < lower || at.position[j] > upper)
        throw ContactPathFailure("contact joint limit violation");
    }
    const auto actual = single_tool ? std::map<std::string, Transform>{} : robot_.forward(full);
    for (size_t j = 0; j < paths.size(); ++j) {
      const auto &path = paths[j];
      const auto pose = single_tool ? robot_.tool(full, name) : actual.at(path.tool);
      const double along = path.progress(pose), alpha = along / path.length;
      const auto projected = path.pose_at(alpha);
      const Eigen::Quaterniond expected(path.pose_at(std::clamp(alpha, 0., 1.)).linear());
      if ((pose.translation() - projected.translation()).norm() > path.tolerance || along < -.005 ||
          along > path.length + .005 || along < previous_along[j] - .005 ||
          Eigen::Quaterniond(pose.linear()).angularDistance(expected) > settings.orientation_tolerance + .005)
        throw ContactPathFailure("contact interpolated spline deviates from Cartesian path: " + path.tool);
      previous_along[j] = along;
    }
  }
  result.success = true;
  result.message = "contact IK interpolation";
  if (bottleneck_joint >= 0 && bottleneck_joint < static_cast<int>(group.names.size()))
    result.message += " | limit=" + bottleneck + "(" + group.names[bottleneck_joint] + ")";
  result.timing.ik_ms = elapsed_ms(before);
  return result;
}
}  // namespace curobo_ros_cpp
