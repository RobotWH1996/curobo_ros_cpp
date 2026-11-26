// SPDX-License-Identifier: Apache-2.0
// 采样、拼接、重定时和校验关节轨迹，生成夹爪动作。
#include "curobo_ros_cpp/trajectory.hpp"
#include <algorithm>
#include <numeric>
#include <set>
#include <utility>

namespace curobo_ros_cpp {
namespace {
// 把 std::vector 看成 Eigen 向量，不拷贝数据。
Eigen::VectorXd vector(const std::vector<double> &values) {
  return Eigen::Map<const Eigen::VectorXd>(values.data(), values.size());
}
std::vector<double> list(const Eigen::VectorXd &values) {
  return {values.data(), values.data() + values.size()};
}
std::vector<size_t> columns(const Trajectory &tr, const std::vector<std::string> &names) {
  std::vector<size_t> result;
  for (const auto &name : names) {
    auto found = std::find(tr.joint_names.begin(), tr.joint_names.end(), name);
    if (found == tr.joint_names.end()) throw std::runtime_error("missing trajectory joint " + name);
    result.push_back(found - tr.joint_names.begin());
  }
  return result;
}
trajectory_msgs::msg::JointTrajectoryPoint point(const TrajectorySample &sample, double time) {
  trajectory_msgs::msg::JointTrajectoryPoint result;
  result.positions = list(sample.position);
  result.velocities = list(sample.velocity);
  result.accelerations = list(sample.acceleration);
  result.time_from_start = duration(time);
  return result;
}
void check_time(const Trajectory &trajectory) {
  if (trajectory.points.size() < 2) throw std::runtime_error("trajectory requires two points");
  double previous = -1.;
  for (const auto &entry : trajectory.points) {
    const double time = seconds(entry.time_from_start);
    if (time <= previous || entry.positions.size() != trajectory.joint_names.size())
      throw std::runtime_error("invalid trajectory layout or clock at " + std::to_string(time) +
                               " after " + std::to_string(previous));
    previous = time;
  }
}
void segment_coefficients(const trajectory_msgs::msg::JointTrajectoryPoint &first,
                          const trajectory_msgs::msg::JointTrajectoryPoint &last, double h,
                          Eigen::MatrixXd &coefficients) {
  const auto n = first.positions.size();
  coefficients.setZero(n, 6);
  coefficients.col(0) = vector(first.positions);
  const bool have_v = first.velocities.size() == n && last.velocities.size() == n;
  const bool have_a = have_v && first.accelerations.size() == n && last.accelerations.size() == n;
  if (have_v) {
    const auto v0 = vector(first.velocities), v1 = vector(last.velocities);
    coefficients.col(1) = v0 * h;
    if (have_a) {
      const auto a0 = vector(first.accelerations), a1 = vector(last.accelerations);
      coefficients.col(2) = .5 * a0 * h * h;
      const Eigen::VectorXd c0 = vector(last.positions) - coefficients.leftCols(3).rowwise().sum();
      const Eigen::VectorXd c1 = v1 * h - coefficients.col(1) - 2 * coefficients.col(2);
      const Eigen::VectorXd c2 = a1 * h * h - 2 * coefficients.col(2);
      coefficients.col(3) = 10 * c0 - 4 * c1 + .5 * c2;
      coefficients.col(4) = -15 * c0 + 7 * c1 - c2;
      coefficients.col(5) = 6 * c0 - 3 * c1 + .5 * c2;
    } else {
      const Eigen::VectorXd delta = vector(last.positions) - vector(first.positions);
      coefficients.col(2) = 3 * delta - (2 * v0 + v1) * h;
      coefficients.col(3) = -2 * delta + (v0 + v1) * h;
    }
  } else
    coefficients.col(1) = vector(last.positions) - vector(first.positions);
}
}  // namespace

TrajectorySampler::TrajectorySampler(const Trajectory &trajectory) : tr(trajectory) {
  if (tr.points.size() < 2) throw std::runtime_error("trajectory requires two points");
}

void TrajectorySampler::load_segment(double time) {
  auto right = std::upper_bound(
      tr.points.begin(), tr.points.end(), time,
      [](double t, const auto &entry) { return t < seconds(entry.time_from_start); });
  index = static_cast<size_t>(std::distance(tr.points.begin(), right));
  index = std::clamp<size_t>(index, 1, tr.points.size() - 1) - 1;
  const auto &first = tr.points[index];
  const auto &last = tr.points[index + 1];
  start = seconds(first.time_from_start);
  end = seconds(last.time_from_start);
  const double h = end - start;
  if (!(h > 0.) || first.positions.size() != tr.joint_names.size() ||
      last.positions.size() != tr.joint_names.size())
    throw std::runtime_error("invalid trajectory layout or clock at " + std::to_string(time));
  segment_coefficients(first, last, h, coefficients);
}

const TrajectorySample &TrajectorySampler::sample(double time) {
  if (coefficients.cols() == 0 || (index > 0 && time < start) ||
      (index + 2 < tr.points.size() && time >= end))
    load_segment(time);
  const double h = end - start;
  const double u = (time - start) / h;
  const double u2 = u * u, u3 = u2 * u, u4 = u2 * u2, u5 = u4 * u;
  const double inv_h = 1. / h, inv_h2 = inv_h * inv_h, inv_h3 = inv_h2 * inv_h;
  const Eigen::Index rows = coefficients.rows();
  if (values.position.size() != rows) {
    values.position.resize(rows);
    values.velocity.resize(rows);
    values.acceleration.resize(rows);
    values.jerk.resize(rows);
  }
  const double *c0 = coefficients.col(0).data();
  const double *c1 = coefficients.col(1).data();
  const double *c2 = coefficients.col(2).data();
  const double *c3 = coefficients.col(3).data();
  const double *c4 = coefficients.col(4).data();
  const double *c5 = coefficients.col(5).data();
  double *pos = values.position.data();
  double *vel = values.velocity.data();
  double *acc = values.acceleration.data();
  double *jrk = values.jerk.data();
  for (Eigen::Index r = 0; r < rows; ++r) {
    const double a0 = c0[r], a1 = c1[r], a2 = c2[r], a3 = c3[r], a4 = c4[r], a5 = c5[r];
    const double p = a0 + a1 * u + a2 * u2 + a3 * u3 + a4 * u4 + a5 * u5;
    const double v = (a1 + 2. * a2 * u + 3. * a3 * u2 + 4. * a4 * u3 + 5. * a5 * u4) * inv_h;
    const double a = (2. * a2 + 6. * a3 * u + 12. * a4 * u2 + 20. * a5 * u3) * inv_h2;
    const double j = (6. * a3 + 24. * a4 * u + 60. * a5 * u2) * inv_h3;
    if (!std::isfinite(p) || !std::isfinite(v) || !std::isfinite(a) || !std::isfinite(j))
      throw std::runtime_error("nonfinite trajectory polynomial");
    pos[r] = p;
    vel[r] = v;
    acc[r] = a;
    jrk[r] = j;
  }
  return values;
}
TrajectorySample sample_trajectory(const Trajectory &tr, double time) {
  return TrajectorySampler(tr).sample(time);
}
// 各段关节名必须一致，时间按上一段终点顺延。
Trajectory concatenate(const std::vector<Trajectory> &segments) {
  Trajectory result;
  double offset = 0.;
  for (const auto &segment : segments) {
    if (segment.points.empty()) throw std::runtime_error("empty trajectory segment");
    if (result.joint_names.empty()) {
      result.joint_names = segment.joint_names;
      result.header = segment.header;
    }
    if (segment.joint_names != result.joint_names)
      throw std::runtime_error("trajectory joint order mismatch");
    const auto first = seconds(segment.points.front().time_from_start);
    double previous = -1.;
    if (!result.points.empty() &&
        (vector(result.points.back().positions) - vector(segment.points.front().positions))
                .cwiseAbs()
                .maxCoeff() > 1e-3)
      throw std::runtime_error("trajectory position discontinuity");
    for (size_t i = 0; i < segment.points.size(); ++i) {
      const double relative = seconds(segment.points[i].time_from_start) - first;
      if (relative <= previous) throw std::runtime_error("nonmonotonic segment");
      previous = relative;
      if (!result.points.empty() && i == 0) continue;
      auto packed = segment.points[i];
      packed.time_from_start = duration(offset + relative);
      result.points.push_back(std::move(packed));
    }
    offset += previous;
  }
  return result;
}
// 只保留指定关节列，时间和导数一起裁剪。
Trajectory select_joints(const Trajectory &tr, const std::vector<std::string> &names) {
  const auto selected = columns(tr, names);
  Trajectory result;
  result.header = tr.header;
  result.joint_names = names;
  for (const auto &entry : tr.points) {
    trajectory_msgs::msg::JointTrajectoryPoint packed;
    packed.time_from_start = entry.time_from_start;
    for (auto i : selected) {
      packed.positions.push_back(entry.positions.at(i));
      if (!entry.velocities.empty()) packed.velocities.push_back(entry.velocities.at(i));
      if (!entry.accelerations.empty()) packed.accelerations.push_back(entry.accelerations.at(i));
    }
    result.points.push_back(std::move(packed));
  }
  return result;
}
// 求解器轨迹补成全身关节。没规划到的关节保持不动。
Trajectory expand_trajectory(const Trajectory &tr, const std::vector<std::string> &names,
                             const JointMap &held) {
  Trajectory result;
  result.header = tr.header;
  result.joint_names = names;
  std::vector<size_t> source;
  for (const auto &name : tr.joint_names) {
    auto found = std::find(names.begin(), names.end(), name);
    if (found == names.end()) throw std::runtime_error("active joint absent from output");
    source.push_back(found - names.begin());
  }
  for (const auto &entry : tr.points) {
    trajectory_msgs::msg::JointTrajectoryPoint packed;
    packed.time_from_start = entry.time_from_start;
    for (const auto &name : names) packed.positions.push_back(held.at(name));
    packed.velocities.resize(names.size(), 0.);
    packed.accelerations.resize(names.size(), 0.);
    for (size_t i = 0; i < source.size(); ++i) {
      packed.positions[source[i]] = entry.positions.at(i);
      if (!entry.velocities.empty()) packed.velocities[source[i]] = entry.velocities.at(i);
      if (!entry.accelerations.empty()) packed.accelerations[source[i]] = entry.accelerations.at(i);
    }
    result.points.push_back(std::move(packed));
  }
  return result;
}
// 夹爪单独插补，手臂关节速度保持为零。
Trajectory gripper_trajectory(const std::vector<std::string> &names, const JointMap &initial,
                              const std::string &joint, double target, double time, double dt) {
  if (time <= 0 || dt <= 0 || !std::isfinite(time))
    throw std::runtime_error("invalid gripper timing");
  Trajectory result;
  result.joint_names = names;
  const auto selected = columns(result, {joint}).front();
  const auto count = std::max<size_t>(2, static_cast<size_t>(std::ceil(time / dt)) + 1);
  const double delta = target - initial.at(joint);
  for (size_t i = 0; i < count; ++i) {
    const double t = time * i / (count - 1), u = t / time;
    trajectory_msgs::msg::JointTrajectoryPoint packed;
    for (const auto &name : names) packed.positions.push_back(initial.at(name));
    packed.velocities.resize(names.size(), 0.);
    packed.accelerations.resize(names.size(), 0.);
    packed.positions[selected] +=
        delta * (10 * std::pow(u, 3) - 15 * std::pow(u, 4) + 6 * std::pow(u, 5));
    packed.velocities[selected] =
        delta * (30 * u * u - 60 * u * u * u + 30 * std::pow(u, 4)) / time;
    packed.accelerations[selected] =
        delta * (60 * u - 180 * u * u + 120 * u * u * u) / (time * time);
    packed.time_from_start = duration(t);
    result.points.push_back(std::move(packed));
  }
  return result;
}
// 张开动作叠到手臂轨迹开头，不单独占用一段。
void overlay_gripper(Trajectory &tr, const std::string &joint, double start, double target,
                     double until) {
  if (until <= 0) return;
  const auto selected = columns(tr, {joint}).front();
  auto found = std::lower_bound(
      tr.points.begin(), tr.points.end(), until,
      [](const auto &entry, double t) { return seconds(entry.time_from_start) < t; });
  if (found != tr.points.end() && found != tr.points.begin() &&
      std::abs(seconds(found->time_from_start) - until) > 1e-8)
    tr.points.insert(found, point(sample_trajectory(tr, until), until));
  for (auto &entry : tr.points) {
    const double u = std::clamp(seconds(entry.time_from_start) / until, 0., 1.),
                 delta = target - start;
    entry.positions[selected] =
        start + delta * (10 * u * u * u - 15 * std::pow(u, 4) + 6 * std::pow(u, 5));
    if (entry.velocities.empty()) entry.velocities.resize(tr.joint_names.size(), 0.);
    if (entry.accelerations.empty()) entry.accelerations.resize(tr.joint_names.size(), 0.);
    entry.velocities[selected] =
        delta * (30 * u * u - 60 * u * u * u + 30 * std::pow(u, 4)) / until;
    entry.accelerations[selected] =
        delta * (60 * u - 180 * u * u + 120 * u * u * u) / (until * until);
  }
}
// 在指定时刻插入停留，后面的时间整体后移。
void insert_hold(Trajectory &tr, double at, double time) {
  if (time <= 0) return;
  auto found =
      std::min_element(tr.points.begin(), tr.points.end(), [&](const auto &a, const auto &b) {
        return std::abs(seconds(a.time_from_start) - at) <
               std::abs(seconds(b.time_from_start) - at);
      });
  if (found == tr.points.end() || std::abs(seconds(found->time_from_start) - at) > 1e-6)
    throw std::runtime_error("hold requires an existing knot");
  for (double value : found->velocities)
    if (std::abs(value) > 1e-5) throw std::runtime_error("hold requires zero velocity");
  for (double value : found->accelerations)
    if (std::abs(value) > 1e-5) throw std::runtime_error("hold requires zero acceleration");
  const auto index = std::distance(tr.points.begin(), found);
  auto held = *found;
  held.time_from_start = duration(at + time);
  for (size_t i = index + 1; i < tr.points.size(); ++i)
    tr.points[i].time_from_start = duration(seconds(tr.points[i].time_from_start) + time);
  tr.points.insert(tr.points.begin() + index + 1, std::move(held));
}
// 末尾再停一段，给夹爪稳定用。
void append_hold(Trajectory &tr, double time) {
  insert_hold(tr, seconds(tr.points.back().time_from_start), time);
}

namespace {
struct JointPoly {
  double c1, c2, c3, c4, c5;
};

void selected_segment_polys(const trajectory_msgs::msg::JointTrajectoryPoint &first,
                            const trajectory_msgs::msg::JointTrajectoryPoint &last, double h,
                            const std::vector<size_t> &selected, std::vector<JointPoly> &out) {
  const auto n = first.positions.size();
  const bool have_v = first.velocities.size() == n && last.velocities.size() == n;
  const bool have_a = have_v && first.accelerations.size() == n && last.accelerations.size() == n;
  out.resize(selected.size());
  for (size_t j = 0; j < selected.size(); ++j) {
    const size_t k = selected[j];
    const double p0 = first.positions[k], p1 = last.positions[k];
    double c1 = 0., c2 = 0., c3 = 0., c4 = 0., c5 = 0.;
    if (have_v) {
      const double v0 = first.velocities[k], v1 = last.velocities[k];
      c1 = v0 * h;
      if (have_a) {
        const double a0 = first.accelerations[k], a1 = last.accelerations[k];
        c2 = .5 * a0 * h * h;
        const double r0 = p1 - p0 - c1 - c2;
        const double r1 = v1 * h - c1 - 2. * c2;
        const double r2 = a1 * h * h - 2. * c2;
        c3 = 10. * r0 - 4. * r1 + .5 * r2;
        c4 = -15. * r0 + 7. * r1 - r2;
        c5 = 6. * r0 - 3. * r1 + .5 * r2;
      } else {
        const double delta = p1 - p0;
        c2 = 3. * delta - (2. * v0 + v1) * h;
        c3 = -2. * delta + (v0 + v1) * h;
      }
    } else {
      c1 = p1 - p0;
    }
    if (!std::isfinite(p0) || !std::isfinite(c1) || !std::isfinite(c2) || !std::isfinite(c3) ||
        !std::isfinite(c4) || !std::isfinite(c5))
      throw std::runtime_error("nonfinite trajectory polynomial");
    out[j] = {c1, c2, c3, c4, c5};
  }
}

template <class Check>
void sample_dynamics(const Trajectory &tr, const std::vector<std::string> &arm,
                     double step, Check check) {
  check_time(tr);
  const auto selected = columns(tr, arm);
  const auto end = seconds(tr.points.back().time_from_start);
  const auto count = std::max<size_t>(2, static_cast<size_t>(std::ceil(end / step)) + 1);
  std::vector<double> times;
  times.reserve(tr.points.size() + count);
  for (const auto &entry : tr.points) times.push_back(seconds(entry.time_from_start));
  for (size_t i = 0; i < count; ++i) times.push_back(end * i / (count - 1));
  std::sort(times.begin(), times.end());
  times.erase(std::unique(times.begin(), times.end()), times.end());
  std::vector<JointPoly> polys;
  size_t seg = 0;
  const size_t last_seg = tr.points.size() - 2;
  double seg_start = seconds(tr.points[0].time_from_start);
  double seg_end = seconds(tr.points[1].time_from_start);
  selected_segment_polys(tr.points[0], tr.points[1], seg_end - seg_start, selected, polys);
  for (const double time : times) {
    while (seg < last_seg && time >= seg_end) {
      ++seg;
      seg_start = seg_end;
      seg_end = seconds(tr.points[seg + 1].time_from_start);
      selected_segment_polys(tr.points[seg], tr.points[seg + 1], seg_end - seg_start, selected, polys);
    }
    const double h = seg_end - seg_start;
    const double inv_h = 1. / h, inv_h2 = inv_h * inv_h, inv_h3 = inv_h2 * inv_h;
    const double u = (time - seg_start) * inv_h;
    const double u2 = u * u, u3 = u2 * u, u4 = u2 * u2;
    for (size_t i = 0; i < selected.size(); ++i) {
      const auto &p = polys[i];
      const double v = (p.c1 + 2. * p.c2 * u + 3. * p.c3 * u2 + 4. * p.c4 * u3 + 5. * p.c5 * u4) * inv_h;
      const double a = (2. * p.c2 + 6. * p.c3 * u + 12. * p.c4 * u2 + 20. * p.c5 * u3) * inv_h2;
      const double j = (6. * p.c3 + 24. * p.c4 * u + 60. * p.c5 * u2) * inv_h3;
      check(time, i, v, a, j);
    }
  }
}
}  // namespace

// 按固定步长采样，超出动态上限就失败。
void validate_dynamics(const Trajectory &tr, const std::vector<std::string> &arm,
                       const DynamicLimits &limits, double step) {
  sample_dynamics(tr, arm, step, [&](double time, size_t i, double velocity, double acceleration,
                                    double jerk) {
    const auto verify = [&](const char *quantity, double measured, double limit) {
      if (std::abs(measured) > limit * 1.015 + 1e-5)
        throw std::runtime_error(std::string("trajectory ") + quantity + " limit exceeded " +
                                 arm[i] + " | time=" + std::to_string(time) +
                                 " s | value=" + std::to_string(measured) +
                                 " | absolute_limit=" + std::to_string(limit));
    };
    verify("velocity", velocity, limits.velocity[i]);
    verify("acceleration", acceleration, limits.acceleration[i]);
    verify("jerk", jerk, limits.jerk[i]);
  });
}

void stretch_trajectory(Trajectory &trajectory, double scale) {
  for (auto &point : trajectory.points) {
    point.time_from_start = duration(seconds(point.time_from_start) * scale);
    for (auto &velocity : point.velocities) velocity /= scale;
    for (auto &acceleration : point.accelerations) acceleration /= scale * scale;
  }
}

namespace {
// Derivatives in normalized segment time u. Each interval owns both endpoint extrema.
std::array<double, 3> segment_dynamic_peaks(const Eigen::MatrixXd &c, size_t joint) {
  const double velocity_start = c(joint, 1), acceleration_start = 2. * c(joint, 2),
               jerk_start = 6. * c(joint, 3), jerk_linear = 24. * c(joint, 4),
               jerk_quadratic = 60. * c(joint, 5);
  const auto velocity = [&](double u) {
    return (((jerk_quadratic / 12. * u + jerk_linear / 6.) * u + jerk_start / 2.) * u +
            acceleration_start) * u + velocity_start;
  };
  const auto acceleration = [&](double u) {
    return ((jerk_quadratic / 3. * u + jerk_linear / 2.) * u + jerk_start) * u + acceleration_start;
  };
  const auto jerk = [&](double u) { return (jerk_quadratic * u + jerk_linear) * u + jerk_start; };
  double max_jerk = std::max(std::abs(jerk(0.)), std::abs(jerk(1.)));
  if (jerk_quadratic != 0.) {
    const double u = -jerk_linear / (2. * jerk_quadratic);
    if (0. < u && u < 1.) max_jerk = std::max(max_jerk, std::abs(jerk(u)));
  }

  // Zeros of jerk split acceleration into monotone pieces and give its extrema.
  std::array<double, 4> boundaries{0.};
  size_t count = 1;
  const auto add_root = [&](double u) {
    if (0. < u && u < 1.) boundaries[count++] = u;
  };
  if (jerk_quadratic == 0.) {
    if (jerk_linear != 0.) add_root(-jerk_start / jerk_linear);
  } else {
    const double discriminant = std::fma(-4. * jerk_quadratic, jerk_start, jerk_linear * jerk_linear);
    if (discriminant >= 0.) {
      const double numerator = -.5 * (jerk_linear + std::copysign(std::sqrt(discriminant), jerk_linear));
      add_root(numerator / jerk_quadratic);
      if (numerator != 0.) add_root(jerk_start / numerator);
    }
  }
  if (count == 3 && boundaries[1] > boundaries[2]) std::swap(boundaries[1], boundaries[2]);
  boundaries[count++] = 1.;
  double max_acceleration = std::abs(acceleration(0.));
  double max_velocity = std::abs(velocity(0.));
  for (size_t i = 1; i < count; ++i) {
    double left = boundaries[i - 1], right = boundaries[i];
    const double a_left = acceleration(left), a_right = acceleration(right);
    max_acceleration = std::max(max_acceleration, std::abs(a_right));
    max_velocity = std::max(max_velocity, std::abs(velocity(right)));
    if ((a_left < 0. && a_right > 0.) || (a_left > 0. && a_right < 0.)) {
      // A zero of acceleration inside this interval is also a velocity extremum.
      for (int iteration = 0; iteration < 40; ++iteration) {
        const double middle = (left + right) / 2.;
        if ((acceleration(middle) < 0.) == (a_left < 0.)) left = middle;
        else right = middle;
      }
      max_velocity = std::max(max_velocity, std::abs(velocity((left + right) / 2.)));
    }
  }
  return {max_velocity, max_acceleration, max_jerk};
}
}  // namespace

double retime_dynamics(Trajectory &trajectory, const std::vector<std::string> &arm,
                       const DynamicLimits &limits) {
  check_time(trajectory);
  const auto selected = columns(trajectory, arm);
  double scale = 1.;
  Eigen::MatrixXd coefficients;
  for (size_t i = 1; i < trajectory.points.size(); ++i) {
    const auto &first = trajectory.points[i - 1], &last = trajectory.points[i];
    const double h = seconds(last.time_from_start) - seconds(first.time_from_start);
    segment_coefficients(first, last, h, coefficients);
    if (!coefficients.allFinite()) throw std::runtime_error("nonfinite trajectory polynomial");
    for (size_t j = 0; j < selected.size(); ++j) {
      const auto [velocity, acceleration, jerk] = segment_dynamic_peaks(coefficients, selected[j]);
      scale = std::max({scale, velocity / limits.velocity[j] / h,
                        std::sqrt(acceleration / limits.acceleration[j]) / h,
                        std::cbrt(jerk / limits.jerk[j]) / h});
    }
  }
  if (scale > 1.) {
    // Timestamps are rounded to nanoseconds when stretched.
    scale *= 1.01;
    stretch_trajectory(trajectory, scale);
  }
  return scale;
}

namespace {
std::pair<double, size_t> retiming_peak(const Trajectory &trajectory,
                                        const std::vector<size_t> &selected,
                                        const DynamicLimits &limits) {
  double worst_ratio = 1.;
  size_t worst_index = 0;
  std::vector<JointPoly> polys;
  for (size_t i = 0; i + 1 < trajectory.points.size(); ++i) {
    const double t0 = seconds(trajectory.points[i].time_from_start);
    const double t1 = seconds(trajectory.points[i + 1].time_from_start);
    const double h = t1 - t0;
    const double inv_h = 1. / h, inv_h2 = inv_h * inv_h, inv_h3 = inv_h2 * inv_h;
    selected_segment_polys(trajectory.points[i], trajectory.points[i + 1], h, selected, polys);
    for (int sample = 0; sample < 9; ++sample) {
      const double u = sample * (1. / 8.);
      const double u2 = u * u, u3 = u2 * u, u4 = u2 * u2;
      for (size_t j = 0; j < selected.size(); ++j) {
        const auto &p = polys[j];
        const double velocity =
            std::abs((p.c1 + 2. * p.c2 * u + 3. * p.c3 * u2 + 4. * p.c4 * u3 + 5. * p.c5 * u4) * inv_h) /
            limits.velocity[j];
        const double acceleration =
            std::abs((2. * p.c2 + 6. * p.c3 * u + 12. * p.c4 * u2 + 20. * p.c5 * u3) * inv_h2) /
            limits.acceleration[j];
        const double jerk =
            std::abs((6. * p.c3 + 24. * p.c4 * u + 60. * p.c5 * u2) * inv_h3) /
            limits.jerk[j];
        if (velocity <= 1. && acceleration <= 1. && jerk <= 1.) continue;
        const double ratio = std::max({velocity, std::sqrt(acceleration), std::cbrt(jerk)});
        if (ratio > 1.005 && ratio * 1.02 > worst_ratio) {
          worst_ratio = ratio * 1.02;
          worst_index = i;
        }
      }
    }
  }
  return {worst_ratio, worst_index};
}
}  // namespace

bool trajectory_needs_retiming(const Trajectory &trajectory,
                               const std::vector<std::string> &arm,
                               const DynamicLimits &limits) {
  check_time(trajectory);
  return retiming_peak(trajectory, columns(trajectory, arm), limits).first > 1.001;
}

// 合并连续路段，并记下每段在总时间轴上的边界。
BlendedTrajectory blend_trajectories(const std::vector<Trajectory> &segments,
                                     const std::vector<std::string> &arm,
                                     const DynamicLimits &limits,
                                     const std::vector<bool> &contacts) {
  auto source = concatenate(segments);
  check_time(source);
  const auto selected = columns(source, arm);
  std::vector<double> boundaries{0.};
  for (const auto &segment : segments)
    boundaries.push_back(
        seconds(duration(boundaries.back() + seconds(segment.points.back().time_from_start) -
                         seconds(segment.points.front().time_from_start))));
  struct Window {
    double left, right;
    std::array<double, 3> old_times, new_times;
    Trajectory curve;
  };
  std::vector<Window> windows;
  double shift = 0.;
  auto speed = [&](const TrajectorySample &value) {
    double squared = 0.;
    for (auto column : selected) squared += value.velocity[column] * value.velocity[column];
    return std::sqrt(squared);
  };
  TrajectorySampler source_sampler(source);
  for (size_t boundary = 1; boundary + 1 < boundaries.size(); ++boundary) {
    const double center = boundaries[boundary];
    std::vector<std::pair<double, double>> left, right;
    double max_left = 0., max_right = 0.;
    for (const auto &entry : source.points) {
      const double t = seconds(entry.time_from_start);
      if (t >= boundaries[boundary - 1] && t < center) {
        const double at = speed(source_sampler.sample(t));
        left.emplace_back(t, at);
        max_left = std::max(max_left, at);
      }
      if (t > center && t <= boundaries[boundary + 1]) {
        const double at = speed(source_sampler.sample(t));
        right.emplace_back(t, at);
        max_right = std::max(max_right, at);
      }
    }
    if (left.empty() || right.empty()) continue;
    auto before = std::find_if(left.rbegin(), left.rend(), [&](const auto &sample) {
      return sample.second >= std::max(.45 * max_left, 1e-5);
    });
    auto after = std::find_if(right.begin(), right.end(), [&](const auto &sample) {
      return sample.second >= std::max(.45 * max_right, 1e-5);
    });
    if (before == left.rend() || after == right.end()) continue;
    const double l = before->first, r = after->first;
    if (!windows.empty() && l < windows.back().right - 1e-8) continue;
    auto first = source_sampler.sample(l);
    auto middle = source_sampler.sample(center);
    auto last = source_sampler.sample(r);
    Eigen::VectorXd incoming = middle.position - first.position,
                    outgoing = last.position - middle.position;
    const double dl = incoming.norm(), dr = outgoing.norm();
    if (std::min(dl, dr) < 1e-7) continue;
    incoming /= dl;
    outgoing /= dr;
    const double pace = std::min(speed(first), speed(last));
    Eigen::VectorXd direction = incoming + outgoing;
    if (contacts.size() == segments.size()) {
      if (contacts[boundary] && !contacts[boundary - 1])
        direction = outgoing;
      else if (contacts[boundary - 1] && !contacts[boundary])
        direction = incoming;
    }
    if (direction.norm() < 1e-4 || incoming.dot(outgoing) < -.98) continue;
    middle.velocity = direction.normalized() * pace;
    middle.acceleration.setZero();
    const double ld =
        std::min(center - l, std::max(.02, 1.1 * dl / std::max(.5 * (speed(first) + pace), 1e-6)));
    const double rd =
        std::min(r - center, std::max(.02, 1.1 * dr / std::max(.5 * (speed(last) + pace), 1e-6)));
    const std::array<double, 3> mapped{l + shift, l + shift + ld, l + shift + ld + rd};
    Trajectory curve;
    curve.joint_names = source.joint_names;
    curve.points = {point(first, mapped[0]), point(middle, mapped[1]), point(last, mapped[2])};
    windows.push_back({l, r, {l, center, r}, mapped, curve});
    shift += ld + rd - (r - l);
  }
  auto map_time = [&](double t) {
    double value = t;
    for (const auto &window : windows) {
      if (t > window.right + 1e-9)
        value += (window.new_times[2] - window.new_times[0]) - (window.right - window.left);
      else if (t >= window.left - 1e-9) {
        const int half = t <= window.old_times[1] ? 0 : 1;
        return window.new_times[half] + (t - window.old_times[half]) *
                                            (window.new_times[half + 1] - window.new_times[half]) /
                                            (window.old_times[half + 1] - window.old_times[half]);
      }
    }
    return value;
  };
  std::set<double> source_times;
  for (const auto &entry : source.points) source_times.insert(seconds(entry.time_from_start));
  for (const auto &window : windows)
    for (double t : window.old_times) source_times.insert(t);
  Trajectory base;
  base.joint_names = source.joint_names;
  base.header = source.header;
  for (double time : source_times) {
    auto sample = source_sampler.sample(time);
    const double mapped = map_time(time);
    for (const auto &window : windows)
      if (time >= window.left - 1e-9 && time <= window.right + 1e-9) {
        sample = sample_trajectory(window.curve, mapped);
        break;
      }
    auto packed = point(sample, mapped);
    if (!base.points.empty() && packed.time_from_start == base.points.back().time_from_start) {
      base.points.back() = std::move(packed);
    } else {
      base.points.push_back(std::move(packed));
    }
  }
  struct Bump {
    double center, width, amount;
  };
  std::vector<Bump> bumps;
  auto clock = [&](double t) {
    std::array<double, 3> result{t, 1., 0.};
    for (const auto &bump : bumps) {
      const double z = (t - bump.center) / bump.width, weight = bump.amount * std::exp(-.5 * z * z);
      result[0] +=
          bump.amount * bump.width * std::sqrt(M_PI / 2.) *
          (std::erf(z / std::sqrt(2.)) - std::erf(-bump.center / bump.width / std::sqrt(2.)));
      result[1] += weight;
      result[2] -= weight * z / bump.width;
    }
    return result;
  };
  Trajectory result;
  for (int iteration = 0; iteration < 24; ++iteration) {
    result = base;
    for (size_t i = 0; i < result.points.size(); ++i) {
      auto &entry = result.points[i];
      const auto warp = clock(seconds(base.points[i].time_from_start));
      const auto velocity = vector(base.points[i].velocities);
      entry.velocities = list(velocity / warp[1]);
      entry.accelerations = list(vector(base.points[i].accelerations) / (warp[1] * warp[1]) -
                                 velocity * warp[2] / std::pow(warp[1], 3));
      entry.time_from_start = duration(warp[0]);
    }
    const auto [worst_ratio, worst_index] = retiming_peak(result, selected, limits);
    if (worst_ratio <= 1.001) break;
    if (iteration == 23) throw std::runtime_error("local retiming failed dynamic limits");
    const double left = seconds(base.points[worst_index].time_from_start),
                 right = seconds(base.points[worst_index + 1].time_from_start),
                 center = (left + right) * .5;
    bumps.push_back(
        {center, std::max(.2, 2 * (right - left)), (worst_ratio - 1.) * clock(center)[1] * 1.08});
  }
  std::vector<double> mapped_boundaries;
  for (double boundary : boundaries) mapped_boundaries.push_back(clock(map_time(boundary))[0]);
  for (size_t i = 1; i + 1 < mapped_boundaries.size(); ++i)
    if (speed(sample_trajectory(result, mapped_boundaries[i])) < 1e-5)
      throw std::runtime_error("via waypoint cannot be passed continuously");
  validate_dynamics(result, arm, limits);
  return {std::move(result), std::move(mapped_boundaries)};
}
}  // namespace curobo_ros_cpp
