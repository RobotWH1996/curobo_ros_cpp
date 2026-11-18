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
// 连续采样同一段时，复用多项式系数和结果缓冲区。
class TrajectorySampler {
 public:
  explicit TrajectorySampler(const Trajectory &trajectory) : tr(trajectory) {
    if (tr.points.size() < 2) throw std::runtime_error("trajectory requires two points");
  }
  const TrajectorySample &sample(double time);

 private:
  void load_segment(double time);

  const Trajectory &tr;
  size_t index = 0;
  double start = 0., end = 0.;
  Eigen::MatrixXd coefficients;
  TrajectorySample values;
};

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
  const auto n = tr.joint_names.size();
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

const TrajectorySample &TrajectorySampler::sample(double time) {
  if (coefficients.cols() == 0 || (index > 0 && time < start) ||
      (index + 2 < tr.points.size() && time >= end))
    load_segment(time);
  const double h = end - start;
  const double u = (time - start) / h;
  std::array<Eigen::VectorXd *, 4> derivatives{
      &values.position, &values.velocity, &values.acceleration, &values.jerk};
  for (int derivative = 0; derivative < 4; ++derivative) {
    auto &value = *derivatives[derivative];
    value.setZero(coefficients.rows());
    for (int power = derivative; power < 6; ++power) {
      double multiplier = 1.;
      for (int k = 0; k < derivative; ++k) multiplier *= power - k;
      value += multiplier * std::pow(u, power - derivative) * coefficients.col(power);
    }
    value /= std::pow(h, derivative);
    if (!value.allFinite())
      throw std::runtime_error("nonfinite trajectory polynomial");
  }
  return values;
}
}  // namespace

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

// 按固定步长采样，超出动态上限就失败。
void validate_dynamics(const Trajectory &tr, const std::vector<std::string> &arm,
                       const DynamicLimits &limits, double step) {
  check_time(tr);
  const auto selected = columns(tr, arm);
  const auto end = seconds(tr.points.back().time_from_start);
  std::set<double> times;
  for (const auto &entry : tr.points) times.insert(seconds(entry.time_from_start));
  const auto count = std::max<size_t>(2, static_cast<size_t>(std::ceil(end / step)) + 1);
  for (size_t i = 0; i < count; ++i) times.insert(end * i / (count - 1));
  TrajectorySampler sampler(tr);
  for (const double time : times) {
    const auto &value = sampler.sample(time);
    for (size_t i = 0; i < selected.size(); ++i) {
      const auto j = selected[i];
      const auto verify = [&](const char *quantity, double measured, double limit) {
        if (std::abs(measured) > limit * 1.015 + 1e-5)
          throw std::runtime_error(std::string("trajectory ") + quantity + " limit exceeded " +
                                   arm[i] + " | time=" + std::to_string(time) +
                                   " s | value=" + std::to_string(measured) +
                                   " | absolute_limit=" + std::to_string(limit));
      };
      verify("velocity", value.velocity[j], limits.velocity[i]);
      verify("acceleration", value.acceleration[j], limits.acceleration[i]);
      verify("jerk", value.jerk[j], limits.jerk[i]);
    }
  }
}

namespace {
std::pair<double, size_t> retiming_peak(const Trajectory &trajectory,
                                        const std::vector<size_t> &selected,
                                        const DynamicLimits &limits) {
  double worst_ratio = 1.;
  size_t worst_index = 0;
  TrajectorySampler sampler(trajectory);
  for (size_t i = 0; i + 1 < trajectory.points.size(); ++i)
    for (int sample = 0; sample < 9; ++sample) {
      const double t = seconds(trajectory.points[i].time_from_start) +
                       (seconds(trajectory.points[i + 1].time_from_start) -
                        seconds(trajectory.points[i].time_from_start)) *
                           sample / 8.;
      const auto &value = sampler.sample(t);
      for (size_t j = 0; j < selected.size(); ++j) {
        const auto k = selected[j];
        const double ratio =
            std::max({std::abs(value.velocity[k]) / limits.velocity[j],
                      std::sqrt(std::abs(value.acceleration[k]) / limits.acceleration[j]),
                      std::cbrt(std::abs(value.jerk[k]) / limits.jerk[j])});
        if (ratio > 1.005 && ratio * 1.02 > worst_ratio) {
          worst_ratio = ratio * 1.02;
          worst_index = i;
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
  for (size_t boundary = 1; boundary + 1 < boundaries.size(); ++boundary) {
    const double center = boundaries[boundary];
    std::vector<double> left, right;
    double max_left = 0., max_right = 0.;
    for (const auto &entry : source.points) {
      const double t = seconds(entry.time_from_start);
      if (t >= boundaries[boundary - 1] && t < center) {
        left.push_back(t);
        max_left = std::max(max_left, speed(sample_trajectory(source, t)));
      }
      if (t > center && t <= boundaries[boundary + 1]) {
        right.push_back(t);
        max_right = std::max(max_right, speed(sample_trajectory(source, t)));
      }
    }
    if (left.empty() || right.empty()) continue;
    auto before = std::find_if(left.rbegin(), left.rend(), [&](double t) {
      return speed(sample_trajectory(source, t)) >= std::max(.45 * max_left, 1e-5);
    });
    auto after = std::find_if(right.begin(), right.end(), [&](double t) {
      return speed(sample_trajectory(source, t)) >= std::max(.45 * max_right, 1e-5);
    });
    if (before == left.rend() || after == right.end()) continue;
    const double l = *before, r = *after;
    if (!windows.empty() && l < windows.back().right - 1e-8) continue;
    auto first = sample_trajectory(source, l), middle = sample_trajectory(source, center),
         last = sample_trajectory(source, r);
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
  TrajectorySampler source_sampler(source);
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
