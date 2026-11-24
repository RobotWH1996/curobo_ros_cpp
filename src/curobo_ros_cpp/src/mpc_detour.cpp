// SPDX-License-Identifier: Apache-2.0
#include "curobo_ros_cpp/mpc_detour.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>

namespace curobo_ros_cpp {
namespace {
bool intersects_voxel(const Eigen::Vector3d &center, const Eigen::Vector3d &motion,
                      double radius_squared, double half_cell) {
  // Between face crossings the squared distance from the segment to the box is quadratic.
  std::array<double, 8> edges{0., 1.};
  size_t count = 2;
  for (int axis = 0; axis < 3; ++axis) {
    if (motion[axis] == 0.) continue;
    for (double face : {-half_cell, half_cell}) {
      const double t = (center[axis] + face) / motion[axis];
      if (0. < t && t < 1.) edges[count++] = t;
    }
  }
  std::sort(edges.begin(), edges.begin() + count);
  for (size_t i = 1; i < count; ++i) {
    const double mid = (edges[i-1] + edges[i]) / 2.;
    Eigen::Vector3d offset = Eigen::Vector3d::Zero(), slope = Eigen::Vector3d::Zero();
    for (int axis = 0; axis < 3; ++axis) {
      const double delta = mid * motion[axis] - center[axis];
      if (std::abs(delta) <= half_cell) continue;
      offset[axis] = -center[axis] - std::copysign(half_cell, delta);
      slope[axis] = motion[axis];
    }
    const double speed2 = slope.squaredNorm();
    const double t = speed2 > 0. ? std::clamp(-offset.dot(slope) / speed2, edges[i-1], edges[i])
                                  : mid;
    if ((offset + t * slope).squaredNorm() < radius_squared) return true;
  }
  return false;
}
}  // namespace

std::optional<std::array<Eigen::Vector3d, 2>> suggest_mpc_detour(
    const Eigen::Vector3d &start, const Eigen::Vector3d &goal,
    const std::vector<Eigen::Vector4d> &gripper, const std::vector<Eigen::Vector3f> &points,
    double pitch, double clearance) {
  const Eigen::Vector3d delta = goal - start;
  if (gripper.empty() || points.empty() || delta.head<2>().norm() < 4. * pitch ||
      std::abs(delta.z()) > 2. * pitch) return std::nullopt;
  const double length = delta.norm();
  const Eigen::Vector3d forward = delta / length;
  Eigen::Vector3d sideways = Eigen::Vector3d::UnitZ().cross(forward).normalized();
  if (sideways.dot(start) > 0.) sideways = -sideways;

  // Use the same occupied cells as the mesh. Bounding spheres only skip distant cells.
  Eigen::Vector3d origin = points.front().cast<double>();
  for (const auto &point : points) origin = origin.cwiseMin(point.cast<double>());
  origin.array() -= pitch;
  std::vector<Eigen::Vector3d> centers;
  centers.reserve(points.size());
  for (const auto &point : points)
    centers.emplace_back(origin.array() + (((point.cast<double>() - origin).array() / pitch)
                                              .floor() + .5) * pitch);

  struct Sweep {
    double radius_squared;
    double reach_squared;
    std::vector<Eigen::Vector3d> points;
  };
  std::vector<Sweep> sweeps;
  double enter = length, leave = 0.;
  bool blocked = false;
  const double half_cell = pitch / 2.;
  for (const auto &ball : gripper) {
    const double radius = ball.w() + clearance;
    const double reach = radius + std::sqrt(3.) * half_cell;
    Sweep sweep{radius * radius, reach * reach, {}};
    const double max_reach = reach + std::hypot(6., 3.) * pitch;
    for (const auto &center : centers) {
      const Eigen::Vector3d p = center - ball.head<3>();
      const double along = p.dot(forward);
      const double side2 = (p - along * forward).squaredNorm();
      const double overshoot = along - std::clamp(along, 0., length);
      if (side2 + overshoot * overshoot > max_reach * max_reach) continue;
      sweep.points.push_back(p);
      if (side2 + overshoot * overshoot >= sweep.reach_squared ||
          !intersects_voxel(p, delta, sweep.radius_squared, half_cell)) continue;
      const double half = std::sqrt(sweep.reach_squared - side2);
      enter = std::min(enter, std::max(0., along - half));
      leave = std::max(leave, std::min(length, along + half));
      blocked = true;
    }
    if (!sweep.points.empty()) sweeps.push_back(std::move(sweep));
  }
  if (!blocked) return std::nullopt;

  const double before = std::max(0., enter - pitch);
  const double after = std::min(length, leave + pitch);
  struct Candidate {
    double length;
    double lateral;
    Eigen::Vector3d first, second;
  };
  std::vector<Candidate> candidates;
  for (int up = 0; up <= 6; ++up)
    for (int side : {0, 1, -1, 2, -2, 3, -3}) {
      if (up == 0 && side == 0) continue;
      const Eigen::Vector3d offset = pitch * (side * sideways + up * Eigen::Vector3d::UnitZ());
      for (double fraction : {1., .5, 0.}) {
        const double entry = fraction * before;
        const double exit = length - fraction * (length - after);
        const Eigen::Vector3d first = entry * forward + offset;
        const Eigen::Vector3d second = exit * forward + offset;
        candidates.push_back({first.norm() + exit - entry + (delta - second).norm(),
                              static_cast<double>(std::abs(side)), first, second});
      }
    }
  std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b) {
    return std::tie(a.length, a.lateral) < std::tie(b.length, b.lateral);
  });
  const auto clear = [&](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
    const Eigen::Vector3d segment = b - a;
    const double inverse_length = 1. / segment.squaredNorm();
    for (const auto &sweep : sweeps)
      for (const auto &point : sweep.points) {
        const Eigen::Vector3d rel = point - a;
        const double u = std::clamp(rel.dot(segment) * inverse_length, 0., 1.);
        const double gap2 = (rel - u * segment).squaredNorm();
        if (gap2 >= sweep.reach_squared) continue;
        if (gap2 < sweep.radius_squared ||
            intersects_voxel(rel, segment, sweep.radius_squared, half_cell)) return false;
      }
    return true;
  };
  for (const auto &candidate : candidates)
    if (clear(Eigen::Vector3d::Zero(), candidate.first) &&
        clear(candidate.first, candidate.second) && clear(candidate.second, delta))
      return std::array<Eigen::Vector3d, 2>{start + candidate.first, start + candidate.second};
  return std::nullopt;
}
}  // namespace curobo_ros_cpp
