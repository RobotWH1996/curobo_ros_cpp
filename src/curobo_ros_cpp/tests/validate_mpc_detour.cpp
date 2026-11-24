// SPDX-License-Identifier: Apache-2.0
#include "curobo_ros_cpp/mpc_detour.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using curobo_ros_cpp::suggest_mpc_detour;

int main() {
  using Eigen::Vector3d;
  using Eigen::Vector3f;
  constexpr double pitch = .02, margin = .005, radius = .03;
  const Vector3d start(.75, 0., .8), goal(.75, .50, .8);
  using Cloud = std::vector<Vector3f>;
  const auto one_ball = [&](const Vector3d &at) {
    return std::vector<Eigen::Vector4d>{{at.x(), at.y(), at.z(), radius}};
  };
  const auto require = [](bool pass, const char *reason) {
    if (!pass) throw std::runtime_error(reason);
  };
  const auto length = [](const auto &path) {
    double result = 0.;
    for (size_t i = 1; i < path.size(); ++i) result += (path[i] - path[i - 1]).norm();
    return result;
  };
  // Independent dense sampling of the physical gripper sphere against occupied boxes.
  const auto clear = [&](const auto &route, const Cloud &cloud) {
    Vector3d origin = cloud.front().cast<double>();
    for (const auto &point : cloud) origin = origin.cwiseMin(point.cast<double>());
    origin.array() -= pitch;
    std::vector<Vector3d> voxels;
    for (const auto &point : cloud)
      voxels.emplace_back(origin.array() + (((point.cast<double>() - origin).array() / pitch)
                                            .floor() + .5) * pitch);
    for (size_t i = 1; i < route.size(); ++i)
      for (int n = 0; n <= 500; ++n) {
        const Vector3d position = route[i - 1] + (n / 500.) * (route[i] - route[i - 1]);
        for (const auto &voxel : voxels)
          if (((voxel - position).array().abs() - pitch / 2.).max(0.).matrix().norm() <
              radius + margin) return false;
      }
    return true;
  };
  const auto path = [&](const auto &from, const auto &to, const Cloud &cloud) {
    const auto waypoints = suggest_mpc_detour(from, to, one_ball(from), cloud, pitch, margin);
    require(waypoints.has_value(), "a compact bypass should exist");
    const auto &[first, second] = *waypoints;
    const std::array<Vector3d, 4> route{from, first, second, to};
    require(clear(route, cloud), "route must clear the independently sampled occupied boxes");
    return route;
  };
  const std::array<Vector3d, 2> straight{start, goal};
  Cloud low;
  for (int x = -8; x <= 8; ++x) low.emplace_back(.741 + x * pitch, .235, .767);
  require(!clear(straight, low), "low obstacle must cross the direct route");
  const auto over = path(start, goal, low);
  const Vector3d old_offset(-.04, 0., .08);
  const std::array<Vector3d, 4> old{start, start + old_offset, goal + old_offset, goal};
  require(over[1].z() > start.z() && over[1].z() < start.z() + .08 &&
          over[1].y() > .05 && over[2].y() < .45 && length(over) < length(old),
          "a low obstacle should need a local, smaller lift");
  const auto reverse = path(goal, start, low);
  require(reverse[1].z() > start.z() && length(reverse) < length(old), "reverse crossing");

  Cloud tall;
  for (int z = -3; z <= 10; ++z) tall.emplace_back(.743, .235, .795 + z * pitch);
  require(!clear(straight, tall), "narrow post must cross the direct route");
  const auto around = path(start, goal, tall);
  require(std::abs(around[1].x() - start.x()) >= pitch && around[1].z() <= start.z() + pitch,
          "prefer a short lateral path around a narrow tall post");

  Cloud wall;
  for (int z = -3; z <= 10; ++z)
    for (int x = -10; x <= 10; ++x) wall.emplace_back(.741 + x * pitch, .235, .795 + z * pitch);
  require(!suggest_mpc_detour(start, goal, one_ball(start), wall, pitch, margin),
          "do not suggest a route through a wide tall wall");
  require(!suggest_mpc_detour(start, goal, one_ball(start), {{.747f, .50f, .798f}}, pitch, margin),
          "do not suggest a route to an obstructed goal");
  require(!suggest_mpc_detour(start, goal, one_ball(start), {{1.4f, .25f, .798f}}, pitch, margin),
          "do not detour around an off-path point");
  std::cout << "Low obstacle geometric path m old=" << length(old) << " new=" << length(over)
            << " lift=" << over[1].z() - start.z() << '\n';
  std::cout << "Tall narrow obstacle lateral m=" << std::abs(around[1].x() - start.x())
            << " length m=" << length(around) << '\n';
  std::cout << "MPC detour geometry validation passed\n";
}
