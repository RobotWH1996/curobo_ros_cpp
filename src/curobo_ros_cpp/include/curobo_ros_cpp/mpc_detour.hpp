// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <Eigen/Core>
#include <array>
#include <optional>
#include <vector>

namespace curobo_ros_cpp {
// Suggest two Cartesian waypoints using the swept gripper at fixed orientation.
// The gripper spheres store world xyz and radius. Whole-arm planning still validates the route.
std::optional<std::array<Eigen::Vector3d, 2>> suggest_mpc_detour(
    const Eigen::Vector3d &start, const Eigen::Vector3d &goal,
    const std::vector<Eigen::Vector4d> &gripper, const std::vector<Eigen::Vector3f> &points,
    double pitch, double clearance);
}  // namespace curobo_ros_cpp
