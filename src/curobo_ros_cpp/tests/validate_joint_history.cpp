// SPDX-License-Identifier: Apache-2.0
#include "curobo_ros_cpp/joint_history.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

using curobo_ros_cpp::JointHistory;

int main() {
  const auto require = [](bool passed, const char *description) {
    if (!passed) throw std::runtime_error(description);
  };
  constexpr int64_t ms = 1'000'000;
  constexpr int64_t capture = 1'795'430'090'969'000'000;
  JointHistory history;
  history.track("arm");
  history.track("gripper");
  require(!history.observe_clock(capture + 200 * ms), "first observation starts clock epoch");
  history.record("arm", capture - 10 * ms, 0.1);
  history.record("arm", capture + 190 * ms, 1.5);
  history.record("arm", capture + 10 * ms, 0.3);  // arrived out of order
  history.record("gripper", capture, 0.01);       // a different publisher
  const auto arm = history.position("arm", capture);
  require(arm && std::abs(*arm - .2) < 1e-12, "old cloud must use interpolated capture pose");
  require(history.position("gripper", capture) == .01, "exact partial sample");
  require(!history.position("arm", capture + 50 * ms), "reject interpolation across a gap");
  require(!history.position("gripper", capture + ms), "no forward extrapolation");
  require(!history.position("arm", capture - 11 * ms), "no backward extrapolation");
  history.record("arm", capture + 10 * ms, std::numeric_limits<double>::quiet_NaN());
  require(!history.position("arm", capture), "do not interpolate through invalid feedback");
  history.record("arm", capture + 3'000 * ms, .6);
  history.record("arm", capture, .7);
  require(!history.position("arm", capture), "expire old samples and ignore late expired samples");
  require(history.observe_clock(capture), "clock rewind must invalidate the cloud timestamp too");
  require(!history.observe_clock(capture), "paused ROS time keeps the current epoch");
  require(!history.position("gripper", capture), "clear all tracks when ROS time moves backward");
  for (int i = 0; i <= 512; ++i)
    history.record("arm", capture + i * ms, .1);
  require(!history.position("arm", capture), "bound history to 512 samples");
  require(history.position("arm", capture + ms) == .1, "keep the next sample within the cap");
  std::cout << "Joint history validation passed\n";
}
