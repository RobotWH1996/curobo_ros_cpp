// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <utility>

namespace curobo_ros_cpp {
// Stamps are ROS time in nanoseconds. The caller serializes reads and writes.
class JointHistory {
 public:
  void track(const std::string &name) { samples_.try_emplace(name); }

  bool observe_clock(int64_t now) {
    const bool rewound = now < clock_;
    if (rewound)
      for (auto &entry : samples_) entry.second.clear();
    clock_ = now;
    return rewound;
  }

  void record(const std::string &name, int64_t stamp, double position) {
    auto found = samples_.find(name);
    if (stamp <= 0 || found == samples_.end()) return;
    auto &samples = found->second;
    if (samples.empty() || stamp > samples.back().first) {
      samples.emplace_back(stamp, position);
    } else {
      if (stamp < samples.back().first - kHistoryNs) return;
      const auto at = lower(samples, stamp);
      if (at->first == stamp) at->second = position;
      else samples.insert(at, {stamp, position});
    }
    while (samples.size() > kMaxSamples || samples.front().first < samples.back().first - kHistoryNs)
      samples.pop_front();
  }

  std::optional<double> position(const std::string &name, int64_t stamp) const {
    const auto &samples = samples_.at(name);
    const auto right = lower(samples, stamp);
    if (right == samples.end()) return std::nullopt;
    if (right->first == stamp)
      return std::isfinite(right->second) ? std::optional<double>(right->second) : std::nullopt;
    if (right == samples.begin()) return std::nullopt;
    const auto left = std::prev(right);
    const auto gap = right->first - left->first;
    if (gap > kMaxGapNs || !std::isfinite(left->second) || !std::isfinite(right->second))
      return std::nullopt;
    const double fraction = static_cast<double>(stamp - left->first) / gap;
    return (1. - fraction) * left->second + fraction * right->second;
  }

 private:
  using Samples = std::deque<std::pair<int64_t, double>>;
  template<class Sequence>
  static auto lower(Sequence &samples, int64_t stamp) -> decltype(samples.begin()) {
    return std::lower_bound(samples.begin(), samples.end(), stamp,
                            [](const auto &sample, int64_t at) { return sample.first < at; });
  }
  static constexpr int64_t kHistoryNs = 2'000'000'000;
  static constexpr size_t kMaxSamples = 512;
  static constexpr int64_t kMaxGapNs = 100'000'000;
  std::map<std::string, Samples> samples_;
  int64_t clock_ = 0;
};
}  // namespace curobo_ros_cpp
