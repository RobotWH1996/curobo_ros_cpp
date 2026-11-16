#!/usr/bin/env bash
# 用可编辑的笛卡尔路点测试 /plan_pose_goal，默认只提交第一条。
set -eo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    cat <<'EOF'
用法 ./src/curobo_ros_cpp/tests/call_opt.sh [--waypoint N] [--waypoints FILE]
默认读取 follow_mpc_waypoints.yaml，只提交第 1 条路点。
服务成功表示轨迹规划通过。是否下发控制器由节点的 execute_trajectory 决定。
EOF
    exit 0
fi

source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
source "$repo_root/install/setup.bash"

exec "${CUROBO_PYTHON:-python3}" - "$script_dir" "$@" <<'PY'
import argparse
import sys
import time
from pathlib import Path

import rclpy
import yaml
from curobo_ros_interfaces.srv import PlanPoseGoal
from geometry_msgs.msg import PoseStamped

script_dir = Path(sys.argv[1])
sys.path.insert(0, str(script_dir))
from follow_mpc_client import load_route

parser = argparse.ArgumentParser(description="测试单条 OPT 笛卡尔路点")
parser.add_argument("--waypoints", type=Path, default=script_dir / "follow_mpc_waypoints.yaml")
parser.add_argument("--waypoint", type=int, default=1)
args = parser.parse_args(sys.argv[2:])
try:
    waypoints = load_route(args.waypoints)["waypoints"]
except (OSError, ValueError, yaml.YAMLError) as error:
    parser.error(str(error))
if not 1 <= args.waypoint <= len(waypoints):
    parser.error(f"--waypoint 必须在 1 到 {len(waypoints)} 之间")

waypoint = waypoints[args.waypoint - 1]
sides = [side for side in ("left", "right") if side in waypoint]
request = PlanPoseGoal.Request()
request.group_name = "both_arms" if len(sides) == 2 else f"{sides[0]}_arm"

def stamped(side):
    pose = PoseStamped()
    pose.header.frame_id = waypoint["frame_id"]
    pose.pose = waypoint[side]
    return pose

request.waypoints = [stamped(sides[0])]
if len(sides) == 2:
    request.other_waypoints = [stamped(sides[1])]

rclpy.init()
node = rclpy.create_node("opt_test_client")
try:
    client = node.create_client(PlanPoseGoal, "/plan_pose_goal")
    if not client.wait_for_service(timeout_sec=10.0):
        sys.exit("找不到 /plan_pose_goal")
    print(f"OPT 请求 ｜ {request.group_name} ｜ 路点 {args.waypoint}/{len(waypoints)} ｜ "
          f"坐标系 {waypoint['frame_id']}", flush=True)
    started = time.monotonic()
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=60.0)
    if not future.done():
        sys.exit("等待 OPT 服务结果超时")
    response = future.result()
    print(f"OPT 结果 ｜ success={response.success} ｜ {(time.monotonic() - started) * 1000:.1f} ms ｜ "
          f"{response.message}")
    sys.exit(0 if response.success else 1)
finally:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
PY
