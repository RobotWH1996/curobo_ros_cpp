#!/usr/bin/env python3
"""Publish a cycling MpcTarget list and start /follow_mpc.

The planning node expires a target after about one second, so this process
keeps republishing the current waypoint. Ctrl+C cancels the action.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import Pose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from curobo_ros_interfaces.action import FollowMpc
from curobo_ros_interfaces.msg import MpcTarget


def _number(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是数字") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} 必须是有限数字")
    return number


def _pose(entry, label):
    if not isinstance(entry, dict):
        raise ValueError(f"{label} 必须包含 position 和 orientation")
    position = entry.get("position") or {}
    orientation = entry.get("orientation") or {}
    pose = Pose()
    pose.position.x = _number(position.get("x"), f"{label}.position.x")
    pose.position.y = _number(position.get("y"), f"{label}.position.y")
    pose.position.z = _number(position.get("z"), f"{label}.position.z")
    pose.orientation.x = _number(orientation.get("x"), f"{label}.orientation.x")
    pose.orientation.y = _number(orientation.get("y"), f"{label}.orientation.y")
    pose.orientation.z = _number(orientation.get("z"), f"{label}.orientation.z")
    pose.orientation.w = _number(orientation.get("w"), f"{label}.orientation.w")
    quaternion = (
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
    norm = math.sqrt(sum(component * component for component in quaternion))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"{label} 的姿态必须是单位四元数")
    if all(abs(component) < 1e-6 for component in quaternion[:3]) and abs(abs(quaternion[3]) - 1.0) < 1e-6:
        raise ValueError(f"{label} 不要用 (0, 0, 0, 1)，请填真实末端姿态")
    return pose


def load_route(path):
    """Read the editable waypoint list. Planning stays in the C++ node."""
    with open(path) as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("路点文件必须是字典")
    frame_id = document.get("frame_id") or "base_link"
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("frame_id 必须是非空字符串")
    publish_hz = _number(document.get("publish_hz", 10.0), "publish_hz")
    if publish_hz < 2.0:
        raise ValueError("publish_hz 至少为 2，否则目标会在节点侧过期")
    timeout = _number(document.get("timeout", 0.0), "timeout")
    position_tolerance = _number(document.get("position_tolerance", 0.0), "position_tolerance")
    orientation_tolerance = _number(document.get("orientation_tolerance", 0.0), "orientation_tolerance")
    raw_waypoints = document.get("waypoints")
    if not isinstance(raw_waypoints, list) or not raw_waypoints:
        raise ValueError("waypoints 至少要有一条")
    waypoints = []
    for index, raw in enumerate(raw_waypoints, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"第 {index} 条路点必须是字典")
        duration_s = _number(raw.get("duration_s"), f"第 {index} 条 duration_s")
        if duration_s <= 0.0:
            raise ValueError(f"第 {index} 条 duration_s 必须大于 0")
        item = {"duration_s": duration_s, "frame_id": raw.get("frame_id") or frame_id}
        for side in ("left", "right"):
            if raw.get(side):
                item[side] = _pose(raw[side], f"第 {index} 条 {side}")
        if "left" not in item and "right" not in item:
            raise ValueError(f"第 {index} 条至少要有 left 或 right")
        waypoints.append(item)
    return {
        "frame_id": frame_id,
        "publish_hz": publish_hz,
        "timeout": timeout,
        "position_tolerance": position_tolerance,
        "orientation_tolerance": orientation_tolerance,
        "waypoints": waypoints,
    }


def spin_until(node, future, timeout_sec=None):
    """Short waits so Ctrl+C is delivered between messages."""
    deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
    while rclpy.ok() and not future.done():
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0.0:
            return False
        rclpy.spin_once(node, timeout_sec=0.1 if remaining is None else min(0.1, remaining))
    return future.done()


def cancel_goal(node, handle, result_future):
    print("正在取消跟随", file=sys.stderr, flush=True)
    deadline = time.monotonic() + 5.0
    acknowledgement = handle.cancel_goal_async()
    if result_future is None:
        result_future = handle.get_result_async()
    try:
        if not spin_until(node, acknowledgement, timeout_sec=5.0):
            print("取消确认超时", file=sys.stderr, flush=True)
        if not spin_until(node, result_future, timeout_sec=max(0.0, deadline - time.monotonic())):
            print("取消后没有收到跟随结果", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        print("再次中断，不再等待取消结果", file=sys.stderr, flush=True)


class RoutePublisher(Node):
    """Republish the active waypoint and advance when its dwell time ends."""

    def __init__(self, route, topic):
        super().__init__("follow_mpc_route_client")
        self.route = route
        self.index = 0
        self.since = time.monotonic()
        self.publisher = self.create_publisher(MpcTarget, topic, 10)
        self.timer = self.create_timer(1.0 / route["publish_hz"], self._tick)
        self.timer.cancel()

    def start_route(self):
        self.since = time.monotonic()
        self._announce()
        self._publish()
        self.timer.reset()

    def _announce(self):
        waypoint = self.route["waypoints"][self.index]
        sides = "、".join(side for side in ("left", "right") if side in waypoint)
        print(
            f"路点 {self.index + 1}/{len(self.route['waypoints'])}，停留 {waypoint['duration_s']:.1f} 秒，{sides}",
            flush=True)

    def _publish(self):
        waypoint = self.route["waypoints"][self.index]
        message = MpcTarget()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = waypoint["frame_id"]
        for side in ("left", "right"):
            pose = waypoint.get(side)
            if pose is None:
                continue
            stamped = getattr(message, side)
            stamped.header.stamp = message.header.stamp
            stamped.header.frame_id = waypoint["frame_id"]
            stamped.pose = pose
            setattr(message, f"has_{side}", True)
        self.publisher.publish(message)

    def _tick(self):
        waypoint = self.route["waypoints"][self.index]
        if time.monotonic() - self.since >= waypoint["duration_s"]:
            self.index = (self.index + 1) % len(self.route["waypoints"])
            self.since = time.monotonic()
            self._announce()
        self._publish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waypoints", required=True, help="循环路点和停留时间")
    parser.add_argument("--topic", default="/curobo/target", help="MpcTarget 话题")
    args = parser.parse_args(argv)
    try:
        route = load_route(args.waypoints)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    print(
        f"共 {len(route['waypoints'])} 条路点，坐标系 {route['frame_id']}，"
        f"按 {route['publish_hz']:.0f} Hz 发到 {args.topic}。Ctrl+C 停止跟随。",
        flush=True)
    # rclpy 的 SIGINT 处理会先关掉 context，取消发不出去。
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = RoutePublisher(route, args.topic)
    client = ActionClient(node, FollowMpc, "follow_mpc")
    handle = None
    goal_future = None
    result_future = None
    try:
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("找不到 /follow_mpc，请先启动规划节点")
        # Wait for target discovery before sending the Action request.
        deadline = time.monotonic() + 5.0
        while node.publisher.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"找不到 {args.topic} 的订阅者")
            rclpy.spin_once(node, timeout_sec=0.05)
        node.start_route()
        goal = FollowMpc.Goal()
        goal.timeout = route["timeout"]
        goal.position_tolerance = route["position_tolerance"]
        goal.orientation_tolerance = route["orientation_tolerance"]
        last_feedback = 0.0

        def feedback(msg):
            nonlocal last_feedback
            now = time.monotonic()
            if now - last_feedback < 0.5:
                return
            last_feedback = now
            update = msg.feedback
            print(
                f"跟随 路点 {node.index + 1}/{len(route['waypoints'])} "
                f"left={update.left_position_error:.3f} m {update.left_orientation_error:.3f} rad "
                f"right={update.right_position_error:.3f} m {update.right_orientation_error:.3f} rad "
                f"plan={update.plan_success} {update.message}",
                flush=True)

        try:
            goal_future = client.send_goal_async(goal, feedback_callback=feedback)
            if not spin_until(node, goal_future):
                raise RuntimeError("没有收到目标确认")
            handle = goal_future.result()
            if handle is None or not handle.accepted:
                raise RuntimeError("/follow_mpc 拒绝了目标")
            print("已启动 /follow_mpc", flush=True)
            result_future = handle.get_result_async()
            if not spin_until(node, result_future):
                raise RuntimeError("没有收到跟随结果")
        except KeyboardInterrupt:
            if handle is None and goal_future is not None and not goal_future.done():
                try:
                    spin_until(node, goal_future, timeout_sec=2.0)
                except KeyboardInterrupt:
                    print("再次中断，不再等待目标句柄", file=sys.stderr, flush=True)
            if handle is None and goal_future is not None and goal_future.done():
                handle = goal_future.result()
            if handle is not None and handle.accepted:
                cancel_goal(node, handle, result_future)
            else:
                print("已中断，未收到可取消的目标句柄", file=sys.stderr, flush=True)
            raise SystemExit(130)
        value = result_future.result()
        print(f"跟随结束 success={value.result.success} {value.result.message}", flush=True)
        if value.result.success:
            return
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("已中断，未提交跟随", file=sys.stderr, flush=True)
        raise SystemExit(130)
    finally:
        node.timer.cancel()
        client.destroy()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
