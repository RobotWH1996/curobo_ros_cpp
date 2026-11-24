#!/usr/bin/python3
"""Cycle the left tool between its current pose and a base_link -Y target."""
import argparse
import copy
import json
import math
import sys
import time

import rclpy
from curobo_ros_interfaces.action import FollowMpc
from geometry_msgs.msg import Pose
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from follow_mpc_client import RoutePublisher, cancel_goal, spin_until


TARGET_TOPIC = "/curobo/target"
ACTION_NAME = "/follow_mpc"
BASE_FRAME = "base_link"
TOOL_FRAME = "left_arm_ee_link"
ARRIVAL_POSITION_M = .015
ARRIVAL_ANGLE_RAD = .10
ARRIVAL_DWELL_S = .35
WAYPOINT_TIMEOUT_S = 25.


def wait_for(node, predicate, seconds):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    return predicate()


def read_pose(node, buffer):
    try:
        stamped = buffer.lookup_transform(BASE_FRAME, TOOL_FRAME, Time())
    except TransformException:
        return None
    age = (node.get_clock().now() - Time.from_msg(stamped.header.stamp)).nanoseconds
    if not 0 <= age < 500_000_000:
        return None
    translation = stamped.transform.translation
    pose = Pose(orientation=stamped.transform.rotation)
    pose.position.x, pose.position.y, pose.position.z = translation.x, translation.y, translation.z
    return pose


def waypoints(start, offset_y):
    offset = copy.deepcopy(start)
    offset.position.y += offset_y
    route = [("去程终点", offset), ("返回原位", start)]
    return [{"name": name, "frame_id": BASE_FRAME, "left": pose} for name, pose in route]


def within_arrival(actual, target):
    pa, pb = actual.position, target.position
    qa, qb = actual.orientation, target.orientation
    dot = qa.x * qb.x + qa.y * qb.y + qa.z * qb.z + qa.w * qb.w
    return (math.dist((pa.x, pa.y, pa.z), (pb.x, pb.y, pb.z)) <= ARRIVAL_POSITION_M
            and abs(dot) >= math.cos(ARRIVAL_ANGLE_RAD / 2.))


class ArrivalRoutePublisher(RoutePublisher):
    def __init__(self):
        super().__init__({"publish_hz": 10., "waypoints": []}, TARGET_TOPIC)
        self.arrived_at = None

    def _announce(self):
        waypoint = self.route["waypoints"][self.index]
        p = waypoint["left"].position
        print(f"路点 {self.index + 1}/{len(self.route['waypoints'])} {waypoint['name']} "
              f"xyz=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})", flush=True)

    def _tick(self):
        now = time.monotonic()
        actual = self.pose_feedback()
        target = self.route["waypoints"][self.index]["left"]
        if actual is not None and within_arrival(actual, target):
            if self.arrived_at is None:
                self.arrived_at = now
            elif now - self.arrived_at >= ARRIVAL_DWELL_S:
                self.index = (self.index + 1) % len(self.route["waypoints"])
                self.since, self.arrived_at = now, None
                self._announce()
        else:
            self.arrived_at = None
        self._publish()


def check_action_server_count(node):
    count = node.count_publishers(f"{ACTION_NAME}/_action/status")
    if count != 1:
        raise RuntimeError(f"{ACTION_NAME} 检测到 {count} 个服务端，请确认仅启动一个规划节点")


def run(node, offset_y):
    tf = Buffer()
    listener = TransformListener(tf, node)
    action = ActionClient(node, FollowMpc, ACTION_NAME)
    prepare = node.create_client(Trigger, "/inex/prepare_mpc")
    stop = node.create_client(Trigger, "/inex/stop_mpc")
    params = node.create_client(GetParameters, "/curobo_motion_planning_node/get_parameters")
    bridge = {}
    bridge_at = cloud_at = 0.

    def on_status(message):
        nonlocal bridge_at
        bridge.update(json.loads(message.data))
        bridge_at = time.monotonic()

    def on_cloud(_message):
        nonlocal cloud_at
        cloud_at = time.monotonic()

    node.create_subscription(String, "/inex/execution_status", on_status, 10)
    if not action.wait_for_server(timeout_sec=4.) or not params.wait_for_service(timeout_sec=3.):
        raise RuntimeError("请先启动普通 MPC 规划节点")
    if not prepare.wait_for_service(timeout_sec=3.) or not stop.wait_for_service(timeout_sec=3.):
        raise RuntimeError("请先用 --mpc-group left_arm 启动桥接")
    query = params.call_async(GetParameters.Request(names=["execute_trajectory", "pointcloud_topic"]))
    if not spin_until(node, query, timeout_sec=3.):
        raise RuntimeError("规划节点参数读取超时")
    execute, cloud = query.result().values
    if execute.type != ParameterType.PARAMETER_BOOL or execute.bool_value:
        raise RuntimeError("SDK 跟随要求规划节点 execute_trajectory=false")
    node.create_subscription(PointCloud2, cloud.string_value or "/all_points", on_cloud,
                             qos_profile_sensor_data, raw=True)

    def inputs_ready():
        now = time.monotonic()
        return (bridge and now - bridge_at < 1. and now - cloud_at < 1.
                and node.publisher.get_subscription_count() and read_pose(node, tf) is not None)

    if not wait_for(node, inputs_ready, 4.):
        raise RuntimeError("未收到新鲜点云、桥接状态、左臂 TF 或目标订阅尚未就绪")
    check_action_server_count(node)
    if bridge["busy"] or node.count_publishers(TARGET_TOPIC) > 1:
        raise RuntimeError("桥接或 MPC 目标话题已被其他任务占用")

    session, sequence = bridge["session"], bridge["seq"] + 1

    def owns_bridge():
        return (bridge.get("session") == session and bridge.get("seq") == sequence
                and bridge.get("key") == "mpc")

    preparation = goal_request = handle = result = None
    try:
        preparation = prepare.call_async(Trigger.Request())
        if not spin_until(node, preparation, timeout_sec=3.):
            raise RuntimeError("桥接准备请求超时")
        response = preparation.result()
        if not response.success:
            raise RuntimeError(response.message)
        if not wait_for(node, lambda: owns_bridge() and bridge["phase"] in ("mpc_ready", "failed"),
                        8.):
            raise RuntimeError("桥接未进入 mpc_ready")
        if bridge["phase"] == "failed":
            raise RuntimeError(bridge.get("detail") or "桥接准备失败")
        start = read_pose(node, tf)
        if start is None or time.monotonic() - cloud_at >= 1.:
            raise RuntimeError("桥接就绪后，左臂 TF 或点云已过期")
        check_action_server_count(node)
        node.route["waypoints"] = waypoints(start, offset_y)
        node.pose_feedback = lambda: read_pose(node, tf)
        print(f"base_link Y 从 {start.position.y:.4f} 到 "
              f"{start.position.y + offset_y:.4f} m，再返回循环。Ctrl+C 停止。", flush=True)
        node.start_route()
        goal_request = action.send_goal_async(FollowMpc.Goal())
        if not spin_until(node, goal_request, timeout_sec=4.):
            raise RuntimeError("/follow_mpc 未返回受理确认")
        handle = goal_request.result()
        if not handle.accepted:
            raise RuntimeError("/follow_mpc 拒绝目标")
        result = handle.get_result_async()
        while not result.done():
            rclpy.spin_once(node, timeout_sec=0.05)
            check_action_server_count(node)
            if time.monotonic() - cloud_at >= 1.:
                raise RuntimeError("点云断流，停止本次跟随")
            if node.count_publishers(TARGET_TOPIC) > 1:
                raise RuntimeError("检测到另一个 MPC 目标发布者，停止本次跟随")
            if not owns_bridge() or not bridge["busy"] or time.monotonic() - bridge_at >= 1.:
                raise RuntimeError(bridge.get("detail") or "桥接会话已结束或状态中断")
            if time.monotonic() - node.since >= WAYPOINT_TIMEOUT_S:
                raise RuntimeError(f"路点 {node.index + 1} 到位超时，停止跟随")
        outcome = result.result().result
        if not outcome.success:
            raise RuntimeError(outcome.message or "/follow_mpc 已退出")
    finally:
        node.timer.cancel()
        owned = (preparation is not None and spin_until(node, preparation, timeout_sec=2.)
                 and preparation.result().success and wait_for(node, owns_bridge, 1.)
                 and bridge["busy"])
        stop_request = stop.call_async(Trigger.Request()) if owned else None
        if handle is None and goal_request is not None and spin_until(node, goal_request, 2.):
            handle = goal_request.result()
        if handle is not None and handle.accepted and (result is None or not result.done()):
            cancel_goal(node, handle, result)
        if stop_request is not None:
            if not spin_until(node, stop_request, 2.) or not wait_for(
                    node, lambda: not owns_bridge() or not bridge["busy"], 6.):
                print("未收到本次桥接的停止确认", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance-cm", type=float, default=10.,
                        help="沿 base_link 负 Y 的往返距离，单位 cm，默认 10")
    args = parser.parse_args()
    if not math.isfinite(args.distance_cm) or args.distance_cm <= 0.:
        parser.error("--distance-cm 必须为正数")
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node = ArrivalRoutePublisher()
    try:
        run(node, -args.distance_cm / 100.)
    except KeyboardInterrupt:
        return 130
    except RuntimeError as error:
        print(error, file=sys.stderr, flush=True)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
