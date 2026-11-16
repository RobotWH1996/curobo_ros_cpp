#!/usr/bin/env python3
"""Request pick/place planning and optionally save the full trajectory JSON.

This process only reads perception, fills the Action goal, and waits for the
result. Planning and execution stay in curobo_motion_planning_node.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import rclpy
import yaml
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.set_message import set_message_fields
from curobo_ros_interfaces.action import Pick, Place, PickPlace


def spin_until(node, future, timeout_sec=None):
    """Short waits so Ctrl+C is delivered between messages."""
    deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
    while rclpy.ok() and not future.done():
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0.0:
            return False
        rclpy.spin_once(node, timeout_sec=0.1 if remaining is None else min(0.1, remaining))
    return future.done()


def save_result(result, action_name, output):
    """Export trajectory data for an external consumer; this function never commands motion."""
    document = {
        "schema": "curobo_task_v2",
        "action": action_name,
        "execution_requested": bool(result.execution_requested),
        "execution_completed": bool(result.execution_completed),
        "complete": bool(result.success),
        "joint_units": "URDF: revolute/continuous=rad, prismatic=m",
        "trajectory_time": "trajectory=whole task; trajectories=stage-local; seconds",
        "result": message_to_ordereddict(result),
    }
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def prepare_goal(data, action_name):
    """Select one task from a shared PickPlace document or use a direct Goal."""
    if not isinstance(data, dict):
        raise ValueError("goal-file 必须包含 YAML/JSON 字典")
    data = copy.deepcopy(data)
    if "perception" in data:
        raise ValueError("perception 不是 Action 字段；话题、类别和超时写在命令行，关闭感知用 --no-perception")
    if "execute" in data:
        raise ValueError("execute 字段已移除；是否下发由节点 debug/deploy 配置决定")
    if action_name != "pick_place" and "task" not in data:
        if set(data) - {"pick", "place", "timeout"}:
            raise ValueError("共用配置的顶层只允许 pick、place、timeout")
        if action_name not in data:
            raise ValueError(f"配置缺少 {action_name}，或使用独立请求的 task 字段")
        data = {"task": data[action_name], "timeout": data.get("timeout", 0.0)}
    names = ("pick", "place") if action_name == "pick_place" else ("task",)
    for name in names:
        task = data.get(name)
        if not isinstance(task, dict):
            raise ValueError(f"{name} 必须是任务字典")
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("pick", "place", "pick_place"), default="pick_place")
    parser.add_argument("--goal-file", required=True)
    parser.add_argument("--output", help="Save full Action Result and trajectory metadata to JSON")
    parser.add_argument("--dry-run", action="store_true", help="打印并校验请求，不提交 Action")
    parser.add_argument("--perception", action=argparse.BooleanOptionalAction, default=True,
                        help="pick / pick_place 从感知读取抓取目标；--no-perception 使用文件里的 tool_pose")
    parser.add_argument("--topic", default="/obj_poses", help="感知话题")
    parser.add_argument("--label", type=int, default=11, help="目标类别：5=BOX，11=ROD，12=HOLE")
    parser.add_argument("--track-id", type=int, help="指定目标 ID；不填则选最高置信度")
    parser.add_argument("--min-score", type=float, default=0.0, help="最低置信度，范围 [0, 1]")
    parser.add_argument("--perception-timeout", type=float, default=10.0, help="等待新感知目标的秒数")
    args = parser.parse_args(argv)
    action = {"pick": Pick, "place": Place, "pick_place": PickPlace}[args.action]
    try:
        with open(args.goal_file) as stream:
            document = yaml.safe_load(stream)
        data = prepare_goal(document, args.action)
        perception = {
            "topic": args.topic,
            "label": args.label,
            "track_id": args.track_id,
            "min_score": args.min_score,
            "wait_timeout": args.perception_timeout,
        }
        goal = action.Goal()
        set_message_fields(goal, data)
    except (OSError, ValueError, TypeError, AttributeError, AssertionError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    live_pick = args.perception and args.action != "place"
    if args.dry_run and not live_pick:
        print(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), end="")
        return
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node("pick_place_task_client")
    client = None
    try:
        if live_pick:
            from perception_pick_target import read_pick_target

            task = data["pick"] if args.action == "pick_place" else data["task"]
            selected = read_pick_target(node, task, perception)
            print("感知目标: " + json.dumps(selected, ensure_ascii=False), file=sys.stderr, flush=True)
            goal = action.Goal()
            set_message_fields(goal, data)
        if args.dry_run:
            print(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), end="")
            return
        client = ActionClient(node, action, args.action)
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Action server unavailable")

        def feedback(msg):
            print(msg.feedback.stage, msg.feedback.message, flush=True)

        started = time.perf_counter()
        future = client.send_goal_async(goal, feedback_callback=feedback)
        print(f"已提交 /{args.action}，等待节点", file=sys.stderr, flush=True)
        handle = None
        try:
            if not spin_until(node, future):
                raise RuntimeError("没有收到目标确认")
            handle = future.result()
            if handle is None or not handle.accepted:
                raise RuntimeError("Goal rejected")
            result_future = handle.get_result_async()
            if not spin_until(node, result_future):
                raise RuntimeError("没有收到任务结果")
        except KeyboardInterrupt:
            try:
                if handle is None and spin_until(node, future, timeout_sec=2.0):
                    handle = future.result()
                if handle is not None and handle.accepted:
                    print("正在取消任务", file=sys.stderr, flush=True)
                    if not spin_until(node, handle.cancel_goal_async(), timeout_sec=5.0):
                        print("取消确认超时", file=sys.stderr, flush=True)
                else:
                    print("已中断，未收到可取消的目标句柄", file=sys.stderr, flush=True)
            except KeyboardInterrupt:
                print("再次中断，不再等待取消确认", file=sys.stderr, flush=True)
            raise
        value = result_future.result().result
        elapsed = time.perf_counter() - started
        if args.output:
            save_result(value, args.action, args.output)
        duration = value.trajectory.points[-1].time_from_start if value.trajectory.points else None
        print(json.dumps({
            "success": value.success, "stage": value.stage, "message": value.message,
            "elapsed_s": round(elapsed, 3),
            "execution_requested": value.execution_requested, "execution_completed": value.execution_completed,
            "joint_names": list(value.trajectory.joint_names), "points": len(value.trajectory.points),
            "duration_s": duration.sec + duration.nanosec * 1e-9 if duration else 0.0,
            "trajectories": len(value.trajectories), "trajectory_stages": list(value.trajectory_stages),
            "output": args.output,
        }, ensure_ascii=False))
        if not value.success:
            raise SystemExit(1)
    finally:
        if client is not None:
            client.destroy()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
