#!/usr/bin/env bash
# 一键打开普通 MPC：先按路点表发布 /curobo/target，再提交 /follow_mpc。
set -eo pipefail

usage() {
    cat <<'EOF'
用法：./src/curobo_ros_cpp/tests/call_follow_mpc.sh [--waypoints 文件] [--topic 话题]

规划节点需要已经在跑。脚本启动后立刻按路点表循环发布目标，并提交 /follow_mpc。
默认路点：src/curobo_ros_cpp/tests/follow_mpc_waypoints.yaml
每条的 duration_s 是停留秒数，发完最后一条回到第一条。
只写 left 或 right 时，另一侧钉在当前末端。
timeout 和容差为 0 时会一直跟到 Ctrl+C，不会因为到位或超时自己结束。
Ctrl+C 会取消 /follow_mpc，并让规划节点停下控制器。再按一次则立即退出。
是否下发控制器由规划节点的 execute_trajectory 决定。
ROS_DOMAIN_ID 沿用当前终端。可用 CUROBO_PYTHON 指定 Python。
EOF
}

waypoints_override=""
passthrough=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --waypoints)
            if [[ $# -lt 2 ]]; then
                echo "缺少 --waypoints 的文件。" >&2
                exit 2
            fi
            waypoints_override="$2"
            shift 2
            ;;
        *) passthrough+=("$1"); shift ;;
    esac
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir" && pwd)"
while [[ "$repo_root" != "/" && ! -f "$repo_root/install/setup.bash" ]]; do
    repo_root="$(dirname -- "$repo_root")"
done
ros_setup="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
workspace_setup="$repo_root/install/setup.bash"
if [[ ! -f "$ros_setup" || ! -f "$workspace_setup" ]]; then
    echo "找不到 ROS/工程环境，请先构建工作空间并检查 ROS_DISTRO。" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ros_setup"
# shellcheck disable=SC1090
source "$workspace_setup"

python_bin="${CUROBO_PYTHON:-$repo_root/../curobo/.venv/bin/python}"
if [[ -z "${CUROBO_PYTHON:-}" && ! -x "$python_bin" ]]; then
    python_bin="$(command -v python3)"
fi
waypoints="${waypoints_override:-$script_dir/follow_mpc_waypoints.yaml}"
printf 'Action: /follow_mpc | 路点: %s | ROS_DOMAIN_ID: %s\n' "$waypoints" "${ROS_DOMAIN_ID:-0}" >&2
exec "$python_bin" "$script_dir/follow_mpc_client.py" --waypoints "$waypoints" "${passthrough[@]}"
