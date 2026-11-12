#!/usr/bin/env bash
set -eo pipefail
workspace_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
curobo_source="${CUROBO_SOURCE_DIR:-${HOME}/wh/curobo}"
warp_directory="${WARP_PACKAGE_DIRECTORY:-$curobo_source/.venv/lib/python3.10/site-packages/warp}"
cmake -S "$curobo_source/cpp_runtime/robot" -B "$workspace_dir/build/native" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUROBO_CUDA_ARCH:-87}" \
  -DBUILD_TESTING=OFF -DWARP_NATIVE_INCLUDE_DIR="$warp_directory/native"
cmake --build "$workspace_dir/build/native" -j4
source /opt/ros/humble/setup.bash
cd "$workspace_dir"
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TARGETED_VALIDATION=OFF \
  -DCUROBO_SOURCE="$curobo_source" -DCUROBO_NATIVE_BUILD="$workspace_dir/build/native" \
  -DWARP_LIBRARY="$warp_directory/bin/warp.so" -DCUDAToolkit_ROOT=/usr/local/cuda "$@"
