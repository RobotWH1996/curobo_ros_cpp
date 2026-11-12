# curobo_ros_cpp

ROS 2 Humble 的 C++ 规划节点。运行时只加载已经导出的 CUDA 内核和几何，用 cuRobo C++/CUDA runtime 做逆解、轨迹优化和 MPC。Python、cuRobo 源码和虚拟环境留在导出机上，目标机携带 `install` 和 `artifacts` 即可。

导出的 CUDA 内核必须和目标 GPU 架构一致，运行时需要 CUDA 12。换 GPU 架构、机器人、控制点数或内部插值后要重新导出。

## 目录

```text
curobo_ros_cpp/
├── src/curobo_ros_interfaces/          # 与 Python 工程共用的消息、服务、Action
│   ├── srv/  PlanJointGoal PlanPoseGoal ComputeIK
│   ├── action/  Pick Place PickPlace FollowMpc
│   └── msg/  PickTask PlaceTask MotionWaypoint MpcTarget MpcPlan
├── src/curobo_ros_cpp/
│   ├── include/curobo_ros_cpp/          # 安装给调用方的声明
│   │   └── detail/                     # 包内实现头文件，不随安装分发
│   ├── src/                            # C++ 实现文件
│   │   ├── main.cpp                    # 进程入口
│   │   ├── node.cpp                    # 参数、订阅、服务和 Action 工作线程
│   │   ├── native_planner.cpp          # 初始化、规划组加载和关节同步
│   │   ├── native_planner_world.cpp    # 场景、点云障碍和携物几何
│   │   ├── native_planner_ik.cpp       # FK、雅可比和逆解
│   │   ├── native_planner_trajopt.cpp  # OPT、RRT/PRM 回退和轨迹验收
│   │   ├── native_planner_contact.cpp  # 接触段直线插补
│   │   ├── native_planner_mpc.cpp      # 原生 MPC 求解与热启动状态
│   │   ├── manipulation.cpp            # 抓放分段、合并、分块提交
│   │   ├── mpc_follow.cpp              # /follow_mpc 周期求解
│   │   ├── executor.cpp                # FollowJointTrajectory；deploy 下不启用
│   │   ├── robot_model.cpp             # URDF、FCL 网格、自碰撞
│   │   ├── pointcloud.cpp              # 裁剪、本体过滤
│   │   └── trajectory.cpp              # 拼接、保持、时间缩放
│   ├── config/
│   │   ├── debug.yaml                  # execute_trajectory: true，可视化开
│   │   ├── deploy.yaml                 # execute_trajectory: false，只发话题
│   │   ├── robots/<robot>.yaml         # 关节、末端、夹爪、控制器名
│   │   └── <robot>_fcl_collision.yaml  # 允许碰撞的连杆对
│   ├── launch/curobo_motion_planning.launch.xml
│   ├── tests/                         # 离线回放、假控制器验证和调用脚本
│   └── scripts/
│       ├── build.sh                    # 编 native runtime，再 colcon
│       ├── export_artifacts.py         # 离线导出 OPT / MPC 内核
│       ├── export_geometry.py          # 离线导出 FCL 几何
│       └── export_scene.py             # 把静态网格烘焙进 YAML
└── artifacts/                          # 离线导出的运行时产物，不参与编译
    ├── geometry/<robot>/               # URDF、碰撞网格
    ├── <group>/opt/<knots>/            # 轨迹优化内核，现为 16 控制点
    ├── <group>/mpc/<knots>/            # MPC 内核，现为 6 控制点、内部插值 4
    └── scenes/                         # 与 scene_model 同名的烘焙场景
```

`<group>` 是 `left_arm`、`right_arm`、`both_arms`。旧布局 `<group>/<knots>` 仍能读；新导出写到 `opt` 和 `mpc` 子目录。

规划入口集中在 `include/curobo_ros_cpp/native_planner.hpp`。`detail/native_planner_group.hpp` 保存各规划实现共享的求解器实例与显存布局。轨迹工具、点云处理、CPU 碰撞模型和控制器执行各有对应头文件。头文件注释描述输入、单位、输出和状态约定，每个实现文件开头概述其职责。

## cuRobo fast

求解器来自 cuRoboV2-Fast 的 `fast` 分支，路径由 `CUROBO_SOURCE_DIR` 指定。导出机用 Python 虚拟环境生成机器人相关的 AOT 产物。运行时加载这些产物和 `libcurobo_robot_runtime.so`。目标机不需要这份源码。

```text
<curobo>/                                  # CUROBO_SOURCE_DIR
├── cpp_runtime/robot                      # C++17 / CUDA 运行时，build.sh 编这一层
│   ├── Kinematics                         # 正运动学和雅可比
│   ├── InverseKinematics                  # 多种子、批量逆解
│   ├── MotionPlanner                      # 关节 / 位姿轨迹优化，失败后再接 RRT
│   ├── MpcController                      # 笛卡尔 MPC
│   ├── Scene                              # 固定容量的立方体、网格、体素
│   └── 本工程依赖的 fast 接口
│       ├── update_fixed_transforms        # 躯干、头、被锁的另一只臂随反馈更新，不重建求解器
│       ├── set_tool_upright               # keep_tool_z_vertical：运动中的工具 Z 轴偏好竖直
│       └── set_joint_tracking             # MPC 在末端零空间里跟踪构型，l 或 zero
├── .venv/.../warp                         # mesh 碰撞后端；头文件给编译，warp.so 打进运行库
└── cpp_runtime/robot/tools/export_robot_aot.py
                                           # export_artifacts.py 调用它，按目标 GPU 出内核
```

`update_fixed_transforms` 会写进运动学、逆解、轨迹优化和 MPC，夹爪与被锁关节变了以后，下一次求解用的是新几何。`set_tool_upright` 只加在正在动的那只工具上，权重来自 `keep_tool_z_vertical`。`set_joint_tracking` 用末端雅可比的零空间把关节往 `mpc_posture` 收，不把末端从目标上拉开。

这三个接口必须在这份 fast 源码里。换提交、机器人模型、控制点数、内部插值或 `CUROBO_CUDA_ARCH` 之后，要重新导出 `artifacts`，并重新跑 `build.sh`。`CUROBO_CUDA_ARCH` 未设置时脚本使用 87。

## 构建

编译机需要 ROS 2 Humble、Eigen、yaml-cpp、FCL、CUDA 12，以及上面的 cuRobo fast 源码和 Warp。

```bash
cd <curobo_ros_cpp>
bash src/curobo_ros_cpp/scripts/build.sh
source install/setup.bash
```

`build.sh` 用 `CUROBO_SOURCE_DIR` 找 fast 源码，用 `CUROBO_CUDA_ARCH` 编译 `libcurobo_robot_runtime.so`，再 colcon 安装本包。安装时把 runtime、`warp.so` 和 `artifacts` 拷进 `install`。

目标机运行时还要有 ROS 2 Humble、`libyaml-cpp`、`libfcl`、`libcudart.so.12` 和匹配的 NVIDIA 驱动。不需要 cuRobo 源码，也不需要 Python 虚拟环境。

## 启动

```bash
source <curobo_ros_cpp>/install/setup.bash
export ROS_DOMAIN_ID=<与机器人网络一致>
ros2 launch curobo_ros_cpp curobo_motion_planning.launch.xml mode:=deploy
```

`mode` 选择 `share/curobo_ros_cpp/config/<mode>.yaml`，默认 `debug`。产物默认从已安装包的 `share/curobo_ros_cpp/artifacts` 加载。也可在启动时用 `artifacts:=<目录>` 指定其他产物。

```text
启动
├── 读 debug.yaml 或 deploy.yaml
├── FCL
│   └── 为导出的碰撞网格建 BVH          # 网格多时占启动的大头
├── OPT
│   ├── 只构造 trajopt_prewarm_groups   # 默认 left_arm；右臂和双臂第一次请求才加载
│   └── 空场景上跑 warmup_iterations 次  # 捕获 CUDA graph
├── 接触 IK
│   └── 预热 pick_place_ik_prewarm_samples  # 默认 33 点一批
└── MPC                                  # mpc_prewarm 默认 false，跟随时才建
```

日志出现「可以接收任务」时，节点已完成初始化。

`debug.yaml` 把 `execute_trajectory` 设为 true，规划后向仿真控制器提交 `FollowJointTrajectory`。`deploy.yaml` 设为 false，只把轨迹发到话题，交给外部控制层。

## 一次抓取怎么规划

位姿目标默认最多 3 次。第 1 次不开图搜索。

```text
预抓取 / 结束段（OPT）
├── 逆解                                   # 位姿目标：32 个种子，从当前关节出发
│   └── FCL 丢掉碰着的终点                 # 保留运动臂与另一臂、躯干的自碰撞
├── 选 2 条种子                            # trajopt_num_seeds
├── 关节空间直线铺成 16 点 B 样条           # 内部再插 trajopt_interpolation_steps=2
├── L-BFGS 50 步                           # 末端误差、平滑、球体对点云网格的碰撞
│   └── keep_tool_z_vertical               # 中间姿态偏好工具 Z 轴竖直
├── 终点再做一次 FCL
└── 按 interpolation_dt 采样，再按 velocity_scale 拉长时间
    └── 失败且图搜索是 rrt                 # 第 2、3 次才允许双向 RRT，给样条换初值
```

结束段的目标已经是关节角，跳过逆解，直接从当前关节优化到目标关节。接触、抬升、撤退不走这棵树，它们是无碰撞查询的笛卡尔插补，但仍检查关节限位、速度限位、直线、姿态和连续性。

`attempts=1` 且 `graph_ms=0` 表示第一次优化已经通过，没有调用 RRT。启动时的 `warmup_iterations` 是空跑，和这里的尝试次数无关。

单独抓取或放置通常分成两块，组合抓放可分成三块。后续块会续接已提交的轨迹。启用执行时，抓放 Action 等控制器结果后结束，整任务超时包括排队、规划和执行等待。任务失败、超时或取消时，节点会请求取消仍在执行的控制器目标。同步 GPU 求解返回后才能响应中断。

```text
/pick
├── 点云冻结一帧                           # 之后结束段只在这帧上抠盒子，不重新读传感器
├── 块 1
│   ├── prepick    OPT，夹爪按张开几何规划
│   ├── pick       直线插补，无碰撞查询
│   ├── close      夹爪行程按 URDF 速度限位计时，再停 gripper_wait_time
│   └── 提交                               # 张开运动叠加在手臂轨迹上
└── 块 2
    ├── 挂上携物                           # 27 个球盖住包围盒，FCL 用包围盒
    ├── lift       直线插补
    ├── end        OPT；点云扣掉携物盒子
    └── 提交
```

`/place` 同样两段：接近和放置用 OPT 与接触插补，释放后把物体留在场景里，撤退沿用接触段规则。`gripper_duration` 仍留在消息里，节点不使用它。

点云过滤由 `self_filter_mode` 控制。`off` 跳过本体判断，`observe` 生成过滤诊断，`apply` 将过滤结果用于规划。单臂组只判断运动臂的本体点。同一体素包含本体点和环境点时，优先保留环境点作为代表。

## 接口

名字和关节顺序来自 `config/robots/<robot>.yaml`。夹爪命令用 URDF 关节单位，手臂和夹爪各走自己的 `FollowJointTrajectory`。

```text
输入
├── /joint_states                          # joint_states_topic
├── <pointcloud_topic>                     # 默认 /all_points；TF 到 base_frame
├── /plan_joint_goal  /plan_pose_goal      # 服务
├── /compute_ik
├── /pick  /place  /pick_place  /follow_mpc # Action
└── /curobo/target                         # MpcTarget，MPC 期间

输出
├── /curobo/planned_trajectory             # 规划出的关节轨迹
├── /display_planned_path                  # debug 下的 MoveIt 预览
├── /curobo/goal_waypoints
├── /curobo/collision_world
└── /curobo/mpc_plan                       # FollowMpc 的周期结果
```

跟随期间持续发送 `MpcTarget` 与运动关节反馈。`mpc_target_max_age` 和 `mpc_joint_state_max_age` 默认都是 1 秒，以本机单调接收时间判断。目标或关节反馈过期时，Action 中止并请求取消控制器目标。`/curobo/mpc_plan` 的成功结果携带交接起点，轨迹点的 `time_from_start` 相对该起点计时。

抓取客户端用法如下。

```bash
./src/curobo_ros_cpp/tests/call_pick_place.sh pick --label <id>
```

脚本只发 Action。下不下控制器由节点的 `execute_trajectory` 决定。

## 回归验证

完成常规构建后，可打开现有的离线与假控制器验证程序。以下命令从工作区根目录运行。

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon build --packages-select curobo_ros_cpp --cmake-args -DBUILD_TARGETED_VALIDATION=ON
source install/setup.bash
export ROS_LOCALHOST_ONLY=1
ROS_DOMAIN_ID=175 build/curobo_ros_cpp/validate_executor
ROS_DOMAIN_ID=176 build/curobo_ros_cpp/validate_snapshot \
  --ros-args --params-file /tmp/curobo_cpp_20261112/case/parameters.yaml \
  -p execute_trajectory:=false
```

`validate_executor` 自行启动假控制器，验证分块续接、取消和确认超时。`validate_snapshot` 使用已录制的关节、TF 和点云，覆盖碰撞、抓放、规划服务和 MPC。回放依赖 `ai` 上已有的 `/tmp/curobo_cpp_20261112/case/` 样本及本工作区的 AOT 产物。

## 离线导出

在带 Python fast 环境的机器上执行。产物必须和运行机的 GPU 架构一致。

```bash
source <curobo>/.venv/bin/activate
python src/curobo_ros_cpp/scripts/export_artifacts.py \
  --curobo <curobo> \
  --python-project <curobo_ros_python> \
  --output artifacts --robot <robot>.yml
python src/curobo_ros_cpp/scripts/export_geometry.py \
  --curobo <curobo> \
  --python-project <curobo_ros_python> \
  --output artifacts/geometry/<robot> --robot <robot>.yml
```

`<robot>.yml` 是 cuRobo 的机器人配置，`<robot>` 与 `config/robots/<robot>.yaml` 对应。脚本默认 OPT 为 16 个控制点、内部插值 2 步，MPC 为 6 个控制点、内部插值 4 步，并为 `left_arm`、`right_arm`、`both_arms` 各导一份。不同机器人使用各自的 `artifacts` 目录。

静态场景里如果有文件网格或其他非盒子几何，先烘焙再放到 `artifacts/scenes`，文件名与 `scene_model` 相同：

```bash
python src/curobo_ros_cpp/scripts/export_scene.py <scene>.yaml artifacts/scenes/<scene>.yaml
```
