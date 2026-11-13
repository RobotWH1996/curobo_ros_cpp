"""C++ planner with robot TF publishing, matching the Python deployment launch."""
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import yaml


def _settings(context):
    share = Path(get_package_share_directory('curobo_ros_cpp'))
    mode = LaunchConfiguration('mode').perform(context)
    config = Path(mode) if os.path.isabs(mode) else share/'config'/(
        mode if Path(mode).suffix else mode+'.yaml')
    if not config.is_file():
        raise RuntimeError(f'规划配置不存在：{config}')
    settings = yaml.safe_load(config.read_text())['curobo_motion_planning_node']['ros__parameters']
    artifacts = Path(LaunchConfiguration('artifacts').perform(context)).expanduser()
    urdf = Path(LaunchConfiguration('robot_urdf').perform(context)).expanduser()
    if not urdf.is_file():
        raise RuntimeError(f'TF URDF 不存在：{urdf}')
    return config,settings,artifacts,urdf


def _launch_setup(context, *args, **kwargs):
    config,settings,artifacts,urdf_path = _settings(context)
    urdf = urdf_path.read_text()
    nodes = []
    if LaunchConfiguration('start_planner').perform(context).lower() == 'true':
        nodes.append(Node(
            package='curobo_ros_cpp',executable='curobo_motion_planning_node',
            name='curobo_motion_planning_node',output='screen',
            parameters=[str(config),{'artifacts_directory':str(artifacts)}],
            arguments=['--ros-args','--log-level',
                       f"curobo_motion_planning_node:={LaunchConfiguration('log_level').perform(context)}"],
        ))

    publish_tf = LaunchConfiguration('publish_robot_tf').perform(context).lower()
    publish_camera = LaunchConfiguration('publish_camera_tf').perform(context).lower()
    if publish_tf not in {'true','false'} or publish_camera not in {'true','false'}:
        raise RuntimeError('publish_robot_tf / publish_camera_tf 必须为 true 或 false')
    if publish_tf == 'true':
        nodes.append(Node(
            package='robot_state_publisher',executable='robot_state_publisher',
            name='robot_state_publisher',output='screen',
            parameters=[{'robot_description':ParameterValue(urdf,value_type=str),
                         'use_sim_time':settings.get('use_sim_time',False)}],
            remappings=[('joint_states',settings.get('joint_states_topic','/joint_states'))],
        ))
    if publish_camera == 'true':
        parent = LaunchConfiguration('camera_parent_frame').perform(context).lstrip('/')
        child = LaunchConfiguration('camera_frame').perform(context).lstrip('/')
        links = {link.attrib['name'] for link in ET.fromstring(urdf).findall('link')}
        if parent not in links or not child or child in links:
            raise RuntimeError('相机静态 TF 父系须存在于 URDF，子系须不在 URDF 中，避免重复发布')
        xyz = LaunchConfiguration('camera_xyz').perform(context).split()
        rpy = LaunchConfiguration('camera_rpy').perform(context).split()
        if any(len(v) != 3 for v in (xyz,rpy)) or not all(
                math.isfinite(float(v)) for v in xyz+rpy):
            raise RuntimeError('camera_xyz / camera_rpy 各须提供三个有限数，单位 m / rad')
        nodes.append(Node(
            package='tf2_ros',executable='static_transform_publisher',
            name='camera_optical_static_tf',output='screen',
            arguments=['--x',xyz[0],'--y',xyz[1],'--z',xyz[2],
                       '--roll',rpy[0],'--pitch',rpy[1],'--yaw',rpy[2],
                       '--frame-id',parent,'--child-frame-id',child],
        ))
    return nodes


def generate_launch_description():
    share = get_package_share_directory('curobo_ros_cpp')
    return LaunchDescription([
        DeclareLaunchArgument('mode',default_value='deploy',
                              description='share/curobo_ros_cpp/config 下的 yaml，或绝对路径'),
        DeclareLaunchArgument('artifacts',default_value=os.path.join(share,'artifacts'),
                              description='导出内核、几何和 URDF 的根目录'),
        DeclareLaunchArgument('start_planner',default_value='true',
                              description='是否启动 C++ 规划节点'),
        DeclareLaunchArgument('log_level',default_value='info',
                              choices=['debug','info','warn','error','fatal']),
        DeclareLaunchArgument('publish_robot_tf',default_value='true',
                              description='根据 /joint_states 发布机器人动态 TF'),
        DeclareLaunchArgument(
            'robot_urdf',
            default_value=os.path.join(share,'artifacts','geometry','lycus','robot.urdf'),
            description='robot_state_publisher 使用的 URDF'),
        DeclareLaunchArgument('publish_camera_tf',default_value='false',
                              description='发布相机父系到光学系的静态外参'),
        DeclareLaunchArgument('camera_parent_frame',default_value='camera_01_link'),
        DeclareLaunchArgument('camera_frame',default_value='camera_01_color_optical_frame'),
        DeclareLaunchArgument('camera_xyz',default_value='0 0 0',
                              description='光学系在父系中的平移，m；启用前核对标定值'),
        DeclareLaunchArgument(
            'camera_rpy',default_value='-1.5707963267948966 0 -1.5707963267948966',
            description='光学系在父系中的 roll pitch yaw，rad'),
        OpaqueFunction(function=_launch_setup),
    ])
