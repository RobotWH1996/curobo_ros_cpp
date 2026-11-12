#!/usr/bin/env python3
"""Offline-only export of the qualified Python robot into native deployment data."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--curobo', type=Path, required=True)
    parser.add_argument('--python-project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--robot', default='lycus.yml')
    parser.add_argument('--output-dt', type=float, default=.02)
    parser.add_argument('--opt-interpolation-steps', type=int, default=2)
    parser.add_argument('--max-trajopt-seeds', type=int, default=4)
    parser.add_argument('--max-ik-seeds', type=int, default=32)
    parser.add_argument('--groups', nargs='+', default=['left_arm', 'right_arm', 'both_arms'])
    parser.add_argument('--opt-knots', nargs='+', type=int, default=[16])
    parser.add_argument('--mpc-knots', nargs='+', type=int, default=[6])
    args = parser.parse_args()
    sys.path[:0] = [str(args.curobo), str(args.python_project / 'src/curobo_ros_python')]
    import numpy as np
    import torch
    import yaml
    from curobo.config_io import load_yaml, join_path
    from curobo.content import get_assets_path, get_robot_configs_path
    from curobo.kinematics import KinematicsCfg
    from curobo_ros_python.planning.groups import apply_robot, GROUP_JOINTS
    from curobo_ros_python.planning.motion_collision import configure_motion_collisions
    from curobo_ros_python.planning.payload import configure_payload_links
    from curobo._src.types import spline_layout

    apply_robot(args.robot)
    # Six action controls have a valid cubic layout. The native MPC deployment
    # extends the author's conservative TrajOpt export floor and validates it.
    spline_layout.MIN_TRAJOPT_KNOTS = 6
    exporter_path = args.curobo / 'cpp_runtime/robot/tools/export_robot_aot.py'
    sys.path.insert(0, str(exporter_path.parent))
    spec = importlib.util.spec_from_file_location('native_export', exporter_path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
    from curobo.content import get_task_configs_path
    create_config = TrajOptSolverCfg.create
    def configured_create(*values, **options):
        options['interpolation_dt']=args.output_dt
        if role == 'opt':
            transition = load_yaml(join_path(str(get_task_configs_path()), 'trajopt/transition_bspline_trajopt.yml'))
            transition['transition_model_cfg']['interpolation_steps'] = args.opt_interpolation_steps
            options['transition_model'] = transition
        else:
            transition = load_yaml(join_path(str(get_task_configs_path()), 'trajopt/transition_bspline_trajopt.yml'))
            transition['transition_model_cfg']['interpolation_steps'] = 4
            options['transition_model'] = transition
        return create_config(*values, **options)
    TrajOptSolverCfg.create = staticmethod(configured_create)
    robot_source = Path(join_path(str(get_robot_configs_path()), args.robot))
    source = load_yaml(str(robot_source))
    digest=hashlib.sha256()
    for file in [robot_source, exporter_path, Path(__file__),
                 args.python_project/'src/curobo_ros_python/curobo_ros_python/planning/motion_collision.py',
                 args.python_project/'src/curobo_ros_python/curobo_ros_python/planning/payload.py']:
        digest.update(file.read_bytes())
    digest.update(subprocess.check_output(['git','-C',str(args.curobo),'rev-parse','HEAD']))
    digest.update(subprocess.check_output(['git','-C',str(args.curobo),'diff','--binary','--','curobo']))
    digest.update(str((args.opt_interpolation_steps,args.max_trajopt_seeds,args.max_ik_seeds,args.output_dt)).encode())
    fingerprint=digest.hexdigest()
    for group in args.groups:
        active = list(GROUP_JOINTS[group])
        for role, knots in [('opt', value) for value in args.opt_knots] + [('mpc', value) for value in args.mpc_knots]:
            output = args.output / group / role / str(knots)
            marker = output / 'deployment.json'
            if marker.is_file():
                layout=json.loads((output/'robot_trajopt_manifest.json').read_text())['spline_layout']
                if json.loads(marker.read_text()).get('fingerprint')==fingerprint and layout['interpolation_steps']==(4 if role=='mpc' else args.opt_interpolation_steps):
                    print(f'REUSE {output}', flush=True)
                    continue
            output.mkdir(parents=True, exist_ok=True)
            robot = configure_payload_links(yaml.safe_load(yaml.safe_dump(source)))
            kin = robot['robot_cfg']['kinematics']
            for joint in GROUP_JOINTS['both_arms']:
                if joint not in active:
                    kin.setdefault('lock_joints', {})[joint] = 0.0
            moving = configure_motion_collisions(robot, active)
            robot_file = output / 'robot.runtime.yml'
            robot_file.write_text(yaml.safe_dump(robot, sort_keys=False))
            options = argparse.Namespace(
                robot_config=robot_file, urdf=None, base_link=None, tool_frame=None,
                attachment_parent_link=None, attachment_spheres=32, out=output,
                target_device=None, kinematics_only=False, max_ik_seeds=args.max_ik_seeds,
                ik_seed_samples=2000, max_ik_goalset=1, num_knots=knots,
                max_trajopt_seeds=args.max_trajopt_seeds, interpolation_buffer_size=4000,
                joint_limit_expansion=0.0, rrt_samples_per_iteration=24,
                rrt_max_iterations=32, rrt_max_nodes_per_tree=1024,
                max_cuboids=32, max_capsules=0, max_cylinders=0,
                max_meshes=8, max_voxel_grids=1,
            )
            print(f'EXPORT {group} knots={knots}', flush=True)
            exporter.export_robot(options)
            config = KinematicsCfg.from_robot_yaml_file(str(robot_file))
            params = config.kinematics_config
            urdf = Path(join_path(str(get_assets_path()), kin['urdf_path']))
            joints = ET.parse(urdf).getroot().findall('joint')
            dynamic = []
            for joint in joints:
                name = joint.get('name')
                mimic = joint.find('mimic')
                source_name = mimic.get('joint') if mimic is not None else name
                if source_name not in kin.get('lock_joints', {}):
                    continue
                child = joint.find('child').get('link')
                row = params.link_name_to_idx_map.get(child)
                if row is None:
                    continue
                if int(params.joint_map_type[row]) != -1:
                    raise ValueError(f'Cannot overwrite active transform {name}')
                origin = joint.find('origin')
                axis = joint.find('axis')
                limit = joint.find('limit')
                dynamic.append(dict(
                    name=name, source=source_name, child=child, row=int(row),
                    kind=joint.get('type'),
                    xyz=[float(x) for x in (origin.get('xyz', '0 0 0') if origin is not None else '0 0 0').split()],
                    rpy=[float(x) for x in (origin.get('rpy', '0 0 0') if origin is not None else '0 0 0').split()],
                    axis=[float(x) for x in (axis.get('xyz', '1 0 0') if axis is not None else '1 0 0').split()],
                    multiplier=float(mimic.get('multiplier', '1')) if mimic is not None else 1.0,
                    offset=float(mimic.get('offset', '0')) if mimic is not None else 0.0,
                ))
            metadata = dict(
                schema=1, role=role, fingerprint=fingerprint, group=group, robot=args.robot, urdf=str(urdf),
                active_joints=params.joint_names, moving_links=sorted(moving),
                default_positions=kin.get('lock_joints', {}), dynamic_joints=dynamic,
                fixed_transforms=params.fixed_transforms.detach().cpu().reshape(-1).tolist(),
                position_lower=params.joint_limits.position[0].detach().cpu().tolist(),
                position_upper=params.joint_limits.position[1].detach().cpu().tolist(),
                robot_sha256=hashlib.sha256(robot_source.read_bytes()).hexdigest(),
                curobo_commit=subprocess.check_output(['git', '-C', str(args.curobo), 'rev-parse', 'HEAD']).decode().strip(),
            )
            marker.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
            print(f'EXPORTED {group} knots={knots}', flush=True)
            torch.cuda.synchronize()


if __name__ == '__main__':
    main()
