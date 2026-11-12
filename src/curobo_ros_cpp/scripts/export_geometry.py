#!/usr/bin/env python3
"""Offline export of the exact retained FCL meshes used by the Python project."""
import argparse
from pathlib import Path
import shutil
import sys

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--python-project',type=Path,required=True)
    parser.add_argument('--curobo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--robot',default='lycus.yml')
    args=parser.parse_args()
    sys.path[:0]=[str(args.curobo),str(args.python_project/'src/curobo_ros_python')]
    import numpy as np
    import yaml
    from curobo.config_io import load_yaml,join_path
    from curobo.content import get_robot_configs_path,get_assets_path
    from curobo_ros_python.planning.mesh_collision import MeshCollisionChecker
    checker=MeshCollisionChecker.from_robot(args.robot,retain_meshes=True)
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for index,(name,mesh) in enumerate(zip(checker.names,checker.meshes)):
        vertices=np.ascontiguousarray(mesh.vertices,dtype='<f8')
        faces=np.ascontiguousarray(mesh.faces,dtype='<i4')
        vertex_file=f'mesh_{index}.f64';face_file=f'mesh_{index}.i32'
        vertices.tofile(args.output/vertex_file);faces.tofile(args.output/face_file)
        rows.append(dict(link=name,vertices=vertex_file,faces=face_file,vertex_count=len(vertices),face_count=len(faces),watertight=bool(mesh.is_watertight)))
    (args.output/'geometry.yaml').write_text(yaml.safe_dump(dict(meshes=rows),sort_keys=False))
    robot_path=Path(join_path(str(get_robot_configs_path()),args.robot))
    kin=load_yaml(str(robot_path))['robot_cfg']['kinematics']
    shutil.copy2(robot_path,args.output/'robot.yml')
    shutil.copy2(join_path(str(get_assets_path()),kin['urdf_path']),args.output/'robot.urdf')
    print('EXPORTED_GEOMETRY',len(rows),flush=True)
if __name__=='__main__':main()
