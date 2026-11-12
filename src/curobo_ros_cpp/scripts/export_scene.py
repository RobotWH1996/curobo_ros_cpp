#!/usr/bin/env python3
"""Bake mesh and non-box scene geometry into portable deployment YAML offline."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    from curobo.scene import Scene
    import numpy as np
    scene=Scene.create(yaml.safe_load(args.input.read_text()))
    boxes={box.name:dict(dims=box.dims,pose=box.pose) for box in scene.cuboid or []}
    scene.cuboid=[]
    meshes=scene.get_mesh_world().mesh
    output=dict(cuboid=boxes,mesh={})
    for mesh in meshes or []:
        vertices,faces=mesh.get_mesh_data()
        output['mesh'][mesh.name]=dict(pose=list(mesh.pose),vertices=[[float(v) for v in row] for row in vertices],faces=[[int(v) for v in row] for row in np.asarray(faces).reshape(-1,3)])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(yaml.safe_dump(output,sort_keys=False))

if __name__=='__main__':
    main()
