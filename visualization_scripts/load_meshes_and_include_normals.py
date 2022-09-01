""" 
    Reads plys defined with pattern in scans_path_pattern. First computes 
    normals for each scan using open3d, then outputs each scan with normal
    information visualized as vertex colors. 

    Example command:
        python ./visualization_scripts/load_meshes_and_include_normals.py \
            --input_path simple_recon_output/HERO_MODEL/scannet/default/meshes/0.04_3.0_open3d_color/ \
            --output_path simple_recon_output/HERO_MODEL/scannet/default/meshes/0.04_3.0_open3d_color_normals/;
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

parser = argparse.ArgumentParser(description='mesh normal visualizer.')

parser.add_argument('--scannet_scans_path_pattern', required=False, default=None,
                        help="Input example string pattern for one scan file. "
                        "For ScanNet it should look something like path_to_scans/SCAN_NAME.ply."
                        "SCAN_NAME will be replaced with each scan's name.")
parser.add_argument('--input_path', required=False, default=None)
parser.add_argument('--output_path', required=True)

args = parser.parse_args()

Path(args.output_path).mkdir(exist_ok=True, parents=True)


if args.scannet_scans_path_pattern:
    scans = ['scene0707_00', 'scene0708_00', 'scene0709_00', 'scene0710_00', 
            'scene0711_00', 'scene0712_00', 'scene0713_00', 'scene0714_00', 
            'scene0715_00', 'scene0716_00', 'scene0717_00', 'scene0718_00', 
            'scene0719_00', 'scene0720_00', 'scene0721_00', 'scene0722_00', 
            'scene0723_00', 'scene0724_00', 'scene0725_00', 'scene0726_00', 
            'scene0727_00', 'scene0728_00', 'scene0729_00', 'scene0730_00', 
            'scene0731_00', 'scene0732_00', 'scene0733_00', 'scene0734_00', 
            'scene0735_00', 'scene0736_00', 'scene0737_00', 'scene0738_00', 
            'scene0739_00', 'scene0740_00', 'scene0741_00', 'scene0742_00', 
            'scene0743_00', 'scene0744_00', 'scene0745_00', 'scene0746_00', 
            'scene0747_00', 'scene0748_00', 'scene0749_00', 'scene0750_00', 
            'scene0751_00', 'scene0752_00', 'scene0753_00', 'scene0754_00', 
            'scene0755_00', 'scene0756_00', 'scene0757_00', 'scene0758_00', 
            'scene0759_00', 'scene0760_00', 'scene0761_00', 'scene0762_00', 
            'scene0763_00', 'scene0764_00', 'scene0765_00', 'scene0766_00', 
            'scene0767_00', 'scene0768_00', 'scene0769_00', 'scene0770_00', 
            'scene0771_00', 'scene0772_00', 'scene0773_00', 'scene0774_00', 
            'scene0775_00', 'scene0776_00', 'scene0777_00', 'scene0778_00', 
            'scene0779_00', 'scene0780_00', 'scene0781_00', 'scene0782_00', 
            'scene0783_00', 'scene0784_00', 'scene0785_00', 'scene0786_00', 
            'scene0787_00', 'scene0788_00', 'scene0789_00', 'scene0790_00', 
            'scene0791_00', 'scene0792_00', 'scene0793_00', 'scene0794_00', 
            'scene0795_00', 'scene0796_00', 'scene0797_00', 'scene0798_00', 
            'scene0799_00', 'scene0800_00', 'scene0801_00', 'scene0802_00', 
            'scene0803_00', 'scene0804_00', 'scene0805_00', 'scene0806_00']

    mesh_paths = [args.scannet_scans_path_pattern.replace("SCAN_NAME", scan) 
                                                            for scan in scans]

elif args.input_path:
    os.chdir(args.input_path)
    mesh_paths = glob.glob("*.ply")

else:
    raise Exception("No valid input path found.")


for path_to_mesh in tqdm(mesh_paths):
    # read mesh
    mesh = o3d.io.read_triangle_mesh(path_to_mesh)
    
    # compute normals and include them as RGB information
    mesh.compute_vertex_normals()
    mesh.vertex_colors = o3d.cuda.pybind.utility.Vector3dVector(
                                    ((1+np.asarray(mesh.vertex_normals))*0.5))

    # write to disk
    mesh_name = path_to_mesh.split("/")[-1].split(".")[0]
    output_path = os.path.join(args.output_path, f"{mesh_name}.ply")

    o3d.io.write_triangle_mesh(output_path, mesh)
