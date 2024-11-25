import os
import trimesh
import numpy as np
import json
import shutil

def normalize_with_cr(mesh_vertices, c, r):
    mesh_vertices -=  c
    mesh_vertices /= r
    return mesh_vertices

import argparse
parser = argparse.ArgumentParser(description='NICP')
parser.add_argument('--input_dir',type=str)
parser.add_argument('--output_dir',type=str)
opt = parser.parse_args()

root = opt.input_dir
out_root = opt.output_dir

for name in os.listdir(out_root):
    out_dir = out_root + name + "/src/"
    os.makedirs(out_dir, exist_ok=True)

    mesh = trimesh.load(root + name + "_lbs_spbs_garment_modified.obj", process=False)
    mesh.export(out_dir + "lbs_spbs_garment_modified.obj")

    for key in ["bottom", "right", "left", "top"]:
        # print(key)
        mesh = trimesh.load(root + name + "_lbs_spbs_garment_modified.obj", process=False)
        with open("../../support_data/nicp_support_data/template_boundary_triangle_strips/mapping/template_indices_mapping_" + key + ".json") as f:
            mapping_dict = json.load(f)
            old_indices = mapping_dict["old_vertex_indices"]
            strip_faces = mapping_dict["strip_faces"]
            boundary_vertices = mesh.vertices[old_indices]
            boundary_strip = trimesh.Trimesh(vertices=boundary_vertices, faces=strip_faces, process=False)
            boundary_strip.export(out_dir + "boundary_strip_" + key + ".obj")
        shutil.copyfile("../../support_data/nicp_support_data/template_boundary_triangle_strips/mapping/template_indices_mapping_" + key + ".json", out_dir + "template_indices_mapping_" + key + ".json")
    out_dir_norm = out_root + name + "/src_norm/"
    os.makedirs(out_dir_norm, exist_ok=True)
    cr = np.load(out_root + name + "/target/cr_info.npz")
    c = cr["c"]
    r = cr["r"]
    for f in os.listdir(out_dir):
        if ".obj" in f:
            norm_mesh = trimesh.load(out_dir + f, process=False)
            norm_mesh.vertices = normalize_with_cr(norm_mesh.vertices, c, r)
            norm_mesh.export(out_dir_norm + f)
    print(name)