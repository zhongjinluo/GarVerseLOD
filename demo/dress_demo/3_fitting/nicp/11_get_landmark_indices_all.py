import numpy as np
import os
import trimesh
from scipy.spatial import cKDTree
import json

def get_closest_indices(vertices, points):
    tree = cKDTree(vertices)
    landmarks = tree.query(points, 1)[1]
    return landmarks

import argparse
parser = argparse.ArgumentParser(description='NICP')
parser.add_argument('--input_dir',type=str)
parser.add_argument('--output_dir',type=str)
opt = parser.parse_args()
root = opt.input_dir + "/"
out_root = opt.output_dir + "/"

src_mapping_dir = "../../support_data/nicp_support_data/src_landmark/"
boundary = dict(np.load("../../support_data/nicp_support_data/boundary.npz", allow_pickle=True)["arr_0"][()])
for name in os.listdir(root):
    fitted_boundary_dir = root + name + "/"
    out_dir = out_root + name + "/"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir + "src/", exist_ok=True)
    os.makedirs(out_dir + "target/", exist_ok=True)
    
    all_fitted_boundary_points = []
    all_src_landmarks = []
    for key in ["bottom", "right", "left", "top"]:
        with open(src_mapping_dir + "template_indices_mapping_" + key + ".json") as f:
            mapping_dict = json.load(f)
            vi_old_to_new_dict = mapping_dict["vi_old_to_new_dict"]
            vi_new_to_old_dict = mapping_dict["vi_new_to_old_dict"]

        fitted_boundary_strip_obj = fitted_boundary_dir + "0_fitted_src_boundary_strip_" + key + ".obj"
        strip = trimesh.load(fitted_boundary_strip_obj, process=False)
        strip_landmarks = []
        for vis in boundary[key]:
            strip_landmarks.append(vi_old_to_new_dict[str(vis[0])])
            strip_landmarks.append(vi_old_to_new_dict[str(vis[1])])
        strip_landmarks = list(set(strip_landmarks))
        strip_boundary_points = strip.vertices[strip_landmarks, :]
        
        for v in strip_boundary_points:
            all_fitted_boundary_points.append(v)
        for vi in strip_landmarks:
            all_src_landmarks.append(vi_new_to_old_dict[str(vi)])
            
    src = trimesh.load(fitted_boundary_dir + "2_lbs_spbs_garment_modified_norm.obj", process=False)
    src = src.subdivide()
    src.export(out_dir + "/src/src_mesh.obj")

    all_src_landmarks = get_closest_indices(src.vertices, src.vertices[all_src_landmarks])
    np.savez(out_dir + "/src/src_landmarks.npz", landmarks=all_src_landmarks)


    # trg
    trg = trimesh.load(fitted_boundary_dir + "3_pred_garment_norm.obj", process=False)
    trg = trg.subdivide()
    trg.export(out_dir + "/target/target_mesh.obj")

    all_trg_landmarks = get_closest_indices(trg.vertices, all_fitted_boundary_points)
    np.savez(out_dir + "/target/target_landmarks.npz", landmarks=all_trg_landmarks)


    