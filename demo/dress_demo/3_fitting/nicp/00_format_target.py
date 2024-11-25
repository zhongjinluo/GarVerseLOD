import os
import trimesh
import numpy as np
import json

def normalize(mesh_vertices):
    bbox_min = np.min(mesh_vertices, axis=0)
    bbox_max = np.max(mesh_vertices, axis=0)
    center = (bbox_min + bbox_max) / 2
    mesh_vertices -=  center
    r = np.max(np.sqrt(np.sum(np.array(mesh_vertices**2), axis=-1)))
    mesh_vertices /= r
    return mesh_vertices, center, r

def normalize_with_cr(mesh_vertices, c, r):
    mesh_vertices -=  c
    mesh_vertices /= r
    return mesh_vertices

import argparse
parser = argparse.ArgumentParser(description='NICP')
parser.add_argument('--input_dir',type=str)
parser.add_argument('--output_dir',type=str)
opt = parser.parse_args()

root = opt.input_dir + "/"
out_root = opt.output_dir + "/"
os.makedirs(out_root, exist_ok=True)
check_list = os.listdir(out_root)

items = []
for f in os.listdir(root):
    if "_top.obj" in f:
        if os.path.isfile(root+ f[0:-8] + "_right.obj") and \
            os.path.isfile(root + f[0:-8] + "_left.obj") and \
                os.path.isfile(root + f[0:-8] + "_bottom.obj"):
            items.append(f[0:-8])

cnt = 0
for name in items:
    print(name, cnt, len(items))
    cnt += 1
    if name in check_list:
        continue
    out_dir = out_root + name + "/target/"
    os.makedirs(out_dir, exist_ok=True)

    mesh = trimesh.load(root + name + ".obj", process=False)
    meshes = mesh.split()
    mesh = meshes[0]
    for m in meshes:
        if m.vertices.shape[0] > mesh.vertices.shape[0]:
            mesh = m
    mesh.export(out_dir + "pred_garment.obj")

    for key in ["bottom", "right", "left", "top"]:
        mesh = trimesh.load(root + name + ".obj", process=False)
        boundary_mesh = trimesh.load(root + name + "_" + key +".obj", process=False)
        boundary_meshes = boundary_mesh.split()
        boundary_mesh = boundary_meshes[0]
        for m in boundary_meshes:
            if m.vertices.shape[0] > boundary_mesh.vertices.shape[0]:
                boundary_mesh = m
        boundary_mesh.export(out_dir + "pred_boundary_mesh_" + key + ".obj")

        inside = boundary_mesh.contains(mesh.vertices)
        indices = np.where(inside==True)[0]
        label_vis = list(indices)
                
        selected_faces = []
        for face in mesh.faces:
            if face[0] in label_vis and face[1] in label_vis and face[2] in label_vis:
            # if face[0] in label_vis or face[1] in label_vis or face[2] in label_vis:
                selected_faces.append(face)
                
        strip_vertices = []
        strip_faces = []   
        vi_old_to_new_dict = {}
        vi_new_to_old_dict = {}
        for face in selected_faces:
            strip_face = []
            for vi in face:
                if str(vi) not in vi_old_to_new_dict.keys():
                    strip_vertices.append(mesh.vertices[vi])
                    new_vi = len(strip_vertices) - 1
                    strip_face.append(new_vi)
                    vi_old_to_new_dict[str(vi)] = int(new_vi)
                    vi_new_to_old_dict[str(new_vi)] = int(vi)
                else:
                    strip_face.append(vi_old_to_new_dict[str(vi)])
            strip_faces.append(strip_face)
        trimesh.Trimesh(vertices=strip_vertices, faces=strip_faces, process=False).export(out_dir + "pred_boundary_strip_" + key + ".obj")

        data_mapping = {
            "vi_old_to_new_dict": vi_old_to_new_dict,
            "vi_new_to_old_dict": vi_new_to_old_dict
        }
        with open(out_dir + 'pred_indices_mapping_' + key + '.json', 'w') as f:
                json.dump(data_mapping, f)

    out_dir_norm = out_root + name + "/target_norm/"
    os.makedirs(out_dir_norm, exist_ok=True)
    norm_mesh = trimesh.load(out_dir + "pred_garment.obj", process=False)
    norm_mesh.vertices, c, r = normalize(norm_mesh.vertices)
    norm_mesh.export(out_dir_norm + "pred_garment.obj")
    for key in ["bottom", "right", "left", "top"]:
        norm_boundary_mesh = trimesh.load(out_dir + "pred_boundary_mesh_" + key + ".obj", process=False)
        norm_boundary_mesh.vertices = normalize_with_cr(norm_boundary_mesh.vertices, c, r)
        norm_boundary_mesh.export(out_dir_norm + "pred_boundary_mesh_" + key + ".obj")
        norm_boundary_strip = trimesh.load(out_dir + "pred_boundary_strip_" + key + ".obj", process=False)
        norm_boundary_strip.vertices = normalize_with_cr(norm_boundary_strip.vertices, c, r)
        norm_boundary_strip.export(out_dir_norm + "pred_boundary_strip_" + key + ".obj")
    np.savez(out_dir + "cr_info.npz", c=c, r=r)
