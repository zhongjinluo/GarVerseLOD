from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree
import os
import numpy as np
import trimesh

import pymeshlab

import argparse
parser = argparse.ArgumentParser(description='NICP')
parser.add_argument('--input_dir',type=str)
opt = parser.parse_args()

root = opt.input_dir
for name in os.listdir(root):
    output_dir = root + name + "/"
    if not os.path.exists(output_dir + '2_final.obj'):
        continue
    if os.path.exists(output_dir + '2_final_refine.obj'):
        continue
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(output_dir + '2_final.obj')
    ms.remeshing_isotropic_explicit_remeshing(targetlen=pymeshlab.Percentage(0.3))
    ms.save_current_mesh(output_dir + '2_final_remesh.obj')

    src = trimesh.load(output_dir + "2_final_remesh.obj", process=False)
    trg = trimesh.load(output_dir + "1_fine.obj", process=False)

    # kd_tree = KDTree(trg.vertices)
    A = trimesh.smoothing.laplacian_calculation(src)
    tree = cKDTree(trg.vertices)
    k = 10
    for i in range(10):
        indices = tree.query(src.vertices, k)[1]
        if k > 1:
            V = np.mean(trg.vertices[indices], axis=1)
        else:
            V = trg[indices]
        src.vertices = A.dot(V)
    src.export(output_dir + "2_final_refine.obj")