from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree
import os
import numpy as np
import trimesh

import pymeshlab

ms = pymeshlab.MeshSet()
ms.load_new_mesh('checks/final.obj')
ms.remeshing_isotropic_explicit_remeshing(targetlen=pymeshlab.Percentage(0.3))
ms.save_current_mesh('checks/final_remesh.obj')

src = trimesh.load("checks/final_remesh.obj", process=False)
trg = trimesh.load("checks/trg.obj", process=False)

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
src.export("checks/final_refine.obj")