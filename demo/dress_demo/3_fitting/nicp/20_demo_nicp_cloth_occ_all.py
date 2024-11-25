# Copyright 2021 by Haozhe Wu, Tsinghua University, Department of Computer Science and Technology.
# All rights reserved.
# This file is part of the pytorch-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import torch
import io3d
import render
import numpy as np
import json
from utils import normalize_mesh, normalize_pcl
from landmark import get_mesh_landmark
from cloth_model import load_bfm_model
from nicp import non_rigid_icp_mesh2pcl, non_rigid_icp_mesh2mesh
import os
import argparse
parser = argparse.ArgumentParser(description='NICP')
parser.add_argument('--input_dir',type=str)
parser.add_argument('--output_dir',type=str)
opt = parser.parse_args()

root = opt.input_dir + "/"
opt.output_dir = opt.output_dir + "/"
for name in os.listdir(root):
    print(name)
    data_dir = opt.input_dir + name + "/"
    output_dir = opt.output_dir + name + "/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir + "/2_final.obj"


    # demo for registering mesh
    # estimate landmark for target meshes
    # the face must face toward z axis
    # the mesh or point cloud must be normalized with normalize_mesh/normalize_pcl function before feed into the nicp process
    device = torch.device('cuda:0')
    meshes = io3d.load_obj_as_mesh(os.path.join(data_dir,'target/target_mesh.obj'), device = device)

    with torch.no_grad():
        norm_meshes, norm_param = normalize_mesh(meshes)
        dummy_render = render.create_dummy_render([1, 0, 0], device = device)
        
        # target_lm_index, lm_mask = get_mesh_landmark(norm_meshes, dummy_render)
        target_lm_index =np.load(os.path.join(data_dir,"target/target_landmarks.npz"))["landmarks"]
        target_lm_index = np.asarray(target_lm_index)
        target_lm_index = target_lm_index[None,...]
        lm_mask = target_lm_index > -1
        target_lm_index = torch.from_numpy(target_lm_index).to(device).long()
        lm_mask = torch.from_numpy(lm_mask).to(device).bool()
        
        bfm_meshes, bfm_lm_index = load_bfm_model(torch.device('cuda:0'),data_dir)


        lm_mask = torch.all(lm_mask, dim = 0)
        bfm_lm_index_m = bfm_lm_index[:, lm_mask]
        target_lm_index_m = target_lm_index[:, lm_mask]
        
    io3d.save_meshes_as_objs([output_dir + "/0_coarse.obj"], bfm_meshes, save_textures = False)
    io3d.save_meshes_as_objs([output_dir + "/1_fine.obj"], norm_meshes, save_textures = False)

    fine_config = json.load(open('config/cloth.json'))
    registered_mesh = non_rigid_icp_mesh2mesh(bfm_meshes, norm_meshes, bfm_lm_index_m, target_lm_index_m, fine_config)
    io3d.save_meshes_as_objs([output_path], registered_mesh, save_textures = False)