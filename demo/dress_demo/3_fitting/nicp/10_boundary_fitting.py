import shutil
import os
import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
import numpy as np

def fit(src_obj, trg_obj, out_obj, device):

    verts_src, faces_src, aux = load_obj(src_obj)
    faces_idx_src = faces_src.verts_idx.to(device)
    verts_src = verts_src.to(device)

    # We read the target 3D model using load_obj
    verts, faces, aux = load_obj(trg_obj)

    # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
    # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
    # For this tutorial, normals and textures are ignored.
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)

    # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0). 
    # (scale, center) will be used to bring the predicted mesh to its original center and scale
    # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    verts_src = verts_src - center
    verts_src = verts_src / scale
    
    center_norm = verts.mean(0)
    center_src_norm = verts_src.mean(0)
    verts_src = verts_src + (center_norm - center_src_norm)

    # We construct a Meshes structure for the target mesh
    src_mesh = Meshes(verts=[verts_src], faces=[faces_idx_src]).to(device)
    trg_mesh = Meshes(verts=[verts], faces=[faces_idx])


    deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)


    # Number of optimization steps
    Niter = 2000
    # Weight for the chamfer loss
    w_chamfer = 1.0 
    # Weight for mesh edge loss
    w_edge = 5.0 
    # Weight for mesh normal consistency
    w_normal = 0.01 
    # Weight for mesh laplacian smoothing
    w_laplacian = 0.1
    # Plot period for the losses
    plot_period = 250
    loop = range(Niter)

    chamfer_losses = []
    laplacian_losses = []
    edge_losses = []
    normal_losses = []

    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()
        
        # Deform the mesh
        new_src_mesh = src_mesh.offset_verts(deform_verts)
        
        # We sample 5k points from the surface of each mesh 
        sample_trg = sample_points_from_meshes(trg_mesh, 500)
        sample_src = sample_points_from_meshes(new_src_mesh, 500)
        
        # We compare the two sets of pointclouds by computing (a) the chamfer loss
        loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
        
        # and (b) the edge length of the predicted mesh
        loss_edge = mesh_edge_loss(new_src_mesh)
        
        # mesh normal consistency
        loss_normal = mesh_normal_consistency(new_src_mesh)
        
        # mesh laplacian smoothing
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
        
        # Weighted sum of the losses
        loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
        
        # Print the losses
        # loop.set_description('total_loss = %.6f' % loss)
        
        # Save the losses for plotting
        chamfer_losses.append(float(loss_chamfer.detach().cpu()))
        edge_losses.append(float(loss_edge.detach().cpu()))
        normal_losses.append(float(loss_normal.detach().cpu()))
        laplacian_losses.append(float(loss_laplacian.detach().cpu()))
        
        # Plot mesh
        # if i % plot_period == 0:
        #     plot_pointcloud(new_src_mesh, title="iter: %d" % i)
        if i % 100 == 0:
            print(i, src_obj)
            
        # Optimization step
        loss.backward()
        optimizer.step()

    # Fetch the verts and faces of the final predicted mesh
    final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

    # Scale normalize back to the original target size
    final_verts = final_verts * scale + center

    # Store the predicted mesh using save_obj
    save_obj(out_obj, final_verts, final_faces)

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='NICP')
    parser.add_argument('--input_dir',type=str)
    parser.add_argument('--output_dir',type=str)
    opt = parser.parse_args()

    root = opt.input_dir
    out_root = opt.output_dir
    device = torch.device("cuda:0")
    for name in os.listdir(root):
        out_dir = out_root + name + "/"
        os.makedirs(out_dir, exist_ok=True)
        for key in ["bottom", "right", "left", "top"]:
        # for key in ["left"]:
            src_obj = root + name + "/src_norm/boundary_strip_" + key + ".obj"
            trg_obj = root + name + "/target_norm/pred_boundary_strip_" + key + ".obj"
            out_obj = out_dir + "0_fitted_src_boundary_strip_" +  key + ".obj"
            fit(src_obj, trg_obj, out_obj, device)
            shutil.copyfile(root + name + "/src_norm/lbs_spbs_garment_modified.obj", out_dir + "2_lbs_spbs_garment_modified_norm.obj")
            shutil.copyfile(root + name + "/target_norm/pred_garment.obj", out_dir + "3_pred_garment_norm.obj")
            shutil.copyfile(src_obj, out_dir + "1_src_" + key + ".obj")
            shutil.copyfile(trg_obj, out_dir + "1_trg_" + key + ".obj")

    