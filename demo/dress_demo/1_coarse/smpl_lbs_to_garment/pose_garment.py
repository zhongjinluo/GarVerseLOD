import os
from smpl.smpl_numpy import SMPL
import trimesh
import numpy as np
from scipy.spatial import cKDTree
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
import torch

from scipy.linalg import expm, norm
from numpy import cross, eye, dot
import argparse
import cv2
import shutil

def M(axis, theta):
    return expm(cross(eye(3), axis/norm(axis)*theta))


from pytorch3d.ops.knn import knn_points

from collections import namedtuple

def sample_blend_closest_points(src: torch.Tensor, ref: torch.Tensor, values: torch.Tensor, K: int = 5, exp: float = 1e-8):
    # not so useful to aggregate all K points
    n_batch, n_points, _ = src.shape
    ret = guard_knn_points(src, ref, K=K)
    dists, vert_ids = ret.dists, ret.idx  # (n_batch, n_points, K)
    values = values.view(-1, values.shape[-1])  # (n, D)
    # sampled = values[vert_ids]  # (n_batch, n_points, K, D)
    disp = 1 / (dists + exp)  # inverse distance: disparity
    weights = disp / disp.sum(dim=-1, keepdim=True)  # normalize distance by K
    dists = torch.einsum('ijk,ijk->ij', dists, weights)
    # sampled *= weights[..., None]  # augment weight in last dim for bones # written separatedly to avoid OOM
    # sampled = sampled.sum(dim=-2)  # sum over second to last for weighted bw
    sampled = torch.einsum('ijkl,ijk->ijl', values[vert_ids], weights)
    return sampled.view(n_batch, n_points, -1), dists.view(n_batch, n_points, 1)

def get_garment_on_spbs_body(mean_body, spbs_body, garment_on_mean_body):
    tree = cKDTree(mean_body.vertices)
    _, t_ii = tree.query(garment_on_mean_body.vertices, k=1)
    offsets = spbs_body.vertices - mean_body.vertices
    garment_vertices_on_spbs_body = garment_on_mean_body.vertices + offsets[t_ii]
    return garment_vertices_on_spbs_body, t_ii

def tpose_points_to_pose_points(pts, bw, A):
    """transform points from the T pose to the pose space
    ppts: n_batch, n_points, 3
    bw: n_batch, 24, n_points
    A: n_batch, 24, 4, 4
    """
    sh = pts.shape
    bw = bw.permute(0, 2, 1)
    A = torch.bmm(bw, A.view(sh[0], 24, -1))
    A = A.view(sh[0], -1, 4, 4)
    R = A[..., :3, :3]
    pts = torch.sum(R * pts[:, :, None], dim=3)
    pts = pts + A[..., :3, 3]
    return pts

def guard_knn_points(src, ref, K):
    ret = knn_points(src, ref, K=K)
    return namedtuple('ret', ['dists', 'idx'])(dists=ret.dists.sqrt(), idx=ret.idx)

def batch_rodrigues(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    rot_dir = poses / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros([batch_size, 1])
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
                       axis=1)
    K = K.reshape([batch_size, 3, 3])

    ident = np.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)

    return rot_mat


def get_rigid_transformation(poses, joints, parents, return_joints=False):
    """
    poses: 24 x 3
    joints: 24 x 3
    parents: 24
    """
    rot_mats = batch_rodrigues(poses)

    # obtain the relative joints
    rel_joints = joints.copy()
    rel_joints[1:] -= joints[parents[1:]]

    # create the transformation matrix
    transforms_mat = np.concatenate([rot_mats, rel_joints[..., None]], axis=2)
    padding = np.zeros([24, 1, 4])
    padding[..., 3] = 1
    transforms_mat = np.concatenate([transforms_mat, padding], axis=1)

    # rotate each part
    transform_chain = [transforms_mat[0]]
    for i in range(1, parents.shape[0]):
        curr_res = np.dot(transform_chain[parents[i]], transforms_mat[i])
        transform_chain.append(curr_res)
    transforms = np.stack(transform_chain, axis=0)

    posed_joints = transforms[:, :3, 3].copy()

    # obtain the rigid transformation
    padding = np.zeros([24, 1])
    joints_homogen = np.concatenate([joints, padding], axis=1)
    rel_joints = np.sum(transforms * joints_homogen[:, None], axis=2)
    transforms[..., 3] = transforms[..., 3] - rel_joints
    transforms = transforms.astype(np.float32)

    if return_joints:
        return transforms, posed_joints
    else:
        return transforms

def post_process(vertices):
    vertices += trans
    vertices *= scale
    vertices *= torch.tensor([1.0, 1.0, -1.0])
    vertices *= torch.tensor([1.0, -1.0, 1.0])
    return vertices

def post_process_align(vertices):
    vertices[:, 0] -= 0.0048
    vertices[:, 1] += 0.1227
    vertices[:, 2] += -0.0046
    vertices *= 100.0
    return vertices


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_folder', type=str, default=None,
                        help='input folder')
    parser.add_argument('--temp_outputs', type=str, default=None,
                        help='temp_outputs')
    parser.add_argument('--out_folder', type=str, default=None,
                    help='output folder')

    iterations = 3
    # mean body shape
    mean_body = trimesh.load("body_mean.obj", process=False) # 1
    mean_body.vertices, mean_body.faces = trimesh.remesh.subdivide_loop(vertices=mean_body.vertices, faces=mean_body.faces, iterations=iterations)

    args = parser.parse_args()
    in_dir = args.in_folder
    temp_outputs = args.temp_outputs
    outputs = args.out_folder
    os.makedirs(outputs, exist_ok=True)
    for f in os.listdir(in_dir):
        name = f[0:-4]
        garment_path = temp_outputs + name + "_garment_on_mean_body.obj"
        npz_path = temp_outputs + name + "_smpl.npz"

        shutil.copyfile(in_dir + f, outputs + f)

        smpl_model = SMPL(sex='neutral', model_dir='../../support_data/smpl_models/')
        smpl_params = dict(np.load(npz_path, allow_pickle=True))
        smpl_params = smpl_params["arr_0"][()]
        pose_params = smpl_params['pose_param'].reshape(-1) # np.zeros(72) # smpl_params['pose_param'].reshape(-1) 要不要 pose blend shape
        shape_params = smpl_params['betas'].reshape(-1)
        scale = smpl_params["scale"] # render + optimize 
        trans = smpl_params["trans"]

        verts, _, hv_t_w_spbs, Jtr_T = smpl_model(pose_params, shape_params)

        spbs_body = trimesh.Trimesh(vertices=hv_t_w_spbs, faces=smpl_model.faces, process=False)
        spbs_body.export(temp_outputs + name + "_tpose_spbs_human.obj")
        spbs_body.vertices, spbs_body.faces = trimesh.remesh.subdivide_loop(vertices=spbs_body.vertices, faces=smpl_model.faces, iterations=iterations)

        # tpose garment on mean body shape
        garment_on_mean_body = trimesh.load(garment_path, process=False) # 3
        garment_vertices_on_spbs_body, t_ii = get_garment_on_spbs_body(mean_body, spbs_body, garment_on_mean_body)  

        garment_on_spbs_body = trimesh.Trimesh(vertices=garment_vertices_on_spbs_body, faces=garment_on_mean_body.faces, process=False)
        garment_on_spbs_body.export(temp_outputs + name + "_tpose_spbs_garment.obj")

        smpl_layer = SMPL_Layer(
            center_idx=0,
            gender='neutral',
            model_root='../../support_data/smpl_models/')
        hv_t_w_spbs = torch.from_numpy(hv_t_w_spbs).float().unsqueeze(0)
        Jtr = torch.from_numpy(Jtr_T).float().unsqueeze(0)
        parents = smpl_layer.kintree_parents
        weights = smpl_layer.th_weights
        poses = smpl_params['pose_param'].reshape(-1)
        A = get_rigid_transformation(poses.reshape(24, 3), Jtr[0].numpy(), np.array(parents), return_joints=False)

        posed_verts = tpose_points_to_pose_points(hv_t_w_spbs, weights[None, ...].permute(0, 2, 1), torch.from_numpy(A))[0]
        posed_verts = post_process(posed_verts)
        spbs_body_with_pose = trimesh.Trimesh(vertices=posed_verts, faces=smpl_model.faces, process=False)
        spbs_body_with_pose.export(temp_outputs + name + "_lbs_spbs_human.obj")
        spbs_body_with_pose.vertices = post_process_align(spbs_body_with_pose.vertices)
        spbs_body_with_pose.export(outputs + name + "_lbs_spbs_human_modified.obj")


        garment_bw, _ = sample_blend_closest_points(torch.from_numpy(garment_vertices_on_spbs_body)[None, ...].float(), hv_t_w_spbs, weights.float())
        posed_garment_verts = tpose_points_to_pose_points(torch.from_numpy(garment_vertices_on_spbs_body)[None, ...], garment_bw.permute(0, 2, 1), torch.from_numpy(A))[0]
        posed_garment_verts = post_process(posed_garment_verts)
        garment_on_spbs_body_with_pose = trimesh.Trimesh(vertices=posed_garment_verts, faces=garment_on_mean_body.faces, process=False)
        garment_on_spbs_body_with_pose.export(temp_outputs + name + "_lbs_spbs_garment.obj")
        garment_on_spbs_body_with_pose.vertices = post_process_align(garment_on_spbs_body_with_pose.vertices)
        garment_on_spbs_body_with_pose.export(outputs + name + "_lbs_spbs_garment_modified.obj")
        print(name)


