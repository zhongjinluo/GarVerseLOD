# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import warnings
import logging
import cv2
warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)

from tqdm.auto import tqdm
from lib.common.render import query_color, image2vid
from lib.renderer.mesh import compute_normal_batch
from lib.common.config import cfg
from lib.common.cloth_extraction import extract_cloth
from lib.dataset.mesh_util import (
    load_checkpoint, update_mesh_shape_prior_losses, get_optim_grid_image, blend_rgb_norm, unwrap,
    remesh, tensor2variable, rot6d_to_rotmat
)

from lib.dataset.TestDataset import TestDataset
# from lib.net.local_affine import LocalAffine
from pytorch3d.structures import Meshes
from apps.ICON import ICON

import os
from termcolor import colored
import argparse
import numpy as np
from PIL import Image
import trimesh
import pickle
import numpy as np
import torch

torch.backends.cudnn.benchmark = True

# from pymaf
def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis

# from pymaf
def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q

# from pymaf
def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3,3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32,
                           device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa

if __name__ == "__main__":

    # loading cfg file
    parser = argparse.ArgumentParser()

    parser.add_argument("-gpu", "--gpu_device", type=int, default=0)
    parser.add_argument("-colab", action="store_true")
    parser.add_argument("-loop_smpl", "--loop_smpl", type=int, default=100)
    parser.add_argument("-patience", "--patience", type=int, default=5)
    parser.add_argument("-vis_freq", "--vis_freq", type=int, default=1000)
    parser.add_argument("-loop_cloth", "--loop_cloth", type=int, default=200)
    parser.add_argument("-hps_type", "--hps_type", type=str, default="pixie")
    parser.add_argument("-export_video", action="store_true")
    parser.add_argument("-in_dir", "--in_dir", type=str, default="./examples")
    parser.add_argument("-out_dir", "--out_dir", type=str, default="./results")
    parser.add_argument('-seg_dir', '--seg_dir', type=str, default=None)
    parser.add_argument("-cfg", "--config", type=str, default="./configs/icon-filter.yaml")

    args = parser.parse_args()

    # cfg read and merge
    cfg.merge_from_file(args.config)
    cfg.merge_from_file("./lib/pymaf/configs/pymaf_config.yaml")

    cfg_show_list = [
        "test_gpus", [args.gpu_device], "mcube_res", 256, "clean_mesh", True, "test_mode", True,
        "batch_size", 1
    ]

    cfg.merge_from_list(cfg_show_list)
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = torch.device(f"cuda:{args.gpu_device}")

    # load model and dataloader
    model = ICON(cfg)
    model = load_checkpoint(model, cfg)

    dataset_param = {
        'image_dir': args.in_dir,
        'seg_dir': args.seg_dir,
        'colab': args.colab,
        'has_det': True,    # w/ or w/o detection
        'hps_type':
            args.hps_type    # pymaf/pare/pixie
    }

    if args.hps_type == "pixie" and "pamir" in args.config:
        print(colored("PIXIE isn't compatible with PaMIR, thus switch to PyMAF", "red"))
        dataset_param["hps_type"] = "pymaf"

    dataset = TestDataset(dataset_param, device)

    print(colored(f"Dataset Size: {len(dataset)}", "green"))

    pbar = tqdm(dataset)
    cnt_data = 0
    total = len(dataset)
    for data in dataset:

        cnt_data += 1
        
        in_tensor = {"smpl_faces": data["smpl_faces"], "image": data["image"]}

        # The optimizer and variables
        optimed_pose = torch.tensor(
            data["body_pose"], device=device, requires_grad=True
        )    # [1,23,3,3]
        optimed_trans = torch.tensor(data["trans"], device=device, requires_grad=True)    # [3]
        optimed_betas = torch.tensor(data["betas"], device=device, requires_grad=True)    # [1,10]
        optimed_orient = torch.tensor(
            data["global_orient"], device=device, requires_grad=True
        )    # [1,1,3,3]
        optimizer_smpl = torch.optim.Adam(
            [optimed_pose, optimed_trans, optimed_betas, optimed_orient],
            lr=1e-3,
            amsgrad=True,
        )
        scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_smpl,
            mode="min",
            factor=0.5,
            verbose=0,
            min_lr=1e-5,
            patience=args.patience,
        )

        losses = {
            "cloth": {
                "weight": 1e1,
                "value": 0.0
            },    # Cloth: Normal_recon - Normal_pred
            "stiffness": {
                "weight": 1e5,
                "value": 0.0
            },    # Cloth: [RT]_v1 - [RT]_v2 (v1-edge-v2)
            "rigid": {
                "weight": 1e5,
                "value": 0.0
            },    # Cloth: det(R) = 1
            "edge": {
                "weight": 0,
                "value": 0.0
            },    # Cloth: edge length
            "nc": {
                "weight": 0,
                "value": 0.0
            },    # Cloth: normal consistency
            "laplacian": {
                "weight": 1e2,
                "value": 0.0
            },    # Cloth: laplacian smoonth
            "normal": {
                "weight": 1e0,
                "value": 0.0
            },    # Body: Normal_pred - Normal_smpl
            "silhouette": {
                "weight": 1e0,
                "value": 0.0
            },    # Body: Silhouette_pred - Silhouette_smpl
        }

        # smpl optimization
        loop_smpl = range(args.loop_smpl)

        per_data_lst = []

        cnt = 0
        for i in loop_smpl:

            cnt += 1

            per_loop_lst = []

            optimizer_smpl.zero_grad()

            # 6d_rot to rot_mat
            optimed_orient_mat = rot6d_to_rotmat(optimed_orient.view(-1, 6)).unsqueeze(0)
            optimed_pose_mat = rot6d_to_rotmat(optimed_pose.view(-1, 6)).unsqueeze(0)

            if dataset_param["hps_type"] != "pixie":
                smpl_out = dataset.smpl_model(
                    betas=optimed_betas,
                    body_pose=optimed_pose_mat,
                    global_orient=optimed_orient_mat,
                    transl=optimed_trans,
                    pose2rot=False,
                )

                smpl_verts = smpl_out.vertices * data["scale"]
                smpl_joints = smpl_out.joints * data["scale"]
            else:
                smpl_verts, smpl_landmarks, smpl_joints = dataset.smpl_model(
                    shape_params=optimed_betas,
                    expression_params=tensor2variable(data["exp"], device),
                    body_pose=optimed_pose_mat,
                    global_pose=optimed_orient_mat,
                    jaw_pose=tensor2variable(data["jaw_pose"], device),
                    left_hand_pose=tensor2variable(data["left_hand_pose"], device),
                    right_hand_pose=tensor2variable(data["right_hand_pose"], device),
                )

                smpl_verts = (smpl_verts + optimed_trans) * data["scale"]
                smpl_joints = (smpl_joints + optimed_trans) * data["scale"]

            smpl_joints *= torch.tensor([1.0, 1.0, -1.0]).to(device)

            if data["type"] == "smpl":
                in_tensor["smpl_joint"] = smpl_joints[:, :24, :]
            elif data["type"] == "smplx" and dataset_param["hps_type"] != "pixie":
                in_tensor["smpl_joint"] = smpl_joints[:, dataset.smpl_joint_ids_24, :]
            else:
                in_tensor["smpl_joint"] = smpl_joints[:, dataset.smpl_joint_ids_24_pixie, :]

            # render optimized mesh (normal, T_normal, image [-1,1])
            in_tensor["T_normal_F"], in_tensor["T_normal_B"] = dataset.render_normal(
                smpl_verts * torch.tensor([1.0, -1.0, -1.0]).to(device), in_tensor["smpl_faces"]
            )
            T_normal_F_save = (
                ((in_tensor["T_normal_F"][0].permute(1, 2, 0) + 1.0) * 255.0 /
                2.0).detach().cpu().numpy().astype(np.uint8)
            )


            T_mask_F, T_mask_B = dataset.render.get_silhouette_image()

            with torch.no_grad():
                in_tensor["normal_F"], in_tensor["normal_B"] = model.netG.normal_filter(in_tensor)

            diff_F_smpl = torch.abs(in_tensor["T_normal_F"] - in_tensor["normal_F"])
            diff_B_smpl = torch.abs(in_tensor["T_normal_B"] - in_tensor["normal_B"])

            losses["normal"]["value"] = (diff_F_smpl + diff_B_smpl).mean()

            # silhouette loss
            smpl_arr = torch.cat([T_mask_F, T_mask_B], dim=-1)[0]
            gt_arr = torch.cat([in_tensor["normal_F"][0], in_tensor["normal_B"][0]],
                               dim=2).permute(1, 2, 0)
            gt_arr = ((gt_arr + 1.0) * 0.5).to(device)
            bg_color = (torch.Tensor([0.5, 0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(device))
            gt_arr = ((gt_arr - bg_color).sum(dim=-1) != 0.0).float()
            diff_S = torch.abs(smpl_arr - gt_arr)
            losses["silhouette"]["value"] = diff_S.mean()

            # Weighted sum of the losses
            smpl_loss = 0.0
            pbar_desc = "Body Fitting --- "
            for k in ["normal", "silhouette"]:
                pbar_desc += f"{k}: {losses[k]['value'] * losses[k]['weight']:.3f} | "
                smpl_loss += losses[k]["value"] * losses[k]["weight"]
            pbar_desc += f"Total: {smpl_loss:.3f}"

            if i % args.vis_freq == 0:

                per_loop_lst.extend(
                    [
                        in_tensor["image"],
                        in_tensor["T_normal_F"],
                        in_tensor["normal_F"],
                        diff_F_smpl / 2.0,
                        diff_S[:, :512].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1),
                    ]
                )
                per_loop_lst.extend(
                    [
                        in_tensor["image"],
                        in_tensor["T_normal_B"],
                        in_tensor["normal_B"],
                        diff_B_smpl / 2.0,
                        diff_S[:, 512:].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1),
                    ]
                )
                per_data_lst.append(get_optim_grid_image(per_loop_lst, None, nrow=5, type="smpl"))
            if cnt < args.loop_smpl:
                smpl_loss.backward()
                optimizer_smpl.step()
                scheduler_smpl.step(smpl_loss)
            in_tensor["smpl_verts"] = smpl_verts * \
                torch.tensor([1.0, 1.0, -1.0]).to(device)

        # visualize the optimization process
        # 1. SMPL Fitting

        os.makedirs(args.out_dir, exist_ok=True)

        norm_pred = (
            ((in_tensor["normal_F"][0].permute(1, 2, 0) + 1.0) * 255.0 /
             2.0).detach().cpu().numpy().astype(np.uint8)
        )

        norm_orig = unwrap(norm_pred, data)
        mask_orig = unwrap(
            np.repeat(data["mask"].permute(1, 2, 0).detach().cpu().numpy(), 3,
                      axis=2).astype(np.uint8),
            data,
        )
        rgb_norm = blend_rgb_norm(data["ori_image"], norm_orig, mask_orig)

        smpl_obj = trimesh.Trimesh(
            in_tensor["smpl_verts"].detach().cpu()[0] * torch.tensor([1.0, 1.0, -1.0]),
            in_tensor['smpl_faces'].detach().cpu()[0],
            process=False,
            maintains_order=True
        )
        smpl_obj.export(args.out_dir + f"/{data['name']}_smpl_raw.obj")

        smpl_obj = trimesh.Trimesh(
            in_tensor["smpl_verts"].detach().cpu()[0] * torch.tensor([1.0, -1.0, 1.0]),
            in_tensor['smpl_faces'].detach().cpu()[0],
            process=False,
            maintains_order=True
        )
        smpl_obj.export(args.out_dir + f"/{data['name']}_smpl.obj")
        mesh = trimesh.Trimesh(vertices=smpl_obj.vertices, faces=smpl_obj.faces, process=False)
        mesh.vertices[:, 0] -= 0.0048
        mesh.vertices[:, 1] += 0.1227
        mesh.vertices[:, 2] += -0.0046
        mesh.vertices *= 100.0
        mesh.export(args.out_dir + f"/{data['name']}_smpl_modified.obj")

        pose_all = torch.cat([optimed_orient, optimed_pose], 1)
        pred_rotmat = rot6d_to_rotmat(pose_all).view(1, 24, 3, 3)
        pose_param = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        smpl_info = {
            'betas': optimed_betas,
            "pose_param": pose_param,
            'pose': optimed_pose,
            'orient': optimed_orient,
            'trans': optimed_trans,
            'scale': data["scale"]
        }

        smpl_info = {
            'betas': optimed_betas.detach().cpu().numpy(),
            "pose_param": pose_param.detach().cpu().numpy(),
            'pose': optimed_pose.detach().cpu().numpy(),
            'orient': optimed_orient.detach().cpu().numpy(),
            'trans': optimed_trans.detach().cpu().numpy().reshape(-1),
            'scale': data["scale"].detach().cpu().numpy()
        }
        np.savez(
            args.out_dir + f"/{data['name']}_smpl.npz", smpl_info, allow_pickle=True
        )
