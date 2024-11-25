from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import logging
import scipy.io as sio
import json

log = logging.getLogger('trimesh')
log.setLevel(40)

def crop_image(img, rect):
    x, y, w, h = rect

    left = abs(x) if x < 0 else 0
    top = abs(y) if y < 0 else 0
    right = abs(img.shape[1]-(x+w)) if x + w >= img.shape[1] else 0
    bottom = abs(img.shape[0]-(y+h)) if y + h >= img.shape[0] else 0

    if img.shape[2] == 4:
        color = [0, 0, 0, 0]
    else:
        color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    x = x + left
    y = y + top

    return new_img[y:(y+h),x:(x+w),:]

def load_trimesh(root_dir):
    folders = os.listdir(root_dir)
    meshs = {}
    for i, f in enumerate(folders):
        if (f[-4:] == '.obj'):
            meshs[f[:-4]] = trimesh.load(os.path.join(root_dir, f))
    return meshs

def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob > 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )


class TrainDataset_Refine(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.projection_mode = 'orthogonal'

        # Path setup
        self.root = self.opt.dataroot
        self.SAMPLE = "geo"
        self.CALIB = 'calib'
        self.NORM_F = 'normal_F_garment_masked'
        # self.MASK = 'garment_mask'

        self.B_MIN = np.array([-128, -100, -128])
        self.B_MAX = np.array([128, 180, 128])

        self.is_train = (phase == 'train')
        self.load_size = self.opt.loadSize

        self.num_sample_inout = self.opt.num_sample_inout
        self.num_sample_color = self.opt.num_sample_color

        self.subjects = self.get_subjects()
        self.augs = 1

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.to_tensor2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])
        # self.mesh_dic = load_trimesh(self.OBJ)

    def get_img_info(self, subject):
        calib_list = []
        render_list = []

        calib_path = os.path.join(self.root, subject[0], self.CALIB, subject[1] + ".txt")
        calib_data = np.loadtxt(calib_path, dtype=float)
        extrinsic = calib_data[:4, :4]
        intrinsic = calib_data[4:8, :4]
        calib_mat = np.matmul(intrinsic, extrinsic)
        calib = torch.from_numpy(calib_mat).float()


        fnorm_path = os.path.join(self.root, subject[0], self.NORM_F, subject[1] + ".png")
        fnorm = Image.open(fnorm_path).convert('RGB')
        # fnorm.save("normal.png")
        fnorm = fnorm.resize((512, 512))

        # mask_path = os.path.join(self.root, subject[0], self.MASK, subject[1] + ".png")
        # mask = cv2.imread(mask_path, -1)
        # cv2.imwrite("mask.png", mask)
        # mask = np.array(mask, dtype=np.float)
        # mask_input = self.to_tensor2(mask.reshape(512, 512, 1)).float()

        if self.is_train:
            fnorm = self.aug_trans(fnorm)
            if self.opt.aug_blur > 0.00001:
                blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                fnorm = fnorm.filter(blur)

        fnorm = self.to_tensor(fnorm)
        render = torch.cat([fnorm], 0)

        render_list.append(render)
        calib_list.append(calib)

        return {
            'img': torch.stack(render_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
        }

    def get_subjects(self):
        all_subjects = []
        dirs = os.listdir(self.root)
        for d in dirs:
            sub_dir = os.path.join(self.root, d, self.CALIB)
            for f in os.listdir(sub_dir):
                all_subjects.append((d, f[0:-4]))
        random.shuffle(all_subjects)
        # all_subjects = [("17122957607431219", f[0:-4])]
        return all_subjects

    def __len__(self):
        return len(self.subjects) * self.augs
    
    def get_sub_samples(self, samples, num):
        random_idx = (torch.rand(num) * samples.shape[0]).long()
        sub_samples = samples[random_idx][:, 0:3]
        sub_labels = samples[random_idx][:, 3:]
        return sub_samples, sub_labels

    def select_sampling_method(self, subject):
        if not self.is_train:
            random.seed(1991)
            np.random.seed(1991)
            torch.manual_seed(1991)
            
        sample_dict = dict(np.load(os.path.join(self.root.replace("thuman2_36views_1", "thuman2_36views_sample_oi"), 
                                    subject[0], self.SAMPLE, "scale_closed_gp.npz"), allow_pickle=True))
        
        sample_dict = sample_dict["arr_0"][()]
        surface_inside_samples = torch.from_numpy(sample_dict["garment_data"]["surface_inside_samples"])
        surface_outside_samples = torch.from_numpy(sample_dict["garment_data"]['surface_outside_samples'])
        random_inside_samples = torch.from_numpy(sample_dict["garment_data"]["random_inside_samples"])
        random_outside_samples = torch.from_numpy(sample_dict["garment_data"]["random_outside_samples"])
        
        strategy = [
            (surface_inside_samples, self.num_sample_inout * 4),
            (surface_outside_samples, self.num_sample_inout * 4),
            (random_inside_samples, self.num_sample_inout // 4),
            (random_outside_samples, self.num_sample_inout // 4)
        ]
        
        samples = []
        labels = []
        for s in strategy:
            sub_samples, sub_labels = self.get_sub_samples(s[0], s[1])
            samples.append(sub_samples)
            labels.append(sub_labels)
        samples = torch.cat(samples, 0)
        labels = torch.cat(labels, 0)
        random_idx = (torch.rand(self.num_sample_inout) * samples.shape[0]).long()
        samples = samples[random_idx]
        labels = labels[random_idx]
        # print(samples.shape, labels.shape)
        
        boundary_samples = torch.from_numpy(sample_dict["boundary_data"])
        random_idx = (torch.rand(4666) * boundary_samples.shape[0]).long()
        boundary_samples = boundary_samples[random_idx]
        # print(boundary_samples.shape, boundary_samples.shape)
        
        samples = torch.cat([samples, boundary_samples[:, 0:3]], 0)
        labels = torch.cat([labels, boundary_samples[:, 3:]], 0)
        
        # mesh_path = os.path.join(self.root, subject[0], self.SAMPLE, "scale.obj")
        # mesh = trimesh.load(mesh_path, process=False)
        # samples = torch.from_numpy(mesh.vertices)
        # labels = torch.zeros((mesh.vertices.shape[0], 1))
        
        return {
             'samples': samples.float().T,
             'labels': labels.float().T
        }

    def get_item(self, index):
        # In case of a missing file or IO error, switch to a random sample instead
        # try:
        sid = index // self.augs

        subject = self.subjects[sid]
        res = {
            'name': subject,
            # 'mesh_path': os.path.join(self.OBJ, subject + '.obj'),
            'sid': sid,
            'b_min': self.B_MIN,
            'b_max': self.B_MAX,
        }
        render_data = self.get_img_info(subject)
        res.update(render_data)

        if self.opt.num_sample_inout:
            sample_data = self.select_sampling_method(subject)
            res.update(sample_data)

        return res

    def __getitem__(self, index):
        return self.get_item(index)
