# from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from model import Embedding
import openmesh as om
import torch
import os
import shutil
import json
import argparse

pcamat = np.load('../../support_data/smgl_data/pcamat.npy')
coeffdic = np.load('../../support_data/smgl_data/coeffdic.npy',allow_pickle=True).item()
# print(coeffdic)

mesh = om.read_polymesh('../../support_data/smgl_data/mean.obj')
points =  mesh.points()
faces = mesh.face_vertex_indices()
maxmin = np.load('../../support_data/smgl_data/maxmin.npy',allow_pickle=True)

pcamat = pcamat[:32, :]
to_tensor = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
embedding = Embedding(is_train=True)
model_CKPT = torch.load("../../support_data/checkpoints/tpose_garment_estimator.pth")
embedding.load_state_dict({k.replace('module.', ''):v for k,v in model_CKPT.items()})
embedding.eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_folder', type=str, default=None,
                        help='input folder')
    parser.add_argument('--out_folder', type=str, default=None,
                        help='output folder')

    args = parser.parse_args()
    root = args.in_folder
    outdir = args.out_folder

    for f in os.listdir(root):
        img = Image.open(os.path.join(root, f))
        img.save(outdir + f)
        img = to_tensor(img)
        img_embedding = embedding(img.unsqueeze(0))
        coeff = img_embedding.detach().numpy().reshape(-1)
        project = np.dot(pcamat.T, coeff)
        newpoints = project.reshape(-1,3) + points
        newmesh = om.PolyMesh(points=newpoints, face_vertex_indices=faces)
        om.write_mesh(os.path.join(outdir, f.replace(".png", "_garment_on_mean_body.obj")), newmesh)
        # om.write_mesh(os.path.join(outdir, f.replace(".png", "_pred.obj")), newmesh)
        print(os.path.join(outdir, f.replace(".png", "_garment_on_mean_body.obj")),)
