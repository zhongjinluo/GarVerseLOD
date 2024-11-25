
import torch
from model import Generate
import sys
import cv2
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',type=str)
parser.add_argument('--output_dir',type=str)
opt = parser.parse_args()

generator = Generate(3, 3)
model_CKPT = torch.load("../support_data/checkpoints/normal_estimator.pth")
generator.load_state_dict(model_CKPT)
generator.cuda().eval()

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

root = opt.input_dir
out_dir = opt.output_dir + '/estimated_normal/'
os.makedirs(out_dir, exist_ok=True)
for f in os.listdir(root):
    for i in range(1):
        rgb = Image.open(os.path.join(root, f))
        rgb = to_tensor(rgb).float()
        pred = generator(rgb.unsqueeze(0).cuda().float())
        preds = (pred.permute(0,2,3,1) + 1) / 2 * 255.0
        preds = preds.cpu().detach().numpy()
        img = np.array(preds[0,:,:,:])[:,:,::-1]
        input = cv2.imread(os.path.join(root, f))
        cv2.imwrite(out_dir + f, img)