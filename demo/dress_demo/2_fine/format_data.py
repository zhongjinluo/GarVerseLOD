import os
import numpy as np
import cv2
import shutil
import argparse
parser = argparse.ArgumentParser(description='NICP')
parser.add_argument('--normal_dir',type=str)
parser.add_argument('--mask_dir',type=str)
parser.add_argument('--output_dir',type=str)
opt = parser.parse_args()

normal_root = opt.normal_dir
mask_root = opt.mask_dir
out_dir = opt.output_dir

def mask_img(img, mask, bg_value=127):
    img_result = np.ones_like(img) * bg_value
    img_result[mask==255] = img[mask==255]
    return img_result

for f in os.listdir(mask_root):
    d = f[0:-4]
    print(d)
    os.makedirs(out_dir + d + "/garment_mask", exist_ok=True)
    os.makedirs(out_dir + d + "/normal_F_masked", exist_ok=True)   
    os.makedirs(out_dir + d + "/normal_F_garment_masked", exist_ok=True)
    os.makedirs(out_dir + d + "/calib", exist_ok=True)
    
    garment_mask_path = os.path.join(mask_root, f)
    mask = cv2.imread(garment_mask_path, -1)
    mask = mask[:, :, 0]
    cv2.imwrite(out_dir + d + "/garment_mask/" + "000.png", mask)
    
    normal_F_path = os.path.join(normal_root, f)
    shutil.copyfile(normal_F_path, out_dir + d + "/normal_F_masked/" + "000.png")

    
    normal_F = cv2.imread(normal_F_path)
    normal_F_garment_masked = mask_img(normal_F, mask)
    cv2.imwrite(out_dir + d + "/normal_F_garment_masked/" + "000.png", normal_F_garment_masked)
    
    shutil.copyfile("cam.txt", out_dir + d + "/calib/" + "000.txt")