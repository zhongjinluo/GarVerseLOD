## GarVerseLOD Demo

Demo Code for Reconstructing Garments from a Single Image (Dresses as an Example).

### Downloading required models and extra data

- Downloading required models and data for GarVerseLOD from [Google Drive](https://drive.google.com/file/d/1ylz5EoVFPmEAhO1cwUjO_zfa-oz5n608/view?usp=sharing). 

  ```
  cd GarVerseLOD/demo/dress_demo/
  unzip support_data.zip
  ```

- Required Models and Data for HPS: We use [ICON's codebase](https://github.com/YuliangXiu/ICON) for SMPL optimization. Here we use [PyMAF](https://github.com/HongwenZhang/PyMAF#necessary-files) for SMPL estimation, but you can easily switch to [PARE (SMPL)](https://github.com/mkocabas/PARE#demo), [PIXIE (SMPL-X)](https://pixie.is.tue.mpg.de/), [HybrIK (SMPL)](https://github.com/Jeff-sjtu/HybrIK) with this codebase. Many thanks to [@YuliangXiu](https://github.com/YuliangXiu) for his excellent work. Please register on ICON's website by following [this instruction](https://github.com/YuliangXiu/ICON/blob/master/docs/installation.md) and then download the required models for HPS:

  ```
  cd GarVerseLOD/demo/dress_demo/1_coarse/ICON_get_smpl/
  bash fetch_data.sh
  bash fetch_hps.sh
  ```

- The directory structure is expected as follows:

  ```
  ├── demo
  │   ├── dress_demo
  │   │   ├── 0_normal_estimator
  │   │   ├── 1_coarse
  │   │   │   ├── ICON_get_smpl
  │   │   │   │   ├── data # HPS
  │   │   │   ├── smpl_lbs_to_garment
  │   │   │   └── tpose_garment_estimator
  │   │   ├── 2_fine
  │   │   ├── 3_fitting
  │   │   ├── inputs
  │   │   │   ├── imgs
  │   │   │   └── masks
  │   │   └── support_data # GarVerseLOD
  ```
  
- We provide an example input in the `inputs/` folder for you to quickly explore our system. You can download more prepared example inputs from our [Google Drive](https://drive.google.com/file/d/1LAWB4tuYRslJEQcn6l8uDPDfaeqKS9Yj/view?usp=sharing).

### Running Demo

  ```
bash demo.sh
  ```
Then you can find all the results in `outputs/results/`. **You can download more prepared example inputs from our [Google Drive](https://drive.google.com/file/d/1LAWB4tuYRslJEQcn6l8uDPDfaeqKS9Yj/view?usp=sharing).**

### Running Demo Step by Step

To walk through our system step by step, please adhere to the following instructions:

- Get normal map

  ```
  # 0_normal_estimator/run.sh
  
  bash 0_normal_estimator/00_predict_normal.sh
  ```

- Get coarse garment

  ```
  # 1_coarse/run.sh
  
  bash 1_coarse/00_get_smpl.sh
  bash 1_coarse/01_get_tpose_garment_on_mean_body.sh
  bash 1_coarse/02_pose_garment.sh
  ```

- Get fine garment

  ```
  # 2_fine/run.sh
  
  bash 2_fine/00_format_data.sh
  bash 2_fine/01_predict_wild.sh
  # bash 2_fine/02_refine_boundary.sh
  ```

  Here we are using 2D-aware boundary predictions, and the code for 3D-aware boundary is in preparation. We have increased the training data, and the 2D-aware results are now significantly closer to the 3D-aware boundary predictions.

- Registration

  ```
  # 3_fitting/run.sh
  
  # boundary fitting
  bash 3_fitting/00_format_target.sh
  bash 3_fitting/01_format_src.sh
  bash 3_fitting/10_boundary_fitting.sh
  bash 3_fitting/11_get_landmark_indices_all.sh
  # nicp
  bash 3_fitting/20_opt_occ_all.sh
  bash 3_fitting/21_refine.sh
  ```

Then you can find all the results in `outputs/results/`. If the result of nicp is not satisfactory, try adjusting the loss weights: `3_fitting/nicp/config/cloth.json`. 

## Prepare Your Data

To prepare your own data, you need to:

- Remove the background and crop the image: [rembg](https://github.com/danielgatis/rembg), [ICON](https://github.com/YuliangXiu/ICON)
- Annotate the garment mask. **Please note that the mask should cover the entire garment, including the invisible region (e.g., the area obstructed by the body, as shown in the following image)**

<img src="./dress_mask.png">