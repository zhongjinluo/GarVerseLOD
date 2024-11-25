cd 1_coarse/smpl_lbs_to_garment/
CUDA_VISIBLE_DEVICES=0 python pose_garment.py --in_folder ../../inputs/imgs/ --out_folder ../../outputs/temp/coarse_garment/ --temp ../../outputs/temp/coarse_temp/ 
cd ../