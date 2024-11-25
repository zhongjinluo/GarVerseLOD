cd 1_coarse/ICON_get_smpl/
CUDA_VISIBLE_DEVICES=0 python -m apps.infer_smpl -cfg ./configs/icon-filter.yaml -gpu 0 -in_dir ../../inputs/imgs/ -out_dir ../../outputs/temp/coarse_temp/ -export_video -loop_smpl 1 -loop_cloth 200 -hps_type pymaf
cd ../