# get tpose garment_on_mean_body
cd ./1_coarse/tpose_garment_estimator
CUDA_VISIBLE_DEVICES=0 python test_wild.py --in_folder ../../inputs/imgs/ --out_folder ../../outputs/temp/coarse_temp/
cd ..
