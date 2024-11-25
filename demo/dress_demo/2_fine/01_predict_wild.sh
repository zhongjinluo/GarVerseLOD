cd 2_fine/
CUDA_VISIBLE_DEVICES=0 python -m apps.predict_refine_wild --dataroot ../outputs/temp/fine_inputs/ --results_path ../outputs/temp/fine_garment/ --resolution 512
cd ..