echo boundary fitting

echo boundary fitting: 00_format_target
bash 3_fitting/00_format_target.sh

echo boundary fitting: 01_format_src
bash 3_fitting/01_format_src.sh

echo boundary fitting: 10_boundary_fitting
bash 3_fitting/10_boundary_fitting.sh

echo boundary fitting: 11_get_landmark_indices_all
bash 3_fitting/11_get_landmark_indices_all.sh


echo fitting

echo fitting: 20_opt_occ_all
bash 3_fitting/20_opt_occ_all.sh

echo fitting: 21_refine
bash 3_fitting/21_refine.sh