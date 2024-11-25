echo coarse garment

echo coarse garment: estimate smpl
bash 1_coarse/00_get_smpl.sh

echo estimate tpose_garment_on_mean_body
bash 1_coarse/01_get_tpose_garment_on_mean_body.sh

echo coarse garment: pose garment
bash 1_coarse/02_pose_garment.sh