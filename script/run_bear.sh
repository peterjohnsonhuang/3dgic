#!/bin/bash

root_dir="./data/"
list="bear"

for i in $list; do
# python train.py --eval \
# -s ${root_dir}${i} \
# -m output/NeRF_Syn/${i}_0823/3dgs \
# --lambda_normal_render_depth 0.01 \
# --lambda_mask_entropy 0.1 \
# --config_file configs/gaussian_dataset/train.json \
# --densification_interval 500 \
# --save_training_vis \
# --iterations 40000

# done
python train.py --eval \
-s ${root_dir}"bear" \
-m output/NeRF_Syn/bear_relighting_0825/neilf \
-c ./output/NeRF_Syn/bear_0823/3dgs/chkpnt40000.pth \
-t neilf \
--lambda_normal_render_depth 0.01 \
--use_global_shs \
--config_file configs/gaussian_dataset/train_tune.json \
--iterations 70000 \
--test_interval 1000 \
--checkpoint_interval 2500 \
--lambda_light 0.01 \
--lambda_base_color 0.005 \
--lambda_base_color_smooth 0.006 \
--lambda_metallic_smooth 0.002 \
--lambda_roughness_smooth 0.002 \
--lambda_visibility 0.1 \
--save_training_vis \
--densification_interval 1000 \
--finetune_visibility \
# --lambda_mask_entropy 0.1 \
done