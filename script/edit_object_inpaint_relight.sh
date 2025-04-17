#!/bin/bash

# Check if the user provided an argument
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <output_folder> <config_file> <check_point>"
    exit 1
fi


output_folder="$1"
config_file="$2"
check_point="$3"

if [ ! -d "$output_folder" ]; then
    echo "Error: Folder '$output_folder' does not exist."
    exit 2
fi



# Remove the selected object
python edit_object_inpaint.py  -m ${output_folder} --config_file ${config_file} --type neilf --skip_test  -c ${check_point} \
--lambda_normal_render_depth 0.01 \
--use_global_shs \
--iterations 20000 \
--lambda_light 0.01 \
--lambda_base_color 0.005 \
--lambda_base_color_smooth 0.006 \
--lambda_metallic_smooth 0.002 \
--lambda_roughness_smooth 0.002 \
--lambda_visibility 0.1 \
--save_training_vis \
# --finetune_visibility \



# --test_interval 1000 \
# --checkpoint_interval 2500 \