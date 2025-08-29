#!/bin/bash

# Check if the user provided an argument
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <output_folder> <config_file> <data_folder>"
    exit 1
fi


output_folder="$1"
config_file="$2"
data_folder="$3"

if [ ! -d "$output_folder" ]; then
    echo "Error: Folder '$output_folder' does not exist."
    exit 2
fi


if [ ! -d "$data_folder" ]; then
    echo "Error: Folder '$data_folder' does not exist."
    exit 2
fi

mkdir ${data_folder}/inpaint_2d_unseen_mask
mkdir ${data_folder}/inpaint_2d_unseen_mask/depth_removal
mkdir ${data_folder}/inpaint_2d_unseen_mask/images
mkdir ${data_folder}/inpaint_2d_unseen_mask/obj_original

cp -r ${output_folder}/train/ours_object_removal/iteration_40000/inpaint_mask_pred/*  ${data_folder}/inpaint_2d_unseen_mask/
cp -r ${output_folder}/train/ours_object_removal/iteration_40000/renders/*  ${data_folder}/inpaint_2d_unseen_mask/images/
cp -r ${output_folder}/train/ours_object_removal/iteration_40000/depth_removal/*  ${data_folder}/inpaint_2d_unseen_mask/depth_removal/
cp -r ${output_folder}/train/ours_object_removal/iteration_40000/gt_objects/*  ${data_folder}/inpaint_2d_unseen_mask/obj_original/


mkdir ${data_folder}/intersect_mask

cp -r ${output_folder}/train/ours_object_removal/iteration_40000/inpaint_mask_pred/*  ${data_folder}/intersect_mask/
