#!/bin/bash

# Check if the user provided an argument
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <output_folder> <config_file>  <data_folder>"
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


mkdir ${data_folder}/object_mask/depth_removal
cp -r ${output_folder}/train/ours_object_removal/iteration_40000/depth/*  ${data_folder}/object_mask/depth_removal


# Remove the selected object
# python edit_object_removal.py -m ${output_folder} --config_file ${config_file} --skip_test --type ${config_type} -c ${check_point}
python edit_object_removal.py -m ${output_folder} --config_file ${config_file} --skip_test --render_intersect
