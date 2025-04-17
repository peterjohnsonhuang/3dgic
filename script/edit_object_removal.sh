#!/bin/bash

# Check if the user provided an argument
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <output_folder> <config_file>  <type>"
    exit 1
fi


output_folder="$1"
config_file="$2"
config_type="$3"
check_point="$4"

if [ ! -d "$output_folder" ]; then
    echo "Error: Folder '$output_folder' does not exist."
    exit 2
fi



# Remove the selected object
# python edit_object_removal.py -m ${output_folder} --config_file ${config_file} --skip_test --type ${config_type} -c ${check_point}
python edit_object_removal.py -m ${output_folder} --config_file ${config_file} --skip_test --render_intersect
