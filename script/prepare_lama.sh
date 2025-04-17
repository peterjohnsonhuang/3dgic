#!/bin/bash

cd lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)

# dataset=bear
# dataset=mipnerf360/kitchen_depth
dataset=mipnerf360/counter_depth
img_dir=./LaMa_test_images/$dataset

python bin/predict.py refine=True model.path=$(pwd)/big-lama indir=$(pwd)/LaMa_test_images/$dataset outdir=$(pwd)/LaMa_test_images/$dataset
# python prepare_pseudo_label.py $(pwd)/LaMa_test_images/$dataset $img_dir