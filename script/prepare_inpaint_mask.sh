#!/bin/bash

cd Tracking-Anything-with-DEVA/

# img_path=../output/mipnerf360/kitchen/train/ours_object_removal/iteration_30000/renders
# mask_path=./output_2d_inpaint_mask/mipnerf360/kitchen
# lama_path=../lama/LaMa_test_images/mipnerf360/kitchen


img_path=../../RelightableGrouping/output/NeRF_Syn/bear_relighting/neilf/train/ours_object_removal/iteration_30000/renders
mask_path=./output_2d_inpaint_mask/r3dgg/bear
lama_path=../lama/LaMa_test_images/r3dgg/bear

# img_path=../output/mipnerf360/kitchen/train/ours_30000/renders
# mask_path=./output_2d_inpaint_mask/mipnerf360/kitchen_excavator
# lama_path=../lama/LaMa_test_images/mipnerf360/kitchen_excavator

# img_path=../output/bear/train/ours_object_removal/iteration_30000/renders
# mask_path=./output_2d_inpaint_mask/bear
# lama_path=../lama/LaMa_test_images/bear

python demo/demo_with_text.py   --chunk_size 4    --img_path $img_path  --amp \
  --temporal_setting semionline --size 480   --output $mask_path  \
  --prompt "big blurry black hole"

python prepare_lama_input.py $img_path $mask_path $lama_path
cd ..