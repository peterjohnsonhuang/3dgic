# [CVPR 2025] 3D Gaussian Inpainting with Depth-Guided Cross-view Consistency (3DGIC)<br>
> Sheng-Yu Huang, Zi-Ting Chou, Yu-Chiang Frank Wang <br>
> [Project Page](https://peterjohnsonhuang.github.io/3dgic-pages/) | [Paper](https://arxiv.org/abs/2502.11801)

The full version of the implementation belongs to the company that sponsored this project. This repository is a suboptimal version, but anyone who wants to compare with our method can treat this repo as official implementation. If you want to get the original results of our method, you can also contact via:
f08942095@ntu.edu.tw 

<div align="center">
  <img src="img_src/3DGIC_CVPR2025.jpg"/>
</div>


Since the original environment is deleted, I've tried build the environment using RTX 5090 GPU with nvidia-driver-570.
#### Install dependencies
for all the packages, please see requirements.txt

#### Install additional pytorch extensions

You can refer to **install.md** to see how I build my environment

## :bookmark_tabs: Todos
We will be releasing all the following contents:
- [x] Training and inference code for 3DGIC
- [x] Provide example of the Bear dataset
- [ ] Demo for relighting


### Running
We will provide almost everything for the bear dataset [here](https://drive.google.com/file/d/1Dc1Q-S5mJIIhE_27DyBPhYIeKGP2zgao/view?usp=sharing) so you can take a look how to put files. 

If you want to prepare the data and train the 3dgs model from scratch, the original data format should be like the preperation results from [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping/blob/main/docs/train.md). After obtain the original data, please run 
:
```
bash script/run_bear.sh
```
to train the original 3DGS model.

Then you need to run the removing process by:
```
CUDA_LAUNCH_BLOCKING=1 bash ./script/edit_object_removal.sh  ./output/NeRF_Syn/bear/3dgs/  ./configs/object_removal/bear.json render ./output/NeRF_Syn/bear/3dgs/chkpnt40000.pth
```
After that, we can obtain the depth-guided mask by:
```
CUDA_LAUNCH_BLOCKING=1 bash ./script/find_depth_guided_mask.sh  ./output/NeRF_Syn/bear/3dgs/  ./configs/object_removal/bear.json ./data/bear/
```
Then you'll have to move the rendered data into you data folder:
```
CUDA_LAUNCH_BLOCKING=1 bash ./script/move_data.sh  ./output/NeRF_Syn/bear/3dgs/  ./configs/object_inpaint/bear_new.json ./data/bear
```

Now we need to obtain the 2D inpainted reference views. Please select **one or more** reference view (in the example I provided, image IDs 1,6,31 are selected), and use any inpainting model you like (e.g., LAMA, SDXL-inpaint) to inpaint the corresponding rgb image in ```data/bear/inpaint_2d_unseen_mask/images``` and depth image in ```data/bear/inpaint_2d_unseen_mask/depth_removal```.  

After you obtained the inpainted images/depths, **replace their original file** in ```data/bear/inpaint_2d_unseen_mask/images``` and ```data/bear/inpaint_2d_unseen_mask/depth_removal```. 

And that's all for the data preperation.



For the 3D inpainting process, just run

```
CUDA_LAUNCH_BLOCKING=1 bash ./script/edit_object_inpaint_spin.sh  ./output/NeRF_Syn/bear/3dgs/  ./configs/object_inpaint/bear_new.json ./data/bear
```

## Citing
If you find our work useful, please consider citing:
```BibTeX
@inproceedings{huang20253d,
  title={3d gaussian inpainting with depth-guided cross-view consistency},
  author={Huang, Sheng-Yu and Chou, Zi-Ting and Wang, Yu-Chiang Frank},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={26704--26713},
  year={2025}
}
```

### Acknowledgement

The code base is derived from [GausianGrouping](https://github.com/lkeab/gaussian-grouping) and [Relightable 3D Gaussian](https://github.com/NJU-3DV/Relightable3DGaussian), please consider citing them if using our code. This code also contain some part of (https://github.com/leo-frank/diff-gaussian-rasterization-depth).

This project is supported in part by the Tron Future Tech
Inc. and the National Science and Technology Council via grant NSTC 113-2634-F-002-005 and NSTC 113-2640-E-002-003, and the Center of Data Intelligence: Technologies, Applications, and Systems, National Taiwan University (grant nos.114L900902, from the Featured Areas Research Center Program within the framework of the Higher Education Sprout Project by the Ministry of Education (MOE) of Taiwan). We also thank the National Center for High-performance Computing (NCHC) for providing computational and storage resources.



