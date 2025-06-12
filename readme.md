
#### Install dependencies
for all the packages, please see requirements.txt

#### Install additional pytorch extensions


```shell
# install knn-cuda
pip install ./submodules/simple-knn

# install bvh
pip install ./bvh

# install relightable 3D Gaussian
pip install ./r3dg-rasterization
```

## :bookmark_tabs: Todos
We will be releasing all the following contents:
- [x] Training and inference code for 3DGIC
- [ ] Provide example of the Bear dataset
- [ ] Demo for relighting


### Running
We run the code in a single NVIDIA GeForce RTX 3090 GPU (24G). we will provide almost everything for the bear dataset so you can take a look how to put files. 

If you want to train the 3dgs model from scratch, please run 
:
```
sh script/run_bear.sh
```

for directly inpaint the bear from our provided 3dgs for the bear scene, please run

```
CUDA_LAUNCH_BLOCKING=1 bash ./script/edit_object_inpaint_spin.sh  ./output/NeRF_Syn/bear_0823/3dgs/  ./configs/object_inpaint/bear_new.json
```





