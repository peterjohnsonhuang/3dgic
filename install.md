create conda env
```
conda create -n 3dgic python=3.10
conda activate 3dgic
```
install cudatoolkit, pytorch
```
conda install nvidia/label/cuda-12.8.1::cuda-toolkit
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```
install other dependencies
```
pip install imageio plyfile tqdm scipy matplotlib scikit-image scikit-learn ipywidgets opencv-python lpips
```
install other extension modules
```
pip install ./submodules/simple-knn #might need to clean first. Chatgpt is your good friend
pip install ./bvh #might need to clean first. Chatgpt is your good friend
pip install ./r3dg-rasterization
cd diff-gaussian-rasterization-depth
python setup.py install
cd ..
```

