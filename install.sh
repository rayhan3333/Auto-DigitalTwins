conda create -n videoartgs python=3.10 -y
conda activate videoartgs
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 "xformers>=0.0.27" --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
pip install -r requirements.txt

# install pytorch3d and tiny-cuda-nn
pip install git+https://github.com/facebookresearch/pytorch3d.git@stable --no-build-isolation
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation

# build pointnet_lib for nearest farthest point sampling
cd utils/pointnet_lib
python setup.py install
cd ../..

# compile pointops2
cd third_party/TAPIP3D/third_party/pointops2
LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH python setup.py install
cd ../../../..

# gslpat
pip install git+https://github.com/nerfstudio-project/gsplat.git

# simple-knn
pip install git+https://gitlab.inria.fr/bkerbl/simple-knn.git --no-build-isolation

