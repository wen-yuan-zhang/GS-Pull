<p align="center" />
<h1 align="center">Neural Signed Distance Function Inference through Splatting 3D Gaussians Pulled on Zero-Level Set</h1>

<p align="center">
    <a href="https://wen-yuan-zhang.github.io/"><strong>Wenyuan Zhang</strong></a>
    ·
    <a href="https://yushen-liu.github.io/"><strong>Yu-Shen Liu</strong></a>
    ·
    <a href="https://h312h.github.io/"><strong>Zhizhong Han</strong></a>
</p>
<h2 align="center">NeurIPS 2024</h2>
<h3 align="center"><a href="https://arxiv.org/abs/2410.14189">Paper</a> | <a href="https://wen-yuan-zhang.github.io/GS-Pull/">Project Page</a></h3>
<div align="center"></div>
<div align="center"></div>
<p align="center">
    <img src="media/overview.png" width="780" />
</p>

In this paper, we propose to seamlessly combine 3D Gaussians with the learning of neural SDFs. Our method provides a novel perspective to jointly learn 3D Gaussians and neural SDFs by more effectively using multi-view consistency and imposing geometry constraints.

# Preprocessed Datasets & Pretrained Meshes

Our preprocessed datasets are provided in [This link](https://drive.google.com/drive/folders/1I3mSRrQ6oMV5nlNkaUtXmLVfCX_z-iRS?usp=sharing).

Pretrained meshes are provided in [This link](https://drive.google.com/drive/folders/13Aao4i-j5bG7Ss-jf6cMBjiit_uFElP5?usp=sharing).

# Setup

## Installation

Clone the repository and create an anaconda environment called gspull using
```shell
git clone git@github.com:wen-yuan-zhang/GS-Pull.git
cd GS-Pull

conda create -n gspull python=3.10
conda activate gspull

conda install pytorch=1.13.0 torchvision=0.14.0 cudatoolkit=11.7 -c pytorch
conda install cudatoolkit-dev=11.7 -c conda-forge

pip install -r requirements.txt
```

To install the differentiable splatting kernel, use 
```shell
cd gaussian_splatting/submodules
pip install diff-gaussian-rasterization
pip install simple-knn
```

To install the C++ extensions for NeuralPull, use 
```shell
cd np_utils/extensions/chamfer_dist
python setup.py install
```

(Optional) To try training UDFs, install udf extraction extensions
```shell
cd custom_mc
python setup.py build_ext --inplace
```

# Training

To train a scene, firstly run original Gaussian Splatting for 7000 iterations
```shell
cd gaussian_splatting
python train.py -s <path to dataset> -m <path to output_dir> --iterations 7000
```
For example, to train scan24 of DTU dataset, use 
```shell
python train.py -s data/DTU/scan24 -m output/DTU/scan24 --iterations 7000
```
The default background color is black. To use white background, you need to add a '-w' argument.

Then train GS-Pull using 
```shell
cd ../
python train.py -s <path to dataset> -c <path to gs checkpoint> --output <path to output_dir>
```
For example, to continue training scan24 of DTU dataset, use
 ```shell
python train.py -s data/DTU/scan24 -c gaussian_splatting/output/DTU/scan24 --output output/DTU/scan24
```
Note that we will identify the scene name in training, so please ensure that the output directory ends with the exact scene name of the dataset.

# Mesh Extraction

To extract meshes from checkpoints, use 
```shell
python extract_mesh.py -s <path to dataset> -g <path to 3DGS checkpoint> -o <path to gspull checkpoint>
```
For example, to extract mesh of scan24 of DTU dataset, use 
```shell
python extract_mesh.py -s data/DTU/scan24/ -g gaussian_splatting/output/DTU/scan24/ -o output/DTU/scan24
```

# Evaluation

To evaluate DTU scenes, put the ground truth of DTU dataset under `data/`, and then use 
```shell
cd evaluation
python clean_eval_dtu_mesh.py  --datadir <path to DTU dataset> --expdir <path to checkpoint dir> --scan <scan id>
```
For example, to evaluate scan24, use 
```shell
python clean_eval_dtu_mesh.py --datadir ../data/DTU --expdir ../output/DTU/scan24 --scan 24
```

To evaluate Tanks and Temples scenes, follow the official evaluation scipts provided by [TNT dataset](https://github.com/isl-org/TanksAndTemples/tree/master/python_toolbox/evaluation). 



# Acknowledgements

This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [SuGaR](https://github.com/Anttwo/SuGaR), [Neural-Pull](https://github.com/mabaorui/NeuralPull) and [CAP-UDF](https://github.com/junshengzhou/CAP-UDF). We thank all the authors for their great repos.


# Citation

If you find our code or paper useful, please consider citing
```bibtex
@inproceedings{zhang2024gspull,
    title = {Neural Signed Distance Function Inference through Splatting 3D Gaussians Pulled on Zero-Level Set},
    author = {Wenyuan Zhang and Yu-Shen Liu and Zhizhong Han},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2024},
}
```

