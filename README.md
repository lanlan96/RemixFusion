<div align="center">
<h1>RemixFusion: Residual-Based Mixed Representation for Large-Scale Online RGB-D Reconstruction</h1>

ACM Transactions on Graphics (to be presented at SIGGRAPH Asia 2025)

<a href="https://arxiv.org/pdf/2507.17594"><img src="https://img.shields.io/badge/arXiv-2507.17594-b31b1b" alt="arXiv"></a>
<a href="https://dl.acm.org/doi/abs/10.1145/3769007" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Paper-VGGT" alt="Paper PDF">
</a>
<a href="https://lanlan96.github.io/RemixFusion/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://huggingface.co/datasets/Kevin1804/RemixFusion"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a>

[Yuqing Lan](https://scholar.google.com/citations?user=laTrw7AAAAAJ&hl=en&oi=ao), [Chenyang Zhu](https://www.zhuchenyang.net/), [Shuaifeng Zhi](https://shuaifengzhi.com/), [Jiazhao Zhang](https://jzhzhang.github.io/), [Zhoufeng Wang](https://github.com/yhanCao), [Renjiao Yi](https://renjiaoyi.github.io/), [Yijie Wang](https://ieeexplore.ieee.org/author/37540196000), [Kai Xu](https://kevinkaixu.net/)
</div>

##  News
- **2025-12-01**: Codes are released.
- **2025-07-17**: The arxiv paper is online.


## 1. Installation

Please create the virtual environment with python3.7 and a 1.13.1 build of PyTorch. The code has been tested on Ubuntu22.04 and CUDA 11.6. The environment can be created like:

```
conda create -n remix python=3.7
conda activate remix
```
Install PyTorch:
```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
Then you can install the dependencies:
```
pip install -r requirements.txt

cd thirdparty/NumpyMarchingCubes
python setup.py install
cd ../../
```
Notes: If you have networks issues when installing libraries like `tiny-cuda-nn`, please try build from source following the original [repository](https://github.com/NVlabs/tiny-cuda-nn?tab=readme-ov-file#pytorch-extension)

## 2. Quick Start
1.Download the [example data](https://drive.google.com/file/d/1kuvn8FejoeU2QU5nKKYspmJDOagQm3OS/view?usp=drive_link) from google drive. 



2.Move it into your local directory and unzip the data. Remember to change the `datadir` in the `configs/study_example.yaml` to your local path.

3.Run the demo.py using the example data for a quick start.
```
python run.py --config ./configs/BS3D/study_example.yaml
```



## 3. Data Preparation
We first introduce the preparation of two indoor large-scale SLAM datasets ([BS3D](https://github.com/jannemus/BS3D) and [uHumans2](https://web.mit.edu/sparklab/datasets/uHumans2/)). Basically, we organize the data like most SLAM datasets (e.g. ScanNet). Then, we introduce other common datasets ([ScanNet](http://www.scan-net.org/), [Replica](https://github.com/facebookresearch/Replica-Dataset), [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)). 

### 3.1 (Optional) Preprocess data of BS3D and uHumans2
Download the raw sequence data from the official websites, and associate the RGB-D pose data via corresponding timestamps. Here, we take the BS3D dataset as an example. You can use the similar procedure for other datasets.

1.The raw data of large-scale benchmark BS3D can be obtained from this [link](https://etsin.fairdata.fi/dataset/3836511a-29ba-4703-98b6-40e59bb5cd50). Please download the specific sequences in the config file. 

2.Use the command to associate RGB-D images and output the organized data.
```
python preprocess/preprocess.py /PATH/BS3D/SCENE_NAME
```

### 3.2 (Recommended) Download Preprocessed Data

### BS3D and uHumans2
We have uploaded the proprocessed sequences to the [hugging face](https://huggingface.co/datasets/Kevin1804/RemixFusion/tree/main/). You can just download the preprocessed datasets (BS3D and uHumans2). First, you need to install `huggingface_hub`
```
pip install -U huggingface_hub
```


Download the datasets using the following command. If you encouter problems of connecting to the hugging face, please search the solutions on your own. Remember to change the `local-dir` to the directory you want. The data takes up to about `27GB`.
```
hf download --repo-type dataset Kevin1804/RemixFusion --local-dir /media/lyq/temp/test/
```

The data structure for BS3D is like this:

<details>
<summary>[Structure for BS3D dataset (click to expand)]</summary>

```
BS3D/
├── cafeteria/                
│   ├── depth/               # Folder containing depth images
│   ├── color/               # Folder containing RGB images
│   ├── all_poses.npy        # All camera extrinsics [N,4,4]
│   ├── poses.txt            # All camera extrinsics [N,8(t,tx,ty,tz,qx,qy,qz,qw)]
│   ├── mesh.ply             # GT LiDAR mesh (w/o color)
│   ├── cafeteria_gt.ply     # GT reconstructed mesh using GT poses
│   ├── cameras.json         # Camera information
│   ├── calibration.yaml     # Calibration details
│   ├── color.txt            # All RGB files path
│   ├── depth.txt            # All Depth files path
```
</details>


The data structure for uHumans2 is like this:

<details>
<summary>[Structure for uHumans2 dataset (click to expand)]</summary>

```
uHumans2/
├── apartment/                
│   ├── depth/               # Folder containing depth images
│   ├── color/               # Folder containing RGB images
│   ├── poses.txt            # All camera extrinsics [N,8(t,tx,ty,tz,qx,qy,qz,qw)]
│   ├── cameras.json         # Camera information
│   ├── associations.yaml    # Association of RGB-D raw data
│   ├── color.txt            # All RGB files path
│   ├── depth.txt            # All Depth files path
```
</details>

### ScanNet
Please follow the procedure on the official [ScanNet](http://www.scan-net.org/) website, and extract RGB-D frames from the `.sens` file using the [code](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py).

The data structure for ScanNet is like this:
<details>
<summary>[Structure for ScanNetV2 dataset (click to expand)]</summary>

```
ScanNet/
├── scene0xxx_0x/
│   ├── frames/
│   │   ├── color/
│   │   │   ├── 0.jpg
│   │   │   ├── 1.jpg
│   │   │   └── ...
│   │   ├── depth/
│   │   │   ├── 0.png
│   │   │   ├── 1.png
│   │   │   └── ...
│   │   ├── pose/
│   │   │   ├── 0.txt
│   │   │   ├── 1.txt
│   │   │   └── ...
│   │   ├── intrinsic/
│   │   │   └── intrinsic_depth.txt
│   └── scene0xxx_0x.txt
│   └── scene0xxx_0x_vh_clean_2.ply
└── ...
```
</details>

### Replica
Download the data using the command (provided by [NICE-SLAM](https://github.com/cvg/nice-slam)) and the data is saved into the `./data/Replica` folder. 
```bash
bash scripts/download_replica.sh
```

The mesh for evaluation is saved as `$OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply`, where the unseen regions are culled using all frames.


### TUM RGB-D
Download the data using the command and the data is saved into the `./data/TUM` folder
```bash
bash scripts/download_tum.sh
```


### Other Datasets
For other datsets (e.g. [FastCaMo](https://drive.google.com/drive/folders/186viK0tSAFVDO_6YJbC3MGXOXzcBQT_z), [FastCaMo-large](https://drive.google.com/drive/folders/186viK0tSAFVDO_6YJbC3MGXOXzcBQT_z) and self captured datasets), please download the data and prepare the data following the procedure mentioned above. 




## 4. Run
In this section, we introduce how to run RemixFusion on different datasets. After you have prepared the datasets according to the instructions above, you can run the following commands to try RemixFusion on the specific sequence.
### BS3D
Please change the `datadir` in the `config/SEQUENCE.yaml` to the root of the processed BS3D sequence. Customize the `--config` to the sequence you want to try.
```
python run.py --config ./configs/BS3D/foobar.yaml
```
### uHumans2
Please change the `datadir` in the `config/SEQUENCE.yaml` to the root of the processed uHumans2 sequence. Customize the `--config` to the sequence you want to try.
```
python run.py --config ./configs/uhumans/apartment.yaml
```
### Replica
Please change the `datadir` in the `config/SEQUENCE.yaml` to the root of the processed Replica sequence. Customize the `--config` to the sequence you want to try.
```
python run.py --config ./configs/Replica/room0.yaml
```
### ScanNet
Please change the `datadir` in the `config/SEQUENCE.yaml` to the root of the processed ScanNet sequence. Customize the `--config` to the sequence you want to try.
```
python run.py --config ./configs/ScanNet/scene0169.yaml
```
### TUM RGB-D
Please change the `datadir` in the `config/SEQUENCE.yaml` to the root of the processed TUM RGB-D sequence. Customize the `--config` to the sequence you want to try.
```
python run.py --config ./configs/Tum/fr1_desk.yaml
```
### FastCaMo-synth
Please change the `datadir` in the `config/SEQUENCE.yaml` to the root of the processed FastCaMo-synth sequence. Customize the `--config` to the sequence you want to try.
```
python run.py --config ./configs/Fast_syn/apartment_1.yaml
```
## 5. Evaluation
### 5.1 Average Trajectory Error
To evaluate the average trajectory error, you can just check the output directory which contains the trajectory results. You can also run the command below with the corresponding pose file. An example is like this:
```bash
python tools/eval_ate.py --est /PATH/YOUR_RESULTS/***.npy --gt /PATH/YOUR_DATASET/***.npy
```

### 5.2 Reconstruction Error

The evaluation of reconstructed mesh consists of two steps.
1. (Step1) We cull the GT meshes using the ground-truth camera poses and depth information. Since the GT meshes are reconstructed from LiDAR scans, they contain large areas that were not observed in the RGB-D images.
```
bash tools/mesh_cull_gt.sh
```
2. (Step2) Use the estimated camera poses and GT depth images to cull the reconstructed mesh. GT depth images are used because we observe that some methods perform bad in depth rendering. Use the rendered depth for culling the mesh will cause severe empty holes. Frustum-based culling, used by previous methods, is not adopted here, as it will reserve a large amount of noisy geometry in empty space.
```
bash tools/mesh_cull_est.sh
```
3. (Step3) Use the following command to evaluate the culled mesh and GT mesh.
```
bash tools/mesh_eval.sh
```
Note: If you only want to evaluate one specific scene, you can modify the command in the above script.
### 5.3 Rendering Error
To evaluate the rendering results, take the following command as an example to output the rendering metrics. Remember to change the `save_ckpt=True` (default: False) for rendering.
```
bash tools/rendering.sh
```

## Acknowledgement
Parts of the codes are modified from [Co-SLAM](https://github.com/apple/ml-cubifyanything), [NICE-SLAM](https://github.com/cvg/nice-slam), [GS-ICP SLAM](https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM), [SplaTAM](https://github.com/spla-tam/SplaTAM). Great Thanks to the authors.


## Citation
If you find our work useful in your research, please consider giving a star ✨ and citing the following paper:
```
@article{lan2025remixfusion,
  title={RemixFusion: Residual-based Mixed Representation for Large-scale Online RGB-D Reconstruction},
  author={Lan, Yuqing and Zhu, Chenyang and Zhi, Shuaifeng and Zhang, Jiazhao and Wang, Zhoufeng and Yi, Renjiao and Wang, Yijie and Xu, Kai},
  journal={ACM Transactions on Graphics},
  volume={45},
  number={1},
  pages={1--19},
  year={2025},
  publisher={ACM New York, NY}
}
```