import os
import warnings
import torch
import argparse
import cv2
import config
from mp_slam.slam import SLAM
from mp_slam.mapper import Mapper
from imageio import imwrite
import numpy as np
from datasets.dataset import get_dataset
from model.scene_rep import JointEncoding



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Arguments to render depth and rgb images.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--ckpt_path', type=str,
                        help='ckpt_path')
    parser.add_argument('--pose_id_list', type=str,
                        help='pose_id_list, whith image to render')
    nice_parser = parser.add_mutually_exclusive_group(required=False)

    nice_parser.add_argument('--pose_path', type=str, default=None, help='do not use gt pose, use your own pose')

    args = parser.parse_args()

    cfg = config.load_config(args.config)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


    #create model
    bounding_box = torch.from_numpy(np.array(cfg['mapping']['bound'])).to("cuda:0")
    frame_reader = get_dataset(cfg)
    #load model
    num_kf = int(frame_reader.num_frames // cfg['mapping']['keyframe_every'] + 1)  
    model = JointEncoding(cfg, bounding_box, num_kf).to("cuda:0")

    ckpt_path = args.ckpt_path   
    load_ckpt = torch.load(ckpt_path)
    model.load_state_dict(load_ckpt['model'])

    if args.pose_path is not None:
        poses = np.load(args.pose_path)

    bounding_box = torch.from_numpy(np.array(cfg['mapping']['bound'])).to("cuda:0")
    marching_cube_bound = torch.from_numpy(np.array(cfg['mapping']['marching_cubes_bound'])).to("cuda:0")
    slam = SLAM(cfg, frame_reader, model, "cuda:0")
    mapper = Mapper(cfg, slam, model)

    mapper.calc_2d_metric(poses, gap=10, save=False)

