import os
# Package imports
import torch
import numpy as np
import torch.nn.functional as F
import argparse
import shutil
import json
import time
import matplotlib.pyplot as plt
from torch.nn import init

# Local imports
import config
from model.scene_rep import JointEncoding
from datasets.dataset import get_dataset

# Multiprocessing imports
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
from mp_slam.tracker import Tracker
from mp_slam.mapper import Mapper
from mp_slam.slam import SLAM

class RemixFusion():
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = get_dataset(config)
        
        self.create_bounds()
        # update the camera intrinsics according to pre-processing config
        self.update_cam()
        
        num_kf = int(self.dataset.num_frames // self.config['mapping']['keyframe_every'] + 1)  
        self.model = JointEncoding(self.config, self.bounding_box, num_kf).to(self.device).share_memory()

        # init the model parameters
        for name, param in self.model.named_parameters():
            if 'rba' in name: #or 'decoder'
                # nn.init.uniform_(param, -0.1, 0.1)
                init.normal_(param, mean=0, std=0.0001)
        
        # set the start method for the multiprocessing
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        # init the SLAM system
        self.slam = SLAM(self.config, self.dataset, self.model, self.device)
        
        # init the tracker
        self.tracker = Tracker(self.config, self.slam, self.model, self.dataset, self.slam.est_c2w_data, self.slam.RO_c2w_data, self.slam.est_c2w_data_rel, self.slam.tracking_idx,  self.slam.mapping_idx, self.slam.tracking_stop_flag, self.slam.pose_gt, self.slam.update_local_MV,  self.slam.keyframeDatabase.all_fuse_pose, self.device)
        
        # init the mapper
        self.mapper = Mapper(self.config, self.slam, self.model)
        
        # first frame mapping
        batch = self.dataset[0]
        self.mapper.first_frame_mapping(batch, self.config['mapping']['first_iters'])
    
    def update_cam(self):
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)
        if 'crop_size' in self.config['cam']:
            crop_size = self.config['cam']['crop_size']
            sx = (crop_size[1]+self.config['cam']['crop_edge']*2) / self.config['cam']['W']
            sy = (crop_size[0]+self.config['cam']['crop_edge']*2) / self.config['cam']['H']
            self.config['cam']['fx'] = sx*self.config['cam']['fx']
            self.config['cam']['fy'] = sy*self.config['cam']['fy']
            self.config['cam']['cx'] = sx*self.config['cam']['cx'] 
            self.config['cam']['cy']  = sy*self.config['cam']['cy'] 
            self.config['cam']['W']  = crop_size[1] + self.config['cam']['crop_edge']*2
            self.config['cam']['H']  = crop_size[0] + self.config['cam']['crop_edge']*2

        # croping will change H, W, cx, cy, so need to change here
        if self.config['cam']['crop_edge'] > 0:
            self.config['cam']['H'] -= self.config['cam']['crop_edge']*2
            self.config['cam']['W'] -= self.config['cam']['crop_edge']*2
            self.config['cam']['cx'] -= self.config['cam']['crop_edge']
            self.config['cam']['cy'] -= self.config['cam']['crop_edge']
            
    def create_bounds(self):
        '''
        Get the pre-defined bounds for the scene
        '''
        self.bounding_box = torch.from_numpy(np.array(self.config['mapping']['bound'])).to(self.device)
        self.marching_cube_bound = torch.from_numpy(np.array(self.config['mapping']['marching_cubes_bound'])).to(self.device)
    
    def tracking(self, rank):
        while True:
            if self.mapping_first_frame[0] == 1:
                print('Start tracking')
                break
            time.sleep(0.5)
        
        self.tracker.run()
    
    def mapping(self, rank):
        self.mapper.run()
    
    def start(self):
        self.processes = []
        p = mp.Process(target=self.mapper.run, args=( ))
        p.start()
        self.processes.append(p)
 
            
    def wait_child_processes(self):
        for p in self.processes:
            p.join()
    
    def run(self):
        self.start()
        self.tracker.run()
        self.wait_child_processes()

if __name__ == '__main__':
            
    print("============================================================")
    print("🚀 RemixFusion is now starting...")
    print("============================================================")

    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    
    args = parser.parse_args()

    cfg = config.load_config(args.config)
    if args.output is not None:
        cfg['data']['output'] = args.output

    print("Saving config...")
    # create the output folder
    save_path = os.path.join(cfg["data"]["output"], cfg['data']['exp_name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save the config file
    with open(os.path.join(save_path, 'config.json'),"w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))

    slam = RemixFusion(cfg)

    slam.start()
    slam.tracker.run()

    slam.wait_child_processes()
