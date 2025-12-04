import time
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.ROtracker import ROTracker
from model.utils import compute_loss, check_orthogonal, orthogonalize_rotation_matrix, orthogonalize_rotation_matrix_tolerate

import numpy as np
from torch.cuda.amp import autocast
import os

class Tracker():
    def __init__(self, config, SLAM, model, dataset, est_c2w_data,  RO_c2w_data, est_c2w_data_rel, tracking_idx, mapping_idx, tracking_stop_flag, pose_gt,  update_local_MV, all_fuse_pose, device) -> None:
        self.config = config
        self.slam = SLAM
        self.dataset = dataset
        self.est_c2w_data = est_c2w_data
        self.RO_c2w_data = RO_c2w_data
        self.est_c2w_data_rel = est_c2w_data_rel
        self.tracking_idx = tracking_idx
        self.mapping_idx = mapping_idx
        self.tracking_stop_flag = tracking_stop_flag
        self.update_local_MV = update_local_MV
        self.all_fuse_pose = all_fuse_pose
        self.share_model = model
        self.pose_gt = pose_gt
        self.data_loader = DataLoader(dataset, num_workers=self.config['data']['num_workers'])
        self.frames_num = len(self.data_loader)
        self.prev_mapping_idx = -1
        self.device = device
    
        self.RO_Tracker = ROTracker(config,self.dataset)

        self.all_poses = []
        

        
    def update_params(self):
        if self.mapping_idx[0] != self.prev_mapping_idx:
            self.model = copy.deepcopy(self.share_model).to(self.device)
            self.prev_mapping_idx = self.mapping_idx[0].clone()
    
    def freeze_model(self):
        '''
        Freeze the model parameters
        '''      
        for param in self.model.parameters():
            param.requires_grad = False
            
    
    def predict_current_pose(self, frame_id, constant_speed=True):
        '''
        Predict current pose from previous pose
        '''
        if frame_id == 1 or (not constant_speed):
            c2w_est_prev = self.est_c2w_data[frame_id-1].to(self.device)
            self.est_c2w_data[frame_id] = c2w_est_prev
            
        else:
            c2w_est_prev_prev = self.RO_c2w_data[frame_id-2].to(self.device)
            c2w_est_prev = self.RO_c2w_data[frame_id-1].to(self.device)
            c2w_est_prev_prev = c2w_est_prev_prev.cpu()
            c2w_est_prev_prev = torch.inverse(c2w_est_prev_prev).to(self.device)
            delta = c2w_est_prev@c2w_est_prev_prev.float()#.inverse()
            self.est_c2w_data[frame_id] = delta@c2w_est_prev
            
            self.est_c2w_data[frame_id][:3,:3]=torch.from_numpy(orthogonalize_rotation_matrix_tolerate(self.est_c2w_data[frame_id][:3,:3])).to(self.device)
        
        return self.est_c2w_data[frame_id]
    
    def tracking(self, batch, frame_id):
        """
        Perform camera pose tracking for the current frame using Robust Odometry (RO).
        
        Args:
            batch (dict): The data batch for the current frame, containing:
                - 'c2w': Ground truth camera pose [B, 4, 4]
                - 'rgb': RGB image [B, H, W, 3]
                - 'depth': Depth image [B, H, W, 1]
                - 'direction': Ray directions [B, H, W, 3]
            frame_id (int): The current frame index.
        """
        #c2w_gt = batch['c2w'][0].to(self.device)

        # Initialize the current pose estimate: use previous tracking result if configured, 
        # otherwise predict using constant speed/motion model.

        cur_c2w = self.predict_current_pose(frame_id, self.config['tracking']['const_speed'])
        
        # Compute Absolute Pose Error (APE) before tracking for later comparison/logging.
        ape_before = self.cal_ape_error(batch["c2w"].squeeze(), cur_c2w[:3, 3])

        # Run RO-based tracking; outputs pose as numpy, updated rgb and depth as numpy arrays.
        RO_pose_np, rgb_np, depth_np = self.RO_Tracker.do_tracking(cur_c2w, None, batch, self.device)
        self.RO_Tracker.RO_pose.append(RO_pose_np)
        cur_c2w = torch.from_numpy(RO_pose_np).float().to(self.device)
        
        # Update estimated and RO pose storage for the current frame.
        self.est_c2w_data[frame_id] = cur_c2w.detach().clone()
        self.RO_c2w_data[frame_id] = cur_c2w.detach().clone()
        
        # Keep a history of all estimated poses
        self.all_poses.append(self.est_c2w_data[frame_id].clone().detach().cpu())
        
        # For non-keyframes, store the pose relative to the corresponding keyframe.
        if frame_id % self.config['mapping']['keyframe_every'] != 0:
            kf_id = frame_id // self.config['mapping']['keyframe_every']
            kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
            c2w_key = self.RO_c2w_data[kf_frame_id]
            c2w_key = c2w_key.cpu()
            c2w_key = torch.inverse(c2w_key).to(self.device)
            delta = self.est_c2w_data[frame_id] @ c2w_key.float()
            self.est_c2w_data_rel[frame_id] = delta

        # Compute and optionally print the updated APE compared to before tracking update.
        self.cal_ape_error(
            batch["c2w"].squeeze(),
            self.est_c2w_data[frame_id][:3, 3],
            ape_before,
            print_flag=self.config["print_ape"],
            prefix="final"
        )

        # Run RO tracker post-processing (may update internal states, do cleanup, etc).
        self.RO_Tracker.post_processing(
            frame_id,
            self.est_c2w_data[frame_id].cpu().numpy(),
            rgb_np,
            depth_np,
            self.est_c2w_data
        )


            
    def cal_ape_error(self, gt, our_t, ape_before=None, print_flag=False, prefix="noname"):
        """
        Calculate the Absolute Pose Error (APE) between ground truth and estimated translation.

        Args:
            gt: Ground truth pose. Could be a torch.Tensor or numpy array, shape (4,4) or similar.
            our_t: Estimated translation vector or pose. Could be a torch.Tensor or numpy array.
            ape_before (float, optional): APE value before update, used for logging improvement.
            print_flag (bool, optional): If True, print the APE computation and success/failure.
            prefix (str, optional): Message prefix for logging output.

        Returns:
            float: The mean absolute error between the translation components of gt and our_t.
        """
        # Handle the case when both inputs are numpy arrays
        if not isinstance(gt, torch.Tensor) and not isinstance(our_t, torch.Tensor):
            ape = np.average(np.abs(gt[:3,3] - our_t))
            return ape
        # Convert gt to tensor if it's not
        if not isinstance(gt, torch.Tensor):
            gt = torch.from_numpy(gt).float()
        # Move gt to correct device if prediction is on CUDA
        if our_t.is_cuda:
            gt = gt.to(self.device)
        # Calculate mean absolute error for translation components
        ape = torch.abs(gt[:3,3] - our_t).mean()
        if not print_flag:
            return ape.item()
        # If print_flag is set and previous APE provided, print improvement/failure
        elif ape_before is not None:
            if ape < ape_before:
                print(prefix, f"success ape: {ape_before:.6f}->{ape:.6f}")
            else:
                print(prefix, f"fail ape: {ape_before:.6f}->{ape:.6f}")
    
    def run(self):
        """
        Main tracking loop for pose estimation and frame processing.
        """
        print("******* tracking process started! *******")
        for idx, batch in tqdm(enumerate(self.data_loader), total=len(self.data_loader)):
            if idx == 0:
                # For the first frame, save the initial pose and initialize RO poses.
                self.all_poses.append(self.est_c2w_data[0].detach().cpu())
                self.RO_c2w_data[0] = self.est_c2w_data[0].detach().clone() 
                continue

            # Wait for the mapping process to reach the required index to ensure tracking and mapping are synchronized.
            while self.mapping_idx[0] < idx - self.config['mapping']['map_every'] - self.config['mapping']['map_every'] // 2:
                time.sleep(0.02)

            # Perform tracking for the current frame.
            self.tracking(batch, idx)
  
            # Update the tracking index.
            self.tracking_idx[0] = idx

        # print('tracking finished') 
        # Indicate that the tracking process has finished.
        self.tracking_stop_flag[0] = 1
        

        
        


            

