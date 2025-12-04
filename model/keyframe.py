import torch
import numpy as np
import random

class KeyFrameDatabase(object):
    def __init__(self, config, H, W, num_kf, num_rays_to_save, device, num_frame=None) -> None:
        self.config = config
        self.keyframes = {}
        self.device = device
        self.rays = torch.zeros((num_kf, num_rays_to_save, 7))
        self.num_rays_to_save = num_rays_to_save
        self.frame_ids = None
        self.H = H
        self.W = W
        self.kf_poses = torch.zeros((num_kf,4,4))
        self.kf_fuse_poses = torch.zeros((num_kf,4,4))
        self.kf_error = torch.zeros((num_kf)).to(device)
        self.kf_error_cnt = torch.zeros((num_kf)).to(device)
        if num_frame is not None:
            self.all_fuse_pose = torch.zeros((num_frame,4,4)).to(device).share_memory_()
    
    def __len__(self):
        return len(self.frame_ids)
    
    def get_length(self):
        return self.__len__()
    
    def sample_single_keyframe_rays(self, rays, option='random',first=False):
        '''
        Sampling strategy for current keyframe rays
        '''
        if option == 'random':
            idxs = random.sample(range(0, self.H*self.W), self.num_rays_to_save)
        elif option == 'filter_depth':
            valid_depth_mask = (rays[..., -1] > 0.0) & (rays[..., -1] <= self.config["cam"]["depth_trunc"])
            rays_valid = rays[valid_depth_mask, :]  # [n_valid, 7]
            num_valid = len(rays_valid)
            if num_valid>self.num_rays_to_save:
                idxs = random.sample(range(0, num_valid), self.num_rays_to_save)
            else:
                idxs = random.sample(range(0, self.H*self.W), self.num_rays_to_save)
                option == "random"
        else:
            raise NotImplementedError()
        if option == "random" or first:
            rays = rays[:, idxs]
        else:
            rays = rays_valid[idxs,:]
        return rays
    
    def attach_ids(self, frame_ids):
        '''
        Attach the frame ids to list
        '''
        if self.frame_ids is None:
            self.frame_ids = frame_ids
        else:
            self.frame_ids = torch.cat([self.frame_ids, frame_ids], dim=0)
    
    def add_keyframe(self, batch, filter_depth=False):
        '''
        Add keyframe rays to the keyframe database
        '''
        # batch direction (Bs=1, H*W, 3)
        first=False
        if batch['frame_id'] == 0:
            first=True
        rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        rays = rays.reshape(1, -1, rays.shape[-1])
        if filter_depth:
            rays = self.sample_single_keyframe_rays(rays, 'filter_depth',first=first)
        else:
            rays = self.sample_single_keyframe_rays(rays,first=first)
        
        if not isinstance(batch['frame_id'], torch.Tensor):
            batch['frame_id'] = torch.tensor([batch['frame_id']])

        self.attach_ids(batch['frame_id'])

        # Store the rays
        self.rays[len(self.frame_ids)-1] = rays
    
    def sample_global_rays(self, bs):
        '''
        Sample rays from self.rays as well as frame_ids
        '''
        num_kf = self.__len__()
        
        idxs = torch.tensor(random.sample(range(num_kf * self.num_rays_to_save), bs))
            
        sample_rays = self.rays[:num_kf].reshape(-1, 7)[idxs]

        frame_ids = self.frame_ids[idxs//self.num_rays_to_save]


        return sample_rays, frame_ids
    
    