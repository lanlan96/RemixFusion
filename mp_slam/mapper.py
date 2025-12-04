import torch
import time
import os
import random
from model.utils import compute_loss, check_orthogonal, orthogonalize_rotation_matrix, orthogonalize_rotation_matrix_tolerate
from model.keyframe import KeyFrameDatabase
import tinycudann as tcnn
import numpy as np
from tqdm import tqdm
import cv2
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from imageio import imwrite
from utils import mse2psnr, ssim
from tools.cull_mesh import cull_one_mesh

try:
  import pycuda.driver as cuda
  import pycuda.autoprimaryctx
  from pycuda.compiler import SourceModule
  import pycuda.gpuarray as gpuarray
  GPU_MODE = 1
except Exception as err:
  print('Warning: {}'.format(err))
  print('Failed to import PyCUDA. Running fusion in CPU mode.')
  GPU_MODE = 0
from torch.cuda.amp import autocast
  
class Holder(cuda.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()
    def get_pointer():
        return self.t.data_ptr()

map_src_mod = SourceModule("""
            __global__ void integrate(float * trgbw_vol,
                                    float * weight_vol,
                                    float * vol_dim,
                                    float * cam_intr,
                                    float * cam_pose,
                                    float * other_params,
                                    float * color_im,
                                    float * depth_im) {
            // Get voxel index
            int gpu_loop_idx = (int) other_params[0];
            int max_threads_per_block = blockDim.x;
            
            // Skip if outside view frustum
            float voxel_size = other_params[1];
            int im_h = (int) other_params[2];
            int im_w = (int) other_params[3];
            float trunc_margin = other_params[4];
            float obs_weight = other_params[5];
            float x_start = other_params[6];
            float x_end = other_params[7];
            float y_start = other_params[8];
            float y_end = other_params[9];
            float z_start = other_params[10];
            float z_end = other_params[11];
            
            int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
            int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;
            int vol_dim_x = (int) vol_dim[0];
            int vol_dim_y = (int) vol_dim[1];
            int vol_dim_z = (int) vol_dim[2];


            if (voxel_idx >= vol_dim_x*vol_dim_y*vol_dim_z){
                return;
            }
               
            float voxel_z = floorf(((float)voxel_idx)/((float)(vol_dim_x*vol_dim_y)));
            float voxel_y = floorf(((float)(voxel_idx-((int)voxel_z)*vol_dim_x*vol_dim_y))/((float)vol_dim_x));
            float voxel_x = (float)(voxel_idx-((int)voxel_z)*vol_dim_x*vol_dim_y-((int)voxel_y)*vol_dim_x);

            
            float pt_x = x_start+((voxel_x)*voxel_size)*(x_end-x_start);
            float pt_y = y_start+((voxel_y)*voxel_size)*(y_end-y_start);
            float pt_z = z_start+((voxel_z)*voxel_size)*(z_end-z_start);

            // World coordinates to camera coordinates
            float tmp_pt_x = pt_x-cam_pose[0*4+3];
            float tmp_pt_y = pt_y-cam_pose[1*4+3];
            float tmp_pt_z = pt_z-cam_pose[2*4+3];
            float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
            float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
            float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;
            
            if (cam_pt_z<=0){
                return;
            }
            
            // Camera coordinates to image pixels
            int pixel_x = __float2int_rn(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
            int pixel_y = __float2int_rn(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);
      
            if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h)
                return;
            
            float depth_value = depth_im[pixel_y*im_w+pixel_x];
            
            // Skip invalid depth
            if (depth_value <= 0)
                return;
            
            // Integrate TSDF
            float vec_x = (((float)pixel_x) - cam_intr[0*3+2])/cam_intr[0*3+0];
            float vec_y = (((float)pixel_y) - cam_intr[1*3+2])/cam_intr[1*3+1];
            float lambda = sqrt(vec_x*vec_x + vec_y*vec_y+1);
            float cam_norm = sqrt(cam_pt_x*cam_pt_x+cam_pt_y*cam_pt_y+cam_pt_z*cam_pt_z);
            
            float depth_diff = (-1.f) * ((1.f / lambda) * cam_norm - depth_value);
            //float depth_diff = depth_value-cam_pt_z;
            
            if (depth_diff < -1*trunc_margin){
                return;
            }
            
            float dist = fmin(1.0f,depth_diff/trunc_margin);
            float w_old = weight_vol[voxel_idx];
            float w_new = w_old + obs_weight;
            
            float new_tsdf=(trgbw_vol[voxel_idx*4]*w_old+obs_weight*dist)/w_new; 

            if (obs_weight <0 && w_old<=1) {
                trgbw_vol[voxel_idx*4] = 1.0;
                weight_vol[voxel_idx] = 0.0;
                trgbw_vol[voxel_idx*4+1] = 0.0;
                trgbw_vol[voxel_idx*4+2] = 0.0;
                trgbw_vol[voxel_idx*4+3] = 0.0;
                return;
            }
            
            if (new_tsdf>1.0){
                return;
            }
            
            trgbw_vol[voxel_idx*4] = new_tsdf;
                
            //Integrate colordist
            float old_b = trgbw_vol[voxel_idx*4+3];
            float old_g = trgbw_vol[voxel_idx*4+2];
            float old_r = trgbw_vol[voxel_idx*4+1];
            float new_b = color_im[(pixel_y*im_w+pixel_x)*3+2];
            float new_g = color_im[(pixel_y*im_w+pixel_x)*3+1];
            float new_r = color_im[(pixel_y*im_w+pixel_x)*3];

    
            new_b = fmin((old_b*w_old+obs_weight*new_b)/w_new,1.0f);
            new_g = fmin((old_g*w_old+obs_weight*new_g)/w_new,1.0f);
            new_r = fmin((old_r*w_old+obs_weight*new_r)/w_new,1.0f);
            trgbw_vol[voxel_idx*4+1] = new_r;
            trgbw_vol[voxel_idx*4+2] = new_g;
            trgbw_vol[voxel_idx*4+3] = new_b;
                
            weight_vol[voxel_idx] = w_new;
            } 
            
            
            __global__ void clean_tsdf(float * tsdf_vol,
                                        float * vol_dim,
                                        float * other_params
                                    ) {
            // Get voxel index
            int gpu_loop_idx = (int) other_params[0];
            int max_threads_per_block = blockDim.x;

            int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
            int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;
            int vol_dim_x = (int) vol_dim[0];
            int vol_dim_y = (int) vol_dim[1];
            int vol_dim_z = (int) vol_dim[2];

            if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z){
                return;
            }   
            tsdf_vol[voxel_idx*4] = 1.0;
            tsdf_vol[voxel_idx*4+1] = 0.0;
            tsdf_vol[voxel_idx*4+2] = 0.0;
            tsdf_vol[voxel_idx*4+3] = 0.0;

            }
            
            """)
       
map_integrate = map_src_mod.get_function("integrate")
cuda_clean_tsdf = map_src_mod.get_function("clean_tsdf")


class Mapper():
    def __init__(self, config, SLAM, model) -> None:
        self.config = config
        self.slam = SLAM
        self.model = model
        self.tracking_idx = SLAM.tracking_idx
        self.mapping_idx = SLAM.mapping_idx
        self.mapping_first_frame = SLAM.mapping_first_frame
        self.tracking_stop_flag = SLAM.tracking_stop_flag
        self.keyframe = SLAM.keyframeDatabase
        self.map_optimizer = SLAM.map_optimizer
        self.rba_optimizer = SLAM.rba_optimizer
        self.device = SLAM.device
        self.dataset = SLAM.dataset
        self.est_c2w_data = SLAM.est_c2w_data
        self.RO_c2w_data = SLAM.RO_c2w_data
        self.est_c2w_data_rel = SLAM.est_c2w_data_rel
        self.update_local_MV = SLAM.update_local_MV
        self.first_BA=True

        self.create_global_volume(self.config["globalV"]["base_resolution"])

    def create_global_volume(self, base_resolution):
        """
        Initialize the global volumetric map for mapping.
        Sets up the volume dimensions, voxel size, volume origin, bounding box, and camera intrinsics.
        Also prepares CUDA GPU parameters for volume processing and grid/block setup.
        
        Args:
            base_resolution (int): Number of voxels per axis for the cube-shaped volume.
        """
        # Set volume dimensions and properties
        self.vol_dim = np.array([base_resolution, base_resolution, base_resolution])
        self.map_box = self.config['mapping']['bound']
        self.voxel_size = 1.0 / base_resolution
        self.vol_origin = np.array([self.map_box[0][0], self.map_box[1][0], self.map_box[2][0]])
        self.box_length = np.array([
            self.map_box[0][1] - self.map_box[0][0],
            self.map_box[1][1] - self.map_box[1][0],
            self.map_box[2][1] - self.map_box[2][0]
        ])
        # Set camera intrinsic matrix
        self.K = np.array([
            [self.dataset.fx, 0, self.dataset.cx],
            [0, self.dataset.fy, self.dataset.cy],
            [0, 0, 1]
        ])
        # Initialize CUDA device and block/grid dimensions for GPU processing
        gpu_dev = cuda.Device(0)
        self.max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
        n_blocks = int(np.ceil(float(np.prod(self.vol_dim)) / float(self.max_gpu_threads_per_block)))
        grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X, int(np.floor(np.cbrt(n_blocks))))
        grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y, int(np.floor(np.sqrt(n_blocks / grid_dim_x))))
        grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z, int(np.ceil(float(n_blocks) / float(grid_dim_x * grid_dim_y))))
        self.max_gpu_grid_dim = np.array([grid_dim_x, grid_dim_y, grid_dim_z]).astype(int)
        # print("map block:", self.max_gpu_threads_per_block)
        # print("map grid:", self.max_gpu_grid_dim)
        self.n_gpu_loops = int(np.ceil(
            float(np.prod(self.vol_dim)) /
            float(np.prod(self.max_gpu_grid_dim) * self.max_gpu_threads_per_block)
        ))
        # Set SDF truncation margin
        self.trunc_margin = self.config["training"]["c_trunc"]
        # print("voxel size: ", self.voxel_size)
        
        
    def save_ckpt(self, save_path):
        '''
        Save the model parameters and the estimated pose
        '''
        save_dict = {'pose': self.est_c2w_data,
                     'pose_rel': self.est_c2w_data_rel,
                     'model': self.model.state_dict()}
        torch.save(save_dict, save_path)
        print(f'Save the checkpoint at {save_path}')
    
    def init_mapvolume(self):
        
        for gpu_loop_idx in range(self.n_gpu_loops):
            cuda_clean_tsdf(
                Holder(self.model.GBV.params),
                cuda.In(self.vol_dim.astype(np.float32)),
                cuda.In(np.asarray([
                                gpu_loop_idx,
                                ], np.float32)),
                block=(self.max_gpu_threads_per_block,1,1),
                grid=(
                    int(self.max_gpu_grid_dim[0]),
                    int(self.max_gpu_grid_dim[1]),
                    int(self.max_gpu_grid_dim[2]),
                )
        )
    
    def first_frame_mapping(self, batch, n_iters=100):
        """
        Perform mapping on the very first frame of the sequence. This function initializes
        the map volume, integrates the first keyframe, and optimizes the mapping network
        based on the input batch for a given number of iterations.

        Args:
            batch (dict): A dictionary containing:
                - 'c2w': Camera-to-world pose matrix, shape [1, 4, 4]
                - 'rgb': RGB image tensor, shape [1, H, W, 3]
                - 'depth': Depth image tensor, shape [1, H, W, 1]
                - 'direction': Ray directions, shape [1, H, W, 3]
            n_iters (int): Number of optimization iterations for mapping (default: 100)

        Returns:
            ret (dict): Output dictionary from the model mapping step
            loss (float): Final loss value after optimization

        Raises:
            ValueError: If called with a frame that is not the first frame (frame_id != 0)
        """
        print('First frame mapping...')
        if batch['frame_id'] != 0:
            raise ValueError('First frame mapping must be the first frame!')
        
        c2w = batch['c2w'].to(self.device)
        # Modify camera pose for specific datasets to face the positive X axis
        if self.config["dataset"] == "Largeindoor": 
            c2w = torch.tensor([[ 0.,  0. , 1. , 0.],
                                [-1.  ,0.,  0.,  0.],
                                [ 0. ,-1.,  0.,  0.],
                                [ 0. , 0.,  0. , 1.]]).to(self.device)
        if self.config["dataset"] == "uhumans":
            temp_pose = torch.tensor([[ 0.,  0. , 1. , 0.],
                                      [-1.  ,0.,  0.,  0.],
                                      [ 0. ,-1.,  0.,  0.],
                                      [ 0. , 0.,  0. , 1.]]).to(self.device)
            c2w[:3,:3] = temp_pose[:3,:3]
        
        # Initialize map volume and integrate the first keyframe
        self.init_mapvolume()
        self.integrate_kf(batch, c2w)
        
        # Save pose information for the first frame
        self.est_c2w_data[0] = c2w
        self.est_c2w_data_rel[0] = c2w
        self.model.rba.update_init_pose(0, c2w.cuda())
        
        self.model.train()

        # Perform mapping optimization for n_iters iterations
        for i in range(n_iters):
            self.map_optimizer.zero_grad()
            indice = self.slam.select_samples(self.slam.dataset.H, self.slam.dataset.W, self.config['mapping']['sample'])
            indice_h, indice_w = indice % (self.slam.dataset.H), indice // (self.slam.dataset.H)
            rays_d_cam = batch['direction'][indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'][indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'][indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            # Forward mapping step and loss computation
            ret = self.model.mapping(rays_o.to(self.device), rays_d.to(self.device), target_s, target_d)
            loss = self.slam.get_loss_from_ret(ret).to(self.device)
            loss.backward()
            self.map_optimizer.step()
           
        # The very first frame is always set as a keyframe
        self.keyframe.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
        
        # self.slam.save_mesh(0, voxel_size=0.03)
        # exit(0)
        
        if self.config['mapping']['first_mesh']:
            self.slam.save_mesh(0)
            self.slam.render_img(0, batch["depth"], batch["rgb"], self.est_c2w_data[0], batch["direction"])

        print('First frame mapping done')
        self.mapping_first_frame[0] = 1
        return ret, loss
    
    def global_mapping(self, batch, cur_frame_id):
        '''
        Global mapping that includes all the keyframes and the current frame
        Params:
            batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W, 1]
            batch['direction']: view direction [1, H, W, 3]
            cur_frame_id: current frame id 
        '''
        poses = torch.stack([self.est_c2w_data[i] for i in range(0, cur_frame_id+1, self.config['mapping']['keyframe_every'])])
        
        # Set up optimizer
        self.map_optimizer.zero_grad()
        self.rba_optimizer.zero_grad()

        current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1) #H,W,7
        current_rays = current_rays.reshape(-1, current_rays.shape[-1]) #H*W,7
        
        with torch.no_grad():
            last_kf_id = ((torch.tensor(cur_frame_id)//self.config['mapping']['keyframe_every']).long()).unsqueeze(-1).unsqueeze(-1)
            poses_last = self.model.rba(last_kf_id)
            poses_all = poses
            poses_all[-1,:,:] = poses_last.squeeze().clone()
        
            
        for i in range(self.config['mapping']['iters']):
            
            rays, ids = self.keyframe.sample_global_rays(self.config['mapping']['sample'])

            idx_cur = random.sample(range(0, self.slam.dataset.H * self.slam.dataset.W),max(self.config['mapping']['sample'] // len(self.keyframe.frame_ids), self.config['mapping']['min_pixels_cur']))
            current_rays_batch = current_rays[idx_cur, :]
        
            rays = torch.cat([rays, current_rays_batch], dim=0) # N, 7 ([2560, 7])
            ids_all = torch.cat([ids//self.config['mapping']['keyframe_every'], -torch.ones((len(idx_cur)))]).to(torch.int64).to(self.device)
            #N+len(idx_cur)

            rays_d_cam = rays[..., :3].to(self.device)
            target_s = rays[..., 3:6].to(self.device)
            target_d = rays[..., 6:7].to(self.device)

            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
            rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
    
            ret = self.model.mapping(rays_o, rays_d, target_s, target_d)  
                    
            loss = self.slam.get_loss_from_ret(ret, smooth=True, iter=i)

            loss.backward(retain_graph=True)

            if (i + 1) % self.config["mapping"]["map_accum_step"] == 0:
                if (i + 1) > self.config["mapping"]["map_wait_step"]:
                    self.map_optimizer.step()
                else:
                    print('Wait update')
                self.map_optimizer.zero_grad()
                self.rba_optimizer.zero_grad()
    
    def global_pose(self, batch, cur_frame_id):
        """
        Runs global bundle adjustment over all keyframes and the current frame.

        Args:
            batch (dict): Contains the following keys:
                - 'c2w': ground truth camera pose [1, 4, 4]
                - 'rgb': RGB image tensor [1, H, W, 3]
                - 'depth': depth image tensor [1, H, W, 1]
                - 'direction': view direction tensor [1, H, W, 3]
            cur_frame_id (int): Index of the current frame.
            update_flag (bool): Optional flag to control update behavior (unused by default).
        """
        # Stack all keyframe poses up to current frame (excluding current frame).
        poses = torch.stack([self.est_c2w_data[i] for i in range(0, cur_frame_id, self.config['mapping']['keyframe_every'])])
        
        # List of all keyframe indices (for updating poses after optimization)
        frame_ids_all = torch.tensor(list(range(0, cur_frame_id+1, self.config['mapping']['keyframe_every'])))

        # Zero gradients for both map and pose optimizers before starting BA
        self.map_optimizer.zero_grad()
        self.rba_optimizer.zero_grad()

        # Prepare current frame rays by concatenating direction, rgb, and depth components.
        current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1) # Shape: [H, W, 7]
        current_rays = current_rays.reshape(-1, current_rays.shape[-1]) # Shape: [H*W, 7]

        # Get indices for all keyframes and current frame
        all_index = torch.arange(0, poses.shape[0]+1).unsqueeze(-1) # Shape: [num_keyframes+1, 1]

        # Get all pose parameters from the pose optimization module (rba)
        poses_all = self.model.rba(all_index)

        for i in range(self.config['mapping']['BA_iters']):
            # Sample global rays from all keyframes
            rays, ids = self.keyframe.sample_global_rays(self.config['mapping']['sample'])

            # Randomly sample rays from the current frame for joint optimization
            idx_cur = random.sample(
                range(0, self.slam.dataset.H * self.slam.dataset.W),
                max(self.config['mapping']['sample'] // len(self.keyframe.frame_ids), self.config['mapping']['min_pixels_cur'])
            )
            current_rays_batch = current_rays[idx_cur, :]
           
            # Concatenate keyframe rays and current frame rays
            rays = torch.cat([rays, current_rays_batch], dim=0)
            
            # Build index tensor for all sampled rays (keyframes and current frame)
            ids_all = torch.cat(
                [ids // self.config['mapping']['keyframe_every'], -torch.ones((len(idx_cur)))]
            ).to(torch.int64).to(self.device)

            # Split rays into components for input to mapping
            rays_d_cam = rays[..., :3].to(self.device) # Camera directions
            target_s = rays[..., 3:6].to(self.device)  # RGB targets
            target_d = rays[..., 6:7].to(self.device)  # Depth targets

            # Transform ray directions and origins from camera to world coordinates
            # rays_d: [N, Bs, 3], rays_o: [N*Bs, 3]
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
            rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
    
            # Forward pass through the mapping model, with clamping enabled
            ret = self.model.mapping(
                rays_o, rays_d, target_s, target_d, clamp=True
            )  
            
            # Compute loss for SLAM update, enable feature similarity and smoothing
            loss = self.slam.get_loss_from_ret(ret, fs=True, smooth=True, iter=i)
            loss.backward(retain_graph=True)

            # Optimize pose parameters at pose_accum_step intervals if opt_pose is True
            if (i + 1) % self.config["mapping"]["pose_accum_step"] == 0 and self.config["mapping"]["opt_pose"]:   
                self.rba_optimizer.step()    
                
                # Update poses_all for the next step
                all_index = torch.arange(0, poses.shape[0]+1).unsqueeze(-1)
                poses_all = self.model.rba(all_index)

                # Empty gradients after the pose step
                self.map_optimizer.zero_grad()
                self.rba_optimizer.zero_grad()
        
        # After BA, update est_c2w_data with optimized poses
        if len(frame_ids_all) > 1 and self.config["mapping"]["opt_pose"]:
            kf_len = torch.arange(len(frame_ids_all)-1).to(self.device)
            kfupid = kf_len * self.config['mapping']['keyframe_every']
            
            if self.config['mapping']['optim_cur']:
                # Update current frame pose and keyframes' poses
                self.est_c2w_data[cur_frame_id] = poses_all[-1:].detach().clone()[0]
                self.est_c2w_data[kfupid] = poses_all[:-1].detach().clone()
            else:
                # Only update keyframes' poses
                self.est_c2w_data[kfupid] = poses_all[:-1].detach().clone()
            
            
    def update_GBV(self,cur_id):
        # Clean init map volume
        self.model.GBV.params[:] = 0.0
        self.init_mapvolume()
        self.model.GBW.params[:] = 0.0

        # Integrate keyframes into the map volume
        for i in range(0, cur_id, self.config['mapping']['keyframe_every']):
            batch = self.dataset[i]
            c2w = self.est_c2w_data[i] #BA pose
            # Integrate keyframe into the map volume
            self.integrate_kf(batch, c2w)

    def convert_relative_pose_npy(self, idx=None):
        """
        Convert stored absolute and relative camera poses to absolute poses for all frames.

        This function generates a numpy array of absolute camera poses for each frame in the dataset,
        by composing the relative pose (delta) with its most recent keyframe's absolute pose.
        If the frame itself is a keyframe, use the stored absolute pose directly.
        Optionally, can process only up to a given frame index (`idx`).

        Args:
            idx (int, optional): Frame index up to which to process poses. If None, process all poses.

        Returns:
            poses_np (np.ndarray): Array of shape (num_frames, 4, 4) with the absolute pose for each frame.
        """
        poses = torch.zeros((len(self.dataset), 4, 4)).to(self.device)
        if idx is not None:
            for i in range(len(self.est_c2w_data[:idx+1])):
                # For keyframe indices, use the absolute pose directly
                if i % self.config['mapping']['keyframe_every'] == 0:
                    poses[i] = self.est_c2w_data[i]
                else:
                    # For non-keyframe, compose relative pose with keyframe's absolute pose
                    kf_id = i // self.config['mapping']['keyframe_every']
                    kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                    c2w_key = self.est_c2w_data[kf_frame_id]
                    delta = self.est_c2w_data_rel[i] 
                    poses[i] = delta @ c2w_key
        else:
            for i in range(len(self.est_c2w_data)):
                # For keyframe indices, use the absolute pose directly
                if i % self.config['mapping']['keyframe_every'] == 0:
                    poses[i] = self.est_c2w_data[i]
                else:
                    # For non-keyframe, compose relative pose with keyframe's absolute pose
                    kf_id = i // self.config['mapping']['keyframe_every']
                    kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                    c2w_key = self.est_c2w_data[kf_frame_id]
                    delta = self.est_c2w_data_rel[i] 
                    poses[i] = delta @ c2w_key
        poses_np = poses.detach().cpu().numpy()
        return poses_np
    
    
    def convert_relative_pose(self, idx=None):
        """
        Convert relative poses to absolute poses for all frames.

        For every frame, this function finds the most recent keyframe and composes 
        the relative pose (delta) with the keyframe's absolute pose to produce 
        the absolute pose for each frame. If the frame itself is a keyframe, 
        its absolute pose is used directly.

        Args:
            idx (int, optional): If specified, process poses only up to frame idx 
                                 (inclusive). If None, use all available poses.

        Returns:
            poses (dict): Mapping from frame index to its absolute pose as a torch tensor.
        """
        poses = {}
        if idx is not None:
            # Only process poses up to frame idx (inclusive)
            for i in range(len(self.est_c2w_data[:idx+1])):
                if i % self.config['mapping']['keyframe_every'] == 0:
                    # Keyframe: use its absolute pose directly
                    poses[i] = self.est_c2w_data[i]
                else:
                    # Non-keyframe: compose relative pose with keyframe's absolute pose
                    kf_id = i // self.config['mapping']['keyframe_every']
                    kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                    c2w_key = self.est_c2w_data[kf_frame_id]
                    delta = self.est_c2w_data_rel[i] 
                    poses[i] = delta @ c2w_key
        else:
            # Process all frames in est_c2w_data
            for i in range(len(self.est_c2w_data)):
                if i % self.config['mapping']['keyframe_every'] == 0:
                    # Keyframe: use its absolute pose directly
                    poses[i] = self.est_c2w_data[i]
                else:
                    # Non-keyframe: compose relative pose with keyframe's absolute pose
                    kf_id = i // self.config['mapping']['keyframe_every']
                    kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                    c2w_key = self.est_c2w_data[kf_frame_id]
                    delta = self.est_c2w_data_rel[i] 
                    poses[i] = delta @ c2w_key

        return poses
    
    def calc_2d_metric_with_results(self, poses, gap=10, scene='cafeteria', save=False):
        """
        Compute 2D evaluation metrics (PSNR, SSIM, LPIPS, Depth L1) between ground truth and rendered RGB/depth images.

        This function loads ground-truth and predicted (rendered) color and depth images, aligns them by depth mask,
        and calculates several quality metrics for each frame (sampled at a given gap). The rendered images are loaded
        from given directories determined by the scene name. Results for each image are printed individually and mean
        values for all metrics are shown at the end.

        Args:
            poses (dict or None): Optional dict of camera poses for each frame index.
            gap (int): Sampling gap to reduce number of evaluation frames.
            scene (str): Scene name used to construct file paths for rendered results.
        """
        psnrs = []
        ssims = []
        lpips = []
        d_L1 = []
        
        # Initialize LPIPS metric computation model (AlexNet backbone, normalized)
        cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to("cuda")

        # Path to color and depth image base directories for the rendered results
        color_base = os.path.join('/home/lyq/myprojects/GS_ICP_SLAM/experiments/results/BS3D/', scene)
        depth_base = os.path.join('/home/lyq/myprojects/GS_ICP_SLAM/experiments/results/BS3D/', scene)

        with torch.no_grad():
            for i in tqdm(range(0, len(self.dataset), gap)):
                print("frame", i)
                # Retrieve current pose if provided
                if poses is not None:
                    c2w = poses[i]
                batch = self.dataset[i]
                gt_rgb = batch['rgb'].cuda()
                gt_depth = batch['depth'].cuda()
                gt_depth_np = gt_depth.cpu().numpy()

                # Define rendered color and depth image paths for the current frame
                # (commented code for uhumans; currently set up for BS3D)
                color_path = os.path.join(color_base, str(i) + '.png') 
                depth_path = os.path.join(depth_base, str(i) + '_d.png')

                # Load rendered RGB and depth images for the current frame
                ours_rgb = cv2.imread(color_path)
                ours_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                ours_rgb = cv2.cvtColor(ours_rgb, cv2.COLOR_BGR2RGB)
                ours_rgb = ours_rgb / 255.
                ours_rgb = torch.from_numpy(ours_rgb).cuda()

                # Convert predicted depth to meters according to PNG scale in config
                ours_depth_np = ours_depth.astype(np.float32) / self.config['cam']['png_depth_scale']

                # Clamp rendered RGB for numerical stability
                ours_rgb = torch.clamp(ours_rgb, 0., 1.).float()

                # Optionally save the rendered RGB/depth images for visualization
                if i % gap == 0 and save:
                    ours_rgb_np = ours_rgb.cpu().numpy()
                    imwrite(os.path.join(self.config['data']['output'], self.config['data']['exp_name'], str(i)+".png"), (ours_rgb_np*255.).astype(np.uint8))
                    render_depth_np = ours_depth_np * 1000.
                    imwrite(os.path.join(self.config['data']['output'], self.config['data']['exp_name'], str(i)+"_d.png"), render_depth_np.astype(np.uint16))
                
                # Create a mask for valid (positive) ground truth depth values
                valid_depth_mask_ = (gt_depth > 0)
                gt_rgb = gt_rgb * valid_depth_mask_[..., None]
                ours_rgb = ours_rgb * valid_depth_mask_[..., None]

                # Compute PSNR
                square_error = (gt_rgb - ours_rgb) ** 2
                mse_error = torch.mean(torch.mean(square_error, axis=2))
                psnr = mse2psnr(mse_error)   
                psnrs += [psnr.detach().cpu()]

                # Compute Depth L1 error (only where GT depth exists)
                depth_L1 = np.abs(
                    gt_depth_np[gt_depth_np > 0.0] - ours_depth_np[gt_depth_np > 0.0]
                ).mean()
                d_L1 += [depth_L1]

                # Compute SSIM
                _, ssim_error = ssim(ours_rgb, gt_rgb)
                ssims += [ssim_error.detach().cpu()]

                # Prepare RGB for LPIPS: shape [C, H, W]
                gt_rgb = gt_rgb.permute(2, 0, 1)
                ours_rgb = ours_rgb.permute(2, 0, 1)
                lpips_value = cal_lpips(gt_rgb.unsqueeze(0), ours_rgb.unsqueeze(0))
                lpips += [lpips_value.detach().cpu()]

                # Free CUDA cache (prevents OOM for large scenes)
                torch.cuda.empty_cache()

                # Print per-frame results
                print(f"{i} psnr:{psnr:.2f},ssim:{ssim_error:.2f},lpips:{lpips_value:.2f},d-l1:{depth_L1:.3f}")
            
            # Convert results lists to numpy arrays and print overall means
            psnrs = np.array(psnrs)
            ssims = np.array(ssims)
            lpips = np.array(lpips)
            d_L1 = np.array(d_L1)
            
            print(f"PSNR: {psnrs.mean():.2f}\nSSIM: {ssims.mean():.3f}\nLPIPS: {lpips.mean():.3f}\nD-L1: {d_L1.mean():.3f}")
    
    def calc_2d_metric(self, poses, gap=10, save=False):
        """
        Calculate 2D image quality metrics (PSNR, SSIM, LPIPS, Depth L1 error) 
        between rendered results and ground truth RGB-D frames for a series of poses.

        Args:
            poses (Tensor): Camera-to-world poses for evaluation, shape (N, 4, 4).
            gap (int): Step size for frame sampling (default: 10, i.e., evaluate every 10th frame).
        """
        psnrs = []
        ssims = []
        lpips = []
        d_L1 = []
        
        # Initialize LPIPS perceptual similarity metric (using AlexNet backbone) on CUDA
        cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to("cuda")
        poses = poses.reshape(-1, 4, 4)
        with torch.no_grad():
            for i in tqdm(range(0, len(self.dataset), gap)):
                print("frame", i)
                c2w = poses[i]
                
                # Load GT RGB and depth, move to CUDA
                batch = self.dataset[i]
                gt_rgb = batch['rgb'].cuda()
                gt_depth = batch['depth'].cuda()
                gt_depth_np = gt_depth.cpu().numpy()

                # Render RGB and depth given pose for frame i
                ours_rgb, ours_depth = self.slam.render_single(i, gt_depth, gt_rgb, c2w, batch["direction"])
                ours_depth_np = ours_depth.cpu().numpy()
                
                ours_rgb = torch.clamp(ours_rgb, 0., 1.)
                
                # Optionally save visualization for debugging (commented out)
                if i % gap == 0 and save:
                    ours_rgb_np = ours_rgb.cpu().numpy()
                    psnr_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'psnr')
                    if not os.path.exists(psnr_path):
                        os.mkdir(psnr_path)
                    imwrite(os.path.join(psnr_path, str(i) + ".png"), (ours_rgb_np * 255.).astype(np.uint8))
                    render_depth_np = ours_depth_np * 1000.
                    imwrite(os.path.join(psnr_path, str(i) + "_d.png"), render_depth_np.astype(np.uint16))
                
                # Mask out invalid depth values
                valid_depth_mask_ = (gt_depth > 0)
                gt_rgb = gt_rgb * valid_depth_mask_[..., None]
                ours_rgb = ours_rgb * valid_depth_mask_[..., None]
                
                # Compute PSNR
                square_error = (gt_rgb - ours_rgb) ** 2
                mse_error = torch.mean(torch.mean(square_error, axis=2))
                psnr = mse2psnr(mse_error)
                psnrs += [psnr.detach().cpu()]
                
                # Compute depth L1 mean error at pixels where GT depth exists
                depth_L1 = np.abs(gt_depth_np[gt_depth_np > 0.0] - ours_depth_np[gt_depth_np > 0.0]).mean()
                d_L1 += [depth_L1]

                # Compute SSIM metric
                _, ssim_error = ssim(ours_rgb, gt_rgb)
                ssims += [ssim_error.detach().cpu()]

                # Compute LPIPS
                gt_rgb = gt_rgb.permute(2, 0, 1)
                ours_rgb = ours_rgb.permute(2, 0, 1)
                lpips_value = cal_lpips(gt_rgb.unsqueeze(0), ours_rgb.unsqueeze(0))
                lpips += [lpips_value.detach().cpu()]

                torch.cuda.empty_cache()
                print(f"{i} psnr:{psnr:.2f},ssim:{ssim_error:.2f},lpips:{lpips_value:.2f},d-l1:{depth_L1:.3f}")
                
            # Convert all metric lists to numpy arrays and report their mean
            psnrs = np.array(psnrs)
            ssims = np.array(ssims)
            lpips = np.array(lpips)
            d_L1 = np.array(d_L1)
            
            print(f"PSNR: {psnrs.mean():.2f}\nSSIM: {ssims.mean():.3f}\nLPIPS: {lpips.mean():.3f}\nD-L1: {d_L1.mean():.3f}")
    
    def post_process_mesh(self, pose_relative_np):
        """
        Post-process the mesh to remove occlusions and save the final mesh.
        """
        # cull mesh
        skip=5
        c2w_list = [pose_relative_np[i] for i in range(0, pose_relative_np.shape[0],skip)]
        mesh_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'mesh.ply')
        save_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'mesh_cull_occulsion.ply')
        cull_one_mesh(self.config, c2w_list, mesh_path, save_path, skip, self.slam.dataset, depth_flag = True, save_unseen=False,
                  remove_occlusion=True, scene_bounds=None,
                  eps=0.1)
        os.remove(mesh_path)
    
    def integrate_kf(self, batch, pose, obs_weight=1.0):
        """
        Integrate an RGB-D keyframe into the TSDF (Truncated Signed Distance Function) volume using GPU acceleration.

        Args:
            batch (dict): Dictionary containing:
                - 'rgb' (Tensor): RGB image of the frame, shape (H, W, 3) or (1, H, W, 3).
                - 'depth' (Tensor): Depth map of the frame, shape (H, W) or (1, H, W).
            pose (Tensor): Camera extrinsic matrix (camera-to-world), shape (4, 4).
            obs_weight (float): Observation weight for this integration (typically 1.0).
        
        This function runs a loop over possible GPU volume splits (for large scenes),
        and calls the CUDA kernel `map_integrate` to fuse the given RGB-D observation into
        the global volumetric map. The kernel receives all necessary camera parameters,
        current TSDF volume data, color and depth images, and grid/block configuration for efficient GPU execution.
        """
        color_im = batch['rgb'].squeeze().to(self.device)
        depth_im = batch['depth'].squeeze().to(self.device)
        im_h, im_w = depth_im.shape 
        
        for gpu_loop_idx in range(self.n_gpu_loops):
            map_integrate(
                Holder(self.model.GBV.params),
                Holder(self.model.GBW.params),
                cuda.In(self.vol_dim.astype(np.float32)),
                cuda.In(self.K.reshape(-1).astype(np.float32)),
                Holder(pose.float().reshape(-1)),
                cuda.In(np.asarray([
                    gpu_loop_idx,
                    self.voxel_size,
                    im_h,
                    im_w,
                    self.trunc_margin,
                    obs_weight,
                    self.map_box[0][0],
                    self.map_box[0][1],
                    self.map_box[1][0],
                    self.map_box[1][1],
                    self.map_box[2][0],
                    self.map_box[2][1],
                ], np.float32)),
                Holder(color_im.float().reshape(-1)),
                Holder(depth_im.float().reshape(-1)),
                block=(self.max_gpu_threads_per_block, 1, 1),
                grid=(
                    int(self.max_gpu_grid_dim[0]),
                    int(self.max_gpu_grid_dim[1]),
                    int(self.max_gpu_grid_dim[2]),
                )
            )
        
    def run(self):
 
        print("******* mapping process started!  *******")
        # Start mapping
        while self.tracking_idx[0]< len(self.dataset)-1:
            while self.tracking_idx[0] <= self.mapping_idx[0] + self.config['mapping']['map_every'] and self.tracking_stop_flag[0]==0:
                time.sleep(0.01)
            if self.first_BA:
                self.model =self.model.to(self.device)
                self.first_BA = False
            current_map_id = int(self.mapping_idx[0] + self.config['mapping']['keyframe_every']) #
            if current_map_id<len(self.dataset):
                batch = self.dataset[current_map_id]
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v[None, ...]
                    else:
                        batch[k] = torch.tensor([v])
                        
                # integrate global volume
                if self.mapping_idx[0] % self.config['mapping']['keyframe_every'] == 0:
                    self.model.rba.update_init_pose(int(current_map_id//self.config['mapping']['keyframe_every']), self.est_c2w_data[current_map_id])
                    self.integrate_kf(batch, self.est_c2w_data[current_map_id])
                    
                # global bundle adjustment and pose update
                self.global_mapping(batch, current_map_id)
                self.global_pose(batch, current_map_id)

                self.mapping_idx[0] = current_map_id


                if self.mapping_idx[0] % self.config['mapping']['keyframe_every'] == 0:
                    self.keyframe.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
            
                # save mesh video
                if self.config['video']['save'] and self.mapping_idx[0] % self.config['video']['save_freq'] == 0:
                    print("saving mesh:",self.mapping_idx[0])
                    self.slam.save_mesh(self.mapping_idx[0], voxel_size=0.075) #0.075
                    torch.cuda.empty_cache()
                    print("saved mesh:", self.mapping_idx[0])
                
                
                if self.mapping_idx[0] % self.config['mesh']['vis']==0:
                    idx = int(self.mapping_idx[0])
                    # save mesh for visualization
                    if not self.config["mesh"]["only_final"]:
                        self.slam.save_mesh(idx, voxel_size=0.1) # voxel_eval
                        print("save mesh:",self.mapping_idx[0])
                    # render RGB/depth image from mesh for evaluation
                    if self.config['mesh']["render_img"] and self.mapping_idx[0]>0:
                        self.slam.render_img(idx, batch["depth"],batch["rgb"],self.est_c2w_data[idx],batch["direction"])
        
                    # evaluate relative poses
                    pose_relative = self.convert_relative_pose(idx)
                    self.slam.pose_eval_func()(self.slam.pose_gt, pose_relative, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), idx, img='frame', name='tracking_result.txt')

        # End of mapping process
        idx = int(self.tracking_idx[0])      
        pose_relative_np = self.convert_relative_pose_npy()
        np.save(os.path.join(self.config['data']['output'], self.config['data']['exp_name'],"all_poses.npy"),pose_relative_np)

        pose_relative = self.convert_relative_pose()
        self.slam.pose_eval_func()(self.slam.pose_gt, pose_relative, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), idx, img='frame', name='tracking_result.txt')
        
        # save_checkpoint for rendering comparison
        if self.config['mapping']['save_ckpt']:
            model_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'checkpoint.pt'.format(idx)) 
            self.save_ckpt(model_savepath)
        # save mesh
        self.slam.save_mesh_final(voxel_size=self.config['mesh']['voxel_final'])
        print("==========================================================================================")
        print("🧩 RemixFusion is closing, waiting for post-processing the reconstructed mesh...")
        print("==========================================================================================")
        # cull mesh
        self.post_process_mesh(pose_relative_np)
        
        exit()
