import numpy as np
import copy
import torch
import os
import time
from skimage import measure
try:
  import pycuda.driver as cuda
  import pycuda.autoprimaryctx
  from pycuda.compiler import SourceModule
  import pycuda.gpuarray as gpuarray
  CUDA_GPU_MODE = 1
except Exception as err:
  print('Warning: {}'.format(err))
  print('Failed to import PyCUDA. Running fusion in CPU mode.')
  CUDA_GPU_MODE = 0


class moving_volume:
    """
    Moving volume of RGB-D Images.
    """
    def __init__(self, cfg, traj, init_pose, gpu_mode=True, start=0):
        self.config = cfg
        # Basic configuration
        self.voxel_size = cfg["volume"]["voxel_size"]  # Size of each voxel in the 3D grid
        self.surface_trunc = cfg["training"]["trunc"]  # Truncation threshold for TSDF
        self.trunc_margin = cfg["volume"]["trunc"]     # Additional truncation margin
        # Extended version parameters
        self.first_len = cfg["volume"]["first_len"]    # First stage length setting
        self.second_len = cfg["volume"]["second_len"]  # Second stage length setting
        self.third_len = cfg["volume"]["third_len"]    # Third stage length setting
        self.more_angel_t = cfg["volume"]["more_angel_t"]  # Flag to enable more angle threshold
        # Centered volume parameters
        self.fix_x = cfg["volume"]["x_config"]["fix"]      # Fixed x dimension or not
        self.fix_y = cfg["volume"]["y_config"]["fix"]      # Fixed y dimension or not
        self.fix_z = cfg["volume"]["z_config"]["fix"]      # Fixed z dimension or not
        self.x_len = cfg["volume"]["x_config"]["len"]      # Length along x
        self.y_len = cfg["volume"]["y_config"]["len"]      # Length along y
        self.z_len = cfg["volume"]["z_config"]["len"]      # Length along z
        self.x_range = cfg["volume"]["x_config"]["range"]  # Range for x
        self.y_range = cfg["volume"]["y_config"]["range"]  # Range for y
        self.z_range = cfg["volume"]["z_config"]["range"]  # Range for z
        self.version = cfg["volume"]["version"]            # Volume versioning
        self.t_treshold = cfg["volume"]["t_treshold"]      # Translation threshold
        self.cut = cfg["RO"]["cut"]                    # Clipping parameter 5
        self.cut_dist = cfg["RO"]["cut_dist"]              # Distance clipping
        self.weight_clamp = cfg["volume"]["weight_clamp"]  # Clamp for weight volume
        self.save_path = os.path.join(cfg['data']['output'], cfg['data']['exp_name'])  # Path to save results

        self.last_pcid = 0             # Last processed point cloud ID
        self.surface_pc = None         # Surface point cloud

        cam_pose_iter = init_pose      # Initial camera pose
            
        # Initialize volume bounds based on initial pose and trajectory
        self.vol_bnds = self.initialize_vol_bnd(cam_pose_iter, traj, self.version)
        '''
        Initialize TSDF volume based on volume bounds
        '''
        self.vol_bnds = np.asarray(self.vol_bnds)
        assert self.vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."
        # Set voxel size as float for further calculations
        self.voxel_size = float(self.voxel_size)
        self.color_const = 256 * 256         # Color constant for TSDF color encoding
        # Compute the number of voxels along each axis (volume dimensions)
        self.vol_dim = np.ceil((self.vol_bnds[:, 1] - self.vol_bnds[:, 0]) / self.voxel_size).copy(order='C').astype(int)
 
        # Update the upper bounds to be consistent with computed volume dimensions
        self.vol_bnds[:, 1] = self.vol_bnds[:, 0] + self.vol_dim * self.voxel_size
        self.vol_origin = self.vol_bnds[:, 0].copy(order='C').astype(np.float32)    # Origin of the volume in world coordinates
        # Store information related to volume bounds for record-keeping or debugging
        self.start_id = 0                      # ID of the starting frame
        self.frame_to_Vrange = {}              # Map from frame to its volume range

        # Logging information about the initialized volume
        # print("vol_dim:", self.vol_dim)
        # print("vol_orgin:", self.vol_origin)
        # print("Voxel volume size: {} x {} x {} - # points: {:,}".format(
        #     self.vol_dim[0], self.vol_dim[1], self.vol_dim[2], 
        #     self.vol_dim[0] * self.vol_dim[1] * self.vol_dim[2]))
        # print("init vol bnd:", self.vol_bnds)

        # Allocate memory for TSDF volume, weight volume, and color volume on the CPU
        self.tsdf_vol_cpu = np.ones(self.vol_dim[0] * self.vol_dim[1] * self.vol_dim[2]).astype(np.float32)   # TSDF values
        self.weight_vol_cpu = np.zeros(self.vol_dim[0] * self.vol_dim[1] * self.vol_dim[2]).astype(np.float32)    # Weights
        self.color_vol_cpu = np.zeros(self.vol_dim[0] * self.vol_dim[1] * self.vol_dim[2]).astype(np.float32)     # Color information
        
        # Initialize and allocate memory for TSDF, weight, and color volumes on the GPU, if enabled
        self.gpu_mode = False
        if gpu_mode and CUDA_GPU_MODE:
            self.gpu_mode = True
            # Allocate memory and copy CPU data to GPU
            self.tsdf_vol_gpu = cuda.mem_alloc(self.tsdf_vol_cpu.nbytes)
            cuda.memcpy_htod(self.tsdf_vol_gpu, self.tsdf_vol_cpu)
            self.weight_vol_gpu = cuda.mem_alloc(self.weight_vol_cpu.nbytes)
            cuda.memcpy_htod(self.weight_vol_gpu, self.weight_vol_cpu)
            self.color_vol_gpu = cuda.mem_alloc(self.color_vol_cpu.nbytes)
            cuda.memcpy_htod(self.color_vol_gpu, self.color_vol_cpu)
            
            # If not full volume, create backup buffers for GPU volumes
            self.tsdf_vol_gpu_back = cuda.mem_alloc(self.tsdf_vol_cpu.nbytes)
            cuda.memcpy_htod(self.tsdf_vol_gpu_back, self.tsdf_vol_cpu)
            self.weight_vol_gpu_back = cuda.mem_alloc(self.weight_vol_cpu.nbytes)
            cuda.memcpy_htod(self.weight_vol_gpu_back, self.weight_vol_cpu)
            self.color_vol_gpu_back = cuda.mem_alloc(self.color_vol_cpu.nbytes)
            cuda.memcpy_htod(self.color_vol_gpu_back, self.color_vol_cpu)
            
            # Determine block/grid sizes for GPU parallelization
            gpu_dev = cuda.Device(0) 
            self.max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
            n_blocks = int(np.ceil(float(np.prod(self.vol_dim)) / float(self.max_gpu_threads_per_block)))
            grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X, int(np.floor(np.cbrt(n_blocks))))
            grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y, int(np.floor(np.sqrt(n_blocks / grid_dim_x))))
            grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z, int(np.ceil(float(n_blocks) / float(grid_dim_x * grid_dim_y))))
            self.max_gpu_grid_dim = np.array([grid_dim_x, grid_dim_y, grid_dim_z]).astype(int)
           
            # Calculate number of loops required to process all volume voxels on the GPU
            self.n_gpu_loops = int(
                np.ceil(
                    float(np.prod(self.vol_dim)) / float(np.prod(self.max_gpu_grid_dim) * self.max_gpu_threads_per_block)
                )
            )
            # print("max_gpu_grid_dim:",self.max_gpu_grid_dim,"block:",self.max_gpu_threads_per_block,"loop:",self.n_gpu_loops)
            
            # Cuda kernel function (C++)
            self.cuda_src_mod = SourceModule("""
            __global__ void swap_rot_trans(float * tsdf_vol,
                                    float * old_tsdf_vol,
                                    float * weight_vol,
                                    float * old_weight_vol,
                                    float * color_vol,
                                    float * old_color_vol,
                                    float * vol_dim,
                                    float * vol_origin,
                                    float * old_origin,
                                    float * old_vol_dim,
                                    float * other_params)
                                    {
            //calculate size of all voxels
            int vol_dim_x = (int) vol_dim[0];
            int vol_dim_y = (int) vol_dim[1];
            int vol_dim_z = (int) vol_dim[2];
            
                                        
            //get voxel index 
            int gpu_loop_idx = (int) other_params[0];
            float voxel_size = other_params[1];
            int max_threads_per_block = blockDim.x;
            int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
            int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;
    
            //exclude the left threads
            if ((voxel_idx >= vol_dim_x*vol_dim_y*vol_dim_z) || voxel_idx<0){
                return;
            }
            
            // Get voxel grid coordinates (note: be careful when casting)
            float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
            float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
            float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);
            

            //new
            float wx = vol_origin[0] + voxel_x*voxel_size;
            float wy = vol_origin[1] + voxel_y*voxel_size;
            float wz = vol_origin[2] + voxel_z*voxel_size;
            
            float voxel_x_old = round((wx - old_origin[0])/voxel_size);
            float voxel_y_old = round((wy - old_origin[1])/voxel_size);
            float voxel_z_old = round((wz - old_origin[2])/voxel_size);

            int old_voxelx = (int)(voxel_x_old);
            int old_voxely = (int)(voxel_y_old);
            int old_voxelz = (int)(voxel_z_old);
        
            int old_vol_dim_x = (int) old_vol_dim[0];
            int old_vol_dim_y = (int) old_vol_dim[1];
            int old_vol_dim_z = (int) old_vol_dim[2];
    
            //old vol_dim
            if ((0<=old_voxelx && old_voxelx<old_vol_dim_x) && (0<=old_voxely && old_voxely<old_vol_dim_y) && (0<=old_voxelz && old_voxelz<old_vol_dim_z)){
                int old_voxel_id =  (int)(old_voxelz + old_voxely*old_vol_dim_z+ old_voxelx*old_vol_dim_y*old_vol_dim_z);
                
                tsdf_vol[voxel_idx] = old_tsdf_vol[old_voxel_id];
                weight_vol[voxel_idx] = old_weight_vol[old_voxel_id];
                color_vol[voxel_idx] = old_color_vol[old_voxel_id];
            }
            else{
                tsdf_vol[voxel_idx] = 1.0;
                weight_vol[voxel_idx] = 0;
                color_vol[voxel_idx] = 0;
                } 
            }                   
                                                                                                   
            __global__ void integrate(float * tsdf_vol,
                                    float * weight_vol,
                                    float * color_vol,
                                    float * vol_dim,
                                    float * vol_origin,
                                    float * cam_intr,
                                    float * cam_pose,
                                    float * other_params,
                                    float * old_bnd,
                                    float * color_im,
                                    float * depth_im) {
            // Get voxel index
            int gpu_loop_idx = (int) other_params[0];
            int max_threads_per_block = blockDim.x;
            float obs_weight = other_params[5];
            float reintegrate_flag = other_params[6];
            float weight_clamp = other_params[7];
            
            int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
            int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;
            int vol_dim_x = (int) vol_dim[0];
            int vol_dim_y = (int) vol_dim[1];
            int vol_dim_z = (int) vol_dim[2];

            if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z){
                return;
            }
            // Get voxel grid coordinates (note: be careful when casting)
            float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
            float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
            float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);
            // Voxel grid coordinates to world coordinates
            float voxel_size = other_params[1];
            
            int origin_x = vol_origin[0];
            int origin_y = vol_origin[1];
            int origin_z = vol_origin[2];
              
            float pt_x = origin_x+(voxel_x)*voxel_size;
            float pt_y = origin_y+(voxel_y)*voxel_size;
            float pt_z = origin_z+(voxel_z)*voxel_size;
            //float pt_x = origin_x+(voxel_x+0.5f)*voxel_size;
            //float pt_y = origin_y+(voxel_y-0.5f)*voxel_size;
            //float pt_z = origin_z+(voxel_z-0.5f)*voxel_size;
            //printf("originx:%f, originy:%f, originz:%f", origin_x,origin_y,origin_z);
            
            if (reintegrate_flag==1){
                if (pt_x<old_bnd[0] || pt_x>=old_bnd[1] || pt_y<old_bnd[2] || pt_y>=old_bnd[3] || pt_z<old_bnd[4] || pt_z>=old_bnd[5]){
                    return;
                }
            }
            
            
            
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

            // Skip if outside view frustum
            int im_h = (int) other_params[2];
            int im_w = (int) other_params[3];
            
            if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h)
                return;
            
            float depth_value = depth_im[pixel_y*im_w+pixel_x];
            
            // Skip invalid depth
            if (depth_value <= 0)
                return;
            
            // Integrate TSDF
            float trunc_margin = other_params[4];
            
            float vec_x = (((float)pixel_x) - cam_intr[0*3+2])/cam_intr[0*3+0];
            float vec_y = (((float)pixel_y) - cam_intr[1*3+2])/cam_intr[1*3+1];
            float lambda = sqrt(vec_x*vec_x + vec_y*vec_y+1);
            float cam_norm = sqrt(cam_pt_x*cam_pt_x+cam_pt_y*cam_pt_y+cam_pt_z*cam_pt_z);
            
            float sdf = (-1.f) * ((1.f / lambda) * cam_norm - depth_value);
            
            if (sdf >= -trunc_margin){
                float dist = fmin(1.0f,sdf/trunc_margin);
                float cur_tsdf = tsdf_vol[voxel_idx];          
                float w_old = weight_vol[voxel_idx];
                float w_new = w_old + obs_weight;
                  
                float new_tsdf=(cur_tsdf*w_old+obs_weight*dist)/(w_new); 
                
                //if (new_tsdf>1.0){
                //    return;
                //}
                
                //test
                float new_weight = w_new;
                //old
                if (weight_clamp==1.0){
                    new_weight = fmin(w_new,128.0f);
                    if (new_weight>40){
                        new_weight=40;
                    } 
                }
                
                tsdf_vol[voxel_idx] = new_tsdf;
                weight_vol[voxel_idx] = new_weight;
                
                if (sdf >= -trunc_margin && sdf <= trunc_margin){
                    float new_color = color_im[pixel_y*im_w+pixel_x];
                    float new_b = floorf(new_color/(256*256));
                    float new_g = floorf((new_color-new_b*256*256)/256);
                    float new_r = new_color-new_b*256*256-new_g*256;
                
                    // Integrate colordist
                    float old_color = color_vol[voxel_idx];
                    float old_b = floorf(old_color/(256*256));
                    float old_g = floorf((old_color-old_b*256*256)/256);
                    float old_r = old_color-old_b*256*256-old_g*256;
                    
                    new_b = fmin(roundf((old_b*w_old+obs_weight*new_b)/w_new),255.0f);
                    new_g = fmin(roundf((old_g*w_old+obs_weight*new_g)/w_new),255.0f);
                    new_r = fmin(roundf((old_r*w_old+obs_weight*new_r)/w_new),255.0f);
                    color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;
                }
                
                if (obs_weight ==-1.0 && w_old<=1 && reintegrate_flag==1.0) {
                    tsdf_vol[voxel_idx] = 1.0;
                    weight_vol[voxel_idx]= 0;
                    color_vol[voxel_idx]=0;
                }
            }    
        }
            __global__ void tri_intepolate(float * tsdf_vol,
                                    float * weight_vol,
                                    float * color_vol,
                                    float * query_pc,
                                    float * tsdf_rgb_np,
                                    float * vol_dim,
                                    float * vol_origin,
                                    float * other_params) {
                                        
            int max_threads_per_block = blockDim.x;
            int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
            int pc_idx = block_idx*max_threads_per_block+threadIdx.x;
            //int pc_idx = blockDim.x*blockIdx.x+threadIdx.x;
            
            float num_pc =  other_params[0];
            float voxel_size =  other_params[1];
            float trunc_margin =  other_params[2];
            float vol_origin_x = vol_origin[0];
            float vol_origin_y = vol_origin[1];
            float vol_origin_z = vol_origin[2];
            int vol_dim_x = (int) vol_dim[0];
            int vol_dim_y = (int) vol_dim[1];
            int vol_dim_z = (int) vol_dim[2];
            
            
            
            if (pc_idx > num_pc){
                return;
            }
            
            float x = query_pc[pc_idx*3];
            float y = query_pc[pc_idx*3+1];
            float z = query_pc[pc_idx*3+2];
            
            int low_x = (int)floor((x-vol_origin_x)/voxel_size);
            int low_y = (int)floor((y-vol_origin_y)/voxel_size);
            int low_z = (int)floor((z-vol_origin_z)/voxel_size);
            
            float x_ori = vol_origin_x + low_x*voxel_size;
            float y_ori = vol_origin_y + low_y*voxel_size;
            float z_ori = vol_origin_z + low_z*voxel_size;
            
            
            
            //make sure the pc in volume, so there is available sdf/rgb value 
            if (low_x<0 || low_x>=vol_dim_x-1 || low_y<0 || low_y>=vol_dim_y-1 || low_z<0 || low_z>=vol_dim_z-1){ //>=
                
                tsdf_rgb_np[pc_idx*5] = 1.0; //-1.0
                tsdf_rgb_np[pc_idx*5+1] = 0.0;
                tsdf_rgb_np[pc_idx*5+2] = 0.0;
                tsdf_rgb_np[pc_idx*5+3] = 0.0;
                tsdf_rgb_np[pc_idx*5+4] = 0.0;
                return;
            }

            float u = (x - x_ori)/voxel_size;
            float v = (y - y_ori)/voxel_size;
            float w = (z - z_ori)/voxel_size;
            
            // three-interpolation
            //calculate the eight corners voxel index
            int low_0 = (int)(low_z + low_y*vol_dim_z+ low_x*vol_dim_y*vol_dim_z);
            int low_1 = low_0+1;
            int low_2 = (int)(low_z + (low_y+1)*vol_dim_z+ low_x*vol_dim_y*vol_dim_z);
            int low_3 = low_2+1;
            int low_4 = (int)(low_z + (low_y)*vol_dim_z+ (low_x+1)*vol_dim_y*vol_dim_z);
            int low_5 = low_4+1;
            int low_6 = (int)(low_z + (low_y+1)*vol_dim_z+ (low_x+1)*vol_dim_y*vol_dim_z);
            int low_7 = low_6+1;
            
            float tv[2][2][2] = {tsdf_vol[low_0],tsdf_vol[low_1],tsdf_vol[low_2],tsdf_vol[low_3],tsdf_vol[low_4],tsdf_vol[low_5],tsdf_vol[low_6],tsdf_vol[low_7]};
            
            float wv[2][2][2] = {weight_vol[low_0],weight_vol[low_1],weight_vol[low_2],weight_vol[low_3],weight_vol[low_4],weight_vol[low_5],weight_vol[low_6],weight_vol[low_7]};
            
            float cvb[2][2][2] = {floorf(color_vol[low_0]/65536),floorf(color_vol[low_1]/65536),floorf(color_vol[low_2]/65536),floorf(color_vol[low_3]/65536),floorf(color_vol[low_4]/65536),floorf(color_vol[low_5]/65536),floorf(color_vol[low_6]/65536),floorf(color_vol[low_7]/65536)};
            
            float cvg[2][2][2] = {floorf((color_vol[low_0]-cvb[0][0][0]*65536)/256),floorf((color_vol[low_1]-cvb[0][0][1]*65536)/256),floorf((color_vol[low_2]-cvb[0][1][0]*65536)/256),floorf((color_vol[low_3]-cvb[0][1][1]*65536)/256),floorf((color_vol[low_4]-cvb[1][0][0]*65536)/256),floorf((color_vol[low_5]-cvb[1][0][1]*65536)/256),floorf((color_vol[low_6]-cvb[1][1][0]*65536)/256),floorf((color_vol[low_7]-cvb[1][1][1]*65536)/256)};
            
            float cvr[2][2][2] = {floorf(color_vol[low_0]-cvb[0][0][0]*65536-cvg[0][0][0]*256),floorf(color_vol[low_1]-cvb[0][0][1]*65536-cvg[0][0][1]*256),floorf(color_vol[low_2]-cvb[0][1][0]*65536-cvg[0][1][0]*256),floorf(color_vol[low_3]-cvb[0][1][1]*65536-cvg[0][1][1]*256),floorf(color_vol[low_4]-cvb[1][0][0]*65536-cvg[1][0][0]*256),floorf(color_vol[low_5]-cvb[1][0][1]*65536-cvg[1][0][1]*256),floorf(color_vol[low_6]-cvb[1][1][0]*65536-cvg[1][1][0]*256),floorf(color_vol[low_7]-cvb[1][1][1]*65536-cvg[1][1][1]*256)};
            
            auto tri_tsdf = 0.0;
            auto tri_cb = 0.0;
            auto tri_cg = 0.0; 
            auto tri_cr = 0.0;
            auto tri_w = 0.0;
            for (int i = 0; i < 2; i++){
                for (int j = 0; j < 2; j++){
                    for (int k = 0; k < 2; k++){
                      tri_tsdf += (i*u + (1 - i)*(1 - u)) *
                              (j*v + (1 - j)*(1 - v)) *
                              (k*w + (1 - k)*(1 - w)) *
                              tv[i][j][k];
                      tri_cb += (i*u + (1 - i)*(1 - u)) *
                              (j*v + (1 - j)*(1 - v)) *
                              (k*w + (1 - k)*(1 - w)) *
                              cvb[i][j][k];
                      tri_cg += (i*u + (1 - i)*(1 - u)) *
                              (j*v + (1 - j)*(1 - v)) *
                              (k*w + (1 - k)*(1 - w)) *
                              cvg[i][j][k];
                      tri_cr += (i*u + (1 - i)*(1 - u)) *
                              (j*v + (1 - j)*(1 - v)) *
                              (k*w + (1 - k)*(1 - w)) *
                              cvr[i][j][k];
                      if (wv[i][j][k]>0){
                          tri_w += 1.0;
                      }
                              }}}
                      
            float w_v = weight_vol[low_0];
            if (tri_w<4){
                w_v = 0;
            }
            
            tsdf_rgb_np[pc_idx*5] = tri_tsdf;
            tsdf_rgb_np[pc_idx*5+1] = floor(tri_cr);
            tsdf_rgb_np[pc_idx*5+2] = floor(tri_cg);
            tsdf_rgb_np[pc_idx*5+3] = floor(tri_cb);
            //tsdf_rgb_np[pc_idx*5+4] = w_v; //weight_vol[low_0];
            tsdf_rgb_np[pc_idx*5+4] = tsdf_vol[low_0];
            
        }



            __global__ void filter_tsdf(float * tsdf_vol,
                                        float * weight_vol,
                                        float * color_vol,
                                        float * vol_dim,
                                        float * other_params
                                        ){
                float weight_threshold=(int) other_params[0];
                int max_threads_per_block = blockDim.x;

                
                int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
                int voxel_idx = block_idx*max_threads_per_block+threadIdx.x;
                int vol_dim_x = (int) vol_dim[0];
                int vol_dim_y = (int) vol_dim[1];
                int vol_dim_z = (int) vol_dim[2];
                if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z){
                return;}
                if (weight_vol[voxel_idx]>=weight_threshold || weight_vol[voxel_idx]==0){
                return;}

                weight_vol[voxel_idx]=0;
                tsdf_vol[voxel_idx]=1;
                color_vol[voxel_idx]=0;
                
                
            }
            
            __global__ void get_truncated_pc(float * tsdf_vol,
                                    float * color_vol,
                                    float * vol_dim,
                                    float * vol_origin,
                                    float * truncated_pc,
                                    //float * pc_count,
                                    float pc_count[1],
                                    float * other_params) {
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
            
            float tsdf = tsdf_vol[voxel_idx];
            float old_color = color_vol[voxel_idx];
            float old_b = floorf(old_color/(256*256));
            float old_g = floorf((old_color-old_b*256*256)/256);
            float old_r = old_color-old_b*256*256-old_g*256;
            
            
            float trunc_dist = other_params[4];

            if (tsdf<=-trunc_dist || tsdf>=trunc_dist){
                return;
            }

            // Get voxel grid coordinates (note: be careful when casting)
            float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
            float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
            float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);
            // Voxel grid coordinates to world coordinates
            float voxel_size = other_params[1];
            
            //float pt_x = vol_origin[0]+(voxel_x)*voxel_size;
            //float pt_y = vol_origin[1]+(voxel_y)*voxel_size;
            //float pt_z = vol_origin[2]+(voxel_z)*voxel_size;
            float pt_x = vol_origin[0]+(voxel_x+0.5f)*voxel_size;
            float pt_y = vol_origin[1]+(voxel_y+0.5f)*voxel_size;
            float pt_z = vol_origin[2]+(voxel_z+0.5f)*voxel_size;
            
            // transform TSDF to SDF[voxel_idx]
            float trunc_margin = other_params[2];
            int pc_num = (int) other_params[3];
            float sdf = tsdf*trunc_margin;
            
            //int count = (int) pc_count[0];
            
            int count = voxel_idx % pc_num;
            
            //printf("pc_num:%d,count:%d\\n",pc_num,count);
            truncated_pc[count*7+0] = pt_x;
            truncated_pc[count*7+1] = pt_y;
            truncated_pc[count*7+2] = pt_z;
            truncated_pc[count*7+3] = sdf;
            truncated_pc[count*7+4] = old_r;
            truncated_pc[count*7+5] = old_g;
            truncated_pc[count*7+6] = old_b;
            
            atomicAdd_system(pc_count,1);
            
            }
            
            __global__ void clean_tsdf(float * tsdf_vol,
                                        float * weight_vol,
                                        float * color_vol,
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
            tsdf_vol[voxel_idx] = 1.0;
            weight_vol[voxel_idx]=0;
            color_vol[voxel_idx]=0;
            }
            
            __global__ void copy_volume(float * tsdf_vol,
                                        float * weight_vol,
                                        float * color_vol,
                                        float * tsdf_vol_back,
                                        float * weight_vol_back,
                                        float * color_vol_back,
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
            tsdf_vol_back[voxel_idx] = tsdf_vol[voxel_idx];
            weight_vol_back[voxel_idx] = weight_vol[voxel_idx];
            color_vol_back[voxel_idx] = color_vol[voxel_idx];
            }
            """)

        self.cuda_swap_rot_trans = self.cuda_src_mod.get_function("swap_rot_trans")
        self.cuda_integrate = self.cuda_src_mod.get_function("integrate")
        self.cuda_clean_tsdf = self.cuda_src_mod.get_function("clean_tsdf")
        self.cuda_get_truncated_pc = self.cuda_src_mod.get_function("get_truncated_pc")
        self.cuda_filter_tsdf = self.cuda_src_mod.get_function("filter_tsdf")
        self.cuda_copy_volume = self.cuda_src_mod.get_function("copy_volume")
        self.cuda_tri_intepolate = self.cuda_src_mod.get_function("tri_intepolate")
        

    def get_truncated_pc(self,pc_num=5000000,trunc_tsdf=0.5): 
      """
      get_truncated_pc from the TSDF volume to help MLP learn better tsdf.
      """
      truncated_pc=np.zeros((pc_num*7)).astype(np.float32) 
      pc_count=np.asarray([0]).astype(np.float32)

      for gpu_loop_idx in range(self.n_gpu_loops):
        self.cuda_get_truncated_pc(self.tsdf_vol_gpu,
                                   self.color_vol_gpu,
                            cuda.In(self.vol_dim.astype(np.float32)),
                            cuda.In(self.vol_origin.astype(np.float32)),
                            cuda.InOut(truncated_pc),
                            cuda.InOut(pc_count),
                            cuda.In(np.asarray([
                              gpu_loop_idx,
                              self.voxel_size,
                              self.trunc_margin, 
                              pc_num,
                              trunc_tsdf, 
                            ], np.float32)),
                            block=(self.max_gpu_threads_per_block,1,1),
                            grid=(
                              int(self.max_gpu_grid_dim[0]),
                              int(self.max_gpu_grid_dim[1]),
                              int(self.max_gpu_grid_dim[2]),
                            )
        )
      truncated_pc=truncated_pc.reshape((-1,7))
      valid_index = (truncated_pc[:,0]!=0.0) & (truncated_pc[:,1]!=0.0) & (truncated_pc[:,2]!=0.0)
      left_pc = truncated_pc[valid_index,:]
      return left_pc
  
    
    def clean_volume(self):
        """
        Reset (clean) the TSDF, weight, and color volumes on the GPU.
        This function iterates over all GPU loop slices, calling the CUDA kernel
        to zero out or refresh the volume data structures, preparing them for new integration.
        """
        for gpu_loop_idx in range(self.n_gpu_loops):
            self.cuda_clean_tsdf(
                self.tsdf_vol_gpu,
                self.weight_vol_gpu,
                self.color_vol_gpu,
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
    
    def update_tsdf_swap_clean(self, vol_bnds, old_bnds):
        """
        Update the TSDF volume bounds during a swap and clean the volume data on the GPU.

        This function sets the new volume bounds, updates the volume dimensions and origin,
        and then iterates over all GPU loop slices to reset (clean) the TSDF, weight, and color
        volumes. This prepares the data structures for new integration after a volume move/swap.

        Args:
            vol_bnds (np.ndarray): New volume bounds (3x2 array).
            old_bnds (np.ndarray): Previous volume bounds (3x2 array), not used in this function but kept for compatibility.
        """
        self.vol_bnds = vol_bnds
        self.vol_dim = np.ceil((self.vol_bnds[:,1] - self.vol_bnds[:,0]) / self.voxel_size).copy(order='C').astype(int)
        self.vol_bnds[:,1] = self.vol_bnds[:,0] + self.vol_dim * self.voxel_size
        self.vol_origin = self.vol_bnds[:,0].copy(order='C').astype(np.float32)
 
        for gpu_loop_idx in range(self.n_gpu_loops):
            self.cuda_clean_tsdf(
                self.tsdf_vol_gpu,
                self.weight_vol_gpu,
                self.color_vol_gpu,
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
    
    def integrate(self, color_im, depth_im, cam_intr, cam_pose, old_bnd, obs_weight=1.,reintegrate_flag=0.0):
        """Integrate an RGB-D frame into the TSDF volume.
        Args:
            color_im (ndarray): An RGB image of shape (H, W, 3).
            depth_im (ndarray): A depth image of shape (H, W).
            cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
            cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
            obs_weight (float): The weight to assign for the current observation. A higher
            value
        """
        im_h, im_w = depth_im.shape
        # Fold RGB color image into a single channel image
        color_im = color_im.astype(np.float32) #RGB feels right

        
        color_im = np.floor(color_im[...,2]*self.color_const + color_im[...,1]*256 + color_im[...,0])
        for gpu_loop_idx in range(self.n_gpu_loops):
            self.cuda_integrate(self.tsdf_vol_gpu,
                                self.weight_vol_gpu,
                                self.color_vol_gpu,
                                cuda.In(self.vol_dim.astype(np.float32)),
                                cuda.In(self.vol_origin.astype(np.float32)),
                                cuda.In(cam_intr.reshape(-1).astype(np.float32)),
                                cuda.In(cam_pose.reshape(-1).astype(np.float32)),
                                cuda.In(np.asarray([
                                gpu_loop_idx,
                                self.voxel_size,
                                im_h,
                                im_w,
                                self.trunc_margin,
                                obs_weight,
                                reintegrate_flag,
                                self.weight_clamp
                                ], np.float32)),
                                cuda.In(old_bnd.reshape(-1).astype(np.float32)),
                                cuda.In(color_im.reshape(-1).astype(np.float32)),
                                cuda.In(depth_im.reshape(-1).astype(np.float32)),
                                block=(self.max_gpu_threads_per_block,1,1),
                                grid=(
                                int(self.max_gpu_grid_dim[0]),
                                int(self.max_gpu_grid_dim[1]),
                                int(self.max_gpu_grid_dim[2]),
                                )
            )
        # print("integrate time",time.time()-time_s)

    
    def tri_interpolate(self, query_pc):
        """tri_interpolate sdf and rgb value according to query_pc.
        Args:
            query_pc (ndarray): point cloud (N, 3).
        """
        tsdf_rgb = np.zeros((query_pc.shape[0],5)) #(88064, 4)
        tsdf_rgb[:,0]=1.0
        num_pc = query_pc.shape[0]
        query_pc_np = query_pc.reshape(-1).astype(np.float32)
        tsdf_rgb_np = tsdf_rgb.reshape(-1).astype(np.float32)
        # print("int(num_pc/(32*32) +1)",int(num_pc/(32*32) +1))
        self.cuda_tri_intepolate(self.tsdf_vol_gpu,
                                 self.weight_vol_gpu,
                            self.color_vol_gpu,
                            cuda.In(query_pc_np),
                            cuda.InOut(tsdf_rgb_np),
                            cuda.In(self.vol_dim.astype(np.float32)),
                            cuda.In(self.vol_origin.astype(np.float32)),
                            cuda.InOut(np.asarray([
                            num_pc,
                            self.voxel_size,
                            self.trunc_margin,
                            ], np.float32)),
                            block=(1024,1,1),
                            grid=(
                            int(num_pc/(32*32) +1),
                            1,
                            1,
                            )
        )
        result = tsdf_rgb_np.reshape(-1,5)
        notvalid_index = (result[:,0]==10.0) & (result[:,1]==0.0) & (result[:,2]==0.0) & (result[:,3]==0.0)
        mask = ~notvalid_index

        return result,mask
    
    def update_tsdf_swap_rot_trans(self, vol_bnds, old_bnds):
        """
        Update TSDF, weight, and color volumes by swapping voxel blocks between coordinate frames 
        when the TSDF volume boundary moves (due to large camera motion or tracking area change).

        Args:
            vol_bnds (np.ndarray): New volume boundaries (3, 2 array).
            old_bnds (np.ndarray): Previous volume boundaries (3, 2 array).
        
        This function recalculates the volume dimensions and origins based on the new bounding box,
        and then performs block-wise data transfer using CUDA to rotate/translate/swap TSDF, weight, 
        and color information from the old volume to the new one.
        """

        start = time.time()

        # Set the new volume bounds
        self.vol_bnds = vol_bnds

        # Compute new volume dimensions and adjust boundaries to match whole voxel multiples
        self.vol_dim = np.ceil((self.vol_bnds[:, 1] - self.vol_bnds[:, 0]) / self.voxel_size).copy(order='C').astype(int)
        self.vol_bnds[:, 1] = self.vol_bnds[:, 0] + self.vol_dim * self.voxel_size
        self.vol_origin = self.vol_bnds[:, 0].copy(order='C').astype(np.float32)

        # Get old volume origin and dimensions
        old_origin = old_bnds[:, 0].copy(order='C').astype(np.float32)
        old_vol_dim = np.ceil((old_bnds[:, 1] - old_bnds[:, 0]) / self.voxel_size).copy(order='C').astype(int)

        endcopy = time.time()
        copy_time = round(endcopy - start, 3)
        # print("copy_time time: ", copy_time)

        # For each CUDA block loop, swap the current and backup (old) volumes
        for gpu_loop_idx in range(self.n_gpu_loops):
            self.cuda_swap_rot_trans(
                self.tsdf_vol_gpu,
                self.tsdf_vol_gpu_back,
                self.weight_vol_gpu,
                self.weight_vol_gpu_back,
                self.color_vol_gpu,
                self.color_vol_gpu_back,
                cuda.In(self.vol_dim.astype(np.float32)),
                cuda.In(self.vol_origin.astype(np.float32)),
                cuda.In(old_origin.astype(np.float32)),
                cuda.In(old_vol_dim.astype(np.float32)),
                cuda.In(np.asarray([
                    gpu_loop_idx,
                    self.voxel_size,
                ], np.float32)),
                block=(self.max_gpu_threads_per_block, 1, 1),
                grid=(
                    int(self.max_gpu_grid_dim[0]),
                    int(self.max_gpu_grid_dim[1]),
                    int(self.max_gpu_grid_dim[2]),
                )
            )

        end = time.time()
        actual_time = round(end - start, 3)
        # print("swap time: ", actual_time)
        
    def filter_tsdf(self, weight_threshold):
        """
        Apply a filter to the TSDF volume using a weight threshold.

        This function launches a CUDA kernel to process the TSDF, weight, and color volumes,
        filtering out voxels whose weights are below the given threshold.

        Args:
            weight_threshold (float): Voxels with a weight lower than this value will be filtered.
        """
        self.cuda_filter_tsdf(
            self.tsdf_vol_gpu,
            self.weight_vol_gpu,
            self.color_vol_gpu,
            cuda.In(self.vol_dim.astype(np.float32)),
            cuda.In(np.asarray([
                weight_threshold
            ], np.float32)),
            block=(self.max_gpu_threads_per_block, 1, 1),
            grid=(
                int(self.max_gpu_grid_dim[0]),
                int(self.max_gpu_grid_dim[1]),
                int(self.max_gpu_grid_dim[2]),
            )
        )

    def copy_volume(self):
        """
        Copy the current TSDF, weight, and color volumes to the backup storage on the GPU.

        This function iterates over all GPU loop slices, calling the CUDA kernel to transfer
        the current volume data to the backup storage. This ensures that the backup volumes
        are always up-to-date with the current active volume, ready for swap operations.
        """
        for gpu_loop_idx in range(self.n_gpu_loops):
            self.cuda_copy_volume(self.tsdf_vol_gpu,
                                self.weight_vol_gpu,
                                self.color_vol_gpu,
                                self.tsdf_vol_gpu_back,
                                self.weight_vol_gpu_back,
                                self.color_vol_gpu_back,
                                cuda.In(self.vol_dim.astype(np.float32)),
                                cuda.In(np.asarray([
                                    gpu_loop_idx 
                                ], np.float32)),
                                block=(self.max_gpu_threads_per_block,1,1),
                                grid=(
                                int(self.max_gpu_grid_dim[0]),
                                int(self.max_gpu_grid_dim[1]),
                                int(self.max_gpu_grid_dim[2]),
                                )
                            )
    
    def initialize_vol_bnd(self,cam_pose_iter,traj,version):
        """
        Initialize the volume bounds based on the camera pose and trajectory.

        This function determines the appropriate volume bounds for the current camera position
        and orientation, using either a center-based approach or a more detailed angle-based
        calculation. The bounds are returned as a 3x2 array, with each row representing a
        dimension (x, y, z) and the two columns representing the minimum and maximum bounds.
        """
        vol_bnds = np.zeros((3,2))
        if version == "center":
            vol_bnds=self.center_volbnd(vol_bnds,cam_pose_iter,traj)
        else:
            vol_bnds=self.more_volbnd(vol_bnds,cam_pose_iter,traj)
        
        return vol_bnds
    

    
    
    def check_move_volume_new(self, cur_id, cam_pose_iter, traj, version="center", larger_flag=False, get_pc=False, gap=100):
        """
        Check whether the TSDF volume needs to be moved and swapped due to significant camera translation or orientation.

        This function determines if the movement of the camera (translation or rotation) exceeds specified thresholds,
        indicating the need to update the TSDF volume bounds and swap the stored volumes accordingly.

        Args:
            cur_id (int): The current frame or camera index.
            cam_pose_iter (np.ndarray): Current camera pose matrix (4x4).
            traj: Trajectory object holding previous reference positions and axis info.
            version (str): Swap logic version, 'center' for translation only, 'more' for considering rotation.
            larger_flag (bool): Whether to use a larger angular threshold for swapping in 'more' mode.
            get_pc (bool): If True, call get_truncated_pc() and store the surface point cloud.
            gap (int): Minimal interval between consecutive point cloud extractions.

        Returns:
            (flag, old_bnds):
                flag (bool): True if the volume was moved/swapped; False otherwise.
                old_bnds (np.ndarray): The previous volume bounds.

        """
        flag = False
        AX = ["x", "y", "z"]
        center_cam = np.round(cam_pose_iter[:3, 3], 0)
        cam_ori = np.asarray([[0], [0], [1]]).astype(np.float32)  # Camera forward axis in local frame
        x_vec = np.asarray([1, 0, 0]).astype(np.float32)
        y_vec = np.asarray([0, 1, 0]).astype(np.float32)
        z_vec = np.asarray([0, 0, 1]).astype(np.float32)

        # Check translation thresholds for each axis
        t_outx = np.abs(cam_pose_iter[0, 3] - traj.kfx) > self.t_treshold
        t_outy = np.abs(cam_pose_iter[1, 3] - traj.kfy) > self.t_treshold
        t_outz = np.abs(cam_pose_iter[2, 3] - traj.kfz) > self.t_treshold
        trans_vector = [
            cam_pose_iter[0, 3] - traj.kfx,
            cam_pose_iter[1, 3] - traj.kfy,
            cam_pose_iter[2, 3] - traj.kfz,
        ]

        old_bnds = copy.deepcopy(self.vol_bnds)
        tmp_bnds = copy.deepcopy(self.vol_bnds)  # Copy current vol_bnds for tentative updates

        # Check if translation along any axis exceeds threshold for swap
        move_flag, movex_flag, movey_flag, movez_flag = False, False, False, False

        if t_outx and not self.fix_x:
            tmp_bnds[0, :] += trans_vector[0]
            traj.kfx = cam_pose_iter[0, 3]
            movex_flag = True
        if t_outy and not self.fix_y:
            tmp_bnds[1, :] += trans_vector[1]
            traj.kfy = cam_pose_iter[1, 3]
            movey_flag = True
        if t_outz and not self.fix_z:
            tmp_bnds[2, :] += trans_vector[2]
            traj.kfz = cam_pose_iter[2, 3]
            movez_flag = True

        move_flag = movex_flag or movey_flag or movez_flag

        if move_flag:
            # Round updated bounds to integers for consistency
            for dim in range(3):
                tmp_bnds[dim, 0] = round(tmp_bnds[dim, 0], 0)
                tmp_bnds[dim, 1] = round(tmp_bnds[dim, 1], 0)

            # If tentative bounds differ from original, proceed with swap
            if not (tmp_bnds == old_bnds).all():
                flag = True
                self.copy_volume()
                self.update_tsdf_swap_rot_trans(tmp_bnds, old_bnds)
                # print("swap volume")
                # self.print_bnd(tmp_bnds, old_bnds)

        # If using the "more" strategy, also consider camera orientation for swapping
        if version == "more":
            tmp_bnds = copy.deepcopy(self.vol_bnds)
            axis2num = {"x": 0, "y": 1, "z": 2}

            # Get world camera direction
            cam_R = cam_pose_iter[:3, :3]
            cam_dir_w = np.matmul(cam_R, cam_ori).squeeze()  # (3,)

            # Project onto each axis and calculate angles
            angelx, flagx = self.require_angle_projection(cam_dir_w, x_vec)
            angely, flagy = self.require_angle_projection(cam_dir_w, y_vec)
            angelz, flagz = self.require_angle_projection(cam_dir_w, z_vec)

            # Order axes by angle
            angel_list = [angelx, angely, angelz]
            flag_list = [flagx, flagy, flagz]
            sorted_angel = sorted(angel_list)
            first, second, third = (
                angel_list.index(sorted_angel[0]),
                angel_list.index(sorted_angel[1]),
                angel_list.index(sorted_angel[2]),
            )
            first_flag, second_flag, third_flag = (
                flag_list[first],
                flag_list[second],
                flag_list[third],
            )
            axis_priority = [first, second, third]
            axis_flag = [first_flag, second_flag, third_flag]
            first_angel = sorted_angel[0]

            angel_threshold = self.more_angel_t
            if larger_flag:
                angel_threshold *= 2

            # If dominant orientation axis changed and within angular threshold, swap
            if first != traj.first and first_angel < angel_threshold:
                traj.kfx = cam_pose_iter[0, 3]
                traj.kfy = cam_pose_iter[1, 3]
                traj.kfz = cam_pose_iter[2, 3]

                vol_bnds = self.more_calculations(
                    tmp_bnds, axis_priority, axis_flag, center_cam
                )
                # If a fixed axis is specified, override those bounds
                if self.fixed_axis is not None:
                    fixed = axis2num[self.fixed_axis]
                    vol_bnds[fixed, 0] = self.fixed_range[0]
                    vol_bnds[fixed, 1] = self.fixed_range[1]

                if not (vol_bnds == old_bnds).all():
                    print("*****moving*****")
                    print(
                        "first,second,first_flag,first_angel,tsdf_cam.first"
                    )
                    print(
                        "debug:",
                        first,
                        second,
                        first_flag,
                        first_angel,
                        traj.first,
                    )
                    print("MORE SWAP")
                    # Optionally extract and store surface point cloud if enough frames have passed
                    if get_pc and (cur_id - self.last_pcid) > gap:
                        self.last_pcid = cur_id
                        surface_pc = self.get_truncated_pc()  # N,7
                        # TODO: try if there is forgettable problem
                        self.surface_pc = surface_pc
                        print("more surface_pc:", surface_pc.shape)

                    self.update_tsdf_swap_rot_trans(vol_bnds, old_bnds)

                    traj.first = first
                    flag = True
        return flag, old_bnds
    
    def frameid_to_Vrange(self,value):
        """
        Map a frame index to the corresponding volume boundary range.

        This function checks if the frame index is within any previously recorded volume range,
        and returns the corresponding volume boundaries if found. If the frame is not in any
        recorded range, it returns the current active volume boundaries.

        Args:
            value (int): The frame index to map.

        Returns:
            np.ndarray: The volume boundaries (3x2 array) corresponding to the frame index.
        """
        # print("self.frame_to_Vrange:",self.frame_to_Vrange,bool(self.frame_to_Vrange))
        if not bool(self.frame_to_Vrange):
            return self.vol_bnds
        for start, end in self.frame_to_Vrange.keys():
            if value >= start and value <= end:
                return self.frame_to_Vrange[(start, end)]
        # not in above range, it indicates this frame is newst,so return current vol_bnds
        return self.vol_bnds
        
    
    

    def more_calculations(self, vol_bnds, axis_priority, axis_flag, center_cam):
        """
        Calculate the volume boundaries based on dominant axis and center camera position.

        This function determines the appropriate volume boundaries for the current camera position
        and orientation, using the dominant axis and center camera position. The boundaries are
        calculated based on the first, second, and third axes, with the first axis being the dominant
        one. The boundaries are returned as a 3x2 array, with each row representing a dimension (x, y, z)
        and the two columns representing the minimum and maximum bounds.
        """
        first,second,third = axis_priority[0],axis_priority[1],axis_priority[2]
        
        vol_bnds[first,0] = center_cam[first]-np.floor(self.first_len/2)*axis_flag[0]-(np.ceil(self.first_len/2) + self.first_len)*(not axis_flag[0])
        vol_bnds[first,1] = center_cam[first]+(np.ceil(self.first_len/2) + self.first_len)*axis_flag[0] +np.floor(self.first_len/2)*(not axis_flag[0])
    
        vol_bnds[second,0] = center_cam[second] - self.second_len
        vol_bnds[second,1] = center_cam[second] + self.second_len
        
        vol_bnds[third,0] = center_cam[third] - self.third_len
        vol_bnds[third,1] = center_cam[third] + self.third_len
        
        return vol_bnds
    
    def center_volbnd(self,vol_bnds,cam_pose_iter,tsdf_cam):
        vol_bnds = np.zeros((3,2))
        tsdf_cam.kfx = cam_pose_iter[0,3]
        tsdf_cam.kfy = cam_pose_iter[1,3]
        tsdf_cam.kfz = cam_pose_iter[2,3]
        center_cam = np.round(cam_pose_iter[:3,3],0)
        # calculate the fixed axis bound range
        vol_bnds[0,0] = center_cam[0]-self.x_len
        vol_bnds[0,1] = center_cam[0]+self.x_len
        
        vol_bnds[1,0] = center_cam[1]-self.y_len
        vol_bnds[1,1] = center_cam[1]+self.y_len
        
        vol_bnds[2,0] = center_cam[2]-self.z_len
        vol_bnds[2,1] = center_cam[2]+self.z_len
        
        return vol_bnds
    
    def more_volbnd(self,vol_bnds,cam_pose_iter,tsdf_cam):
        vol_bnds = np.zeros((3,2))
        # update tsdf_cam.kfx y z
        tsdf_cam.kfx = cam_pose_iter[0,3]
        tsdf_cam.kfy = cam_pose_iter[1,3]
        tsdf_cam.kfz = cam_pose_iter[2,3]
        center_cam = np.round(cam_pose_iter[:3,3],0)
        axis2num = {"x":0,"y":1,"z":2}
        self.fixed_axis=None
        if self.fix_x:
            self.fixed_axis='x'
            self.fixed_range=self.x_range
        if self.fix_y:
            self.fixed_axis='y'
            self.fixed_range=self.y_range
        if self.fix_z:
            self.fixed_axis='z'
            self.fixed_range=self.z_range
        if self.fixed_axis is not None:
            fixed = axis2num[self.fixed_axis]
            vol_bnds[fixed,0] = self.fixed_range[0]
            vol_bnds[fixed,1] = self.fixed_range[1]
        
        # calculate bnds according to angle
        cam_ori=np.asarray([[0],[0],[1]]).astype(np.float32) #3,1
        x_vec = np.asarray([1,0,0]).astype(np.float32) #3, 
        y_vec = np.asarray([0,1,0]).astype(np.float32) #3, 
        z_vec = np.asarray([0,0,1]).astype(np.float32) #3, 
        #camera frustum direction
        cam_R=cam_pose_iter[:3,:3] 

        cam_dir_w =np.matmul(cam_R,cam_ori).squeeze() #3, 
        # print("cam_R:",cam_R, cam_dir_w, self.fixed_axis)
        angelx,flagx = self.require_angle_projection(cam_dir_w, x_vec,fixed=self.fixed_axis )
        angely,flagy = self.require_angle_projection(cam_dir_w, y_vec,fixed=self.fixed_axis)
        angelz,flagz = self.require_angle_projection(cam_dir_w, z_vec,fixed=self.fixed_axis)
        # select the closer axis as the first
        angel_list = [angelx,angely,angelz]
        flag_list = [flagx,flagy,flagz]
        sorted_angel = sorted(angel_list)
        first,second,third=angel_list.index(sorted_angel[0]),angel_list.index(sorted_angel[1]),angel_list.index(sorted_angel[2])
        first_flag,second_flag,third_flag=flag_list[first],flag_list[second],flag_list[third]
        axis_priority = [first,second,third]
        axis_flag = [first_flag,second_flag,third_flag]
        # update tsdf_cam.first axis
        tsdf_cam.first = first
        vol_bnds = self.more_calculations(vol_bnds,axis_priority,axis_flag,center_cam)
        if self.fixed_axis is not None:
            fixed = axis2num[self.fixed_axis]
            vol_bnds[fixed,0] = self.fixed_range[0]
            vol_bnds[fixed,1] = self.fixed_range[1]
        return vol_bnds
    
    def require_angle(self,x,y,absolute=False):
        """
        Calculate the angle between two vectors and determine if it is positive or negative.

        This function computes the cosine of the angle between two vectors, x and y, and then
        calculates the angle in radians and degrees. If the absolute flag is False, it returns
        the angle value and a flag indicating if the angle is positive or negative. If the
        absolute flag is True, it returns only the angle value.
        """

        module_x=np.sqrt(x.dot(x))
        module_y=np.sqrt(y.dot(y))

        dot_value=x.dot(y)

        cos_theta=dot_value/(module_x*module_y+1e-3)

        angle_radian=np.arccos(cos_theta)

        angle_value=angle_radian*180/np.pi
        
        if absolute == False:
            flag = 1
            if angle_value>90:
                angle_value = 180-angle_value
                flag = -1
            return angle_value,flag
        else:
            return angle_value 
    
    
    def require_angle_projection(self,x,y,absolute=False,fixed="z"):
        if fixed == "x":
            a = x[1:]
            b = y[1:]
        if fixed == "y":
            a = x[0::2]
            b = y[0::2]
        if fixed == "z":
            a = x[:2]
            b = y[:2]
        
        if absolute==False:
            angel,flag=self.require_angle(a,b,absolute)
            return angel,flag
        if absolute==True:
            angel=self.require_angle(a,b,absolute)
            return angel
    
    def print_bnd(self,vol_bnds,old_bnds):
        """
        Print the volume boundaries for x, y, and z dimensions.

        This function prints the previous and new volume boundaries for each dimension,
        formatted to one decimal place. It helps visualize the changes in volume boundaries
        during tracking or volume updates.
        """
        print(f"x:[{old_bnds[0,0]:.1f},{old_bnds[0,1]:.1f}] -> [{vol_bnds[0,0]:.1f},{vol_bnds[0,1]:.1f}]")
        print(f"y:[{old_bnds[1,0]:.1f},{old_bnds[1,1]:.1f}] -> [{vol_bnds[1,0]:.1f},{vol_bnds[1,1]:.1f}]")
        print(f"z:[{old_bnds[2,0]:.1f},{old_bnds[2,1]:.1f}] -> [{vol_bnds[2,0]:.1f},{vol_bnds[2,1]:.1f}]")
    
    def get_volume_all(self):
        """
        Retrieve the TSDF, weight, and color volumes from the GPU to the CPU.

        This function copies the GPU-resident TSDF, weight, and color volume data
        to the CPU memory for further processing or visualization. It ensures that
        the data is synchronized between the GPU and CPU for seamless integration.
        """
        if self.gpu_mode:
            cuda.memcpy_dtoh(self.tsdf_vol_cpu, self.tsdf_vol_gpu)
            cuda.memcpy_dtoh(self.weight_vol_cpu, self.weight_vol_gpu)
            cuda.memcpy_dtoh(self.color_vol_cpu, self.color_vol_gpu)
        return self.tsdf_vol_cpu, self.weight_vol_cpu, self.color_vol_cpu
    
    
    def get_mesh(self):
        """
        Compute a mesh from the voxel volume using marching cubes.
        """
        tsdf_vol, weight_vol, color_vol = self.get_volume_all() #300，200，300
        tsdf_vol = tsdf_vol.reshape(self.vol_dim[0],self.vol_dim[1],self.vol_dim[2])
        weight_vol = weight_vol.reshape(self.vol_dim[0],self.vol_dim[1],self.vol_dim[2])
        color_vol = color_vol.reshape(self.vol_dim[0],self.vol_dim[1],self.vol_dim[2])
        
        w_x = np.ones_like(weight_vol)
        w_y = np.ones_like(weight_vol)
        w_z = np.ones_like(weight_vol)
        for i in range(w_x.shape[0]-1):
            w_x[i,...]=w_x[i+1,...]
        for i in range(w_y.shape[1]-1):
            w_y[:,i,:]=w_y[:,i+1,:]
        for i in range(w_z.shape[2]-1):
            w_z[:,:,i]=w_z[:,:,i+1]
        # Marching cubes
        mask = (weight_vol>0) & (w_x>0) & (w_y>0) & (w_z>0) & (tsdf_vol>-0.99) & (tsdf_vol<0.99) & (tsdf_vol>-0.5)
        # verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0 , mask=mask)
        verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0 )#, mask=mask)
        # verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0 , mask=mask)
        verts_ind = np.round(verts).astype(int)
        verts = verts*self.voxel_size+self.vol_origin  # voxel grid coordinates to world coordinates

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:,0], verts_ind[:,1], verts_ind[:,2]]
        colors_b = np.floor(rgb_vals/self.color_const)
        colors_g = np.floor((rgb_vals-colors_b*self.color_const)/256)
        colors_r = rgb_vals-colors_b*self.color_const-colors_g*256
        colors = np.floor(np.asarray([colors_r,colors_g,colors_b])).T
        colors = colors.astype(np.uint8)
        return verts, faces, norms, colors
    

    def get_point_cloud(self):
        """
        Extract a point cloud from the voxel volume.
        """
        tsdf_vol, weight_vol, color_vol = self.get_volume_all() #300，200，300
        tsdf_vol = tsdf_vol.reshape(self.vol_dim[0],self.vol_dim[1],self.vol_dim[2])
        weight_vol = weight_vol.reshape(self.vol_dim[0],self.vol_dim[1],self.vol_dim[2])
        color_vol = color_vol.reshape(self.vol_dim[0],self.vol_dim[1],self.vol_dim[2])
        
        mask = (weight_vol>0)
        
        # Marching cubes
        print("\n===============saving tsdf volume pc=================")
        print("MAX:",tsdf_vol.max(),"MIN:",tsdf_vol.min(),"origin:",self.vol_origin)
        verts = measure.marching_cubes(tsdf_vol, level=0, mask=mask)[0]
        #NEW delete not updated -truncated region
        verts_ind = np.round(verts).astype(int)

        verts = verts*self.voxel_size + self.vol_origin
        
        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self.color_const)     
        colors_g = np.floor((rgb_vals - colors_b*self.color_const) / 256)
        colors_r = rgb_vals - colors_b*self.color_const - colors_g*256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)

        pc = np.hstack([verts, colors])
        
        return pc
    
    def pcwrite(self,filename, xyzrgb):
        """
        Save a point cloud to a polygon .ply file.
        """
        xyz = xyzrgb[:, :3]
        rgb = xyzrgb[:, 3:].astype(np.uint8)

        # Write header
        ply_file = open(filename,'w')
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex %d\n"%(xyz.shape[0]))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")

        for i in range(xyz.shape[0]):
                ply_file.write("%f %f %f %d %d %d\n"%(
            xyz[i, 0], xyz[i, 1], xyz[i, 2],
            rgb[i, 0], rgb[i, 1], rgb[i, 2],
            ))
                
    def meshwrite(self, filename, verts, faces, norms, colors):
        """
        Save a 3D mesh to a polygon .ply file.
        """
        # Write header
        ply_file = open(filename,'w')
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex %d\n"%(verts.shape[0]))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property float nx\n")
        ply_file.write("property float ny\n")
        ply_file.write("property float nz\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("element face %d\n"%(faces.shape[0]))
        ply_file.write("property list uchar int vertex_index\n")
        ply_file.write("end_header\n")

        # Write vertex list
        for i in range(verts.shape[0]):
            ply_file.write("%f %f %f %f %f %f %d %d %d\n"%(
            verts[i,0], verts[i,1], verts[i,2],
            norms[i,0], norms[i,1], norms[i,2],
            colors[i,0], colors[i,1], colors[i,2],
            ))

        # Write face list
        for i in range(faces.shape[0]):
            ply_file.write("3 %d %d %d\n"%(faces[i,0], faces[i,1], faces[i,2]))

        ply_file.close()
        
if __name__ == '__main__':
    cfg = {}
    mv =moving_volume(cfg,None,None)