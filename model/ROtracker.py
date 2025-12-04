import torch
from tqdm import tqdm
import random
from model.Volume import moving_volume
import random
import cv2
import os
import random
from copy import deepcopy
import numpy as np
from model.utils import compute_loss, check_orthogonal, orthogonalize_rotation_matrix, orthogonalize_rotation_matrix_tolerate
from model.traj import Trajectory
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
from datasets.utils import get_camera_rays, alphanum_key, as_intrinsics_matrix

class Holder(cuda.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()
    def get_pointer():
        return self.t.data_ptr()

class ROTracker(object):
    def __init__(self, cfg, data_stream) -> None:
        super(ROTracker, self).__init__()
        self.cfg = cfg
        self.last_kfframe_id = 0
        self.last_frame = None

        self.data_stream = data_stream
        # General configuration: saving output, experiment/paths, scaling and verbosity
        self.save = cfg["RO"]["save_volume"]
        self.save_path = os.path.join(os.path.join(self.cfg['data']['output'], self.cfg['data']['exp_name']))
        if self.save:
            if not os.path.exists(os.path.join(self.save_path, "RO")):
                os.mkdir(os.path.join(self.save_path, "RO/"))
        self.estimate_pose = []
        self.depth_scale = cfg["cam"]["png_depth_scale"]
        self.larger_flag = False      

        # Optimization and search configuration
        self.init_size = cfg["RO"]["init_size"]
        self.scaling_coefficient = cfg["RO"]["scaling_coefficient"]
        self.particle_iter_lens = cfg["RO"]["particle_iter_lens"]
        self.PST_path = cfg["RO"]["PST_path"]
        self.PST_size = cfg["RO"]["PST_size"]
        self.fix_level_index = cfg["RO"]["fix_level_index"]
        self.count_search = cfg["RO"]["count_search"]
        self.filter_weight = cfg["RO"]["filter_weight"]
        self.mlp_trunc = cfg["training"]["trunc"]
        self.cut = cfg["RO"]["cut"]
        self.cut_dist = cfg["RO"]["cut_dist"]
        self.truncation = cfg['volume']['trunc']
        self.sample_range = cfg['RO']['sample_range']
        self.iterative_scale = cfg['RO']['iterative_scale']
        self.get_pc = cfg["training"]["surface_weight"] > 0 
        # Initialize trajectory recorder and dataset boundaries
        tracking_path = "./results/"
        self.traj = Trajectory(tracking_path)
        self.start_frame = 0
        self.end_frame = len(self.data_stream)

        # Load initial frame, pose, and set up local TSDF volume
        init_batch = self.data_stream[0]
        init_pose = init_batch["c2w"].squeeze().cpu().numpy()
        if cfg["dataset"] == "Largeindoor":
            init_pose = np.array([[0., 0., 1., 0.],
                                  [-1., 0., 0., 0.],
                                  [0., -1., 0., 0.],
                                  [0., 0., 0., 1.]])
        depth_np = init_batch["depth"].squeeze().cpu().numpy()
        rgb_np = np.floor((init_batch["rgb"].squeeze().cpu().numpy()) * 255.)
        self.RO_pose = []
        self.MV = moving_volume(cfg, self.traj, init_pose, start=0)

        # Image dimensions and camera intrinsics
        self.lastkfid = 0
        self.im_h = self.data_stream.H
        self.im_w = self.data_stream.W
        self.K = np.array([
            [self.data_stream.fx, 0.0, self.data_stream.cx],
            [0.0, self.data_stream.fy, self.data_stream.cy],
            [0.0, 0.0, 1.0]
        ])

        # print("self.im_h:", self.im_h, "self.im_w:", self.im_w)
        # print("RO k:", self.K)

        # GPU buffer initialization for vertex and normal maps
        self.depth_vertex = np.ones(self.im_h * self.im_w * 4, dtype=np.float32)
        self.normal_vertex = np.ones(self.im_h * self.im_w * 3, dtype=np.float32)


        self.depth_vertex_gpu = cuda.mem_alloc(self.depth_vertex.nbytes)
        self.normal_vertex_gpu = cuda.mem_alloc(self.normal_vertex.nbytes)
        self.depth_map_gpu = cuda.mem_alloc(int(self.depth_vertex.nbytes / 3))


        # States for saving, visualization, as well as search/optimization
        self.kfvis_id = 0
        self.save_volume = cfg['RO']["save_volume"]
        self.save_freq = cfg['RO']["save_freq"]
        self.move_frameid = 0
        self.initialize_search_size = np.zeros((6))
        self.previous_frame_success = False
        self.tiff_index = [
            0, 1 + 20, 2 + 40, 3, 4 + 20, 5 + 40, 6 + 0, 7 + 20, 8 + 40,
            9 + 0, 10 + 20, 11 + 40, 12 + 0, 13 + 20, 14 + 40,
            15 + 0, 16 + 20, 17 + 40, 18 + 0, 19 + 20
        ]
        self.depth_level = [32, 16, 8, 32, 16, 8, 32, 16, 8, 32, 16, 8, 32, 16, 8, 32, 16, 8, 32, 16]
        self.downsample = [int(self.im_h / i) * int(self.im_w / i) for i in self.depth_level]

        # Load PST for multi-resolution search
        self.readpst(self.PST_path, self.PST_size)

        # Global transformation state
        self.current_global_R = np.zeros((3, 3), dtype=np.float32)
        self.current_global_T = np.zeros((3), dtype=np.float32)

        # Integrate initial frame to TSDF volume
        self.MV.integrate(rgb_np, depth_np, self.K, init_pose, self.MV.vol_bnds, obs_weight=1.)

        # Optionally, save the initial mesh
        if self.cfg["RO"]["save_volume"]:
            verts, faces, norms, colors = self.MV.get_mesh()
            tsdfpc_path = os.path.join(self.save_path, "RO" + "/" + str(0) + ".ply")
            self.MV.meshwrite(tsdfpc_path, verts, faces, norms, colors)
            
        # PyCUDA source code for tracking
        self.cuda_src_mod = SourceModule("""
        #include <curand_kernel.h>
        extern "C" {                                  
        __global__ void compute_tsdf_value(float * tsdf_vol,
                                    float * weight_vol,
                                    float * depth_map,
                                    float * depth_vertex,
                                    float * search_value,
                                    float * search_count,
                                    float * search_size,
                                    const float current_global_R[9],
                                    const float current_global_T[3],
                                    float * q_transform,
                                    float * other_params,
                                    float * cam_intr,
                                    float * normal_vertex){
            int vol_dim_x = (int) other_params[0];
            int vol_dim_y = (int) other_params[1];
            int vol_dim_z = (int) other_params[2];

            //float vol_origin_x = (int) other_params[3];
            //float vol_origin_y = (int) other_params[4];
            //float vol_origin_z = (int) other_params[5];
            int vol_origin_x = (int) other_params[3];
            int vol_origin_y = (int) other_params[4];
            int vol_origin_z = (int) other_params[5];


            float voxel_size = other_params[6];

            int node_size= (int) other_params[7];
            int level= (int)other_params[8];
            int im_h = (int) other_params[9]*level;
            int im_w = (int) other_params[10]*level;

            int level_index = (int) other_params[11];
            int img_H =  (int) other_params[12];
            int img_W =  (int) other_params[13];      

            
            int node=blockDim.x*blockIdx.x+threadIdx.x;
            int pi=(blockDim.y*blockIdx.y+threadIdx.y)*level+level_index;
            int pj=(blockDim.z*blockIdx.z+threadIdx.z)*level+level_index;
            
            if ( pi>im_h-1 || pj>im_w-1 || pi<0 || pj<0 || node>=node_size){
                return;
            }
            
            if(normal_vertex[(pi*img_W+pj)*3]==0 && normal_vertex[(pi*img_W+pj)*3+1]==0 && normal_vertex[(pi*img_W+pj)*3+2]==0){
                return;
            }

            //depth world_to_image
            float x=depth_vertex[(pi*img_W+pj)*4];
            float y=depth_vertex[(pi*img_W+pj)*4+1];
            float z=depth_vertex[(pi*img_W+pj)*4+2];
            float gt_tsdf = depth_vertex[(pi*img_W+pj)*4+3];
            
            //float debugx=depth_vertex[(pi*img_W+pj)*4];
            //float debugy=depth_vertex[(pi*img_W+pj)*4+1];
            //float debugz = depth_vertex[(pi*img_W+pj)*4+2];

            if (x==0 && y==0 && z==0){
                return;
            }

            
            //rotate the vertex to the global pose
            float global_x = current_global_R[0] * x + current_global_R[1] * y + current_global_R[2] * z ; 
            float global_y = current_global_R[3] * x + current_global_R[4] * y + current_global_R[5] * z ; 
            float global_z = current_global_R[6] * x + current_global_R[7] * y + current_global_R[8] * z ; 

            float t_x=q_transform[node*6+0] * search_size[0]; //0.00100031
            float t_y=q_transform[node*6+1] * search_size[1];
            float t_z=q_transform[node*6+2] * search_size[2];

            float q1=q_transform[node*6+3] * search_size[3];
            float q2=q_transform[node*6+4] * search_size[4];
            float q3=q_transform[node*6+5] * search_size[5];
            float q0=sqrt(1-q1*q1-q2*q2-q3*q3);;//real part

            float q_w= -(global_x*q1 + global_y*q2 + global_z*q3);
            float q_x = q0*global_x - q3*global_y + q2*global_z ;
            float q_y = q3*global_x + q0*global_y - q1*global_z;
            float q_z =-q2*global_x + q1*global_y + q0*global_z ;
            
            x= q_x*q0 + q_w*(-q1) - q_z*(-q2) + q_y*(-q3) + t_x + current_global_T[0];
            y= q_y*q0 + q_z*(-q1) + q_w*(-q2) - q_x*(-q3) + t_y + current_global_T[1];
            z= q_z*q0 - q_y*(-q1) + q_x*(-q2) + q_w*(-q3) + t_z + current_global_T[2];

            float vertex_cur_cam_x = x - current_global_T[0];
            float vertex_cur_cam_y = y - current_global_T[1];
            float vertex_cur_cam_z = z - current_global_T[2];
            
            float cam_x = current_global_R[0]*vertex_cur_cam_x+current_global_R[3]*vertex_cur_cam_y+current_global_R[6]*vertex_cur_cam_z ;
            float cam_y = current_global_R[1]*vertex_cur_cam_x+current_global_R[4]*vertex_cur_cam_y+current_global_R[7]*vertex_cur_cam_z ;
            float cam_z = current_global_R[2]*vertex_cur_cam_x+current_global_R[5]*vertex_cur_cam_y+current_global_R[8]*vertex_cur_cam_z ;
    
            int pixel_x = (int)((cam_x*cam_intr[0])/cam_z+cam_intr[2]+0.5f);
            int pixel_y = (int)((cam_y*cam_intr[4])/cam_z+cam_intr[5]+0.5f);
            
            
            if (pixel_x>=0 && pixel_y>=0 && pixel_x<img_W && pixel_y<img_H && cam_z>=0){
                int voxel_x = (int)roundf((x-vol_origin_x)/voxel_size);
                int voxel_y = (int)roundf((y-vol_origin_y)/voxel_size);
                int voxel_z = (int)roundf((z-vol_origin_z)/voxel_size);
                //change 0->1 to be the same as c++ version
                if (voxel_x<1 || voxel_x>=vol_dim_x-1 || voxel_y<1 || voxel_y>=vol_dim_y-1 || voxel_z<1 || voxel_z>=vol_dim_z-1){
                return;
                }

                int index=voxel_z+voxel_y*vol_dim_z+voxel_x*vol_dim_y*vol_dim_z;
                
                if (index > vol_dim_x*vol_dim_y*vol_dim_z){
                return;
                }
                          
                //float add_value = abs(tsdf_vol[index]);
                float add_value = abs(tsdf_vol[index]-gt_tsdf);
                
                
                unsigned long long u_node = (unsigned long long)(node);
                
                atomicAdd_system(search_value+u_node,add_value);
                atomicAdd_system(search_count+node,1);
            }
            
            return;

        }
 
        __global__ void compute_vertex(float * depth,
                          float * depth_vertex,
                          float * cam_intr,
                          float * other_params) {
            int im_h = (int) other_params[0];
            int im_w = (int) other_params[1];
            float cutdist =  other_params[2];
            float trunc =  other_params[3];
            int seed_num =  int(other_params[4]);
            float sample_range =  other_params[5];

            int pi=blockDim.x*blockIdx.x+threadIdx.x;
            int pj=blockDim.y*blockIdx.y+threadIdx.y;

            
            if (  pi>im_h-1 || pj>im_w-1 || pi<0 || pj<0   ){
              return;
            }
            
            float depth_value = depth[pi*im_w+pj];
            if (depth_value>cutdist){
                depth_value = 0.f;
            }
            
            if ( depth_value<=0){
              //mark the pixel is invalid, and filter it in eval tsdf
              depth_vertex[(pi*im_w+pj)*4]=0.0;
              depth_vertex[(pi*im_w+pj)*4+1]=0.0;
              depth_vertex[(pi*im_w+pj)*4+2]=0.0;
              depth_vertex[(pi*im_w+pj)*4+3]=0.0;
              return;
            }
            
            unsigned long long seed = seed_num; //123456789;
            curandState state;
            curand_init(seed, pi, 0, &state); // init random state
            
            //float sample = (curand_uniform(&state)*2)-1;
            //float z_val = sample*trunc; // [-trunc,+trunc]
            
            //float sample = (curand_uniform(&state)*4)-3; // good for all except 106
            //float z_val = sample*trunc; // [-3*trunc,+1*trunc]
            

            float sample = (curand_uniform(&state)*(sample_range+1))-sample_range;
            float z_val = sample*trunc; 
        
            if (sample_range<1){
                sample = (curand_uniform(&state)*2*sample_range)-sample_range;
                z_val = sample*trunc; 
            }
            
            float gt_tsdf = -sample;
            
            if (z_val<-1*trunc){
                gt_tsdf = 1.0;
            }
            
            if (z_val>1*trunc){
                gt_tsdf = 1.0;
            }
            
            //printf("z_val:%f,gt_Tsdf:%f\\n",z_val,-sample);
            float c_z=depth_value + z_val;
            float c_x=((float)pj-cam_intr[0*3+2])*c_z/cam_intr[0];
            float c_y=((float)pi-cam_intr[1*3+2])*c_z/cam_intr[1*3+1];

            depth_vertex[(pi*im_w+pj)*4]=c_x;
            depth_vertex[(pi*im_w+pj)*4+1]=c_y;
            depth_vertex[(pi*im_w+pj)*4+2]=c_z;
            depth_vertex[(pi*im_w+pj)*4+3] = gt_tsdf;

        }
        
        __global__ void compute_normal(
                                float * depth_vertex,
                                float * depth_normal,
                                float * other_params
                                ) {
            int im_h = (int) other_params[0];
            int im_w = (int) other_params[1];

            int pi=blockDim.x*blockIdx.x+threadIdx.x;
            int pj=blockDim.y*blockIdx.y+threadIdx.y;

            if (  pi>im_h-2 || pj>im_w-2 ||pi<1 || pj<1){
                return;
            }
            //depth vertex to normal
            int left=pi*im_w+pj-1;
            int right=pi*im_w+pj+1;
            int up=(pi-1)*im_w+pj;
            int down=(pi+1)*im_w+pj;
            int center=pi*im_w+pj;
            

            if (depth_vertex[center*4+2] ==0 || depth_vertex[left*4+2] ==0 || depth_vertex[right*4+2] ==0 || depth_vertex[up*4+2] ==0 || depth_vertex[down*4+2] ==0){
                depth_normal[(pi*im_w+pj)*3]  =0.f;
                depth_normal[(pi*im_w+pj)*3+1]=0.f;
                depth_normal[(pi*im_w+pj)*3+2]=0.f;
                return;
            }
            else{
                float hor_x=depth_vertex[left*4]-depth_vertex[right*4];
                float hor_y=depth_vertex[left*4+1]-depth_vertex[right*4+1];
                float hor_z=depth_vertex[left*4+2]-depth_vertex[right*4+2];

                float ver_x=depth_vertex[up*4]-depth_vertex[down*4];
                float ver_y=depth_vertex[up*4+1]-depth_vertex[down*4+1];
                float ver_z=depth_vertex[up*4+2]-depth_vertex[down*4+2];

                float normal_x=-hor_z*ver_y+hor_y*ver_z;
                float normal_y=hor_z*ver_x-hor_x*ver_z;
                float normal_z=-hor_y*ver_x+hor_x*ver_y;
                float lens=sqrt(normal_x*normal_x+normal_y*normal_y+normal_z*normal_z);
                normal_x =normal_x/lens;
                normal_y =normal_y/lens;
                normal_z =normal_z/lens;
                
                if (normal_z>0){
                normal_x *=-1;
                normal_y *=-1;
                normal_z *=-1;
                }
                
                depth_normal[(pi*im_w+pj)*3]  =normal_x;
                depth_normal[(pi*im_w+pj)*3+1]=normal_y;
                depth_normal[(pi*im_w+pj)*3+2]=normal_z;

            }
   
        }  
        }
         """, no_extern_c=True)
        self.cuda_compute_tsdf_value = self.cuda_src_mod.get_function("compute_tsdf_value") 
        self.cuda_compute_vertex = self.cuda_src_mod.get_function("compute_vertex")
        self.cuda_compute_normal = self.cuda_src_mod.get_function("compute_normal")


    def init_searchsize(self):
        """
        Initialize the search size and related vectors for transformation search.
        - iter_trans_vector: Stores the current transformation parameters (translation and rotation).
        - search_size: Stores the current search window/range for each transformation parameter.
        - previous_search_size: Keeps the previous search size for comparison or tracking convergence.
        - search_size[...] = self.init_size: Sets the initial search size (should be provided elsewhere).
        """
        self.iter_trans_vector = np.zeros((6), dtype=np.float32)  # [x, y, z, rx, ry, rz]
        self.search_size = np.zeros((6), dtype=np.float32)
        self.previous_search_size = np.zeros((6), dtype=np.float32)
        self.search_size[...] = self.init_size
    

    
    def init_depth_vertex(self, depth_im, cam_intr):
        """
        Initialize the depth vertex buffer on the GPU by invoking the CUDA kernel that computes
        the 3D vertices given a depth image and camera intrinsics. This method uploads the depth
        image to the GPU, prepares the necessary parameters, and launches the CUDA kernel to
        compute the vertex map, with random seed support for stochastic sampling.

        Args:
            depth_im (np.ndarray): The depth image in floating point, shape [H, W].
            cam_intr (np.ndarray): The camera intrinsic parameters, shape [3, 3] or flat (9,).

        Side Effects:
            Fills self.depth_vertex_gpu (device buffer) with 3D coordinates for each pixel.
        """
        cuda.memcpy_htod(self.depth_map_gpu, depth_im.reshape(-1).astype(np.float32))
        seed_num = random.randint(1, 1000000)
        self.cuda_compute_vertex(self.depth_map_gpu,
                            self.depth_vertex_gpu,
                            cuda.In(cam_intr.reshape(-1).astype(np.float32)),
                            cuda.In(np.asarray([
                            self.im_h,
                            self.im_w,
                            self.cut_dist,
                            self.truncation,
                            seed_num,
                            self.sample_range
                            ], np.float32)),
                            block=(int((self.im_h+32-1)/32),int((self.im_w+32-1)/32),1),
                            grid=(32,32,1)
                            )
    
    def init_normal(self):
        """
        Initialize the normal map on the GPU by launching the CUDA kernel that computes vertex normals
        from the depth vertex map. This method uses the current values for image height and width,
        and stores the result in normal_vertex_gpu.
        """
        self.cuda_compute_normal(
            self.depth_vertex_gpu,
            self.normal_vertex_gpu,
            cuda.In(np.asarray([
                self.im_h,
                self.im_w
            ], np.float32)),
            block=(int((self.im_h+32-1)/32), int((self.im_w+32-1)/32), 1),
            grid=(32, 32, 1)
        )
    
    def get_PST(self, tiff_index):
        """
        Retrieve a sub-array of PST (Particle Sampling Transform) candidates given the tiff index.
        
        The tiff_index encodes both a class and an intra-class index. 
        - Each PST_class contains up to 20 sub-classes.
        - Each PST_class_index identifies a group within the class (e.g., for different resolutions).
        - Returns a view of ALL_PST with shape [N, 6], where N may be 10240, 3072, or 1024 depending on the class.
        
        Args:
            tiff_index (int): Index encoding PST class and subclass.
        
        Returns:
            np.ndarray: Corresponding PST sub-array of transformation candidates.
        """
        PST_class = tiff_index // 20
        PST_class_num = tiff_index - PST_class * 20
        PST_class_index = PST_class_num // 3
        return self.ALL_PST[PST_class][PST_class_index, ...]  # (e.g., 10240x6, 3072x6, 1024x6)
        
    
    def update_PST(self, tsdf, mean_transform, min_scale=1e-3, scale=0.09):
        """
        Update the search size (perturbation scale) for the particle sampling transform (PST).
        
        Args:
            tsdf (float): The minimum TSDF value for the transformation candidate.
            mean_transform (np.ndarray): Array of 7 elements representing the translation (0:3) and quaternion (3:7) parameters.
            min_scale (float): The minimum scale to ensure non-zero search size for each parameter.
            scale (float): A scaling factor applied to the search size update.
        """
        # Compute the absolute translation components plus a minimum scale
        s_tx = abs(mean_transform[0]) + min_scale
        s_ty = abs(mean_transform[1]) + min_scale
        s_tz = abs(mean_transform[2]) + min_scale

        # Compute the absolute quaternion (rotation) components plus a minimum scale
        s_qx = abs(mean_transform[4]) + min_scale
        s_qy = abs(mean_transform[5]) + min_scale
        s_qz = abs(mean_transform[6]) + min_scale

        # Calculate the L2 norm of the search direction vector (translation and rotation)
        trans_norm = np.sqrt(s_tx**2 + s_ty**2 + s_tz**2 + s_qx**2 + s_qy**2 + s_qz**2)

        # Normalize translation and quaternion directions
        normal_tx = s_tx / trans_norm
        normal_ty = s_ty / trans_norm
        normal_tz = s_tz / trans_norm
        normal_qx = s_qx / trans_norm
        normal_qy = s_qy / trans_norm
        normal_qz = s_qz / trans_norm

        # Use the normalized directions, TSDF value, and scaling factor to update search size
        self.search_size[3] = scale * tsdf * normal_qx + min_scale
        self.search_size[4] = scale * tsdf * normal_qy + min_scale
        self.search_size[5] = scale * tsdf * normal_qz + min_scale
        self.search_size[0] = scale * tsdf * normal_tx + min_scale
        self.search_size[1] = scale * tsdf * normal_ty + min_scale
        self.search_size[2] = scale * tsdf * normal_tz + min_scale

        return
    
    def evaluate_tsdf(self, cur_id, level, node_size, cam_intr, level_index):
        """
        Evaluate the TSDF (Truncated Signed Distance Function) for the current transformation candidates.
        This function launches a CUDA kernel to compute the average TSDF value for each transformation
        candidate, given the current pose and other parameters.

        Parameters:
            cur_id (int): Current frame or process ID.
            level (int): Pyramid or processing level (relates to image downsampling).
            node_size (int): Number of transformation candidates.
            cam_intr (np.ndarray): Camera intrinsic parameters (flattened).
            level_index (int): The current level index for accessing pyramid data.
            value_type: Type of value to be evaluated.

        Returns:
            tuple of np.ndarray:
                - The normalized search values (average TSDF per candidate).
                - Raw TSDF sum values for each candidate.
                - Count of samples per candidate.
        """
        # Allocate arrays to store TSDF value sums and sample counts for each candidate
        search_value = np.zeros((self.transform_candidate.shape[0])).astype(np.float32)
        search_count = np.zeros((self.transform_candidate.shape[0])).astype(np.float32)
        
        # Transfer current global rotation and translation to the GPU
        R_GPU = gpuarray.to_gpu(self.current_global_R.reshape(-1).astype(np.float32))
        T_GPU = gpuarray.to_gpu(self.current_global_T.reshape(-1).astype(np.float32))

        # Launch the CUDA kernel to compute TSDF values for all transformation candidates
        self.cuda_compute_tsdf_value(
                                    self.MV.tsdf_vol_gpu,
                                    self.MV.weight_vol_gpu, 
                                    self.depth_map_gpu,
                                    self.depth_vertex_gpu,
                                    cuda.InOut(search_value),
                                    cuda.InOut(search_count),
                                    cuda.In(self.search_size.astype(np.float32)),
                                    R_GPU,
                                    T_GPU,
                                    cuda.In(self.transform_candidate.reshape(-1).astype(np.float32)),
                                    cuda.In(np.asarray([
                                    self.MV.vol_dim[0],
                                    self.MV.vol_dim[1],
                                    self.MV.vol_dim[2],
                                    self.MV.vol_origin[0],
                                    self.MV.vol_origin[1],
                                    self.MV.vol_origin[2],
                                    self.MV.voxel_size,
                                    node_size,
                                    level,
                                    int(self.im_h/level),
                                    int(self.im_w/level),
                                    level_index,
                                    self.im_h,
                                    self.im_w,
                                    cur_id,
                                    self.cut,
                                    self.cut_dist
                                    ], np.float32)),
                                    cuda.In(cam_intr.reshape(-1).astype(np.float32)),
                                    self.normal_vertex_gpu,
                                    block=(32*32,1,1),  
                                    grid=( int(node_size/(32*32)),int(self.im_h/level),int(self.im_w/level))            
                                    )
        
        f_search_value = search_value
            
        # Return normalized TSDF values, raw sums, and counts
        return f_search_value / (search_count + 1e-6), f_search_value, search_count
    
    def cal_transform(self, search_value):
        """
        Calculate the weighted average (mean) transformation parameters (translation and quaternion)
        from a list of TSDF search values and candidate transformations.

        Args:
            search_value (np.ndarray): Array of TSDF (Truncated Signed Distance Function) fit values
                                       for each transformation candidate.

        Returns:
            tuple: (success_flag, min_tsdf, mean_transform)
                success_flag (bool): Whether a valid set of candidates was found (at least one below origin_tsdf).
                min_tsdf (float): The minimum mean TSDF value found (used for optimization).
                mean_transform (np.ndarray): The mean transformation as a 7-element array
                                             [tx, ty, tz, qw, qx, qy, qz] (quaternion).
        """
        mean_transform = np.zeros((7), dtype=np.float32)
        origin_tsdf = search_value[0]

        # Initialize sum variables for weighted average
        sum_tx = 0.0
        sum_ty = 0.0
        sum_tz = 0.0
        sum_qw = 0.0
        sum_qx = 0.0
        sum_qy = 0.0
        sum_qz = 0.0
        sum_weight = 0.0
        sum_tsdf = 0.0
        count_search = 0

        # Loop through each candidate and aggregate weighted values
        for j in range(1, len(search_value)):
            if search_value[j] < origin_tsdf:
                tx = self.transform_candidate[j][0]
                ty = self.transform_candidate[j][1]
                tz = self.transform_candidate[j][2]
                qx = self.transform_candidate[j][3]
                qy = self.transform_candidate[j][4]
                qz = self.transform_candidate[j][5]
                cur_fit = search_value[j]
                weight = origin_tsdf - cur_fit  # Higher weight for better fits

                sum_tx += tx * weight
                sum_ty += ty * weight
                sum_tz += tz * weight
                sum_qx += qx * weight
                sum_qy += qy * weight
                sum_qz += qz * weight

                # Scale quaternion components for normalization
                qx = qx * self.search_size[3]
                qy = qy * self.search_size[4]
                qz = qz * self.search_size[5]

                # Safety check for quaternion validity (must have positive sqrt)
                if (1 - qx*qx - qy*qy - qz*qz < 0):
                    print("qx qy qz:", qx, qy, qz)
                    print("search size:", self.search_size)
                    print("sqrt content", 1 - qx*qx - qy*qy - qz*qz)
                    print("weight:", weight)
                    print("origin_tsdf:", origin_tsdf)
                    print("cur_fit:", cur_fit)
                    exit(0)
                qw = np.sqrt(1 - qx*qx - qy*qy - qz*qz)
                sum_qw += qw * weight
                sum_weight += weight
                sum_tsdf += cur_fit * weight
                count_search += 1

                # Stop if enough candidates were considered
                if count_search == self.count_search:
                    break

        # If no valid candidates found, return failure
        if count_search <= 0:
            success = False
            min_tsdf = origin_tsdf
            return False, min_tsdf, mean_transform

        # Otherwise, compute the weighted mean for each parameter
        mean_tsdf = sum_tsdf / sum_weight
        mean_transform[0] = (sum_tx / sum_weight) * self.search_size[0]
        mean_transform[1] = (sum_ty / sum_weight) * self.search_size[1]
        mean_transform[2] = (sum_tz / sum_weight) * self.search_size[2]
        qww = (sum_qw / sum_weight)
        qxx = (sum_qx / sum_weight) * self.search_size[3]
        qyy = (sum_qy / sum_weight) * self.search_size[4]
        qzz = (sum_qz / sum_weight) * self.search_size[5]

        # Check normalization for quaternion (otherwise print warning)
        if (1 - qxx*qxx - qyy*qyy - qzz*qzz) < 0:
            print("WRONG QW ********************************************")

        # Normalize the quaternion part of transformation
        lens = 1 / np.sqrt(qww*qww + qxx*qxx + qyy*qyy + qzz*qzz)
        mean_transform[3] = qww * lens
        mean_transform[4] = qxx * lens
        mean_transform[5] = qyy * lens
        mean_transform[6] = qzz * lens

        min_tsdf = mean_tsdf
        return True, min_tsdf, mean_transform
    

    
    
    def random_optimization(self, cur_id, cam_pose, rgb_im, depth_im, cam_intr, beta=0.9, inherit=False):
        """
        Perform random optimization to refine camera pose using particle sampling and TSDF evaluation.
        
        Args:
            cur_id (int): Current frame index.
            cam_pose (np.ndarray): Initial camera pose (4x4 extrinsic matrix).
            rgb_im (np.ndarray): Current RGB image.
            depth_im (np.ndarray): Current depth image.
            cam_intr (np.ndarray): Camera intrinsics.
            beta (float): Exponential moving average factor for search size update.
            inherit (bool): Whether to inherit the previous search size for optimization.

        Returns:
            np.ndarray: Optimized camera pose (4x4 matrix).
        """
        # Initialize global rotation and translation from input pose
        self.current_global_R = cam_pose[:3, :3].copy()
        self.current_global_T = cam_pose[:3, 3].copy()
        
        # Inherit search size from previous frame if required and successful
        if inherit is True and self.previous_frame_success:
            self.search_size = self.initialize_search_size
        else:
            self.init_searchsize()
        
        # Prepare data on device for TSDF evaluation
        self.init_depth_vertex(depth_im, cam_intr)
        self.init_normal()
        
        previous_success = False
        success = False
        count_particle = 0
        level_index = 5

        # Iterate for the number of defined particle steps
        for i in range(self.particle_iter_lens):
            if not success:
                count_particle = 0

            PST_class = count_particle % 3
            # Sample candidate transformations from PST table
            self.transform_candidate = self.get_PST(self.tiff_index[count_particle])
            level = self.depth_level[count_particle]

            # Evaluate TSDF and choose candidate transformation (rgb-based or depth-based)
            search_value, sv, sc = self.evaluate_tsdf(cur_id, level, self.PST_size[PST_class], cam_intr, level_index)

            # Find the best transformation parameters
            success, min_tsdf, mean_transform = self.cal_transform(search_value)

            # Unpack transformation increment (translation + quaternion)
            current_T_incremental = mean_transform[:3]
            qw = mean_transform[3]
            qx = mean_transform[4]
            qy = mean_transform[5]
            qz = mean_transform[6]

            # If successful, apply transformation increment to tracker state
            if success:
                if count_particle < 19:
                    count_particle += 1
                # Quaternion to rotation matrix using standard formula
                current_R_incremental = np.array([
                    [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw),     2 * (qx*qz + qy*qw)],
                    [2 * (qx*qy + qz*qw),     1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
                    [2 * (qx*qz - qy*qw),     2 * (qy*qz + qx*qw),     1 - 2 * (qx*qx + qy*qy)]
                ], dtype=np.float32)
                self.current_global_T += current_T_incremental
                self.current_global_R = np.matmul(current_R_incremental, self.current_global_R)

            # Update multi-resolution grid index
            if self.fix_level_index:
                level_index = 1
            else:
                level_index += 5

            level_index = level_index % (self.depth_level[count_particle])

            # Update search space size for next iteration
            self.update_PST(min_tsdf, mean_transform, scale=self.scaling_coefficient)

            # Exponential moving average of search size if two successive successes
            if previous_success and success:
                self.search_size[0] = beta * self.search_size[0] + (1 - beta) * self.previous_search_size[0]
                self.search_size[1] = beta * self.search_size[1] + (1 - beta) * self.previous_search_size[1]
                self.search_size[2] = beta * self.search_size[2] + (1 - beta) * self.previous_search_size[2]
                self.search_size[3] = beta * self.search_size[3] + (1 - beta) * self.previous_search_size[3]
                self.search_size[4] = beta * self.search_size[4] + (1 - beta) * self.previous_search_size[4]
                self.search_size[5] = beta * self.search_size[5] + (1 - beta) * self.previous_search_size[5]

            # Save current search size for potential next step
            elif success:
                if self.iterative_scale:
                    previous_success = True
                self.previous_search_size[0] = self.search_size[0]
                self.previous_search_size[1] = self.search_size[1]
                self.previous_search_size[2] = self.search_size[2]
                self.previous_search_size[3] = self.search_size[3]
                self.previous_search_size[4] = self.search_size[4]
                self.previous_search_size[5] = self.search_size[5]

            if not success:
                previous_success = False

            # On first iteration, store if this frame successfully initialized the search region
            if i == 0:
                if success:
                    self.initialize_search_size = self.search_size
                    self.previous_frame_success = True
                else:
                    self.previous_frame_success = False

        # Compose final 4x4 transformation matrix and return
        cam_pose_iter = np.eye(4, dtype=np.float32)
        cam_pose_iter[:3, :3] = self.current_global_R
        cam_pose_iter[:3, 3] = self.current_global_T

        return cam_pose_iter

        
    def readpst(self, PST_path, PST_size):
        """
        Load Particle Sampling Transform (PST) candidate arrays from TIFF files for multi-resolution search.

        Args:
            PST_path (str): Path to the directory containing PST .tiff files.
            PST_size (list or tuple): Number of candidates in each PST class/dataset for multi-resolution (e.g. [10240, 3072, 1024]).

        This function initializes the ALL_PST dictionary with three arrays, each corresponding to a different
        resolution/class. For each index in self.tiff_index, the appropriate TIFF file is loaded into the relevant 
        portion of the PST array for fast access during tracking.
        """
        # Initialize storage arrays for the different PST resolutions/classes
        pst_tmp0 = np.zeros((len(self.tiff_index)//3 + 1, PST_size[0], 6), dtype=np.float32)
        pst_tmp1 = np.zeros((len(self.tiff_index)//3 + 1, PST_size[1], 6), dtype=np.float32)
        pst_tmp2 = np.zeros((len(self.tiff_index)//3, PST_size[2], 6), dtype=np.float32)
        self.ALL_PST = {0: pst_tmp0, 1: pst_tmp1, 2: pst_tmp2}

        # For each tiff index, load the corresponding PST candidates from disk
        for i in range(len(self.tiff_index)):
            PST_index = self.tiff_index[i]
            PST_class = PST_index // 20
            PST_class_num = PST_index - PST_class * 20
            PST_class_index = PST_class_num // 3

            pst_filename = os.path.join(
                PST_path,
                f"pst_{PST_size[PST_class]}_{PST_class_num}.tiff"
            )
            # Load the candidate array from TIFF and arrange as contiguous float32
            self.ALL_PST[PST_class][PST_class_index, ...] = np.ascontiguousarray(
                cv2.imread(pst_filename, -1)
            )


    def do_tracking(self, init_pose, decoder, batch, device):
        """
        Perform Random Optimization for pose estimation for the given frame.
        
        Args:
            init_pose (torch.Tensor or np.ndarray): The initial camera pose (4x4).
            decoder: Not used (compatibility placeholder).
            batch (dict): Current frame data containing 'frame_id', 'c2w', 'depth', and 'rgb'.
            device: Torch device (not used here but for compatibility).
        
        Returns:
            cam_pose_iter (np.ndarray): The optimized camera pose (4x4, numpy array) after tracking.
            rgb_np (np.ndarray): The RGB image for the current frame, scaled to 0-255.
            depth_np (np.ndarray): The depth image for the current frame as numpy array.
        """
        cur_id = batch["frame_id"]

        # Convert the initial pose to numpy if given as a torch tensor
        if isinstance(init_pose, torch.Tensor):
            init_pose = init_pose.detach().cpu().numpy()
        
        # Compute Absolute Pose Error (APE) before tracking
        ape_before = self.cal_ape_error(batch["c2w"].squeeze(), init_pose[:3, 3])
        self.ape_before = ape_before

        # Get the depth and rgb images as numpy arrays
        depth_np = batch["depth"].squeeze().cpu().numpy()
        rgb_np = np.floor((batch["rgb"].squeeze().cpu().numpy()) * 255)
        gt_pose = batch["c2w"].squeeze().cpu().numpy()

        # Store the ground truth pose for this frame
        self.gt_pose = gt_pose
        cur_id = cur_id

        # Run the tracking optimization
        cam_pose_iter = self.random_optimization(cur_id, init_pose, rgb_np, depth_np, self.K) 

        # Tracking finished, return optimized pose and frame data
        return cam_pose_iter, rgb_np, depth_np
    
    

    def post_processing(self,cur_id,cam_pose_iter,rgb,depth,est_c2w_data):
        '''
        This function handles post-tracking operations including TSDF volume update and mesh export.
        '''
        # if (cur_id+1)%100==0:
        #     self.MV.filter_tsdf(self.filter_weight) 
        #     print("fresh TSDF VOLUME")
            
        # Check if TSDF volume needs to shift based on the current camera pose.
        move_flag, old_volbnd = self.MV.check_move_volume_new(
            cur_id, cam_pose_iter, self.traj, version=self.MV.version, 
            larger_flag=self.larger_flag, get_pc=self.get_pc
        )

        if move_flag:
            # If this is the first volume move, start from 0, otherwise from the previous start_id
            if self.MV.start_id == 0:   
                start = 0 
            else:
                start = self.MV.start_id
            end = cur_id - 1
            self.MV.start_id = cur_id
            # Record the old volume boundaries for the previous range of frames
            self.MV.frame_to_Vrange[(start, end)] = old_volbnd
            self.larger_flag = False
            self.move_frameid = cur_id

        # Integrate the new RGB-D data into the TSDF volume
        self.MV.integrate(rgb, depth, self.K, cam_pose_iter, old_volbnd, obs_weight=1.)
        
        # Export and save a mesh from the local TSDF volume at user-defined intervals
        if self.save_volume and (cur_id % self.save_freq == 0 or cur_id == self.end_frame - 1):
            verts, faces, norms, colors = self.MV.get_mesh()
            tsdfpc_path = os.path.join(self.save_path, "RO" + "/" + str(cur_id) + ".ply")
            self.MV.meshwrite(tsdfpc_path, verts, faces, norms, colors)


    def cal_ape_error(self, gt, our_t):
        """
        Calculate the Absolute Pose Error (APE) between the ground truth pose and estimated translation.

        Args:
            gt (np.ndarray or torch.Tensor): Ground truth pose (4x4 matrix or compatible).
            our_t (np.ndarray or torch.Tensor): Estimated translation vector or compatible.

        Returns:
            float: The mean absolute error between the ground truth and estimated translation.
        """
        if not isinstance(gt, torch.Tensor) and not isinstance(our_t, torch.Tensor):
            # Both inputs are numpy arrays, use numpy operations
            ape = np.average(np.abs(gt[:3,3]-our_t))
            return ape
        if not isinstance(gt, torch.Tensor):
            # Convert ground truth to torch tensor for compatibility
            gt = torch.from_numpy(gt).float()
            if our_t.is_cuda:
                gt = gt.cuda()
        # Compute mean absolute difference in translation
        ape = torch.abs(gt[:3,3]-our_t).mean()
        return ape.item()
    
  