import torch
import numpy as np
import open3d as o3d
import random
import os
from packaging import version
import open3d as o3d
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from imageio import imwrite

# Local imports
from model.keyframe import KeyFrameDatabase
from utils import coordinates, extract_mesh, extract_mesh_github
from tools.eval_ate import pose_evaluation, pose_evaluation_na, align_ba
from utils import at_to_transform_matrix, qt_to_transform_matrix, matrix_to_axis_angle, matrix_to_quaternion



class SLAM():
    def __init__(self, config, dataset, model, device):
        self.config = config
        self.device = device
        self.dataset = dataset
        self.model = model
        self.create_bounds()
        self.create_pose_data()
        self.get_pose_representation()

        self.create_share_data()
        self.keyframeDatabase = self.create_kf_database(config)
        self.create_optimizer()
        self.vis_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'])
        self.scene = self.config['data']['datadir'].split("/")[-1]
        
    def pose_eval_func(self):
        return pose_evaluation
    
    def pose_eval_func_na(self):
        return pose_evaluation_na
    
    def align_ba(self):
        return align_ba
    
    def create_share_data(self):
        self.create_pose_data()
        self.mapping_first_frame = torch.zeros((1)).int().share_memory_()
        self.mapping_idx = torch.zeros((1)).share_memory_()
        self.tracking_idx = torch.zeros((1)).share_memory_()
        self.tracking_stop_flag = torch.zeros((1)).int().share_memory_()
        self.update_local_MV = torch.zeros((1)).share_memory_()
  
        
    
    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
    def get_pose_representation(self):
        '''
        Get the pose representation axis-angle or quaternion
        '''
        if self.config['training']['rot_rep'] == 'axis_angle':
            self.matrix_to_tensor = matrix_to_axis_angle
            self.matrix_from_tensor = at_to_transform_matrix
        
        elif self.config['training']['rot_rep'] == "quat":
            # print("Using quaternion as rotation representation")
            self.matrix_to_tensor = matrix_to_quaternion
            self.matrix_from_tensor = qt_to_transform_matrix
        else:
            raise NotImplementedError
        
    def create_pose_data(self):
        '''
        Create the pose data
        '''
        num_frames = self.dataset.num_frames
        self.est_c2w_data = torch.zeros((num_frames, 4, 4)).to(self.device).share_memory_()
        self.est_c2w_data_rel = torch.zeros((num_frames, 4, 4)).to(self.device).share_memory_()
        
        self.RO_c2w_data = torch.zeros((num_frames, 4, 4)).to(self.device).share_memory_()
        
        self.load_gt_pose() 
    
    def create_bounds(self):
        '''
        Get the pre-defined bounds for the scene
        '''
        self.bounding_box = torch.from_numpy(np.array(self.config['mapping']['bound'])).to(self.device)
        self.marching_cube_bound = torch.from_numpy(np.array(self.config['mapping']['marching_cubes_bound'])).to(self.device)

    def create_kf_database(self, config):  
        '''
        Create the keyframe database
        '''
        num_kf = int(self.dataset.num_frames // self.config['mapping']['keyframe_every'] + 1)  
        # print('#kf:', num_kf)
        # print('#Pixels to save:', self.dataset.num_rays_to_save)
        return KeyFrameDatabase(config, 
                                self.dataset.H, 
                                self.dataset.W, 
                                num_kf, 
                                self.dataset.num_rays_to_save, 
                                self.device,
                                len(self.dataset))
    
    def load_gt_pose(self):
        '''
        Load the ground truth pose
        '''
        self.pose_gt = torch.zeros((self.dataset.num_frames, 4, 4))
        for i, pose in enumerate(self.dataset.poses):
            self.pose_gt[i] = pose
 
    def save_state_dict(self, save_path):
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

    def load_ckpt(self, load_path):
        '''
        Load the model parameters and the estimated pose
        '''
        dict = torch.load(load_path)
        self.model.load_state_dict(dict['model'])
        self.est_c2w_data = dict['pose']
        self.est_c2w_data_rel = dict['pose_rel']

    def select_samples(self, H, W, samples):
        '''
        randomly select samples from the image
        '''
        indice = random.sample(range(H * W), int(samples))
        indice = torch.tensor(indice)
        return indice

    def get_loss_from_ret(self, ret, rgb=True, sdf=True, depth=True, fs=True, smooth=False, tracking=False, iter=0):
        """
        Compute the total training loss from network output.

        Args:
            ret (dict): Output dictionary containing losses and values from model.
            rgb (bool): Whether to include RGB loss in total loss.
            sdf (bool): Whether to include SDF loss in total loss.
            depth (bool): Whether to include depth loss in total loss.
            fs (bool): Whether to include free space loss in total loss.
            smooth (bool): Whether to apply smoothness loss.
            tracking (bool): Not used, for compatibility.
            iter (int): Iteration index, for logging.

        Returns:
            loss (float/tensor): Total computed loss as a weighted sum of selected losses.
        """
        loss = 0
        if rgb:
            loss += self.config['training']['rgb_weight'] * ret['rgb_res_loss']
        if depth:
            loss += self.config['training']['depth_weight'] * ret['depth_res_loss']
        if sdf:
            loss += self.config['training']['sdf_weight'] * ret["sdf_res_loss"]
        if fs:
            loss += self.config['training']['fs_weight'] * ret["fs_res_loss"]

        if smooth and self.config['training']['smooth_weight'] > 0:
            res_smooth = self.smoothness(
                self.config['training']['smooth_pts'],
                self.config['training']['smooth_vox'],
                margin=self.config['training']['smooth_margin']
            )
            loss += self.config['training']['smooth_weight'] * res_smooth

        # print(
        #     iter, "rgb:%4f,depth:%4f,sdf:%4f,fs:%4f,loss:%4f" % (
        #         self.config['training']['rgb_weight'] * ret['rgb_res_loss'],
        #         self.config['training']['depth_weight'] * ret['depth_res_loss'],
        #         self.config['training']['sdf_weight'] * ret["sdf_res_loss"],
        #         self.config['training']['fs_weight'] * ret["fs_res_loss"],
        #         loss
        #     )
        # )

        return loss
    

    def smoothness(self, sample_points=256, voxel_size=0.1, margin=0.05, color=False):
        '''
        Smoothness loss of feature grid
        '''
        volume = self.bounding_box[:, 1] - self.bounding_box[:, 0]

        grid_size = (sample_points-1) * voxel_size
        offset_max = self.bounding_box[:, 1]-self.bounding_box[:, 0] - grid_size - 2 * margin

        offset = torch.rand(3).to(offset_max) * offset_max + margin
        coords = coordinates(sample_points - 1, 'cpu', flatten=False).float().to(volume)
        pts = (coords + torch.rand((1,1,1,3)).to(volume)) * voxel_size + self.bounding_box[:, 0] + offset

        if self.config['grid']['tcnn_encoding']:
            pts_tcnn = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

        sdf_res = self.model.query_sdf_res(pts_tcnn, embed=True)

        tv_x_res = torch.pow(sdf_res[1:,...]-sdf_res[:-1,...], 2).sum()
        tv_y_res = torch.pow(sdf_res[:,1:,...]-sdf_res[:,:-1,...], 2).sum()
        tv_z_res = torch.pow(sdf_res[:,:,1:,...]-sdf_res[:,:,:-1,...], 2).sum()

        loss_res = (tv_x_res + tv_y_res + tv_z_res)/ (sample_points**3)

        return loss_res
    
    def get_rays_from_batch(self, batch, c2w_est, indices):
        '''
        Get the rays from the batch
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
            c2w_est: [4, 4]
            indices: [N]
        Returns:
            rays_o: [N, 3]
            rays_d: [N, 3]
            target_s: [N, 3]
            target_d: [N, 1]
            c2w_gt: [4, 4]
        '''
        rays_d_cam = batch['direction'].reshape(-1, 3)[indices].to(self.device)
        target_s = batch['rgb'].reshape(-1, 3)[indices].to(self.device)
        target_d = batch['depth'].reshape(-1, 1)[indices].to(self.device)
        rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:3, :3], -1)
        rays_o = c2w_est[None, :3, -1].repeat(rays_d.shape[0], 1)
        c2w_gt = batch['c2w'][0].to(self.device)

        if torch.sum(torch.isnan(rays_d_cam)):
            print('warning rays_d_cam')
        
        if torch.sum(torch.isnan(c2w_est)):
            print('warning c2w_est')

        return rays_o, rays_d, target_s, target_d, c2w_gt
    
    
    def update_pose_array(self, frame_id):
        if torch.sum(torch.isnan(self.est_c2w_data[frame_id])):
            print('tracking warning')
        self.model.pose_array.add_params(self.est_c2w_data[frame_id].to(self.device), frame_id)

    def convert_relative_pose(self):
        poses = {}
        for i in range(len(self.est_c2w_data)):
            if i % self.config['mapping']['keyframe_every'] == 0:
                poses[i] = self.est_c2w_data[i]
            else:
                kf_id = i // self.config['mapping']['keyframe_every']
                kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                c2w_key = self.est_c2w_data[kf_frame_id]
                delta = self.est_c2w_data_rel[i] 
                poses[i] = delta @ c2w_key
        
        return poses

    def create_optimizer(self):
        """
        Create optimizers for mapping and pose refinement.
        This sets up two Adam optimizers:
        - map_optimizer: Optimizes decoder and embedding networks for mapping.
        - rba_optimizer: Optimizes the pose refinement network.
        If the grid is not shared (oneGrid=False), a color embedding network is also included.
        """
        trainable_parameters = [
            {'params': self.model.decoder_res.parameters(), 'weight_decay': 1e-6, 'lr': self.config['mapping']['lr_decoder']},
            {'params': self.model.embed_res_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed_res']}
        ]
        rba_parameter = [{'params': self.model.rba.parameters(), 'weight_decay': 1e-6, 'eps': 1e-15, 'lr': self.config['mapping']['lr_pose']}]

        self.map_optimizer = optim.Adam(trainable_parameters, betas=(0.9, 0.99))
        self.rba_optimizer = optim.Adam(rba_parameter, betas=(0.9, 0.99))
    
     
    
    def render_single(self, frame_id, gt_depth, gt_color, cam_pose, ray_d, prefix=None, gap=1):
        """
        Render a single RGB-D prediction using the current model state.

        Args:
            frame_id (int): The index of the frame to render.
            gt_depth (torch.Tensor): The ground truth depth image.
            gt_color (torch.Tensor): The ground truth color image.
            cam_pose (torch.Tensor or np.ndarray): Camera pose (4x4 matrix).
            ray_d (torch.Tensor): Ray directions for each pixel.
            prefix (str, optional): Optional prefix for saving outputs.
            gap (int, optional): Subsampling step for visualization (default: 1).

        Returns:
            color_np (np.ndarray): The predicted RGB image.
            depth_np (np.ndarray): The predicted depth image.
        """
        with torch.no_grad():
            # Subsample input images and rays for rendering
            gt_color = gt_color.squeeze(0)[::gap, ::gap, :]
            gt_depth = gt_depth.squeeze(0)[::gap, ::gap]
            ray_d = ray_d.squeeze()[::gap, ::gap, ...]

            gt_depth_np = gt_depth.cpu().numpy()
            
            # Ensure cam_pose is a Torch tensor
            if isinstance(cam_pose, torch.Tensor):
                c2w = cam_pose.squeeze().detach()
            else:
                c2w = torch.from_numpy(cam_pose).to(self.device)
            
            target_s = gt_color.reshape(-1, 3).to(self.device)
            target_d = gt_depth.reshape(-1, 1).to(self.device)
            
            # Compute ray origins and directions in world coordinates
            rays_o = c2w[:3, -1].repeat(1, target_s.shape[0], 1).reshape(-1, 3)
            ray_d = ray_d.to(self.device)
            rays_d = torch.sum(ray_d.reshape(-1, 3).unsqueeze(1) * c2w[None, :3, :3], -1)
            # [HW, 1, 3] * [1, 3, 3] = (HW, 3, 3) -> (HW, 3)

            rays_d = rays_d.reshape(-1, 3)

            rays_o = rays_o.to(self.device)
            rays_d = rays_d.to(self.device)
            
            # Perform prediction using the model's mapping function
            ret = self.model.mapping(rays_o, rays_d, target_s, target_d)
            depth = ret["depth_res"]
            color = ret["rgb_res"]
            
            # Reshape predicted results to image format
            depth_np = depth.reshape(gt_depth_np.shape[0], gt_depth_np.shape[1])
            color_np = color.reshape(gt_depth_np.shape[0], gt_depth_np.shape[1], 3)

            return color_np, depth_np
    
    
    
    def save_mesh(self, i, voxel_size=0.05):
        """
        Save the reconstructed mesh to a file.
        
        Args:
            i (int): Index or ID of the mesh to be saved, used for naming the output file.
            voxel_size (float, optional): Voxel size for marching cubes algorithm (default: 0.05).
        """
        mesh_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'mesh_track{}.ply'.format(int(i)))
        
        color_func = self.model.query_color_residual
        extract_mesh_github(self.model.query_sdf_res, 
                            self.model.query_w_res,
                            self.config, 
                            self.bounding_box, 
                            color_func=color_func, 
                            marching_cube_bound=self.marching_cube_bound, 
                            voxel_size=voxel_size, 
                            mesh_savepath=mesh_savepath)
        # print("saved to ", mesh_savepath)
        
    def save_mesh_final(self, voxel_size=0.05):
        """
        Save the reconstructed mesh to a file.
        
        Args:
            i (int): Index or ID of the mesh to be saved, used for naming the output file.
            voxel_size (float, optional): Voxel size for marching cubes algorithm (default: 0.05).
        """
        mesh_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'mesh.ply')
        
        color_func = self.model.query_color_residual
        extract_mesh_github(self.model.query_sdf_res, 
                            self.model.query_w_res,
                            self.config, 
                            self.bounding_box, 
                            color_func=color_func, 
                            marching_cube_bound=self.marching_cube_bound, 
                            voxel_size=voxel_size, 
                            mesh_savepath=mesh_savepath)
        # print("saved to ", mesh_savepath)
        
    def save_mesh_explicit(self, i, voxel_size=0.05):
        """
        Save the explicitly reconstructed mesh to a file.

        Args:
            i (int): Index or identifier for the mesh file name.
            voxel_size (float, optional): The voxel size for the marching cubes algorithm (default: 0.05).
        """
        mesh_savepath = os.path.join(
            self.config['data']['output'], 
            self.config['data']['exp_name'], 
            'mesh_track{}_ex.ply'.format(int(i))
        )
        color_func = self.model.query_color_ex
        extract_mesh_github(
            self.model.query_sdf_ex, 
            self.model.query_w_res,
            self.config, 
            self.bounding_box, 
            color_func=color_func, 
            marching_cube_bound=self.marching_cube_bound, 
            voxel_size=voxel_size, 
            mesh_savepath=mesh_savepath
        )
        print("explicit mesh saved to ", mesh_savepath)
    
    def render_img(self, frame_id, gt_depth, gt_color, cam_pose, ray_d, prefix=None, gap=4, step=None):
        """
        Render images using the model mapping and visualize the results.
        
        Args:
            frame_id (int): The current frame index for saving outputs.
            gt_depth (torch.Tensor): Ground truth depth map, shape [1, H, W] or [1, H, W, 1].
            gt_color (torch.Tensor): Ground truth RGB image, shape [1, H, W, 3].
            cam_pose (torch.Tensor): Camera-to-world transformation matrix, shape [1, 4, 4] or [4, 4].
            ray_d (torch.Tensor): Ray directions, shape [H, W, 3] or similar.
            prefix (str, optional): Optional prefix for output file naming.
            gap (int, optional): Pixel step for subsampling (default: 4).
            step (int, optional): Current training or iteration step (default: None).
        """
        with torch.no_grad():
            # Subsample depth, color, and rays for visualization
            gt_color = gt_color.squeeze(0)[::gap, ::gap, :]
            gt_depth = gt_depth.squeeze(0)[::gap, ::gap]
            ray_d = ray_d.squeeze()[::gap, ::gap, ...]

            # Convert to numpy for error calculation and visualization
            gt_depth_np = gt_depth.cpu().numpy()
            gt_color_np = gt_color.cpu().numpy()
            c2w = cam_pose.squeeze().detach()

            # Prepare target color and depth for evaluation
            target_s = gt_color.reshape(-1, 3).to(self.device)
            target_d = gt_depth.reshape(-1, 1).to(self.device)

            # Compute ray origins (all from camera center)
            rays_o = c2w[:3, -1].repeat(1, target_s.shape[0], 1).reshape(-1, 3)
            ray_d = ray_d.to(self.device)
            # Compute ray directions in world coordinates
            rays_d = torch.sum(ray_d.reshape(-1, 3).unsqueeze(1) * cam_pose[None, :3, :3], -1)
            rays_d = rays_d.reshape(-1, 3)

            rays_o = rays_o.to(self.device)
            rays_d = rays_d.to(self.device)

            # Model inference
            ret = self.model.mapping(rays_o, rays_d, target_s, target_d)
            depth = ret["depth_res"]
            color = ret["rgb_res"]

            # Predict to numpy for visualization
            depth_np = depth.detach().cpu().numpy().reshape(gt_depth_np.shape[0], gt_depth_np.shape[1])
            color_np = color.detach().cpu().numpy().reshape(gt_depth_np.shape[0], gt_depth_np.shape[1], 3)

            # Compute residual images (error)
            depth_residual = np.abs(gt_depth_np - depth_np)
            depth_residual[gt_depth_np == 0.0] = 0.0

            color_residual = np.abs(gt_color_np - color_np)
            color_residual[gt_depth_np == 0.0] = 0.0

            # Clamp color ranges
            gt_color_np = np.clip(gt_color_np, 0, 1)
            color_np = np.clip(color_np, 0, 1)
            color_residual = np.clip(color_residual, 0, 1)

            # Optionally save color result every 5 steps
            if step is not None:
                if step % 5 == 0:
                    imwrite(os.path.join(self.vis_dir, "rgb_{}.png".format(str(step))), (color_np * 255.).astype(np.uint8))

            # Mask out invalid areas
            depth_np[gt_depth_np == 0.0] = 0.0
            color_np[gt_depth_np == 0.0] = 0.0

            # Create a figure for visualization: depth, color, residuals
            fig, axs = plt.subplots(2, 3)
            fig.tight_layout()
            max_depth = np.max(gt_depth_np)

            axs[0, 0].imshow(gt_depth_np, cmap="plasma", vmin=0, vmax=max_depth)
            axs[0, 0].set_title('Input Depth')
            axs[0, 0].set_xticks([])
            axs[0, 0].set_yticks([])

            axs[0, 1].imshow(depth_np, cmap="plasma", vmin=0, vmax=max_depth)
            axs[0, 1].set_title('Generated Depth')
            axs[0, 1].set_xticks([])
            axs[0, 1].set_yticks([])

            axs[0, 2].imshow(depth_residual, cmap="plasma", vmin=0, vmax=max_depth)
            axs[0, 2].set_title('Depth Residual')
            axs[0, 2].set_xticks([])
            axs[0, 2].set_yticks([])

            axs[1, 0].imshow(gt_color_np, cmap="plasma")
            axs[1, 0].set_title('Input RGB')
            axs[1, 0].set_xticks([])
            axs[1, 0].set_yticks([])

            axs[1, 1].imshow(color_np, cmap="plasma")
            axs[1, 1].set_title('Generated RGB')
            axs[1, 1].set_xticks([])
            axs[1, 1].set_yticks([])

            axs[1, 2].imshow(color_residual, cmap="plasma")
            axs[1, 2].set_title('RGB Residual')
            axs[1, 2].set_xticks([])
            axs[1, 2].set_yticks([])
            plt.subplots_adjust(wspace=0, hspace=0)

            # Save the visualization results with appropriate naming
            if prefix is not None:
                output_path = os.path.join(self.vis_dir, str(frame_id) + "_" + prefix + ".jpg")
            else:
                output_path = os.path.join(self.vis_dir, str(frame_id) + ".jpg")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2, dpi=300)
            plt.cla()
            plt.clf()
    
    