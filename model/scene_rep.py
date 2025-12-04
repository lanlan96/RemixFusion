
import torch
import torch.nn as nn
import numpy as np
from .encodings import get_encoder
from .decoder import ColorSDFNet
from .utils import sample_pdf, batchify, mse2psnr, compute_loss, get_sdf_loss
from torch.autograd import grad
from .rba import RBA
import tinycudann as tcnn

        
class JointEncoding(nn.Module):
    def __init__(self, config, bound_box, num_kf=None):
        super(JointEncoding, self).__init__()
        self.config = config
        self.bounding_box = bound_box
        self.num_kf = num_kf
        self.get_resolution()
        self.get_encoding(config)
        self.get_decoder(config)
        self.count=0

    def get_resolution(self):
        '''
        Get the resolution of the grid
        '''
        dim_max = (self.bounding_box[:,1] - self.bounding_box[:,0]).max()
        if self.config['grid']['voxel_sdf'] > 10:
            self.resolution_sdf = self.config['grid']['voxel_sdf']
        else:
            self.resolution_sdf = int(dim_max / self.config['grid']['voxel_sdf'])
        
        if self.config['grid']['voxel_color'] > 10:
            self.resolution_color = self.config['grid']['voxel_color']
        else:
            self.resolution_color = int(dim_max / self.config['grid']['voxel_color'])
        
        # print('SDF resolution:', self.resolution_sdf)

    def get_encoding(self, config, GBV=True):
        '''
        Get the encoding of the scene representation
        '''
        # Coordinate encoding
        self.embedpos_fn, self.input_ch_pos = get_encoder(config['pos']['enc'], n_bins=self.config['pos']['n_bins'])
        
        self.embed_res_fn, embed_out = get_encoder(config['grid']['enc'], log2_hashmap_size=config['grid']['hash_size'], desired_resolution=self.resolution_sdf)
        
        self.input_ch = embed_out
            
        if GBV:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            n_levels = self.config["globalV"]["n_levels"]
            per_level_scale = self.config["globalV"]["per_level_scale"]
            base_resolution = self.config["globalV"]["base_resolution"]
            n_features_per_level = self.config["globalV"]["n_features_per_level"]

            # global volume 
            GBV = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                    "otype": "Grid",
                    "type": "Dense",
                    "n_levels": n_levels,
                    "n_features_per_level": n_features_per_level,
                    "base_resolution": base_resolution,
                    "per_level_scale": per_level_scale,
                    "interpolation": "Linear"},
                dtype=torch.float
            )
            GBV.to(self.device)
            GBV.requires_grad_(False)
            self.GBV = GBV
            # self.GBV.params[:] = 0.0 #initialize

            # global volume weight
            GBW = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                        "otype": "Grid",
                        "type": "Dense",
                        "n_levels": n_levels,
                        "n_features_per_level": 1,
                        "base_resolution": base_resolution,
                        "per_level_scale": per_level_scale,
                        "interpolation": "Linear"}, # Linear
                    dtype=torch.float
            )
            GBW.to(self.device)
            GBW.requires_grad_(False)
            self.GBW = GBW
            self.GBW.params[:] = 0.0 # initialize
            

    def get_decoder(self, config):
        '''
        Get the decoder of the scene representation
        '''
        self.decoder_res = ColorSDFNet(config, input_ch=self.input_ch, input_ch_pos=self.input_ch_pos)

        self.color_net_res = batchify(self.decoder_res.color_net, None)
        self.sdf_net_res = batchify(self.decoder_res.sdf_net, None)

        self.rba = RBA(self.num_kf,scale=config['mapping']['pose_scale'])

    def sdf2weights(self, sdf, z_vals, args=None):
        '''
        Convert signed distance function to weights.

        Params:
            sdf: [N_rays, N_samples]
            z_vals: [N_rays, N_samples]
        Returns:
            weights: [N_rays, N_samples]
        '''
        weights = torch.sigmoid(sdf / args['training']['trunc']) * torch.sigmoid(-sdf / args['training']['trunc'])

        signs = sdf[:, 1:] * sdf[:, :-1]
        mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
        inds = torch.argmax(mask, axis=1) # return the first indice
        inds = inds[..., None]
        z_min = torch.gather(z_vals, 1, inds) # The first surface
        mask = torch.where(z_vals < z_min + args['data']['sc_factor'] * args['training']['trunc'], torch.ones_like(z_vals), torch.zeros_like(z_vals))
        
        weights = weights * mask
        return weights / (torch.sum(weights, axis=-1, keepdims=True) + 1e-8)
    
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
                
    #@torch.cuda.amp.autocast(enabled=True)
    def raw2outputs(self, raw, z_vals):
        """
        Compute final rendered outputs from raw network predictions and sampled depths via volume rendering.
        
        Args:
            raw (Tensor): Raw network output, shape [N_rays, N_samples, 4]. The last channel is SDF value.
            z_vals (Tensor): Sampled depth values along each ray, shape [N_rays, N_samples].

        Returns:
            rgb_map (Tensor): Rendered RGB color per ray, shape [N_rays, 3].
            depth_map (Tensor): Rendered depth (expected) per ray, shape [N_rays].
        """
        # Extract predicted RGB for each sample on each ray
        rgb = raw[..., :3]  # [N_rays, N_samples, 3]
        # Compute weights per sample from SDF predictions using the configured truncation
        weights = self.sdf2weights(raw[..., 3], z_vals, args=self.config)

        # Calculate per-ray RGB by weighting and summing sample colors
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        # Calculate per-ray depth as weighted sum of sampled z values
        depth_map = torch.sum(weights * z_vals, -1)  # [N_rays]

        return rgb_map, depth_map
    
    
    
    #@torch.cuda.amp.autocast(enabled=True)
    def query_sdf(self, query_points, return_geo=False, embed=False):
        '''
        Get the SDF value of the query points
        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            sdf: [N_rays, N_samples]
            geo_feat: [N_rays, N_samples, channel]
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])
  
        embedded = self.embed_fn(inputs_flat)
        if embed:

            return torch.reshape(embedded, list(query_points.shape[:-1]) + [embedded.shape[-1]])

        embedded_pos = self.embedpos_fn(inputs_flat)
        out = self.sdf_net(torch.cat([embedded, embedded_pos], dim=-1))
        sdf, geo_feat = out[..., :1], out[..., 1:]
        
        sdf = torch.reshape(sdf, list(query_points.shape[:-1]))
        if not return_geo:
            return sdf
        geo_feat = torch.reshape(geo_feat, list(query_points.shape[:-1]) + [geo_feat.shape[-1]])
        
        return sdf, geo_feat
    
    #@torch.cuda.amp.autocast(enabled=True)
    def query_sdf_res(self, query_points, return_geo=False, embed=False):
        '''
        Get the SDF value of the query points
        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            sdf: [N_rays, N_samples]
            geo_feat: [N_rays, N_samples, channel]
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])
  
        embedded = self.embed_res_fn(inputs_flat)
        if embed:
            return torch.reshape(embedded, list(query_points.shape[:-1]) + [embedded.shape[-1]])

        embedded_pos = self.embedpos_fn(inputs_flat)
        
        ex_Trgb = self.GBV(inputs_flat)
        tmp_tsdf = ex_Trgb[...,0] * self.config['training']['c_trunc']
        tmp_tsdf = tmp_tsdf/self.config['training']['trunc']

        tmp_tsdf = torch.clamp(tmp_tsdf,-1,1)
        
        out = self.sdf_net_res(torch.cat([embedded, embedded_pos,tmp_tsdf.unsqueeze(-1)], dim=-1))
        sdf, geo_feat = out[..., :1], out[..., 1:]
        

        
        sdf[...,0] += tmp_tsdf 
        
        
        sdf = torch.reshape(sdf, list(query_points.shape[:-1]))
        if not return_geo:
            return sdf
        geo_feat = torch.reshape(geo_feat, list(query_points.shape[:-1]) + [geo_feat.shape[-1]])
        
        return sdf, geo_feat
    
    def query_sdf_ex(self, query_points, return_geo=False, embed=False):
        '''
        Get the SDF value of the query points
        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            sdf: [N_rays, N_samples]
            geo_feat: [N_rays, N_samples, channel]
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])
  
        ex_Trgb = self.GBV(inputs_flat)
        sdf = ex_Trgb[...,0]
        sdf = torch.reshape(sdf, list(query_points.shape[:-1]))

        return sdf

    
    #@torch.cuda.amp.autocast(enabled=True)
    def query_w_res(self, query_points, return_geo=False, embed=False):
        '''
        Get the SDF value of the query points
        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            sdf: [N_rays, N_samples]
            geo_feat: [N_rays, N_samples, channel]
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])
        ex_w = self.GBW(inputs_flat)
        ex_w = torch.reshape(ex_w, list(query_points.shape[:-1]))

        return ex_w
      
    #@torch.cuda.amp.autocast(enabled=True)
    def query_color_residual(self, query_points):
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])

        emebed_residual = self.embed_res_fn(inputs_flat) #c:32
        
        embe_pos = self.embedpos_fn(inputs_flat)

        ex_Trgb = self.GBV(inputs_flat)

        raw_residual = self.decoder_res(emebed_residual, embe_pos, ex_Trgb[...,:1], ex_Trgb[...,1:])

        raw_residual[...,:3] += ex_Trgb[...,1:]
        
        return raw_residual[...,:3]
    
    def query_color_ex(self, query_points):
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])

        ex_Trgb = self.GBV(inputs_flat)
        
        raw_residual = torch.zeros_like(ex_Trgb).to(self.device)

        raw_residual[...,:3] = ex_Trgb[...,1:]
        raw_residual[...,3] = ex_Trgb[...,0]
        
        return raw_residual[...,:3]
    
    
    #@torch.cuda.amp.autocast(enabled=True)
    def query_color_sdf(self, query_points, ranged_mask = None):
        '''
        Query the color and sdf at query_points.

        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            raw: [N_rays, N_samples, 4]
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])

        emebed_residual = self.embed_res_fn(inputs_flat)
        
        embe_pos_res = self.embedpos_fn(inputs_flat)

        ex_Trgb = self.GBV(inputs_flat)
        tmp_tsdf = ex_Trgb[...,0] * self.config['training']['c_trunc']
        tmp_tsdf = tmp_tsdf/self.config['training']['trunc']
        if self.clamp:
            threshold = self.config['mapping']['clamp']
            tmp_tsdf = torch.clamp(tmp_tsdf,-threshold, threshold)
            cin_tsdf = torch.clamp(tmp_tsdf,-1,1)
        else:
            tmp_tsdf = torch.clamp(tmp_tsdf,-1,1)
        
        if self.clamp: 
            raw_residual = self.decoder_res(emebed_residual, embe_pos_res, cin_tsdf.unsqueeze(-1), ex_Trgb[...,1:])
        else:
            raw_residual = self.decoder_res(emebed_residual, embe_pos_res, tmp_tsdf.unsqueeze(-1), ex_Trgb[...,1:])

        raw_residual[...,:3] += ex_Trgb[...,1:]
        raw_residual[...,3] += tmp_tsdf
        
        raw = torch.cat([raw_residual], dim=-1) 
        
        return raw
    
    #@torch.cuda.amp.autocast(enabled=True)
    def query_color_sdf_tracking(self, query_points):
        '''
        Query the color and sdf at query_points.

        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            raw: [N_rays, N_samples, 4]
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])
        emebed_full = self.embed_fn(inputs_flat)
        embe_pos = self.embedpos_fn(inputs_flat)

        raw = self.decoder(emebed_full, embe_pos)
        
        return raw
    
    #@torch.cuda.amp.autocast(enabled=True)
    def run_network(self, inputs, flat=False):
        """
        Run the network on a batch of inputs.

        Args:
            inputs (torch.Tensor): Input query points of shape [N_rays, N_samples, 3] or [N, 3].
            flat (bool, optional): If True, return the output as a flat tensor of shape [N, 4];
                                   if False, reshape back to [N_rays, N_samples, 4]. Default is False.

        Returns:
            torch.Tensor: Network outputs corresponding to queried inputs. Shape is either [N, 4] if flat=True,
                          or [N_rays, N_samples, 4] if flat=False.
        """
        # Flatten input points to [N, 3], where N = N_rays * N_samples
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        
        # Normalize input coordinates to [0, 1] range if using TCNN encoding
        if self.config['grid']['tcnn_encoding']:
            inputs_flat = (inputs_flat - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        
        # For very large batches, use batchify for memory efficiency; otherwise process directly
        if inputs_flat.shape[0] > 1000000:
            outputs_flat = batchify(self.query_color_sdf)(inputs_flat)
        else:
            outputs_flat = batchify(self.query_color_sdf, None)(inputs_flat)
        
        # If flat is True, return [N, 4] tensor directly
        if flat:
            return outputs_flat
        
        # Otherwise, reshape to match original input batch structure [N_rays, N_samples, 4]
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs
    
    
    
    #@torch.cuda.amp.autocast(enabled=True)
    def render_rays(self, rays_o, rays_d, target_d=None, tracking=False,frameid=None, render_flag=False):
        '''
        Params:
            rays_o: [N_rays, 3]
            rays_d: [N_rays, 3]
            target_d: [N_rays, 1]

        '''
        n_rays = rays_o.shape[0]
        
        range_d = self.config['training']['range_d']
        n_range_d = self.config['training']['n_range_d']
        n_samples_d = self.config['training']['n_samples_d']
        # Sample depth
        if target_d is not None:
            z_samples = torch.linspace(-range_d, range_d, steps=n_range_d).to(target_d) 
            z_samples = z_samples[None, :].repeat(n_rays, 1) + target_d
            z_samples[target_d.squeeze()<=0] = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], steps=n_range_d).to(target_d) 

            if n_samples_d > 0:
                z_vals = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], n_samples_d)[None, :].repeat(n_rays, 1).to(rays_o)
                z_vals, z_index = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

            else:
                z_vals = z_samples
        else:
            z_vals = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], self.config['training']['n_samples']).to(rays_o)
            z_vals = z_vals[None, :].repeat(n_rays, 1) # [n_rays, n_samples]
        
        # Perturb sampling depths
        if self.config['training']['perturb'] > 0.:
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            z_vals = (lower + (upper - lower) * torch.rand(z_vals.shape).to(rays_o))
            
        pts = rays_o[...,None,:] + rays_d[...,None,:] * (z_vals[...,:,None]) # [N_rays, N_samples, 3]

        raw = self.run_network(pts)

        ret = {}
        
        rgb_res_map,  depth_res_map = self.raw2outputs(raw[...,:4], z_vals)

        ret["rgb_res_map"] = rgb_res_map
        ret["depth_res_map"] = depth_res_map
        ret['z_vals'] = z_vals
        ret['raw'] = raw

        return ret
    
    
    #@torch.cuda.amp.autocast(enabled=True)
    def mapping(self, rays_o, rays_d, target_rgb, target_d, tracking=False, render_flag=False, clamp=False):
        '''
        Perform mapping step for neural implicit scene representation.

        Args:
            rays_o (Tensor): Ray origins of shape (Bs, 3).
            rays_d (Tensor): Ray directions of shape (Bs, 3).
            target_rgb (Tensor): Target RGB values (Bs, 3).
            target_d (Tensor): Target depth values (Bs, 1).
            global_step (int, optional): Current training step.
            tracking (bool, optional): If True, indicates tracking mode.
            frame_id (Tensor, optional): Frame indices for pose correction.
            index_h (Tensor, optional): Height indices for selection.
            index_w (Tensor, optional): Width indices for selection.
            delta_flag (bool, optional): Reserved flag, unused.
            middle_mask (Tensor, optional): Middle mask for SDF loss.
            render_flag (bool, optional): If True, run render mode.
            clamp (bool, optional): If True, clamp outputs.

        Returns:
            dict: Dictionary containing losses and output maps, or rendering outputs if not training.
        '''
        # Store clamp state
        self.clamp = clamp

        # Run the rendering/prediction pipeline for the given rays
        rend_dict = self.render_rays(rays_o, rays_d, target_d=target_d, tracking=tracking, render_flag=render_flag)
        
        # If model is in eval mode (not training), simply return the rendered outputs
        if not self.training:
            return rend_dict
        
        # Create a valid mask for depth values that are in the allowed range
        valid_depth_mask = (target_d.squeeze() > 0.) * (target_d.squeeze() < self.config['cam']['depth_trunc'])  # shape: (N,)
        # Expand to match rgb channels for weighted loss computation
        rgb_weight = valid_depth_mask.clone().unsqueeze(-1) 
        
        # Set missing rgb weights for invalid depths to a configured small value
        rgb_weight[rgb_weight == 0] = self.config['training']['rgb_missing']

        # Compute RGB residual loss using weighted difference
        rgb_res_loss = compute_loss(rend_dict["rgb_res_map"] * rgb_weight, target_rgb * rgb_weight)

        # Compute depth residual loss using only valid depth pixels
        depth_res_loss = compute_loss(
            rend_dict["depth_res_map"].squeeze()[valid_depth_mask], 
            target_d.squeeze()[valid_depth_mask]
        )
        
        # Compute SDF (Signed Distance Function) related losses
        z_vals = rend_dict['z_vals']                # Sample depths along rays
        tsdf_res = rend_dict['raw'][..., 3]         # SDF/TSDF output channel
        truncation = self.config['training']['trunc'] * self.config['data']['sc_factor']

        # fs_res_loss: free space loss; sdf_res_loss: SDF regression loss
        fs_res_loss, sdf_res_loss = get_sdf_loss(
            z_vals, target_d, tsdf_res, truncation, loss_type='l2', middle_mask=valid_depth_mask
        )
        
        # Pack all the computed losses and outputs into a dictionary for optimization
        ret = {
            "rgb_res_loss": rgb_res_loss,
            "depth_res_loss": depth_res_loss,
            "sdf_res_loss": sdf_res_loss,
            "fs_res_loss": fs_res_loss,
            "rgb_res": rend_dict["rgb_res_map"],
            "depth_res": rend_dict["depth_res_map"],
        }

        return ret