import os
import torch
import numpy as np
import trimesh
import marching_cubes as mcubes
import cv2
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from skimage import measure
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix, rotation_6d_to_matrix, quaternion_to_axis_angle


def add_noise(pose, rotation_noise=0.1, translation_noise=0.05):
    """
    Add random noise to pose (4x4 transformation matrix).
    Args:
        pose (np.ndarray): 4x4 transformation matrix.
        rotation_noise (float): Maximum rotation noise (radians) for each axis.
        translation_noise (float): Maximum translation noise for each axis (same scale as pose units).
    Returns:
        noise_matrix (np.ndarray): 4x4 noisy transformation matrix.
    """

    # Initialize identity rotation matrix
    R = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])  

    # Initialize zero translation vector
    t = np.array([0, 0, 0])  
    
    # Generate random small rotation (in axis-angle form) and translation error
    rotation_error = np.random.uniform(-rotation_noise, rotation_noise, size=(3,))
    translation_error = np.random.uniform(-translation_noise, translation_noise, size=(3,))
    
    # Convert axis-angle rotation error to a rotation matrix (Rodrigues' formula)
    perturbation_matrix = np.zeros((3, 3))  # Placeholder for rotation matrix
    cv2.Rodrigues(rotation_error, perturbation_matrix)

    # Compute new rotation matrix and translation with noise
    noisy_R = R @ perturbation_matrix
    noisy_t = t + translation_error

    # Use noisy_R as the rotation noise matrix
    Q = noisy_R

    # Initialize output transformation as identity
    noise_matrix = np.eye(4)

    # Apply translation noise to input pose
    noise_matrix[:3, 3] = pose[:3, 3] + noisy_t
    # Apply rotation noise to input pose rotation
    noise_matrix[:3, :3] = Q @ pose[:3, :3]

    return noise_matrix

#### GO-Surf ####
def coordinates(voxel_dim, device: torch.device, flatten=True):
    if type(voxel_dim) is int:
        nx = ny = nz = voxel_dim
    else:
        nx, ny, nz = voxel_dim[0], voxel_dim[1], voxel_dim[2]
    x = torch.arange(0, nx, dtype=torch.long, device=device)
    y = torch.arange(0, ny, dtype=torch.long, device=device)
    z = torch.arange(0, nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")

    if not flatten:
        return torch.stack([x, y, z], dim=-1)

    return torch.stack((x.flatten(), y.flatten(), z.flatten()))


def getVoxels(x_max, x_min, y_max, y_min, z_max, z_min, voxel_size=None, resolution=None):

    if not isinstance(x_max, float):
        x_max = float(x_max)
        x_min = float(x_min)
        y_max = float(y_max)
        y_min = float(y_min)
        z_max = float(z_max)
        z_min = float(z_min)
    
    if voxel_size is not None:
        Nx = round((x_max - x_min) / voxel_size + 0.0005)
        Ny = round((y_max - y_min) / voxel_size + 0.0005)
        Nz = round((z_max - z_min) / voxel_size + 0.0005)

        tx = torch.linspace(x_min, x_max, Nx + 1)
        ty = torch.linspace(y_min, y_max, Ny + 1)
        tz = torch.linspace(z_min, z_max, Nz + 1)
    else:
        tx = torch.linspace(x_min, x_max, resolution)
        ty = torch.linspace(y_min, y_max,resolution)
        tz = torch.linspace(z_min, z_max, resolution)


    return tx, ty, tz

def get_batch_query_fn(query_fn, num_args=1 ,device=None):

    if device is not None:
        if num_args == 1:
            fn = lambda f, i0, i1: query_fn(f[i0:i1, None, :].to(device))
        else:
            fn = lambda f, f1, i0, i1: query_fn(f[i0:i1, None, :].to(device), f1[i0:i1, :].to(device))
    else:
        if num_args == 1:
            fn = lambda f, i0, i1: query_fn(f[i0:i1, None, :])
        else:
            fn = lambda f, f1, i0, i1: query_fn(f[i0:i1, None, :], f1[i0:i1, :])

    return fn


@torch.no_grad()
def extract_mesh_github(query_fn, query_w_fn, config, bounding_box, marching_cube_bound=None, color_func = None, voxel_size=None, resolution=None, isolevel=0.0, scene_name='', mesh_savepath=''):
    '''
    Extracts mesh from the scene model using marching cubes (Adapted from NeuralRGBD)
    '''
    # Query network on dense 3d grid of points
    if marching_cube_bound is None:
        marching_cube_bound = bounding_box

    x_min, y_min, z_min = marching_cube_bound[:, 0]
    x_max, y_max, z_max = marching_cube_bound[:, 1]
    tx, ty, tz = getVoxels(x_max, x_min, y_max, y_min, z_max, z_min, voxel_size, resolution)
    query_pts = torch.stack(torch.meshgrid(tx, ty, tz, indexing='ij'), -1).to(torch.float32)
    
    sh = query_pts.shape
    flat = query_pts.reshape([-1, 3])
    bounding_box_cpu = bounding_box.cpu()

    if config['grid']['tcnn_encoding']:
        flat = (flat - bounding_box_cpu[:, 0]) / (bounding_box_cpu[:, 1] - bounding_box_cpu[:, 0])
    fn = get_batch_query_fn(query_fn, device=bounding_box.device)
    w_fn = get_batch_query_fn(query_w_fn, device=bounding_box.device)

    chunk = 1024 * 64
    raw = [fn(flat, i, i + chunk).cpu().data.numpy() for i in range(0, flat.shape[0], chunk)]
    weight = [w_fn(flat, i, i + chunk).cpu().data.numpy() for i in range(0, flat.shape[0], chunk)]
    
    raw = np.concatenate(raw, 0).astype(np.float32)
    raw = np.reshape(raw, list(sh[:-1]) + [-1])
    
    weight = np.concatenate(weight, 0).astype(np.float32)
    weight = np.reshape(weight, list(sh[:-1]) + [-1])


    tsdf = raw.squeeze()   # [X, Y, Z]


    mask = (weight.squeeze() > 0)


    invalid_back = tsdf > -0.95   

    mask = mask #& (~invalid_back)
    
    # mask = (weight>0) #* (raw>-0.8)
    # print('Running Marching Cubes')
    # vertices, triangles = mcubes.marching_cubes(raw.squeeze(), isolevel, truncation=3.0)
    vertices, triangles, _, _ = measure.marching_cubes(raw.squeeze(), level=isolevel, mask=mask.squeeze())


    # normalize vertex positions
    vertices[:, :3] /= np.array([[tx.shape[0] - 1, ty.shape[0] - 1, tz.shape[0] - 1]])

    # Rescale and translate
    tx = tx.cpu().data.numpy()
    ty = ty.cpu().data.numpy()
    tz = tz.cpu().data.numpy()
    
    scale = np.array([tx[-1] - tx[0], ty[-1] - ty[0], tz[-1] - tz[0]])
    offset = np.array([tx[0], ty[0], tz[0]])
    vertices[:, :3] = scale[np.newaxis, :] * vertices[:, :3] + offset

    # Transform to metric units
    vertices[:, :3] = vertices[:, :3] / config['data']['sc_factor'] - config['data']['translation']


    if color_func is not None:
        if config['grid']['tcnn_encoding']:
            vertices = np.ascontiguousarray(vertices)
            vert_flat = (torch.from_numpy(vertices).to(bounding_box) - bounding_box[:, 0]) / (bounding_box[:, 1] - bounding_box[:, 0])

        fn_color = get_batch_query_fn(color_func, 1)

        chunk = 1024 * 64
        raw = [fn_color(vert_flat,  i, i + chunk).cpu().data.numpy() for i in range(0, vert_flat.shape[0], chunk)]

        sh = vert_flat.shape
        
        raw = np.concatenate(raw, 0).astype(np.float32)
        color = np.reshape(raw, list(sh[:-1]) + [-1])
        color = np.clip(color, 0, 1) * 255
        mesh = trimesh.Trimesh(vertices, triangles, process=False, vertex_colors=color)

    else:
        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles, process=False)

    os.makedirs(os.path.split(mesh_savepath)[0], exist_ok=True)
    mesh.export(mesh_savepath)

    # print('Mesh saved')
    return mesh
#### #### 




#### NeuralRGBD ####
@torch.no_grad()
def extract_mesh(query_fn, config, bounding_box, marching_cube_bound=None, color_func = None, voxel_size=None, resolution=None, isolevel=0.0, scene_name='', mesh_savepath='',mesh_bound=None):
    '''
    Extracts mesh from the scene model using marching cubes (Adapted from NeuralRGBD)
    '''
    with torch.no_grad():
        # Query network on dense 3d grid of points
        if marching_cube_bound is None:
            marching_cube_bound = bounding_box

        x_min, y_min, z_min = marching_cube_bound[:, 0]
        x_max, y_max, z_max = marching_cube_bound[:, 1]

        tx, ty, tz = getVoxels(x_max, x_min, y_max, y_min, z_max, z_min, voxel_size, resolution)
        query_pts = torch.stack(torch.meshgrid(tx, ty, tz, indexing='ij'), -1).to(torch.float32)
        chunk = 500000
        
        sh = query_pts.shape
        if mesh_bound is None:
            flat = query_pts.reshape([-1, 3]).to(bounding_box[:, 0]) #8KW,3
        else:
            flat = query_pts.reshape([-1, 3]) #8KW,3
        flat_np = flat.cpu().numpy()
        print("flat:",flat.shape)
        #TODO:filter those out of volume, and remove noisy pts
        # if mesh_bound is not None:
        mask = np.zeros(flat.shape[0])
        
        min_x = np.amin(mesh_bound.vertices[:, 0])
        max_x = np.amax(mesh_bound.vertices[:, 0])
        min_y = np.amin(mesh_bound.vertices[:, 1])
        max_y = np.amax(mesh_bound.vertices[:, 1])
        min_z = np.amin(mesh_bound.vertices[:, 2])
        max_z = np.amax(mesh_bound.vertices[:, 2])
        
        min_x = min_x*1.02 if min_x<0 else min_x*0.98
        min_y = min_y*1.02 if min_y<0 else min_y*0.98
        min_z = min_z*1.02 if min_z<0 else min_z*0.98
        max_x = max_x*1.02 if max_x>0 else max_x*0.98
        max_y = max_y*1.02 if max_y>0 else max_y*0.98
        max_z = max_z*1.02 if max_z>0 else max_z*0.98
        
        print(min_x, max_x, min_y, max_y, min_z,max_z)
        # mask = []
        c1=(min_x<flat_np[:,0]).astype(np.bool)
        c2=(max_x>flat_np[:,0]).astype(np.bool)
        c3=(min_y<flat_np[:,1]).astype(np.bool)
        c4=(max_y>flat_np[:,1]).astype(np.bool)
        c5=(min_z<flat_np[:,2]).astype(np.bool)
        c6=(max_z>flat_np[:,2]).astype(np.bool)
        # mask =  np.where(c1 & c2 & c3 & c4 & c5 & c6, np.ones(flat_np.shape[0]),np.zeros(flat_np.shape[0]))
        indices =  np.where(c1 & c2 & c3 & c4 & c5 & c6)[0]
        mask[indices] = 1
        mask=mask.astype(bool)

        #TODO:
        flat = flat.to(bounding_box[:, 0])
        
        if config['grid']['tcnn_encoding']:
            flat = (flat - bounding_box[:, 0]) / (bounding_box[:, 1] - bounding_box[:, 0])

        fn = get_batch_query_fn(query_fn)


        print("flat:",flat)
        raw = [fn(flat, i, i + chunk).cpu().data.numpy() for i in range(0, flat.shape[0], chunk)]
        
        raw = np.concatenate(raw, 0).astype(np.float32)
        if mesh_bound is not None:
            raw[~mask] = 1 #100 #-1
        print("raw:",raw[mask])
        raw = np.reshape(raw, list(sh[:-1]) + [-1])

        print('Running Marching Cubes')
        vertices, triangles = mcubes.marching_cubes(raw.squeeze(), isolevel, truncation=3.0)
        print('done', vertices.shape, triangles.shape)

        # normalize vertex positions
        vertices[:, :3] /= np.array([[tx.shape[0] - 1, ty.shape[0] - 1, tz.shape[0] - 1]])

        # Rescale and translate
        tx = tx.cpu().data.numpy()
        ty = ty.cpu().data.numpy()
        tz = tz.cpu().data.numpy()
        
        scale = np.array([tx[-1] - tx[0], ty[-1] - ty[0], tz[-1] - tz[0]])
        offset = np.array([tx[0], ty[0], tz[0]])
        vertices[:, :3] = scale[np.newaxis, :] * vertices[:, :3] + offset

        # Transform to metric units
        vertices[:, :3] = vertices[:, :3] / config['data']['sc_factor'] - config['data']['translation']

        if color_func is not None:
            #TODO:
            vert_flat = torch.from_numpy(vertices).to(bounding_box)
            if config['grid']['tcnn_encoding']:
                vert_flat = (torch.from_numpy(vertices).to(bounding_box) - bounding_box[:, 0]) / (bounding_box[:, 1] - bounding_box[:, 0])
            
            fn_color = get_batch_query_fn(color_func, 1)

            chunk = 1024 * 64
            raw = [fn_color(vert_flat,  i, i + chunk).cpu().data.numpy() for i in range(0, vert_flat.shape[0], chunk)]

            sh = vert_flat.shape
            
            raw = np.concatenate(raw, 0).astype(np.float32)
            color = np.reshape(raw, list(sh[:-1]) + [-1])
            color = np.clip(color, 0, 1) * 255
            mesh = trimesh.Trimesh(vertices, triangles, process=False, vertex_colors=color)
        else:
            # Create mesh
            mesh = trimesh.Trimesh(vertices, triangles, process=False)

        
        os.makedirs(os.path.split(mesh_savepath)[0], exist_ok=True)
        mesh.export(mesh_savepath)

        print('Mesh saved')
        return mesh
    #### #### 



def mse2psnr(x):
    return -10.*torch.log(x)/torch.log(torch.tensor(10.))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img, gt, window_size=11, size_average=True):
    img = torch.where(gt!=0, img, 0.)
    channel = img.size(-3)
    window = create_window(window_size, channel)

    if img.is_cuda:
        window = window.cuda(img.get_device())
    window = window.type_as(img)

    return _ssim(img, gt, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map, ssim_map.mean()
    else:
        return ssim_map, ssim_map.mean(1).mean(1).mean(1)




def axis_angle_to_matrix(data):
    batch_dims = data.shape[:-1]

    theta = torch.norm(data, dim=-1, keepdim=True)
    omega = data / theta

    omega1 = omega[...,0:1]
    omega2 = omega[...,1:2]
    omega3 = omega[...,2:3]
    zeros = torch.zeros_like(omega1)

    K = torch.concat([torch.concat([zeros, -omega3, omega2], dim=-1)[...,None,:],
                      torch.concat([omega3, zeros, -omega1], dim=-1)[...,None,:],
                      torch.concat([-omega2, omega1, zeros], dim=-1)[...,None,:]], dim=-2)
    I = torch.eye(3).expand(*batch_dims,3,3).to(data)

    return I + torch.sin(theta).unsqueeze(-1) * K + (1. - torch.cos(theta).unsqueeze(-1)) * (K @ K)

def matrix_to_axis_angle(rot):
    """
    :param rot: [N, 3, 3]
    :return:
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(rot))

def at_to_transform_matrix(rot, trans):
    """
    :param rot: axis-angle [bs, 3]
    :param trans: translation vector[bs, 3]
    :return: transformation matrix [b, 4, 4]
    """
    bs = rot.shape[0]
    T = torch.eye(4).to(rot)[None, ...].repeat(bs, 1, 1)
    R = axis_angle_to_matrix(rot)
    T[:, :3, :3] = R
    T[:, :3, 3] = trans
    return T

def qt_to_transform_matrix(rot, trans):
    """
    :param quad: axis-angle [bs, 4]
    :param trans: translation vector[bs, 3]
    :return: transformation matrix [b, 4, 4]
    """
    bs = rot.shape[0]
    T = torch.eye(4).to(rot)[None, ...].repeat(bs, 1, 1)
    R = quaternion_to_matrix(rot)
    T[:, :3, :3] = R
    T[:, :3, 3] = trans
    return T

def six_t_to_transform_matrix(rot, trans):
    """
    :param rot: 6d rotation [bs, 6]
    :param trans: translation vector[bs, 3]
    :return: transformation matrix [b, 4, 4]
    """
    bs = rot.shape[0]
    T = torch.eye(4).to(rot)[None, ...].repeat(bs, 1, 1)
    R = rotation_6d_to_matrix(rot)
    T[:, :3, :3] = R
    T[:, :3, 3] = trans
    return 