# package imports
import torch
import torch.nn.functional as F
from math import exp, log, floor
import numpy as np 
from scipy.spatial.transform import Rotation


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

def matrixtoeular(Rmatrix):
    r =  Rotation.from_matrix(Rmatrix)
    euler_angles = r.as_euler('xyz', degrees=True)
    return euler_angles

def start_timing():
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return start, end

def end_timing(start, end):
    torch.cuda.synchronize()
    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end)
    return elapsed_time

def check_orthogonal(matrix):
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()
    transpose_matrix = np.transpose(matrix)
    
    product_matrix = np.dot(matrix, transpose_matrix)
    
    diagonal_elements = np.diagonal(product_matrix)
    # print("diagonal_elements:",diagonal_elements)
    if not np.allclose(diagonal_elements, 1, atol=1e-6):
        return False
    
    non_diagonal_elements = product_matrix - np.diag(diagonal_elements)
    if not np.allclose(non_diagonal_elements, 0, atol=1e-6):
        return False
    
    return True

def orthogonalize_rotation_matrix_tolerate(matrix, epsilon=1e-10):
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()
    U, _, V = np.linalg.svd(matrix)
    orthogonal_matrix = np.dot(U, V)
    orthogonal_matrix[np.abs(orthogonal_matrix - 1) < epsilon] = 1
    orthogonal_matrix[np.abs(orthogonal_matrix + 1) < epsilon] = -1
    return orthogonal_matrix

def orthogonalize_rotation_matrix(matrix):
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()
    q1 = matrix[:, 0] / np.linalg.norm(matrix[:, 0])
    
    q2 = matrix[:, 1] - np.dot(matrix[:, 1], q1) * q1
    q2 = q2 / np.linalg.norm(q2)
    
    q3 = np.cross(q1, q2)
    
    orthogonal_matrix = np.column_stack((q1, q2, q3))
    
    return orthogonal_matrix

def mse2psnr(x):
    '''
    MSE to PSNR
    '''
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x)

def coordinates(voxel_dim, device: torch.device):
    '''
    Params: voxel_dim: int or tuple of int
    Return: coordinates of the voxel grid
    '''
    if type(voxel_dim) is int:
        nx = ny = nz = voxel_dim
    else:
        nx, ny, nz = voxel_dim[0], voxel_dim[1], voxel_dim[2]
    x = torch.arange(0, nx, dtype=torch.long, device=device)
    y = torch.arange(0, ny, dtype=torch.long, device=device)
    z = torch.arange(0, nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")

    return torch.stack((x.flatten(), y.flatten(), z.flatten()))

def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    '''
    Params:
        bins: torch.Tensor, (Bs, N_samples)
        weights: torch.Tensor, (Bs, N_samples)
        N_importance: int
    Return:
        samples: torch.Tensor, (Bs, N_importance)
    '''
    # device = weights.get_device()
    device = weights.device
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True) # Bs, N_samples-2
    cdf = torch.cumsum(pdf, -1) 
    cdf = torch.cat([torch.zeros_like(cdf[..., :1], device=device), cdf], -1) # Bs, N_samples-1
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / N_importance, 1. - 0.5 / N_importance, steps=N_importance, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom, device=device), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def batchify(fn, chunk=1024*64, ): #chunk=1024*64
    """
    Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        raw_list =[]

        for i in range(0, inputs.shape[0], chunk):

            raw_chunk = fn(inputs[i:i+chunk])
            raw_list.append(raw_chunk)

        raw = torch.cat(raw_list, 0)

        return raw#, delta
    
    return ret


def get_masks(z_vals, target_d, truncation):
    '''
    Params:
        z_vals: torch.Tensor, (Bs, N_samples)
        target_d: torch.Tensor, (Bs,)
        truncation: float
    Return:
        front_mask: torch.Tensor, (Bs, N_samples)
        sdf_mask: torch.Tensor, (Bs, N_samples)
        fs_weight: float
        sdf_weight: float
    '''

    # before truncation
    front_mask = torch.where(z_vals < (target_d - truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals))
    # after truncation
    back_mask = torch.where(z_vals > (target_d + truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals))
    # valid mask
    depth_mask = torch.where(target_d > 0.0, torch.ones_like(target_d), torch.zeros_like(target_d))
    # Valid sdf regionn
    sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask

    num_fs_samples = torch.count_nonzero(front_mask)
    num_sdf_samples = torch.count_nonzero(sdf_mask)
    num_samples = num_sdf_samples + num_fs_samples
    fs_weight = 1.0 - num_fs_samples / num_samples
    sdf_weight = 1.0 - num_sdf_samples / num_samples

    return front_mask, sdf_mask, fs_weight, sdf_weight

def compute_loss(prediction, target, loss_type='l2'):
    '''
    Params: 
        prediction: torch.Tensor, (Bs, N_samples)
        target: torch.Tensor, (Bs, N_samples)
        loss_type: str
    Return:
        loss: torch.Tensor, (1,)
    '''

    if loss_type == 'l2':
        return F.mse_loss(prediction, target)
    elif loss_type == 'l1':
        return F.l1_loss(prediction, target)

    raise Exception('Unsupported loss type')



def get_sdf_loss(z_vals, target_d, predicted_sdf, truncation, loss_type=None, grad=None, middle_mask=None):
    """
    Compute SDF-based losses for depth supervision and free space constraints.

    Args:
        z_vals (torch.Tensor): Sampled depths along rays, (Bs, N_samples).
        target_d (torch.Tensor): Ground truth depth values, (Bs,).
        predicted_sdf (torch.Tensor): Predicted SDF/TSDF values at each sample, (Bs, N_samples).
        truncation (float): Truncation value (usually in meters in world units).
        loss_type (str, optional): Type of loss to use ('l1', 'l2', ...). Default: None.
        grad (torch.Tensor, optional): Gradient of the SDF prediction (for Eikonal loss). Default: None.
        middle_mask (torch.Tensor, optional): Mask (Bs,) to suppress loss on certain rays. Default: None.

    Returns:
        fs_loss (torch.Tensor): Free space loss scalar.
        sdf_loss (torch.Tensor): SDF regression loss scalar.
        eikonal_loss (torch.Tensor, optional): Eikonal loss scalar (if grad is not None).
    """
    # Compute spatial masks and weighting factors for free space and SDF losses.
    front_mask, sdf_mask, fs_weight, sdf_weight = get_masks(z_vals, target_d, truncation)

    # Optionally mask out losses for specific rays (e.g., invalid/missing data).
    if middle_mask is not None:
        front_mask *= middle_mask[..., None]
        sdf_mask *= middle_mask[..., None]

    # Free space loss: penalize SDF predictions < 0 in free space (in front of surface depth).
    fs_loss = compute_loss(predicted_sdf * front_mask, torch.ones_like(predicted_sdf) * front_mask, loss_type) * fs_weight

    # SDF regression loss: force SDF prediction at the surface to match observed depth.
    sdf_loss = compute_loss((z_vals + predicted_sdf * truncation) * sdf_mask, target_d * sdf_mask, loss_type) * sdf_weight

    # Optionally compute Eikonal loss (enforces |grad SDF| = 1 for true SDFs).
    if grad is not None:
        eikonal_loss = (((grad.norm(2, dim=-1) - 1) ** 2) * sdf_mask / sdf_mask.sum()).sum()
        return fs_loss, sdf_loss, eikonal_loss

    return fs_loss, sdf_loss

def matrixtoeular(Rmatrix):
    r = Rotation.from_matrix(Rmatrix)
    # transform matrix to eular
    euler_angles = r.as_euler('xyz', degrees=True)
    return euler_angles