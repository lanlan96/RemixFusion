
import torch
import torch.nn as nn
from kornia.geometry.conversions import angle_axis_to_rotation_matrix as at_to_transform_matrix
from kornia.geometry.conversions import rotation_matrix_to_angle_axis as matrix_to_axis_angle


def make_c2w(r, t, first=False):
    """
    :param r:  (3, ) axis-angle             torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4)
    """
    c2w = torch.eye(4).type_as(r).unsqueeze(0).repeat(r.shape[0],1,1)

    R = at_to_transform_matrix(r)  # (3, 3)
    c2w[:,:3, :3] = R
    c2w[:,:3, 3] = t

    return c2w

# RBA: Residual Bundle Adjustment
class RBA(nn.Module):
    def __init__(self, num_cams, init_c2w=None, layers=2, scale=1e-2, out_dir=None):
        """
        :param num_cams: Number of keyframes (cameras)
        :param init_c2w: (N, 4, 4) torch tensor of initial camera-to-world transforms
        """
        super(RBA, self).__init__()
        self.num_cams = num_cams
        self.scale = scale
        self.out_dir = out_dir
        if init_c2w is not None:
            self.init_c2w = init_c2w.clone().detach()
            self.init_r = []
            self.init_t = []
            for idx in range(num_cams):
                r_init = matrix_to_axis_angle(self.init_c2w[idx][:3, :3].reshape([1, 3, 3])).reshape(-1)
                t_init = self.init_c2w[idx][:3, 3].reshape(-1)
                self.init_r.append(r_init)
                self.init_t.append(t_init)
            self.init_r = torch.stack(self.init_r)  # [num_cams, 3]
            self.init_t = torch.stack(self.init_t)  # [num_cams, 3]
        else:
            self.init_r = torch.zeros(size=(num_cams, 3), dtype=torch.float32).cuda()
            self.init_t = torch.zeros(size=(num_cams, 3), dtype=torch.float32).cuda()
            self.init_c2w = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(num_cams, 1, 1).cuda()

        d_in = 7  # cam_id (1) + rotation(3) + translation(3) = 7

        activation_func = nn.ELU(inplace=True)

        layers_list = nn.Sequential(nn.Linear(d_in, 256),
                                    activation_func)
        for i in range(layers):
            layers_list.append(nn.Sequential(nn.Linear(256, 256),
                                             activation_func))
        layers_list.append(nn.Linear(256, 6))
        
        self.layers = nn.Sequential(*layers_list)

    def get_init_pose(self, cam_id):
        return self.init_c2w[cam_id]

    def update_init_pose(self, cam_id, c2w):
        self.init_c2w[cam_id] = c2w.clone().detach()
        r_init = matrix_to_axis_angle(self.init_c2w[cam_id][:3, :3].reshape([1, 3, 3]).contiguous()).reshape(-1)
        t_init = self.init_c2w[cam_id][:3, 3].reshape(-1)
        self.init_r[cam_id] = r_init
        self.init_t[cam_id] = t_init

    def forward(self, cam_id):
        if not isinstance(cam_id, torch.Tensor):
            if cam_id == 0:
                return self.init_c2w[0]
        else:
            cam_id_tensor = cam_id.type_as(self.init_c2w)

        # Normalize to [-1, 1]
        cam_id_tensor = (cam_id_tensor / self.num_cams) * 2 - 1

        init_r = self.init_r[cam_id].squeeze()
        init_t = self.init_t[cam_id].squeeze()

        if len(init_r.shape) == 1:
            init_r = init_r.unsqueeze(0)
            init_t = init_t.unsqueeze(0)
        inputs = torch.cat([cam_id_tensor, init_r, init_t], dim=-1)

        out = self.layers(inputs) * self.scale

        if 0 in cam_id:
            out[0, ...] = 0
        r = out[:, :3] + self.init_r[cam_id].squeeze()
        t = out[:, 3:] + self.init_t[cam_id].squeeze()

        c2w = make_c2w(r, t)
        return c2w


