import os
import argparse
import glob

import numpy as np
import torch
import imageio
import trimesh
import pyrender
from tqdm import tqdm
from copy import deepcopy

import sys
sys.path.append('../')

import config
from datasets.dataset import get_dataset

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def load_virt_cam_poses(path):
    poses = []
    pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                        key=lambda x: int(os.path.basename(x)[:-4]))
    for pose_path in pose_paths:
        with open(pose_path, "r") as f:
            lines = f.readlines()
        ls = []
        for line in lines:
            l = list(map(float, line.split(' ')))
            ls.append(l)
        c2w = np.array(ls).reshape(4, 4)
        # virt_cams are stored under OpenCV convention, so need to convert to OpenGL
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        poses.append(c2w)

    assert len(poses) > 0, "Make sure the path: {} really has virtual views!".format(path)
    print("Added {} virtual views from {}".format(len(poses), path))

    return poses



def load_cam_intrinsics_from_cfg(cfg):
    K = np.array([[cfg['cam']['fx'], .0, cfg['cam']['cx']],
                  [.0, cfg['cam']['fy'], cfg['cam']['cy']],
                  [.0, .0, 1.0]]).reshape(3, 3)
    H, W = cfg["cam"]["H"], cfg["cam"]["W"]
    return K, H, W


def render_depth_maps(mesh, poses, K, H, W, near=0.01, far=5.0):
    """
    :param mesh: Mesh to be rendered
    :param poses: list of camera poses (c2w under OpenGL convention)
    :param K: camera intrinsics [3, 3]
    :param W: width of image plane
    :param H: height of image plane
    :param near: near clip
    :param far: far clip
    :return: list of rendered depth images [H, W]
    """
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=near, zfar=far)
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)
    renderer = pyrender.OffscreenRenderer(W, H)
    render_flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY

    depth_maps = []
    for pose in poses:
        scene.set_pose(camera_node, pose)
        depth = renderer.render(scene, render_flags)
        depth_maps.append(depth)

    return depth_maps


def render_depth_maps_doublesided(mesh, poses, K, H, W, near=0.01, far=10.0):
    depth_maps_1 = render_depth_maps(mesh, poses, K, H, W, near=near, far=far)
    mesh.faces[:, [1, 2]] = mesh.faces[:, [2, 1]]
    depth_maps_2 = render_depth_maps(mesh, poses, K, H, W, near=near, far=far)
    mesh.faces[:, [1, 2]] = mesh.faces[:, [2, 1]]  # it's a pass by reference, so I restore the original order

    depth_maps = []
    for i in tqdm(range(len(depth_maps_1))):
        depth_map = np.where(depth_maps_1[i] > 0, depth_maps_1[i], depth_maps_2[i])
        depth_map = np.where((depth_maps_2[i] > 0) & (depth_maps_2[i] < depth_map), depth_maps_2[i], depth_map) #save the nearest depth
        depth_maps.append(depth_map)
        # print("depth_maps: ", depth_map.shape) #700 1260 m
        # print("depth_maps: ", depth_map)

    return depth_maps


def cull_by_bounds(points, scene_bounds, eps=0.02):
    """
    :param points:
    :param scene_bounds:
    :param eps:
    :return:
    """
    inside_mask = np.all(points >= (scene_bounds[0] - eps), axis=1) & np.all(points <= (scene_bounds[1] + eps), axis=1)
    return inside_mask

def pose_preprocess(all_pose):
    '''
    numpy array to torch tensor N,4,4
    '''
    all_RT = []
    for i in range(0, all_pose.shape[0]):
        T = all_pose[i,1:4]
        # quad = all_pose[i,4:]
        quad  = np.zeros(4)
        quad[0] = all_pose[i,7]
        quad[1] = all_pose[i,4]
        quad[2] = all_pose[i,5]
        quad[3] = all_pose[i,6]
        r = Rotation.from_quat(quad)
        R = r.as_matrix()
        RT_torch = torch.eye(4)
        RT_torch[:3,3]=torch.from_numpy(T)
        RT_torch[:3,:3]=torch.from_numpy(R)
        RT_torch=RT_torch.unsqueeze(0)
        all_RT.append(RT_torch)

    final = torch.cat(all_RT, dim=0)  # cat on dim=0
    return final

def cull_from_one_pose(points, pose, K, H, W, remove_occlusion=True, rendered_depth=None, eps=0.03):
    """
    :param points: mesh vertices [V, 3] np array
    :param pose: c2w under OpenGL convention (right-up-back) [3, 3] np array
    :param K: camera intrinsics [3, 3] np array
    :param rendered_depth: rendered depth image (optional)
    :param remove_occlusion:
    :return:
    """
    c2w = deepcopy(pose)
    # convert to OpenCV pose
    #TODO: originally reserved
    # c2w[:3, 1] *= -1
    # c2w[:3, 2] *= -1
    
    w2c = np.linalg.inv(c2w)
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]

    # pts under camera coordinate frame
    camera_space = rotation @ points.transpose() + translation[:, None]  # [3, N]
    uvz = (K @ camera_space).transpose()  # [N, 3]
    pz = uvz[:, 2] + 1e-8
    px = uvz[:, 0] / pz
    py = uvz[:, 1] / pz

    # step 1: frustum
    in_frustum_mask = (0 <= px) & (px <= W - 1) & (0 <= py) & (py <= H - 1) & (pz > 0)
    u = np.clip(px, 0, W - 1).astype(np.int32)
    v = np.clip(py, 0, H - 1).astype(np.int32)

    # step 2: self occlusion
    obs_mask = in_frustum_mask
    if remove_occlusion:
        assert rendered_depth is not None, "remove_occlusion requires rendered depth image!!!"
        obs_mask = in_frustum_mask & (pz < (rendered_depth[v, u] + eps))

    return in_frustum_mask.astype(np.int32), obs_mask.astype(np.int32)


def apply_culling_strategy(dataset, depth_flag, skip, points, poses, K, H, W,
                           rendered_depth_list=None,
                           remove_occlusion=True,
                           verbose=False,
                           virt_cam_starts=-1,
                           eps=0.03):
    """
    :param points:
    :param poses:
    :param K:
    :param H:
    :param W:
    :param rendered_depth_list:
    :param remove_occlusion:
    :param verbose:
    :param virt_cam_starts:
    :return:
    """

    in_frustum_mask = np.zeros(points.shape[0])
    obs_mask = np.zeros(points.shape[0])
    if depth_flag:
        idx = [i for i in dataset.frame_ids[::skip]]
    for i, pose in enumerate(tqdm(poses)):
        if verbose:
            print('Processing pose ' + str(i + 1) + ' out of ' + str(len(poses)))
        if depth_flag:
            # print("debug",idx[i],len(dataset))
            rendered_depth = dataset[idx[i]]['depth'].cpu().numpy()
        else:
            rendered_depth = rendered_depth_list[i] if rendered_depth_list is not None else None
        in_frustum, obs = cull_from_one_pose(points, pose, K, H, W,
                                             rendered_depth=rendered_depth,
                                             remove_occlusion=remove_occlusion,
                                             eps=eps)
        obs_mask = obs_mask + obs
        # virtual camera views shouldn't contribute to in_frustum_mask, it only adds more entries to obs_mask
        if virt_cam_starts < 0 or i < virt_cam_starts:
            in_frustum_mask = in_frustum_mask + in_frustum

    return in_frustum_mask, obs_mask


def cull_one_mesh(cfg, c2w_list, mesh_path, save_path, skip, dataset, depth_flag = None, save_unseen=False, remove_occlusion=True,
                  virtual_cameras=False, virt_cam_path=None, subdivide=False, max_edge=0.05,
                  scene_bounds=None, th_obs=0, eps=0.03, silent=True, platform='egl'):
    """
    :param cfg:
    :param c2w_list: list of camera poses (c2w under OpenGL)
    :param mesh_path:
    :param save_path:
    :param save_unseen: save unobserved vertices
    :param remove_occlusion:
    :param virtual_cameras:
    :param virt_cam_path: path to the dir that saves virtual camera views (optional)
    :param scene_bounds:
    :param th_obs:
    :param silent:
    :param platform:
    :return:
    """
    # load original mesh
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    vertices = mesh.vertices  # [V, 3]
    triangles = mesh.faces  # [F, 3]
    colors = mesh.visual.vertex_colors  # [V, 3]

    # cull with the bounding box first
    if scene_bounds is not None:
        inside_mask = cull_by_bounds(vertices, scene_bounds)
        inside_mask = inside_mask[triangles[:, 0]] | inside_mask[triangles[:, 1]] | inside_mask[triangles[:, 2]]
        triangles = triangles[inside_mask, :]

    os.environ['PYOPENGL_PLATFORM'] = platform
    # K, H, W = load_cam_intrinsics_from_cfg(cfg)
    K = np.array([
            [dataset.fx, 0, dataset.cx],
            [0, dataset.fy, dataset.cy],
            [0, 0, 1]
        ])
    H, W = dataset.H, dataset.W

    # add virtual cameras to camera poses list
    if virtual_cameras:
        virt_cam_starts = len(c2w_list)
        if virt_cam_path is None:
            virt_cam_path = os.path.join(cfg['data']['datadir'], 'virtual_cameras')  # hardcoded path
        c2w_list = c2w_list + load_virt_cam_poses(virt_cam_path)
    else:
        virt_cam_starts = -1

    # we don't need to subdivided mesh to render depth, so do the rendering before updating the mesh
    if remove_occlusion:
        if depth_flag:
            rendered_depth_maps = []
        else:
            print("rendering depth maps...")
            rendered_depth_maps = render_depth_maps_doublesided(mesh, c2w_list, K, H, W, near=0.01, far=10.0)
    else:
        rendered_depth_maps = None

    # update the mesh vertices and faces
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    mesh.remove_unreferenced_vertices()

    # start culling
    points = vertices[:, :3]  # [V, 3]
    in_frustum_mask, obs_mask = apply_culling_strategy(dataset, depth_flag, skip, points, c2w_list, K, H, W,
                                                       rendered_depth_list=rendered_depth_maps,
                                                       verbose=(not silent),
                                                       remove_occlusion=remove_occlusion,
                                                       virt_cam_starts=virt_cam_starts,
                                                       eps=eps)
    inf1 = in_frustum_mask[triangles[:, 0]]  # [F, 3]
    inf2 = in_frustum_mask[triangles[:, 1]]
    inf3 = in_frustum_mask[triangles[:, 2]]
    in_frustum_mask = (inf1 > th_obs) | (inf2 > th_obs) | (inf3 > th_obs)
    if remove_occlusion:
        obs1 = obs_mask[triangles[:, 0]]
        obs2 = obs_mask[triangles[:, 1]]
        obs3 = obs_mask[triangles[:, 2]]
        obs_mask = (obs1 > th_obs) | (obs2 > th_obs) | (obs3 > th_obs)
        valid_mask = in_frustum_mask & obs_mask  # [F,]
    else:
        valid_mask = in_frustum_mask
    triangles_observed = triangles[valid_mask, :]

    # save culled mesh
    mesh = trimesh.Trimesh(vertices, triangles_observed, vertex_colors=colors, process=False)
    mesh.remove_unreferenced_vertices()
    mesh.export(save_path)
    print("Mesh is saved to ", save_path)
    # save unobserved points
    if save_unseen:
        triangles_not_observed = triangles[~valid_mask, :]
        mesh_not_observed = trimesh.Trimesh(vertices, triangles_not_observed, process=False)
        mesh_not_observed.remove_unreferenced_vertices()
        save_dir = os.path.dirname(save_path)
        scene_name = save_path.split("/")[-1].split("_")[0]
        mesh_not_observed.export("{}/{}_unseen.ply".format(save_dir, scene_name))
        pc_unseen = mesh_not_observed.vertices
        np.save("{}/{}_pc_unseen.npy".format(save_dir, scene_name), pc_unseen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for mesh culling script."
    )
    parser.add_argument("--config", required=True, type=str, help="config file for the scene of input mesh")
    parser.add_argument("--input_mesh", required=True, type=str, help="path to the mesh to be culled")
    parser.add_argument("--output_mesh", type=str, help="path to save the culled mesh (optional)")
    parser.add_argument("--remove_occlusion", dest="remove_occlusion", action="store_true")
    parser.add_argument("--virtual_cameras", dest="virtual_cameras", action="store_true")
    parser.add_argument("--virt_cam_path", type=str, help="path to virtual camera DIR (optional)")
    parser.add_argument("--gt_pose", dest="gt_pose", action="store_true")
    parser.add_argument("--gt_depth", dest="gt_depth", action="store_true")
    parser.add_argument("--ckpt_path", type=str, help="path to checkpoint file")
    parser.add_argument("--skip", type=int, default=2) #default:2
    parser.add_argument("--th_obs", type=int, default=0)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--range", type=int, default=0)
    parser.add_argument('--pose', type=str, default="default",help='path to estimated pose')
    parser.add_argument("--transform", dest="transform", action="store_true")
    
    args = parser.parse_args()

    if args.remove_occlusion and args.virtual_cameras:
        print("Using Co-SLAM culling strategy...")
        suffix = "virt_cams"
    elif args.remove_occlusion:
        print("Using Neural-RGBD/GO-Surf culling strategy...")
        suffix = "occlusion"
    else:
        print("Using iMAP/NICE-SLAM culling strategy...")
        suffix = "frustum"

    mesh_path = args.input_mesh
    if args.output_mesh is None:
        save_path = mesh_path.replace(".ply", "_cull_{}.ply".format(suffix))
    else:
        save_path = args.output_mesh

    
    depth_flag = False
    cfg = config.load_config(args.config,cull_mesh=True)

    dataset = get_dataset(cfg)
    if args.gt_depth:
        depth_flag = True
        
    if args.gt_pose:
        if args.range>0:
            # c2w_list = [np.array(dataset.poses[i]).astype(np.float32) for i in dataset.frame_ids[::args.skip]][:args.range]
            c2w_list = [np.array(dataset.poses[i]).astype(np.float32) for i in dataset.frame_ids[:args.range][::args.skip]]
        else:
            c2w_list = [np.array(dataset.poses[i]).astype(np.float32) for i in dataset.frame_ids[::args.skip]]
            # gt depth for occlusion removal
    else:
        if args.pose == "default":
            assert args.ckpt_path is not None and os.path.exists(args.ckpt_path), "Please ensure you provided ckpt path and it exists!!!"
            c2w_list_dict = torch.load(args.ckpt_path)["pose"]
            c2w_list = [c2w_list_dict[i].cpu().numpy() for i in range(0, len(c2w_list_dict.keys()), args.skip)]
        else:
            if "txt" in args.pose:
                all_pose_np = []
                all_pose = np.loadtxt(args.pose)[:,1:]# tx ty tz qx qy qz qw
                for i in range(all_pose.shape[0]):
                    T_quad=all_pose[i]
                    trans=T_quad[:3]
                    rot = torch.from_numpy(T_quad[3:])
                    # rot = torch.from_numpy(np.array([T_quad[6],T_quad[3],T_quad[4],T_quad[5]])) #Mipsfusion
                    
                    T = torch.eye(4)
                    R = quaternion_to_matrix(rot).squeeze() #require qw qx qy qz
                    T[:3, :3] = R
                    T[:3, 3] = torch.from_numpy(trans)
                    all_pose_np.append(T.cpu().numpy())
                all_pose_np = np.array(all_pose_np)
            else:
                all_pose_np = np.load(args.pose)
            if args.transform:
                all_pose_np[:,:3, 1] *= -1
                all_pose_np[:,:3, 2] *= -1
            if args.range>0:
                all_pose_np=all_pose_np[:args.range]
            c2w_list = [all_pose_np[i] for i in range(0, all_pose_np.shape[0], args.skip)]

    cull_one_mesh(cfg, c2w_list, mesh_path, save_path, args.skip, dataset, depth_flag = depth_flag, save_unseen=False,
                  remove_occlusion=args.remove_occlusion, scene_bounds=None,
                  eps=args.eps, th_obs=args.th_obs, silent=True, platform='egl',
                  virtual_cameras=args.virtual_cameras, virt_cam_path=args.virt_cam_path)


#python cull_mesh.py --config ../configs/BS3D/study.yaml --input_mesh  /home/lyq/myprojects/remix_files/remixfusion_results/results/BS3D_badslam/study/study_changed_mesh.ply  --gt_pose  --remove_occlusion --gt_depth --eps 0.1 --skip 5
