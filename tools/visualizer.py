import argparse
import os
import time

import numpy as np
import torch
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader

import config
from viz import SLAMFrontend
from datasets.dataset import get_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments to visualize the SLAM process.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one inconfig file')
    parser.add_argument('--mesh_dir', type=str,
                        help='output folder, this have higher priority, can overwrite the one inconfig file')
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    parser.set_defaults(nice=True)
    parser.add_argument('--save_rendering',
                        action='store_true', help='save rendering video to `vis.mp4` in output folder ')
    parser.add_argument('--vis_input_frame',
                        action='store_true', help='visualize input frames')
    parser.add_argument('--no_gt_traj',
                        action='store_true', help='not visualize gt trajectory')
    args = parser.parse_args()
    cfg = config.load_config(args.config)
    scale = 1.0 #cfg['scale']
    output = cfg['data']['output'] if args.output is None else args.output
    mesh_dir =args.mesh_dir
    if args.vis_input_frame:
        frame_reader = get_dataset(cfg)
        frame_loader = DataLoader(
            frame_reader, batch_size=1, shuffle=False, num_workers=4)
    ckptsdir = f'{output}/ckpts'
    

    frame_reader = get_dataset(cfg)
    estimate_c2w_list = np.load(os.path.join(cfg['data']['output'], 'all_poses.npy')) #bs3d

    gt_poses = frame_reader.poses
    # print(len(gt_poses))
    gt_poses = torch.cat(gt_poses,dim=0).reshape(-1,4,4)
    gt_c2w_list = gt_poses.cpu().numpy()
    print("estimate_c2w_list",estimate_c2w_list.shape)
    print("gt_c2w_list",gt_c2w_list.shape)
    N = len(frame_reader)

    frontend = SLAMFrontend(output, init_pose=estimate_c2w_list[0], cam_scale=0.3,
                            save_rendering=args.save_rendering, near=0,
                            estimate_c2w_list=estimate_c2w_list, gt_c2w_list=gt_c2w_list).start()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #(*'mp4v')
    
    # video = cv2.VideoWriter('/home/lyq/Videos/building2_img_Test.mp4', fourcc, 30, (640, 720))

    
    time.sleep(5)
    for i in tqdm(range(0, N)):
        # show every second frame for speed up
        # if args.vis_input_frame and i % 2 == 0:
        if args.vis_input_frame and i % 2 == 0:

            ret = frame_reader[i]
            idx = ret['frame_id']
            gt_color = ret['rgb']
            gt_depth = ret['depth']
            gt_c2w = ret['c2w']

            depth_np = gt_depth.numpy()
            color_np = (gt_color.numpy()*255).astype(np.uint8)
            
            # max_depth = np.minimum(8.0,np.max(depth_np))
            max_depth = 7.0 #8.0
            depth_np = depth_np/max_depth*255
            depth_np = np.clip(depth_np, 0, 255).astype(np.uint8)
            depth_np = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET) #cv2.COLORMAP_JET
            color_np = np.clip(color_np, 0, 255)
            whole = np.concatenate([color_np, depth_np], axis=0)
            H, W, _ = whole.shape
            whole = cv2.resize(whole, (W//2, H//2))
            
            # video.write(whole[:, :, ::-1])
            cv2.imshow(f'Input RGB-D Sequence', whole[:, :, ::-1])

            cv2.waitKey(1)
        time.sleep(0.03)
        meshfile = f'{mesh_dir}/mesh_track{i:d}_cull_occlusion.ply'
        # print((W//2, H//2))
        if os.path.isfile(meshfile):
            print(meshfile)
            frontend.update_mesh(meshfile)
        frontend.update_pose(1, estimate_c2w_list[i], gt=False)
        if not args.no_gt_traj:
            frontend.update_pose(1, gt_c2w_list[i], gt=True)
        # the visualizer might get stucked if update every frame
        # with a long sequence (10000+ frames)
        if i % 10 == 0:
            frontend.update_cam_trajectory(i, gt=False)
            if not args.no_gt_traj:
                frontend.update_cam_trajectory(i, gt=True)

    # video.release()
    print("ending")
    if args.save_rendering:
        time.sleep(1)
        os.system(f"/usr/bin/ffmpeg -f image2 -r 30 -pattern_type glob -i '{output}/tmp_rendering/*.jpg' -y {output}/vis.mp4")
