#!/bin/bash


scenes="cafeteria lounge foobar corridor hub juice study waiting" 

echo "Start running on BS3D dataset..."

for sc in ${scenes}
do
  echo "running on ${sc}..."
  # REMIXFUSION
  python eval_recon_revised.py --rec_mesh /home/lyq/myprojects/Remixfusion-sig/output/BS3D/${sc}/test/${sc}_cull_occlusion.ply --gt_mesh /media/lyq/data/dataset/BS3D/${sc}/mesh_cull_occlusion.ply --dataset_type BS3D -3d --icp_th 0.3 --com_th 0.1 
  # MonoGS
  # python eval_recon_revised.py --rec_mesh /home/lyq/myprojects/RTG-SLAM/output/dataset/BS3D_slam/${sc}/image/mesh_gap10_cull_occlusion.ply --gt_mesh /media/lyq/data/dataset/BS3D/${sc}/mesh_cull_occlusion.ply --dataset_type BS3D -3d --icp_th 0.3 --com_th 0.1 
  # ESLAM
  # python eval_recon_revised.py --rec_mesh /media/lyq/兰博Data/Remix-结果/ESLAM-renderandmesh/${sc}/final_mesh_cull_occlusion.ply --gt_mesh /media/lyq/data/dataset/BS3D/${sc}/mesh_cull_occlusion.ply --dataset_type BS3D -3d --icp_th 0.3 --com_th 0.1 
  # Splatam
  # python eval_recon_revised.py --rec_mesh /home/lyq/myprojects/SplaTAM/experiments/bs3d/${sc}_seed0/eval/mesh_cull_occlusion.ply --gt_mesh /media/lyq/data/dataset/BS3D/${sc}/mesh_cull_occlusion.ply --dataset_type BS3D -3d --icp_th 0.3 --com_th 0.1 
  # GSICPSLAM
  # python eval_recon_revised.py --rec_mesh /home/lyq/myprojects/GS_ICP_SLAM/experiments/results/BS3D_b/${sc}/mesh_cull_occlusion.ply --gt_mesh /media/lyq/data/dataset/BS3D/${sc}/mesh_cull_occlusion.ply --dataset_type BS3D -3d --icp_th 0.3 
  # RTG-SLAM
  # python eval_recon_revised.py --rec_mesh /home/lyq/myprojects/RTG-SLAM/output/dataset/BS3D_slam/${sc}/image/mesh_gap10_cull_occlusion.ply --gt_mesh /media/lyq/data/dataset/BS3D/${sc}/mesh_cull_occlusion.ply --dataset_type BS3D -3d --icp_th 0.3 --com_th 0.1
  # Co-SLAM
  # python eval_recon_revised.py --rec_mesh /home/lyq/myprojects/remix_files/remixfusion_results/results/BS3D_coslam/${sc}/${sc}_cull_occlusion.ply --gt_mesh /media/lyq/data/dataset/BS3D/${sc}/mesh_cull_occlusion.ply --dataset_type BS3D -3d --icp_th 0.3 --com_th 0.1 
  # Photo-SLAM
  # python eval_recon_revised.py --rec_mesh /home/lyq/myprojects/Photo-SLAM/results/bs3d/${sc}/mesh_gap10_cull_occlusion_transformed.ply --gt_mesh /media/lyq/data/dataset/BS3D/${sc}/mesh_cull_occlusion.ply --dataset_type BS3D -3d --icp_th 0.3 --com_th 0.1 
  # LoopSplat
  # python eval_recon_revised.py --rec_mesh /home/lyq/myprojects/LoopSplat/cleaned_mesh_cull_occlusion.ply --gt_mesh /media/lyq/data/dataset/BS3D/${sc}/mesh_cull_occlusion.ply --dataset_type BS3D -3d --icp_th 0.3 --com_th 0.1 
  # MIPSFusion
  # python eval_recon_revised.py --rec_mesh /home/lyq/myprojects/remix_files/remixfusion_results/results/BS3D_mips/${sc}/final_mesh_cull_occlusion.ply --gt_mesh /media/lyq/data/dataset/BS3D/${sc}/mesh_cull_occlusion.ply --dataset_type BS3D -3d --icp_th 0.3 --com_th 0.1 

  
  echo $sc done!
  echo -e "\n"
done
