#!/bin/bash

 scenes="cafeteria corridor foobar hub juice lounge study waiting"

echo "Start running on BS3D dataset..."

for sc in ${scenes}
do
  echo "running on ${sc}..."
  python -u ../rendering_eval.py ../configs/BS3D/${sc}.yaml --ckpt_path ../output/BS3D/${sc}/test/checkpoint.pt  --pose_path ../output/BS3D/${sc}/test/all_poses.npy  > log/rendering_evaluation_${sc}.log
  echo $sc done!
done
