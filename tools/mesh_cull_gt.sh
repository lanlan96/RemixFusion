#!/bin/bash

scenes="cafeteria corridor foobar hub juice lounge study waiting" 

echo "Start running on BS3D dataset..."

for sc in ${scenes}
do
  echo "running on ${sc}..."
  python cull_mesh.py --config ../configs/BS3D/${sc}.yaml --input_mesh  /media/lyq/data/dataset/BS3D/${sc}/mesh.ply --gt_pose --remove_occlusion --gt_depth --eps 0.1 --skip 5 
  echo $sc done!
  sleep 5
done

