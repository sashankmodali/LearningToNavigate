#!/bin/bash

variable=$(conda info | grep -i 'base environment' | awk '{ print $4 }')
source ${variable}/etc/profile.d/conda.sh
conda activate habitat
python nslam.py --split val --eval 1 --train_global 0 --train_local 0 --train_slam 0 --auto_gpu_config 0  --load_global pretrained_models/model_best.global --load_local pretrained_models/model_best.local --load_slam pretrained_models/model_best.slam -n 1 --print_images 1 --task "point_nav"
python generate_video.py