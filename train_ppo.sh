#!/bin/bash

variable=$(conda info | grep -i 'base environment' | awk '{ print $4 }')
source ${variable}/etc/profile.d/conda.sh
conda activate habitat

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

python train_ppo.py --log-file log_file.p --checkpoint-folder ./checkpoints --sim-gpu-id 0 --pth-gpu-id 0 --num-processes 2 --num-mini-batch 2 --checkpoint-interval 1000 --num-updates 1000000 --load-train 68000