#!/bin/bash

variable=$(conda info | grep -i 'base environment' | awk '{ print $4 }')
source ${variable}/etc/profile.d/conda.sh
conda activate habitat

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export GLOG_minloglevel=2

cd habitat-lab

python -u habitat_baselines/run.py --exp-config habitat_baselines/config/pointnav/ppo_pointnav_cs593.yaml --run-type train

cd ../
