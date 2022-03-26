#!/bin/bash

variable=$(conda info | grep -i 'base environment' | awk '{ print $4 }')
source ${variable}/etc/profile.d/conda.sh
conda activate habitat
python main.py --task milestone1 --print_images 1