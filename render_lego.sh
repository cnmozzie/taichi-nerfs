#!/bin/bash

set -euo pipefail

export DATA_DIR=./Synthetic_NeRF

python3 render.py \
    --root_dir $DATA_DIR/Lego \
    --exp_name Lego --perf \
    --val_only --ckpt_path ckpts/nsvf/Lego/epoch=19-v5.ckpt \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 #--no_save_test --gui \
    #--half2_opt \
    
