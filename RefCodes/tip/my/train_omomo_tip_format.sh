#!/bin/bash

# Training script for OMOMO data using TIP-format pipeline
# This script follows TIP's original training approach

TRAIN_DIRS="../../process/processed_data_OMOMO/train"
VAL_DIRS="../../process/processed_data_OMOMO/test"
SAVE_PATH="output/tip_omomo_format"

# Alternatively, use multiple datasets:
# TRAIN_DIRS="../../process/processed_data_IMHD_split/train ../../process/processed_data_BEHAVE_split/train ../../process/processed_data_OMOMO/train"
# VAL_DIRS="../../process/processed_data_IMHD_split/test ../../process/processed_data_BEHAVE_split/test ../../process/processed_data_OMOMO/test"

python train_tip_format.py \
    --train_dirs ${TRAIN_DIRS} \
    --val_dirs ${VAL_DIRS} \
    --save_path ${SAVE_PATH} \
    --batch_size 128 \
    --epochs 200 \
    --seq_len 60 \
    --lr 2e-4 \
    --weight_decay 1e-5 \
    --clip 5.0 \
    --optim AdamW \
    --rnn_nhid 512 \
    --tf_nhid 1024 \
    --tf_in_dim 256 \
    --n_heads 16 \
    --tf_layers 4 \
    --past_dropout 0.8 \
    --in_dropout 0.0 \
    --rnn_dropout 0.0 \
    --noise_input_hist 0.1 \
    --lambda_obj 1.0 \
    --fps 30.0 \
    --num_workers 8 \
    --patience 20 \
    --seed 42 \
    --cosine_lr \
    --cuda \
    --use_object_imu \
    # --with_acc_sum \
    # --warm_start output/tip_omomo_format/best.pt


