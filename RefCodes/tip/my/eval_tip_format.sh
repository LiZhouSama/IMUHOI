#!/bin/bash

# Evaluation script for TIP-format trained models

# Path to trained model checkpoint
WEIGHTS="checkpoints/tip_omomo_format/best.pt"

# Evaluation data directory
DATA_DIRS="../../process/processed_data_OMOMO/test"

# Path to SMPLH body model (for FK and position metrics)
SMPLH_PATH="../../smpl_models/smplh/male/model.npz"

# Run evaluation
python eval_tip_format.py \
    --data_dirs ${DATA_DIRS} \
    --weights ${WEIGHTS} \
    --smplh_path ${SMPLH_PATH} \
    --seq_len 60 \
    --fps 30.0 \
    --root_supervision vel \
    --imu_noise_std 0.0 \
    --rnn_nhid 512 \
    --tf_nhid 1024 \
    --tf_in_dim 256 \
    --n_heads 16 \
    --tf_layers 4 \
    --past_dropout 0.8 \
    --use_object_imu \
    --eval_contacts \
    # --with_acc_sum  # Uncomment if model was trained with acc_sum

# For multiple datasets:
# DATA_DIRS="../../process/processed_data_IMHD_split/test ../../process/processed_data_BEHAVE_split/test ../../process/processed_data_OMOMO/test"

# For noisy evaluation (testing robustness):
# --imu_noise_std 0.1


