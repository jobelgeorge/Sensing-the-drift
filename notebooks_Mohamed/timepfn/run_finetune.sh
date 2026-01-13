#!/bin/bash

# Define paths
PROJECT_ROOT=$(pwd)
TIMEPFN_DIR="$PROJECT_ROOT/TimePFN"
DATA_ROOT="$PROJECT_ROOT/datasets_challenge/"
DATA_FILE="train_merged_finetune.csv"
CHECKPOINT_PATH="$TIMEPFN_DIR/checkpoints/TimePFN/checkpoint.pth"

# 1. Move into TimePFN directory so imports work correctly
cd "$TIMEPFN_DIR" || exit

# 2. Run Fine-Tuning
python -u run.py \
  --is_training 1 \
  --root_path "$DATA_ROOT" \
  --data_path "$DATA_FILE" \
  --model_id sensing_drift_ft \
  --model TimePFN \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 8 \
  --d_layers 1 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --d_model 1024 \
  --d_ff 512 \
  --n_heads 8 \
  --des 'Exp' \
  --itr 1 \
  --patch_size 16 \
  --embed_dim 256 \
  --train_epochs 5 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --load 1 \
  --load_path "$CHECKPOINT_PATH" \
  --max_channel 161 \
  --target sideSlip