#!/bin/bash
# =============================================================================
# scripts/rl_grpo/train_rl_tgs.sh - Single-GPU / Multi-GPU Launch Script
# =============================================================================

export PYTHONPATH=$PYTHONPATH:$(pwd)
export TOKENIZERS_PARALLELISM=false

# Default paths
CKPT_DIR="./results_real/epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05/checkpoint-551"
MODEL_PATH="./pretrained_weights/Llama-2-7b-chat-hf"
OUTPUT_DIR="./work_dirs_RL/tgs_grpo"

# Hyperparameters
MAX_STEPS=1000
NUM_GEN=4
LR=3e-6
BETA=0.04
DEVICE="cuda:0"

# Execute single-GPU GRPO trainer
python scripts/rl_grpo/train_rl_tgs.py \
    --ckpt_dir "${CKPT_DIR}" \
    --model_name_or_path "${MODEL_PATH}" \
    --out_dir "${OUTPUT_DIR}" \
    --max_steps "${MAX_STEPS}" \
    --num_generations "${NUM_GEN}" \
    --learning_rate "${LR}" \
    --beta "${BETA}" \
    --device "${DEVICE}"
