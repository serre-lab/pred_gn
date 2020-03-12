#!/bin/bash
#SBATCH --mem=80G
#SBATCH -n 20
#SBATCH -J GN_R2D
#SBATCH -p gpu 
#SBATCH -o GN_R2D_Moments_CPC.out
#SBATCH -e GN_R2D_Moments_CPC.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:8
#SBATCH --account=carney-tserre-condo
#SBATCH -C quadrortx

source activate sf

EXP_NAME="GN_R2D_Moments_CPC"

python tools/run_net.py \
  --cfg configs/Moments/GN_R2D_18_PRED_CPC.yaml \
  NUM_GPUS 8 \
  DATA.NUM_FRAMES 18 \
  TRAIN.BATCH_SIZE 48 \
  DATA_LOADER.NUM_WORKERS 8 \
  OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME

