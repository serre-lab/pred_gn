#!/bin/bash
#SBATCH --mem=40G
#SBATCH -n 20
#SBATCH -J GN_CPC
#SBATCH -p gpu 
#SBATCH -o GN_R2D_Moments_CPC.out
#SBATCH -e GN_R2D_Moments_CPC.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=carney-tserre-condo
#SBATCH -C quadrortx

source activate sf

EXP_NAME="GN_R2D_Moments_CPC"

python tools/run_net.py \
  --cfg configs/Moments/GN_R2D_18_PRED_CPC.yaml \
  DATA.COLOR_AUGMENTATION False \
  NUM_GPUS 2 \
  DATA.NUM_FRAMES 18 \
  DATA.NUM_REPEATED_SAMPLES 2 \
  TRAIN.BATCH_SIZE 2 \
  DATA_LOADER.NUM_WORKERS 8 \
  OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME

