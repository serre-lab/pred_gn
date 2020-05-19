#!/bin/bash
#SBATCH --mem=120G
#SBATCH -n 30
#SBATCH -J GN_CPC
#SBATCH -p gpu 
#SBATCH -o GN_R2D_Moments_CPC.out
#SBATCH -e GN_R2D_Moments_CPC.err
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:10
#SBATCH --account=carney-tserre-condo
#SBATCH -C quadrortx

source activate sf

EXP_NAME="GN_R2D_Moments_CPC"

python tools/run_net.py \
  --cfg configs/Moments/GN_R2D_18_PRED_CPC.yaml \
  DATA.COLOR_AUGMENTATION False \
  NUM_GPUS 10 \
  DATA.NUM_FRAMES 18 \
  DATA.NUM_REPEATED_SAMPLES 3 \
  TRAIN.BATCH_SIZE 20 \
  DATA_LOADER.NUM_WORKERS 8 \
  OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME

