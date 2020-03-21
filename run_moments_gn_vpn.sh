#!/bin/bash
#SBATCH --mem=40G
#SBATCH -n 20
#SBATCH -J GN_R2D
#SBATCH -p gpu 
#SBATCH -o GN_R2D_Moments_VPN_viz.out
#SBATCH -e GN_R2D_Moments_VPN_viz.err
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=carney-tserre-condo
# SBATCH -C v100

source activate sf

# EXP_NAME="GN_R2D_Moments_VPN"

# python tools/run_net.py \
#   --cfg configs/Moments/GN_R2D_18_PRED_VPN.yaml \
#   DATA.COLOR_AUGMENTATION True \
#   NUM_GPUS 4 \
#   DATA.NUM_FRAMES 18 \
#   DATA.NUM_REPEATED_SAMPLES 2 \
#   TRAIN.BATCH_SIZE 16 \
#   DATA_LOADER.NUM_WORKERS 8 \
#   OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME

python tools/visualize_dataset.py \
  --cfg configs/Moments/GN_R2D_18_PRED_VPN.yaml \
  DATA.COLOR_AUGMENTATION True \
  NUM_GPUS 1 \
  DATA.NUM_FRAMES 18 \
  DATA.NUM_REPEATED_SAMPLES 4 \
  TRAIN.BATCH_SIZE 2 \
  DATA_LOADER.NUM_WORKERS 8 \
  OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME

