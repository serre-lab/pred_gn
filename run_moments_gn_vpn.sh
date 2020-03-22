#!/bin/bash
#SBATCH --mem=80G
#SBATCH -n 30
#SBATCH -J GN_VPN
#SBATCH -p gpu 
#SBATCH -o GN_R2D_Moments_VPN.out
#SBATCH -e GN_R2D_Moments_VPN.err
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=carney-tserre-condo
#SBATCH -C v100

source activate sf

EXP_NAME="GN_R2D_Moments_VPN"

python tools/run_net.py \
  --cfg configs/Moments/GN_R2D_18_PRED_VPN.yaml \
  DATA.COLOR_AUGMENTATION False \
  NUM_GPUS 4 \
  DATA.NUM_FRAMES 18 \
  DATA.NUM_REPEATED_SAMPLES 5 \
  TRAIN.BATCH_SIZE 8 \
  DATA_LOADER.NUM_WORKERS 8 \
  OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME



