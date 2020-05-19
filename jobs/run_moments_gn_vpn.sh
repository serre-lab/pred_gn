#!/bin/bash
#SBATCH --mem=120G
#SBATCH -n 30
#SBATCH -J GN_VPN_S
#SBATCH -p gpu 
#SBATCH -o GN_R2D_Moments_VPN_S.out
#SBATCH -e GN_R2D_Moments_VPN_S.err
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:10
#SBATCH --account=carney-tserre-condo
#SBATCH -C quadrortx
# SBATCH -C v100

module load anaconda/3-5.2.0
source activate sf

# EXP_NAME="GN_R2D_Moments_VPN_E"

# python tools/run_net.py \
#   --cfg configs/Moments/GN_R2D_18_PRED_VPN.yaml \
#   DATA.COLOR_AUGMENTATION False \
#   NUM_GPUS 6 \
#   DATA.NUM_FRAMES 18 \
#   DATA.NUM_REPEATED_SAMPLES 2 \
#   TRAIN.BATCH_SIZE 24 \
#   DATA_LOADER.NUM_WORKERS 8 \
#   OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME

EXP_NAME="GN_R2D_Moments_VPN_Simple"

python tools/run_net.py \
  --cfg configs/Moments/GN_R2D_18_PRED_VPN.yaml \
  DATA.COLOR_AUGMENTATION False \
  NUM_GPUS 10 \
  MODEL.MODEL_NAME GN_VPN \
  DATA.NUM_FRAMES 18 \
  DATA.NUM_REPEATED_SAMPLES 2 \
  TRAIN.BATCH_SIZE 18 \
  DATA_LOADER.NUM_WORKERS 8 \
  OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME

# MODEL.MODEL_NAME GN_VPN \