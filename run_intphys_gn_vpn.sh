#!/bin/bash
#SBATCH --mem=120G
#SBATCH -n 30
#SBATCH -J IP_GN_VPN_18ts
#SBATCH -p gpu 
#SBATCH -o GN_IP_VPN_BU_18ts_L1.out
#SBATCH -e GN_IP_VPN_BU_18ts_L1.err
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:10
#SBATCH --account=carney-tserre-condo
#SBATCH -C quadrortx
# SBATCH -C v100

module load anaconda/3-5.2.0
source activate sf

EXP_NAME="GN_IntPhys_VPN_BU_18ts_L1"

# python tools/run_net.py \
#   --cfg configs/IntPhys/GN_VPN.yaml \
#   DATA.COLOR_AUGMENTATION False \
#   NUM_GPUS 10 \
#   DATA.SAMPLING_RATE 2 \
#   DATA.NUM_FRAMES 18 \
#   TRAIN.BATCH_SIZE 30 \
#   SOLVER.MAX_EPOCH 50 \
#   PREDICTIVE.CPC False \
#   DATA_LOADER.NUM_WORKERS 8 \
#   OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME

python tools/run_net.py \
  --cfg configs/IntPhys/GN_VPN.yaml \
  DATA.COLOR_AUGMENTATION False \
  NUM_GPUS 10 \
  DATA.SAMPLING_RATE 2 \
  DATA.NUM_FRAMES 18 \
  TRAIN.BATCH_SIZE 20 \
  SOLVER.MAX_EPOCH 100 \
  PREDICTIVE.CPC True \
  DATA_LOADER.NUM_WORKERS 8 \
  OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME
