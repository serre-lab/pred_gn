#!/bin/bash
#SBATCH --mem=30G
#SBATCH -n 20
#SBATCH -J IntPhys_GN_VPN
#SBATCH -p gpu 
#SBATCH -o GN_R2D_intphys_VPN.out
#SBATCH -e GN_R2D_intphys_VPN.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=carney-tserre-condo
# SBATCH -C quadrortx
# SBATCH -C v100

module load anaconda/3-5.2.0
source activate sf

EXP_NAME="GN_R2D_IntPhys_VPN_Simple_test"

python tools/run_net.py \
  --cfg configs/IntPhys/GN_VPN.yaml \
  DATA.COLOR_AUGMENTATION False \
  NUM_GPUS 2 \
  DATA.SAMPLING_RATE 1 \
  DATA.NUM_FRAMES 24 \
  TRAIN.BATCH_SIZE 4 \
  PREDICTIVE.CPC False \
  DATA_LOADER.NUM_WORKERS 8 \
  OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME

# python tools/visualize_dataset.py \
#   --cfg configs/IntPhys/GN_VPN.yaml \
#   DATA.COLOR_AUGMENTATION True \
#   NUM_GPUS 1 \
#   NUM_GPUS 1 \
#   DATA.NUM_FRAMES 18 \
#   DATA.NUM_REPEATED_SAMPLES 2 \
#   TRAIN.BATCH_SIZE 2 \
#   DATA_LOADER.NUM_WORKERS 8 \
#   OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME