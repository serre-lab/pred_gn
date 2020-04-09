#!/bin/bash
#SBATCH --mem=120G
#SBATCH -n 30
#SBATCH -J MM_PN_basic
#SBATCH -p gpu 
#SBATCH -o Prednet_MM_basic.out
#SBATCH -e Prednet_MM_basic.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=carney-tserre-condo
#SBATCH -C v100
# SBATCH -C v100 quadrortx

module load anaconda/3-5.2.0
source activate sf

# EXP_NAME="GN_IP_VPN_PRED_18ts_Sup_GN_W_k3_jit_L1"
EXP_NAME="Prednet_intphys_basic_01"

python tools/run_net.py \
  --cfg configs/IntPhys/Prednet.yaml \
  NUM_GPUS 2 \
  DATA.NUM_FRAMES 20 \
  TRAIN.BATCH_SIZE 100 \
  SOLVER.MAX_EPOCH 100 \
  PREDICTIVE.CPC False \
  DATA_LOADER.NUM_WORKERS 8 \
  OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME

# python tools/visualize_dataset.py \
#   --cfg configs/IntPhys/Prednet.yaml \
#   NUM_GPUS 2 \
#   DATA.NUM_FRAMES 20 \
#   TRAIN.BATCH_SIZE 40 \
#   SOLVER.MAX_EPOCH 100 \
#   PREDICTIVE.CPC False \
#   DATA_LOADER.NUM_WORKERS 8 \
#   OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME


