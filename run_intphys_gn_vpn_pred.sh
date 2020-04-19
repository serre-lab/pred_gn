#!/bin/bash
#SBATCH --mem=120G
#SBATCH -n 20
#SBATCH -J IP_GN_prednet_GNR_noW
#SBATCH -p gpu 
#SBATCH -o GN_IntPhys_prednet_GNR_noW_cbp.out
#SBATCH -e GN_IntPhys_prednet_GNR_noW_cbp.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:5
#SBATCH --account=carney-tserre-condo
#SBATCH -C quadrortx
# SBATCH -C v100 quadrortx


module load anaconda/3-5.2.0
source activate sf

# EXP_NAME="GN_IP_VPN_PRED_18ts_Sup_GN_W_k3_jit_L1"
EXP_NAME="GN_IntPhys_prednet_GNR_noW_cbp"

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
  NUM_GPUS 5 \
  SOLVER.BASE_LR 1e-3 \
  DATA.SAMPLING_RATE 2 \
  DATA.NUM_FRAMES 20 \
  TRAIN.BATCH_SIZE 50 \
  SOLVER.MAX_EPOCH 100 \
  PREDICTIVE.CPC False \
  DATA_LOADER.NUM_WORKERS 8 \
  MODEL.MODEL_NAME GN_PRED \
  GN.RECURRENT_BN GNR \
  GN.FEEDFORWARD_BN GNR \
  OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME

# MODEL.MODEL_NAME GN_PRED \

# python tools/run_net.py \
#   --cfg configs/IntPhys/GN_VPN.yaml \
#   DATA.COLOR_AUGMENTATION False \
#   NUM_GPUS 6 \
#   DATA.SAMPLING_RATE 2 \
#   DATA.NUM_FRAMES 18 \
#   TRAIN.BATCH_SIZE 16 \
#   SOLVER.MAX_EPOCH 100 \
#   PREDICTIVE.CPC False \
#   DATA_LOADER.NUM_WORKERS 8 \
#   GN.RECURRENT_BN GN \
#   GN.FEEDFORWARD_BN GN \
#   OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME
