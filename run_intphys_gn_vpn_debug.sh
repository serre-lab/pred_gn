#!/bin/bash
#SBATCH --mem=30G
#SBATCH -n 20
#SBATCH -J IntPhys_GN_VPN
#SBATCH -p gpu 
#SBATCH -o GN_IntPhys_test.out
#SBATCH -e GN_IntPhys_test.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=carney-tserre-condo
#SBATCH -C quadrortx
# SBATCH -C v100

module load anaconda/3-5.2.0
source activate sf

EXP_NAME="GN_IntPhys_test"

# python tools/run_net.py \
#   --cfg configs/IntPhys/GN_VPN.yaml \
#   DATA.COLOR_AUGMENTATION False \
#   NUM_GPUS 2 \
#   DATA.SAMPLING_RATE 1 \
#   DATA.NUM_FRAMES 24 \
#   TRAIN.BATCH_SIZE 4 \
#   PREDICTIVE.CPC False \
#   DATA_LOADER.NUM_WORKERS 8 \
#   OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME

# python tools/visualize_dataset.py \
#   --cfg configs/IntPhys/GN_VPN_SEG.yaml \
#   DATA.COLOR_AUGMENTATION False \
#   NUM_GPUS 1 \
#   DATA.NUM_FRAMES 18 \
#   DATA.NUM_REPEATED_SAMPLES 2 \
#   TRAIN.BATCH_SIZE 2 \
#   DATA_LOADER.NUM_WORKERS 8 \
#   PREDICTIVE.CPC False \
#   MODEL.MODEL_NAME GN_PRED \
#   GN.RECURRENT_BN GN \
#   GN.FEEDFORWARD_BN GN \
#   TRAIN.CHECKPOINT_FILE_PATH /users/azerroug/data/azerroug/slowfast/outputs/GN_IntPhys_prednet_GN_18ts_L1_2/checkpoints/checkpoint_epoch_00044.pyth \
#   OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME

## GN_IntPhys_prednet_GN_18ts_L1_2/checkpoints/checkpoint_epoch_0004.pyth
## GN_IntPhys_VPN_PRED_18ts_Sup_GN_W_k3_L1/checkpoints/checkpoint_epoch_00035.pyth


python tools/run_net.py \
  --cfg configs/IntPhys/GN_VPN.yaml \
  DATA.COLOR_AUGMENTATION False \
  NUM_GPUS 2 \
  DATA.SAMPLING_RATE 2 \
  DATA.NUM_FRAMES 20 \
  TRAIN.BATCH_SIZE 4 \
  SOLVER.MAX_EPOCH 100 \
  PREDICTIVE.CPC False \
  DATA_LOADER.NUM_WORKERS 8 \
  MODEL.MODEL_NAME GN_PRED \
  GN.RECURRENT_BN GN \
  GN.FEEDFORWARD_BN GN \
  OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME