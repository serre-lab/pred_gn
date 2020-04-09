#!/bin/bash
#SBATCH --mem=120G
#SBATCH -n 30
#SBATCH -J MM_GN_basic_01
#SBATCH -p gpu 
#SBATCH -o GN_MM_basic_TDR_01.out
#SBATCH -e GN_MM_basic_TDR_01.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:6
#SBATCH --account=carney-tserre-condo
#SBATCH -C quadrortx
# SBATCH -C v100 quadrortx

module load anaconda/3-5.2.0
source activate sf

# EXP_NAME="GN_IP_VPN_PRED_18ts_Sup_GN_W_k3_jit_L1"
EXP_NAME="GN_mmnist_basic_TDR_01"

python tools/run_net.py \
  --cfg configs/Mmnist/GN_VPN.yaml \
  NUM_GPUS 6 \
  DATA.NUM_FRAMES 20 \
  TRAIN.BATCH_SIZE 240 \
  SOLVER.MAX_EPOCH 100 \
  PREDICTIVE.CPC False \
  DATA_LOADER.NUM_WORKERS 8 \
  MODEL.MODEL_NAME GN_PRED \
  GN.RECURRENT_BN GN \
  GN.FEEDFORWARD_BN GN \
  OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME


