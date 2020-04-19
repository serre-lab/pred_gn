#!/bin/bash
#SBATCH --mem=120G
#SBATCH -n 20
#SBATCH -J IP_GN
#SBATCH -p gpu 
#SBATCH -o GN_IntPhys.out
#SBATCH -e GN_IntPhys.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:5
#SBATCH --account=carney-tserre-condo
#SBATCH -C quadrortx

# v100 quadrortx

module load anaconda/3-5.2.0
source activate sf

EXP_NAME="GN_IntPhys"

# all config options are in configs/{dataset}/{cfg}.yaml
CONFIG="configs/IntPhys/GN_VPN.yaml"

NUM_GPUS=2

# number of training video frames, the total number of frames is 100 for IntPhys 
NUM_FRAMES=20
SAMPLING_RATE=2

BASE_LR=1e-3 
BATCH_SIZE=20
MAX_EPOCH=100

# Model to train
MODEL_NAME="GN_PRED"
# GN_PRED is the closest to prednet (top-down then bottomup)
# GN_VPN is the closest to prednet (bottomup then top-down)

# enable training with CPC
CPC=False

# GNR is groupnorm with running statistics through the video (decay=0.5) other options are in slowfast/batchnorm.py
RECURRENT_BN="GNR" #GN
# relevent if there are normalization in feedforward layers (including the prediction layer)
FEEDFORWARD_BN="GNR" #GN

python tools/run_net.py \
  --cfg $CONFIG \
  NUM_GPUS $NUM_GPUS \
  SOLVER.BASE_LR $BASE_LR \
  DATA.SAMPLING_RATE $SAMPLING_RATE \
  DATA.NUM_FRAMES $NUM_FRAMES \
  TRAIN.BATCH_SIZE $BATCH_SIZE \
  SOLVER.MAX_EPOCH $MAX_EPOCH \
  PREDICTIVE.CPC $CPC \
  MODEL.MODEL_NAME $MODEL_NAME \
  GN.RECURRENT_BN $RECURRENT_BN \
  GN.FEEDFORWARD_BN $FEEDFORWARD_BN \
  OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME
