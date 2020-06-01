#!/bin/bash
#SBATCH --mem=40G
#SBATCH -n 6
#SBATCH -J MM_PN_C_E
#SBATCH -p gpu 
#SBATCH -o cur_logs/%x_%J.out
#SBATCH -e cur_logs/%x_%J.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=carney-tserre-condo
# SBATCH -C quadrortx
# SBATCH -C v100 quadrortx titanrtx

module load anaconda/3-5.2.0
source activate sf


EXP_NAME="PredNet_Intphys_C_E_AUG"

MODEL_NAME="PredNet_E" # PredNet PredNet_E
RNN_CELL='ConvLSTMCell_C' # ConvLSTMCell_CGpool ConvLSTMCell_CG1x1
# RNN_CELL='ConvLSTMCell_CG1x1_noF'
# RNN_CELL='ConvLSTMCell_CG1x1_noO'

BATCH_SIZE=30
N_FRAMES=20
NUM_GPUS=2

EVAL_PERIOD=5
BASE_LR=1e-3

DEBUG=False

port=$(shuf -i10005-10099 -n1)
dist_url="tcp://localhost:$port"

DATE=`date +%Y-%m-%d_%H-%M-%S`

OUT_DIR="/users/azerroug/data/azerroug/slowfast/outputs/PredNet/${DATE}_${EXP_NAME}"


# OUT_DIR="/users/azerroug/data/azerroug/slowfast/outputs/PredNet/2020-05-22_22-30-50_Prednet_intphys_CG1x1"
# OUT_DIR="/users/azerroug/data/azerroug/slowfast/outputs/PredNet/2020-05-22_22-30-21_Prednet_intphys_CGpool"


# NEP_ID='MOT-45'
# NEP_ID $NEP_ID \
python tools/run_net.py \
  --cfg configs/IntPhys/Prednet.yaml \
  --init_method $dist_url \
  DEBUG $DEBUG \
  NAME $EXP_NAME \
  NUM_GPUS $NUM_GPUS \
  DATA.NUM_FRAMES $N_FRAMES \
  TRAIN.ENABLE True \
  TRAIN.BATCH_SIZE $BATCH_SIZE \
  TRAIN.EVAL_PERIOD $EVAL_PERIOD \
  TRAIN.CHECKPOINT_PERIOD 5 \
  SOLVER.MAX_EPOCH 150 \
  SOLVER.BASE_LR $BASE_LR \
  PREDICTIVE.CPC False \
  PREDNET.CELL $RNN_CELL \
  MODEL.MODEL_NAME $MODEL_NAME \
  DATA_LOADER.NUM_WORKERS 8 \
  OUTPUT_DIR $OUT_DIR \
