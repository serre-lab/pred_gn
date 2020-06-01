#!/bin/bash
#SBATCH --mem=60G
#SBATCH -n 2
#SBATCH -J MM_PN_hGRU
#SBATCH -p gpu 
#SBATCH -o cur_logs/%x_%J.out
#SBATCH -e cur_logs/%x_%J.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=carney-tserre-condo
# SBATCH -C quadrortx
# SBATCH -C v100 

module load anaconda/3-5.2.0
source activate sf

EXP_NAME="Prednet_hGRU_intphys_nonorm"

MODEL_NAME="PredNet_hGRU"
BATCH_SIZE=20
N_FRAMES=20
NUM_GPUS=1

EVAL_PERIOD=5

port=$(shuf -i10005-10099 -n1)
dist_url="tcp://localhost:$port"

RECURRENT_BN="''" # "GN"

DEBUG=False

DATE=`date +%Y-%m-%d_%H-%M-%S`

OUT_DIR="/users/azerroug/data/azerroug/slowfast/outputs/PredNet/${DATE}_${EXP_NAME}"

python tools/run_net.py \
  --cfg configs/IntPhys/Prednet.yaml \
  --init_method $dist_url \
  DEBUG $DEBUG \
  NAME $EXP_NAME \
  NUM_GPUS $NUM_GPUS \
  DATA.NUM_FRAMES $N_FRAMES \
  TRAIN.BATCH_SIZE $BATCH_SIZE \
  TRAIN.EVAL_PERIOD $EVAL_PERIOD \
  TRAIN.CHECKPOINT_PERIOD 5 \
  SOLVER.MAX_EPOCH 100 \
  PREDICTIVE.CPC False \
  GN.RECURRENT_BN $RECURRENT_BN \
  MODEL.MODEL_NAME $MODEL_NAME \
  DATA_LOADER.NUM_WORKERS 8 \
  OUTPUT_DIR $OUT_DIR \


# OUT_DIR="/users/azerroug/data/azerroug/slowfast/outputs/PredNet/05-19-2020_15-48-32_Prednet_hGRU_intphys"
# NEP_ID='MOT-5'

# python tools/run_net.py \
#   --cfg configs/IntPhys/Prednet.yaml \
#   --init_method $dist_url \
#   DEBUG $DEBUG \
#   NEP_ID $NEP_ID \
#   NAME $EXP_NAME \
#   NUM_GPUS $NUM_GPUS \
#   DATA.NUM_FRAMES $N_FRAMES \
#   TRAIN.BATCH_SIZE $BATCH_SIZE \
#   SOLVER.MAX_EPOCH 200 \
#   PREDICTIVE.CPC False \
#   MODEL.MODEL_NAME $MODEL_NAME \
#   DATA_LOADER.NUM_WORKERS 8 \
#   OUTPUT_DIR $OUT_DIR \
