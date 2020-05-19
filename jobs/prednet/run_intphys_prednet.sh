#!/bin/bash
#SBATCH --mem=60G
#SBATCH -n 4
#SBATCH -J MM_PN_basic
#SBATCH -p gpu 
#SBATCH -o cur_logs/%x_%J.out
#SBATCH -e cur_logs/%x_%J.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=carney-tserre-condo
#SBATCH -C quadrortx
# SBATCH -C v100 quadrortx

module load anaconda/3-5.2.0
source activate sf


EXP_NAME="Prednet_intphys"

MODEL_NAME="PredNet"
BATCH_SIZE=20
N_FRAMES=20
NUM_GPUS=2

DEBUG=False

port=$(shuf -i10005-10099 -n1)
dist_url="tcp://localhost:$port"

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
  SOLVER.MAX_EPOCH 100 \
  PREDICTIVE.CPC False \
  MODEL.MODEL_NAME $MODEL_NAME \
  DATA_LOADER.NUM_WORKERS 8 \
  OUTPUT_DIR $OUT_DIR \


# python tools/visualize_dataset.py \
#   --cfg configs/IntPhys/Prednet.yaml \
#   NUM_GPUS 2 \
#   DATA.NUM_FRAMES 20 \
#   TRAIN.BATCH_SIZE 40 \
#   SOLVER.MAX_EPOCH 100 \
#   PREDICTIVE.CPC False \
#   DATA_LOADER.NUM_WORKERS 8 \
#   OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME


