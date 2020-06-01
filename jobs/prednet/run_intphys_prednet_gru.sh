#!/bin/bash
#SBATCH --mem=60G
#SBATCH -n 3
#SBATCH -J MM_PN_GRU
#SBATCH -p gpu 
#SBATCH -o cur_logs/%x_%J.out
#SBATCH -e cur_logs/%x_%J.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=carney-tserre-condo
# SBATCH -C quadrortx
# SBATCH -C v100 quadrortx titanrtx

module load anaconda/3-5.2.0
source activate sf

DEBUG=False

EXP_NAME="Prednet_GRU_intphys" #_lesion_out

MODEL_NAME="PredNet_GRU"
RNN_CELL="GRU" # GRU_in GRU_out GRU_rnn
BATCH_SIZE=16
N_FRAMES=20
NUM_GPUS=1

EVAL_PERIOD=5
BASE_LR=0.0005

########################################################
port=$(shuf -i10005-10099 -n1)
dist_url="tcp://localhost:$port"

DATE=`date +%Y-%m-%d_%H-%M-%S`
OUT_DIR="/users/azerroug/data/azerroug/slowfast/outputs/PredNet/${DATE}_${EXP_NAME}"

# OUT_DIR="/users/azerroug/data/azerroug/slowfast/outputs/PredNet/2020-05-22_09-56-16_Prednet_GRU_intphys"
# NEP_ID='MOT-33'
  # NEP_ID $NEP_ID \

########################################################
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
  SOLVER.MAX_EPOCH 150 \
  SOLVER.BASE_LR $BASE_LR \
  PREDICTIVE.CPC False \
  MODEL.MODEL_NAME $MODEL_NAME \
  PREDNET.CELL $RNN_CELL \
  DATA_LOADER.NUM_WORKERS 8 \
  OUTPUT_DIR $OUT_DIR \


# OUT_DIR="/users/azerroug/data/azerroug/slowfast/outputs/PredNet/05-19-2020_15-48-32_Prednet_hGRU_intphys"
# NEP_ID='MOT-5'
