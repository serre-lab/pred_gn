#!/bin/bash
#SBATCH --mem=20G
#SBATCH -n 2
#SBATCH -J SM_PN_C_E
#SBATCH -p gpu 
#SBATCH -o cur_logs/%x_%J.out
#SBATCH -e cur_logs/%x_%J.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0
#SBATCH --account=carney-tserre-condo
# SBATCH -C quadrortx
# SBATCH -C v100 quadrortx titanrtx

module load anaconda/3-5.2.0
source activate sf

#IDX=$SLURM_ARRAY_TASK_ID
# IDX
DATA=("one_bars_t"\
        "one_bars_r"\
        "one_bars_s"\
        "one_rnd_t"\
        "one_rnd_r"\
        "one_rnd_s"\
        )

DATA_NAME=${DATA[$IDX]}

EXP_NAME="SM_PN_fb_focal_best"

MODEL_NAME="SmallGN"
# MODEL_NAME="SmallPredNet" # PredNet PredNet_E
# RNN_CELL='ConvLSTMCell_C'
LAYERS=[6,16,32,64] 
# LAYERS=[12,24,48] #,96,96 
LOSSES="[['FocalLoss'],[1]]" #L1Loss FocalLoss CrossEntropy
# LOSSES="[['FocalLoss','CPC'],[1,5e-3]]" #L1Loss FocalLoss
EVALS="['mse','Acc','IoU']"

BATCH_SIZE=30
N_FRAMES=20
NUM_GPUS=1

EVAL_PERIOD=10
BASE_LR=5e-4

CHECKPOINT_PERIOD=10
SUMMARY_PERIOD=500

DEBUG=True

port=$(shuf -i10005-10099 -n1)
dist_url="tcp://localhost:$port"

DATE=`date +%Y-%m-%d_%H-%M-%S`

OUT_DIR="/users/azerroug/data/azerroug/slowfast/outputs/PredNet/${DATE}_${EXP_NAME}"

#run_s_net
python tools/debug.py \
  --cfg configs/SimpleMotion/Prednet.yaml \
  --init_method $dist_url \
  DEBUG $DEBUG \
  NAME $EXP_NAME \
  NUM_GPUS $NUM_GPUS \
  DATA.NUM_FRAMES $N_FRAMES \
  DATA.NAME $DATA_NAME \
  TRAIN.ENABLE True \
  TRAIN.BATCH_SIZE $BATCH_SIZE \
  TRAIN.EVAL_PERIOD $EVAL_PERIOD \
  TRAIN.CHECKPOINT_PERIOD $CHECKPOINT_PERIOD \
  SOLVER.MAX_EPOCH 150 \
  SOLVER.BASE_LR $BASE_LR \
  PREDICTIVE.CPC True \
  PREDNET.CELL $RNN_CELL \
  PREDNET.LAYERS $LAYERS \
  PREDNET.LOSSES $LOSSES \
  PREDNET.EVALS $EVALS \
  SUMMARY_PERIOD $SUMMARY_PERIOD \
  MODEL.MODEL_NAME $MODEL_NAME \
  DATA_LOADER.NUM_WORKERS 8 \
  OUTPUT_DIR $OUT_DIR \
