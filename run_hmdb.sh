#!/bin/bash
#SBATCH --mem=25G
#SBATCH -n 20
#SBATCH -J slow_fast_C2D
#SBATCH -p gpu 
#SBATCH -o C2D_8x8_R50_HMDB.out
#SBATCH -e C2D_8x8_R50_HMDB.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=carney-tserre-condo
# ## SBATCH -C titan

source activate sf

EXP_NAME="C2D_8x8_R50_HMDB"

python tools/run_net.py \
  --cfg configs/HMDB/C2D_8x8_R50.yaml \
  NUM_GPUS 4 \
  TRAIN.BATCH_SIZE 16 \
  OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/C2D_8x8_R50_HMDB

# nohup \
# python tools/run_net.py \
#   --cfg configs/HMDB/C2D_8x8_R50.yaml \
#   NUM_GPUS 4 \
#   TRAIN.BATCH_SIZE 32 \
#   > $EXP_NAME.log 2>&1 &