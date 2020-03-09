#!/bin/bash
#SBATCH --mem=40G
#SBATCH -n 20
#SBATCH -J GN_R2D
#SBATCH -p gpu 
#SBATCH -o GN_R2D_Moments.out
#SBATCH -e GN_R2D_Moments.err
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=carney-tserre-condo
# ## SBATCH -C titan

source activate sf

EXP_NAME="GN_R2D_Moments_2"

python tools/run_net.py \
  --cfg configs/Moments/GN_R2D_18.yaml \
  NUM_GPUS 2 \
  DATA.NUM_FRAMES 32 \
  TRAIN.BATCH_SIZE 32 \
  DATA_LOADER.NUM_WORKERS 8 \
  OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME
