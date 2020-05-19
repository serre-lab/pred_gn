#!/bin/bash
#SBATCH --mem=60G
#SBATCH -n 20
#SBATCH -J GN_R3D
#SBATCH -p gpu 
#SBATCH -o R3D_Moments.out
#SBATCH -e R3D_Moments.err
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=carney-tserre-condo
# ## SBATCH -C titan

source activate sf

EXP_NAME="R3D_Moments"

python tools/run_net.py \
  --cfg configs/Moments/R3D_18.yaml \
  NUM_GPUS 2 \
  DATA.NUM_FRAMES 16 \
  TRAIN.BATCH_SIZE 64 \
  DATA_LOADER.NUM_WORKERS 8 \
  OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME
