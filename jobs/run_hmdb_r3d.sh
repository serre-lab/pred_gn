#!/bin/bash
#SBATCH --mem=25G
#SBATCH -n 20
#SBATCH -J R3D_SF
#SBATCH -p gpu 
#SBATCH -o R3D_18_HMDB.out
#SBATCH -e R3D_18_HMDB.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=carney-tserre-condo
# ## SBATCH -C titan

source activate sf

EXP_NAME="R3D_18_HMDB"

python tools/run_net.py \
  --cfg configs/HMDB/R3D_18.yaml \
  NUM_GPUS 4 \
  TRAIN.BATCH_SIZE 16 \
  OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME
