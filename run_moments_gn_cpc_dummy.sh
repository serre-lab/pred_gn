#!/bin/bash
#SBATCH --mem=80G
#SBATCH -n 25
#SBATCH -J GN_R2D_DIST
#SBATCH -p gpu 
#SBATCH -o GN_R2D_Moments_CPC_dummy_1.out
#SBATCH -e GN_R2D_Moments_CPC_dummy_1.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=carney-tserre-condo
#SBATCH -C v100

source activate sf

EXP_NAME="GN_R2D_Moments_CPC_Dist"

#tcp://172.20.216.5:1234
NCCL_DEBUG="INFO" \
python tools/run_net.py \
  --cfg configs/Moments/GN_R2D_18_PRED_CPC.yaml \
  --init_method tcp://172.25.210.4:1234 \
  --num_shards 2 \
  --shard_id 1 \
  DATA.NUM_FRAMES 16 \
  TRAIN.BATCH_SIZE 44 \
  DATA_LOADER.NUM_WORKERS 8 \
  NUM_GPUS 4 \
  OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME


# 172.25.210.1

# 172.20.210.5
# 172.25.210.5


# 172.25.203.18
# 172.20.203.18