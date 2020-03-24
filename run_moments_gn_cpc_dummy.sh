#!/bin/bash
#SBATCH --mem=120G
#SBATCH -n 30
#SBATCH -J GN_VPN
#SBATCH -p gpu 
#SBATCH -o GN_R2D_Moments_VPN_2.out
#SBATCH -e GN_R2D_Moments_VPN_2.err
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:10
#SBATCH --account=carney-tserre-condo
#SBATCH -C quadrortx

source activate sf

EXP_NAME="GN_R2D_Moments_CPC_Dist"

#tcp://172.20.216.5:1234
NCCL_DEBUG="INFO" \
python tools/run_net.py \
  --cfg configs/Moments/GN_R2D_18_PRED_VPN.yaml \
  --init_method tcp://172.20.216.5:1234 \
  --num_shards 3 \
  --shard_id 2 \
  DATA.COLOR_AUGMENTATION False \
  NUM_GPUS 10 \
  DATA.NUM_FRAMES 18 \
  DATA.NUM_REPEATED_SAMPLES 3 \
  TRAIN.BATCH_SIZE 20 \
  DATA_LOADER.NUM_WORKERS 8 \
  SUMMARY_PERIOD 1000 \
  OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME


# 172.20.216.5
# 172.25.216.5

# 172.25.210.1

# 172.20.210.5
# 172.25.210.5


# 172.25.203.18
# 172.20.203.18