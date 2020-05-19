#!/bin/bash
#SBATCH --mem=120G
#SBATCH -n 30
#SBATCH -J GN_VPN_S
#SBATCH -p gpu 
#SBATCH -o GN_R2D_Moments_VPN_S_1.out
#SBATCH -e GN_R2D_Moments_VPN_S_1.err
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:10
#SBATCH --account=carney-tserre-condo
#SBATCH -C quadrortx

module load anaconda/3-5.2.0
source activate sf

EXP_NAME="GN_R2D_Moments_VPN_S_Dist"

#tcp://172.20.216.5:1234
NCCL_DEBUG="INFO" \
python tools/run_net.py \
  --cfg configs/Moments/GN_R2D_18_PRED_VPN.yaml \
  --init_method tcp://172.20.216.5:1234 \
  --num_shards 2 \
  --shard_id 1 \
  DATA.COLOR_AUGMENTATION False \
  MODEL.MODEL_NAME GN_VPN \
  NUM_GPUS 10 \
  DATA.NUM_FRAMES 18 \
  DATA.NUM_REPEATED_SAMPLES 2 \
  TRAIN.BATCH_SIZE 18 \
  DATA_LOADER.NUM_WORKERS 8 \
  SUMMARY_PERIOD 1000 \
  OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME

# 172.20.216.7

# 172.20.216.5
# 172.25.216.5

# 172.25.210.1

# 172.20.210.5
# 172.25.210.5


# 172.25.203.18
# 172.20.203.18