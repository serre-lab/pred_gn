#!/bin/bash
#SBATCH --mem=30G
#SBATCH -n 20
#SBATCH -J GN_VPN_2
#SBATCH -p gpu 
#SBATCH -o GN_R2D_Moments_VPN_test_viz.out
#SBATCH -e GN_R2D_Moments_VPN_test_viz.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=carney-tserre-condo
#SBATCH -C titanrtx

module load anaconda/3-5.2.0
source activate sf

EXP_NAME="GN_R2D_Moments_VPN_test_viz"

python tools/run_net.py \
  --cfg configs/Moments/GN_R2D_18_PRED_VPN.yaml \
  DATA.COLOR_AUGMENTATION False \
  NUM_GPUS 2 \
  DATA.NUM_FRAMES 18 \
  DATA.NUM_REPEATED_SAMPLES 2 \
  TRAIN.BATCH_SIZE 2 \
  DATA_LOADER.NUM_WORKERS 8 \
  SUMMARY_PERIOD 10 \
  OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME

# python tools/visualize_dataset.py \
#   --cfg configs/Moments/GN_R2D_18_PRED_VPN.yaml \
#   DATA.COLOR_AUGMENTATION True \
#   NUM_GPUS 1 \
#   DATA.NUM_FRAMES 18 \
#   DATA.NUM_REPEATED_SAMPLES 2 \
#   TRAIN.BATCH_SIZE 2 \
#   DATA_LOADER.NUM_WORKERS 8 \
#   OUTPUT_DIR /users/azerroug/data/azerroug/slowfast/outputs/$EXP_NAME
