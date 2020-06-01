#!/bin/bash
#SBATCH --mem=10G
#SBATCH -n 10
#SBATCH -J augment_dataset
#SBATCH -o augment_dataset.out
#SBATCH -e augment_dataset.err
#SBATCH --time=10:00:00
#SBATCH --account=carney-tserre-condo

module load anaconda/3-5.2.0
source activate sf

# source activate py36

python motion_bench.py
