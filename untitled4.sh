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

# python motion_bench.py
# python motion_data.py
# python load_icons.py

# python slowfast/datasets/simple_motion_data.py
# mv *.npy ../scratch/simple_motion/

python motion_data.py

# python run_inference.py