#!/bin/bash
#SBATCH --job-name=multiview-ablation-patches-shared
#SBATCH --output=multiview-ablation-%j.out
#SBATCH --error=multiview-ablation-%j.err
#SBATCH --partition=your_slurm_partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1

# 1. Clear environment and load cluster-provided software
module purge

# 2. Activate your local environment (Choose Option A or Option B)

conda activate multiview-env  # <-- Change to your environment name

bash /users/k24058220/multiview-crl/run_training_create.sh
