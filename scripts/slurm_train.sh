#!/bin/bash
#SBATCH --job-name=timothy-train
#SBATCH --account=YOUR_PROJECT_ID
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

cd ${PROJECT_ROOT}
mkdir -p logs

module load LUMI/25.09 partition/G PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250527

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1

# Install missing packages
singularity exec $SIF pip install --user h5py tqdm openpyxl omegaconf scipy 2>&1 | tail -3

# LOWHO CV (primary — tests extrapolation to unseen drought severity)
singularity exec $SIF python3 scripts/train_timothy.py --config configs/timothy.yaml --cv lowho --output-dir results

# LOPO CV
singularity exec $SIF python3 scripts/train_timothy.py --config configs/timothy.yaml --cv lopo --output-dir results
