#!/bin/bash
#SBATCH --job-name=timothy-features
#SBATCH --account=YOUR_PROJECT_ID
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/extract_features_%j.out
#SBATCH --error=logs/extract_features_%j.err

cd ${PROJECT_ROOT}
mkdir -p logs features

module load LUMI/25.09 partition/G PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250527

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1

# Install missing packages into user site-packages (cached across runs)
singularity exec $SIF pip install --user h5py tqdm openpyxl 2>&1 | tail -3

singularity exec $SIF python3 scripts/extract_features.py --experiment all --batch_size 64 --num_workers 4
