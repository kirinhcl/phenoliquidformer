#!/bin/bash
#SBATCH --job-name=timothy-novi-baselines
#SBATCH --account=YOUR_PROJECT_ID
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/novi_baselines_%j.out
#SBATCH --error=logs/novi_baselines_%j.err

cd ${PROJECT_ROOT}
mkdir -p logs

module load LUMI/25.09 partition/G PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250527

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1

singularity exec $SIF pip install --user h5py tqdm openpyxl omegaconf scipy pyyaml ncps xgboost scikit-learn 2>&1 | tail -1

echo "=== Stage 1: Classical baselines (no VI) ==="
singularity exec $SIF python3 scripts/train_yield_baselines.py \
    --experiment exp01 \
    --no-vi

echo "=== Stage 2: Transformer (no VI) ==="
singularity exec $SIF python3 scripts/train_yield_transformer.py \
    --experiment exp01 \
    --no-vi
