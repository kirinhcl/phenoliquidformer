#!/bin/bash
#SBATCH --job-name=timothy-yield
#SBATCH --account=YOUR_PROJECT_ID
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/yield_%j.out
#SBATCH --error=logs/yield_%j.err

cd ${PROJECT_ROOT}
mkdir -p logs

module load LUMI/25.09 partition/G PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250527

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1

singularity exec $SIF pip install --user h5py tqdm openpyxl omegaconf scipy pyyaml torchdiffeq 2>&1 | tail -1

# Step 1: Train multimodal teacher with LOPO CV on exp01
echo "=== Stage 1: Train Teacher ==="
singularity exec $SIF python3 scripts/train_yield_teacher.py \
    --config configs/timothy.yaml \
    --experiment exp01 \
    --output-dir results/yield_teacher

# Step 2: Distill image-only student
echo "=== Stage 2: Distill Student ==="
singularity exec $SIF python3 scripts/train_yield_student.py \
    --config configs/timothy.yaml \
    --experiment exp01 \
    --teacher-dir results/yield_teacher/exp01 \
    --output-dir results/yield_student
