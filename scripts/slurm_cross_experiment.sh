#!/bin/bash
#SBATCH --job-name=timothy-cross
#SBATCH --account=YOUR_PROJECT_ID
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cross_experiment_%j.out
#SBATCH --error=logs/cross_experiment_%j.err

cd ${PROJECT_ROOT}
mkdir -p logs

module load LUMI/25.09 partition/G PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250527

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1

singularity exec $SIF pip install --user h5py tqdm openpyxl omegaconf scipy 2>&1 | tail -1

# Q6: Train on exp01, test on exp02 (vernalization generalization)
singularity exec $SIF python3 scripts/train_cross_experiment.py \
    --train-exp exp01 --test-exp exp02 --output-dir results/cross_exp01_to_exp02

# Q6: Train on exp02, test on exp03 (regrowth generalization)
singularity exec $SIF python3 scripts/train_cross_experiment.py \
    --train-exp exp02 --test-exp exp03 --output-dir results/cross_exp02_to_exp03
