#!/bin/bash
#SBATCH --job-name=timothy-lt-ablation
#SBATCH --account=YOUR_PROJECT_ID
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lt_ablation_%j.out
#SBATCH --error=logs/lt_ablation_%j.err

cd ${PROJECT_ROOT}
mkdir -p logs

module load LUMI/25.09 partition/G PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250527

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1

singularity exec $SIF pip install --user h5py tqdm openpyxl omegaconf scipy pyyaml ncps xgboost scikit-learn 2>&1 | tail -1

LT_OVERRIDES="model.liquid.hidden_dim=64 model.liquid.n_layers=2"

echo "=== Ablation 1/4: Drop Image (ChlF + Env only) ==="
singularity exec $SIF python3 scripts/train_liquid_teacher.py \
    --experiment exp01 \
    --model-type liquid_transformer \
    --no-vi \
    --drop-modality image \
    --output-dir results/lt_ablation/drop_image \
    --override $LT_OVERRIDES

echo "=== Ablation 2/4: Drop ChlF (Image + Env only) ==="
singularity exec $SIF python3 scripts/train_liquid_teacher.py \
    --experiment exp01 \
    --model-type liquid_transformer \
    --no-vi \
    --drop-modality fluor \
    --output-dir results/lt_ablation/drop_fluor \
    --override $LT_OVERRIDES

echo "=== Ablation 3/4: Image only ==="
singularity exec $SIF python3 scripts/train_liquid_teacher.py \
    --experiment exp01 \
    --model-type liquid_transformer \
    --no-vi \
    --drop-modality fluor env \
    --output-dir results/lt_ablation/image_only \
    --override $LT_OVERRIDES

echo "=== Ablation 4/4: ChlF only ==="
singularity exec $SIF python3 scripts/train_liquid_teacher.py \
    --experiment exp01 \
    --model-type liquid_transformer \
    --no-vi \
    --drop-modality image env \
    --output-dir results/lt_ablation/fluor_only \
    --override $LT_OVERRIDES

echo "=== All ablations complete ==="
echo ""
echo "--- Summary ---"
for variant in drop_image drop_fluor image_only fluor_only; do
    if [ -f "results/lt_ablation/${variant}/exp01/summary.json" ]; then
        echo "${variant}:"
        cat "results/lt_ablation/${variant}/exp01/summary.json"
        echo ""
    fi
done
