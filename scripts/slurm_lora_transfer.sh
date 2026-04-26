#!/bin/bash
#SBATCH --job-name=timothy-lora
#SBATCH --account=YOUR_PROJECT_ID
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lora_transfer_%j.out
#SBATCH --error=logs/lora_transfer_%j.err

cd ${PROJECT_ROOT}
mkdir -p logs

module load LUMI/25.09 partition/G PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250527

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1

singularity exec $SIF pip install --user h5py tqdm openpyxl omegaconf scipy pyyaml ncps xgboost scikit-learn 2>&1 | tail -1

echo "=== LoRA Cross-Stage Transfer Re-run (corrected Exp02 DW) ==="
echo "Source model: results/lt_h64_n2_res/exp01 (pre-trained LT, 48-fold LOPO)"
echo "Target:       exp02 (40 plants, corrected endpoint, DW mean=27.35 g)"
echo ""

singularity exec $SIF python3 scripts/train_lora_transfer.py \
    --config configs/timothy.yaml \
    --source-dir results/lt_h64_n2_res/exp01 \
    --output-dir results/lora_lt_transfer_fixed \
    --model-type liquid_transformer

echo ""
echo "=== LoRA Transfer complete ==="
if [ -f "results/lora_lt_transfer_fixed/lora_summary.json" ]; then
    echo "--- Summary ---"
    cat results/lora_lt_transfer_fixed/lora_summary.json
fi
