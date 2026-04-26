#!/bin/bash
#SBATCH --job-name=timothy-lt-interp
#SBATCH --account=YOUR_PROJECT_ID
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/lt_interp_%j.out
#SBATCH --error=logs/lt_interp_%j.err

# Liquid Transformer interpretability analysis on LUMI.
# - 48-fold saliency (d dw_pred / d input_t) for image / fluor / env
# - Latent trajectory PCA (fold 0 model)
# - Liquid time-constant weight distributions (fold 0 model)
#
# Produces:
#   results/liquid_interpretability/saliency_mean.npy       (3, T)
#   results/liquid_interpretability/saliency_per_plant.npy  (N, T, 3)
#   results/liquid_interpretability/saliency_das.npy        (T,)
#   results/liquid_interpretability/saliency_meta.json
#   results/liquid_interpretability/time_constants.json
#   paper/figures/output/fig_saliency_real.{pdf,png}
#   paper/figures/output/fig_latent_trajectories_lt.{pdf,png}
#   paper/figures/output/fig_time_constants_lt.{pdf,png}

cd ${PROJECT_ROOT}
mkdir -p logs results/liquid_interpretability

module load LUMI/25.09 partition/G PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250527

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1

singularity exec $SIF pip install --user h5py tqdm openpyxl omegaconf scipy pyyaml ncps scikit-learn matplotlib 2>&1 | tail -1

echo "=== Liquid Transformer interpretability ==="
singularity exec $SIF python3 scripts/analyze_liquid_interpretability.py \
    --model-dir results/lt_h64_n2_res/exp01 \
    --experiment exp01 \
    --override model.liquid.hidden_dim=64 model.liquid.n_layers=2

echo ""
echo "=== Output files ==="
ls -la results/liquid_interpretability/
echo ""
ls -la paper/figures/output/fig_saliency_real.* \
       paper/figures/output/fig_latent_trajectories_lt.* \
       paper/figures/output/fig_time_constants_lt.* 2>/dev/null
