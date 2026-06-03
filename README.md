# LiquidFormer

Code release for **"LiquidFormer: A Liquid Neural Network for Cross-Stage Multimodal Yield Prediction in Timothy Grass"**, submitted to *Computers and Electronics in Agriculture*. (This repository was published under its earlier working name, `phenoliquidformer`.)

LiquidFormer is a hybrid architecture combining a gated Transformer-style multimodal fusion block, a Closed-form Continuous-time (CfC) recurrent backbone, and an attention-pooling readout, adapted across developmental stages via LoRA, for dry-weight yield prediction in Timothy grass (*Phleum pratense*) under drought. A phenology-aware variant (PhenologyCfC, with time constants modulated by φ = DAS/max DAS) is retained for the headline checkpoint but contributes no statistically significant gain (Wilcoxon p = 0.37) and is reported as an architectural ablation rather than a load-bearing design choice.

## Architecture

```
src/
├── data/dataset.py             # TimothyDroughtDataset — multimodal time-series loader
├── model/
│   ├── encoder.py              # ViewAggregation (image view pooling)
│   ├── gating.py               # ModalityProjection + ModalityGating
│   ├── temporal.py             # TemporalTransformer (legacy baseline)
│   ├── temporal_attention.py   # TemporalAttentionPooling (4-head query token)
│   ├── heads.py                # WHCRegressionHead, BiomassTrajectoryHead, YieldHead
│   ├── phenology_cfc.py        # PhenologyCfCCell + PhenologyCfC (τ modulated by φ=DAS/max_DAS)
│   ├── liquid_transformer.py   # LiquidFormer (321K params) — PRIMARY MODEL
│   ├── phenology_liquid_model.py  # PhenologyLiquidModel (PhenologyCfC + attention pooling; φ variant)
│   ├── timothy_model.py        # Pure Transformer baseline (602K params)
│   ├── liquid_model.py         # Legacy Liquid NN / CfC (66K params)
│   ├── yield_model.py          # Teacher-Student Neural ODE (legacy distillation)
│   └── ode_dynamics.py         # Neural ODE (73K params)
├── training/
│   ├── cv.py                   # LeaveOnePlantOutCV, LeaveOneWHCOutCV, CrossExperimentCV
│   ├── losses.py               # MultiTaskLoss (Huber + MSE)
│   ├── trainer.py              # Training loop with early stopping
│   └── distillation_loss.py    # Teacher-student distillation losses (legacy)
└── utils/config.py             # OmegaConf YAML loader
```

## Installation

```bash
git clone https://github.com/<your-user>/phenoliquidformer.git
cd phenoliquidformer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The Timothy drought dataset is hosted at NaPPI (National Plant Phenotyping Infrastructure, University of Helsinki). The raw multimodal data are not redistributed in this repository; please contact the authors for access.

Expected layout:

```
data/
├── plant_metadata.csv
├── timepoint_metadata.csv
├── 2023-Timothy-01-Nonvernalized/
│   ├── FCQ_Timothy-01.xlsx
│   ├── DigBio_Timothy-01.xlsx
│   └── Envdata_Timothy-01_Probes.xlsx
└── 2024-Timothy-02-Vernalized/
    └── EndPoint_Timothy-02_Weight+Flowering_fixed2.xlsx
features/
└── exp01_dinov2.h5            # 768-dim DINOv2 embeddings (extracted via scripts/extract_features.py)
```

## Reproducing the paper

### 1. Build metadata

```bash
python scripts/build_metadata.py
```

### 2. Extract DINOv2 features (one-time)

```bash
python scripts/extract_features.py --experiment exp01
```

### 3. Train baselines

```bash
python scripts/train_yield_baselines.py        # Ridge, RF, XGBoost, group-mean
python scripts/train_yield_transformer.py      # Pure Transformer baseline
```

### 4. Train LiquidFormer

```bash
python scripts/train_liquid_teacher.py \
    --experiment exp01 \
    --model-type liquid_transformer \
    --output-dir results/lt_h64_n2_res \
    --override model.liquid.hidden_dim=64 model.liquid.n_layers=2
```

### 5. LoRA cross-stage adaptation

```bash
python scripts/train_lora_transfer.py \
    --pretrained results/lt_h64_n2_res/exp01 \
    --target-experiment exp02 \
    --rank 2
```

### 6. Statistics and figures

```bash
python scripts/compute_paper_stats.py          # Wilcoxon, bootstrap CI, Cohen's d
python scripts/analyze_liquid_interpretability.py   # Saliency + latent trajectories
```

### SLURM (LUMI)

The `scripts/slurm_*.sh` scripts target the LUMI supercomputer with AMD MI250X GPUs. Replace `YOUR_PROJECT_ID` and set `PROJECT_ROOT` before submitting:

```bash
export PROJECT_ROOT=/path/to/phenoliquidformer
sbatch scripts/slurm_lt_ablation.sh
```

## Key results (Exp01, 48-fold leave-one-plant-out; deep models are 5-seed means ± SD)

| Model | Params | MAE (g) | R² |
|---|---|---|---|
| **LiquidFormer** | 321K | **1.83** | **0.797 ± 0.059** |
| Transformer | 602K | 1.92 | 0.761 ± 0.058 |
| GRU+fusion | 276K | 2.03 | 0.736 ± 0.074 |
| XGBoost | — | 2.96 | 0.46 |
| Random Forest | — | 3.03 | 0.45 |
| Ridge | — | 2.81 | 0.37 |
| Group-mean | — | 3.60 | 0.34 |

LiquidFormer and the Transformer are statistically equivalent in mean accuracy (paired Wilcoxon p = 0.48, ns); LiquidFormer's advantage is parameter efficiency (47% fewer parameters), a marginal edge over a matched GRU backbone (LiquidFormer vs GRU+fusion p = 0.054), and an inspectable continuous-time gate — not headline accuracy. A rank-2 LoRA adapter (~500 trainable parameters) recovers the cross-stage Exp01→Exp02 collapse, matching full fine-tuning while updating ~10³× fewer parameters.

## Citation

If you use this code, please cite:

```bibtex
@article{lu2026liquidformer,
  title   = {LiquidFormer: A Liquid Neural Network for Cross-Stage
             Multimodal Yield Prediction in Timothy Grass},
  author  = {Lu, Chenghao and Liu, Chang and Poque, Sylvain and Yu, Kang and
             Himanen, Kristiina and Su, Xiang},
  journal = {Computers and Electronics in Agriculture},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

- NaPPI (National Plant Phenotyping Infrastructure), University of Helsinki, for plant phenotyping data
- LUMI supercomputer (CSC, Finland) for compute resources
- The CfC implementation is built on [`ncps`](https://github.com/mlech26l/ncps) (Hasani et al.)
- Image embeddings produced by [DINOv2](https://github.com/facebookresearch/dinov2)
- Multimodal fusion components adapted from the Faba/LUPIN project
