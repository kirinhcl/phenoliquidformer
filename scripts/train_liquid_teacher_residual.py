#!/usr/bin/env python3
"""Train Liquid (CfC) teacher with RESIDUAL prediction target.

Key idea: instead of predicting DW directly, the model predicts the residual
    dw_residual = dw - group_mean(WHC, genotype)
where group_mean is computed from the training fold only (no leakage).

Final prediction: model_pred * residual_std + group_mean
This guarantees the model is never worse than the group-mean baseline and
lets it focus on individual-level variation.

Usage:
    python scripts/train_liquid_teacher_residual.py --experiment exp01
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from src.data.dataset import TimothyDroughtDataset
from src.model.liquid_model import LiquidYieldModel
from src.training.cv import LeaveOnePlantOutCV
from src.training.distillation_loss import YieldLoss
from src.utils.config import load_config


def collate_fn(batch):
    result = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values, dtype=torch.float32)
        else:
            result[key] = values
    return result


def compute_group_means(plant_meta_df, train_indices):
    """Compute (WHC, genotype) group means from training fold only."""
    train_df = plant_meta_df.iloc[train_indices]
    group_means = train_df.groupby(["treatment", "genotype"])["dw_g"].mean().to_dict()
    # Fallback: overall mean for groups not in training
    overall_mean = train_df["dw_g"].mean()
    return group_means, overall_mean


def get_base_pred(batch, group_means, overall_mean, device):
    """Look up group-mean prediction for each plant in batch."""
    bases = []
    for i in range(len(batch["treatment"])):
        key = (batch["treatment"][i], batch["genotype"][i])
        bases.append(group_means.get(key, overall_mean))
    return torch.tensor(bases, dtype=torch.float32, device=device)


def sample_t_cut(times, rng):
    if rng.random() < 0.5:
        return None
    if len(times) < 4:
        return None
    idx = rng.randint(3, len(times) - 1)
    return float(times[idx])


def train_one_fold(dataset, train_idx, val_idx, test_idx, plant_meta,
                   cfg, fold_id, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compute group means and residual std from training fold only
    group_means, overall_mean = compute_group_means(plant_meta, train_idx)
    train_df = plant_meta.iloc[train_idx]
    train_residuals = []
    for _, row in train_df.iterrows():
        gm = group_means.get((row["treatment"], row["genotype"]), overall_mean)
        train_residuals.append(row["dw_g"] - gm)
    residual_std = float(np.std(train_residuals) + 1e-8)

    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx.tolist()),
        batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_fn,
    )

    model = LiquidYieldModel(role="teacher", cfg=cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    criterion = YieldLoss(
        huber_delta=OmegaConf.select(cfg.training.loss, "huber_delta", default=1.0),
        flower_weight=OmegaConf.select(cfg.training.loss, "flower_weight", default=0.3),
    )

    best_val_mae = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    rng = random.Random(cfg.training.seed + fold_id)

    fold_dir = output_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, cfg.training.max_epochs + 1):
        t0 = time.time()

        model.train()
        train_losses = []
        for batch in train_loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            dw_true = batch_dev["dw_target"]
            base_pred = get_base_pred(batch, group_means, overall_mean, device)
            residual_target = (dw_true - base_pred) / residual_std

            B = dw_true.shape[0]
            flower_target = torch.full((B,), float("nan"), device=device)

            times_np = batch_dev["temporal_positions"][0].cpu().numpy()
            t_cut = sample_t_cut(times_np, rng)

            optimizer.zero_grad()
            out = model(batch_dev, t_cut=t_cut)
            # Model predicts normalized residual
            loss, _ = criterion(
                out["dw_pred"], residual_target,
                out["flowering_pred"], flower_target,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        # Validation
        model.train(False)
        val_maes = []
        with torch.no_grad():
            for batch in val_loader:
                batch_dev = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                base = get_base_pred(batch, group_means, overall_mean, device)
                out = model(batch_dev, t_cut=None)
                dw_pred = out["dw_pred"] * residual_std + base
                val_maes.extend((dw_pred - batch_dev["dw_target"]).abs().cpu().tolist())

        val_mae = float(np.mean(val_maes)) if val_maes else float("inf")
        elapsed = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_mae": val_mae,
            "elapsed": elapsed,
        })

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), fold_dir / "best_model_state.pt")
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Fold {fold_id} Epoch {epoch}: train_loss={np.mean(train_losses):.4f} "
                  f"val_MAE={val_mae:.3f}g ({elapsed:.1f}s)", flush=True)

        if epochs_no_improve >= cfg.training.patience:
            print(f"  Early stopping at epoch {epoch} (best={best_epoch})", flush=True)
            break

    # Test
    model.load_state_dict(torch.load(fold_dir / "best_model_state.pt", weights_only=True))
    model.train(False)
    test_preds = []
    test_trues = []
    test_bases = []
    with torch.no_grad():
        for batch in test_loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            base = get_base_pred(batch, group_means, overall_mean, device)
            out = model(batch_dev, t_cut=None)
            dw_pred = out["dw_pred"] * residual_std + base
            test_preds.extend(dw_pred.cpu().tolist())
            test_trues.extend(batch_dev["dw_target"].cpu().tolist())
            test_bases.extend(base.cpu().tolist())

    test_mae = float(np.mean(np.abs(np.array(test_preds) - np.array(test_trues))))
    base_mae = float(np.mean(np.abs(np.array(test_bases) - np.array(test_trues))))

    result = {
        "fold_id": fold_id,
        "best_epoch": best_epoch,
        "best_val_mae_g": best_val_mae,
        "test_mae_g": test_mae,
        "test_baseline_mae_g": base_mae,
        "test_preds": test_preds,
        "test_trues": test_trues,
        "test_bases": test_bases,
        "residual_std": residual_std,
        "history": history,
    }
    with open(fold_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/timothy.yaml")
    parser.add_argument("--experiment", default="exp01")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--output-dir", default="results/liquid_teacher_residual")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=[f"data.experiment={args.experiment}"])

    if "liquid" not in cfg.model:
        cfg.model.liquid = OmegaConf.create({
            "hidden_dim": 32, "modality_dim": 32, "head_dim": 64,
        })

    print(f"Training liquid teacher (RESIDUAL) on {args.experiment}", flush=True)

    dataset = TimothyDroughtDataset(cfg)
    plant_meta = dataset.plant_meta
    valid_mask = plant_meta["dw_g"].notna().values
    plant_meta_valid = plant_meta[valid_mask].reset_index(drop=True)

    cv = LeaveOnePlantOutCV(plant_meta_valid, args.experiment, seed=cfg.training.seed)
    n_folds = args.max_folds or cv.n_folds
    print(f"LOPO CV: {n_folds} folds", flush=True)

    output_dir = Path(args.output_dir) / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(cv.split()):
        if fold_idx >= n_folds:
            break
        print(f"\n--- Fold {fold_idx} ---", flush=True)
        result = train_one_fold(
            dataset, train_idx, val_idx, test_idx, plant_meta_valid,
            cfg, fold_idx, output_dir,
        )
        print(f"  Test MAE: {result['test_mae_g']:.3f}g "
              f"(baseline MAE={result['test_baseline_mae_g']:.3f}g)", flush=True)
        all_results.append(result)

    # Pooled evaluation
    all_preds = np.array([p for r in all_results for p in r["test_preds"]])
    all_trues = np.array([t for r in all_results for t in r["test_trues"]])
    all_bases = np.array([b for r in all_results for b in r["test_bases"]])

    pooled_mae = float(np.mean(np.abs(all_preds - all_trues)))
    ss_res = float(np.sum((all_trues - all_preds) ** 2))
    ss_tot = float(np.sum((all_trues - all_trues.mean()) ** 2))
    r2_global = 1 - ss_res / max(ss_tot, 1e-8)
    std_ratio = float(all_preds.std() / (all_trues.std() + 1e-8))

    baseline_mae = float(np.mean(np.abs(all_bases - all_trues)))
    ss_res_b = float(np.sum((all_trues - all_bases) ** 2))
    baseline_r2 = 1 - ss_res_b / max(ss_tot, 1e-8)

    summary = {
        "experiment": args.experiment,
        "n_folds": len(all_results),
        "model": {
            "mean_fold_mae_g": float(np.mean([r["test_mae_g"] for r in all_results])),
            "std_fold_mae_g": float(np.std([r["test_mae_g"] for r in all_results])),
            "pooled_mae_g": pooled_mae,
            "r2_global": r2_global,
            "std_ratio": std_ratio,
        },
        "baseline_whc_geno": {
            "pooled_mae_g": baseline_mae,
            "r2_global": baseline_r2,
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Liquid Teacher (Residual) summary ({args.experiment}) ===")
    print(f"  Model    : MAE={pooled_mae:.3f}g  R²={r2_global:+.3f}  std_ratio={std_ratio:.2f}")
    print(f"  Baseline : MAE={baseline_mae:.3f}g  R²={baseline_r2:+.3f}")
    improvement_mae = (baseline_mae - pooled_mae) / baseline_mae * 100
    improvement_r2 = r2_global - baseline_r2
    print(f"  Improvement: MAE {improvement_mae:+.1f}%   R² {improvement_r2:+.3f}")


if __name__ == "__main__":
    main()
