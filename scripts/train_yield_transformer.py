#!/usr/bin/env python3
"""Train the Transformer teacher model for DW yield prediction.

Uses the existing TimothyDroughtModel (ViewAgg + ModalityProj + Gating +
TemporalTransformer) but targets DW instead of WHC.

Usage:
    python scripts/train_yield_transformer.py --experiment exp01
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
from src.model.timothy_model import TimothyDroughtModel
from src.training.cv import LeaveOnePlantOutCV
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


def normalize_targets(plant_meta, col):
    v = plant_meta[col].dropna().values
    return float(v.mean()), float(v.std() + 1e-8)


def sample_t_cut(times, rng):
    if rng.random() < 0.5:
        return None
    if len(times) < 4:
        return None
    idx = rng.randint(3, len(times) - 1)
    return float(times[idx])


def train_one_fold(dataset, train_idx, val_idx, test_idx, cfg,
                   fold_id, output_dir, dw_mean, dw_std):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model = TimothyDroughtModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    loss_fn = torch.nn.SmoothL1Loss()

    best_val_mae = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    rng = random.Random(cfg.training.seed + fold_id)

    fold_dir = output_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.training.max_epochs + 1):
        t0 = time.time()

        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            dw_target = (batch["dw_target"] - dw_mean) / dw_std

            # Random truncation via active_mask
            times_np = batch["temporal_positions"][0].cpu().numpy()
            t_cut = sample_t_cut(times_np, rng)
            if t_cut is not None:
                mask = batch["temporal_positions"][0] <= t_cut
                T_used = int(mask.sum().item())
                batch["active_mask"][:, T_used:] = False

            optimizer.zero_grad()
            outputs = model(batch)
            # Use WHC head output as DW prediction (same architecture, different target)
            dw_pred = outputs["whc_pred"]
            loss = loss_fn(dw_pred, dw_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.train(False)
        val_maes = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                outputs = model(batch)
                dw_pred = outputs["whc_pred"] * dw_std + dw_mean
                val_maes.extend((dw_pred - batch["dw_target"]).abs().cpu().tolist())

        val_mae = float(np.mean(val_maes)) if val_maes else float("inf")
        elapsed = time.time() - t0

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
    test_preds, test_trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(batch)
            dw_pred = outputs["whc_pred"] * dw_std + dw_mean
            test_preds.extend(dw_pred.cpu().tolist())
            test_trues.extend(batch["dw_target"].cpu().tolist())

    test_mae = float(np.mean(np.abs(np.array(test_preds) - np.array(test_trues))))

    result = {
        "fold_id": fold_id, "best_epoch": best_epoch,
        "best_val_mae_g": best_val_mae, "test_mae_g": test_mae,
        "test_preds": test_preds, "test_trues": test_trues,
    }
    with open(fold_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/timothy.yaml")
    parser.add_argument("--experiment", default="exp01")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--output-dir", default="results/yield_transformer")
    parser.add_argument("--no-vi", action="store_true", help="Exclude vegetation indices from model input")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=[f"data.experiment={args.experiment}"])

    if args.no_vi:
        OmegaConf.update(cfg, "model.use_vi", False)

    print(f"Training Transformer teacher (yield) on {args.experiment} (use_vi={not args.no_vi})", flush=True)

    dataset = TimothyDroughtDataset(cfg)
    plant_meta = dataset.plant_meta
    valid_mask = plant_meta["dw_g"].notna().values
    plant_meta_valid = plant_meta[valid_mask].reset_index(drop=True)

    dw_mean, dw_std = normalize_targets(plant_meta_valid, "dw_g")
    print(f"DW stats: mean={dw_mean:.2f}, std={dw_std:.2f}", flush=True)

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
            dataset, train_idx, val_idx, test_idx,
            cfg, fold_idx, output_dir, dw_mean, dw_std,
        )
        print(f"  Test MAE: {result['test_mae_g']:.3f}g", flush=True)
        all_results.append(result)

    # Pooled metrics
    all_preds = np.array([p for r in all_results for p in r["test_preds"]])
    all_trues = np.array([t for r in all_results for t in r["test_trues"]])
    pooled_mae = float(np.mean(np.abs(all_preds - all_trues)))
    ss_res = float(np.sum((all_trues - all_preds) ** 2))
    ss_tot = float(np.sum((all_trues - all_trues.mean()) ** 2))
    r2 = 1 - ss_res / max(ss_tot, 1e-8)
    std_ratio = float(all_preds.std() / (all_trues.std() + 1e-8))

    summary = {
        "experiment": args.experiment,
        "n_folds": len(all_results),
        "mean_test_mae_g": pooled_mae,
        "std_test_mae_g": float(np.std([r["test_mae_g"] for r in all_results])),
        "r2_global": r2,
        "std_ratio": std_ratio,
        "dw_mean": dw_mean,
        "dw_std": dw_std,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Transformer yield summary ({args.experiment}) ===")
    print(f"  MAE={pooled_mae:.3f}g  R²={r2:+.3f}  std_ratio={std_ratio:.2f}")


if __name__ == "__main__":
    main()
