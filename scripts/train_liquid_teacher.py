#!/usr/bin/env python3
"""Train the Liquid Neural Network (CfC) teacher model for yield prediction.

Usage:
    python scripts/train_liquid_teacher.py --experiment exp01
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


def normalize_targets(plant_meta, target_col):
    values = plant_meta[target_col].dropna().values
    return float(values.mean()), float(values.std() + 1e-8)


def sample_t_cut(times, rng):
    if rng.random() < 0.5:
        return None
    if len(times) < 4:
        return None
    idx = rng.randint(3, len(times) - 1)
    return float(times[idx])


def _apply_modality_drop(batch: dict, drop_modalities: list[str]) -> dict:
    """Zero out specified modalities in the batch for ablation studies."""
    if not drop_modalities:
        return batch
    if "image" in drop_modalities:
        batch["image_mask"] = torch.zeros_like(batch["image_mask"])
    if "fluor" in drop_modalities:
        batch["fluorescence"] = torch.zeros_like(batch["fluorescence"])
        if "fluor_mask" in batch:
            batch["fluor_mask"] = torch.zeros_like(batch["fluor_mask"])
    if "env" in drop_modalities:
        if "environment" in batch:
            batch["environment"] = torch.zeros_like(batch["environment"])
    return batch


def train_one_fold(dataset, train_idx, val_idx, test_idx, cfg,
                   fold_id, output_dir, dw_mean, dw_std, model_type="liquid",
                   drop_modalities=None):
    drop_modalities = drop_modalities or []
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

    if model_type == "liquid_transformer":
        from src.model.liquid_transformer import LiquidTransformerModel
        model = LiquidTransformerModel(role="teacher", cfg=cfg).to(device)
    elif model_type == "phenology":
        from src.model.phenology_liquid_model import PhenologyLiquidModel
        model = PhenologyLiquidModel(role="teacher", cfg=cfg).to(device)
    else:
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
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch = _apply_modality_drop(batch, drop_modalities)
            dw_target_raw = batch["dw_target"]
            dw_target = (dw_target_raw - dw_mean) / dw_std
            B = dw_target.shape[0]
            flower_target = torch.full((B,), float("nan"), device=device)

            times_np = batch["temporal_positions"][0].cpu().numpy()
            t_cut = sample_t_cut(times_np, rng)

            optimizer.zero_grad()
            out = model(batch, t_cut=t_cut)

            loss, _ = criterion(
                out["dw_pred"], dw_target,
                out["flowering_pred"], flower_target,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.train(False)
        val_maes = []
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                batch = _apply_modality_drop(batch, drop_modalities)
                out = model(batch, t_cut=None)
                dw_pred = out["dw_pred"] * dw_std + dw_mean
                dw_true = batch["dw_target"]
                val_maes.extend((dw_pred - dw_true).abs().cpu().tolist())
                val_preds.extend(dw_pred.cpu().tolist())
                val_trues.extend(dw_true.cpu().tolist())

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
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch = _apply_modality_drop(batch, drop_modalities)
            out = model(batch, t_cut=None)
            dw_pred = out["dw_pred"] * dw_std + dw_mean
            test_preds.extend(dw_pred.cpu().tolist())
            test_trues.extend(batch["dw_target"].cpu().tolist())

    test_mae = float(np.mean(np.abs(np.array(test_preds) - np.array(test_trues))))

    result = {
        "fold_id": fold_id,
        "best_epoch": best_epoch,
        "best_val_mae_g": best_val_mae,
        "test_mae_g": test_mae,
        "test_preds": test_preds,
        "test_trues": test_trues,
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
    parser.add_argument("--output-dir", default="results/liquid_teacher")
    parser.add_argument(
        "--model-type", choices=["liquid", "phenology", "liquid_transformer"], default="liquid",
        help="Model type: liquid (baseline CfC), phenology (PhenologyCfC + TemporalAttention), "
             "or liquid_transformer (Transformer fusion + PhenologyCfC)",
    )
    parser.add_argument("--no-vi", action="store_true", help="Exclude vegetation indices from model input")
    parser.add_argument(
        "--drop-modality", nargs="*", default=[],
        choices=["image", "fluor", "env"],
        help="Zero out specified modalities for ablation (e.g. --drop-modality image)",
    )
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides, e.g. model.liquid.hidden_dim=64")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=[f"data.experiment={args.experiment}"] + args.override)

    if args.no_vi:
        OmegaConf.update(cfg, "model.use_vi", False)

    if "liquid" not in cfg.model:
        cfg.model.liquid = OmegaConf.create({
            "hidden_dim": 32, "modality_dim": 32, "head_dim": 64,
        })

    drop_label = f", drop={args.drop_modality}" if args.drop_modality else ""
    print(f"Training liquid teacher on {args.experiment} (use_vi={not args.no_vi}{drop_label})", flush=True)

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
            model_type=args.model_type,
            drop_modalities=args.drop_modality,
        )
        print(f"  Test MAE: {result['test_mae_g']:.3f}g", flush=True)
        all_results.append(result)

    test_maes = [r["test_mae_g"] for r in all_results]

    # Pool all predictions for global R² and std ratio
    all_preds = np.array([p for r in all_results for p in r["test_preds"]])
    all_trues = np.array([t for r in all_results for t in r["test_trues"]])
    ss_res = float(np.sum((all_trues - all_preds) ** 2))
    ss_tot = float(np.sum((all_trues - all_trues.mean()) ** 2))
    r2_global = 1 - ss_res / max(ss_tot, 1e-8)
    std_ratio = float(all_preds.std() / (all_trues.std() + 1e-8))

    summary = {
        "experiment": args.experiment,
        "n_folds": len(all_results),
        "mean_test_mae_g": float(np.mean(test_maes)),
        "std_test_mae_g": float(np.std(test_maes)),
        "r2_global": r2_global,
        "std_ratio": std_ratio,
        "dw_mean": dw_mean,
        "dw_std": dw_std,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Liquid Teacher summary ({args.experiment}) ===")
    print(f"  DW MAE:   {summary['mean_test_mae_g']:.3f} +/- {summary['std_test_mae_g']:.3f} g")
    print(f"  R²:       {r2_global:+.3f}")
    print(f"  std_ratio: {std_ratio:.2f}")
    print(f"  (WHC+Geno baseline R²=0.64 for reference)")


if __name__ == "__main__":
    main()
