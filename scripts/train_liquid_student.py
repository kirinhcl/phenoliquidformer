#!/usr/bin/env python3
"""Distill Liquid teacher into image-only student with progressive truncation.

The student is a LiquidYieldModel(role='student') that takes only image features
(no fluor, no VI, no WHC, no genotype). It learns from:
    - Ground-truth DW (hard target)
    - Teacher's predictions (soft target)
    - Teacher's hidden state sequence (latent trajectory alignment via cosine sim)

Progressive truncation: during training, randomly truncate observations at
various DAS values. During test, evaluate at multiple truncation points to
build an "accuracy vs time" curve.

Usage:
    python scripts/train_liquid_student.py \\
        --experiment exp01 \\
        --teacher-dir results/liquid_teacher/exp01 \\
        --output-dir results/liquid_student
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
    """During training, randomly truncate to encourage early-prediction robustness."""
    r = rng.random()
    if r < 0.3:
        return None  # full sequence
    if len(times) < 4:
        return None
    idx = rng.randint(3, len(times) - 1)
    return float(times[idx])


def anneal_alpha(epoch, total_epochs, start=0.3, end=0.7):
    """Hard-target weight α increases over training."""
    progress = min(epoch / max(total_epochs, 1), 1.0)
    return start + (end - start) * progress


def distillation_loss(student_out, teacher_out, residual_target, flower_target,
                      yield_criterion, alpha, beta, gamma=0.3):
    """Combined distillation loss.

    L_hard : student vs ground truth (hard label)
    L_soft : student vs teacher output (soft label)
    L_traj : student hidden sequence vs teacher hidden sequence (cosine)
    """
    metrics = {}

    # Hard target loss
    l_hard, hard_m = yield_criterion(
        student_out["dw_pred"], residual_target,
        student_out["flowering_pred"], flower_target,
    )
    metrics["l_hard"] = float(l_hard.item())

    # Soft target loss (detached teacher)
    dw_soft = F.smooth_l1_loss(
        student_out["dw_pred"], teacher_out["dw_pred"].detach()
    )
    fl_soft = F.smooth_l1_loss(
        student_out["flowering_pred"], teacher_out["flowering_pred"].detach()
    )
    l_soft = dw_soft + 0.3 * fl_soft
    metrics["l_soft"] = float(l_soft.item())

    # Hidden sequence alignment (cosine similarity)
    s_seq = student_out["h_seq"]  # (B, T, hidden)
    t_seq = teacher_out["h_seq"].detach()
    T_min = min(s_seq.shape[1], t_seq.shape[1])
    s = F.normalize(s_seq[:, :T_min], dim=-1)
    t = F.normalize(t_seq[:, :T_min], dim=-1)
    cos_sim = (s * t).sum(dim=-1)  # (B, T_min)
    l_traj = (1.0 - cos_sim).mean()
    metrics["l_traj"] = float(l_traj.item())

    total = alpha * l_hard + beta * l_soft + gamma * l_traj
    metrics["l_total"] = float(total.item())
    return total, metrics


def train_student_fold(dataset, train_idx, val_idx, test_idx,
                       teacher_ckpt, cfg, fold_id, output_dir,
                       dw_mean, dw_std):
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

    # Load frozen teacher
    teacher = LiquidYieldModel(role="teacher", cfg=cfg).to(device)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device, weights_only=True))
    teacher.train(False)
    for p in teacher.parameters():
        p.requires_grad = False

    # Create student
    student = LiquidYieldModel(role="student", cfg=cfg).to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
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

        alpha = anneal_alpha(epoch, cfg.training.max_epochs, 0.3, 0.7)
        beta = 1.0 - alpha

        student.train()
        train_losses = []
        for batch in train_loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            dw_target_raw = batch_dev["dw_target"]
            dw_target_norm = (dw_target_raw - dw_mean) / dw_std
            B = dw_target_raw.shape[0]
            flower_target = torch.full((B,), float("nan"), device=device)

            times_np = batch_dev["temporal_positions"][0].cpu().numpy()
            t_cut = sample_t_cut(times_np, rng)

            optimizer.zero_grad()
            with torch.no_grad():
                teacher_out = teacher(batch_dev, t_cut=t_cut)
            student_out = student(batch_dev, t_cut=t_cut)

            loss, _ = distillation_loss(
                student_out, teacher_out, dw_target_norm, flower_target,
                criterion, alpha=alpha, beta=beta, gamma=0.3,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.training.gradient_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        # Validation (full time)
        student.train(False)
        val_maes = []
        with torch.no_grad():
            for batch in val_loader:
                batch_dev = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                out = student(batch_dev, t_cut=None)
                dw_pred = out["dw_pred"] * dw_std + dw_mean
                val_maes.extend((dw_pred - batch_dev["dw_target"]).abs().cpu().tolist())

        val_mae = float(np.mean(val_maes)) if val_maes else float("inf")
        elapsed = time.time() - t0
        history.append({
            "epoch": epoch, "alpha": alpha, "beta": beta,
            "train_loss": float(np.mean(train_losses)),
            "val_mae": val_mae, "elapsed": elapsed,
        })

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(student.state_dict(), fold_dir / "best_student_state.pt")
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Fold {fold_id} Epoch {epoch}: α={alpha:.2f} "
                  f"train_loss={np.mean(train_losses):.4f} val_MAE={val_mae:.3f}g "
                  f"({elapsed:.1f}s)", flush=True)

        if epochs_no_improve >= cfg.training.patience:
            print(f"  Early stopping at {epoch} (best={best_epoch})", flush=True)
            break

    # Test at multiple truncation points (progressive truncation)
    student.load_state_dict(torch.load(fold_dir / "best_student_state.pt", weights_only=True))
    student.train(False)
    test_results = {}
    for t_cut_val in [15, 20, 25, 30, 40, None]:
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch_dev = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                out = student(batch_dev, t_cut=t_cut_val)
                dw_pred = out["dw_pred"] * dw_std + dw_mean
                preds.extend(dw_pred.cpu().tolist())
                trues.extend(batch_dev["dw_target"].cpu().tolist())
        mae = float(np.mean(np.abs(np.array(preds) - np.array(trues))))
        test_results[str(t_cut_val)] = {"mae": mae, "preds": preds, "trues": trues}

    result = {
        "fold_id": fold_id,
        "best_epoch": best_epoch,
        "best_val_mae_g": best_val_mae,
        "test_by_tcut": test_results,
        "history": history,
    }
    with open(fold_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/timothy.yaml")
    parser.add_argument("--experiment", default="exp01")
    parser.add_argument("--teacher-dir", required=True)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--output-dir", default="results/liquid_student")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=[f"data.experiment={args.experiment}"])
    if "liquid" not in cfg.model:
        cfg.model.liquid = OmegaConf.create({
            "hidden_dim": 32, "modality_dim": 32, "head_dim": 64,
        })

    print(f"Distilling liquid student on {args.experiment}", flush=True)

    dataset = TimothyDroughtDataset(cfg)
    plant_meta = dataset.plant_meta
    valid_mask = plant_meta["dw_g"].notna().values
    plant_meta_valid = plant_meta[valid_mask].reset_index(drop=True)

    # Use teacher's DW stats
    teacher_summary = Path(args.teacher_dir) / "summary.json"
    with open(teacher_summary) as f:
        tdata = json.load(f)
    dw_mean = tdata["dw_mean"]
    dw_std = tdata["dw_std"]
    print(f"Teacher DW stats: mean={dw_mean:.2f}, std={dw_std:.2f}")

    cv = LeaveOnePlantOutCV(plant_meta_valid, args.experiment, seed=cfg.training.seed)
    n_folds = args.max_folds or cv.n_folds
    print(f"LOPO CV: {n_folds} folds", flush=True)

    output_dir = Path(args.output_dir) / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(cv.split()):
        if fold_idx >= n_folds:
            break
        teacher_ckpt = Path(args.teacher_dir) / f"fold_{fold_idx}" / "best_model_state.pt"
        if not teacher_ckpt.exists():
            print(f"  Skip fold {fold_idx}: no teacher checkpoint")
            continue
        print(f"\n--- Fold {fold_idx} ---", flush=True)
        result = train_student_fold(
            dataset, train_idx, val_idx, test_idx,
            teacher_ckpt, cfg, fold_idx, output_dir, dw_mean, dw_std,
        )
        tm_full = result["test_by_tcut"]["None"]["mae"]
        tm_20 = result["test_by_tcut"]["20"]["mae"]
        print(f"  Test MAE (full/t=20): {tm_full:.3f}g / {tm_20:.3f}g", flush=True)
        all_results.append(result)

    # Aggregate by t_cut (pooled across folds)
    summary = {
        "experiment": args.experiment,
        "n_folds": len(all_results),
        "dw_mean": dw_mean,
        "dw_std": dw_std,
        "by_tcut": {},
    }
    for t_cut in ["15", "20", "25", "30", "40", "None"]:
        pooled_preds = []
        pooled_trues = []
        for r in all_results:
            pooled_preds.extend(r["test_by_tcut"][t_cut]["preds"])
            pooled_trues.extend(r["test_by_tcut"][t_cut]["trues"])
        preds = np.array(pooled_preds)
        trues = np.array(pooled_trues)
        mae = float(np.mean(np.abs(preds - trues)))
        ss_res = float(np.sum((trues - preds) ** 2))
        ss_tot = float(np.sum((trues - trues.mean()) ** 2))
        r2 = 1 - ss_res / max(ss_tot, 1e-8)
        std_ratio = float(preds.std() / (trues.std() + 1e-8))
        summary["by_tcut"][t_cut] = {
            "pooled_mae_g": mae,
            "r2_global": r2,
            "std_ratio": std_ratio,
        }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Liquid Student summary ({args.experiment}) ===")
    for tc, s in summary["by_tcut"].items():
        label = f"t_cut=DAS {tc}" if tc != "None" else "t_cut=full"
        print(f"  {label:<20} MAE={s['pooled_mae_g']:.3f}g  R²={s['r2_global']:+.3f}  "
              f"std_ratio={s['std_ratio']:.2f}")


if __name__ == "__main__":
    main()
