#!/usr/bin/env python3
"""Distill teacher model into image-only student.

Student architecture: same LatentODE but with image modality only, no context.
Distillation losses: hard (true DW) + soft (teacher DW) + latent trajectory alignment.

Usage:
    python scripts/train_yield_student.py \\
        --experiment exp01 \\
        --teacher-dir results/yield_teacher/exp01 \\
        --output-dir results/yield_student/exp01
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
from src.model.yield_model import TimothyYieldModel
from src.training.cv import LeaveOnePlantOutCV
from src.training.distillation_loss import DistillationLoss
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


def sample_t_cut(times, rng):
    if rng.random() < 0.5:
        return None
    if len(times) < 4:
        return None
    idx = rng.randint(3, len(times) - 1)
    return float(times[idx])


def anneal_alpha(epoch, total_epochs, start=0.3, end=0.7):
    """Gradually increase hard-target weight from start to end."""
    progress = min(epoch / max(total_epochs, 1), 1.0)
    return start + (end - start) * progress


def train_student_fold(dataset, train_idx, val_idx, test_idx, cfg,
                       teacher_ckpt, fold_id, output_dir, dw_mean, dw_std):
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
    teacher = TimothyYieldModel(role="teacher", cfg=cfg).to(device)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device, weights_only=True))
    teacher.train(False)
    for p in teacher.parameters():
        p.requires_grad = False

    # Create student
    student = TimothyYieldModel(role="student", cfg=cfg).to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    criterion = DistillationLoss(
        alpha_hard=0.3, beta_soft=0.7, gamma_traj=0.3,
        delta_recon=0.1, epsilon_smooth=0.01,
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
        criterion.set_weights(alpha=alpha, beta=beta)

        student.train()
        train_losses = []
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            dw_target_raw = batch["dw_target"]
            dw_target = (dw_target_raw - dw_mean) / dw_std
            B = dw_target.shape[0]
            flower_target = torch.full((B,), float("nan"), device=device)

            times_np = batch["temporal_positions"][0].cpu().numpy()
            t_cut = sample_t_cut(times_np, rng)

            optimizer.zero_grad()
            # Teacher forward (frozen)
            with torch.no_grad():
                teacher_out = teacher(batch, t_cut=t_cut)
            # Student forward
            student_out = student(batch, t_cut=t_cut)

            # Reconstruction and smoothness closures
            active_mask = batch["active_mask"][:, : student_out["h_traj"].shape[1]]

            def recon_fn():
                return student.ode.reconstruction_loss(
                    student_out["h_traj"], student_out["obs_used"], active_mask
                )

            def smooth_fn():
                return student.ode.smoothness_loss(student_out["h_traj"])

            loss, _ = criterion(
                student_out=student_out,
                teacher_out=teacher_out,
                dw_target=dw_target,
                flower_target=flower_target,
                recon_loss_fn=recon_fn,
                smoothness_loss_fn=smooth_fn,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.training.gradient_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        student.train(False)
        val_maes = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                out = student(batch, t_cut=None)
                dw_pred = out["dw_pred"] * dw_std + dw_mean
                val_maes.extend((dw_pred - batch["dw_target"]).abs().cpu().tolist())

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

    # Test (multiple truncation points)
    student.load_state_dict(torch.load(fold_dir / "best_student_state.pt", weights_only=True))
    student.train(False)
    test_results = {}
    for t_cut in [15, 20, 25, 30, 40, None]:
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                out = student(batch, t_cut=t_cut)
                dw_pred = out["dw_pred"] * dw_std + dw_mean
                preds.extend(dw_pred.cpu().tolist())
                trues.extend(batch["dw_target"].cpu().tolist())
        mae = float(np.mean(np.abs(np.array(preds) - np.array(trues))))
        test_results[str(t_cut)] = {"mae": mae, "preds": preds, "trues": trues}

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
    parser.add_argument("--output-dir", default="results/yield_student")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=[f"data.experiment={args.experiment}"])

    if "ode" not in cfg.model:
        cfg.model.ode = OmegaConf.create({
            "latent_dim": 64, "vf_hidden": 64, "decoder_hidden": 64,
            "method": "dopri5", "rtol": 1e-3, "atol": 1e-4,
        })
    if "heads" not in cfg.model:
        cfg.model.heads = OmegaConf.create({})
    if "yield_hidden" not in cfg.model.heads:
        cfg.model.heads.yield_hidden = 64

    print(f"Distilling student on {args.experiment}", flush=True)

    dataset = TimothyDroughtDataset(cfg)
    plant_meta = dataset.plant_meta
    valid_mask = plant_meta["dw_g"].notna().values
    plant_meta_valid = plant_meta[valid_mask].reset_index(drop=True)

    # Load teacher normalization stats from summary
    teacher_summary = Path(args.teacher_dir) / "summary.json"
    with open(teacher_summary) as f:
        tdata = json.load(f)
    dw_mean = tdata["dw_mean"]
    dw_std = tdata["dw_std"]
    print(f"Using teacher DW stats: mean={dw_mean:.2f}, std={dw_std:.2f}")

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
            print(f"  Skipping fold {fold_idx}: no teacher checkpoint")
            continue
        print(f"\n--- Fold {fold_idx} ---", flush=True)
        result = train_student_fold(
            dataset, train_idx, val_idx, test_idx,
            cfg, teacher_ckpt, fold_idx, output_dir, dw_mean, dw_std,
        )
        tm_full = result["test_by_tcut"]["None"]["mae"]
        tm_20 = result["test_by_tcut"]["20"]["mae"]
        print(f"  Test MAE (full / t=20): {tm_full:.3f}g / {tm_20:.3f}g", flush=True)
        all_results.append(result)

    # Aggregate by t_cut
    summary = {
        "experiment": args.experiment,
        "n_folds": len(all_results),
        "dw_mean": dw_mean,
        "dw_std": dw_std,
        "by_tcut": {},
    }
    for t_cut in ["15", "20", "25", "30", "40", "None"]:
        maes = [r["test_by_tcut"][t_cut]["mae"] for r in all_results]
        summary["by_tcut"][t_cut] = {
            "mean_mae_g": float(np.mean(maes)),
            "std_mae_g": float(np.std(maes)),
        }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Student summary ({args.experiment}) ===")
    for tc, stats in summary["by_tcut"].items():
        print(f"  t_cut={tc}: MAE={stats['mean_mae_g']:.3f} +/- {stats['std_mae_g']:.3f} g")


if __name__ == "__main__":
    main()
