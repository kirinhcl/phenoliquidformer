#!/usr/bin/env python3
"""Cross-experiment evaluation: test Liquid models trained on exp01 on exp02/exp03.

Tests both teacher (multimodal + context) and student (image-only) models
trained on exp01 against exp02 data to measure transfer degradation.

Usage:
    python scripts/eval_cross_experiment_liquid.py \\
        --teacher-dir results/liquid_teacher/exp01 \\
        --student-dir results/liquid_student/exp01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.dataset import TimothyDroughtDataset
from src.model.liquid_model import LiquidYieldModel
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


def evaluate_model(model, dataset, dw_mean, dw_std, device, t_cut=None):
    """Run model on entire dataset, return preds and trues."""
    model.to(device)
    model.train(False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    all_preds = []
    all_trues = []
    all_plants = []
    all_treatments = []
    all_genotypes = []

    with torch.no_grad():
        for batch in loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Handle max_T mismatch: pad or truncate
            test_T = batch_dev["images"].shape[1]
            # We need to handle the case where test dataset has different T
            # than training dataset. The CfC handles variable-length natively.

            out = model(batch_dev, t_cut=t_cut)
            dw_pred = out["dw_pred"] * dw_std + dw_mean

            all_preds.extend(dw_pred.cpu().tolist())

            dw_true = batch_dev["dw_target"]
            all_trues.extend(dw_true.cpu().tolist())
            all_plants.extend(batch["plant_id"])
            all_treatments.extend(batch["treatment"])
            all_genotypes.extend(batch["genotype"])

    return {
        "preds": np.array(all_preds),
        "trues": np.array(all_trues),
        "plants": all_plants,
        "treatments": all_treatments,
        "genotypes": all_genotypes,
    }


def compute_metrics(preds, trues):
    mae = float(np.mean(np.abs(preds - trues)))
    ss_res = float(np.sum((trues - preds) ** 2))
    ss_tot = float(np.sum((trues - trues.mean()) ** 2))
    r2 = 1 - ss_res / max(ss_tot, 1e-8)
    std_ratio = float(preds.std() / max(trues.std(), 1e-8))
    return {"mae": mae, "r2": r2, "std_ratio": std_ratio}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/timothy.yaml")
    parser.add_argument("--teacher-dir", required=True)
    parser.add_argument("--student-dir", default=None)
    parser.add_argument("--output-dir", default="results/liquid_cross_experiment")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load teacher normalization stats
    teacher_summary_path = Path(args.teacher_dir) / "summary.json"
    with open(teacher_summary_path) as f:
        teacher_summary = json.load(f)
    dw_mean = teacher_summary["dw_mean"]
    dw_std = teacher_summary["dw_std"]
    print(f"Teacher DW stats: mean={dw_mean:.2f}, std={dw_std:.2f}")

    # Load exp01 config as base
    cfg = load_config(args.config)
    if "liquid" not in cfg.model:
        cfg.model.liquid = OmegaConf.create({
            "hidden_dim": 32, "modality_dim": 32, "head_dim": 64,
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Use fold_0's checkpoint as representative model
    teacher_ckpt = Path(args.teacher_dir) / "fold_0" / "best_model_state.pt"

    # Also try averaging predictions across all folds for more stable estimate
    fold_dirs = sorted(Path(args.teacher_dir).glob("fold_*"))
    n_folds_available = len([d for d in fold_dirs if (d / "best_model_state.pt").exists()])
    print(f"Available teacher folds: {n_folds_available}")

    # Evaluate on exp01 (within-experiment baseline), exp02
    for test_exp in ["exp01", "exp02"]:
        print(f"\n{'='*60}")
        print(f"Evaluating on {test_exp}")
        print(f"{'='*60}")

        cfg_test = load_config(args.config, overrides=[f"data.experiment={test_exp}"])
        if "liquid" not in cfg_test.model:
            cfg_test.model.liquid = OmegaConf.create({
                "hidden_dim": 32, "modality_dim": 32, "head_dim": 64,
            })
        test_dataset = TimothyDroughtDataset(cfg_test)

        # Filter to plants with DW data
        valid_mask = test_dataset.plant_meta["dw_g"].notna().values
        valid_indices = np.where(valid_mask)[0].tolist()
        if not valid_indices:
            print(f"  No DW targets in {test_exp}, skipping")
            continue

        from torch.utils.data import Subset
        test_subset = Subset(test_dataset, valid_indices)
        print(f"  {len(valid_indices)} plants with DW targets")
        print(f"  DW range: {test_dataset.plant_meta.loc[valid_mask, 'dw_g'].min():.1f} - "
              f"{test_dataset.plant_meta.loc[valid_mask, 'dw_g'].max():.1f} g")

        exp_results = {"experiment": test_exp, "n_plants": len(valid_indices)}

        # --- Teacher (ensemble over folds for stability) ---
        print(f"\n  --- Teacher (multimodal) ---")
        teacher_preds_all = []
        for fd in fold_dirs:
            ckpt = fd / "best_model_state.pt"
            if not ckpt.exists():
                continue
            teacher = LiquidYieldModel(role="teacher", cfg=cfg_test)
            teacher.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            res = evaluate_model(teacher, test_subset, dw_mean, dw_std, device)
            teacher_preds_all.append(res["preds"])
            if len(teacher_preds_all) >= 10:  # use up to 10 folds for ensemble
                break

        # Ensemble mean
        teacher_preds = np.mean(teacher_preds_all, axis=0)
        teacher_trues = res["trues"]
        tm = compute_metrics(teacher_preds, teacher_trues)
        print(f"  Teacher (ensemble {len(teacher_preds_all)} folds): "
              f"MAE={tm['mae']:.3f}g  R²={tm['r2']:+.3f}  std_ratio={tm['std_ratio']:.2f}")
        exp_results["teacher"] = tm

        # --- Student (image-only, ensemble) ---
        if args.student_dir:
            print(f"\n  --- Student (image-only) ---")
            student_fold_dirs = sorted(Path(args.student_dir).glob("fold_*"))
            student_preds_all = []
            for fd in student_fold_dirs:
                ckpt = fd / "best_student_state.pt"
                if not ckpt.exists():
                    continue
                student = LiquidYieldModel(role="student", cfg=cfg_test)
                student.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
                res_s = evaluate_model(student, test_subset, dw_mean, dw_std, device)
                student_preds_all.append(res_s["preds"])
                if len(student_preds_all) >= 10:
                    break

            student_preds = np.mean(student_preds_all, axis=0)
            sm = compute_metrics(student_preds, teacher_trues)
            print(f"  Student (ensemble {len(student_preds_all)} folds): "
                  f"MAE={sm['mae']:.3f}g  R²={sm['r2']:+.3f}  std_ratio={sm['std_ratio']:.2f}")
            exp_results["student"] = sm

        # --- WHC+Geno baseline ---
        meta = test_dataset.plant_meta[valid_mask].reset_index(drop=True)
        group_means = meta.groupby(["treatment", "genotype"])["dw_g"].mean().to_dict()
        base_preds = meta.apply(
            lambda r: group_means.get((r["treatment"], r["genotype"]), meta["dw_g"].mean()), axis=1
        ).values
        bm = compute_metrics(base_preds, meta["dw_g"].values)
        print(f"  Baseline (WHC+Geno): MAE={bm['mae']:.3f}g  R²={bm['r2']:+.3f}")
        exp_results["baseline"] = bm

        # Per-treatment breakdown
        print(f"\n  Per-treatment Teacher MAE:")
        for treat in sorted(meta["treatment"].unique()):
            mask = meta["treatment"] == treat
            if mask.sum() > 0:
                idx = np.where(mask.values)[0]
                t_mae = float(np.mean(np.abs(teacher_preds[idx] - teacher_trues[idx])))
                print(f"    {treat}: MAE={t_mae:.3f}g (n={mask.sum()})")

        all_results[test_exp] = exp_results

    # Save
    with open(output_dir / "cross_experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    # Summary comparison
    print(f"\n{'='*60}")
    print("CROSS-EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Experiment':<15} {'Teacher MAE':<15} {'Student MAE':<15} {'Baseline MAE':<15}")
    for exp_id, r in all_results.items():
        t_mae = r.get("teacher", {}).get("mae", "-")
        s_mae = r.get("student", {}).get("mae", "-")
        b_mae = r.get("baseline", {}).get("mae", "-")
        t_str = f"{t_mae:.3f}g" if isinstance(t_mae, float) else t_mae
        s_str = f"{s_mae:.3f}g" if isinstance(s_mae, float) else s_mae
        b_str = f"{b_mae:.3f}g" if isinstance(b_mae, float) else b_mae
        print(f"{exp_id:<15} {t_str:<15} {s_str:<15} {b_str:<15}")


if __name__ == "__main__":
    main()
