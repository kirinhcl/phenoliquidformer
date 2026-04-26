#!/usr/bin/env python3
"""Cross-experiment training: train on one experiment, test on another.

Addresses Q6: Can a model trained on non-vernalized plants generalize
to vernalized/regrowth plants?

Usage:
    python scripts/train_cross_experiment.py --train-exp exp01 --test-exp exp02
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader, Subset

from src.data.dataset import TimothyDroughtDataset
from src.model.timothy_model import TimothyDroughtModel
from src.training.trainer import Trainer
from src.utils.config import load_config


def collate_fn(batch: list[dict]) -> dict:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/timothy.yaml")
    parser.add_argument("--train-exp", required=True, choices=["exp01", "exp02", "exp03"])
    parser.add_argument("--test-exp", required=True, choices=["exp01", "exp02", "exp03"])
    parser.add_argument("--output-dir", default="results/cross_experiment")
    args = parser.parse_args()

    print(f"Cross-experiment: train on {args.train_exp}, test on {args.test_exp}")

    # Load train dataset
    cfg_train = load_config(args.config, overrides=[f"data.experiment={args.train_exp}"])
    train_dataset = TimothyDroughtDataset(cfg_train)
    print(f"Train dataset ({args.train_exp}): {len(train_dataset)} plants, max_T={train_dataset.max_T}")

    # Load test dataset
    cfg_test = load_config(args.config, overrides=[f"data.experiment={args.test_exp}"])
    test_dataset = TimothyDroughtDataset(cfg_test)
    print(f"Test dataset ({args.test_exp}): {len(test_dataset)} plants, max_T={test_dataset.max_T}")

    # Split train into train/val (85/15)
    rng = np.random.RandomState(42)
    n = len(train_dataset)
    indices = rng.permutation(n)
    n_val = max(1, int(n * 0.15))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_loader = DataLoader(
        Subset(train_dataset, train_idx.tolist()),
        batch_size=cfg_train.training.batch_size, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        Subset(train_dataset, val_idx.tolist()),
        batch_size=cfg_train.training.batch_size, shuffle=False, collate_fn=collate_fn,
    )

    model = TimothyDroughtModel(cfg_train)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        cfg=cfg_train, fold_id=0, checkpoint_dir=output_dir,
    )

    result = trainer.train()
    print(f"Training done. Best val MAE: {result['best_val_mae']:.4f}")

    # Evaluate on test experiment
    model.load_state_dict(
        torch.load(output_dir / "best_model_state.pt", map_location=trainer.device, weights_only=True)
    )
    model.to(trainer.device)
    model.train(False)

    test_loader = DataLoader(
        test_dataset, batch_size=cfg_test.training.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    all_preds = []
    all_targets = []
    all_plants = []
    all_gates = []

    with torch.no_grad():
        for batch in test_loader:
            batch_dev = {
                k: v.to(trainer.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # Handle max_T mismatch: pad or truncate temporal tensors
            test_T = batch_dev["images"].shape[1]
            model_T = train_dataset.max_T
            if test_T != model_T:
                for key in ["images", "image_mask", "fluorescence", "fluor_mask",
                            "environment", "vi", "temporal_positions", "active_mask",
                            "digital_biomass", "biomass_mask"]:
                    t = batch_dev[key]
                    if t.dim() >= 2 and t.shape[1] == test_T:
                        if test_T < model_T:
                            pad_shape = list(t.shape)
                            pad_shape[1] = model_T - test_T
                            pad = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
                            batch_dev[key] = torch.cat([t, pad], dim=1)
                        else:
                            batch_dev[key] = t[:, :model_T]

            outputs = model(batch_dev)
            all_preds.extend(outputs["whc_pred"].cpu().tolist())
            all_targets.extend(batch["whc_target"].tolist())
            all_plants.extend(batch["plant_id"])

            gates = outputs["modality_gates"].cpu().numpy()
            active = batch_dev["active_mask"][:, :model_T].cpu().numpy()
            for b in range(gates.shape[0]):
                valid = active[b].astype(bool)
                mean_g = gates[b][valid].mean(axis=0) if valid.any() else gates[b].mean(axis=0)
                all_gates.append(mean_g)

    preds = np.array(all_preds)
    targets = np.array(all_targets)
    mae = float(np.mean(np.abs(preds - targets)))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-8))
    rho, p_rho = stats.spearmanr(preds, targets)

    mean_gates = np.mean(all_gates, axis=0)

    print(f"\n=== Cross-experiment: {args.train_exp} -> {args.test_exp} ===")
    print(f"  Test plants: {len(preds)}")
    print(f"  MAE:      {mae:.4f}")
    print(f"  R2:       {r2:.4f}")
    print(f"  Spearman: {float(rho):.4f} (p={float(p_rho):.4e})")
    print(f"  Gates: Image={mean_gates[0]:.3f} Fluor={mean_gates[1]:.3f} "
          f"Env={mean_gates[2]:.3f} VI={mean_gates[3]:.3f}")

    cross_result = {
        "train_exp": args.train_exp,
        "test_exp": args.test_exp,
        "n_train": len(train_dataset),
        "n_test": len(test_dataset),
        "test_mae": mae,
        "test_r2": r2,
        "test_spearman": float(rho),
        "gate_weights": {
            "image": float(mean_gates[0]), "fluor": float(mean_gates[1]),
            "env": float(mean_gates[2]), "vi": float(mean_gates[3]),
        },
        "per_plant": [
            {"plant_id": p, "whc_target": float(t), "whc_pred": float(pr)}
            for p, t, pr in zip(all_plants, targets, preds)
        ],
    }
    with open(output_dir / "cross_experiment_results.json", "w") as f:
        json.dump(cross_result, f, indent=2)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
