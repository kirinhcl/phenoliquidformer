"""Main training script for Timothy drought model.

Usage:
    python scripts/train_timothy.py [--config configs/timothy.yaml] [--cv lopo|lowho]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import TimothyDroughtDataset
from src.model.timothy_model import TimothyDroughtModel
from src.training.cv import LeaveOnePlantOutCV, LeaveOneWHCOutCV
from src.training.trainer import Trainer
from src.utils.config import load_config


def collate_fn(batch: list[dict]) -> dict:
    """Collate batch of samples, stacking tensors and collecting metadata."""
    result = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values, dtype=torch.float32)
        else:
            result[key] = values  # strings
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/timothy.yaml")
    parser.add_argument("--cv", default="lowho", choices=["lopo", "lowho"])
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment = cfg.data.experiment
    print(f"Training Timothy model | experiment={experiment} | cv={args.cv}")

    # Load dataset
    dataset = TimothyDroughtDataset(cfg)
    plant_meta = dataset.plant_meta
    print(f"Dataset: {len(dataset)} plants, max_T={dataset.max_T}")

    # Set up CV
    if args.cv == "lopo":
        cv = LeaveOnePlantOutCV(plant_meta, experiment, seed=cfg.training.seed)
    else:
        cv = LeaveOneWHCOutCV(plant_meta, experiment, seed=cfg.training.seed)

    n_folds = args.max_folds or cv.n_folds
    print(f"CV strategy: {args.cv}, {n_folds} folds")

    output_dir = Path(args.output_dir) / f"{experiment}_{args.cv}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(cv.split()):
        if fold_idx >= n_folds:
            break

        print(f"\n--- Fold {fold_idx} ---")
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        train_loader = DataLoader(
            Subset(dataset, train_idx.tolist()),
            batch_size=cfg.training.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx.tolist()),
            batch_size=cfg.training.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # Create model
        model = TimothyDroughtModel(cfg)
        checkpoint_dir = output_dir / f"fold_{fold_idx}"

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            fold_id=fold_idx,
            checkpoint_dir=checkpoint_dir,
        )

        result = trainer.train()

        # Evaluate on test set
        model.load_state_dict(
            torch.load(checkpoint_dir / "best_model_state.pt", weights_only=True)
        )
        test_loader = DataLoader(
            Subset(dataset, test_idx.tolist()),
            batch_size=cfg.training.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        test_metrics = trainer._validate.__wrapped__(trainer)  # noqa: reuse validate
        # Actually just call it properly by swapping the loader
        trainer.val_loader = test_loader
        test_metrics = trainer._validate()
        result["test_metrics"] = test_metrics
        print(
            f"  Test: MAE={test_metrics['whc_mae']:.4f}, "
            f"R2={test_metrics.get('r2', 0):.3f}"
        )

        all_results.append(result)

    # Summary
    test_maes = [r["test_metrics"]["whc_mae"] for r in all_results]
    test_r2s = [r["test_metrics"].get("r2", 0) for r in all_results]
    print(f"\n=== Summary ({args.cv}, {len(all_results)} folds) ===")
    print(f"  MAE: {np.mean(test_maes):.4f} +/- {np.std(test_maes):.4f}")
    print(f"  R2:  {np.mean(test_r2s):.3f} +/- {np.std(test_r2s):.3f}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(
            {
                "cv": args.cv,
                "experiment": experiment,
                "n_folds": len(all_results),
                "mean_mae": float(np.mean(test_maes)),
                "std_mae": float(np.std(test_maes)),
                "mean_r2": float(np.mean(test_r2s)),
                "std_r2": float(np.std(test_r2s)),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
