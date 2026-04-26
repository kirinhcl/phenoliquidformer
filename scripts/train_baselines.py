#!/usr/bin/env python3
"""Classical ML baselines for WHC regression.

Uses time-aggregated tabular features (fluorescence + VI + digital biomass)
with Random Forest, XGBoost, and SVR.

Usage:
    python scripts/train_baselines.py --experiment exp01 --cv lowho
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.data.dataset import TimothyDroughtDataset
from src.training.cv import LeaveOnePlantOutCV, LeaveOneWHCOutCV
from src.utils.config import load_config


def extract_tabular_features(dataset: TimothyDroughtDataset) -> tuple[np.ndarray, np.ndarray]:
    """Extract time-aggregated features from each plant.

    For each modality (fluorescence, VI, digital biomass), compute:
    mean, std, slope (linear trend), min, max over valid timepoints.

    Returns:
        X: (N, n_features) feature matrix
        y: (N,) WHC targets
    """
    all_features = []
    all_targets = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        active = sample["active_mask"].numpy().astype(bool)
        T_active = active.sum()

        features = []

        # Fluorescence: (T, 98) → aggregate
        fluor = sample["fluorescence"].numpy()[active]
        if len(fluor) > 0:
            features.extend(fluor.mean(axis=0).tolist())
            features.extend(fluor.std(axis=0).tolist())
        else:
            features.extend([0.0] * dataset.fluor_dim * 2)

        # Vegetation indices: (T, 11) → aggregate
        vi = sample["vi"].numpy()[active]
        if len(vi) > 0:
            features.extend(vi.mean(axis=0).tolist())
            features.extend(vi.std(axis=0).tolist())
        else:
            features.extend([0.0] * dataset.vi_dim * 2)

        # Digital biomass: (T,) → aggregate + slope
        biomass = sample["digital_biomass"].numpy()[active]
        bm_mask = sample["biomass_mask"].numpy()[active]
        valid_bm = biomass[bm_mask.astype(bool)]
        if len(valid_bm) > 1:
            features.append(float(valid_bm.mean()))
            features.append(float(valid_bm.std()))
            features.append(float(valid_bm.max()))
            features.append(float(valid_bm[-1] - valid_bm[0]))  # growth
            # Slope via linear regression
            x_time = np.arange(len(valid_bm), dtype=np.float64)
            slope = np.polyfit(x_time, valid_bm.astype(np.float64), 1)[0]
            features.append(float(slope))
        else:
            features.extend([0.0] * 5)

        # Environment: (T, 5) → aggregate
        env = sample["environment"].numpy()[active]
        if len(env) > 0:
            features.extend(env.mean(axis=0).tolist())
        else:
            features.extend([0.0] * 5)

        all_features.append(features)
        all_targets.append(sample["whc_target"])

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)

    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y


def run_baseline(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    cv_splits,
) -> dict:
    """Run a baseline model across CV folds."""
    fold_results = []

    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(cv_splits):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if model_name == "rf":
            model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        elif model_name == "xgboost":
            try:
                from xgboost import XGBRegressor
                model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
            except ImportError:
                print("XGBoost not available, skipping.")
                return {}
        elif model_name == "svr":
            model = SVR(kernel="rbf", C=1.0)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        rho, _ = stats.spearmanr(preds, y_test)
        rho = float(rho) if not np.isnan(rho) else 0.0

        fold_results.append({"fold": fold_idx, "mae": mae, "r2": r2, "spearman": rho})

    maes = [r["mae"] for r in fold_results]
    r2s = [r["r2"] for r in fold_results]
    return {
        "model": model_name,
        "n_folds": len(fold_results),
        "mean_mae": float(np.mean(maes)),
        "std_mae": float(np.std(maes)),
        "mean_r2": float(np.mean(r2s)),
        "std_r2": float(np.std(r2s)),
        "folds": fold_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/timothy.yaml")
    parser.add_argument("--experiment", default="exp01")
    parser.add_argument("--cv", default="lowho", choices=["lopo", "lowho"])
    parser.add_argument("--output-dir", default="results/baselines")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=[f"data.experiment={args.experiment}"])
    print(f"Baselines | experiment={args.experiment} | cv={args.cv}")

    dataset = TimothyDroughtDataset(cfg)
    print(f"Extracting tabular features for {len(dataset)} plants...")
    X, y = extract_tabular_features(dataset)
    print(f"Feature matrix: {X.shape}, targets: {y.shape}")

    plant_meta = dataset.plant_meta
    if args.cv == "lopo":
        cv = LeaveOnePlantOutCV(plant_meta, args.experiment, seed=42)
    else:
        cv = LeaveOneWHCOutCV(plant_meta, args.experiment, seed=42)

    output_dir = Path(args.output_dir) / f"{args.experiment}_{args.cv}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for model_name in ["rf", "svr", "xgboost"]:
        print(f"\n--- {model_name.upper()} ---")
        # Need to regenerate CV splits each time
        if args.cv == "lopo":
            cv = LeaveOnePlantOutCV(plant_meta, args.experiment, seed=42)
        else:
            cv = LeaveOneWHCOutCV(plant_meta, args.experiment, seed=42)
        result = run_baseline(model_name, X, y, cv.split())
        if result:
            print(f"  MAE: {result['mean_mae']:.4f} +/- {result['std_mae']:.4f}")
            print(f"  R2:  {result['mean_r2']:.3f} +/- {result['std_r2']:.3f}")
            all_results.append(result)

    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'baseline_results.json'}")


if __name__ == "__main__":
    main()
