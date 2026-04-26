#!/usr/bin/env python3
"""Classical ML baselines for DW yield prediction.

Baselines:
    1. Group-mean: WHC+Genotype group mean (zero-param lookup)
    2. Linear Regression: on time-aggregated features
    3. Ridge Regression: L2-regularized linear
    4. Random Forest
    5. XGBoost
    6. SVR

All use LOPO CV with honest per-fold group mean computation.
Features: time-aggregated (mean, std, slope, max) of fluor + VI + digital biomass.

Usage:
    python scripts/train_yield_baselines.py --experiment exp01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.data.dataset import TimothyDroughtDataset
from src.training.cv import LeaveOnePlantOutCV
from src.utils.config import load_config


def extract_tabular_features(
    dataset: TimothyDroughtDataset,
    use_vi: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract time-aggregated features for each plant.

    Args:
        dataset: TimothyDroughtDataset instance.
        use_vi: If False, skip VI feature extraction for no-VI ablation.

    Returns X (N, n_features), y (N,), feature_names
    """
    all_features = []
    all_targets = []
    feature_names = None

    for idx in range(len(dataset)):
        sample = dataset[idx]
        active = sample["active_mask"].numpy().astype(bool)

        features = {}

        # Fluorescence: (T, 98) -> mean, std
        fluor = sample["fluorescence"].numpy()[active]
        if len(fluor) > 0:
            for stat_name, stat_fn in [("mean", np.mean), ("std", np.std)]:
                vals = stat_fn(fluor, axis=0)
                for i, v in enumerate(vals):
                    features[f"fluor_{i}_{stat_name}"] = float(v)
        else:
            for stat_name in ["mean", "std"]:
                for i in range(dataset.fluor_dim):
                    features[f"fluor_{i}_{stat_name}"] = 0.0

        # VI: (T, 11) -> mean, std, slope
        if use_vi:
            vi = sample["vi"].numpy()[active]
            if len(vi) > 1:
                for i in range(dataset.vi_dim):
                    features[f"vi_{i}_mean"] = float(vi[:, i].mean())
                    features[f"vi_{i}_std"] = float(vi[:, i].std())
                    x_t = np.arange(len(vi), dtype=float)
                    slope = np.polyfit(x_t, vi[:, i], 1)[0]
                    features[f"vi_{i}_slope"] = float(slope)
            elif len(vi) == 1:
                for i in range(dataset.vi_dim):
                    features[f"vi_{i}_mean"] = float(vi[0, i])
                    features[f"vi_{i}_std"] = 0.0
                    features[f"vi_{i}_slope"] = 0.0
            else:
                for i in range(dataset.vi_dim):
                    for s in ["mean", "std", "slope"]:
                        features[f"vi_{i}_{s}"] = 0.0

        # Digital biomass: EXCLUDED — it's computed from RGB area
        # and is a near-direct proxy for DW. Including it would be
        # an unfair advantage over models that don't use it.

        # WHC as feature
        features["whc"] = float(sample["whc_target"])

        # Genotype as 0/1
        features["is_noreng"] = 1.0 if "Noreng" in sample["genotype"] else 0.0

        if feature_names is None:
            feature_names = sorted(features.keys())

        row = [features[k] for k in feature_names]
        all_features.append(row)
        all_targets.append(float(sample["dw_target"]) if not np.isnan(sample["dw_target"]) else 0.0)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, feature_names


def run_baseline(model_name, X, y, cv_splits, plant_meta):
    """Run one baseline model across LOPO folds."""
    fold_preds = []
    fold_trues = []

    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(cv_splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if model_name == "linear":
            model = LinearRegression()
        elif model_name == "ridge":
            model = Ridge(alpha=1.0)
        elif model_name == "rf":
            model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
        elif model_name == "xgboost":
            try:
                from xgboost import XGBRegressor
                model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
            except ImportError:
                return None
        elif model_name == "svr":
            model = SVR(kernel="rbf", C=1.0)
        elif model_name == "group_mean":
            # Simple group mean baseline
            train_df = plant_meta.iloc[train_idx]
            test_row = plant_meta.iloc[test_idx[0]]
            group_means = train_df.groupby(["treatment", "genotype"])["dw_g"].mean().to_dict()
            overall_mean = train_df["dw_g"].mean()
            key = (test_row["treatment"], test_row["genotype"])
            pred = group_means.get(key, overall_mean)
            fold_preds.append(pred)
            fold_trues.append(y_test[0])
            continue
        else:
            return None

        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        fold_preds.extend(pred.tolist())
        fold_trues.extend(y_test.tolist())

    preds = np.array(fold_preds)
    trues = np.array(fold_trues)
    mae = float(np.mean(np.abs(preds - trues)))
    ss_res = float(np.sum((trues - preds) ** 2))
    ss_tot = float(np.sum((trues - trues.mean()) ** 2))
    r2 = 1 - ss_res / max(ss_tot, 1e-8)
    std_ratio = float(preds.std() / max(trues.std(), 1e-8))

    return {"model": model_name, "mae": mae, "r2": r2, "std_ratio": std_ratio}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/timothy.yaml")
    parser.add_argument("--experiment", default="exp01")
    parser.add_argument("--output-dir", default="results/yield_baselines")
    parser.add_argument("--no-vi", action="store_true", help="Exclude vegetation indices from features")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=[f"data.experiment={args.experiment}"])
    use_vi = not args.no_vi
    print(f"Yield baselines on {args.experiment} (use_vi={use_vi})", flush=True)

    dataset = TimothyDroughtDataset(cfg)
    plant_meta = dataset.plant_meta
    valid_mask = plant_meta["dw_g"].notna().values
    plant_meta_valid = plant_meta[valid_mask].reset_index(drop=True)

    print(f"Extracting features for {len(dataset)} plants...")
    X, y, feature_names = extract_tabular_features(dataset, use_vi=use_vi)
    print(f"Feature matrix: {X.shape} ({len(feature_names)} features)")

    output_dir = Path(args.output_dir) / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    baselines = ["group_mean", "linear", "ridge", "rf", "svr", "xgboost"]
    all_results = []

    for model_name in baselines:
        cv = LeaveOnePlantOutCV(plant_meta_valid, args.experiment, seed=42)
        result = run_baseline(model_name, X, y, cv.split(), plant_meta_valid)
        if result:
            print(f"  {result['model']:<15} MAE={result['mae']:.3f}g  "
                  f"R²={result['r2']:+.3f}  std_ratio={result['std_ratio']:.2f}", flush=True)
            all_results.append(result)

    # Add Liquid results for reference
    liquid_teacher_path = Path("results/liquid_teacher") / args.experiment / "summary.json"
    if liquid_teacher_path.exists():
        with open(liquid_teacher_path) as f:
            liq = json.load(f)
        all_results.append({
            "model": "liquid_teacher",
            "mae": liq["mean_test_mae_g"],
            "r2": liq["r2_global"],
            "std_ratio": liq["std_ratio"],
        })
    liquid_student_path = Path("results/liquid_student") / args.experiment / "summary.json"
    if liquid_student_path.exists():
        with open(liquid_student_path) as f:
            liq_s = json.load(f)
        tc_full = liq_s["by_tcut"]["None"]
        all_results.append({
            "model": "liquid_student",
            "mae": tc_full["pooled_mae_g"],
            "r2": tc_full["r2_global"],
            "std_ratio": tc_full["std_ratio"],
        })

    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"YIELD PREDICTION COMPARISON ({args.experiment})")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'MAE (g)':<12} {'R²':<12} {'std_ratio':<12}")
    print("-" * 56)
    for r in sorted(all_results, key=lambda x: x["mae"]):
        print(f"{r['model']:<20} {r['mae']:<12.3f} {r['r2']:<+12.3f} {r['std_ratio']:<12.2f}")

    print(f"\nResults saved to {output_dir / 'baseline_results.json'}")


if __name__ == "__main__":
    main()
