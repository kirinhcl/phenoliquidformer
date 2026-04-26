#!/usr/bin/env python3
"""CEA-level statistical comparison of all yield prediction methods.

Computes:
    1. Per-fold MAE for each method (48 paired observations)
    2. Pairwise Wilcoxon signed-rank tests
    3. Friedman test across all methods
    4. Nemenyi post-hoc with critical difference diagram
    5. Bootstrap 95% CI for pooled MAE and R²
    6. Additional metrics: RMSE, MAPE
    7. Scatter plots (true vs predicted) for key methods

Usage:
    python scripts/statistical_tests.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.linewidth": 0.8, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

OUT = Path("results/statistical_tests")
OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT = Path("paper/figures/output")
FIG_OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Collect fold-level predictions from all methods
# ============================================================

def load_dl_fold_preds(results_dir, pred_key="test_preds", true_key="test_trues"):
    """Load per-fold predictions from DL results."""
    preds_per_fold = []
    trues_per_fold = []
    for fd in sorted(Path(results_dir).glob("fold_*")):
        mp = fd / "metrics.json"
        if not mp.exists():
            continue
        with open(mp) as f:
            m = json.load(f)
        if pred_key in m:
            preds_per_fold.append(m[pred_key][0] if len(m[pred_key]) == 1 else m[pred_key])
            trues_per_fold.append(m[true_key][0] if len(m[true_key]) == 1 else m[true_key])
        elif "test_by_tcut" in m:
            tc = m["test_by_tcut"]["None"]
            preds_per_fold.append(tc["preds"][0] if len(tc["preds"]) == 1 else tc["preds"])
            trues_per_fold.append(tc["trues"][0] if len(tc["trues"]) == 1 else tc["trues"])
    # Flatten if nested
    flat_preds = [p if isinstance(p, (int, float)) else p[0] for p in preds_per_fold]
    flat_trues = [t if isinstance(t, (int, float)) else t[0] for t in trues_per_fold]
    return np.array(flat_preds), np.array(flat_trues)


def compute_classical_fold_preds(dataset_module):
    """Recompute classical ML fold-level predictions via LOPO."""
    from src.data.dataset import TimothyDroughtDataset
    from src.training.cv import LeaveOnePlantOutCV
    from src.utils.config import load_config
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    cfg = load_config("configs/timothy.yaml", overrides=["data.experiment=exp01"])
    dataset = TimothyDroughtDataset(cfg)
    plant_meta = dataset.plant_meta
    valid_mask = plant_meta["dw_g"].notna().values
    pm = plant_meta[valid_mask].reset_index(drop=True)

    # Extract features (same as train_yield_baselines.py, no biomass)
    X_rows = []
    y_vals = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        active = sample["active_mask"].numpy().astype(bool)
        features = []
        fluor = sample["fluorescence"].numpy()[active]
        if len(fluor) > 0:
            features.extend(fluor.mean(axis=0).tolist())
            features.extend(fluor.std(axis=0).tolist())
        else:
            features.extend([0.0] * dataset.fluor_dim * 2)
        vi = sample["vi"].numpy()[active]
        if len(vi) > 1:
            for i in range(dataset.vi_dim):
                features.append(float(vi[:, i].mean()))
                features.append(float(vi[:, i].std()))
                x_t = np.arange(len(vi), dtype=float)
                features.append(float(np.polyfit(x_t, vi[:, i], 1)[0]))
        else:
            features.extend([0.0] * dataset.vi_dim * 3)
        features.append(float(sample["whc_target"]))
        features.append(1.0 if "Noreng" in sample["genotype"] else 0.0)
        X_rows.append(features)
        y_vals.append(float(sample["dw_target"]) if not np.isnan(sample["dw_target"]) else 0.0)

    X = np.nan_to_num(np.array(X_rows, dtype=np.float32))
    y = np.array(y_vals, dtype=np.float32)

    cv = LeaveOnePlantOutCV(pm, "exp01", seed=42)
    results = {}
    models_cfg = {
        "Ridge": lambda: Ridge(alpha=1.0),
        "RF": lambda: RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
        "SVR": lambda: SVR(kernel="rbf", C=1.0),
    }
    try:
        from xgboost import XGBRegressor
        models_cfg["XGBoost"] = lambda: XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
    except ImportError:
        pass

    for name, model_fn in models_cfg.items():
        fold_preds, fold_trues = [], []
        cv_iter = LeaveOnePlantOutCV(pm, "exp01", seed=42)
        for train_idx, val_idx, test_idx in cv_iter.split():
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            model = model_fn()
            model.fit(X_tr, y[train_idx])
            pred = model.predict(X_te)
            fold_preds.extend(pred.tolist())
            fold_trues.extend(y[test_idx].tolist())
        results[name] = (np.array(fold_preds), np.array(fold_trues))

    # Group-mean baseline
    gm_preds, gm_trues = [], []
    cv_iter = LeaveOnePlantOutCV(pm, "exp01", seed=42)
    for train_idx, val_idx, test_idx in cv_iter.split():
        train_df = pm.iloc[train_idx]
        test_row = pm.iloc[test_idx[0]]
        group_means = train_df.groupby(["treatment", "genotype"])["dw_g"].mean().to_dict()
        overall_mean = train_df["dw_g"].mean()
        key = (test_row["treatment"], test_row["genotype"])
        gm_preds.append(group_means.get(key, overall_mean))
        gm_trues.append(test_row["dw_g"])
    results["Group-mean"] = (np.array(gm_preds), np.array(gm_trues))

    return results


def compute_metrics(preds, trues):
    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    mape = float(np.mean(np.abs(preds - trues) / np.maximum(np.abs(trues), 1e-8)) * 100)
    ss_res = np.sum((trues - preds) ** 2)
    ss_tot = np.sum((trues - trues.mean()) ** 2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-8))
    std_ratio = float(preds.std() / max(trues.std(), 1e-8))
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2, "std_ratio": std_ratio}


def bootstrap_ci(preds, trues, metric_fn, n_boot=10000, alpha=0.05):
    """Bootstrap 95% CI for a metric."""
    rng = np.random.RandomState(42)
    n = len(preds)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        boot_vals.append(metric_fn(preds[idx], trues[idx]))
    lower = np.percentile(boot_vals, 100 * alpha / 2)
    upper = np.percentile(boot_vals, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def main():
    print("=" * 70)
    print("CEA-LEVEL STATISTICAL ANALYSIS")
    print("=" * 70)

    # Load all DL predictions
    all_methods = {}

    # Liquid Teacher
    p, t = load_dl_fold_preds("results/liquid_teacher/exp01")
    if len(p) == 48:
        all_methods["Liquid Teacher"] = (p, t)

    # Transformer
    p, t = load_dl_fold_preds("results/yield_transformer/exp01")
    if len(p) == 48:
        all_methods["Transformer"] = (p, t)

    # Liquid Student
    p, t = load_dl_fold_preds("results/liquid_student/exp01")
    if len(p) == 48:
        all_methods["Liquid Student"] = (p, t)

    # Neural ODE
    p, t = load_dl_fold_preds("results/yield_teacher/exp01")
    if len(p) == 48:
        all_methods["Neural ODE"] = (p, t)

    # Classical ML
    print("\nComputing classical ML fold predictions...")
    classical = compute_classical_fold_preds(None)
    all_methods.update(classical)

    print(f"\nLoaded {len(all_methods)} methods")
    for name, (p, t) in all_methods.items():
        print(f"  {name}: {len(p)} predictions")

    # ============================================================
    # 2. Comprehensive metrics table
    # ============================================================
    print("\n" + "=" * 70)
    print("COMPREHENSIVE METRICS")
    print("=" * 70)

    metrics_table = []
    for name, (preds, trues) in all_methods.items():
        m = compute_metrics(preds, trues)
        # Bootstrap CI for MAE
        mae_lo, mae_hi = bootstrap_ci(preds, trues,
                                       lambda p, t: np.mean(np.abs(p - t)))
        r2_lo, r2_hi = bootstrap_ci(preds, trues,
                                     lambda p, t: 1 - np.sum((t - p) ** 2) / max(np.sum((t - t.mean()) ** 2), 1e-8))
        m["mae_ci"] = f"[{mae_lo:.2f}, {mae_hi:.2f}]"
        m["r2_ci"] = f"[{r2_lo:.2f}, {r2_hi:.2f}]"
        m["name"] = name
        metrics_table.append(m)

    metrics_df = pd.DataFrame(metrics_table).sort_values("mae")
    print(f"\n{'Model':<20} {'MAE(g)':<10} {'MAE 95%CI':<18} {'RMSE':<10} {'R²':<10} {'R² 95%CI':<18} {'MAPE%':<10}")
    print("-" * 96)
    for _, r in metrics_df.iterrows():
        print(f"{r['name']:<20} {r['mae']:<10.3f} {r['mae_ci']:<18} {r['rmse']:<10.3f} "
              f"{r['r2']:<+10.3f} {r['r2_ci']:<18} {r['mape']:<10.1f}")

    metrics_df.to_csv(OUT / "comprehensive_metrics.csv", index=False)

    # ============================================================
    # 3. Pairwise Wilcoxon signed-rank tests
    # ============================================================
    print("\n" + "=" * 70)
    print("PAIRWISE WILCOXON SIGNED-RANK TESTS (on fold-level |error|)")
    print("=" * 70)

    method_names = list(all_methods.keys())
    n_methods = len(method_names)
    fold_errors = {}
    for name, (preds, trues) in all_methods.items():
        fold_errors[name] = np.abs(preds - trues)

    wilcoxon_results = []
    print(f"\n{'Method A':<20} {'Method B':<20} {'p-value':<12} {'Significant?'}")
    print("-" * 64)
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            a, b = method_names[i], method_names[j]
            try:
                stat, p = stats.wilcoxon(fold_errors[a], fold_errors[b])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            except ValueError:
                p = 1.0
                sig = "(tied)"
            wilcoxon_results.append({"method_a": a, "method_b": b, "p_value": p})
            if p < 0.1:  # only print interesting ones
                print(f"{a:<20} {b:<20} {p:<12.4e} {sig}")

    pd.DataFrame(wilcoxon_results).to_csv(OUT / "wilcoxon_pairwise.csv", index=False)

    # ============================================================
    # 4. Friedman test
    # ============================================================
    print("\n" + "=" * 70)
    print("FRIEDMAN TEST (all methods)")
    print("=" * 70)

    error_matrix = np.column_stack([fold_errors[name] for name in method_names])
    try:
        friedman_stat, friedman_p = stats.friedmanchisquare(*[error_matrix[:, i] for i in range(n_methods)])
        print(f"Friedman chi²={friedman_stat:.2f}, p={friedman_p:.2e}")
        if friedman_p < 0.05:
            print("Significant — methods differ in performance")
        else:
            print("Not significant — no evidence of difference")
    except Exception as e:
        print(f"Friedman test failed: {e}")

    # ============================================================
    # 5. Critical Difference Diagram (Nemenyi)
    # ============================================================
    print("\n" + "=" * 70)
    print("AVERAGE RANKS (for Nemenyi CD diagram)")
    print("=" * 70)

    # Rank methods per fold (1 = best)
    ranks = np.zeros_like(error_matrix)
    for i in range(error_matrix.shape[0]):
        ranks[i] = stats.rankdata(error_matrix[i])
    avg_ranks = ranks.mean(axis=0)

    rank_order = np.argsort(avg_ranks)
    print(f"\n{'Rank':<6} {'Model':<20} {'Avg Rank':<10}")
    for pos, idx in enumerate(rank_order):
        print(f"{pos+1:<6} {method_names[idx]:<20} {avg_ranks[idx]:<10.2f}")

    # Save for external CD diagram tool
    rank_df = pd.DataFrame({
        "model": [method_names[i] for i in rank_order],
        "avg_rank": [avg_ranks[i] for i in rank_order],
    })
    rank_df.to_csv(OUT / "average_ranks.csv", index=False)

    # ============================================================
    # 6. Scatter plots (true vs predicted)
    # ============================================================
    print("\nGenerating scatter plots...")

    key_methods = ["Liquid Teacher", "Transformer", "Liquid Student", "RF", "Group-mean"]
    key_methods = [m for m in key_methods if m in all_methods]

    fig, axes = plt.subplots(1, len(key_methods), figsize=(4 * len(key_methods), 4))
    if len(key_methods) == 1:
        axes = [axes]

    colors = {"Liquid Teacher": "#e6550d", "Transformer": "#756bb1",
              "Liquid Student": "#31a354", "RF": "#636363", "Group-mean": "#969696"}

    for ax, name in zip(axes, key_methods):
        preds, trues = all_methods[name]
        m = compute_metrics(preds, trues)
        c = colors.get(name, "#333")

        ax.scatter(trues, preds, c=c, s=30, alpha=0.7, edgecolors="white", linewidth=0.5)
        # Perfect prediction line
        lims = [min(trues.min(), preds.min()) - 1, max(trues.max(), preds.max()) + 1]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
        # Regression line
        z = np.polyfit(trues, preds, 1)
        x_line = np.linspace(lims[0], lims[1], 100)
        ax.plot(x_line, np.polyval(z, x_line), c=c, lw=1.5, alpha=0.8)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("True DW (g)")
        ax.set_ylabel("Predicted DW (g)")
        ax.set_title(f"{name}\nMAE={m['mae']:.2f}g  R²={m['r2']:.2f}", fontsize=9)
        ax.set_aspect("equal")

    fig.suptitle("True vs Predicted Dry Weight (48-fold LOPO)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "scatter_true_vs_pred.pdf")
    fig.savefig(FIG_OUT / "scatter_true_vs_pred.png", dpi=150)
    plt.close(fig)
    print(f"  Scatter plots saved")

    print(f"\nAll results saved to {OUT}")
    print("Done!")


if __name__ == "__main__":
    main()
