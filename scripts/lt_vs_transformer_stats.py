"""
Statistical comparison: Liquid Transformer vs Pure Transformer
48-fold Leave-One-Plant-Out CV on exp01

Tests:
  - Wilcoxon signed-rank test (paired, two-sided)  [primary]
  - Paired t-test                                   [secondary]
  - Cohen's d effect size

Models compared:
  - Liquid Transformer: results/lt_h64_n2_res/exp01  (h=64, n_layers=2, residual)
  - Pure Transformer:   results/yield_transformer/exp01
"""

import json
import pathlib

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_ROOT = pathlib.Path(__file__).parent.parent / "results"
LT_DIR       = RESULTS_ROOT / "lt_h64_n2_res"    / "exp01"
TRANS_DIR    = RESULTS_ROOT / "yield_transformer" / "exp01"
OUTPUT_FILE  = RESULTS_ROOT / "lt_vs_transformer_stats.json"


# ---------------------------------------------------------------------------
# Load per-fold test MAE
# ---------------------------------------------------------------------------
def load_fold_maes(exp_dir: pathlib.Path) -> list[float]:
    """Return per-fold test_mae_g values sorted by fold index."""
    fold_dirs = sorted(
        [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    maes = []
    for fold_dir in fold_dirs:
        metrics_path = fold_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
        with metrics_path.open() as f:
            metrics = json.load(f)
        maes.append(float(metrics["test_mae_g"]))
    return maes


lt_maes    = np.array(load_fold_maes(LT_DIR))
trans_maes = np.array(load_fold_maes(TRANS_DIR))

assert len(lt_maes) == len(trans_maes), (
    f"Fold count mismatch: LT={len(lt_maes)}, Transformer={len(trans_maes)}"
)
n_folds = len(lt_maes)


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------
lt_mean    = float(lt_maes.mean())
lt_std     = float(lt_maes.std(ddof=1))
trans_mean = float(trans_maes.mean())
trans_std  = float(trans_maes.std(ddof=1))

diff      = lt_maes - trans_maes  # positive -> LT worse
diff_mean = float(diff.mean())
diff_std  = float(diff.std(ddof=1))


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test (primary, two-sided)
# ---------------------------------------------------------------------------
wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
    lt_maes, trans_maes, alternative="two-sided"
)

# ---------------------------------------------------------------------------
# Paired t-test (secondary, two-sided)
# ---------------------------------------------------------------------------
ttest_stat, ttest_p = stats.ttest_rel(
    lt_maes, trans_maes, alternative="two-sided"
)

# ---------------------------------------------------------------------------
# Cohen's d  (paired: mean(diff) / std(diff))
# ---------------------------------------------------------------------------
cohens_d = diff_mean / diff_std if diff_std > 0 else float("nan")


# ---------------------------------------------------------------------------
# Significance label
# ---------------------------------------------------------------------------
def sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
SEP = "-" * 70

print(SEP)
print("  Statistical Comparison: Liquid Transformer vs Pure Transformer  (exp01 LOPO)")
print(SEP)
print(f"  Folds: {n_folds}")
print()
print(f"  {'Model':<30}  {'Mean MAE (g)':>13}  {'Std MAE (g)':>12}")
print(f"  {'-'*30}  {'-'*13}  {'-'*12}")
print(f"  {'Liquid Transformer (LT)':<30}  {lt_mean:>13.4f}  {lt_std:>12.4f}")
print(f"  {'Pure Transformer':<30}  {trans_mean:>13.4f}  {trans_std:>12.4f}")
print(f"  {'Difference (LT - Trans)':<30}  {diff_mean:>+13.4f}  {diff_std:>12.4f}")
print()
print(f"  {'Test':<32}  {'Statistic':>10}  {'p-value':>10}  {'Sig':>4}")
print(f"  {'-'*32}  {'-'*10}  {'-'*10}  {'-'*4}")
print(f"  {'Wilcoxon signed-rank (primary)':<32}  "
      f"{wilcoxon_stat:>10.4f}  {wilcoxon_p:>10.4f}  {sig_label(wilcoxon_p):>4}")
print(f"  {'Paired t-test (secondary)':<32}  "
      f"{ttest_stat:>10.4f}  {ttest_p:>10.4f}  {sig_label(ttest_p):>4}")
print()

effect_dir = "LT better" if cohens_d < 0 else "LT worse"
print(f"  Cohen's d (paired):  {cohens_d:+.4f}  ({effect_dir})")
print()

if diff_mean < 0:
    interp = f"Liquid Transformer is BETTER by {abs(diff_mean):.4f} g mean MAE."
elif diff_mean > 0:
    interp = f"Liquid Transformer is WORSE by {diff_mean:.4f} g mean MAE."
else:
    interp = "Models are identical on average."
print(f"  Interpretation: {interp}")
print(SEP)


# ---------------------------------------------------------------------------
# Save JSON results
# ---------------------------------------------------------------------------
output = {
    "experiment": "exp01",
    "cv": "leave_one_plant_out_48fold",
    "n_folds": n_folds,
    "liquid_transformer": {
        "results_dir": str(LT_DIR),
        "mean_test_mae_g": lt_mean,
        "std_test_mae_g": lt_std,
        "per_fold_mae_g": lt_maes.tolist(),
    },
    "pure_transformer": {
        "results_dir": str(TRANS_DIR),
        "mean_test_mae_g": trans_mean,
        "std_test_mae_g": trans_std,
        "per_fold_mae_g": trans_maes.tolist(),
    },
    "difference_lt_minus_transformer": {
        "mean_g": diff_mean,
        "std_g": diff_std,
    },
    "wilcoxon_signed_rank": {
        "statistic": float(wilcoxon_stat),
        "p_value": float(wilcoxon_p),
        "significance": sig_label(wilcoxon_p),
    },
    "paired_t_test": {
        "statistic": float(ttest_stat),
        "p_value": float(ttest_p),
        "significance": sig_label(ttest_p),
    },
    "cohens_d": float(cohens_d),
    "notes": (
        "LT = lt_h64_n2_res (Liquid Transformer, h=64, n_layers=2, residual). "
        "Positive diff means LT is worse than pure Transformer."
    ),
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_FILE.open("w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Results saved to: {OUTPUT_FILE}\n")
