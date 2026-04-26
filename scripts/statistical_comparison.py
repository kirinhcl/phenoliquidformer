"""
Statistical comparison: Phenology Liquid NN vs Baseline Liquid NN
48-fold Leave-One-Plant-Out CV on exp01

Tests:
  - Wilcoxon signed-rank test (paired, two-sided)  [primary]
  - Paired t-test                                   [secondary]
  - Cohen's d effect size
"""

import json
import pathlib

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_ROOT  = pathlib.Path(__file__).parent.parent / "results"
PHENOLOGY_DIR = RESULTS_ROOT / "phenology_teacher" / "exp01"
BASELINE_DIR  = RESULTS_ROOT / "liquid_teacher"    / "exp01"
OUTPUT_FILE   = RESULTS_ROOT / "phenology_vs_baseline_stats.json"


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


phenology_maes = np.array(load_fold_maes(PHENOLOGY_DIR))
baseline_maes  = np.array(load_fold_maes(BASELINE_DIR))

assert len(phenology_maes) == len(baseline_maes), (
    f"Fold count mismatch: phenology={len(phenology_maes)}, "
    f"baseline={len(baseline_maes)}"
)
n_folds = len(phenology_maes)


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------
pheno_mean = float(phenology_maes.mean())
pheno_std  = float(phenology_maes.std(ddof=1))
base_mean  = float(baseline_maes.mean())
base_std   = float(baseline_maes.std(ddof=1))

diff      = phenology_maes - baseline_maes   # positive → phenology worse
diff_mean = float(diff.mean())
diff_std  = float(diff.std(ddof=1))


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test (primary, two-sided)
# ---------------------------------------------------------------------------
wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
    phenology_maes, baseline_maes, alternative="two-sided"
)

# ---------------------------------------------------------------------------
# Paired t-test (secondary, two-sided)
# ---------------------------------------------------------------------------
ttest_stat, ttest_p = stats.ttest_rel(
    phenology_maes, baseline_maes, alternative="two-sided"
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
SEP = "-" * 68

print(SEP)
print("  Statistical Comparison: Phenology LNN vs Baseline LNN  (exp01 LOPO)")
print(SEP)
print(f"  Folds: {n_folds}")
print()
print(f"  {'Model':<24}  {'Mean MAE (g)':>13}  {'Std MAE (g)':>12}")
print(f"  {'-'*24}  {'-'*13}  {'-'*12}")
print(f"  {'Phenology LNN':<24}  {pheno_mean:>13.4f}  {pheno_std:>12.4f}")
print(f"  {'Baseline LNN':<24}  {base_mean:>13.4f}  {base_std:>12.4f}")
print(f"  {'Difference (Pheno-Base)':<24}  {diff_mean:>+13.4f}  {diff_std:>12.4f}")
print()
print(f"  {'Test':<32}  {'Statistic':>10}  {'p-value':>10}  {'Sig':>4}")
print(f"  {'-'*32}  {'-'*10}  {'-'*10}  {'-'*4}")
print(f"  {'Wilcoxon signed-rank (primary)':<32}  "
      f"{wilcoxon_stat:>10.4f}  {wilcoxon_p:>10.4f}  {sig_label(wilcoxon_p):>4}")
print(f"  {'Paired t-test (secondary)':<32}  "
      f"{ttest_stat:>10.4f}  {ttest_p:>10.4f}  {sig_label(ttest_p):>4}")
print()

effect_dir = "phenology better" if cohens_d < 0 else "phenology worse"
print(f"  Cohen's d (paired):  {cohens_d:+.4f}  ({effect_dir})")
print()

if diff_mean < 0:
    interp = f"Phenology LNN is BETTER by {abs(diff_mean):.4f} g mean MAE."
elif diff_mean > 0:
    interp = f"Phenology LNN is WORSE by {diff_mean:.4f} g mean MAE."
else:
    interp = "Models are identical on average."
print(f"  Interpretation: {interp}")
print(SEP)


# ---------------------------------------------------------------------------
# Save JSON results
# ---------------------------------------------------------------------------
output = {
    "experiment": "exp01",
    "n_folds": n_folds,
    "phenology_lnn": {
        "mean_test_mae_g": pheno_mean,
        "std_test_mae_g": pheno_std,
        "per_fold_mae_g": phenology_maes.tolist(),
    },
    "baseline_lnn": {
        "mean_test_mae_g": base_mean,
        "std_test_mae_g": base_std,
        "per_fold_mae_g": baseline_maes.tolist(),
    },
    "difference_phenology_minus_baseline": {
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
    "cohens_d": cohens_d,
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_FILE.open("w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Results saved to: {OUTPUT_FILE}\n")
