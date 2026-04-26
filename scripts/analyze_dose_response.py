#!/usr/bin/env python3
"""Dose-response analysis: trait vs WHC level across timepoints.

Computes EC50, ANOVA, and dose-response curves for key traits.
Addresses Q1: At what WHC level do critical physiological transitions occur?

Usage:
    python scripts/analyze_dose_response.py --experiment exp01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

OUTPUT_DIR = Path("results/dose_response")


def sigmoid(x, L, k, x0, b):
    """4-parameter logistic (sigmoid) dose-response curve."""
    return L / (1 + np.exp(-k * (x - x0))) + b


def estimate_ec50(whc_values: np.ndarray, trait_values: np.ndarray) -> float | None:
    """Estimate EC50 (WHC level at 50% trait reduction) via sigmoid fit."""
    try:
        # Normalize trait to 0-1 range
        t_min, t_max = trait_values.min(), trait_values.max()
        if t_max - t_min < 1e-8:
            return None
        t_norm = (trait_values - t_min) / (t_max - t_min)
        popt, _ = curve_fit(
            sigmoid, whc_values, t_norm,
            p0=[1, 10, 0.5, 0], maxfev=5000,
            bounds=([0, 0, 0.1, -1], [2, 100, 1.0, 1])
        )
        ec50 = popt[2]  # x0 parameter = inflection point
        return float(ec50)
    except (RuntimeError, ValueError):
        return None


def load_experiment_data(experiment: str) -> pd.DataFrame:
    """Load digital biomass data for an experiment."""
    exp_dirs = {
        "exp01": "data/2023-Timothy-01-Nonvernalized",
        "exp02": "data/2024-Timothy-02-Vernalized",
        "exp03": "data/2024-Timothy-03-Regrowth",
    }
    db_path = Path(exp_dirs[experiment])
    db_file = list(db_path.glob("DigBio_Timothy-*.xlsx"))[0]
    df = pd.read_excel(db_file)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def analyze_dose_response(experiment: str) -> None:
    """Run dose-response analysis for one experiment."""
    df = load_experiment_data(experiment)
    output_dir = OUTPUT_DIR / experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mean digital biomass per plant per timepoint
    df["Plant ID"] = df["Plant ID"].astype(str).str.strip()
    df["Treatment"] = df["Treatment"].astype(str).str.strip()

    # Parse WHC as numeric
    df["WHC"] = df["Treatment"].str.extract(r"WHC-(\d+)").astype(float) / 100

    # Group by treatment and DAS
    grouped = df.groupby(["Treatment", "WHC", "DAS"])["Digital Biomass"].agg(["mean", "sem", "count"]).reset_index()

    # --- Plot 1: Growth curves per WHC level ---
    fig, ax = plt.subplots(figsize=(10, 6))
    whc_levels = sorted(grouped["WHC"].unique())
    colors = plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(whc_levels)))

    for whc, color in zip(whc_levels, colors):
        subset = grouped[grouped["WHC"] == whc].sort_values("DAS")
        ax.plot(subset["DAS"], subset["mean"], "o-", color=color, label=f"WHC-{int(whc*100)}%", linewidth=2)
        ax.fill_between(subset["DAS"], subset["mean"] - subset["sem"], subset["mean"] + subset["sem"], alpha=0.2, color=color)

    ax.set_xlabel("Days After Sowing (DAS)")
    ax.set_ylabel("Digital Biomass")
    ax.set_title(f"Growth Curves by WHC Level — {experiment}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "growth_curves.pdf", dpi=300)
    fig.savefig(output_dir / "growth_curves.png", dpi=150)
    plt.close(fig)

    # --- ANOVA at each timepoint ---
    anova_results = []
    for das, das_df in df.groupby("DAS"):
        groups = [g["Digital Biomass"].dropna().values for _, g in das_df.groupby("Treatment")]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            h_stat, p_val = stats.kruskal(*groups)
            anova_results.append({"DAS": das, "H_statistic": h_stat, "p_value": p_val})

    anova_df = pd.DataFrame(anova_results)
    anova_df.to_csv(output_dir / "kruskal_wallis_per_das.csv", index=False)
    print(f"  Kruskal-Wallis: {(anova_df['p_value'] < 0.05).sum()}/{len(anova_df)} timepoints significant")

    # --- EC50 estimation per timepoint ---
    ec50_results = []
    for das, das_df in df.groupby("DAS"):
        das_summary = das_df.groupby("WHC")["Digital Biomass"].mean()
        if len(das_summary) >= 4:
            ec50 = estimate_ec50(das_summary.index.values, das_summary.values)
            ec50_results.append({"DAS": das, "EC50": ec50})

    ec50_df = pd.DataFrame(ec50_results)
    ec50_df.to_csv(output_dir / "ec50_per_das.csv", index=False)

    valid_ec50 = ec50_df.dropna(subset=["EC50"])
    if len(valid_ec50) > 0:
        print(f"  EC50 range: {valid_ec50['EC50'].min():.2f} - {valid_ec50['EC50'].max():.2f}")

    # --- Plot 2: EC50 trajectory over time ---
    if len(valid_ec50) > 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(valid_ec50["DAS"], valid_ec50["EC50"], "o-", color="darkred", linewidth=2)
        ax.set_xlabel("Days After Sowing (DAS)")
        ax.set_ylabel("EC50 (WHC level)")
        ax.set_title(f"EC50 Trajectory — {experiment}")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="WHC-50%")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "ec50_trajectory.pdf", dpi=300)
        fig.savefig(output_dir / "ec50_trajectory.png", dpi=150)
        plt.close(fig)

    print(f"  Results saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="exp01", choices=["exp01", "exp02", "exp03", "all"])
    args = parser.parse_args()

    experiments = ["exp01", "exp02", "exp03"] if args.experiment == "all" else [args.experiment]
    for exp in experiments:
        print(f"\n=== Dose-Response: {exp} ===")
        analyze_dose_response(exp)


if __name__ == "__main__":
    main()
