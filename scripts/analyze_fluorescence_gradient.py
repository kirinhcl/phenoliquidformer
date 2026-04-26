#!/usr/bin/env python3
"""Fluorescence analysis across drought gradient.

Identifies which ChlF parameters are earliest and strongest drought indicators.
Computes Spearman correlation of each parameter with WHC level over time.

Usage:
    python scripts/analyze_fluorescence_gradient.py --experiment exp01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = Path("results/fluorescence")

EXP_PATHS = {
    "exp01": "data/2023-Timothy-01-Nonvernalized/FCQ_Timothy-01.xlsx",
    "exp02": "data/2024-Timothy-02-Vernalized/FCQ_Timothy-02.xlsx",
    "exp03": "data/2024-Timothy-03-Regrowth/FCQ_Timothy-03.xlsx",
}


def analyze_fluorescence(experiment: str) -> None:
    output_dir = OUTPUT_DIR / experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(EXP_PATHS[experiment])
    df.columns = [str(c).strip() for c in df.columns]
    df["Treatment"] = df["Treatment"].astype(str).str.strip()
    df["WHC"] = df["Treatment"].str.extract(r"WHC-(\d+)").astype(float) / 100

    # Identify fluorescence columns (numeric columns after metadata)
    exclude = {"Obs", "Camera Position", "Size", "WHC"}
    meta_end = 20
    fluor_cols = [
        c for i, c in enumerate(df.columns)
        if i >= meta_end and c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"  {len(fluor_cols)} fluorescence parameters")

    # --- Spearman correlation: each ChlF param vs WHC at each timepoint ---
    das_values = sorted(df["DAS"].dropna().unique())
    corr_matrix = np.full((len(fluor_cols), len(das_values)), np.nan)
    pval_matrix = np.full((len(fluor_cols), len(das_values)), np.nan)

    for j, das in enumerate(das_values):
        das_df = df[df["DAS"] == das]
        whc = das_df["WHC"].values
        for i, col in enumerate(fluor_cols):
            vals = pd.to_numeric(das_df[col], errors="coerce").values
            valid = ~(np.isnan(vals) | np.isnan(whc))
            if valid.sum() >= 5:
                rho, p = stats.spearmanr(whc[valid], vals[valid])
                corr_matrix[i, j] = rho
                pval_matrix[i, j] = p

    # --- Heatmap: fluorescence param x timepoint ---
    fig, ax = plt.subplots(figsize=(14, max(8, len(fluor_cols) * 0.15)))
    im = ax.imshow(corr_matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(das_values)))
    ax.set_xticklabels([int(d) for d in das_values], rotation=45, fontsize=6)
    ax.set_yticks(range(len(fluor_cols)))
    ax.set_yticklabels(fluor_cols, fontsize=5)
    ax.set_xlabel("DAS")
    ax.set_ylabel("Fluorescence Parameter")
    ax.set_title(f"Spearman Correlation with WHC Level — {experiment}")
    fig.colorbar(im, ax=ax, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(output_dir / "fluorescence_heatmap.pdf", dpi=300)
    fig.savefig(output_dir / "fluorescence_heatmap.png", dpi=150)
    plt.close(fig)

    # --- Top drought-sensitive parameters ---
    # Average absolute correlation across timepoints
    mean_abs_corr = np.nanmean(np.abs(corr_matrix), axis=1)
    ranking = pd.DataFrame({
        "parameter": fluor_cols,
        "mean_abs_correlation": mean_abs_corr,
        "mean_correlation": np.nanmean(corr_matrix, axis=1),
    }).sort_values("mean_abs_correlation", ascending=False)

    ranking.to_csv(output_dir / "parameter_ranking.csv", index=False)
    print(f"  Top 10 drought-sensitive parameters:")
    for _, row in ranking.head(10).iterrows():
        print(f"    {row['parameter']:20s} |rho|={row['mean_abs_correlation']:.3f}  rho={row['mean_correlation']:.3f}")

    # --- Key parameters time-series by WHC ---
    key_params = ranking.head(4)["parameter"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    whc_levels = sorted(df["WHC"].unique())
    colors = plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(whc_levels)))

    for ax, param in zip(axes.flat, key_params):
        for whc, color in zip(whc_levels, colors):
            subset = df[df["WHC"] == whc].groupby("DAS")[param].agg(["mean", "sem"]).reset_index()
            ax.plot(subset["DAS"], subset["mean"], "o-", color=color, label=f"WHC-{int(whc*100)}%", linewidth=1.5, markersize=3)
        ax.set_title(param, fontsize=10)
        ax.set_xlabel("DAS")

    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Top Drought-Sensitive ChlF Parameters — {experiment}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "top_parameters.pdf", dpi=300)
    fig.savefig(output_dir / "top_parameters.png", dpi=150)
    plt.close(fig)

    print(f"  Results saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="exp01", choices=["exp01", "exp02", "exp03", "all"])
    args = parser.parse_args()

    experiments = ["exp01", "exp02", "exp03"] if args.experiment == "all" else [args.experiment]
    for exp in experiments:
        print(f"\n=== Fluorescence: {exp} ===")
        analyze_fluorescence(exp)


if __name__ == "__main__":
    main()
