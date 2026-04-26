#!/usr/bin/env python3
"""Genotype comparison: Jauniai vs Noreng under drought gradient.

Addresses Q2: Do northern-adapted (Noreng) and southern-adapted (Jauniai)
genotypes differ in drought response strategies?

Usage:
    python scripts/analyze_genotype_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = Path("results/genotype_comparison")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Exp01 data (both genotypes)
    df = pd.read_excel("data/2023-Timothy-01-Nonvernalized/DigBio_Timothy-01.xlsx")
    df.columns = [str(c).strip() for c in df.columns]
    df["Genotype"] = df["Genotype"].astype(str).str.strip()
    df["Treatment"] = df["Treatment"].astype(str).str.strip()
    df["WHC"] = df["Treatment"].str.extract(r"WHC-(\d+)").astype(float) / 100

    # Load endpoint data
    ep = pd.read_excel("data/2023-Timothy-01-Nonvernalized/EndPoint_Timothy-01_Weight+Flowering.xlsx")
    ep.columns = [str(c).strip() for c in ep.columns]
    ep["Genotype"] = ep["Genotype"].astype(str).str.strip()
    ep["Treatment"] = ep["Treatment"].astype(str).str.strip()
    ep["WHC"] = ep["Treatment"].str.extract(r"WHC-(\d+)").astype(float) / 100

    # --- Plot 1: Growth curves by genotype × WHC ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    whc_levels = sorted(df["WHC"].unique())
    colors = plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(whc_levels)))

    for ax, genotype in zip(axes, ["Jauniai", "Noreng"]):
        gdf = df[df["Genotype"] == genotype]
        for whc, color in zip(whc_levels, colors):
            subset = gdf[gdf["WHC"] == whc].groupby("DAS")["Digital Biomass"].agg(["mean", "sem"]).reset_index()
            ax.plot(subset["DAS"], subset["mean"], "o-", color=color, label=f"WHC-{int(whc*100)}%", linewidth=1.5, markersize=3)
        ax.set_title(genotype)
        ax.set_xlabel("DAS")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Digital Biomass")
    fig.suptitle("Growth Curves: Jauniai vs Noreng (Exp01)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "growth_curves_by_genotype.pdf", dpi=300)
    fig.savefig(OUTPUT_DIR / "growth_curves_by_genotype.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Endpoint biomass by genotype × WHC ---
    fw_col = [c for c in ep.columns if "Fresh Weight" in c][0]
    dw_col = [c for c in ep.columns if "Dry Weight" in c][0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in zip(axes, [fw_col, dw_col], ["Fresh Weight (g)", "Dry Weight (g)"]):
        for i, genotype in enumerate(["Jauniai", "Noreng"]):
            gep = ep[ep["Genotype"] == genotype]
            means = gep.groupby("WHC")[col].mean()
            sems = gep.groupby("WHC")[col].sem()
            offset = (i - 0.5) * 0.015
            ax.errorbar(means.index + offset, means.values, yerr=sems.values,
                        fmt="o-", label=genotype, capsize=3, linewidth=2)
        ax.set_xlabel("WHC Level")
        ax.set_ylabel(title)
        ax.legend()

    fig.suptitle("Endpoint Biomass: Genotype Comparison (Exp01)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "endpoint_biomass_genotype.pdf", dpi=300)
    fig.savefig(OUTPUT_DIR / "endpoint_biomass_genotype.png", dpi=150)
    plt.close(fig)

    # --- Two-way ANOVA: Genotype × WHC interaction ---
    interaction_results = []
    for das, das_df in df.groupby("DAS"):
        # Mann-Whitney U between genotypes at each WHC level
        for whc in whc_levels:
            subset = das_df[das_df["WHC"] == whc]
            j_vals = subset[subset["Genotype"] == "Jauniai"]["Digital Biomass"].dropna()
            n_vals = subset[subset["Genotype"] == "Noreng"]["Digital Biomass"].dropna()
            if len(j_vals) >= 2 and len(n_vals) >= 2:
                u_stat, p_val = stats.mannwhitneyu(j_vals, n_vals, alternative="two-sided")
                interaction_results.append({
                    "DAS": das, "WHC": whc, "U_statistic": u_stat,
                    "p_value": p_val, "Jauniai_mean": j_vals.mean(), "Noreng_mean": n_vals.mean()
                })

    inter_df = pd.DataFrame(interaction_results)
    inter_df.to_csv(OUTPUT_DIR / "genotype_mannwhitney.csv", index=False)
    sig = inter_df[inter_df["p_value"] < 0.05]
    print(f"  Significant genotype differences: {len(sig)}/{len(inter_df)} (DAS x WHC combinations)")
    print(f"  Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
