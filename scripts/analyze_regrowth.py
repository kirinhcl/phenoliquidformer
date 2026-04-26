#!/usr/bin/env python3
"""Regrowth analysis: drought history effect on recovery after cutting.

Addresses Q4: Does drought history affect regrowth capacity?

Usage:
    python scripts/analyze_regrowth.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = Path("results/regrowth")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Exp03 digital biomass
    df = pd.read_excel("data/2024-Timothy-03-Regrowth/DigBio_Timothy-03.xlsx")
    df.columns = [str(c).strip() for c in df.columns]
    df["Plant ID"] = df["Plant ID"].astype(str).str.strip()
    df["Genotype"] = df["Genotype"].astype(str).str.strip()
    df["Treatment"] = df["Treatment"].astype(str).str.strip()
    df["WHC"] = df["Treatment"].str.extract(r"WHC-(\d+)").astype(float) / 100

    # --- Plot 1: Regrowth curves per WHC ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    whc_levels = sorted(df["WHC"].unique())
    colors = plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(whc_levels)))

    for ax, genotype in zip(axes, ["Jauniai", "Noreng"]):
        gdf = df[df["Genotype"] == genotype]
        for whc, color in zip(whc_levels, colors):
            subset = gdf[gdf["WHC"] == whc].groupby("DAS")["Digital Biomass"].agg(["mean", "sem"]).reset_index()
            ax.plot(subset["DAS"], subset["mean"], "o-", color=color, label=f"WHC-{int(whc*100)}%", linewidth=2, markersize=4)
        ax.set_title(f"{genotype} — Regrowth")
        ax.set_xlabel("DAS")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Digital Biomass")
    fig.suptitle("Regrowth Curves After Cutting (Exp03)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "regrowth_curves.pdf", dpi=300)
    fig.savefig(OUTPUT_DIR / "regrowth_curves.png", dpi=150)
    plt.close(fig)

    # --- Regrowth rate: slope of biomass over time per plant ---
    rate_results = []
    for (pid, genotype, whc), plant_df in df.groupby(["Plant ID", "Genotype", "WHC"]):
        plant_df = plant_df.sort_values("DAS")
        das = plant_df["DAS"].values.astype(float)
        bm = plant_df["Digital Biomass"].values.astype(float)
        valid = ~np.isnan(bm)
        if valid.sum() >= 3:
            slope = np.polyfit(das[valid], bm[valid], 1)[0]
            rate_results.append({
                "plant_id": pid, "genotype": genotype, "whc": whc,
                "regrowth_rate": slope, "final_biomass": bm[valid][-1],
            })

    rate_df = pd.DataFrame(rate_results)
    rate_df.to_csv(OUTPUT_DIR / "regrowth_rates.csv", index=False)

    # --- Plot 2: Regrowth rate vs WHC ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for genotype, marker in [("Jauniai", "o"), ("Noreng", "s")]:
        gdf = rate_df[rate_df["genotype"] == genotype]
        means = gdf.groupby("whc")["regrowth_rate"].mean()
        sems = gdf.groupby("whc")["regrowth_rate"].sem()
        ax.errorbar(means.index, means.values, yerr=sems.values,
                    fmt=f"{marker}-", label=genotype, capsize=3, linewidth=2)

    ax.set_xlabel("WHC Level (drought history)")
    ax.set_ylabel("Regrowth Rate (biomass/day)")
    ax.set_title("Regrowth Rate vs Drought History (Exp03)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "regrowth_rate_vs_whc.pdf", dpi=300)
    fig.savefig(OUTPUT_DIR / "regrowth_rate_vs_whc.png", dpi=150)
    plt.close(fig)

    # --- Stats: Kruskal-Wallis for regrowth rate across WHC ---
    for genotype in ["Jauniai", "Noreng"]:
        gdf = rate_df[rate_df["genotype"] == genotype]
        groups = [g["regrowth_rate"].values for _, g in gdf.groupby("whc")]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            h, p = stats.kruskal(*groups)
            print(f"  {genotype}: Kruskal-Wallis H={h:.2f}, p={p:.4f}")

    # Spearman correlation: WHC vs regrowth rate
    rho, p = stats.spearmanr(rate_df["whc"], rate_df["regrowth_rate"])
    print(f"  Overall: Spearman rho={rho:.3f}, p={p:.4f}")
    print(f"  Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
