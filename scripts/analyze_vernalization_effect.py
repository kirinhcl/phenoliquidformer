#!/usr/bin/env python3
"""Vernalization effect: Exp01 (non-vernalized) vs Exp02 (vernalized).

Addresses Q3: How does vernalization reprogram drought response?
Compares Jauniai only (present in both experiments) at matching WHC levels.

Usage:
    python scripts/analyze_vernalization_effect.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = Path("results/vernalization_effect")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plant_meta = pd.read_csv("data/plant_metadata.csv")

    # Load endpoint data for both experiments
    ep1 = pd.read_excel("data/2023-Timothy-01-Nonvernalized/EndPoint_Timothy-01_Weight+Flowering.xlsx")
    ep1.columns = [str(c).strip() for c in ep1.columns]
    ep1["experiment"] = "exp01"
    ep1["vernalized"] = False

    ep2 = pd.read_excel("data/2024-Timothy-02-Vernalized/EndPoint_Timothy-02_Weight+Flowering.xlsx")
    ep2.columns = [str(c).strip() for c in ep2.columns]
    ep2["experiment"] = "exp02"
    ep2["vernalized"] = True

    # Standardize column names
    for ep in [ep1, ep2]:
        ep["Genotype"] = ep[[c for c in ep.columns if "Genotype" in c][0]].astype(str).str.strip()
        ep["Treatment"] = ep[[c for c in ep.columns if "Treatment" in c][0]].astype(str).str.strip()
        ep["WHC"] = ep["Treatment"].str.extract(r"WHC-(\d+)").astype(float) / 100
        fw_col = [c for c in ep.columns if "Fresh Weight" in c][0]
        dw_col = [c for c in ep.columns if "Dry Weight" in c][0]
        ep["fw_g"] = pd.to_numeric(ep[fw_col], errors="coerce")
        ep["dw_g"] = pd.to_numeric(ep[dw_col], errors="coerce")
        flower_col = [c for c in ep.columns if "Flower" in c]
        if flower_col:
            ep["flowering"] = pd.to_numeric(ep[flower_col[0]], errors="coerce")

    # Filter Jauniai only, matching WHC levels (30-90%)
    common_whc = set(ep1["WHC"].unique()) & set(ep2["WHC"].unique())
    j1 = ep1[(ep1["Genotype"] == "Jauniai") & (ep1["WHC"].isin(common_whc))]
    j2 = ep2[(ep2["Genotype"] == "Jauniai") & (ep2["WHC"].isin(common_whc))]

    combined = pd.concat([j1, j2], ignore_index=True)

    # --- Plot 1: Endpoint biomass: vernalized vs non-vernalized ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in zip(axes, ["fw_g", "dw_g"], ["Fresh Weight (g)", "Dry Weight (g)"]):
        for vern, label, marker in [(False, "Non-vernalized (Exp01)", "o"), (True, "Vernalized (Exp02)", "s")]:
            subset = combined[combined["vernalized"] == vern]
            means = subset.groupby("WHC")[col].mean()
            sems = subset.groupby("WHC")[col].sem()
            ax.errorbar(means.index, means.values, yerr=sems.values,
                        fmt=f"{marker}-", label=label, capsize=3, linewidth=2)
        ax.set_xlabel("WHC Level")
        ax.set_ylabel(title)
        ax.legend()

    fig.suptitle("Vernalization Effect on Endpoint Biomass (Jauniai)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "vernalization_endpoint.pdf", dpi=300)
    fig.savefig(OUTPUT_DIR / "vernalization_endpoint.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Flowering (vernalized only) ---
    if "flowering" in combined.columns:
        vern_data = combined[combined["vernalized"]]
        if vern_data["flowering"].notna().any():
            fig, ax = plt.subplots(figsize=(7, 5))
            flower_by_whc = vern_data.groupby("WHC")["flowering"].agg(["mean", "sem"]).reset_index()
            ax.bar(flower_by_whc["WHC"].astype(str), flower_by_whc["mean"],
                   yerr=flower_by_whc["sem"], capsize=3, color="mediumpurple")
            ax.set_xlabel("WHC Level")
            ax.set_ylabel("Number of Flowers")
            ax.set_title("Flowering Under Drought (Vernalized Jauniai)")
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / "flowering_by_whc.pdf", dpi=300)
            fig.savefig(OUTPUT_DIR / "flowering_by_whc.png", dpi=150)
            plt.close(fig)

    # --- Statistical comparison ---
    stat_results = []
    for whc in sorted(common_whc):
        v0 = j1[j1["WHC"] == whc]["fw_g"].dropna()
        v1 = j2[j2["WHC"] == whc]["fw_g"].dropna()
        if len(v0) >= 2 and len(v1) >= 2:
            u, p = stats.mannwhitneyu(v0, v1, alternative="two-sided")
            stat_results.append({
                "WHC": whc,
                "nonvern_mean_fw": v0.mean(),
                "vern_mean_fw": v1.mean(),
                "change_pct": (v1.mean() - v0.mean()) / v0.mean() * 100,
                "p_value": p,
            })

    stat_df = pd.DataFrame(stat_results)
    stat_df.to_csv(OUTPUT_DIR / "vernalization_stats.csv", index=False)
    print(f"  Vernalization comparison (Jauniai, {len(common_whc)} WHC levels):")
    for _, r in stat_df.iterrows():
        sig = "*" if r["p_value"] < 0.05 else ""
        print(f"    WHC-{int(r['WHC']*100)}%: {r['change_pct']:+.1f}% change {sig}")
    print(f"  Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
