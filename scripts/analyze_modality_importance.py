#!/usr/bin/env python3
"""Modality importance analysis via gate weights from trained model.

Addresses Q5: Which modality is most informative at each drought severity level?
Extracts gate weights from the trained model across all plants and timepoints.

Usage:
    python scripts/analyze_modality_importance.py --checkpoint results/exp01_lowho/fold_0/best_model_state.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import TimothyDroughtDataset
from src.model.timothy_model import TimothyDroughtModel
from src.utils.config import load_config

OUTPUT_DIR = Path("results/modality_importance")
MODALITY_NAMES = ["Image", "Fluorescence", "Environment", "Vegetation Index"]


def collate_fn(batch: list[dict]) -> dict:
    result = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values, dtype=torch.float32)
        else:
            result[key] = values
    return result


def extract_gate_weights(
    model: TimothyDroughtModel,
    dataset: TimothyDroughtDataset,
    device: str,
) -> pd.DataFrame:
    """Extract gate weights for all plants at all timepoints."""
    model.to(device)
    model.train(False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    records = []
    with torch.no_grad():
        for batch in loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(batch_dev)
            gates = outputs["modality_gates"].cpu().numpy()  # (B, T, 4)
            active = batch["active_mask"].numpy()  # (B, T)
            tp = batch["temporal_positions"].numpy()  # (B, T)

            for b in range(gates.shape[0]):
                for t in range(gates.shape[1]):
                    if not active[b, t]:
                        continue
                    records.append({
                        "plant_id": batch["plant_id"][b],
                        "treatment": batch["treatment"][b],
                        "whc": float(batch["whc_target"][b]),
                        "das": float(tp[b, t]),
                        "gate_image": gates[b, t, 0],
                        "gate_fluor": gates[b, t, 1],
                        "gate_env": gates[b, t, 2],
                        "gate_vi": gates[b, t, 3],
                    })

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/timothy.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model_state.pt")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_config(args.config)
    dataset = TimothyDroughtDataset(cfg)
    model = TimothyDroughtModel(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    print(f"Extracting gate weights for {len(dataset)} plants...")
    gate_df = extract_gate_weights(model, dataset, device)
    gate_df.to_csv(OUTPUT_DIR / "gate_weights.csv", index=False)

    # --- Plot 1: Gate weights by WHC level (averaged over time) ---
    gate_cols = ["gate_image", "gate_fluor", "gate_env", "gate_vi"]
    by_whc = gate_df.groupby("whc")[gate_cols].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(by_whc))
    width = 0.2
    for i, (col, name) in enumerate(zip(gate_cols, MODALITY_NAMES)):
        ax.bar(x + i * width, by_whc[col], width, label=name)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"WHC-{int(w*100)}%" for w in by_whc.index])
    ax.set_ylabel("Gate Weight")
    ax.set_title("Modality Importance by Drought Severity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gates_by_whc.pdf", dpi=300)
    fig.savefig(OUTPUT_DIR / "gates_by_whc.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Gate weights heatmap (WHC x DAS) per modality ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, col, name in zip(axes.flat, gate_cols, MODALITY_NAMES):
        pivot = gate_df.pivot_table(values=col, index="whc", columns="das", aggfunc="mean")
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{int(w*100)}%" for w in pivot.index])
        ax.set_xticks(range(0, len(pivot.columns), max(1, len(pivot.columns)//8)))
        ax.set_xticklabels([int(d) for d in pivot.columns[::max(1, len(pivot.columns)//8)]])
        ax.set_title(name)
        ax.set_xlabel("DAS")
        ax.set_ylabel("WHC")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Modality Gate Weights: WHC x Time", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gates_heatmap.pdf", dpi=300)
    fig.savefig(OUTPUT_DIR / "gates_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
