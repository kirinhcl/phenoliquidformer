#!/usr/bin/env python3
"""Cross-experiment transfer analysis.

Addresses Q6: Can a model trained on non-vernalized plants generalize to
vernalized/regrowth plants? Uses t-SNE to visualize learned embeddings.

Usage:
    python scripts/analyze_cross_experiment.py --checkpoint results/exp01_lowho/fold_0/best_model_state.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from src.data.dataset import TimothyDroughtDataset
from src.model.timothy_model import TimothyDroughtModel
from src.utils.config import load_config

OUTPUT_DIR = Path("results/cross_experiment")


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


def extract_embeddings(
    model: TimothyDroughtModel,
    dataset: TimothyDroughtDataset,
    device: str,
) -> pd.DataFrame:
    """Extract CLS embeddings and predictions for all plants."""
    model.to(device)
    model.train(False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    embeddings = []
    metadata = []

    with torch.no_grad():
        for batch in loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(batch_dev)
            cls_emb = outputs["cls_embedding"].cpu().numpy()  # (B, 128)
            whc_pred = outputs["whc_pred"].cpu().numpy()  # (B,)

            for b in range(cls_emb.shape[0]):
                embeddings.append(cls_emb[b])
                metadata.append({
                    "plant_id": batch["plant_id"][b],
                    "experiment": batch["experiment"][b],
                    "genotype": batch["genotype"][b],
                    "treatment": batch["treatment"][b],
                    "whc_target": float(batch["whc_target"][b]),
                    "whc_pred": float(whc_pred[b]),
                })

    emb_array = np.array(embeddings)
    meta_df = pd.DataFrame(metadata)
    return emb_array, meta_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/timothy.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model trained on one experiment
    cfg = load_config(args.config, overrides=["data.experiment=all"])
    dataset = TimothyDroughtDataset(cfg)
    model = TimothyDroughtModel(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    print(f"Extracting embeddings for {len(dataset)} plants (all experiments)...")
    emb_array, meta_df = extract_embeddings(model, dataset, device)

    # t-SNE on embeddings
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb_array) - 1))
    coords = tsne.fit_transform(emb_array)
    meta_df["tsne_x"] = coords[:, 0]
    meta_df["tsne_y"] = coords[:, 1]
    meta_df.to_csv(OUTPUT_DIR / "embeddings.csv", index=False)

    # --- Plot 1: Colored by experiment ---
    fig, ax = plt.subplots(figsize=(8, 6))
    exp_colors = {"exp01": "tab:blue", "exp02": "tab:orange", "exp03": "tab:green"}
    exp_labels = {"exp01": "Non-vernalized", "exp02": "Vernalized", "exp03": "Regrowth"}
    for exp_id, color in exp_colors.items():
        mask = meta_df["experiment"] == exp_id
        ax.scatter(meta_df.loc[mask, "tsne_x"], meta_df.loc[mask, "tsne_y"],
                   c=color, label=exp_labels[exp_id], alpha=0.7, s=30)
    ax.set_title("t-SNE of Learned Embeddings (by Experiment)")
    ax.legend()
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tsne_by_experiment.pdf", dpi=300)
    fig.savefig(OUTPUT_DIR / "tsne_by_experiment.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Colored by WHC level ---
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(meta_df["tsne_x"], meta_df["tsne_y"],
                    c=meta_df["whc_target"], cmap="RdYlBu", alpha=0.7, s=30)
    fig.colorbar(sc, ax=ax, label="WHC Level")
    ax.set_title("t-SNE of Learned Embeddings (by WHC)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tsne_by_whc.pdf", dpi=300)
    fig.savefig(OUTPUT_DIR / "tsne_by_whc.png", dpi=150)
    plt.close(fig)

    # --- Cross-experiment prediction accuracy ---
    for exp_id in ["exp01", "exp02", "exp03"]:
        mask = meta_df["experiment"] == exp_id
        if mask.sum() == 0:
            continue
        preds = meta_df.loc[mask, "whc_pred"].values
        targets = meta_df.loc[mask, "whc_target"].values
        mae = np.mean(np.abs(preds - targets))
        print(f"  {exp_id}: MAE={mae:.4f} (n={mask.sum()})")

    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
