#!/usr/bin/env python3
"""Visualize temporal attention weights from trained LiquidTransformerModel (h64, n_layers=2, residual).

Loads the fold-0 checkpoint from results/lt_h64_n2_res/exp01/fold_0/best_model_state.pt,
runs inference on all exp01 plants, averages per-head attention weights across plants,
and produces a publication-quality figure (heatmap + line plot) saved to
paper/figures/final/fig_lt_attention_weights.{png,pdf}.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.dataset import TimothyDroughtDataset
from src.model.liquid_transformer import LiquidTransformerModel
from src.utils.config import load_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAS_VALUES = [1, 4, 6, 8, 11, 15, 18, 22, 25, 26, 29, 32, 36, 39, 43, 46, 50, 51]
N_TIMEPOINTS = len(DAS_VALUES)
N_HEADS = 4

CHECKPOINT_PATH = PROJECT_ROOT / "results/lt_h64_n2_res/exp01/fold_0/best_model_state.pt"
CONFIG_PATH = str(PROJECT_ROOT / "configs/timothy.yaml")
OUT_DIR = PROJECT_ROOT / "paper/figures/final"

# Nature / NPG colour palette
NATURE_COLORS = [
    "#E64B35",  # coral
    "#4DBBD5",  # sky
    "#00A087",  # teal
    "#3C5488",  # navy
]
HEAD_COLORS = NATURE_COLORS[:N_HEADS]

# ---------------------------------------------------------------------------
# Matplotlib style — Nature / NPG
# ---------------------------------------------------------------------------

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 6,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def collate_fn(batch: list[dict]) -> dict:
    result: dict = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values, dtype=torch.float32)
        else:
            result[key] = values
    return result


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: Path, cfg) -> LiquidTransformerModel:
    OmegaConf.update(cfg, "model.liquid.hidden_dim", 64)
    OmegaConf.update(cfg, "model.liquid.n_layers", 2)
    model = LiquidTransformerModel(role="teacher", cfg=cfg)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Collect attention weights across dataset
# ---------------------------------------------------------------------------


def collect_attention_weights(
    model: LiquidTransformerModel,
    dataset: TimothyDroughtDataset,
    batch_size: int = 8,
    device: torch.device | None = None,
) -> np.ndarray:
    """Return array of shape (N_plants, N_heads, T) with per-plant attention weights."""
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_weights: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = model(batch, t_cut=None)
            # attn_weights: (B, H, T)
            attn = out["attn_weights"].cpu().numpy()
            all_weights.append(attn)

    return np.concatenate(all_weights, axis=0)  # (N_plants, H, T)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.08) -> None:
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def make_figure(mean_attn: np.ndarray, das_values: list[int]) -> plt.Figure:
    """Build the two-panel figure.

    Panel A: heatmap  — (N_heads rows) x (T cols), colour = attention weight.
    Panel B: line plot — per-head curves + grand mean.

    Args:
        mean_attn: (N_heads, T) array of attention weights averaged over plants.
        das_values: DAS values for each time point (length T).

    Returns:
        matplotlib Figure.
    """
    T = len(das_values)
    H = mean_attn.shape[0]

    double_col_in = 183 / 25.4  # 183 mm double-column width
    fig = plt.figure(figsize=(double_col_in, double_col_in * 0.60))

    gs = gridspec.GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[1.4, 1.0],
        hspace=0.60,
    )

    x = np.arange(T)
    x_labels = [str(d) for d in das_values]

    # ── Panel A: Heatmap ─────────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[0])
    panel_label(ax_heat, "A")

    vmin = float(mean_attn.min())
    vmax = float(mean_attn.max())
    if vmax - vmin < 1e-6:
        vmin = vmin - 1e-4
        vmax = vmax + 1e-4

    im = ax_heat.imshow(
        mean_attn,
        aspect="auto",
        cmap="YlOrRd",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax_heat.set_xticks(x)
    ax_heat.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=5.5)
    ax_heat.set_yticks(np.arange(H))
    ax_heat.set_yticklabels([f"Head {i + 1}" for i in range(H)], fontsize=6)
    ax_heat.set_xlabel("Days after sowing (DAS)", fontsize=7)
    ax_heat.set_title(
        "Temporal attention weights per head — LT (h64, 2-layer, residual), exp01 N=48",
        fontsize=8,
        pad=4,
    )

    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical", fraction=0.025, pad=0.01)
    cbar.set_label("Attention weight", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    cbar.outline.set_linewidth(0.3)

    das_40_idx = min(range(T), key=lambda i: abs(das_values[i] - 40))
    ax_heat.axvline(das_40_idx, color="#444444", linewidth=0.7, linestyle="--", alpha=0.7)

    # ── Panel B: Line plot ────────────────────────────────────────────────
    ax_line = fig.add_subplot(gs[1])
    panel_label(ax_line, "B")

    for h_idx in range(H):
        ax_line.plot(
            x,
            mean_attn[h_idx],
            color=HEAD_COLORS[h_idx],
            linewidth=0.9,
            alpha=0.65,
            label=f"Head {h_idx + 1}",
        )

    grand_mean = mean_attn.mean(axis=0)
    ax_line.plot(
        x,
        grand_mean,
        color="#111111",
        linewidth=1.8,
        label="Mean (all heads)",
        zorder=5,
    )

    ax_line.set_xticks(x)
    ax_line.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=5.5)
    ax_line.set_ylabel("Attention weight", fontsize=7)
    ax_line.set_xlabel("Days after sowing (DAS)", fontsize=7)
    ax_line.set_title("Average temporal attention profile across heads", fontsize=8, pad=4)
    ax_line.set_xlim(-0.5, T - 0.5)

    ax_line.axvline(
        das_40_idx,
        color="#888888",
        linewidth=0.7,
        linestyle="--",
        alpha=0.8,
    )
    ymax = float(grand_mean.max()) * 1.05
    ax_line.set_ylim(bottom=0, top=ymax * 1.25)
    ax_line.text(
        das_40_idx + 0.2,
        ymax * 1.12,
        "DAS 40\n(early prediction\nthreshold)",
        fontsize=4.5,
        color="#555555",
        va="top",
    )

    ax_line.legend(
        loc="upper left",
        fontsize=5.5,
        ncol=2,
        frameon=False,
        handlelength=1.2,
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading config ...")
    cfg = load_config(CONFIG_PATH)

    # Inject liquid sub-config if absent
    if OmegaConf.select(cfg, "model.liquid", default=None) is None:
        cfg.model.liquid = OmegaConf.create(
            {"hidden_dim": 64, "modality_dim": 64, "n_layers": 2, "head_dim": 64}
        )

    print(f"Loading model from:\n  {CHECKPOINT_PATH}")
    model = load_model(CHECKPOINT_PATH, cfg)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")
    print(f"  CfC layers: {len(model.cfc_layers)}")

    print("Loading dataset (exp01) ...")
    dataset = TimothyDroughtDataset(cfg)
    print(f"  Plants: {len(dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    print("Collecting attention weights ...")
    attn_all = collect_attention_weights(model, dataset, batch_size=8, device=device)
    print(f"  Shape: {attn_all.shape}  (plants x heads x timepoints)")

    mean_attn = attn_all.mean(axis=0)  # (H, T)

    print("\nPer-head statistics:")
    for h in range(mean_attn.shape[0]):
        peak_das = DAS_VALUES[int(np.argmax(mean_attn[h]))]
        print(
            f"  Head {h + 1}: "
            f"min={mean_attn[h].min():.4f}  "
            f"max={mean_attn[h].max():.4f}  "
            f"peak DAS={peak_das}"
        )
    grand_mean = mean_attn.mean(axis=0)
    peak_das_mean = DAS_VALUES[int(np.argmax(grand_mean))]
    print(f"  Grand mean peak DAS: {peak_das_mean}")

    print("\nBuilding figure ...")
    fig = make_figure(mean_attn, DAS_VALUES)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "fig_lt_attention_weights.png"
    pdf_path = OUT_DIR / "fig_lt_attention_weights.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"\nSaved:\n  {png_path}\n  {pdf_path}")


if __name__ == "__main__":
    main()
