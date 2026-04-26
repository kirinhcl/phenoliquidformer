"""Visualize how phenological position (phi) modulates CfC time constants (tau)
for the LiquidTransformerModel (h64, n_layers=2, residual).

Panel A: phi_proj embedding magnitude (L2 norm) vs phi [0, 1] — Layer 1.
Panel B: phi_proj embedding magnitude (L2 norm) vs phi [0, 1] — Layer 2.
Panel C: Distribution of t_interp at early / mid / late growth stages (Layer 1, violin plots).
Panel D: Distribution of t_interp at early / mid / late growth stages (Layer 2, violin plots).

Saves:
    paper/figures/final/fig_lt_tau_modulation.png
    paper/figures/final/fig_lt_tau_modulation.pdf

Usage:
    python scripts/visualize_lt_tau_modulation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

# ── project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

from src.model.liquid_transformer import LiquidTransformerModel
from src.utils.config import load_config

# ── constants ─────────────────────────────────────────────────────────────────
CHECKPOINT = PROJECT_ROOT / "results/lt_h64_n2_res/exp01/fold_0/best_model_state.pt"
CONFIG_PATH = PROJECT_ROOT / "configs/timothy.yaml"
OUT_DIR = PROJECT_ROOT / "paper/figures/final"
N_PHI = 200          # resolution for phi sweep panels
N_SAMPLES = 512      # random backbone draws per phi value for violin panels
DEVICE = "cpu"

# Nature NPG stage palette
STAGE_COLORS: dict[str, str] = {
    "Early\n(φ < 0.3)": "#4878CF",
    "Mid\n(0.3 ≤ φ < 0.7)": "#6ACC65",
    "Late\n(φ ≥ 0.7)": "#D65F5F",
}
# Distinct colours per layer for phi_proj norm curves
LAYER_COLORS = ["#2D6A4F", "#7B2D8B"]


# ── model loading ─────────────────────────────────────────────────────────────

def load_model() -> LiquidTransformerModel:
    cfg = load_config(str(CONFIG_PATH))
    # Ensure liquid config matches checkpoint architecture
    if OmegaConf.select(cfg, "model.liquid", default=None) is None:
        cfg.model.liquid = OmegaConf.create(
            {"hidden_dim": 64, "modality_dim": 64, "n_layers": 2, "head_dim": 64}
        )
    OmegaConf.update(cfg, "model.liquid.hidden_dim", 64)
    OmegaConf.update(cfg, "model.liquid.n_layers", 2)
    model = LiquidTransformerModel(role="teacher", cfg=cfg)
    state = torch.load(str(CHECKPOINT), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# ── phi_proj norm sweep ───────────────────────────────────────────────────────

def phi_proj_norms(cell, phi_vals: np.ndarray) -> np.ndarray:
    """Return L2 norm of phi_proj(phi) for each phi in phi_vals."""
    phi_t = torch.tensor(phi_vals, dtype=torch.float32).unsqueeze(-1)  # (N, 1)
    with torch.no_grad():
        emb = cell.phi_proj(phi_t)   # (N, backbone_units)
    return emb.norm(dim=-1).numpy()


# ── t_interp distributions ────────────────────────────────────────────────────

def compute_t_interp_for_phis(
    cell,
    phi_vals: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute t_interp over random backbone activations for a set of phi values.

    For each phi in phi_vals we draw n_samples random backbone activation vectors
    and compute t_interp = sigmoid(time_a * ts + time_b).  The unit time step
    ts=1 is used so that t_interp reflects the trained temporal sensitivity.

    Returns flat array of all t_interp values (len(phi_vals) * n_samples * hidden_size).
    """
    backbone_units = cell.phi_proj.weight.shape[1]

    backbone_noise = torch.tensor(
        rng.standard_normal((n_samples, backbone_units)).astype(np.float32)
    )
    ts = torch.ones(n_samples, 1)   # unit time step, broadcasts over hidden dim

    collected: list[np.ndarray] = []
    with torch.no_grad():
        for phi in phi_vals:
            phi_t = torch.full((n_samples,), float(phi))
            phi_emb = cell.phi_proj(phi_t.unsqueeze(-1))    # (B, backbone_units)
            x_phi = backbone_noise + phi_emb                 # (B, backbone_units)
            ta = cell.time_a(x_phi)                          # (B, hidden_size)
            tb = cell.time_b(x_phi)                          # (B, hidden_size)
            t_interp = torch.sigmoid(ta * ts + tb)           # (B, hidden_size)
            collected.append(t_interp.reshape(-1).numpy())

    return np.concatenate(collected)


# ── violin helper ─────────────────────────────────────────────────────────────

def draw_half_violin(
    ax: plt.Axes,
    data: np.ndarray,
    x_pos: float,
    color: str,
    width: float = 0.38,
) -> None:
    """Right-facing half violin with IQR bar and median dot."""
    kde = gaussian_kde(data, bw_method=0.12)
    y_grid = np.linspace(np.percentile(data, 0.5), np.percentile(data, 99.5), 300)
    density = kde(y_grid)
    density = density / density.max() * width

    ax.fill_betweenx(y_grid, x_pos, x_pos + density, color=color, alpha=0.72, lw=0)
    ax.plot(x_pos + density, y_grid, color=color, lw=0.7, alpha=0.9)

    q25, q50, q75 = np.percentile(data, [25, 50, 75])
    ax.plot([x_pos, x_pos], [q25, q75],
            color="white", lw=2.2, solid_capstyle="round", zorder=3)
    ax.plot([x_pos, x_pos], [q25, q75],
            color=color, lw=1.2, solid_capstyle="round", zorder=4)
    ax.scatter([x_pos], [q50],
               s=14, color="white", zorder=5, edgecolors=color, linewidths=1.0)


# ── NPG style ─────────────────────────────────────────────────────────────────

def apply_npg_style() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.2,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ── figure assembly ───────────────────────────────────────────────────────────

def make_figure(
    phi_sweep: np.ndarray,
    norms_per_layer: list[np.ndarray],
    stage_data_per_layer: list[dict[str, np.ndarray]],
) -> plt.Figure:
    """Build 2x2 figure: (phi_proj norms | t_interp violins) for each of the 2 layers.

    Args:
        phi_sweep: (N_PHI,) array of phi values in [0, 1].
        norms_per_layer: list of 2 arrays, each (N_PHI,) — phi_proj L2 norms.
        stage_data_per_layer: list of 2 dicts mapping stage label -> t_interp samples.
    """
    apply_npg_style()

    fig = plt.figure(figsize=(7.2, 5.8))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        wspace=0.44, hspace=0.52,
        left=0.09, right=0.97, top=0.91, bottom=0.10,
    )

    panel_labels = [("a", "b"), ("c", "d")]

    for layer_idx in range(2):
        norms = norms_per_layer[layer_idx]
        stage_data = stage_data_per_layer[layer_idx]
        color = LAYER_COLORS[layer_idx]
        layer_num = layer_idx + 1

        # ── Phi-proj norm sweep ───────────────────────────────────────────
        ax_a = fig.add_subplot(gs[layer_idx, 0])

        ax_a.fill_between(phi_sweep, 0, norms, color=color, alpha=0.14)
        ax_a.plot(phi_sweep, norms, color=color, lw=1.6)

        y_top = norms.max() * 1.08
        ax_a.set_ylim(0, y_top)

        ax_a.axvline(0.3, color="#aaaaaa", lw=0.75, ls="--", zorder=0)
        ax_a.axvline(0.7, color="#aaaaaa", lw=0.75, ls="--", zorder=0)

        label_y = y_top * 0.97
        for x_mid, stage_label, stage_color in [
            (0.15, "Early",  "#4878CF"),
            (0.50, "Mid",    "#6ACC65"),
            (0.85, "Late",   "#D65F5F"),
        ]:
            ax_a.text(x_mid, label_y, stage_label, ha="center", va="top",
                      fontsize=6.5, color=stage_color, fontstyle="italic")

        ax_a.set_xlabel("Phenological position φ")
        ax_a.set_ylabel("‖φ_proj(φ)‖₂  (a.u.)")
        ax_a.set_xlim(0, 1)
        ax_a.set_title(
            panel_labels[layer_idx][0],
            loc="left", fontweight="bold", fontsize=10, pad=4,
        )
        ax_a.text(
            0.5, 1.03,
            f"Layer {layer_num}: φ_proj norm vs phenological position",
            transform=ax_a.transAxes, ha="center", va="bottom",
            fontsize=7.5,
        )

        # ── t_interp violin plots ─────────────────────────────────────────
        ax_b = fig.add_subplot(gs[layer_idx, 1])

        VIOLIN_WIDTH = 0.40
        labels = list(stage_data.keys())
        positions = [1.0, 2.0, 3.0]

        for pos, stage_label in zip(positions, labels):
            stage_color = STAGE_COLORS[stage_label]
            draw_half_violin(ax_b, stage_data[stage_label], pos, stage_color,
                             width=VIOLIN_WIDTH)
            med = np.median(stage_data[stage_label])
            ax_b.text(pos + VIOLIN_WIDTH + 0.08, med, f"{med:.2f}",
                      va="center", ha="left", fontsize=6.0, color="#444")

        ax_b.set_xticks(positions)
        ax_b.set_xticklabels(labels, fontsize=6.5)
        ax_b.set_xlim(0.55, 3.0 + VIOLIN_WIDTH + 0.30)
        ax_b.set_ylim(0, 1)
        ax_b.set_ylabel("t_interp  (interpolation factor)")
        ax_b.set_title(
            panel_labels[layer_idx][1],
            loc="left", fontweight="bold", fontsize=10, pad=4,
        )
        ax_b.text(
            0.5, 1.03,
            f"Layer {layer_num}: t_interp distribution by growth stage",
            transform=ax_b.transAxes, ha="center", va="bottom",
            fontsize=7.5,
        )

    # Shared figure title
    fig.suptitle(
        "Phenological position modulates CfC liquid time constants\n"
        "LiquidTransformer (h64, 2-layer, residual)",
        fontsize=9, y=0.98, va="top",
    )

    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading LiquidTransformerModel (h64, n_layers=2, fold 0) ...")
    model = load_model()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  CfC layers: {len(model.cfc_layers)}")

    phi_sweep = np.linspace(0.0, 1.0, N_PHI)

    phi_early = np.linspace(0.00, 0.299, 60)
    phi_mid   = np.linspace(0.30, 0.699, 80)
    phi_late  = np.linspace(0.70, 1.000, 60)

    norms_per_layer: list[np.ndarray] = []
    stage_data_per_layer: list[dict[str, np.ndarray]] = []

    rng = np.random.default_rng(42)

    for layer_idx in range(len(model.cfc_layers)):
        cell = model.cfc_layers[layer_idx].cell
        layer_num = layer_idx + 1

        print(f"\nLayer {layer_num}:")
        print(f"  Sweeping phi_proj norms over {N_PHI} phi values ...")
        norms = phi_proj_norms(cell, phi_sweep)
        norms_per_layer.append(norms)

        print(f"  Computing t_interp distributions per growth stage ...")
        stage_data: dict[str, np.ndarray] = {
            "Early\n(φ < 0.3)":       compute_t_interp_for_phis(
                cell, phi_early, N_SAMPLES // 4, rng),
            "Mid\n(0.3 ≤ φ < 0.7)":   compute_t_interp_for_phis(
                cell, phi_mid,   N_SAMPLES // 4, rng),
            "Late\n(φ ≥ 0.7)":         compute_t_interp_for_phis(
                cell, phi_late,  N_SAMPLES // 4, rng),
        }
        stage_data_per_layer.append(stage_data)

        for label, vals in stage_data.items():
            q25, q50, q75 = np.percentile(vals, [25, 50, 75])
            print(
                f"    {label.replace(chr(10), ' '):30s}  "
                f"median={q50:.3f}  IQR=[{q25:.3f}, {q75:.3f}]  n={len(vals):,}"
            )

    print("\nRendering figure ...")
    fig = make_figure(phi_sweep, norms_per_layer, stage_data_per_layer)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        out_path = OUT_DIR / f"fig_lt_tau_modulation{suffix}"
        fig.savefig(out_path)
        print(f"  Saved -> {out_path}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
