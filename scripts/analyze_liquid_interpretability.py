#!/usr/bin/env python3
"""Interpretability analysis for the Liquid Transformer main model.

Three analyses unique to continuous-time models:
    1. Temporal Gradient Saliency: d(yield)/d(input_t) per modality per timestep
    2. Latent Trajectory Phase Space: PCA visualization of h(t) by WHC
    3. Liquid Time Constants: tau distribution from trained PhenologyCfC cells

For each LOPO fold, we load that fold's trained checkpoint and compute saliency
on its held-out test plant. This yields 48 per-plant saliency arrays (one per
fold) that we aggregate for Fig 7 Panel B.

Usage:
    python scripts/analyze_liquid_interpretability.py \\
        --model-dir results/lt_h64_n2_res/exp01 \\
        --override model.liquid.hidden_dim=64 model.liquid.n_layers=2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from src.data.dataset import TimothyDroughtDataset
from src.model.liquid_transformer import LiquidTransformerModel
from src.model.phenology_cfc import PhenologyCfCCell, _lecun_tanh
from src.training.cv import LeaveOnePlantOutCV
from src.utils.config import load_config

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.linewidth": 0.8, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

OUT = Path("results/liquid_interpretability")
FIG_OUT = Path("paper/figures/output")
OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)

WHC_COLORS = {
    0.25: "#d73027", 0.30: "#f46d43", 0.40: "#fdae61",
    0.50: "#fee090", 0.70: "#abd9e9", 0.90: "#4575b4",
}

MODALITIES = ["image", "fluor", "env"]


def collate_fn(batch):
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


def load_fold_model(model_dir: Path, fold_id: int, cfg, device) -> LiquidTransformerModel:
    ckpt_path = model_dir / f"fold_{fold_id}" / "best_model_state.pt"
    model = LiquidTransformerModel(role="teacher", cfg=cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.train(False)
    return model


def saliency_for_plant(model, batch: dict, device):
    batch_dev = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    images = batch_dev["images"].detach().clone().requires_grad_(True)
    fluor = batch_dev["fluorescence"].detach().clone().requires_grad_(True)
    env = batch_dev["environment"].detach().clone().requires_grad_(True)

    batch_dev["images"] = images
    batch_dev["fluorescence"] = fluor
    batch_dev["environment"] = env

    model.zero_grad()
    out = model(batch_dev, t_cut=None)
    dw_pred = out["dw_pred"]
    if dw_pred.ndim > 0:
        dw_pred = dw_pred.sum()
    dw_pred.backward()

    active = batch_dev["active_mask"][0].cpu().numpy().astype(bool)
    T = int(active.sum())

    img_g = images.grad[0].norm(dim=(-1, -2)).detach().cpu().numpy()
    flu_g = fluor.grad[0].norm(dim=-1).detach().cpu().numpy()
    env_g = env.grad[0].norm(dim=-1).detach().cpu().numpy()

    saliency_full = np.stack([img_g, flu_g, env_g], axis=-1)
    saliency = saliency_full[active]
    das_vec = batch_dev["temporal_positions"][0].detach().cpu().numpy()[active]
    return saliency, das_vec, T


def aggregate_saliency(per_plant_saliency, per_plant_das):
    ref_das = per_plant_das[0]
    T = len(ref_das)
    n_plants = len(per_plant_saliency)

    stack = np.zeros((n_plants, T, len(MODALITIES)))
    for i, sal in enumerate(per_plant_saliency):
        T_i = min(sal.shape[0], T)
        for m in range(len(MODALITIES)):
            col = sal[:T_i, m]
            mx = col.max()
            if mx > 0:
                col = col / mx
            stack[i, :T_i, m] = col

    mean_matrix = stack.mean(axis=0).T  # (n_modalities, T)
    return mean_matrix, ref_das


def run_saliency_all_folds(model_dir, dataset, cv, cfg, device, max_folds):
    per_plant_saliency = []
    per_plant_das = []
    per_plant_whc = []
    per_plant_genotype = []
    per_plant_plant_id = []

    loader_all = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    batches = list(loader_all)

    for fold_idx, (_, _, test_idx) in enumerate(cv.split()):
        if max_folds is not None and fold_idx >= max_folds:
            break
        plant_pos = int(test_idx[0])
        batch = batches[plant_pos]

        model = load_fold_model(model_dir, fold_idx, cfg, device)
        saliency, das_vec, T = saliency_for_plant(model, batch, device)

        per_plant_saliency.append(saliency)
        per_plant_das.append(das_vec)
        per_plant_whc.append(float(batch["whc_target"][0]))
        per_plant_genotype.append(batch["genotype"][0])
        per_plant_plant_id.append(batch["plant_id"][0])

        if (fold_idx + 1) % 8 == 0 or fold_idx == 0:
            print(f"  fold {fold_idx:2d}: plant={per_plant_plant_id[-1]} "
                  f"WHC={per_plant_whc[-1]:.2f} T={T}", flush=True)

    return {
        "saliency": per_plant_saliency,
        "das": per_plant_das,
        "whc": per_plant_whc,
        "genotype": per_plant_genotype,
        "plant_id": per_plant_plant_id,
    }


def plot_saliency_heatmap(mean_matrix, ref_das):
    fig, ax = plt.subplots(figsize=(7, 3.0))
    im = ax.imshow(mean_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(len(MODALITIES)))
    ax.set_yticklabels([m.capitalize() for m in MODALITIES], fontsize=9)
    ax.set_xticks(range(len(ref_das)))
    ax.set_xticklabels([f"{int(d)}" for d in ref_das], fontsize=7)
    ax.set_xlabel("DAS")
    ax.set_title("Liquid Transformer temporal saliency (mean over LOPO folds)", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Normalized saliency", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "fig_saliency_real.pdf")
    fig.savefig(FIG_OUT / "fig_saliency_real.png", dpi=150)
    plt.close(fig)
    print("  saved fig_saliency_real.pdf/png")


class _SigmaRecorder:
    """Monkey-patches PhenologyCfCCell.forward to capture sigma (time-interp factor).

    Usage:
        with _SigmaRecorder(model) as rec:
            model(batch)
        sigmas = rec.per_step    # list of (B, hidden) tensors per call
    """

    def __init__(self, model):
        self.cells = [m for m in model.modules() if isinstance(m, PhenologyCfCCell)]
        self._orig = []
        self.per_cell: list[list[torch.Tensor]] = [[] for _ in self.cells]

    def __enter__(self):
        for idx, cell in enumerate(self.cells):
            self._orig.append(cell.forward)
            bucket = self.per_cell[idx]

            def patched(x, hx, ts, phi, _cell=cell, _bucket=bucket):
                x_cat = torch.cat([x, hx], dim=-1)
                backbone_out = _lecun_tanh(_cell.backbone(x_cat))
                phi_emb = _cell.phi_proj(phi.unsqueeze(-1))
                x_phi = backbone_out + phi_emb
                ta = _cell.time_a(x_phi)
                tb = _cell.time_b(x_phi)
                ts_exp = ts.unsqueeze(-1)
                t_interp = torch.sigmoid(ta * ts_exp + tb)
                _bucket.append(t_interp.detach().cpu())
                f1 = torch.tanh(_cell.ff1(backbone_out))
                f2 = torch.tanh(_cell.ff2(backbone_out))
                h_new = f1 * (1.0 - t_interp) + t_interp * f2
                return h_new, h_new

            cell.forward = patched
        return self

    def __exit__(self, exc_type, exc, tb):
        for cell, orig in zip(self.cells, self._orig):
            cell.forward = orig


def extract_latent_trajectories(model, dataset, device):
    """Forward every plant through `model` once. Capture full h_seq and sigma(t).

    Returns per-plant lists:
        trajectories: list of (T, hidden) ndarrays
        sigmas_last:  list of (T, hidden) ndarrays — sigma from the LAST CfC layer
        whc_list, genotype_list, das_list
    """
    model.to(device)
    model.train(False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    trajectories, sigmas_last = [], []
    whc_list, genotype_list, das_list = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with _SigmaRecorder(model) as rec:
                out = model(batch_dev)
            T = int(batch_dev["active_mask"][0].sum().item())
            h_seq = out["h_seq"][0, :T].cpu().numpy()
            # rec.per_cell[-1] is a list of T tensors (each shape (B=1, hidden))
            last_sigma = torch.cat(rec.per_cell[-1], dim=0)[:T].numpy()  # (T, hidden)
            trajectories.append(h_seq)
            sigmas_last.append(last_sigma)
            whc_list.append(float(batch["whc_target"][0]))
            genotype_list.append(batch["genotype"][0])
            das_list.append(batch_dev["temporal_positions"][0, :T].cpu().numpy())

    return trajectories, sigmas_last, whc_list, genotype_list, das_list


def compute_fluor_rho_per_das(experiment: str, out_path: Path) -> None:
    """Save per-DAS |Spearman rho(QY_max, WHC)| from the fixed FCQ file for `experiment`."""
    if spearmanr is None:
        print("  scipy not available; skipping rho computation")
        return
    import pandas as pd
    fcq_dir = {
        "exp01": Path("data/2023-Timothy-01-Nonvernalized"),
        "exp02": Path("data/2024-Timothy-02-Vernalized"),
        "exp03": Path("data/2024-Timothy-03-Regrowth"),
    }[experiment]
    stem = {
        "exp01": "FCQ_Timothy-01",
        "exp02": "FCQ_Timothy-02",
        "exp03": "FCQ_Timothy-03",
    }[experiment]
    fixed = fcq_dir / f"{stem}_fixed.xlsx"
    src = fixed if fixed.exists() else (fcq_dir / f"{stem}.xlsx")
    df = pd.read_excel(src)
    df.columns = [str(c).strip() for c in df.columns]
    df["WHC"] = df["Treatment"].str.extract(r"(\d+)").astype(float) / 100

    rows = []
    for das in sorted(df["DAS"].dropna().unique()):
        sub = df[df["DAS"] == das][["QY_max", "WHC"]].dropna()
        if len(sub) < 5:
            continue
        rho, p = spearmanr(sub["QY_max"], sub["WHC"])
        rows.append({"das": int(das), "abs_rho": float(abs(rho)),
                     "p_value": float(p), "n": int(len(sub))})
    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["das", "abs_rho", "p_value", "n"])
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {out_path} (source={src.name})")


def plot_latent_trajectories(trajectories, whc_list, das_list):
    all_points = np.vstack(trajectories)
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_points)

    traj_2d = []
    offset = 0
    for traj in trajectories:
        T = traj.shape[0]
        traj_2d.append(all_2d[offset:offset + T])
        offset += T

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for traj, whc in zip(traj_2d, whc_list):
        color = WHC_COLORS.get(whc, "#999")
        ax.plot(traj[:, 0], traj[:, 1], "-", color=color, alpha=0.4, lw=1)
        ax.scatter(traj[-1, 0], traj[-1, 1], c=color, s=40, zorder=5,
                   edgecolors="black", linewidth=0.5)
    for whc, color in sorted(WHC_COLORS.items()):
        ax.scatter([], [], c=color, s=40, label=f"WHC-{int(whc*100)}%")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("A. Latent trajectories by WHC", fontweight="bold")

    ax = axes[1]
    for traj, das in zip(traj_2d, das_list):
        sc = ax.scatter(traj[:, 0], traj[:, 1], c=das, cmap="viridis",
                        s=15, alpha=0.6, vmin=0, vmax=55)
        ax.plot(traj[:, 0], traj[:, 1], "-", color="gray", alpha=0.15, lw=0.5)
    fig.colorbar(sc, ax=ax, label="DAS")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("B. Trajectories colored by time", fontweight="bold")

    fig.suptitle("Phase space of Liquid Transformer hidden states",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "fig_latent_trajectories_lt.pdf")
    fig.savefig(FIG_OUT / "fig_latent_trajectories_lt.png", dpi=150)
    plt.close(fig)
    return {
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "n_plants": len(trajectories),
    }


def analyze_time_constants(model):
    tc = {}
    for name, param in model.named_parameters():
        if "time_a" in name or "time_b" in name or "phi_proj" in name:
            vals = param.detach().cpu().numpy().flatten()
            tc[name] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "values": vals.tolist(),
            }
    return tc


def plot_time_constants(time_constants):
    all_vals: dict[str, list[float]] = {}
    for name, data in time_constants.items():
        short = name.split(".")[-1]
        all_vals.setdefault(short, []).extend(data["values"])

    if not all_vals:
        print("  no time constants found")
        return

    fig, axes = plt.subplots(1, len(all_vals), figsize=(4 * len(all_vals), 3.5))
    if len(all_vals) == 1:
        axes = [axes]
    colors = ["#2166ac", "#b2182b", "#1b9e77", "#e6550d"]
    for ax, ((name, vals), color) in zip(axes, zip(all_vals.items(), colors)):
        vals = np.array(vals)
        ax.hist(vals, bins=30, color=color, alpha=0.75, edgecolor="black", linewidth=0.4)
        ax.axvline(vals.mean(), color="black", ls="--", lw=1.3,
                   label=f"mean={vals.mean():.3f}")
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Count")
        ax.set_title(f"{name} (n={len(vals)})", fontweight="bold", fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle("Learned liquid time-constant weights (PhenologyCfC)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "fig_time_constants_lt.pdf")
    fig.savefig(FIG_OUT / "fig_time_constants_lt.png", dpi=150)
    plt.close(fig)
    print("  saved fig_time_constants_lt.pdf/png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/timothy.yaml")
    parser.add_argument("--model-dir", default="results/lt_h64_n2_res/exp01")
    parser.add_argument("--experiment", default="exp01")
    parser.add_argument("--max-folds", type=int, default=None,
                        help="Limit number of folds (for quick testing)")
    parser.add_argument("--override", nargs="*",
                        default=["model.liquid.hidden_dim=64", "model.liquid.n_layers=2"],
                        help="Config overrides matching training time")
    parser.add_argument("--skip-latent", action="store_true")
    parser.add_argument("--skip-tau", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu")

    overrides = [f"data.experiment={args.experiment}"] + list(args.override)
    cfg = load_config(args.config, overrides=overrides)
    if "liquid" not in cfg.model:
        cfg.model.liquid = OmegaConf.create({"hidden_dim": 64, "n_layers": 2, "head_dim": 64})

    model_dir = Path(args.model_dir)
    with open(model_dir / "summary.json") as f:
        summary = json.load(f)
    print(f"Loaded summary: MAE={summary['mean_test_mae_g']:.3f}g "
          f"R2={summary['r2_global']:.3f}")

    dataset = TimothyDroughtDataset(cfg)
    plant_meta = dataset.plant_meta
    valid_mask = plant_meta["dw_g"].notna().values
    plant_meta_valid = plant_meta[valid_mask].reset_index(drop=True)
    print(f"Dataset: {len(plant_meta_valid)} plants with valid DW in {args.experiment}")

    cv = LeaveOnePlantOutCV(plant_meta_valid, args.experiment, seed=cfg.training.seed)

    print("\n=== Analysis 1: Temporal Gradient Saliency (per-fold) ===")
    sal_data = run_saliency_all_folds(model_dir, dataset, cv, cfg, device, args.max_folds)

    mean_matrix, ref_das = aggregate_saliency(sal_data["saliency"], sal_data["das"])
    np.save(OUT / "saliency_mean.npy", mean_matrix)
    np.save(OUT / "saliency_das.npy", ref_das)

    lengths = [s.shape[0] for s in sal_data["saliency"]]
    T_ref = min(lengths)
    stacked = np.stack([s[:T_ref] for s in sal_data["saliency"]], axis=0)
    np.save(OUT / "saliency_per_plant.npy", stacked)

    with open(OUT / "saliency_meta.json", "w") as f:
        json.dump({
            "modalities": MODALITIES,
            "n_plants": len(sal_data["saliency"]),
            "das": ref_das.tolist(),
            "whc": sal_data["whc"],
            "genotype": sal_data["genotype"],
            "plant_id": sal_data["plant_id"],
        }, f, indent=2)

    plot_saliency_heatmap(mean_matrix, ref_das)

    print("\nMean saliency matrix (rows: modalities, cols: DAS):")
    print("DAS:", [int(d) for d in ref_das])
    for m_idx, m_name in enumerate(MODALITIES):
        vals = ["%.2f" % v for v in mean_matrix[m_idx]]
        print(f"  {m_name:>6}: {vals}")

    if not args.skip_latent:
        print("\n=== Analysis 2: Latent Trajectory Phase Space + Sigma (fold 0) ===")
        model0 = load_fold_model(model_dir, 0, cfg, device)
        traj, sigmas, whc_l, geno_l, das_l = extract_latent_trajectories(model0, dataset, device)
        info = plot_latent_trajectories(traj, whc_l, das_l)
        print(f"  PCA explained variance (PC1, PC2): {info['pca_explained_variance'][:2]}")

        # Pad trajectories and sigmas to common T for downstream figures
        lengths = [h.shape[0] for h in traj]
        T_ref = min(lengths)
        latent_stack = np.stack([h[:T_ref] for h in traj], axis=0)   # (N, T, hidden)
        sigma_stack = np.stack([s[:T_ref] for s in sigmas], axis=0)  # (N, T, hidden)
        das_ref = das_l[0][:T_ref].astype(int)
        np.save(OUT / "latent_per_plant.npy", latent_stack)
        np.save(OUT / "sigma_per_plant.npy", sigma_stack)
        np.save(OUT / "latent_das.npy", das_ref)
        with open(OUT / "latent_meta.json", "w") as f:
            json.dump({
                "n_plants": len(traj),
                "T": int(T_ref),
                "hidden_dim": int(latent_stack.shape[-1]),
                "das": das_ref.tolist(),
                "whc": whc_l,
                "genotype": geno_l,
            }, f, indent=2)
        print(f"  saved latent_per_plant.npy {latent_stack.shape} "
              f"sigma_per_plant.npy {sigma_stack.shape}")

    print("\n=== Analysis 2b: Per-DAS |rho|(QY_max, WHC) ===")
    compute_fluor_rho_per_das(args.experiment, OUT / "fluor_rho_per_das.csv")

    if not args.skip_tau:
        print("\n=== Analysis 3: Liquid Time Constants (fold 0) ===")
        model0 = load_fold_model(model_dir, 0, cfg, device)
        tc = analyze_time_constants(model0)
        with open(OUT / "time_constants.json", "w") as f:
            json.dump(tc, f, indent=2)
        plot_time_constants(tc)
        print(f"  extracted {len(tc)} tau-related parameter groups")

    print(f"\nAll artifacts saved under {OUT}/")


if __name__ == "__main__":
    main()
