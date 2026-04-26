#!/usr/bin/env python3
"""Yield prediction ablation experiments for Liquid NN.

Tests:
    1. Modality ablation: drop image / drop fluor / drop VI
    2. Context ablation: drop WHC / drop genotype / drop both
    3. Distillation ablation: student trained from scratch (no teacher)

Usage:
    python scripts/run_yield_ablations.py
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from ncps.torch import CfC
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.dataset import TimothyDroughtDataset
from src.model.encoder import ViewAggregation
from src.model.liquid_model import LiquidYieldModel, ModalityProjector, YieldHead
from src.training.cv import LeaveOnePlantOutCV
from src.training.distillation_loss import YieldLoss
from src.utils.config import load_config


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


def sample_t_cut(times, rng):
    if rng.random() < 0.5:
        return None
    if len(times) < 4:
        return None
    return float(times[rng.randint(3, len(times) - 1)])


class AblationModel(nn.Module):
    """Flexible Liquid model for ablation testing."""

    def __init__(self, cfg, use_image=True, use_fluor=True, use_vi=True,
                 use_whc=True, use_genotype=True):
        super().__init__()
        mcfg = cfg.model if "model" in cfg else cfg
        hidden_dim = OmegaConf.select(mcfg, "liquid.hidden_dim", default=32)
        mod_dim = OmegaConf.select(mcfg, "liquid.modality_dim", default=32)

        self.use_image = use_image
        self.use_fluor = use_fluor
        self.use_vi = use_vi
        self.use_whc = use_whc
        self.use_genotype = use_genotype

        if use_image:
            self.view_agg = ViewAggregation(mcfg.encoder_output_dim)
            self.image_proj = nn.Sequential(
                nn.LayerNorm(mcfg.modality.image_dim), nn.Linear(mcfg.modality.image_dim, mod_dim), nn.Tanh())
        if use_fluor:
            self.fluor_proj = nn.Sequential(
                nn.LayerNorm(mcfg.modality.fluor_dim), nn.Linear(mcfg.modality.fluor_dim, mod_dim), nn.Tanh())
        if use_vi:
            self.vi_proj = nn.Sequential(
                nn.LayerNorm(mcfg.modality.vi_dim), nn.Linear(mcfg.modality.vi_dim, mod_dim), nn.Tanh())

        obs_dim = mod_dim * (int(use_image) + int(use_fluor) + int(use_vi))
        ctx_dim = int(use_whc) + 2 * int(use_genotype)
        per_step = obs_dim + 1 + ctx_dim  # +1 for dt

        self.cfc = CfC(per_step, hidden_dim, batch_first=True, return_sequences=True)
        self.yield_head = YieldHead(hidden_dim, OmegaConf.select(mcfg, "liquid.head_dim", default=64))
        self.ctx_dim = ctx_dim

    def forward(self, batch, t_cut=None):
        device = next(self.parameters()).device
        parts = []
        if self.use_image:
            img = self.view_agg(batch["images"], batch["image_mask"])
            parts.append(self.image_proj(img))
        if self.use_fluor:
            parts.append(self.fluor_proj(batch["fluorescence"]))
        if self.use_vi:
            parts.append(self.vi_proj(batch["vi"]))
        obs = torch.cat(parts, dim=-1)

        times = batch["temporal_positions"][0]
        dt = torch.zeros_like(times)
        dt[1:] = times[1:] - times[:-1]
        dt_norm = dt / max(float(dt.max().item()), 1.0)
        dt_feat = dt_norm.unsqueeze(0).unsqueeze(-1).expand(obs.shape[0], -1, 1)

        if t_cut is not None:
            T = int((times <= t_cut).sum().item())
            obs, dt_feat = obs[:, :T], dt_feat[:, :T]
        else:
            T = obs.shape[1]

        step_in = torch.cat([obs, dt_feat], dim=-1)
        if self.ctx_dim > 0:
            ctx_parts = []
            B = obs.shape[0]
            if self.use_whc:
                ctx_parts.append(batch["whc_target"].view(B, 1).float())
            if self.use_genotype:
                geno = torch.zeros(B, 2, device=device)
                for i, g in enumerate(batch["genotype"]):
                    if "Jauniai" in g: geno[i, 0] = 1.0
                    elif "Noreng" in g: geno[i, 1] = 1.0
                ctx_parts.append(geno)
            ctx = torch.cat(ctx_parts, dim=-1).unsqueeze(1).expand(-1, T, -1)
            step_in = torch.cat([step_in, ctx], dim=-1)

        active = batch["active_mask"][:, :T].unsqueeze(-1).float()
        step_in = step_in * active
        h_seq, h_final = self.cfc(step_in)
        out = self.yield_head(h_final)
        return {"dw_pred": out["dw"], "flowering_pred": out["flowering"]}


def run_ablation(name, dataset, plant_meta, cfg, dw_mean, dw_std,
                 output_dir, max_folds=None, **model_kwargs):
    print(f"\n{'='*50}")
    print(f"Ablation: {name}")
    print(f"{'='*50}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cv = LeaveOnePlantOutCV(plant_meta, "exp01", seed=42)
    n_folds = max_folds or cv.n_folds
    criterion = YieldLoss()

    fold_preds, fold_trues = [], []
    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(cv.split()):
        if fold_idx >= n_folds:
            break
        train_loader = DataLoader(Subset(dataset, train_idx.tolist()),
                                  batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(Subset(dataset, val_idx.tolist()),
                                batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(Subset(dataset, test_idx.tolist()),
                                 batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_fn)

        model = AblationModel(cfg, **model_kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        rng = random.Random(42 + fold_idx)

        best_val, best_state, patience_counter = float("inf"), None, 0
        for epoch in range(1, cfg.training.max_epochs + 1):
            model.train()
            for batch in train_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                t_cut = sample_t_cut(batch["temporal_positions"][0].cpu().numpy(), rng)
                optimizer.zero_grad()
                out = model(batch, t_cut=t_cut)
                dw_target = (batch["dw_target"] - dw_mean) / dw_std
                flower_target = torch.full_like(dw_target, float("nan"))
                loss, _ = criterion(out["dw_pred"], dw_target, out["flowering_pred"], flower_target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
                optimizer.step()

            model.train(False)
            val_maes = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    out = model(batch)
                    pred = out["dw_pred"] * dw_std + dw_mean
                    val_maes.extend((pred - batch["dw_target"]).abs().cpu().tolist())
            val_mae = float(np.mean(val_maes))
            if val_mae < best_val:
                best_val = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= cfg.training.patience:
                break

        if best_state:
            model.load_state_dict(best_state)
        model.to(device)
        model.train(False)
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = model(batch)
                pred = out["dw_pred"] * dw_std + dw_mean
                fold_preds.extend(pred.cpu().tolist())
                fold_trues.extend(batch["dw_target"].cpu().tolist())

    preds = np.array(fold_preds)
    trues = np.array(fold_trues)
    mae = float(np.mean(np.abs(preds - trues)))
    ss_res = np.sum((trues - preds) ** 2)
    ss_tot = np.sum((trues - trues.mean()) ** 2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-8))

    result = {"name": name, "mae": mae, "r2": r2, "n_folds": len(fold_preds), **model_kwargs}
    abl_dir = output_dir / name.replace(" ", "_").lower()
    abl_dir.mkdir(parents=True, exist_ok=True)
    with open(abl_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"  MAE={mae:.3f}g  R²={r2:+.3f}  (n={len(fold_preds)} folds)", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/timothy.yaml")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--output-dir", default="results/yield_ablations")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=["data.experiment=exp01"])
    if "liquid" not in cfg.model:
        cfg.model.liquid = OmegaConf.create({"hidden_dim": 32, "modality_dim": 32, "head_dim": 64})

    dataset = TimothyDroughtDataset(cfg)
    pm = dataset.plant_meta
    pm = pm[pm["dw_g"].notna()].reset_index(drop=True)
    dw_mean, dw_std = float(pm["dw_g"].mean()), float(pm["dw_g"].std() + 1e-8)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    # Full model (reference)
    all_results.append(run_ablation("Full", dataset, pm, cfg, dw_mean, dw_std, output_dir, args.max_folds,
                                     use_image=True, use_fluor=True, use_vi=True, use_whc=True, use_genotype=True))

    # Modality ablation
    all_results.append(run_ablation("Drop Image", dataset, pm, cfg, dw_mean, dw_std, output_dir, args.max_folds,
                                     use_image=False, use_fluor=True, use_vi=True, use_whc=True, use_genotype=True))
    all_results.append(run_ablation("Drop Fluor", dataset, pm, cfg, dw_mean, dw_std, output_dir, args.max_folds,
                                     use_image=True, use_fluor=False, use_vi=True, use_whc=True, use_genotype=True))
    all_results.append(run_ablation("Drop VI", dataset, pm, cfg, dw_mean, dw_std, output_dir, args.max_folds,
                                     use_image=True, use_fluor=True, use_vi=False, use_whc=True, use_genotype=True))

    # Context ablation
    all_results.append(run_ablation("Drop WHC", dataset, pm, cfg, dw_mean, dw_std, output_dir, args.max_folds,
                                     use_image=True, use_fluor=True, use_vi=True, use_whc=False, use_genotype=True))
    all_results.append(run_ablation("Drop Genotype", dataset, pm, cfg, dw_mean, dw_std, output_dir, args.max_folds,
                                     use_image=True, use_fluor=True, use_vi=True, use_whc=True, use_genotype=False))
    all_results.append(run_ablation("Drop All Context", dataset, pm, cfg, dw_mean, dw_std, output_dir, args.max_folds,
                                     use_image=True, use_fluor=True, use_vi=True, use_whc=False, use_genotype=False))

    # Student from scratch (no distillation)
    all_results.append(run_ablation("Image Only (scratch)", dataset, pm, cfg, dw_mean, dw_std, output_dir, args.max_folds,
                                     use_image=True, use_fluor=False, use_vi=False, use_whc=False, use_genotype=False))

    # Summary
    with open(output_dir / "ablation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    full_mae = all_results[0]["mae"]
    print(f"{'Variant':<25} {'MAE(g)':<10} {'R²':<10} {'Δ MAE':<10}")
    print("-" * 55)
    for r in all_results:
        delta = r["mae"] - full_mae
        print(f"{r['name']:<25} {r['mae']:<10.3f} {r['r2']:<+10.3f} {delta:<+10.3f}")


if __name__ == "__main__":
    main()
