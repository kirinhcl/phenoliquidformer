#!/usr/bin/env python3
"""LoRA cross-experiment transfer: adapt exp01-trained Liquid NN to exp02.

Freezes the base model and adds low-rank adapters to CfC layers.
Tests multiple ranks and sample sizes to build a transfer efficiency curve.

Key outputs:
    1. LoRA rank sweep: rank ∈ {2, 4, 8, 16} at full exp02 data
    2. Sample efficiency: n ∈ {5, 10, 20, 30, 40} at best rank
    3. Layer-specific LoRA: only time params vs only ff params vs all
    4. Baselines: direct transfer (no adapt), full fine-tune, train from scratch

Usage:
    python scripts/train_lora_transfer.py \\
        --source-dir results/lt_h64_n2_res/exp01 \\
        --output-dir results/lora_transfer \\
        --model-type liquid_transformer
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from src.data.dataset import TimothyDroughtDataset
from src.model.liquid_model import LiquidYieldModel
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


# ============================================================
# LoRA Implementation
# ============================================================

class LoRALinear(nn.Module):
    """Low-Rank Adapter wrapping an existing Linear layer."""

    def __init__(self, original: nn.Linear, rank: int = 4) -> None:
        super().__init__()
        self.original = original
        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        d_in = original.in_features
        d_out = original.out_features
        self.lora_A = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        self.rank = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = x @ self.lora_A @ self.lora_B
        return base_out + lora_out

    def extra_repr(self) -> str:
        return f"rank={self.rank}, in={self.original.in_features}, out={self.original.out_features}"


def apply_lora(model: nn.Module, rank: int, target_layers: str = "all") -> int:
    """Apply LoRA to specific layers of the CfC model.

    Args:
        model: LiquidYieldModel or LiquidTransformerModel
        rank: LoRA rank
        target_layers: "all", "ff" (ff1/ff2 only), "time" (time_a/time_b only)

    Returns:
        Number of trainable parameters
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze yield head (always adapts)
    for param in model.yield_head.parameters():
        param.requires_grad = True

    lora_count = 0

    # Apply LoRA to CfC cell's linear layers.
    # LiquidYieldModel wraps ncps CfC → cell at layer.rnn_cell
    # LiquidTransformerModel wraps PhenologyCfC → cell at layer.cell
    for layer in model.cfc_layers:
        cell = getattr(layer, "cell", None) or getattr(layer, "rnn_cell", None)
        if cell is None:
            continue

        # Identify target linear layers
        targets = {}
        if target_layers in ("all", "ff"):
            if hasattr(cell, "ff1") and isinstance(cell.ff1, nn.Linear):
                targets["ff1"] = cell.ff1
            if hasattr(cell, "ff2") and isinstance(cell.ff2, nn.Linear):
                targets["ff2"] = cell.ff2
        if target_layers in ("all", "time"):
            if hasattr(cell, "time_a") and isinstance(cell.time_a, nn.Linear):
                targets["time_a"] = cell.time_a
            if hasattr(cell, "time_b") and isinstance(cell.time_b, nn.Linear):
                targets["time_b"] = cell.time_b

        # Also adapt backbone if present
        if target_layers == "all" and hasattr(cell, "backbone"):
            backbone = cell.backbone
            if isinstance(backbone, nn.Linear):
                targets["backbone"] = backbone
            elif isinstance(backbone, (nn.Sequential, nn.ModuleList)):
                for i, sub in enumerate(backbone):
                    if isinstance(sub, nn.Linear):
                        targets[f"backbone_{i}"] = sub

        for name, linear in targets.items():
            lora_layer = LoRALinear(linear, rank=rank)
            # Replace in the cell
            if name == "ff1":
                cell.ff1 = lora_layer
            elif name == "ff2":
                cell.ff2 = lora_layer
            elif name == "time_a":
                cell.time_a = lora_layer
            elif name == "time_b":
                cell.time_b = lora_layer
            elif name == "backbone":
                cell.backbone = lora_layer
            elif name.startswith("backbone_"):
                idx = int(name.split("_")[1])
                cell.backbone[idx] = lora_layer
            lora_count += rank * (linear.in_features + linear.out_features)

    # For LiquidTransformerModel: also adapt gate network and residual_proj when
    # target_layers == "all"
    if target_layers == "all":
        if hasattr(model, "modality_gating") and model.modality_gating is not None:
            gate_net = getattr(model.modality_gating, "gate_network", None)
            if gate_net is not None:
                for param in gate_net.parameters():
                    param.requires_grad = True
        if hasattr(model, "residual_proj") and isinstance(model.residual_proj, nn.Linear):
            for param in model.residual_proj.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable


def load_source_model(source_dir, fold, cfg, device, model_type="liquid"):
    """Load a pre-trained exp01 model.

    Args:
        source_dir: Directory containing fold subdirectories with checkpoints.
        fold: Fold index to load.
        cfg: OmegaConf config.
        device: Torch device.
        model_type: "liquid" for LiquidYieldModel, "liquid_transformer" for
            LiquidTransformerModel.
    """
    if model_type == "liquid_transformer":
        from src.model.liquid_transformer import LiquidTransformerModel
        model = LiquidTransformerModel(role="teacher", cfg=cfg)
    else:
        model = LiquidYieldModel(role="teacher", cfg=cfg)

    ckpt_path = Path(source_dir) / f"fold_{fold}" / "best_model_state.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if any(k.startswith("cfc.") for k in ckpt):
        ckpt = {(k.replace("cfc.", "cfc_layers.0.", 1) if k.startswith("cfc.") else k): v
                for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    return model


def evaluate_on_dataset(model, dataset, dw_mean, dw_std, device):
    """Evaluate model on full dataset, return MAE and R²."""
    model.to(device)
    model.train(False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch)
            dw = out["dw_pred"] * dw_std + dw_mean
            preds.extend(dw.cpu().tolist())
            trues.extend(batch["dw_target"].cpu().tolist())
    preds, trues = np.array(preds), np.array(trues)
    mae = float(np.mean(np.abs(preds - trues)))
    ss_res = float(np.sum((trues - preds) ** 2))
    ss_tot = float(np.sum((trues - trues.mean()) ** 2))
    r2 = float(1 - ss_res / max(ss_tot, 1e-8))
    return mae, r2, preds, trues


def fine_tune(model, train_loader, val_loader, dw_mean_target, dw_std_target,
              cfg, device, max_epochs=100, patience=20):
    """Fine-tune (LoRA or full) on target experiment data."""
    model.to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.lr, weight_decay=cfg.training.weight_decay,
    )
    criterion = YieldLoss()

    best_val_mae = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            dw_target = (batch["dw_target"] - dw_mean_target) / dw_std_target
            B = dw_target.shape[0]
            flower_target = torch.full((B,), float("nan"), device=device)
            optimizer.zero_grad()
            out = model(batch)
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
                dw = out["dw_pred"] * dw_std_target + dw_mean_target
                val_maes.extend((dw - batch["dw_target"]).abs().cpu().tolist())
        val_mae = float(np.mean(val_maes))

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return best_val_mae


def run_lopo_experiment(name, source_dir, cfg, target_dataset, target_meta,
                        dw_mean_t, dw_std_t, device, output_dir,
                        lora_rank=None, lora_layers="all",
                        train_from_scratch=False, full_finetune=False,
                        n_train_samples=None, source_fold=0,
                        model_type="liquid"):
    """Run one LOPO experiment on target data.

    Args:
        model_type: "liquid" for LiquidYieldModel, "liquid_transformer" for
            LiquidTransformerModel. Used when instantiating from scratch and
            when loading the source checkpoint.
    """
    print(f"\n  --- {name} ---", flush=True)

    cv = LeaveOnePlantOutCV(target_meta, "exp02", seed=42)
    fold_preds, fold_trues = [], []

    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(cv.split()):
        # Subsample training data if requested
        if n_train_samples is not None and len(train_idx) > n_train_samples:
            rng = np.random.RandomState(42 + fold_idx)
            train_idx = rng.choice(train_idx, n_train_samples, replace=False)

        train_loader = DataLoader(
            Subset(target_dataset, train_idx.tolist()),
            batch_size=min(8, len(train_idx)), shuffle=True, collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            Subset(target_dataset, val_idx.tolist()),
            batch_size=8, shuffle=False, collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            Subset(target_dataset, test_idx.tolist()),
            batch_size=8, shuffle=False, collate_fn=collate_fn,
        )

        if train_from_scratch:
            if model_type == "liquid_transformer":
                from src.model.liquid_transformer import LiquidTransformerModel
                model = LiquidTransformerModel(role="teacher", cfg=cfg).to(device)
            else:
                model = LiquidYieldModel(role="teacher", cfg=cfg).to(device)
        else:
            model = load_source_model(source_dir, source_fold, cfg, device, model_type=model_type)

        if lora_rank is not None:
            trainable = apply_lora(model, rank=lora_rank, target_layers=lora_layers)
        elif full_finetune:
            trainable = sum(p.numel() for p in model.parameters())
        elif not train_from_scratch:
            # Direct transfer — no training
            model.to(device)
            model.train(False)
            with torch.no_grad():
                for batch in test_loader:
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    out = model(batch)
                    dw = out["dw_pred"] * dw_std_t + dw_mean_t
                    fold_preds.extend(dw.cpu().tolist())
                    fold_trues.extend(batch["dw_target"].cpu().tolist())
            continue
        else:
            trainable = sum(p.numel() for p in model.parameters())

        # Fine-tune
        fine_tune(model, train_loader, val_loader, dw_mean_t, dw_std_t, cfg, device)

        # Test
        model.train(False)
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = model(batch)
                dw = out["dw_pred"] * dw_std_t + dw_mean_t
                fold_preds.extend(dw.cpu().tolist())
                fold_trues.extend(batch["dw_target"].cpu().tolist())

    preds = np.array(fold_preds)
    trues = np.array(fold_trues)
    mae = float(np.mean(np.abs(preds - trues)))
    ss_res = float(np.sum((trues - preds) ** 2))
    ss_tot = float(np.sum((trues - trues.mean()) ** 2))
    r2 = float(1 - ss_res / max(ss_tot, 1e-8))

    result = {"name": name, "mae": mae, "r2": r2, "n_preds": len(preds)}
    if lora_rank:
        result["lora_rank"] = lora_rank
        result["lora_layers"] = lora_layers
    if n_train_samples:
        result["n_train_samples"] = n_train_samples

    print(f"  {name}: MAE={mae:.3f}g  R²={r2:+.3f} (n={len(preds)})", flush=True)

    exp_dir = output_dir / name.replace(" ", "_").replace("=", "").lower()
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/timothy.yaml")
    parser.add_argument("--source-dir", default="results/lt_h64_n2_res/exp01")
    parser.add_argument("--output-dir", default="results/lora_transfer")
    parser.add_argument(
        "--model-type",
        choices=["liquid", "liquid_transformer"],
        default="liquid_transformer",
        help=(
            "'liquid' for LiquidYieldModel (original CfC), "
            "'liquid_transformer' for LiquidTransformerModel (PhenologyCfC + Transformer fusion)."
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  model_type: {args.model_type}", flush=True)

    cfg = load_config(args.config, overrides=["data.experiment=exp02"])
    if "liquid" not in cfg.model:
        cfg.model.liquid = OmegaConf.create({"hidden_dim": 32, "modality_dim": 32, "head_dim": 64})

    # Apply model-type-specific config overrides
    if args.model_type == "liquid_transformer":
        OmegaConf.update(cfg, "model.liquid.hidden_dim", 64)
        OmegaConf.update(cfg, "model.liquid.n_layers", 2)

    target_dataset = TimothyDroughtDataset(cfg)
    target_meta = target_dataset.plant_meta
    target_meta = target_meta[target_meta["dw_g"].notna()].reset_index(drop=True)
    dw_mean_t = float(target_meta["dw_g"].mean())
    dw_std_t = float(target_meta["dw_g"].std() + 1e-8)
    print(f"Target (exp02): {len(target_meta)} plants, DW mean={dw_mean_t:.2f}, std={dw_std_t:.2f}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    common = dict(source_dir=args.source_dir, cfg=cfg, target_dataset=target_dataset,
                  target_meta=target_meta, dw_mean_t=dw_mean_t, dw_std_t=dw_std_t,
                  device=device, output_dir=output_dir, model_type=args.model_type)

    # 1. Direct transfer (no adaptation) — lower bound
    all_results.append(run_lopo_experiment("Direct transfer", **common))

    # 2. Train from scratch on exp02 — upper bound
    all_results.append(run_lopo_experiment("From scratch", **common, train_from_scratch=True))

    # 3. Full fine-tune
    all_results.append(run_lopo_experiment("Full fine-tune", **common, full_finetune=True))

    # 4. LoRA rank sweep (all layers)
    for rank in [2, 4, 8, 16]:
        all_results.append(run_lopo_experiment(
            f"LoRA rank={rank}", **common, lora_rank=rank, lora_layers="all"))

    # 5. Layer-specific LoRA (at best rank from step 4)
    best_lora = min([r for r in all_results if "lora_rank" in r], key=lambda x: x["mae"])
    best_rank = best_lora.get("lora_rank", 4)
    print(f"\nBest LoRA rank: {best_rank}", flush=True)

    all_results.append(run_lopo_experiment(
        f"LoRA rank={best_rank} ff-only", **common, lora_rank=best_rank, lora_layers="ff"))
    all_results.append(run_lopo_experiment(
        f"LoRA rank={best_rank} time-only", **common, lora_rank=best_rank, lora_layers="time"))

    # 6. Sample efficiency (with best LoRA config)
    for n_samples in [5, 10, 20]:
        all_results.append(run_lopo_experiment(
            f"LoRA rank={best_rank} n={n_samples}", **common,
            lora_rank=best_rank, lora_layers="all", n_train_samples=n_samples))

    # Save summary
    with open(output_dir / "lora_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("LORA TRANSFER SUMMARY (exp01 → exp02)")
    print(f"{'='*60}")
    print(f"{'Method':<35} {'MAE(g)':<10} {'R²':<10}")
    print("-" * 55)
    for r in sorted(all_results, key=lambda x: x["mae"]):
        print(f"{r['name']:<35} {r['mae']:<10.3f} {r['r2']:<+10.3f}")


if __name__ == "__main__":
    main()
