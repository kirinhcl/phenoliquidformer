"""Tests for LiquidTransformerModel hybrid architecture."""

import torch
import pytest
from omegaconf import OmegaConf
from src.model.liquid_transformer import LiquidTransformerModel


def _make_cfg():
    return OmegaConf.create({
        "model": {
            "encoder_output_dim": 768,
            "modality": {
                "image_dim": 768,
                "fluor_dim": 98,
                "env_dim": 5,
                "vi_dim": 11,
                "hidden_dim": 128,
                "gate_hidden": 64,
            },
            "liquid": {
                "hidden_dim": 32,
                "modality_dim": 32,
                "n_layers": 1,
                "head_dim": 64,
            },
            "phenology": {
                "backbone_units": 128,
                "backbone_layers": 1,
                "num_attention_heads": 4,
                "max_das": 51,
            },
        },
    })


def _make_batch(B: int = 2, T: int = 18, V: int = 4) -> dict:
    return {
        "images": torch.randn(B, T, V, 768),
        "image_mask": torch.ones(B, T, V, dtype=torch.bool),
        "fluorescence": torch.randn(B, T, 98),
        "fluor_mask": torch.ones(B, T, dtype=torch.bool),
        "environment": torch.randn(B, T, 5),
        "vi": torch.randn(B, T, 11),
        "temporal_positions": torch.linspace(1, 51, T).unsqueeze(0).expand(B, -1),
        "active_mask": torch.ones(B, T, dtype=torch.bool),
        "whc_target": torch.tensor([0.9, 0.25] * B)[:B],
        "genotype": (["Jauniai", "Noreng"] * B)[:B],
    }


def test_forward_teacher():
    cfg = _make_cfg()
    model = LiquidTransformerModel(role="teacher", cfg=cfg)
    batch = _make_batch()
    out = model(batch)

    assert out["dw_pred"].shape == (2,)
    assert out["attn_weights"].shape == (2, 4, 18)
    assert out["modality_gates"] is not None
    assert out["modality_gates"].shape == (2, 18, 3), (
        f"Expected gates shape (2, 18, 3), got {out['modality_gates'].shape}"
    )
    assert not torch.isnan(out["dw_pred"]).any(), "NaN in dw_pred"


def test_forward_student():
    cfg = _make_cfg()
    model = LiquidTransformerModel(role="student", cfg=cfg)
    batch = _make_batch()
    out = model(batch)

    assert out["dw_pred"].shape == (2,)
    assert out["modality_gates"] is None


def test_t_cut_truncation():
    cfg = _make_cfg()
    model = LiquidTransformerModel(role="teacher", cfg=cfg)
    batch = _make_batch(T=18)
    # Cut at DAS 25 — roughly the first 9 timesteps of linspace(1, 51, 18)
    out = model(batch, t_cut=25.0)
    assert out["dw_pred"].shape == (2,)
    assert not torch.isnan(out["dw_pred"]).any()


def test_param_count():
    cfg = _make_cfg()
    model = LiquidTransformerModel(role="teacher", cfg=cfg)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nLiquidTransformer params: {total:,}")
    assert total < 250_000, f"Model has {total} params, expected < 250K"
    assert total > 100_000, f"Model has {total} params, expected > 100K"


def test_backward_pass():
    cfg = _make_cfg()
    model = LiquidTransformerModel(role="teacher", cfg=cfg)
    batch = _make_batch()
    out = model(batch)
    loss = out["dw_pred"].sum()
    loss.backward()

    # Gradients flow to gating
    assert model.modality_gating.gate_network[0].weight.grad is not None, (
        "No gradient in modality_gating.gate_network[0].weight"
    )
    # Gradients flow to PhenologyCfC cell's phi_proj
    for i, layer in enumerate(model.cfc_layers):
        assert layer.cell.phi_proj.weight.grad is not None, (
            f"No gradient in cfc_layers[{i}].cell.phi_proj.weight"
        )


def test_h_seq_shape():
    cfg = _make_cfg()
    model = LiquidTransformerModel(role="teacher", cfg=cfg)
    batch = _make_batch(B=3, T=18)
    out = model(batch)
    hidden_dim = 32
    assert out["h_seq"].shape == (3, 18, hidden_dim)
    assert out["h_final"].shape == (3, hidden_dim)
