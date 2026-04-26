import torch
import pytest
from omegaconf import OmegaConf
from src.model.phenology_liquid_model import PhenologyLiquidModel


def _make_cfg():
    return OmegaConf.create({
        "model": {
            "encoder_output_dim": 768,
            "modality": {
                "image_dim": 768,
                "fluor_dim": 98,
                "vi_dim": 11,
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


def _make_batch(B=2, T=18, V=4):
    return {
        "images": torch.randn(B, T, V, 768),
        "image_mask": torch.ones(B, T, V, dtype=torch.bool),
        "fluorescence": torch.randn(B, T, 98),
        "vi": torch.randn(B, T, 11),
        "temporal_positions": torch.linspace(1, 51, T).unsqueeze(0).expand(B, -1),
        "active_mask": torch.ones(B, T, dtype=torch.bool),
        "whc_target": torch.tensor([0.9, 0.25]),
        "genotype": ["Jauniai", "Noreng"],
    }


def test_forward_teacher():
    cfg = _make_cfg()
    model = PhenologyLiquidModel(role="teacher", cfg=cfg)
    batch = _make_batch()
    out = model(batch)
    assert out["dw_pred"].shape == (2,)
    assert out["attn_weights"].shape == (2, 4, 18)
    assert not torch.isnan(out["dw_pred"]).any()


def test_forward_student():
    cfg = _make_cfg()
    model = PhenologyLiquidModel(role="student", cfg=cfg)
    batch = _make_batch()
    out = model(batch)
    assert out["dw_pred"].shape == (2,)
    assert out["attn_weights"].shape == (2, 4, 18)


def test_no_vi_in_projector():
    cfg = _make_cfg()
    model = PhenologyLiquidModel(role="teacher", cfg=cfg)
    assert not model.modality_proj.use_vi
    assert model.modality_proj.out_dim_total == 64


def test_param_count_under_75k():
    cfg = _make_cfg()
    model = PhenologyLiquidModel(role="teacher", cfg=cfg)
    total = sum(p.numel() for p in model.parameters())
    assert total < 75_000, f"Model has {total} params, expected < 75K"
