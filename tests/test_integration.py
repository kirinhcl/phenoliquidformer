# tests/test_integration.py
import torch
import pytest
from omegaconf import OmegaConf
from src.model.phenology_liquid_model import PhenologyLiquidModel
from src.training.distillation_loss import YieldLoss


def _make_cfg():
    return OmegaConf.create({
        "model": {
            "encoder_output_dim": 768,
            "modality": {"image_dim": 768, "fluor_dim": 98, "vi_dim": 11},
            "liquid": {
                "hidden_dim": 32, "modality_dim": 32,
                "n_layers": 1, "head_dim": 64,
            },
            "phenology": {
                "backbone_units": 128, "backbone_layers": 1,
                "num_attention_heads": 4, "max_das": 51,
            },
        },
    })


def _make_batch(B=2, T=18, V=4, device="cpu"):
    return {
        "images": torch.randn(B, T, V, 768, device=device),
        "image_mask": torch.ones(B, T, V, dtype=torch.bool, device=device),
        "fluorescence": torch.randn(B, T, 98, device=device),
        "vi": torch.randn(B, T, 11, device=device),
        "temporal_positions": torch.linspace(
            1, 51, T, device=device,
        ).unsqueeze(0).expand(B, -1),
        "active_mask": torch.ones(B, T, dtype=torch.bool, device=device),
        "whc_target": torch.tensor([0.9, 0.25], device=device),
        "genotype": ["Jauniai", "Noreng"],
        "dw_target": torch.tensor([25.0, 18.0], device=device),
    }


def test_forward_backward_pass():
    """Full forward + backward pass without errors."""
    cfg = _make_cfg()
    model = PhenologyLiquidModel(role="teacher", cfg=cfg)
    criterion = YieldLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch = _make_batch()
    out = model(batch)

    flower_target = torch.full((2,), float("nan"))
    loss, metrics = criterion(
        out["dw_pred"], batch["dw_target"],
        out["flowering_pred"], flower_target,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Verify gradients flowed to phi_proj
    for layer in model.cfc_layers:
        phi_grad = layer.cell.phi_proj.weight.grad
        assert phi_grad is not None, "phi_proj should receive gradients"
        assert phi_grad.abs().sum() > 0, "phi_proj gradients should be nonzero"


def test_t_cut_truncation():
    """Model works with t_cut (early prediction)."""
    cfg = _make_cfg()
    model = PhenologyLiquidModel(role="teacher", cfg=cfg)
    batch = _make_batch()
    out = model(batch, t_cut=30.0)
    assert out["dw_pred"].shape == (2,)
    # attn_weights T dimension should be truncated
    T_trunc = out["attn_weights"].shape[2]
    assert T_trunc < 18
