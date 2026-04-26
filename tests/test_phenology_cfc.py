import torch
import pytest
from src.model.phenology_cfc import PhenologyCfCCell


def test_phenology_cfc_cell_output_shape():
    cell = PhenologyCfCCell(input_size=68, hidden_size=32)
    x = torch.randn(4, 68)
    hx = torch.zeros(4, 32)
    ts = torch.ones(4)
    phi = torch.tensor([0.0, 0.3, 0.7, 1.0])
    h_out, h_new = cell(x, hx, ts, phi)
    assert h_out.shape == (4, 32)
    assert h_new.shape == (4, 32)


def test_phenology_cfc_cell_phi_zero_matches_baseline():
    cell = PhenologyCfCCell(input_size=68, hidden_size=32)
    x = torch.randn(4, 68)
    hx = torch.zeros(4, 32)
    ts = torch.ones(4)
    phi = torch.zeros(4)
    h_out, _ = cell(x, hx, ts, phi)
    assert h_out.shape == (4, 32)
    assert not torch.isnan(h_out).any()


def test_phenology_cfc_cell_different_phi_different_output():
    torch.manual_seed(42)
    cell = PhenologyCfCCell(input_size=68, hidden_size=32)
    torch.nn.init.normal_(cell.phi_proj.weight, std=0.5)
    x = torch.randn(4, 68)
    hx = torch.zeros(4, 32)
    ts = torch.ones(4)
    h_early, _ = cell(x, hx, ts, torch.zeros(4))
    h_late, _ = cell(x, hx, ts, torch.ones(4))
    assert not torch.allclose(h_early, h_late, atol=1e-5)
