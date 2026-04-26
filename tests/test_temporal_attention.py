import torch
import pytest
from src.model.temporal_attention import TemporalAttentionPooling


def test_output_shape():
    pool = TemporalAttentionPooling(hidden_dim=32, num_heads=4)
    h_seq = torch.randn(4, 18, 32)
    mask = torch.ones(4, 18, dtype=torch.bool)
    pooled, attn = pool(h_seq, mask)
    assert pooled.shape == (4, 32)
    assert attn.shape == (4, 4, 18)


def test_attention_weights_sum_to_one():
    pool = TemporalAttentionPooling(hidden_dim=32, num_heads=4)
    h_seq = torch.randn(2, 10, 32)
    mask = torch.ones(2, 10, dtype=torch.bool)
    _, attn = pool(h_seq, mask)
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_masking_zeroes_invalid_steps():
    pool = TemporalAttentionPooling(hidden_dim=32, num_heads=4)
    h_seq = torch.randn(2, 10, 32)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[:, 7:] = False
    _, attn = pool(h_seq, mask)
    assert (attn[:, :, 7:] == 0).all()
    sums = attn[:, :, :7].sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_no_nan_with_all_masked():
    pool = TemporalAttentionPooling(hidden_dim=32, num_heads=4)
    h_seq = torch.randn(1, 5, 32)
    mask = torch.zeros(1, 5, dtype=torch.bool)
    pooled, attn = pool(h_seq, mask)
    assert not torch.isnan(pooled).any()
