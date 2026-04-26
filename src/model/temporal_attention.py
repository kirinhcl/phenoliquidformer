"""Multi-head temporal attention pooling over CfC hidden sequences."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TemporalAttentionPooling(nn.Module):
    """Multi-head attention pooling over a temporal hidden sequence.

    Uses learnable query vectors (one per head) to attend over hidden states.
    Returns pooled representation and per-head attention weights for visualization.
    """

    def __init__(self, hidden_dim: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim

        self.queries = nn.Parameter(torch.randn(1, num_heads, self.head_dim) * 0.02)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h_seq: Tensor, active_mask: Tensor) -> tuple[Tensor, Tensor]:
        B, T, D = h_seq.shape
        H, d = self.num_heads, self.head_dim

        K = self.k_proj(h_seq).view(B, T, H, d).transpose(1, 2)  # (B, H, T, d)
        V = self.v_proj(h_seq).view(B, T, H, d).transpose(1, 2)  # (B, H, T, d)
        Q = self.queries.expand(B, -1, -1)  # (B, H, d)

        scores = torch.einsum("bhd,bhtd->bht", Q, K) / (d ** 0.5)

        inv_mask = ~active_mask.unsqueeze(1).expand(-1, H, -1)
        scores = scores.masked_fill(inv_mask, float("-inf"))

        all_masked = ~active_mask.any(dim=1)
        if all_masked.any():
            scores[all_masked] = 0.0

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = attn_weights.masked_fill(inv_mask, 0.0)

        context = torch.einsum("bht,bhtd->bhd", attn_weights, V)
        context = context.reshape(B, D)

        pooled = self.out_proj(context)
        return pooled, attn_weights
