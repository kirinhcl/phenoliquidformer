"""Modules for aggregating multi-view image embeddings."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ViewAggregation(nn.Module):
    """Attention-based pooling over multiple camera views per timestep."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim: int = dim
        self.query: nn.Parameter = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: Tensor, view_mask: Tensor) -> Tensor:
        """Aggregate view embeddings with attention.

        Args:
            x: View embeddings of shape (B, T, V, D).
            view_mask: Boolean mask of shape (B, T, V) where True indicates a valid view.

        Returns:
            Aggregated embeddings of shape (B, T, D).
        """
        view_mask = view_mask.bool()
        # x: (B, T, V, D)
        scores = (x * self.query).sum(dim=-1) / math.sqrt(self.dim)  # (B, T, V)
        scores = scores.masked_fill(~view_mask, float("-inf"))
        all_missing = ~view_mask.any(dim=-1)  # (B, T)
        if all_missing.any():
            scores = scores.masked_fill(all_missing.unsqueeze(-1), 0.0)

        attn = torch.softmax(scores, dim=-1)  # (B, T, V)
        if all_missing.any():
            attn = attn.masked_fill(all_missing.unsqueeze(-1), 0.0)

        aggregated = torch.sum(attn.unsqueeze(-1) * x, dim=-2)  # (B, T, D)
        if all_missing.any():
            aggregated = aggregated.masked_fill(all_missing.unsqueeze(-1), 0.0)
        return aggregated
