"""Prediction heads for Timothy drought model."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class WHCRegressionHead(nn.Module):
    """Predict continuous WHC level from pooled temporal representation.

    Pools over time dimension using attention, then maps to scalar WHC prediction.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        self.pool_query = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.input_dim = input_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor, active_mask: Tensor) -> Tensor:
        """Predict WHC level.

        Args:
            x: Temporal tokens (B, T, input_dim).
            active_mask: (B, T) bool, True = valid timestep.

        Returns:
            WHC predictions (B,) in [0, 1] range.
        """
        # Attention pooling over time
        scores = (x * self.pool_query).sum(dim=-1) / (self.input_dim ** 0.5)  # (B, T)
        scores = scores.masked_fill(~active_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)  # (B, T)
        attn = attn.masked_fill(~active_mask, 0.0)
        pooled = (attn.unsqueeze(-1) * x).sum(dim=1)  # (B, input_dim)

        pooled = self.norm(pooled)
        return self.mlp(pooled).squeeze(-1)  # (B,)


class BiomassTrajectoryHead(nn.Module):
    """Predict digital biomass trajectory (per-timestep regression)."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Predict biomass per timestep.

        Args:
            x: Temporal tokens (B, T, input_dim).

        Returns:
            Biomass predictions (B, T).
        """
        x = self.norm(x)
        return self.mlp(x).squeeze(-1)  # (B, T)
