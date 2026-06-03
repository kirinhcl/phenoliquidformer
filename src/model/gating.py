"""Modality gating module for adaptive feature fusion."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ModalityProjection(nn.Module):
    """Projects modalities to common 128-dim space with mask tokens for missing data."""

    def __init__(
        self,
        image_dim: int = 768,
        fluor_dim: int = 92,
        env_dim: int = 5,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim: int = hidden_dim

        # Projection MLPs for each modality
        self.image_proj: nn.Sequential = self._make_proj(image_dim, hidden_dim)
        self.fluor_proj: nn.Sequential = self._make_proj(fluor_dim, hidden_dim)
        self.env_proj: nn.Sequential = self._make_proj(env_dim, hidden_dim)

        # LayerNorm per modality to prevent gate collapse from scale mismatch
        self.image_norm: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.fluor_norm: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.env_norm: nn.LayerNorm = nn.LayerNorm(hidden_dim)

        # Mask tokens for image and fluorescence (only modalities with masks)
        self.image_mask_token: nn.Parameter = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.fluor_mask_token: nn.Parameter = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def _make_proj(self, input_dim: int, hidden_dim: int) -> nn.Sequential:
        """Create projection MLP: Linear → ReLU → Dropout → Linear."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        image_emb: Tensor,
        fluorescence: Tensor,
        environment: Tensor,
        image_mask: Tensor,
        fluor_mask: Tensor,
    ) -> list[Tensor]:
        """Project modalities to common space and apply mask tokens.

        Args:
            image_emb: Image embeddings of shape (B, T, 768).
            fluorescence: Fluorescence data of shape (B, T, 92).
            environment: Environment data of shape (B, T, 5).
            image_mask: Boolean mask for image of shape (B, T), True = valid.
            fluor_mask: Boolean mask for fluorescence of shape (B, T), True = valid.

        Returns:
            List of 3 projected modalities, each of shape (B, T, 128).
        """
        batch_size, time_steps, _ = image_emb.shape

        # Project all modalities and normalize to common scale
        image_proj = self.image_norm(self.image_proj(image_emb))
        fluor_proj = self.fluor_norm(self.fluor_proj(fluorescence))
        env_proj = self.env_norm(self.env_proj(environment))

        # Apply mask tokens where data is invalid
        image_token = self.image_mask_token.expand(batch_size, time_steps, -1)
        image_proj = torch.where(image_mask.unsqueeze(-1), image_proj, image_token)

        fluor_token = self.fluor_mask_token.expand(batch_size, time_steps, -1)
        fluor_proj = torch.where(fluor_mask.unsqueeze(-1), fluor_proj, fluor_token)

        return [image_proj, fluor_proj, env_proj]


class ModalityGating(nn.Module):
    """Learns per-timestep importance weights for adaptive modality fusion."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_modalities: int = 4,
        gate_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_dim: int = hidden_dim
        self.num_modalities: int = num_modalities

        # Gate network: concat → Linear → ReLU → Dropout → Linear → Softmax
        self.gate_network: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, gate_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gate_hidden, num_modalities),
        )

    def forward(self, modality_features: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Compute gated fusion of modality features.

        Args:
            modality_features: List of 4 tensors, each of shape (B, T, 128).

        Returns:
            fused: Weighted sum of modalities, shape (B, T, 128).
            gates: Softmax weights, shape (B, T, 4), sum to 1.0 along last dim.
        """
        # Concatenate all modalities
        concat = torch.cat(modality_features, dim=-1)  # (B, T, 512)

        # Compute gates via network and apply softmax
        gate_logits = self.gate_network(concat)  # (B, T, 4)
        gates = torch.softmax(gate_logits, dim=-1)  # (B, T, 4)

        # Stack modalities and apply gates
        stacked = torch.stack(modality_features, dim=-1)  # (B, T, 128, 4)
        fused = (stacked * gates.unsqueeze(-2)).sum(dim=-1)  # (B, T, 128)

        return fused, gates
