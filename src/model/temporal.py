"""Temporal transformer for multimodal sequences."""

from __future__ import annotations

import math
from typing import Optional, cast

import torch
from torch import Tensor, nn


def sinusoidal_pe(dag_values: Tensor, dim: int) -> Tensor:
    """Continuous sinusoidal positional encoding from DAG values.

    Args:
        dag_values: (B, T) float tensor of DAG values.
        dim: Embedding dimension.

    Returns:
        Positional encoding of shape (B, T, dim).
    """
    pe = torch.zeros(*dag_values.shape, dim, device=dag_values.device, dtype=dag_values.dtype)
    position = dag_values.unsqueeze(-1)  # (B, T, 1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=dag_values.device, dtype=dag_values.dtype)
        * -(math.log(10000.0) / dim)
    )
    pe[..., 0::2] = torch.sin(position * div_term)
    pe[..., 1::2] = torch.cos(position * div_term)
    return pe


class TransformerEncoderLayerWithWeights(nn.TransformerEncoderLayer):
    """Transformer encoder layer that stores attention weights."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.attention_weights: Optional[Tensor] = None

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        attn_output, attn_weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        self.attention_weights = attn_weights
        return self.dropout1(attn_output)


class TemporalTransformer(nn.Module):
    """Transformer encoder with continuous sinusoidal positional encoding."""

    def __init__(
        self,
        dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.dim: int = dim
        self.cls_token: nn.Parameter = nn.Parameter(torch.zeros(1, 1, dim))
        self.causal: bool = causal

        encoder_layer = TransformerEncoderLayerWithWeights(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
        )
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            cast(nn.TransformerEncoderLayer, encoder_layer),
            num_layers=num_layers,
        )
        self.attention_weights: list[Tensor] = []

    def forward(
        self,
        x: Tensor,
        temporal_positions: Tensor,
        active_mask: Tensor,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Encode temporal sequence with CLS token.

        Args:
            x: (B, T, dim)
            temporal_positions: (B, T) float DAG values
            active_mask: (B, T) bool mask where True indicates active positions

        Returns:
            cls_embedding: (B, dim)
            temporal_tokens: (B, T, dim)
            attention_weights: list of (B, H, T+1, T+1) per layer
        """
        # x: (B, T, dim)
        pe = sinusoidal_pe(temporal_positions, self.dim)  # (B, T, dim)
        x = x + pe  # (B, T, dim)

        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, dim)
        cls_positions = torch.zeros(
            batch_size,
            1,
            device=x.device,
            dtype=temporal_positions.dtype,
        )
        cls_pe = sinusoidal_pe(cls_positions, self.dim)  # (B, 1, dim)
        cls_token = cls_token + cls_pe

        x = torch.cat([cls_token, x], dim=1)  # (B, T+1, dim)

        active_mask = active_mask.bool()
        cls_active = torch.ones(batch_size, 1, device=active_mask.device, dtype=torch.bool)
        active_mask_with_cls = torch.cat([cls_active, active_mask], dim=1)  # (B, T+1)
        padding_mask = ~active_mask_with_cls  # True indicates padding

        attn_mask = None
        if self.causal:
            seq_len = x.shape[1]
            causal_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
            causal_mask[0, :] = False
            for i in range(1, seq_len):
                causal_mask[i, 0] = False
                causal_mask[i, 1 : i + 1] = False
            attn_mask = torch.zeros(seq_len, seq_len, device=x.device)
            attn_mask.masked_fill_(causal_mask, float("-inf"))

        encoded = self.encoder(
            x,
            mask=attn_mask,
            src_key_padding_mask=padding_mask,
        )  # (B, T+1, dim)

        cls_embedding = encoded[:, 0, :]  # (B, dim)
        temporal_tokens = encoded[:, 1:, :]  # (B, T, dim)
        self.attention_weights = []
        for layer in self.encoder.layers:
            typed_layer = cast(TransformerEncoderLayerWithWeights, layer)
            if typed_layer.attention_weights is None:
                continue
            self.attention_weights.append(cast(Tensor, typed_layer.attention_weights))
        return cls_embedding, temporal_tokens, self.attention_weights
