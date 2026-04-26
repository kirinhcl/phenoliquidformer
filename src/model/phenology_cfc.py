"""Phenology-aware Closed-form Continuous-time (CfC) cell.

Extends the standard CfC formulation with a phenological phase signal phi
(normalized DAS position in [0, 1]) that modulates the time constants
(time_a, time_b) without affecting the state transition pathway (ff1, ff2).
"""

import torch
import torch.nn as nn
from typing import Tuple


def _lecun_tanh(x: torch.Tensor) -> torch.Tensor:
    """LeCun activation: 1.7159 * tanh(0.666 * x)."""
    return 1.7159 * torch.tanh(0.666 * x)


class PhenologyCfCCell(nn.Module):
    """CfC cell with phenology-gated time constants.

    Args:
        input_size: Dimensionality of the input vector.
        hidden_size: Dimensionality of the hidden state.
        backbone_units: Width of the backbone projection layer.
            Defaults to 4 * hidden_size.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        backbone_units: int | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if backbone_units is None:
            backbone_units = 4 * hidden_size
        self.backbone_units = backbone_units

        # Backbone: concatenate input + hidden state
        self.backbone = nn.Linear(input_size + hidden_size, backbone_units)

        # State transition layers (not modulated by phi)
        self.ff1 = nn.Linear(backbone_units, hidden_size)
        self.ff2 = nn.Linear(backbone_units, hidden_size)

        # Time constant layers (receive phi-modulated backbone output)
        self.time_a = nn.Linear(backbone_units, hidden_size)
        self.time_b = nn.Linear(backbone_units, hidden_size)

        # Phenology projection: scalar phi -> backbone_units
        self.phi_proj = nn.Linear(1, backbone_units, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                nn.init.xavier_uniform_(w)

    def forward(
        self,
        x: torch.Tensor,
        hx: torch.Tensor,
        ts: torch.Tensor,
        phi: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single time-step forward pass.

        Args:
            x:   (B, input_size)  — input features.
            hx:  (B, hidden_size) — previous hidden state.
            ts:  (B,)             — elapsed time / delta-t.
            phi: (B,)             — normalized DAS position in [0, 1].

        Returns:
            (h_out, h_new): both (B, hidden_size). h_out == h_new for CfC.
        """
        # Backbone (unmodulated)
        x_cat = torch.cat([x, hx], dim=-1)                # (B, input+hidden)
        backbone_out = _lecun_tanh(self.backbone(x_cat))   # (B, backbone_units)

        # State transition: uses unmodulated backbone output
        f1 = torch.tanh(self.ff1(backbone_out))            # (B, hidden_size)
        f2 = torch.tanh(self.ff2(backbone_out))            # (B, hidden_size)

        # Phenology modulation for time constants
        phi_emb = self.phi_proj(phi.unsqueeze(-1))         # (B, backbone_units)
        x_phi = backbone_out + phi_emb                     # (B, backbone_units)

        # Time constants (phi-modulated)
        ta = self.time_a(x_phi)                            # (B, hidden_size)
        tb = self.time_b(x_phi)                            # (B, hidden_size)

        # Interpolation factor: sigmoid(time_a * ts + time_b)
        ts_exp = ts.unsqueeze(-1)                          # (B, 1) broadcasts over hidden
        t_interp = torch.sigmoid(ta * ts_exp + tb)         # (B, hidden_size)

        # CfC state update
        h_new = f1 * (1.0 - t_interp) + t_interp * f2    # (B, hidden_size)

        return h_new, h_new


class PhenologyCfC(nn.Module):
    """RNN wrapper around PhenologyCfCCell.

    Loops over the time dimension calling PhenologyCfCCell at each step.
    Drop-in replacement for ncps CfC when phi_seq is available.

    Args:
        input_size:     Input feature dimensionality.
        hidden_size:    Hidden state dimensionality.
        backbone_units: Backbone projection width (default: 4 * hidden_size).
        batch_first:    If True, input/output tensors are (B, T, F).
                        If False, they are (T, B, F). Default: True.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        backbone_units: int | None = None,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.cell = PhenologyCfCCell(input_size, hidden_size, backbone_units)

    def forward(
        self,
        x_seq: torch.Tensor,
        phi_seq: torch.Tensor,
        ts_seq: torch.Tensor | None = None,
        hx: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over a full sequence.

        Args:
            x_seq:   (B, T, F) or (T, B, F) — input sequence.
            phi_seq: (B, T) or (T, B)        — normalized DAS per step.
            ts_seq:  (B, T) or (T, B)        — delta-t per step.
                     If None, defaults to ones.
            hx:      (B, hidden_size)         — initial hidden state.
                     If None, defaults to zeros.

        Returns:
            outputs: (B, T, hidden_size) or (T, B, hidden_size)
            h_last:  (B, hidden_size)
        """
        if self.batch_first:
            B, T, _ = x_seq.shape
        else:
            T, B, _ = x_seq.shape
            x_seq = x_seq.transpose(0, 1)      # -> (B, T, F)
            phi_seq = phi_seq.transpose(0, 1)   # -> (B, T)
            if ts_seq is not None:
                ts_seq = ts_seq.transpose(0, 1)

        if hx is None:
            hx = torch.zeros(B, self.hidden_size, device=x_seq.device, dtype=x_seq.dtype)

        if ts_seq is None:
            ts_seq = torch.ones(B, T, device=x_seq.device, dtype=x_seq.dtype)

        outputs = []
        h = hx
        for t in range(T):
            h_out, h = self.cell(
                x_seq[:, t, :],
                h,
                ts_seq[:, t],
                phi_seq[:, t],
            )
            outputs.append(h_out.unsqueeze(1))  # (B, 1, hidden_size)

        out_seq = torch.cat(outputs, dim=1)     # (B, T, hidden_size)

        if not self.batch_first:
            out_seq = out_seq.transpose(0, 1)   # -> (T, B, hidden_size)

        return out_seq, h
