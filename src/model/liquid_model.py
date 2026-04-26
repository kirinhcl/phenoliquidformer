"""Liquid Neural Network (CfC) teacher-student model for Timothy yield prediction.

Replaces the Neural ODE core with Closed-form Continuous-time (CfC) liquid cells
from the ncps library. CfC has:
    - Small parameter count (good for small samples)
    - Stable liquid time constants (built-in regularization)
    - No ODE solver in loop (fast training)
    - Designed for irregular time series

We provide irregular timing information as an additional input feature
(dt since previous observation) to avoid an ncps timespans shape-handling issue.
"""

from __future__ import annotations

import torch
from ncps.torch import CfC
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from .encoder import ViewAggregation


class ModalityProjector(nn.Module):
    """Project each modality to common space and concatenate."""

    def __init__(
        self,
        image_dim: int = 768,
        fluor_dim: int = 98,
        vi_dim: int = 11,
        out_dim: int = 32,
        use_fluor: bool = True,
        use_vi: bool = True,
    ) -> None:
        super().__init__()
        self.use_fluor = use_fluor
        self.use_vi = use_vi

        self.image_proj = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, out_dim),
            nn.Tanh(),
        )
        if use_fluor:
            self.fluor_proj = nn.Sequential(
                nn.LayerNorm(fluor_dim),
                nn.Linear(fluor_dim, out_dim),
                nn.Tanh(),
            )
        if use_vi:
            self.vi_proj = nn.Sequential(
                nn.LayerNorm(vi_dim),
                nn.Linear(vi_dim, out_dim),
                nn.Tanh(),
            )

        self.out_dim_total = out_dim * (1 + int(use_fluor) + int(use_vi))

    def forward(
        self,
        image_emb: Tensor,
        fluorescence: Tensor | None,
        vi: Tensor | None,
    ) -> Tensor:
        parts = [self.image_proj(image_emb)]
        if self.use_fluor and fluorescence is not None:
            parts.append(self.fluor_proj(fluorescence))
        if self.use_vi and vi is not None:
            parts.append(self.vi_proj(vi))
        return torch.cat(parts, dim=-1)


class YieldHead(nn.Module):
    """Predict DW and flowering count from final hidden state."""

    def __init__(self, hidden_dim: int, mlp_dim: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mlp_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
        )
        self.dw_out = nn.Linear(mlp_dim, 1)
        self.flower_out = nn.Linear(mlp_dim, 1)

    def forward(self, h: Tensor) -> dict[str, Tensor]:
        s = self.shared(h)
        return {
            "dw": self.dw_out(s).squeeze(-1),
            "flowering": self.flower_out(s).squeeze(-1),
        }


class LiquidYieldModel(nn.Module):
    """Teacher-student Liquid (CfC) model for yield prediction.

    role='teacher': image + fluor + vi + context (WHC, genotype)
    role='student': image only
    """

    def __init__(self, role: str, cfg: DictConfig) -> None:
        super().__init__()
        assert role in ("teacher", "student")
        self.role = role
        mcfg = cfg.model if "model" in cfg else cfg

        hidden_dim = OmegaConf.select(mcfg, "liquid.hidden_dim", default=32)
        modality_out = OmegaConf.select(mcfg, "liquid.modality_dim", default=32)

        # View aggregation
        self.view_agg = ViewAggregation(mcfg.encoder_output_dim)

        # Modality projection
        if role == "teacher":
            use_vi = OmegaConf.select(mcfg, "use_vi", default=True)
            self.modality_proj = ModalityProjector(
                image_dim=mcfg.modality.image_dim,
                fluor_dim=mcfg.modality.fluor_dim,
                vi_dim=mcfg.modality.vi_dim,
                out_dim=modality_out,
                use_fluor=True, use_vi=use_vi,
            )
            context_dim = 3  # WHC + genotype (2-dim one-hot)
        else:
            self.modality_proj = ModalityProjector(
                image_dim=mcfg.modality.image_dim,
                fluor_dim=mcfg.modality.fluor_dim,
                vi_dim=mcfg.modality.vi_dim,
                out_dim=modality_out,
                use_fluor=False, use_vi=False,
            )
            context_dim = 0

        # Total per-step feature dim: modalities + dt + context (replicated per step)
        per_step_dim = self.modality_proj.out_dim_total + 1 + context_dim  # +1 for dt

        # Stacked CfC layers
        n_layers = OmegaConf.select(mcfg, "liquid.n_layers", default=1)
        cfc_layers = []
        for i in range(n_layers):
            in_dim = per_step_dim if i == 0 else hidden_dim
            cfc_layers.append(CfC(
                input_size=in_dim,
                units=hidden_dim,
                batch_first=True,
                return_sequences=True,
            ))
        self.cfc_layers = nn.ModuleList(cfc_layers)
        self.n_layers = n_layers

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        # Yield head
        self.yield_head = YieldHead(
            hidden_dim=hidden_dim,
            mlp_dim=OmegaConf.select(mcfg, "liquid.head_dim", default=64),
        )

    def _build_context(self, batch: dict) -> Tensor | None:
        if self.role != "teacher":
            return None
        B = batch["images"].shape[0]
        device = batch["images"].device
        whc = batch["whc_target"].view(B, 1).float()
        genotype_list = batch["genotype"]
        geno_oh = torch.zeros(B, 2, device=device, dtype=torch.float32)
        for i, g in enumerate(genotype_list):
            if "Jauniai" in g:
                geno_oh[i, 0] = 1.0
            elif "Noreng" in g:
                geno_oh[i, 1] = 1.0
        return torch.cat([whc, geno_oh], dim=-1)  # (B, 3)

    def forward(
        self,
        batch: dict,
        t_cut: float | None = None,
    ) -> dict[str, Tensor]:
        # Aggregate views
        images = batch["images"]
        image_mask = batch["image_mask"]
        image_emb = self.view_agg(images, image_mask)  # (B, T, D)

        fluor = batch["fluorescence"] if self.role == "teacher" else None
        vi = batch["vi"] if self.role == "teacher" else None

        obs = self.modality_proj(image_emb, fluor, vi)  # (B, T, obs_dim)

        # Time deltas
        times = batch["temporal_positions"][0]  # (T,)
        dt = torch.zeros_like(times)
        dt[1:] = times[1:] - times[:-1]
        # Normalize dt to [0, 1] by max
        max_dt = max(float(dt.max().item()), 1.0)
        dt_norm = dt / max_dt
        dt_feat = dt_norm.unsqueeze(0).unsqueeze(-1).expand(obs.shape[0], -1, 1)  # (B, T, 1)

        # Apply truncation
        if t_cut is not None:
            mask = times <= t_cut
            T_used = int(mask.sum().item())
            obs = obs[:, :T_used]
            dt_feat = dt_feat[:, :T_used]
        else:
            T_used = obs.shape[1]

        # Build per-step input: [obs, dt] + replicated context
        step_input = torch.cat([obs, dt_feat], dim=-1)  # (B, T_used, obs_dim+1)
        if self.context_dim > 0:
            ctx = self._build_context(batch)
            ctx_rep = ctx.unsqueeze(1).expand(-1, T_used, -1)  # (B, T_used, ctx_dim)
            step_input = torch.cat([step_input, ctx_rep], dim=-1)

        # Apply active mask: zero out invalid time steps
        active = batch["active_mask"][:, :T_used].unsqueeze(-1).float()
        step_input = step_input * active

        # Stacked CfC forward
        x = step_input
        for layer in self.cfc_layers:
            h_seq, h_final = layer(x)
            x = h_seq  # output of layer i is input of layer i+1

        # Use final hidden state for prediction
        yield_out = self.yield_head(h_final)

        return {
            "h_seq": h_seq,
            "h_final": h_final,
            "dw_pred": yield_out["dw"],
            "flowering_pred": yield_out["flowering"],
        }
