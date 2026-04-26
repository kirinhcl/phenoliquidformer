"""PhenologyLiquidModel: combines PhenologyCfC + TemporalAttentionPooling.

Structurally similar to LiquidYieldModel but with three differences:
  1. VI modality is dropped (use_vi=False always).
  2. PhenologyCfC replaces ncps CfC — receives phi_seq (normalized DAS).
  3. TemporalAttentionPooling replaces the final hidden state h_final.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from .encoder import ViewAggregation
from .liquid_model import ModalityProjector, YieldHead
from .phenology_cfc import PhenologyCfC
from .temporal_attention import TemporalAttentionPooling


class PhenologyLiquidModel(nn.Module):
    """Teacher-student Liquid (PhenologyCfC) model for yield prediction.

    role='teacher': image + fluor + context (WHC, genotype). VI always dropped.
    role='student': image only.
    """

    def __init__(self, role: str, cfg: DictConfig) -> None:
        super().__init__()
        assert role in ("teacher", "student")
        self.role = role
        mcfg = cfg.model if "model" in cfg else cfg

        hidden_dim = OmegaConf.select(mcfg, "liquid.hidden_dim", default=32)
        modality_out = OmegaConf.select(mcfg, "liquid.modality_dim", default=32)
        n_layers = OmegaConf.select(mcfg, "liquid.n_layers", default=1)
        head_dim = OmegaConf.select(mcfg, "liquid.head_dim", default=64)

        backbone_units = OmegaConf.select(mcfg, "phenology.backbone_units", default=128)
        num_attention_heads = OmegaConf.select(mcfg, "phenology.num_attention_heads", default=4)
        self.max_das = OmegaConf.select(mcfg, "phenology.max_das", default=51)

        # View aggregation
        self.view_agg = ViewAggregation(mcfg.encoder_output_dim)

        # Modality projection — VI always disabled
        if role == "teacher":
            self.modality_proj = ModalityProjector(
                image_dim=mcfg.modality.image_dim,
                fluor_dim=mcfg.modality.fluor_dim,
                vi_dim=mcfg.modality.vi_dim,
                out_dim=modality_out,
                use_fluor=True,
                use_vi=False,
            )
            context_dim = 3  # WHC + genotype one-hot (2-dim)
        else:
            self.modality_proj = ModalityProjector(
                image_dim=mcfg.modality.image_dim,
                fluor_dim=mcfg.modality.fluor_dim,
                vi_dim=mcfg.modality.vi_dim,
                out_dim=modality_out,
                use_fluor=False,
                use_vi=False,
            )
            context_dim = 0

        # Per-step feature dim: modalities + dt + context
        per_step_dim = self.modality_proj.out_dim_total + 1 + context_dim  # +1 for dt

        # Stacked PhenologyCfC layers
        cfc_layers = []
        for i in range(n_layers):
            in_dim = per_step_dim if i == 0 else hidden_dim
            cfc_layers.append(PhenologyCfC(
                input_size=in_dim,
                hidden_size=hidden_dim,
                backbone_units=backbone_units,
                batch_first=True,
            ))
        self.cfc_layers = nn.ModuleList(cfc_layers)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        # Temporal attention pooling
        self.attn_pool = TemporalAttentionPooling(
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
        )

        # Yield head
        self.yield_head = YieldHead(hidden_dim=hidden_dim, mlp_dim=head_dim)

    def _build_context(self, batch: dict) -> Tensor | None:
        if self.role != "teacher":
            return None
        B = batch["images"].shape[0]
        device = batch["images"].device
        whc = batch["whc_target"].view(B, 1).float().to(device)
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
        # VI always excluded
        obs = self.modality_proj(image_emb, fluor, vi=None)  # (B, T, obs_dim)

        # Time deltas and phi (normalized DAS)
        times = batch["temporal_positions"][0]  # (T,)
        dt = torch.zeros_like(times)
        dt[1:] = times[1:] - times[:-1]
        max_dt = max(float(dt.max().item()), 1.0)
        dt_norm = dt / max_dt

        B = obs.shape[0]
        phi_seq = (times / self.max_das).clamp(0.0, 1.0)   # (T,)
        phi_seq = phi_seq.unsqueeze(0).expand(B, -1)         # (B, T)
        dt_feat = dt_norm.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)  # (B, T, 1)
        ts_seq = dt_norm.unsqueeze(0).expand(B, -1)          # (B, T)

        # Apply truncation
        if t_cut is not None:
            mask = times <= t_cut
            T_used = int(mask.sum().item())
            obs = obs[:, :T_used]
            dt_feat = dt_feat[:, :T_used]
            phi_seq = phi_seq[:, :T_used]
            ts_seq = ts_seq[:, :T_used]
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
        active_mask = batch["active_mask"][:, :T_used]  # (B, T_used)

        # Stacked PhenologyCfC forward
        x = step_input
        for layer in self.cfc_layers:
            h_seq, _h_last = layer(x, phi_seq, ts_seq=ts_seq)
            x = h_seq

        # Temporal attention pooling instead of final hidden state
        pooled, attn_weights = self.attn_pool(h_seq, active_mask)  # pooled: (B, D)

        yield_out = self.yield_head(pooled)

        return {
            "h_seq": h_seq,
            "h_final": pooled,
            "dw_pred": yield_out["dw"],
            "flowering_pred": yield_out["flowering"],
            "attn_weights": attn_weights,
        }
