"""Liquid Transformer: Transformer multimodal fusion + CfC temporal dynamics.

Combines:
- ModalityProjection + ModalityGating from Transformer (cross-modal fusion)
- PhenologyCfC (continuous-time dynamics with phenology-aware tau)
- TemporalAttentionPooling (interpretable temporal attention)
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from .encoder import ViewAggregation
from .gating import ModalityGating, ModalityProjection
from .liquid_model import YieldHead
from .phenology_cfc import PhenologyCfC
from .temporal_attention import TemporalAttentionPooling


class LiquidTransformerModel(nn.Module):
    """Hybrid model: Transformer multimodal fusion + Liquid temporal dynamics."""

    def __init__(self, role: str, cfg: DictConfig) -> None:
        super().__init__()
        assert role in ("teacher", "student")
        self.role = role
        mcfg = cfg.model if "model" in cfg else cfg

        hidden_dim = OmegaConf.select(mcfg, "liquid.hidden_dim", default=32)
        fusion_dim = OmegaConf.select(mcfg, "modality.hidden_dim", default=128)
        gate_hidden = OmegaConf.select(mcfg, "modality.gate_hidden", default=64)

        # Phenology config
        pcfg_backbone_units = OmegaConf.select(mcfg, "phenology.backbone_units", default=128)
        pcfg_num_heads = OmegaConf.select(mcfg, "phenology.num_attention_heads", default=4)
        self.max_das = OmegaConf.select(mcfg, "phenology.max_das", default=51)
        # Ablation flag: when True, force phi_t = 0 at every step so the
        # PhenologyCfC cell becomes equivalent to a standard CfC (the
        # phi_proj weights still exist but receive a zero input and
        # therefore contribute nothing to time-constant pre-activations).
        self.disable_phenology = OmegaConf.select(mcfg, "phenology.disable", default=False)
        # Temporal backbone selector. "cfc" (default) uses the
        # PhenologyCfC continuous-time cell; "gru" swaps in a vanilla
        # nn.GRU at the same hidden_dim, holding the rest of the
        # LiquidFormer architecture (Transformer fusion, residual,
        # attention pooling, yield head) fixed. This isolates the
        # continuous-time vs discrete-time RNN axis for the benchmark.
        self.temporal_backbone = OmegaConf.select(mcfg, "temporal.backbone", default="cfc")

        # 1. View aggregation
        self.view_agg = ViewAggregation(mcfg.encoder_output_dim)

        # 2. Transformer-style multimodal fusion
        if role == "teacher":
            # Teacher: image + fluor + env (3 modalities)
            self.modality_proj = ModalityProjection(
                image_dim=mcfg.modality.image_dim,
                fluor_dim=mcfg.modality.fluor_dim,
                env_dim=mcfg.modality.env_dim,
                hidden_dim=fusion_dim,
            )
            num_mods = 3  # image, fluor, env
            self.context_dim = 3  # WHC + genotype (2-dim one-hot)

            self.modality_gating = ModalityGating(
                hidden_dim=fusion_dim,
                num_modalities=num_mods,
                gate_hidden=gate_hidden,
            )
        else:
            # Student: image only — simple projector, no gating
            self.modality_proj = None
            self.modality_gating = None
            self.image_proj = nn.Sequential(
                nn.Linear(mcfg.modality.image_dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
            )
            self.context_dim = 0

        # 3. PhenologyCfC temporal core
        # CfC input: fused features (fusion_dim) + context (context_dim).
        # dt is passed as ts_seq to PhenologyCfC, not concatenated.
        cfc_input_dim = fusion_dim + self.context_dim
        n_layers = OmegaConf.select(mcfg, "liquid.n_layers", default=1)
        cfc_layers = []
        for i in range(n_layers):
            in_dim = cfc_input_dim if i == 0 else hidden_dim
            if self.temporal_backbone == "gru":
                cfc_layers.append(nn.GRU(
                    input_size=in_dim,
                    hidden_size=hidden_dim,
                    batch_first=True,
                ))
            else:
                cfc_layers.append(PhenologyCfC(
                    input_size=in_dim,
                    hidden_size=hidden_dim,
                    backbone_units=pcfg_backbone_units,
                    batch_first=True,
                ))
        self.cfc_layers = nn.ModuleList(cfc_layers)
        self.hidden_dim = hidden_dim

        # Residual projection: fused (fusion_dim) → hidden_dim for skip connection
        self.residual_proj = nn.Linear(fusion_dim, hidden_dim)

        # 4. Temporal attention pooling
        self.temporal_pool = TemporalAttentionPooling(
            hidden_dim=hidden_dim,
            num_heads=pcfg_num_heads,
        )

        # 5. Yield head
        self.yield_head = YieldHead(
            hidden_dim=hidden_dim,
            mlp_dim=OmegaConf.select(mcfg, "liquid.head_dim", default=64),
            dropout=OmegaConf.select(mcfg, "liquid.dropout", default=0.1),
        )

    def _build_context(self, batch: dict) -> Tensor:
        """Build teacher context vector: WHC + genotype one-hot. Returns (B, 3)."""
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
        images = batch["images"]
        image_mask = batch["image_mask"]
        image_emb = self.view_agg(images, image_mask)  # (B, T, 768)

        # Multimodal fusion (teacher) or image-only projection (student)
        if self.role == "teacher":
            fluorescence = batch["fluorescence"]
            fluor_mask = batch.get(
                "fluor_mask",
                torch.ones(fluorescence.shape[:2], dtype=torch.bool, device=fluorescence.device),
            )
            environment = batch.get(
                "environment",
                torch.zeros(
                    fluorescence.shape[0], fluorescence.shape[1], 5,
                    device=fluorescence.device,
                ),
            )
            image_active = image_mask.any(dim=-1)  # (B, T)

            modality_features = self.modality_proj(
                image_emb, fluorescence, environment,
                image_active, fluor_mask,
            )
            fused, gates = self.modality_gating(modality_features)  # (B, T, 128)
        else:
            fused = self.image_proj(image_emb)  # (B, T, 128)
            gates = None

        # Time deltas and phi
        times = batch["temporal_positions"][0]  # (T,)
        dt = torch.zeros_like(times)
        dt[1:] = times[1:] - times[:-1]
        max_dt = max(float(dt.max().item()), 1.0)
        dt_norm = dt / max_dt  # normalized delta-t for ts_seq

        phi_seq = (times / self.max_das).clamp(0, 1)  # (T,)
        if self.disable_phenology:
            phi_seq = torch.zeros_like(phi_seq)
        phi_seq_b = phi_seq.unsqueeze(0).expand(fused.shape[0], -1)  # (B, T)
        dt_seq_b = dt_norm.unsqueeze(0).expand(fused.shape[0], -1)   # (B, T)

        # Truncation
        if t_cut is not None:
            mask = times <= t_cut
            T_used = int(mask.sum().item())
            fused = fused[:, :T_used]
            phi_seq_b = phi_seq_b[:, :T_used]
            dt_seq_b = dt_seq_b[:, :T_used]
        else:
            T_used = fused.shape[1]

        # Build CfC input: fused + context (teacher) or fused only (student)
        step_input = fused  # (B, T_used, fusion_dim)
        if self.context_dim > 0:
            ctx = self._build_context(batch)          # (B, 3)
            ctx_rep = ctx.unsqueeze(1).expand(-1, T_used, -1)  # (B, T_used, 3)
            step_input = torch.cat([step_input, ctx_rep], dim=-1)

        # Zero out invalid time steps
        active = batch["active_mask"][:, :T_used]  # (B, T_used)
        step_input = step_input * active.unsqueeze(-1).float()

        # Residual: project fused features to hidden_dim for skip connection
        residual = self.residual_proj(fused[:, :T_used])  # (B, T_used, hidden_dim)
        residual = residual * active.unsqueeze(-1).float()

        # Recurrent temporal processing. CfC backbone uses phi/dt; GRU
        # backbone just consumes the step_input sequence.
        x = step_input
        for layer in self.cfc_layers:
            if self.temporal_backbone == "gru":
                h_seq, _ = layer(x)  # nn.GRU returns (output, h_n)
            else:
                h_seq, _ = layer(x, phi_seq=phi_seq_b, ts_seq=dt_seq_b)
            x = h_seq

        # Add residual connection: CfC output + projected fused features
        h_seq = h_seq + residual

        # Temporal attention pooling
        pooled, attn_weights = self.temporal_pool(h_seq, active)

        yield_out = self.yield_head(pooled)

        return {
            "h_seq": h_seq,
            "h_final": pooled,
            "dw_pred": yield_out["dw"],
            "flowering_pred": yield_out["flowering"],
            "attn_weights": attn_weights,
            "modality_gates": gates,
        }
