"""Timothy drought model — multimodal fusion for WHC regression."""

from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from .encoder import ViewAggregation
from .gating import ModalityGating, ModalityProjection
from .heads import BiomassTrajectoryHead, WHCRegressionHead
from .temporal import TemporalTransformer


class TimothyDroughtModel(nn.Module):
    """Multimodal drought severity estimation model.

    Architecture (adapted from LUPIN/StressDetectionModel):
        1. ViewAggregation: (B, T, V=4, 768) → (B, T, 768)
        2. ModalityProjection: 4 modalities → 128-dim common space
        3. ModalityGating: adaptive fusion with learned weights
        4. TemporalTransformer: (B, T, 128) → temporal reasoning
        5. WHCRegressionHead: predict continuous WHC level
        6. BiomassTrajectoryHead (optional): predict biomass trajectory
    """

    def __init__(self, model_config: DictConfig) -> None:
        super().__init__()
        cfg = model_config.model if "model" in model_config else model_config

        self.use_vi: bool = OmegaConf.select(cfg, "use_vi", default=True)
        self.enabled_modalities: list[str] = OmegaConf.select(
            cfg, "ablation.enabled_modalities", default=["image", "fluor", "env", "vi"]
        )
        self.fusion_mode: str = OmegaConf.select(
            cfg, "ablation.fusion_mode", default="gating"
        )
        self.temporal_mode: str = OmegaConf.select(
            cfg, "ablation.temporal_mode", default="transformer"
        )
        self.causal_mask: bool = OmegaConf.select(
            cfg, "ablation.causal_mask", default=False
        )

        # View aggregation
        self.view_agg = ViewAggregation(cfg.encoder_output_dim)

        num_mods: int = 4 if self.use_vi else 3

        # Modality projection
        self.modality_proj = ModalityProjection(
            image_dim=cfg.modality.image_dim,
            fluor_dim=cfg.modality.fluor_dim,
            env_dim=cfg.modality.env_dim,
            vi_dim=cfg.modality.vi_dim,
            hidden_dim=cfg.modality.hidden_dim,
            use_vi=self.use_vi,
        )

        # Modality gating
        self.modality_gating = ModalityGating(
            hidden_dim=cfg.modality.hidden_dim,
            num_modalities=num_mods,
            gate_hidden=cfg.modality.gate_hidden,
        )

        # Temporal transformer
        self.temporal = TemporalTransformer(
            dim=cfg.temporal.dim,
            num_layers=cfg.temporal.num_layers,
            num_heads=cfg.temporal.num_heads,
            ff_dim=cfg.temporal.ff_dim,
            dropout=cfg.temporal.dropout,
            causal=self.causal_mask,
        )

        # Concat fusion alternative
        self.concat_fusion: nn.Module | None = None
        if self.fusion_mode == "concat":
            self.concat_fusion = nn.Sequential(
                nn.Linear(cfg.modality.hidden_dim * num_mods, cfg.modality.hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(cfg.modality.hidden_dim * 2, cfg.modality.hidden_dim),
            )

        # MLP temporal alternative
        self.temporal_mlp: nn.Module | None = None
        if self.temporal_mode == "mlp":
            self.temporal_mlp = nn.Sequential(
                nn.Linear(cfg.temporal.dim, cfg.temporal.dim * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(cfg.temporal.dim * 4, cfg.temporal.dim),
            )

        # Prediction heads
        self.whc_head = WHCRegressionHead(
            input_dim=cfg.temporal.dim,
            hidden_dim=cfg.heads.whc_hidden_dim,
        )
        self.biomass_head = BiomassTrajectoryHead(
            input_dim=cfg.temporal.dim,
            hidden_dim=cfg.heads.biomass_hidden_dim,
        )

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run forward pass.

        Args:
            batch: dict with keys: images, image_mask, fluorescence, fluor_mask,
                   environment, vi, temporal_positions, active_mask

        Returns:
            dict with: whc_pred, biomass_pred, modality_gates, cls_embedding
        """
        # 1. View aggregation
        images = batch["images"]  # (B, T, V=4, 768)
        image_mask = batch["image_mask"]  # (B, T, V)
        image_emb = self.view_agg(images, image_mask)  # (B, T, 768)
        image_active = image_mask.any(dim=-1)  # (B, T)

        # 2. Modality projection
        fluorescence = batch["fluorescence"]  # (B, T, fluor_dim)
        fluor_mask = batch["fluor_mask"]  # (B, T)
        environment = batch["environment"]  # (B, T, 5)
        vi = batch["vi"] if self.use_vi else None  # (B, T, vi_dim) or None

        modality_features = self.modality_proj(
            image_emb, fluorescence, environment, vi, image_active, fluor_mask
        )

        # Zero out disabled modalities (ablation; vi is already excluded when use_vi=False)
        modality_names = ["image", "fluor", "env"] + (["vi"] if self.use_vi else [])
        for i, name in enumerate(modality_names):
            if name not in self.enabled_modalities:
                modality_features[i] = torch.zeros_like(modality_features[i])

        # 3. Modality fusion
        num_mods = len(modality_features)
        if self.fusion_mode == "concat":
            concat = torch.cat(modality_features, dim=-1)
            fused = self.concat_fusion(concat)
            B, T = fused.shape[0], fused.shape[1]
            gates = torch.ones(B, T, num_mods, device=fused.device) / num_mods
        else:
            fused, gates = self.modality_gating(modality_features)

        # 4. Temporal encoding
        temporal_positions = batch["temporal_positions"]  # (B, T)
        active_mask = batch["active_mask"]  # (B, T)

        if self.temporal_mode == "transformer":
            cls_emb, temporal_tokens, _ = self.temporal(
                fused, temporal_positions, active_mask
            )
        elif self.temporal_mode == "mlp":
            temporal_tokens = self.temporal_mlp(fused)
            cls_emb = temporal_tokens.mean(dim=1)
        elif self.temporal_mode == "none":
            temporal_tokens = fused
            cls_emb = fused.mean(dim=1)
        else:
            raise ValueError(f"Unknown temporal_mode: {self.temporal_mode}")

        # 5. Predictions
        whc_pred = self.whc_head(temporal_tokens, active_mask)  # (B,)
        biomass_pred = self.biomass_head(temporal_tokens)  # (B, T)

        return {
            "whc_pred": whc_pred,
            "biomass_pred": biomass_pred,
            "modality_gates": gates,
            "cls_embedding": cls_emb,
        }
