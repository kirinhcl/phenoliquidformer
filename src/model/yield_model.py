"""Teacher-Student Neural ODE for Timothy yield prediction.

Teacher: full multimodal (image + fluor + VI) + context (WHC + genotype) -> LatentODE
Student: image-only -> LatentODE

Both use identical Latent ODE architecture; they differ only in input configuration.
The student is distilled from the teacher via output matching + latent trajectory
alignment.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from .encoder import ViewAggregation
from .ode_dynamics import LatentODE


class ModalityProjector(nn.Module):
    """Project each modality to a common dimension and concatenate.

    Unlike the gating approach, this is a simple concat fusion that avoids
    gate-collapse issues identified in the ablation study.
    """

    def __init__(
        self,
        image_dim: int = 768,
        fluor_dim: int = 98,
        vi_dim: int = 11,
        out_dim: int = 64,
        use_fluor: bool = True,
        use_vi: bool = True,
    ) -> None:
        super().__init__()
        self.use_fluor = use_fluor
        self.use_vi = use_vi

        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, out_dim),
            nn.Tanh(),
        )
        if use_fluor:
            self.fluor_proj = nn.Sequential(
                nn.Linear(fluor_dim, out_dim),
                nn.Tanh(),
            )
        if use_vi:
            self.vi_proj = nn.Sequential(
                nn.Linear(vi_dim, out_dim),
                nn.Tanh(),
            )

        self.out_dim_total = out_dim * (1 + int(use_fluor) + int(use_vi))

    def forward(
        self,
        image_emb: Tensor,  # (B, T, 768)
        fluorescence: Tensor | None,  # (B, T, 98) or None
        vi: Tensor | None,  # (B, T, 11) or None
    ) -> Tensor:
        parts = [self.image_proj(image_emb)]
        if self.use_fluor and fluorescence is not None:
            parts.append(self.fluor_proj(fluorescence))
        if self.use_vi and vi is not None:
            parts.append(self.vi_proj(vi))
        return torch.cat(parts, dim=-1)


class YieldHead(nn.Module):
    """Predict DW and flowering from final latent state."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
        )
        self.dw_out = nn.Linear(hidden_dim, 1)
        self.flower_out = nn.Linear(hidden_dim, 1)

    def forward(self, h_final: Tensor) -> dict[str, Tensor]:
        s = self.shared(h_final)
        return {
            "dw": self.dw_out(s).squeeze(-1),           # (B,)
            "flowering": self.flower_out(s).squeeze(-1), # (B,) — Poisson rate (log)
        }


class TimothyYieldModel(nn.Module):
    """Unified teacher / student model.

    Role determines input configuration:
        teacher: uses image + fluor + vi + context (WHC, genotype)
        student: uses image only, no context
    """

    def __init__(
        self,
        role: str,
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        assert role in ("teacher", "student")
        self.role = role
        mcfg = cfg.model if "model" in cfg else cfg

        # View aggregation for image features
        self.view_agg = ViewAggregation(mcfg.encoder_output_dim)

        # Modality projection — teacher uses all, student only image
        modality_out = 32  # per-modality output dim
        if role == "teacher":
            self.modality_proj = ModalityProjector(
                image_dim=mcfg.modality.image_dim,
                fluor_dim=mcfg.modality.fluor_dim,
                vi_dim=mcfg.modality.vi_dim,
                out_dim=modality_out,
                use_fluor=True,
                use_vi=True,
            )
            context_dim = 3  # WHC scalar (1) + genotype one-hot (2)
        else:  # student
            self.modality_proj = ModalityProjector(
                image_dim=mcfg.modality.image_dim,
                fluor_dim=mcfg.modality.fluor_dim,
                vi_dim=mcfg.modality.vi_dim,
                out_dim=modality_out,
                use_fluor=False,
                use_vi=False,
            )
            context_dim = 0  # completely blind

        obs_dim = self.modality_proj.out_dim_total

        # Latent ODE
        self.ode = LatentODE(
            obs_dim=obs_dim,
            latent_dim=OmegaConf.select(mcfg, "ode.latent_dim", default=64),
            context_dim=context_dim,
            vf_hidden=OmegaConf.select(mcfg, "ode.vf_hidden", default=64),
            decoder_hidden=OmegaConf.select(mcfg, "ode.decoder_hidden", default=64),
            ode_method=OmegaConf.select(mcfg, "ode.method", default="dopri5"),
            ode_rtol=OmegaConf.select(mcfg, "ode.rtol", default=1e-3),
            ode_atol=OmegaConf.select(mcfg, "ode.atol", default=1e-4),
        )

        # Yield head
        self.yield_head = YieldHead(
            latent_dim=OmegaConf.select(mcfg, "ode.latent_dim", default=64),
            hidden_dim=OmegaConf.select(mcfg, "heads.yield_hidden", default=64),
        )

        self.context_dim = context_dim

    def _build_observations(self, batch: dict) -> Tensor:
        """Aggregate multimodal inputs into a single observation tensor."""
        # View aggregation for images
        images = batch["images"]         # (B, T, V=4, D)
        image_mask = batch["image_mask"] # (B, T, V)
        image_emb = self.view_agg(images, image_mask)  # (B, T, D)

        fluor = batch["fluorescence"] if self.role == "teacher" else None
        vi = batch["vi"] if self.role == "teacher" else None

        obs = self.modality_proj(image_emb, fluor, vi)  # (B, T, obs_dim)
        return obs

    def _build_context(self, batch: dict) -> Tensor | None:
        """Build context vector for teacher; None for student."""
        if self.role != "teacher":
            return None

        B = batch["images"].shape[0]
        device = batch["images"].device

        whc = batch["whc_target"].view(B, 1).float()  # (B, 1)
        # Genotype one-hot: assume "Jauniai"=[1,0], "Noreng"=[0,1]
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
        """Run forward pass.

        Args:
            batch: dict with images, image_mask, fluorescence, fluor_mask, vi,
                   temporal_positions, active_mask, whc_target, genotype, ...
            t_cut: if given, truncate integration at this DAS

        Returns:
            dict with keys: h_traj, dw_pred, flowering_pred, time_used
        """
        # Build observations
        obs = self._build_observations(batch)  # (B, T, obs_dim)

        # Get measurement times (DAS). temporal_positions is (B, T), but they
        # should be identical across plants within an experiment — take the
        # first plant's times.
        times = batch["temporal_positions"][0]  # (T,)
        active_mask = batch["active_mask"]  # (B, T)

        # Build context
        context = self._build_context(batch)

        # Integrate
        h_traj, time_used = self.ode.integrate(
            observations=obs,
            times=times,
            active_mask=active_mask,
            context=context,
            t_cut=t_cut,
        )

        # Predict yield from final latent state
        h_final = h_traj[:, -1]  # (B, latent_dim)
        yield_out = self.yield_head(h_final)

        return {
            "h_traj": h_traj,           # (B, T_used, latent_dim)
            "time_used": time_used,     # (T_used,)
            "obs_used": obs[:, : len(time_used)],  # (B, T_used, obs_dim)
            "dw_pred": yield_out["dw"],            # (B,)
            "flowering_pred": yield_out["flowering"],  # (B,)
        }
