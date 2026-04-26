"""Distillation losses for teacher-student Neural ODE framework.

Components:
    L_hard  : student predictions vs ground truth labels
    L_soft  : student predictions vs teacher predictions (frozen)
    L_traj  : student latent trajectory vs teacher latent trajectory
    L_recon : student latent decoded back to observations (reconstruction)
    L_smooth: regularizer on dh/dt magnitude

Total: L = α · L_hard + β · L_soft + γ · L_traj + δ · L_recon + ε · L_smooth

With alpha annealing: α increases from 0.3 to 0.7, β decreases from 0.7 to 0.3
across training epochs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class YieldLoss(nn.Module):
    """Hard-target yield loss: Huber for DW, Poisson NLL for flowering."""

    def __init__(self, huber_delta: float = 1.0, flower_weight: float = 0.3) -> None:
        super().__init__()
        self.dw_loss = nn.SmoothL1Loss(beta=huber_delta)
        self.flower_weight = flower_weight

    def forward(
        self,
        dw_pred: Tensor,
        dw_target: Tensor,
        flower_pred: Tensor,
        flower_target: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        # DW: Huber on normalized targets (assume pre-normalized)
        valid_dw = ~torch.isnan(dw_target)
        if valid_dw.any():
            l_dw = self.dw_loss(dw_pred[valid_dw], dw_target[valid_dw])
        else:
            l_dw = torch.tensor(0.0, device=dw_pred.device)

        # Flowering: Poisson NLL (predict log rate)
        valid_fl = ~torch.isnan(flower_target)
        if valid_fl.any():
            rate = F.softplus(flower_pred[valid_fl]) + 1e-6  # ensure positive
            target_f = flower_target[valid_fl]
            l_flower = (rate - target_f * torch.log(rate)).mean()
        else:
            l_flower = torch.tensor(0.0, device=dw_pred.device)

        total = l_dw + self.flower_weight * l_flower
        return total, {
            "l_dw": float(l_dw.item()),
            "l_flower": float(l_flower.item()),
        }


class DistillationLoss(nn.Module):
    """Teacher-student distillation loss with latent trajectory alignment."""

    def __init__(
        self,
        alpha_hard: float = 0.5,
        beta_soft: float = 0.5,
        gamma_traj: float = 0.3,
        delta_recon: float = 0.1,
        epsilon_smooth: float = 0.01,
        huber_delta: float = 1.0,
        flower_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.alpha = alpha_hard
        self.beta = beta_soft
        self.gamma = gamma_traj
        self.delta = delta_recon
        self.epsilon = epsilon_smooth
        self.yield_loss = YieldLoss(huber_delta, flower_weight)

    def set_weights(
        self,
        alpha: float | None = None,
        beta: float | None = None,
    ) -> None:
        """Update hard/soft weights (for alpha annealing)."""
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta

    def forward(
        self,
        student_out: dict,
        teacher_out: dict | None,
        dw_target: Tensor,
        flower_target: Tensor,
        recon_loss_fn,
        smoothness_loss_fn,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute total loss.

        Args:
            student_out: dict with dw_pred, flowering_pred, h_traj, obs_used
            teacher_out: dict with same keys (detached) or None (pure training)
            dw_target, flower_target: ground-truth targets
            recon_loss_fn: callable(h_traj, obs, mask) -> reconstruction loss
            smoothness_loss_fn: callable(h_traj) -> smoothness loss
        """
        metrics: dict[str, float] = {}

        # Hard-target loss
        l_hard, hard_metrics = self.yield_loss(
            student_out["dw_pred"], dw_target,
            student_out["flowering_pred"], flower_target,
        )
        metrics.update(hard_metrics)
        metrics["l_hard"] = float(l_hard.item())

        total = self.alpha * l_hard

        # Soft-target loss (distillation)
        if teacher_out is not None:
            dw_soft = F.smooth_l1_loss(
                student_out["dw_pred"],
                teacher_out["dw_pred"].detach(),
            )
            fl_soft = F.smooth_l1_loss(
                student_out["flowering_pred"],
                teacher_out["flowering_pred"].detach(),
            )
            l_soft = dw_soft + 0.3 * fl_soft
            metrics["l_soft"] = float(l_soft.item())
            total = total + self.beta * l_soft

            # Latent trajectory alignment
            # Student and teacher have same latent_dim; align at each time step
            # Align along cosine similarity to be scale-invariant
            s_traj = student_out["h_traj"]  # (B, T, latent_dim)
            t_traj = teacher_out["h_traj"].detach()  # (B, T, latent_dim)

            # Handle possibly different T due to different t_cut; take min
            T_min = min(s_traj.shape[1], t_traj.shape[1])
            s_t = s_traj[:, :T_min]
            t_t = t_traj[:, :T_min]

            # Cosine distance (1 - cos_sim)
            s_n = F.normalize(s_t, dim=-1)
            t_n = F.normalize(t_t, dim=-1)
            cos_sim = (s_n * t_n).sum(dim=-1)  # (B, T_min)
            l_traj = (1.0 - cos_sim).mean()
            metrics["l_traj"] = float(l_traj.item())
            total = total + self.gamma * l_traj

        # Reconstruction loss (student only)
        # Need active_mask — computed in model
        # We'll accept a callable that knows how to do this
        if recon_loss_fn is not None:
            l_recon = recon_loss_fn()
            if l_recon is not None:
                metrics["l_recon"] = float(l_recon.item())
                total = total + self.delta * l_recon

        # Smoothness regularizer
        if smoothness_loss_fn is not None:
            l_smooth = smoothness_loss_fn()
            if l_smooth is not None:
                metrics["l_smooth"] = float(l_smooth.item())
                total = total + self.epsilon * l_smooth

        metrics["l_total"] = float(total.item())
        return total, metrics
