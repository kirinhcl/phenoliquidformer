"""Loss functions for Timothy drought model."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class WHCRegressionLoss(nn.Module):
    """Huber (smooth L1) loss for WHC level prediction."""

    def __init__(self, delta: float = 0.1) -> None:
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=delta)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.loss_fn(pred, target)


class BiomassTrajectoryLoss(nn.Module):
    """MSE loss for digital biomass trajectory prediction."""

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Compute masked MSE loss.

        Args:
            pred: (B, T) predicted biomass.
            target: (B, T) target biomass.
            mask: (B, T) bool, True = valid timestep.
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        diff = (pred - target) ** 2
        return (diff * mask.float()).sum() / mask.sum()


class MultiTaskLoss(nn.Module):
    """Combined WHC regression + biomass trajectory loss."""

    def __init__(
        self,
        whc_weight: float = 1.0,
        biomass_weight: float = 0.5,
        huber_delta: float = 0.1,
    ) -> None:
        super().__init__()
        self.whc_weight = whc_weight
        self.biomass_weight = biomass_weight
        self.whc_loss = WHCRegressionLoss(delta=huber_delta)
        self.biomass_loss = BiomassTrajectoryLoss()

    def forward(
        self,
        whc_pred: Tensor,
        whc_target: Tensor,
        biomass_pred: Tensor,
        biomass_target: Tensor,
        biomass_mask: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute multi-task loss.

        Returns:
            total_loss: Weighted sum of losses.
            loss_dict: Individual loss values for logging.
        """
        l_whc = self.whc_loss(whc_pred, whc_target)
        l_bio = self.biomass_loss(biomass_pred, biomass_target, biomass_mask)
        total = self.whc_weight * l_whc + self.biomass_weight * l_bio

        return total, {
            "whc_loss": l_whc.item(),
            "biomass_loss": l_bio.item(),
            "total_loss": total.item(),
        }
