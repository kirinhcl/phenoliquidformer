"""Training loop for Timothy drought model."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
from omegaconf import DictConfig
from scipy import stats
from torch.utils.data import DataLoader

from src.training.losses import MultiTaskLoss


class Trainer:
    """Training loop with early stopping, gradient clipping, and multi-task loss."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        cfg: DictConfig,
        fold_id: int,
        checkpoint_dir: Union[str, Path],
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.fold_id = fold_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        tcfg = cfg.training
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay
        )

        self.criterion = MultiTaskLoss(
            whc_weight=tcfg.loss.whc_weight,
            biomass_weight=tcfg.loss.biomass_weight,
        )

        self.max_epochs = tcfg.max_epochs
        self.patience = tcfg.patience
        self.gradient_clip = tcfg.gradient_clip

        self.best_val_mae = float("inf")
        self.epochs_without_improvement = 0
        self.best_epoch = 0

    def train(self) -> dict[str, Any]:
        """Run full training loop. Returns dict with best metrics and history."""
        history: list[dict[str, float]] = []

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch()
            val_metrics = self._validate()
            elapsed = time.time() - t0

            metrics = {
                "epoch": epoch,
                "train_loss": train_metrics["total_loss"],
                "train_whc_mae": train_metrics["whc_mae"],
                "val_loss": val_metrics["total_loss"],
                "val_whc_mae": val_metrics["whc_mae"],
                "val_r2": val_metrics.get("r2", 0.0),
                "val_spearman": val_metrics.get("spearman", 0.0),
                "elapsed": elapsed,
            }
            history.append(metrics)

            if val_metrics["whc_mae"] < self.best_val_mae:
                self.best_val_mae = val_metrics["whc_mae"]
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self._save_checkpoint("best_model_state.pt")
            else:
                self.epochs_without_improvement += 1

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"  Fold {self.fold_id} Epoch {epoch}: "
                    f"train_loss={train_metrics['total_loss']:.4f} "
                    f"val_MAE={val_metrics['whc_mae']:.4f} "
                    f"val_R2={val_metrics.get('r2', 0):.3f} "
                    f"({elapsed:.1f}s)"
                )

            if self.epochs_without_improvement >= self.patience:
                print(f"  Early stopping at epoch {epoch} (best={self.best_epoch})")
                break

        result = {
            "fold_id": self.fold_id,
            "best_epoch": self.best_epoch,
            "best_val_mae": self.best_val_mae,
            "history": history,
        }
        with open(self.checkpoint_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)

        return result

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        whc_errors = []

        for batch in self.train_loader:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            self.optimizer.zero_grad()
            outputs = self.model(batch)

            loss, loss_dict = self.criterion(
                outputs["whc_pred"],
                batch["whc_target"],
                outputs["biomass_pred"],
                batch["digital_biomass"],
                batch["biomass_mask"],
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip
            )
            self.optimizer.step()

            total_loss += loss_dict["total_loss"]
            mae = (outputs["whc_pred"] - batch["whc_target"]).abs().detach().cpu()
            whc_errors.extend(mae.tolist())

        n = len(self.train_loader)
        return {
            "total_loss": total_loss / max(n, 1),
            "whc_mae": float(np.mean(whc_errors)) if whc_errors else 0.0,
        }

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        self.model.train(False)
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in self.val_loader:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = self.model(batch)

            _, loss_dict = self.criterion(
                outputs["whc_pred"],
                batch["whc_target"],
                outputs["biomass_pred"],
                batch["digital_biomass"],
                batch["biomass_mask"],
            )
            total_loss += loss_dict["total_loss"]
            all_preds.extend(outputs["whc_pred"].cpu().tolist())
            all_targets.extend(batch["whc_target"].cpu().tolist())

        n = len(self.val_loader)
        preds = np.array(all_preds)
        targets = np.array(all_targets)

        metrics: dict[str, float] = {
            "total_loss": total_loss / max(n, 1),
            "whc_mae": float(np.mean(np.abs(preds - targets))) if len(preds) > 0 else 0.0,
        }

        if len(preds) > 1:
            ss_res = np.sum((targets - preds) ** 2)
            ss_tot = np.sum((targets - targets.mean()) ** 2)
            metrics["r2"] = float(1 - ss_res / max(ss_tot, 1e-8))
            rho, _ = stats.spearmanr(preds, targets)
            metrics["spearman"] = float(rho) if not np.isnan(rho) else 0.0

        return metrics

    def _save_checkpoint(self, filename: str) -> None:
        torch.save(self.model.state_dict(), self.checkpoint_dir / filename)
