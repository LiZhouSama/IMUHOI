"""Loss definitions for GPNet training."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import articulate as art


class GPNetLoss(nn.Module):
    """Multi-task loss for the three-stage GPNet pipeline."""

    def __init__(self, weights: Dict[str, float] | None = None):
        super().__init__()
        self.weights = {
            'pl_position': 1.0,
            'pl_orientation': 0.5,
            'ik_position': 1.0,
            'ik_orientation': 0.5,
            'rr_rotation': 1.0,
            'vr_velocity': 0.5,
            'vr_stationary': 1.0,
        }
        if weights:
            self.weights.update(weights)

        self.smooth_l1 = nn.SmoothL1Loss()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred = torch.cat(predictions, dim=0)
        target = torch.cat(targets, dim=0)

        pl_pred = pred[:, :18]
        ik_pred = pred[:, 18:90]
        rr_pred = pred[:, 90:180]
        vr_pred = pred[:, 180:]

        pl_tgt = target[:, :18]
        ik_tgt = target[:, 18:90]
        rr_tgt = target[:, 90:180]
        vr_tgt = target[:, 180:]

        loss_pl_position = self.smooth_l1(pl_pred[:, :15], pl_tgt[:, :15])
        gR_pred = art.math.normalize_tensor(pl_pred[:, 15:])
        gR_tgt = art.math.normalize_tensor(pl_tgt[:, 15:])
        loss_pl_orientation = (1.0 - (gR_pred * gR_tgt).sum(dim=-1)).mean()

        loss_ik_position = self.smooth_l1(ik_pred[:, :69], ik_tgt[:, :69])
        gR2_pred = art.math.normalize_tensor(ik_pred[:, 69:])
        gR2_tgt = art.math.normalize_tensor(ik_tgt[:, 69:])
        loss_ik_orientation = (1.0 - (gR2_pred * gR2_tgt).sum(dim=-1)).mean()

        loss_rr_rotation = self.mse(rr_pred, rr_tgt)

        loss_vr_velocity = self.smooth_l1(vr_pred[:, :4], vr_tgt[:, :4])
        loss_vr_stationary = self.bce(vr_pred[:, 4:], vr_tgt[:, 4:])

        total = (
            self.weights['pl_position'] * loss_pl_position +
            self.weights['pl_orientation'] * loss_pl_orientation +
            self.weights['ik_position'] * loss_ik_position +
            self.weights['ik_orientation'] * loss_ik_orientation +
            self.weights['rr_rotation'] * loss_rr_rotation +
            self.weights['vr_velocity'] * loss_vr_velocity +
            self.weights['vr_stationary'] * loss_vr_stationary
        )

        return total, {
            'pl_position': loss_pl_position.detach(),
            'pl_orientation': loss_pl_orientation.detach(),
            'ik_position': loss_ik_position.detach(),
            'ik_orientation': loss_ik_orientation.detach(),
            'rr_rotation': loss_rr_rotation.detach(),
            'vr_velocity': loss_vr_velocity.detach(),
            'vr_stationary': loss_vr_stationary.detach(),
        }
