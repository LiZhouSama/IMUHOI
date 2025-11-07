"""
Extended loss definitions for GPNet training with object tracking.
Based on gp_training/losses.py with additional object velocity loss.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import articulate as art


class GPNetWithObjectLoss(nn.Module):
    """
    Multi-task loss for GPNet pipeline extended with object tracking.
    Maintains GP's original loss structure and adds object velocity loss.
    """
    
    def __init__(self, weights: Dict[str, float] | None = None):
        super().__init__()
        # Default weights matching GP
        self.weights = {
            'pl_position': 1.0,
            'pl_orientation': 0.5,
            'ik_position': 1.0,
            'ik_orientation': 0.5,
            'rr_rotation': 1.0,
            'vr_velocity': 0.5,
            'vr_stationary': 1.0,
            'obj_velocity': 1.0,  # New: object velocity loss weight
        }
        if weights:
            self.weights.update(weights)
        
        self.smooth_l1 = nn.SmoothL1Loss()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        obj_pred: Optional[List[torch.Tensor]] = None,
        obj_target: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: List of predicted tensors [T, 189] per sequence
            targets: List of target tensors [T, 189] per sequence
            obj_pred: Optional list of object velocity predictions [T, 3]
            obj_target: Optional list of object velocity targets [T, 3]
            
        Returns:
            (total_loss, loss_dict)
        """
        # Concatenate sequences
        pred = torch.cat(predictions, dim=0)  # [N, 189]
        target = torch.cat(targets, dim=0)  # [N, 189]
        
        # Split predictions by stage (matching GP format)
        # PL: [0:18] = pRB (15) + gR (3)
        # IK: [18:90] = pRJ (69) + gR (3)
        # RR: [90:180] = rrj (90)
        # VR: [180:189] = vRR (4) + stationary (5)
        
        pl_pred = pred[:, :18]
        ik_pred = pred[:, 18:90]
        rr_pred = pred[:, 90:180]
        vr_pred = pred[:, 180:]
        
        pl_tgt = target[:, :18]
        ik_tgt = target[:, 18:90]
        rr_tgt = target[:, 90:180]
        vr_tgt = target[:, 180:]
        
        # PL losses (Pose & Localization)
        loss_pl_position = self.smooth_l1(pl_pred[:, :15], pl_tgt[:, :15])
        gR_pred = art.math.normalize_tensor(pl_pred[:, 15:])
        gR_tgt = art.math.normalize_tensor(pl_tgt[:, 15:])
        loss_pl_orientation = (1.0 - (gR_pred * gR_tgt).sum(dim=-1)).mean()
        
        # IK losses (Inverse Kinematics)
        loss_ik_position = self.smooth_l1(ik_pred[:, :69], ik_tgt[:, :69])
        gR2_pred = art.math.normalize_tensor(ik_pred[:, 69:])
        gR2_tgt = art.math.normalize_tensor(ik_tgt[:, 69:])
        loss_ik_orientation = (1.0 - (gR2_pred * gR2_tgt).sum(dim=-1)).mean()
        
        # RR losses (Root Rotation)
        loss_rr_rotation = self.mse(rr_pred, rr_tgt)
        
        # VR losses (Velocity & Root)
        loss_vr_velocity = self.smooth_l1(vr_pred[:, :4], vr_tgt[:, :4])
        loss_vr_stationary = self.bce(vr_pred[:, 4:], vr_tgt[:, 4:])
        
        # Object losses (new)
        loss_obj_velocity = torch.tensor(0.0, device=pred.device)
        if obj_pred is not None and obj_target is not None:
            obj_p = torch.cat(obj_pred, dim=0)  # [N, 3]
            obj_t = torch.cat(obj_target, dim=0)  # [N, 3]
            loss_obj_velocity = self.smooth_l1(obj_p, obj_t)
        
        # Total weighted loss
        total = (
            self.weights['pl_position'] * loss_pl_position +
            self.weights['pl_orientation'] * loss_pl_orientation +
            self.weights['ik_position'] * loss_ik_position +
            self.weights['ik_orientation'] * loss_ik_orientation +
            self.weights['rr_rotation'] * loss_rr_rotation +
            self.weights['vr_velocity'] * loss_vr_velocity +
            self.weights['vr_stationary'] * loss_vr_stationary +
            self.weights['obj_velocity'] * loss_obj_velocity
        )
        
        loss_dict = {
            'pl_position': loss_pl_position.detach(),
            'pl_orientation': loss_pl_orientation.detach(),
            'ik_position': loss_ik_position.detach(),
            'ik_orientation': loss_ik_orientation.detach(),
            'rr_rotation': loss_rr_rotation.detach(),
            'vr_velocity': loss_vr_velocity.detach(),
            'vr_stationary': loss_vr_stationary.detach(),
            'obj_velocity': loss_obj_velocity.detach(),
        }
        
        return total, loss_dict


class SimplifiedVRWithObjectLoss(nn.Module):
    """
    Simplified loss focusing only on VR and object (for fine-tuning).
    """
    
    def __init__(
        self,
        lambda_vr: float = 1.0,
        lambda_static: float = 0.5,
        lambda_obj: float = 1.0
    ):
        super().__init__()
        self.lambda_vr = lambda_vr
        self.lambda_static = lambda_static
        self.lambda_obj = lambda_obj
        
        self.smooth_l1 = nn.SmoothL1Loss()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        vr_pred: torch.Tensor,
        vr_target: torch.Tensor,
        obj_pred: Optional[torch.Tensor] = None,
        obj_target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            vr_pred: [B, T, 9] VR predictions (4 velocity + 5 stationary)
            vr_target: [B, T, 9] VR targets
            obj_pred: [B, T, 3] object velocity predictions
            obj_target: [B, T, 3] object velocity targets
        """
        # VR velocity loss
        loss_vr = self.smooth_l1(vr_pred[..., :4], vr_target[..., :4])
        
        # VR stationary loss
        loss_static = self.bce(vr_pred[..., 4:], vr_target[..., 4:])
        
        # Object velocity loss
        loss_obj = torch.tensor(0.0, device=vr_pred.device)
        if obj_pred is not None and obj_target is not None:
            loss_obj = self.smooth_l1(obj_pred, obj_target)
        
        total = (
            self.lambda_vr * loss_vr +
            self.lambda_static * loss_static +
            self.lambda_obj * loss_obj
        )
        
        loss_dict = {
            'vr': loss_vr.detach(),
            'static': loss_static.detach(),
            'obj': loss_obj.detach()
        }
        
        return total, loss_dict

