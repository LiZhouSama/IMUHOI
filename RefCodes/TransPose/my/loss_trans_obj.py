# -*- coding: utf-8 -*-
"""
Loss definitions for TransPose with object estimation.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

_DEFAULT_FPS = 60.0


def _multi_scale_velocity_loss(pred: torch.Tensor,
                               target: torch.Tensor,
                               scales: Sequence[int]) -> torch.Tensor:
    """
    Implements the multi-scale velocity loss described in the TransPose paper.
    """
    losses = []
    B, T, C = pred.shape
    for n in scales:
        if n <= 0 or T < n:
            continue
        blocks = T // n
        if blocks == 0:
            continue
        pred_block = pred[:, :blocks * n].reshape(B, blocks, n, C).mean(dim=2)
        target_block = target[:, :blocks * n].reshape(B, blocks, n, C).mean(dim=2)
        diff = pred_block - target_block
        losses.append((diff.pow(2).sum(dim=-1) * n).mean())
    if not losses:
        return pred.new_tensor(0.0)
    return sum(losses) / len(losses)


def transpose_with_object_loss(
    pred: Dict[str, torch.Tensor],
    gt: Dict[str, torch.Tensor],
    weights: Optional[Dict[str, float]] = None,
    velocity_scales: Iterable[int] = (1, 3, 9, 27),
    return_tensors: bool = True,
) -> Tuple[torch.Tensor, Dict]:
    """
    Computes the combined loss for human pose, translation and object motion.
    
    Args:
        pred: 预测字典
        gt: ground truth字典
        weights: 损失权重
        velocity_scales: 多尺度velocity loss的尺度
        return_tensors: 如果为True，返回tensor形式的metrics（避免GPU同步）；
                       如果为False，返回float形式（需要GPU同步）
    
    Returns:
        total_loss: 总损失
        metrics: 如果return_tensors=True，返回detached tensor字典；否则返回float字典
    """
    weight_cfg = {
        'leaf': 1.0,
        'full': 1.0,
        'pose': 5.0,
        'contact': 1.0,
        'velocity_local': 0.05,
        'root_velocity': 1.0,
        'root_position': 1.0,
        'obj_velocity': 0.1,
        'obj_position': 0.5,
    }
    if weights is not None:
        weight_cfg.update(weights)

    loss_leaf = F.mse_loss(pred['leaf_pos'], gt['leaf_pos'])
    loss_full = F.mse_loss(pred['full_pos'], gt['full_pos'])
    loss_pose = F.mse_loss(pred['reduced_pose'], gt['reduced_pose'])
    loss_contact = F.binary_cross_entropy_with_logits(pred['contact_logits'], gt['contact'])

    loss_vel_local = _multi_scale_velocity_loss(pred['velocity_local'], gt['velocity_local'], velocity_scales)
    loss_root_vel = F.mse_loss(pred['root_velocity_world'], gt['root_velocity_world'])
    loss_root_pos = F.mse_loss(pred['root_position_world'], gt['root_position_world'])

    loss_obj_vel = F.mse_loss(pred['obj_velocity'], gt['obj_velocity'])
    loss_obj_pos = F.mse_loss(pred['obj_position'], gt['obj_position'])

    total = (
        weight_cfg['leaf'] * loss_leaf +
        weight_cfg['full'] * loss_full +
        weight_cfg['pose'] * loss_pose +
        weight_cfg['contact'] * loss_contact +
        weight_cfg['velocity_local'] * loss_vel_local +
        weight_cfg['root_velocity'] * loss_root_vel +
        weight_cfg['root_position'] * loss_root_pos +
        weight_cfg['obj_velocity'] * loss_obj_vel +
        weight_cfg['obj_position'] * loss_obj_pos
    )

    if return_tensors:
        # ✅ 返回detached tensor，避免GPU同步
        metrics = {
            'loss_leaf': loss_leaf.detach(),
            'loss_full': loss_full.detach(),
            'loss_pose': loss_pose.detach(),
            'loss_contact': loss_contact.detach(),
            'loss_velocity_local': loss_vel_local.detach(),
            'loss_root_velocity': loss_root_vel.detach(),
            'loss_root_position': loss_root_pos.detach(),
            'loss_obj_velocity': loss_obj_vel.detach(),
            'loss_obj_position': loss_obj_pos.detach(),
        }
    else:
        # 只在需要时才转换为float（epoch末尾）
        metrics = {
            'loss_leaf': float(loss_leaf.detach().cpu()),
            'loss_full': float(loss_full.detach().cpu()),
            'loss_pose': float(loss_pose.detach().cpu()),
            'loss_contact': float(loss_contact.detach().cpu()),
            'loss_velocity_local': float(loss_vel_local.detach().cpu()),
            'loss_root_velocity': float(loss_root_vel.detach().cpu()),
            'loss_root_position': float(loss_root_pos.detach().cpu()),
            'loss_obj_velocity': float(loss_obj_vel.detach().cpu()),
            'loss_obj_position': float(loss_obj_pos.detach().cpu()),
        }
    return total, metrics

