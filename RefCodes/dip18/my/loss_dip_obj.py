from typing import Optional, Dict

import torch
import torch.nn.functional as F


def _finite_difference(sequence: torch.Tensor) -> torch.Tensor:
    """
    Computes forward finite differences along the temporal axis.
    """
    diff = sequence[:, 1:] - sequence[:, :-1]
    pad_shape = list(diff.shape)
    pad_shape[1] = 1
    padding = torch.zeros(pad_shape, dtype=sequence.dtype, device=sequence.device)
    return torch.cat([padding, diff], dim=1)

def loss_p_obj(
    p_pred: torch.Tensor,
    p_gt: torch.Tensor,
    obj_v_pred: torch.Tensor,
    obj_v_gt: torch.Tensor,
    obj_p_pred: Optional[torch.Tensor] = None,
    obj_p_gt: Optional[torch.Tensor] = None,
    w_human_pose: float = 1.0,
    w_obj_vel: float = 1.0,
    w_obj_pos: float = 1.0,
) -> (torch.Tensor, Dict[str, float]):
    """
    Loss variant without the human velocity term.
    """
    metrics: Dict[str, float] = {}

    loss_pose = F.mse_loss(p_pred, p_gt)
    metrics["loss_pose"] = loss_pose.item()

    loss_obj_vel = F.mse_loss(obj_v_pred, obj_v_gt)
    metrics["loss_obj_vel"] = loss_obj_vel.item()

    if obj_p_pred is not None and obj_p_gt is not None:
        loss_obj_pos = F.mse_loss(obj_p_pred, obj_p_gt)
    else:
        loss_obj_pos = torch.zeros_like(loss_obj_vel)
    metrics["loss_obj_pos"] = loss_obj_pos.item()

    total = w_human_pose * loss_pose + w_obj_vel * loss_obj_vel + w_obj_pos * loss_obj_pos
    metrics["loss_total"] = total.item()
    return total, metrics

