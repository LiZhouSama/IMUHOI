from typing import Dict, Tuple

import torch

from learning_utils import loss_q_only_2axis


def tip_human_object_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    human_root_dim: int,
    lambda_obj: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Reproduce the legacy TIP loss: quaternion/velocity loss on the human branch
    plus an MSE penalty on the object velocity (last three channels).
    """
    pred_flat = prediction.reshape(-1, prediction.size(-1))
    tgt_flat = target.reshape(-1, target.size(-1))

    human_pred = pred_flat[:, :human_root_dim]
    human_tgt = tgt_flat[:, :human_root_dim]
    loss_human = loss_q_only_2axis(human_tgt, human_pred)

    obj_pred = pred_flat[:, -3:]
    obj_tgt = tgt_flat[:, -3:]
    loss_obj = torch.mean((obj_pred - obj_tgt) ** 2)

    total = loss_human + lambda_obj * loss_obj
    stats = {
        "loss_human": float(loss_human.item()),
        "loss_obj": float(loss_obj.item()),
    }
    return total, stats


__all__ = ["tip_human_object_loss"]
