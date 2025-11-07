import torch
import torch.nn.functional as F


def _ensure_tensor(data):
    if isinstance(data, list):
        return torch.stack(data, dim=0)
    return data


def loss_vp_obj_trans(v, p, gt_vel, gt_pose, obj_v_pred, obj_v_gt,
                     contact_pred, contact_gt, root_vel_local_pred, root_vel_local_gt,
                     root_vel_pred, root_vel_gt, root_trans_pred, root_trans_gt,
                     w_human: float = 1.0, w_obj: float = 1.0,
                     w_contact: float = 1.0, w_root_vel_local: float = 1.0, w_root_vel: float = 1.0,
                     w_root_trans: float = 1.0):
                     
    vel_indices = [0, 1, 2, 3, 0, 3, 4, 5]  # 根据posers_config排序，腿、躯干、手
    target_vel = gt_vel[:, :, vel_indices].reshape(gt_vel.size(0), gt_vel.size(1), -1)
    loss_v = F.mse_loss(v, target_vel)

    target_pose = gt_pose.reshape(gt_pose.size(0), gt_pose.size(1), -1)
    loss_p = F.mse_loss(p, target_pose)

    loss_obj = F.mse_loss(obj_v_pred, obj_v_gt)
    loss_contact = F.binary_cross_entropy_with_logits(contact_pred, contact_gt)
    loss_root_vel_local = F.mse_loss(root_vel_local_pred, root_vel_local_gt)
    loss_root_vel = F.mse_loss(root_vel_pred, root_vel_gt)
    loss_root_trans = F.mse_loss(root_trans_pred, root_trans_gt)

    total_loss = (w_human * (loss_v + loss_p) +
                  w_obj * loss_obj +
                  w_contact * loss_contact +
                  w_root_vel_local * loss_root_vel_local +
                  w_root_vel * loss_root_vel +
                  w_root_trans * loss_root_trans)
    return total_loss, {
        'loss_v': loss_v.item(),
        'loss_p': loss_p.item(),
        'loss_obj': loss_obj.item(),
        'loss_contact': loss_contact.item(),
        'loss_root_vel_local': loss_root_vel_local.item(),
        'loss_root_vel': loss_root_vel.item(),
        'loss_root_trans': loss_root_trans.item(),
    }


