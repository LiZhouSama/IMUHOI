import torch.nn.functional as F


def loss_vp_obj(v, p, gt_vel, gt_pose, obj_v_pred, obj_v_gt, w_human: float = 1.0, w_obj: float = 1.0):
    
    vel_indices = [0, 1, 2, 3, 0, 3, 4, 5]  # 根据posers_config排序，腿、躯干、手
    target_vel = gt_vel[:, :, vel_indices].reshape(gt_vel.size(0), gt_vel.size(1), -1)
    loss_v = F.mse_loss(v, target_vel)

    target_pose = gt_pose.reshape(gt_pose.size(0), gt_pose.size(1), -1)
    loss_p = F.mse_loss(p, target_pose)

    loss_obj = F.mse_loss(obj_v_pred, obj_v_gt)

    total_loss = w_human * (loss_v + loss_p) + w_obj * loss_obj
    return total_loss, {
        'loss_v': loss_v.item(),
        'loss_p': loss_p.item(),
        'loss_obj': loss_obj.item(),
    }

def loss_p_obj(p, gt_pose, obj_v_pred, obj_v_gt, w_human: float = 1.0, w_obj: float = 1.0):
    target_pose = gt_pose.reshape(gt_pose.size(0), gt_pose.size(1), -1)
    loss_pose = F.mse_loss(p, target_pose)
    
    loss_obj = F.mse_loss(obj_v_pred, obj_v_gt)

    total_loss = w_human * loss_pose + w_obj * loss_obj
    return total_loss, {
        'loss_pose': loss_pose.item(),
        'loss_obj': loss_obj.item(),
    }