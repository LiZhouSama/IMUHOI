import torch.nn.functional as F
import pytorch3d.transforms as transforms
def loss_p_obj(p, gt_pose, obj_v_pred, obj_v_gt, w_human: float = 1.0, w_obj: float = 1.0):
    target_pose = transforms.rotation_6d_to_matrix(gt_pose.reshape(gt_pose.size(0), gt_pose.size(1), 22, 6))
    target_pose = target_pose.view(gt_pose.size(0), gt_pose.size(1), -1)
    loss_pose = F.mse_loss(p, target_pose)
    
    loss_obj = F.mse_loss(obj_v_pred, obj_v_gt)

    total_loss = w_human * loss_pose + w_obj * loss_obj
    return total_loss, {
        'loss_p': loss_pose.item(),
        'loss_obj': loss_obj.item(),
    }