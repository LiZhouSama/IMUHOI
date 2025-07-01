import torch
import torch.nn as nn


class ContactAwareLoss(nn.Module):
    """
    接触感知损失函数 - 通过loss引导模型学习合理的接触行为
    """
    def __init__(self, contact_distance=0.1, ramp_up_steps=1000, loss_weights=None):
        super().__init__()
        self.contact_distance = contact_distance
        self.ramp_up_steps = ramp_up_steps
        
        # 默认损失权重
        default_weights = {
            'contact_distance': 1.0,
            'contact_velocity': 0.5,
            'approach_smoothness': 0.3,
        }
        self.loss_weights = loss_weights if loss_weights is not None else default_weights
        
    def forward(self, pred_hand_pos, pred_obj_pos, contact_probs, training_step=0):
        """
        计算接触感知损失
        
        Args:
            pred_hand_pos: [bs, seq, 2, 3] 预测的手部位置（左手、右手）
            pred_obj_pos: [bs, seq, 3] 预测的物体位置
            contact_probs: [bs, seq, 3] 接触概率（左手、右手、双手）
            training_step: 当前训练步数，用于渐进式权重
            
        Returns:
            total_loss: 总的接触损失
            loss_dict: 各项损失的详细信息
        """
        bs, seq_len = pred_hand_pos.shape[:2]
        device = pred_hand_pos.device
        losses = {}
        
        # 1. 接触距离一致性损失：接触概率高时，手应该接近物体目标距离
        # 计算手到物体的距离 [bs, seq, 2]
        hand_obj_dist = torch.norm(
            pred_hand_pos - pred_obj_pos.unsqueeze(2),  # [bs, seq, 2, 3] - [bs, seq, 1, 3]
            dim=-1
        )
        
        # 距离误差：实际距离与目标距离的差异
        distance_error = torch.abs(hand_obj_dist - self.contact_distance)  # [bs, seq, 2]
        
        # 用接触概率加权：接触概率越高，距离约束越强
        contact_distance_loss = contact_probs[:, :, :2] * distance_error  # [bs, seq, 2]
        losses['contact_distance'] = contact_distance_loss.mean()
        
        # 2. 接触时速度一致性损失：接触时手和物体应该有相似的运动
        if seq_len > 1:
            hand_velocity = pred_hand_pos[:, 1:] - pred_hand_pos[:, :-1]  # [bs, seq-1, 2, 3]
            obj_velocity = pred_obj_pos[:, 1:] - pred_obj_pos[:, :-1]      # [bs, seq-1, 3]
            
            # 计算速度差异
            velocity_diff = torch.norm(
                hand_velocity - obj_velocity.unsqueeze(2),  # [bs, seq-1, 2, 3] - [bs, seq-1, 1, 3]
                dim=-1
            )  # [bs, seq-1, 2]
            
            # 用接触概率加权
            contact_velocity_loss = contact_probs[:, 1:, :2] * velocity_diff  # [bs, seq-1, 2]
            losses['contact_velocity'] = contact_velocity_loss.mean()
        else:
            losses['contact_velocity'] = torch.tensor(0.0, device=device)
        
        # 3. 首次接触平滑性损失：鼓励平滑的接近行为
        if seq_len > 5:  # 需要足够的帧数
            # 检测首次接触
            contact_threshold = 0.5
            contact_binary = contact_probs[:, :, :2] > contact_threshold  # [bs, seq, 2]
            first_contact = contact_binary & (~torch.roll(contact_binary, 1, dims=1))  # [bs, seq, 2]
            first_contact[:, 0] = False  # 第一帧不算首次接触
            
            # 在首次接触前后的窗口内计算平滑性
            window_size = 3
            smoothness_losses = []
            
            for b in range(bs):
                for hand_idx in range(2):
                    contact_frames = first_contact[b, :, hand_idx].nonzero(as_tuple=True)[0]
                    
                    for contact_frame in contact_frames:
                        if contact_frame >= window_size and contact_frame < seq_len - window_size:
                            # 提取接触前后的距离轨迹
                            start_frame = max(0, contact_frame - window_size)
                            end_frame = min(seq_len, contact_frame + window_size + 1)
                            
                            distance_trajectory = hand_obj_dist[b, start_frame:end_frame, hand_idx]
                            
                            # 计算轨迹的平滑性（二阶差分）
                            if len(distance_trajectory) >= 3:
                                second_diff = torch.diff(distance_trajectory, n=2)
                                smoothness_loss = second_diff.abs().mean()
                                smoothness_losses.append(smoothness_loss)
            
            if smoothness_losses:
                losses['approach_smoothness'] = torch.stack(smoothness_losses).mean()
            else:
                losses['approach_smoothness'] = torch.tensor(0.0, device=device)
        else:
            losses['approach_smoothness'] = torch.tensor(0.0, device=device)
        
        # 4. 计算总损失（带阶段性权重）
        # 基础损失的渐进权重
        ramp_up_factor = min(1.0, training_step / self.ramp_up_steps)
        weighted_losses = {}
        total_loss = torch.tensor(0.0, device=device)
        
        for loss_name, loss_value in losses.items():
            weight = self.loss_weights.get(loss_name, 1.0)
            
            weighted_loss = weight * loss_value * ramp_up_factor
                
            weighted_losses[loss_name] = weighted_loss
            total_loss = total_loss + weighted_loss
        
        # 返回详细的损失信息
        loss_dict = {
            'total_contact_loss': total_loss,
            'ramp_up_factor': ramp_up_factor,
            **losses,  # 原始损失
            **{f'weighted_{k}': v for k, v in weighted_losses.items()}  # 加权后的损失
        }
        
        return total_loss, loss_dict
    
    def update_weights(self, new_weights):
        """更新损失权重"""
        self.loss_weights.update(new_weights)
    
    def get_current_ramp_factor(self, training_step):
        """获取当前的渐进因子"""
        return min(1.0, training_step / self.ramp_up_steps) 