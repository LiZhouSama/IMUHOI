import torch.nn
from torch.nn.functional import relu
import torch
from human_body_prior.body_model.body_model import BodyModel
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d
from utils.utils import global2local

from configs.global_config import FRAME_RATE, IMU_JOINTS_ROT, IMU_JOINT_NAMES, joint_set, acc_scale



# 定义辅助函数
INITIAL_STATE_DIM = 64
def lerp(val, low, high):
    """线性插值"""
    return low + (high - low) * val

def create_rnn_state_projection(initial_state_dim: int, hidden_size: int, num_layers: int, bidirectional: bool, device: torch.device) -> torch.nn.Linear:
    """创建用于将压缩初始状态映射到 (h0, c0) 拼接向量的线性层"""
    num_directions = 2 if bidirectional else 1
    return torch.nn.Linear(initial_state_dim, 2 * num_layers * num_directions * hidden_size).to(device)

def compute_rnn_initial_state(proj_layer: torch.nn.Linear, rnn_module: 'RNN', compressed_initial_state: torch.Tensor, batch_size: int):
    """由压缩初始状态生成 LSTM 的 (h0, c0) 初始状态元组"""
    projected_state = proj_layer(compressed_initial_state)
    num_layers = rnn_module.n_rnn_layer
    num_directions = rnn_module.num_directions
    hidden_size = rnn_module.n_hidden
    expected_shape = (batch_size, 2 * num_layers * num_directions, hidden_size)
    projected_state = projected_state.view(expected_shape)
    h_0_c_0 = torch.split(projected_state, num_layers * num_directions, dim=1)
    h_0 = h_0_c_0[0].permute(1, 0, 2).contiguous()
    c_0 = h_0_c_0[1].permute(1, 0, 2).contiguous()
    return (h_0, c_0)

class RNN(torch.nn.Module):
    """
    RNN模块，包括线性输入层、RNN和线性输出层。
    """
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=True, dropout=0.2):
        super(RNN, self).__init__()
        self.n_hidden = n_hidden
        self.n_rnn_layer = n_rnn_layer
        self.num_directions = 2 if bidirectional else 1
        # Set batch_first=True for LSTM
        self.rnn = torch.nn.LSTM(n_hidden, n_hidden, n_rnn_layer, bidirectional=bidirectional, batch_first=True)
        self.linear1 = torch.nn.Linear(n_input, n_hidden)
        self.linear2 = torch.nn.Linear(n_hidden * self.num_directions, n_output)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, h=None):
        # Input x is expected to be [batch_size, seq_len, n_input] due to batch_first=True
        # No need to unsqueeze/squeeze if input has the correct shape
        x = relu(self.linear1(self.dropout(x)))
        # Pass initial hidden state h (h_0, c_0) if provided
        x, h_out = self.rnn(x, h)
        # Output x is [batch_size, seq_len, n_hidden * num_directions]
        # Apply linear layer to each time step's output
        # Reshape x to [batch_size * seq_len, n_hidden * num_directions] before linear layer
        # Then reshape back to [batch_size, seq_len, n_output]
        batch_size, seq_len, _ = x.shape
        x = x.reshape(-1, self.n_hidden * self.num_directions)
        x = self.linear2(x)
        x = x.reshape(batch_size, seq_len, -1)

        return x, h_out # Return sequence output and final hidden state


class VelocityContactModule(torch.nn.Module):
    """
    模块1: 速度估计 + 手部接触预测
    输入: 所有IMU数据
    输出: 物体速度, 叶子节点速度, 手部接触概率
    """
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        self.num_human_imus = cfg.num_human_imus if hasattr(cfg, 'num_human_imus') else 6
        self.imu_dim = cfg.imu_dim if hasattr(cfg, 'imu_dim') else 9
        hidden_dim_multiplier = cfg.hidden_dim_multiplier if hasattr(cfg, 'hidden_dim_multiplier') else 1

        # 计算IMU输入维度
        n_human_imu = self.num_human_imus * self.imu_dim
        n_obj_imu = 1 * self.imu_dim
        
        # 手部接触网络相关索引
        self.lhand_imu_idx = IMU_JOINT_NAMES.index('left_hand')
        self.rhand_imu_idx = IMU_JOINT_NAMES.index('right_hand')
        
        # 物体运动检测阈值 (基于IMU数据)
        self.obj_imu_acc_threshold = cfg.obj_imu_acc_threshold if hasattr(cfg, 'obj_imu_acc_threshold') else 0.01  # IMU加速度变化阈值
        self.obj_imu_ori_threshold = cfg.obj_imu_ori_threshold if hasattr(cfg, 'obj_imu_ori_threshold') else 0.005  # IMU方向变化阈值
        self.contact_suppression_strength = cfg.contact_suppression_strength if hasattr(cfg, 'contact_suppression_strength') else 0.95  # 当无运动时将接触概率降低到此比例
        
        # 物体速度抑制参数
        self.velocity_suppression_strength = cfg.velocity_suppression_strength if hasattr(cfg, 'velocity_suppression_strength') else 0.9  # 当无运动时将速度降低到此比例
        
        # 速度估计网络 - 分离为两个独立的网络
        self.obj_velocity_net = RNN(n_obj_imu, 3, 128 * hidden_dim_multiplier)
        self.leaf_velocity_net = RNN(n_human_imu, joint_set.n_leaf * 3, 128 * hidden_dim_multiplier)
        
        # 手部接触预测网络
        n_hand_contact_input = 3 * self.imu_dim + 2 * 3 + 3  # 双手IMU + 物体IMU + 双手速度 + 物体速度
        self.hand_contact_net = RNN(n_hand_contact_input, 3, 128 * hidden_dim_multiplier)
        
        # 初始状态维度
        self.initial_state_dim = INITIAL_STATE_DIM
        
        # 用于生成初始状态的线性层
        self.proj_obj_velocity_state = create_rnn_state_projection(self.initial_state_dim, self.obj_velocity_net.n_hidden, self.obj_velocity_net.n_rnn_layer, self.obj_velocity_net.num_directions == 2, device)
        self.proj_leaf_velocity_state = create_rnn_state_projection(self.initial_state_dim, self.leaf_velocity_net.n_hidden, self.leaf_velocity_net.n_rnn_layer, self.leaf_velocity_net.num_directions == 2, device)
        self.proj_hand_contact_state = create_rnn_state_projection(self.initial_state_dim, self.hand_contact_net.n_hidden, self.hand_contact_net.n_rnn_layer, self.hand_contact_net.num_directions == 2, device)
    
    def detect_object_motion_from_imu(self, obj_imu):
        """
        基于物体IMU数据检测物体运动状态，参考preprocess.py中的逻辑
        
        Args:
            obj_imu: 物体IMU数据 [batch_size, seq_len, 9] 
                     格式通常为 [acc_x, acc_y, acc_z, ori_6d] (前3维加速度，后6维方向)
            
        Returns:
            motion_mask: 物体运动掩码 [batch_size, seq_len] (True表示有运动)
        """
        batch_size, seq_len, imu_dim = obj_imu.shape
        device = obj_imu.device
        
        # 初始化运动掩码，第一帧默认为False
        motion_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        if seq_len <= 1:
            return motion_mask
        
        # 提取加速度信息 (前3维)
        obj_acc = obj_imu[:, :, :3]  # [batch_size, seq_len, 3]
        
        # 提取方向信息 (后6维，6D旋转表示)
        obj_ori_6d = obj_imu[:, :, 3:]  # [batch_size, seq_len, 6]
        
        # 将6D旋转表示转换为旋转矩阵，以与preprocess.py保持一致
        from pytorch3d.transforms import rotation_6d_to_matrix
        obj_rot_mat = rotation_6d_to_matrix(obj_ori_6d.reshape(-1, 6)).reshape(batch_size, seq_len, 3, 3)
        
        # 计算相邻帧的加速度变化
        acc_diff = torch.norm(obj_acc[:, 1:] - obj_acc[:, :-1], dim=2)  # [batch_size, seq_len-1]
        
        # 计算相邻帧的旋转变化 (使用Frobenius范数，与preprocess.py保持一致)
        rot_diff = torch.norm(obj_rot_mat[:, 1:] - obj_rot_mat[:, :-1], dim=(2, 3))  # [batch_size, seq_len-1]
        
        # 使用配置的IMU检测阈值
        acc_threshold = self.obj_imu_acc_threshold  # IMU加速度变化阈值
        ori_threshold = self.obj_imu_ori_threshold  # IMU方向变化阈值 (对应旋转矩阵的Frobenius范数)
        
        # 判断运动：加速度或旋转变化超过阈值
        motion_detected = (acc_diff > acc_threshold) | (rot_diff > ori_threshold)
        
        # 填充运动掩码 (第一帧保持False)
        motion_mask[:, 1:] = motion_detected
        
        return motion_mask
    
    def forward(self, imu_data, compressed_initial_state, use_object_data=True):
        """
        前向传播
        
        Args:
            imu_data: [bs, seq, n_imu] IMU数据
            compressed_initial_state: [bs, initial_state_dim] 压缩的初始状态
            use_object_data: 是否使用物体数据
        
        Returns:
            dict: 包含速度和接触预测的字典
        """
        batch_size, seq_len, _ = imu_data.shape
        device = imu_data.device
        
        # 分离人体和物体IMU
        human_imu_data = imu_data[:, :, :self.num_human_imus*self.imu_dim]
        obj_imu_data = imu_data[:, :, self.num_human_imus*self.imu_dim:]
        
        # 如果不使用物体数据，将物体IMU设为0
        if not use_object_data:
            obj_imu_data = torch.zeros_like(obj_imu_data)
        
        # 速度估计 - 使用分离的网络
        # 物体速度估计
        obj_velocity_initial_state = compute_rnn_initial_state(self.proj_obj_velocity_state, self.obj_velocity_net, compressed_initial_state, batch_size)
        pred_obj_vel, _ = self.obj_velocity_net(obj_imu_data, obj_velocity_initial_state)
        
        # 叶子节点速度估计
        leaf_velocity_initial_state = compute_rnn_initial_state(self.proj_leaf_velocity_state, self.leaf_velocity_net, compressed_initial_state, batch_size)
        pred_leaf_vel_flat, _ = self.leaf_velocity_net(human_imu_data, leaf_velocity_initial_state)
        pred_leaf_vel = pred_leaf_vel_flat.reshape(batch_size, seq_len, joint_set.n_leaf, 3)
        
        # 提取双手速度
        lhand_vel = pred_leaf_vel[:, :, 3, :]  # 左手速度
        rhand_vel = pred_leaf_vel[:, :, 4, :]  # 右手速度
        
        # 手部接触预测
        human_imu_reshaped = human_imu_data.reshape(batch_size, seq_len, self.num_human_imus, self.imu_dim)
        lhand_imu = human_imu_reshaped[:, :, self.lhand_imu_idx, :]
        rhand_imu = human_imu_reshaped[:, :, self.rhand_imu_idx, :]
        
        # 构建手部接触网络的输入
        imu_feat = torch.cat([lhand_imu, rhand_imu, obj_imu_data], dim=2)
        hand_vel_feat = torch.cat([lhand_vel, rhand_vel], dim=2)
        hand_contact_input = torch.cat([imu_feat, hand_vel_feat, pred_obj_vel], dim=2)
        
        # 获取手部接触网络的初始状态
        hand_contact_initial_state = compute_rnn_initial_state(self.proj_hand_contact_state, self.hand_contact_net, compressed_initial_state, batch_size)
        
        # 手部接触预测
        pred_hand_contact_prob_logits, _ = self.hand_contact_net(hand_contact_input, hand_contact_initial_state)
        pred_hand_contact_prob = torch.sigmoid(pred_hand_contact_prob_logits)
        
        # 应用基于物体运动的先验调整
        if use_object_data and obj_imu_data.numel() > 0:
            # 基于物体IMU数据检测物体运动状态
            motion_mask = self.detect_object_motion_from_imu(obj_imu_data)  # [batch_size, seq_len]
            
            # 1. 物体速度抑制：当物体没有运动时，大幅降低预测速度
            # 保存抑制前的原始速度预测
            pred_obj_vel_before_suppression = pred_obj_vel.clone()
            
            velocity_suppression_factor = torch.where(
                motion_mask.unsqueeze(-1),  # [batch_size, seq_len, 1]
                torch.ones_like(pred_obj_vel),  # 有运动时不调整
                torch.full_like(pred_obj_vel, 1.0 - self.velocity_suppression_strength)  # 无运动时降低速度
            )
            
            # 应用速度抑制因子
            pred_obj_vel = pred_obj_vel * velocity_suppression_factor
            
            # 2. 手部接触概率抑制：当物体没有运动时，大幅降低接触概率
            # motion_mask为False的位置表示物体没有运动
            contact_suppression_factor = torch.where(
                motion_mask.unsqueeze(-1),  # [batch_size, seq_len, 1]
                torch.ones_like(pred_hand_contact_prob),  # 有运动时不调整
                torch.full_like(pred_hand_contact_prob, 1.0 - self.contact_suppression_strength)  # 无运动时降低概率
            )
            
            # 应用接触概率抑制因子
            pred_hand_contact_prob = pred_hand_contact_prob * contact_suppression_factor

        pred_obj_vel_watch = pred_obj_vel.detach().cpu().numpy()
        obj_imu_data_watch = obj_imu_data.detach().cpu().numpy()
        
        # 构建返回字典
        result = {
            "pred_obj_vel": pred_obj_vel,
            "pred_leaf_vel": pred_leaf_vel,
            "pred_leaf_vel_flat": pred_leaf_vel_flat,
            "pred_hand_contact_prob": pred_hand_contact_prob
        }
        
        # 添加运动检测和抑制相关的调试信息（如果使用了物体数据）
        if use_object_data and obj_imu_data.numel() > 0:
            result.update({
                "obj_motion_mask": motion_mask,  # 物体运动掩码 [batch_size, seq_len]
                "velocity_suppression_factor": velocity_suppression_factor,  # 速度抑制因子 [batch_size, seq_len, 3]
                "contact_suppression_factor": contact_suppression_factor,  # 接触抑制因子 [batch_size, seq_len, 3]
                "pred_obj_vel_before_suppression": pred_obj_vel_before_suppression,  # 抑制前的原始速度预测
            })
        
        return result


class HumanPoseModule(torch.nn.Module):
    """
    模块2: 人体姿态 + 足部接触 + 人体平移
    输入: 人体IMU + 叶子节点速度
    输出: 人体姿态, 足部接触概率, 人体平移
    """
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        self.num_human_imus = cfg.num_human_imus if hasattr(cfg, 'num_human_imus') else 6
        self.imu_dim = cfg.imu_dim if hasattr(cfg, 'imu_dim') else 9
        self.joint_dim = cfg.joint_dim if hasattr(cfg, 'joint_dim') else 6
        self.num_joints = cfg.num_joints if hasattr(cfg, 'num_joints') else 22
        hidden_dim_multiplier = cfg.hidden_dim_multiplier if hasattr(cfg, 'hidden_dim_multiplier') else 1
        
        # 计算输入维度
        n_human_imu = self.num_human_imus * self.imu_dim
        
        # 姿态估计级联架构
        self.pose_s1 = RNN(n_human_imu + joint_set.n_leaf * 3, joint_set.n_leaf * 3, 256 * hidden_dim_multiplier)
        self.pose_s2 = RNN(joint_set.n_leaf * 3 + n_human_imu, joint_set.n_full * 3, 64 * hidden_dim_multiplier)
        self.pose_s3 = RNN(joint_set.n_full * 3 + n_human_imu, joint_set.n_reduced * 6, 128 * hidden_dim_multiplier)

        # 足部接触概率预测网络
        self.contact_prob_net = RNN(joint_set.n_leaf * 3 + n_human_imu, 2, 64 * hidden_dim_multiplier)

        # 基于运动学的速度分支预测网络
        self.trans_b2 = RNN(joint_set.n_full * 3 + n_human_imu, 3, 256 * hidden_dim_multiplier, bidirectional=False)
        
        # 初始状态维度
        self.initial_state_dim = INITIAL_STATE_DIM
        
        # 用于生成初始状态的线性层
        self.proj_s1_state = create_rnn_state_projection(self.initial_state_dim, self.pose_s1.n_hidden, self.pose_s1.n_rnn_layer, self.pose_s1.num_directions == 2, device)
        self.proj_s2_state = create_rnn_state_projection(self.initial_state_dim, self.pose_s2.n_hidden, self.pose_s2.n_rnn_layer, self.pose_s2.num_directions == 2, device)
        self.proj_s3_state = create_rnn_state_projection(self.initial_state_dim, self.pose_s3.n_hidden, self.pose_s3.n_rnn_layer, self.pose_s3.num_directions == 2, device)
        self.proj_contact_state = create_rnn_state_projection(self.initial_state_dim, self.contact_prob_net.n_hidden, self.contact_prob_net.n_rnn_layer, self.contact_prob_net.num_directions == 2, device)
        self.proj_trans_b2_state = create_rnn_state_projection(self.initial_state_dim, self.trans_b2.n_hidden, self.trans_b2.n_rnn_layer, self.trans_b2.num_directions == 2, device)
        
        # SMPL Body Model for FK
        self.body_model = BodyModel(bm_fname=cfg.bm_path, num_betas=16).to(device)
        for p in self.body_model.parameters():
            p.requires_grad = False
        
        # 注册buffer
        self.register_buffer('parents_tensor', self.body_model.kintree_table[0].long())
        self.register_buffer('imu_joints_rot', torch.tensor(IMU_JOINTS_ROT, dtype=torch.long))
        self.register_buffer('gravity_velocity', torch.tensor([0.0, -0.018, 0], dtype=torch.float32))
        
        # 参数
        self.vel_scale = cfg.vel_scale if hasattr(cfg, 'vel_scale') else 1.66
        self.prob_threshold_min = cfg.prob_threshold_min if hasattr(cfg, 'prob_threshold_min') else 0.5
        self.prob_threshold_max = cfg.prob_threshold_max if hasattr(cfg, 'prob_threshold_max') else 0.9
        self.floor_y = cfg.floor_y if hasattr(cfg, 'floor_y') else 0.0
        
        # 足部关节索引
        self.left_foot_idx, self.right_foot_idx = 7, 8 
        self.wrist_l_idx, self.wrist_r_idx = 20, 21

    def _reduced_global_to_local_6d(self, imu_rotation_6d, reduced_pose_6d):
        """将预测的reduced全局旋转转换为完整的局部6D旋转表示"""
        bs_seq = reduced_pose_6d.shape[0]
        device = reduced_pose_6d.device

        # 转换输入为旋转矩阵
        R_imu = rotation_6d_to_matrix(imu_rotation_6d)
        R_reduced_pred = rotation_6d_to_matrix(reduced_pose_6d.reshape(bs_seq, joint_set.n_reduced, 6))
        
        # 构建完整的全局旋转矩阵
        R_global_full = torch.eye(3, device=device).repeat(bs_seq, self.num_joints, 1, 1)
        R_global_full[:, joint_set.reduced] = R_reduced_pred
        R_global_full[:, IMU_JOINTS_ROT] = R_imu
        
        # 使用global2local转换为局部旋转矩阵
        local_rotmats = global2local(R_global_full, self.parents_tensor)
        local_rotmats[:,0] = R_imu[:, -1]
        local_rotmats[:, [20, 21, 7, 8]] = torch.eye(3, device=device).repeat(bs_seq, 4, 1, 1)

        # 转换回6D表示
        local_pose_6d = matrix_to_rotation_6d(local_rotmats.reshape(-1, 3, 3)).reshape(bs_seq, self.num_joints * 6)

        return local_pose_6d
        
    def _prob_to_weight(self, p):
        """将概率值转换为权重"""
        threshold_min = float(self.prob_threshold_min)
        threshold_max = float(self.prob_threshold_max)
        
        if threshold_min >= threshold_max:
            return torch.full_like(p, 0.5).unsqueeze(1)

        clamped_p = p.clamp(threshold_min, threshold_max)
        weight = (clamped_p - threshold_min) / (threshold_max - threshold_min)
        return weight.unsqueeze(1)
    
    def forward(self, human_imu_data, pred_leaf_vel_flat, compressed_initial_state):
        """
        前向传播

        Args:
            human_imu_data: [bs, seq, n_human_imu] 人体IMU数据
            pred_leaf_vel_flat: [bs, seq, n_leaf*3] 叶子节点速度
            compressed_initial_state: [bs, initial_state_dim] 压缩的初始状态

        Returns:
            dict: 包含姿态、接触和平移预测的字典
        """
        batch_size, seq_len, _ = human_imu_data.shape
        device = human_imu_data.device
        
        # 级联姿态估计
        s1_initial_state = compute_rnn_initial_state(self.proj_s1_state, self.pose_s1, compressed_initial_state, batch_size)
        s2_initial_state = compute_rnn_initial_state(self.proj_s2_state, self.pose_s2, compressed_initial_state, batch_size)
        s3_initial_state = compute_rnn_initial_state(self.proj_s3_state, self.pose_s3, compressed_initial_state, batch_size)
        contact_initial_state = compute_rnn_initial_state(self.proj_contact_state, self.contact_prob_net, compressed_initial_state, batch_size)
        trans_b2_initial_state = compute_rnn_initial_state(self.proj_trans_b2_state, self.trans_b2, compressed_initial_state, batch_size)

        # 第一阶段：预测关键关节位置
        s1_input = torch.cat([human_imu_data, pred_leaf_vel_flat], dim=2)
        s1_output, _ = self.pose_s1(s1_input, s1_initial_state)
        pred_leaf_pos = s1_output.reshape(batch_size, seq_len, joint_set.n_leaf, 3)

        # 第二阶段：预测所有关节位置
        s2_input = torch.cat([s1_output, human_imu_data], dim=2)
        s2_output, _ = self.pose_s2(s2_input, s2_initial_state)
        pred_full_pos = s2_output.reshape(batch_size, seq_len, joint_set.n_full, 3)

        # 第三阶段：预测关节旋转
        s3_input = torch.cat([s2_output, human_imu_data], dim=2)
        s3_output, _ = self.pose_s3(s3_input, s3_initial_state)
        pred_reduced_global_pose_6d = s3_output
        
        # 转换为局部姿态
        human_imu_flat = human_imu_data.reshape(-1, self.num_human_imus, self.imu_dim)
        human_imu_rot_flat = human_imu_flat[:, :, 3:9]
        pred_motion_6d_flat = self._reduced_global_to_local_6d(
            human_imu_rot_flat,
            pred_reduced_global_pose_6d.reshape(-1, joint_set.n_reduced * 6)
        )
        pred_motion = pred_motion_6d_flat.reshape(batch_size, seq_len, -1)
        
        # 足部接触概率预测
        contact_prob_input = torch.cat([s1_output, human_imu_data], dim=2)
        contact_probability, _ = self.contact_prob_net(contact_prob_input, contact_initial_state)
        contact_probability = torch.sigmoid(contact_probability)
        contact_probability_flat = contact_probability.reshape(-1, 2)
        
        # 速度分支B2预测
        trans_b2_input = torch.cat([s2_output, human_imu_data], dim=2)
        velocity_b2, _ = self.trans_b2(trans_b2_input, trans_b2_initial_state)
        velocity_b2_flat = velocity_b2.reshape(-1, 3)
        
        # 计算人体平移
        pose_mat_flat = rotation_6d_to_matrix(pred_motion_6d_flat.reshape(-1, 22, 6))
        root_orient_mat_flat = pose_mat_flat[:, 0]
        pose_axis_angle_flat = matrix_to_axis_angle(pose_mat_flat).reshape(batch_size*seq_len, -1)
        
        # Forward Kinematics
        body_model_output = self.body_model(
            root_orient=pose_axis_angle_flat[:, :3],
            pose_body=pose_axis_angle_flat[:, 3:],
        )
        j_flat = body_model_output.Jtr[:, :self.num_joints, :]

        # 计算速度分支B1
        j_seq = j_flat.reshape(batch_size, seq_len, self.num_joints, 3)
        left_foot_pos = j_seq[:, :, self.left_foot_idx, :]
        right_foot_pos = j_seq[:, :, self.right_foot_idx, :]
        
        prev_left_foot_pos = torch.roll(left_foot_pos, shifts=1, dims=1)
        prev_right_foot_pos = torch.roll(right_foot_pos, shifts=1, dims=1)
        prev_left_foot_pos[:, 0, :] = left_foot_pos[:, 0, :]
        prev_right_foot_pos[:, 0, :] = right_foot_pos[:, 0, :]
        
        lfoot_vel = (prev_left_foot_pos - left_foot_pos)
        rfoot_vel = (prev_right_foot_pos - right_foot_pos)
        lfoot_vel_flat = lfoot_vel.reshape(-1, 3)
        rfoot_vel_flat = rfoot_vel.reshape(-1, 3)
        
        contact_indices = contact_probability_flat.max(dim=1).indices
        chosen_foot_vel = torch.where(contact_indices.unsqueeze(1) == 0, rfoot_vel_flat, lfoot_vel_flat)
        tran_b1_vel = self.gravity_velocity.unsqueeze(0) + chosen_foot_vel
        
        # 计算速度分支B2
        velocity_b2_flat = torch.bmm(root_orient_mat_flat, velocity_b2_flat.unsqueeze(-1)).squeeze(-1)
        tran_b2_vel = velocity_b2_flat * self.vel_scale / FRAME_RATE
        
        # 融合速度分支
        max_contact_prob = contact_probability_flat.max(dim=1).values
        weight = self._prob_to_weight(max_contact_prob)
        velocity_flat = lerp(tran_b2_vel, tran_b1_vel, weight)
        
        # 计算最终平移
        velocity = velocity_flat.reshape(batch_size, seq_len, 3)
        trans = torch.cumsum(velocity, dim=1)
        
        # 计算手部位置（用于物体平移估计）
        pred_root_orient_6d = pred_motion[:, :, :6]
        pred_body_pose_6d = pred_motion[:, :, 6:]
        
        pred_root_orient_6d_flat = pred_root_orient_6d.reshape(-1, 6)
        pred_body_pose_6d_flat = pred_body_pose_6d.reshape(-1, 21, 6)
        pred_root_orient_mat_flat = rotation_6d_to_matrix(pred_root_orient_6d_flat)
        pred_body_pose_mat_flat = rotation_6d_to_matrix(pred_body_pose_6d_flat.reshape(-1, 6)).reshape(-1, 21, 3, 3)
        pred_root_orient_axis_flat = matrix_to_axis_angle(pred_root_orient_mat_flat)
        pred_body_pose_axis_flat = matrix_to_axis_angle(pred_body_pose_mat_flat.reshape(-1, 3, 3)).reshape(-1, 21*3)
        pred_transl_flat = trans.reshape(-1, 3)
        
        pred_pose_body_input = {
            'root_orient': pred_root_orient_axis_flat, 
            'pose_body': pred_body_pose_axis_flat, 
            'trans': pred_transl_flat
        }
        pred_smplh_out = self.body_model(**pred_pose_body_input)
        pred_joints_all = pred_smplh_out.Jtr.view(batch_size, seq_len, -1, 3)
        
        # 提取手部位置
        lhand_pos_global = pred_joints_all[:, :, self.wrist_l_idx, :]
        rhand_pos_global = pred_joints_all[:, :, self.wrist_r_idx, :]
        hand_positions = torch.stack([lhand_pos_global, rhand_pos_global], dim=2)
        
        return {
            "pred_leaf_pos": pred_leaf_pos,
            "pred_full_pos": pred_full_pos,
            "motion": pred_motion,
            "tran_b2_vel": velocity_b2_flat.reshape(batch_size, seq_len, 3),
            "contact_probability": contact_probability,
            "root_vel": velocity * FRAME_RATE,
            "root_pos": trans,
            "pred_hand_pos": hand_positions,
            "hands_pos_feat": torch.cat([lhand_pos_global, rhand_pos_global], dim=2)
        }

class UnifiedObjectTransModule(torch.nn.Module):
    """
    统一物体位置估计模块：左/右手FK + IMU速度 + 门控逐帧融合
    """
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        self.imu_dim = cfg.imu_dim if hasattr(cfg, 'imu_dim') else 9
        self.num_human_imus = cfg.num_human_imus if hasattr(cfg, 'num_human_imus') else 6
        hidden_dim_multiplier = cfg.hidden_dim_multiplier if hasattr(cfg, 'hidden_dim_multiplier') else 1

        # 噪声与门控参数
        self.gating_prior_beta = getattr(cfg, 'gating_prior_beta', 0)     # 先验强度: gating_prior_beta控制先验知识的影响程度
        self.gating_temperature = getattr(cfg, 'gating_temperature', 5.0)   # 温度控制: 通过gating_temperature控制权重的锐度，温度越低权重越集中
        
        # 门控权重平滑参数（仅在推理时使用，训练时保持真实梯度）
        self.gating_smoothing_enabled = getattr(cfg, 'gating_smoothing_enabled', False)
        self.gating_smoothing_alpha = getattr(cfg, 'gating_smoothing_alpha', 0.6)  # 平滑系数，越大越平滑
        self.gating_max_change = getattr(cfg, 'gating_max_change', 0.25)  # 相邻帧最大权重变化（选择性平滑）

        # 初始状态维度
        self.initial_state_dim = INITIAL_STATE_DIM

        # 分支输入维度
        n_fk_branch_input = 34  # obj_rot6 + hand_pos3 + hand_contact1 + obj_imu9 + hand_imu9 + obj_vel3 + rot_delta3
        n_imu_branch_input = 21 # obj_imu9 + obj_vel3 + rot_delta3 + obj_rot6
        n_gating_input = 9      # contact_prob3 + obj_vel3 + obj_imu_acc3

        # 左/右手: 回归物体系单位方向(3) + 长度(1)
        self.lhand_fk_head = RNN(n_fk_branch_input, 4, 128 * hidden_dim_multiplier)
        self.rhand_fk_head = RNN(n_fk_branch_input, 4, 128 * hidden_dim_multiplier)

        # IMU分支: 回归速度(3)
        # self.imu_head = RNN(n_imu_branch_input, 3, 128 * hidden_dim_multiplier)

        # 门控: 输出3路logits
        self.gating_head = RNN(n_gating_input, 3, 64 * hidden_dim_multiplier)

        # 初始状态投影
        self.proj_lhand_fk_state = create_rnn_state_projection(self.initial_state_dim, self.lhand_fk_head.n_hidden, self.lhand_fk_head.n_rnn_layer, self.lhand_fk_head.num_directions == 2, device)
        self.proj_rhand_fk_state = create_rnn_state_projection(self.initial_state_dim, self.rhand_fk_head.n_hidden, self.rhand_fk_head.n_rnn_layer, self.rhand_fk_head.num_directions == 2, device)
        # self.proj_imu_state = create_rnn_state_projection(self.initial_state_dim, self.imu_head.n_hidden, self.imu_head.n_rnn_layer, self.imu_head.num_directions == 2, device)
        self.proj_gate_state = create_rnn_state_projection(self.initial_state_dim, self.gating_head.n_hidden, self.gating_head.n_rnn_layer, self.gating_head.num_directions == 2, device)

    
    def predict_object_position_from_contact(self, pred_hand_contact_prob, pred_hand_positions, obj_rot_matrix, gt_obj_trans):
        """
        通过FK从预测的接触和手位置计算物体位置
        FK公式: \hat p_O = \hat p_H + ^WR_O · 接触首帧的 ^Ov_{HO} · 接触首帧的| \hat p_H -  \hat p_O|
        
        Args:
            pred_hand_contact_prob: [bs, seq, 3] 手部接触概率 (左手、右手、物体)
            pred_hand_positions: [bs, seq, 2, 3] 预测的手部位置 [左手、右手]
            obj_rot_matrix: [bs, seq, 3, 3] 物体旋转矩阵 ^WR_O
            gt_obj_trans: [bs, seq, 3] 真实物体位置（仅用于第一个接触段的初始化）
        
        Returns:
            computed_obj_trans: [bs, seq, 3] 预测的物体位置
            fk_bone_info: dict, FK计算过程中的骨长和方向信息，包含：
                - fk_lhand_bone_length: [bs, seq] FK左手虚拟骨长
                - fk_rhand_bone_length: [bs, seq] FK右手虚拟骨长
                - fk_lhand_direction: [bs, seq, 3] FK左手方向（物体坐标系）
                - fk_rhand_direction: [bs, seq, 3] FK右手方向（物体坐标系）
        """
        batch_size, seq_len, _ = pred_hand_contact_prob.shape
        device = pred_hand_contact_prob.device
        
        # 初始化输出
        computed_obj_trans = gt_obj_trans.clone()  # 开始时使用真值位置作为基础
        
        # 初始化FK骨长和方向信息
        fk_lhand_bone_length = torch.zeros(batch_size, seq_len, device=device)
        fk_rhand_bone_length = torch.zeros(batch_size, seq_len, device=device)
        fk_lhand_direction = torch.zeros(batch_size, seq_len, 3, device=device)
        fk_rhand_direction = torch.zeros(batch_size, seq_len, 3, device=device)
        
        # 阈值设定
        contact_threshold = 0.5
        
        # 处理每个batch
        for b in range(batch_size):
            # 提取当前batch的数据
            lhand_contact_prob = pred_hand_contact_prob[b, :, 0]  # [seq]
            rhand_contact_prob = pred_hand_contact_prob[b, :, 1]  # [seq]
            lhand_pos = pred_hand_positions[b, :, 0, :]  # [seq, 3]
            rhand_pos = pred_hand_positions[b, :, 1, :]  # [seq, 3]
            obj_rot_mat = obj_rot_matrix[b]  # [seq, 3, 3]
            gt_obj_pos = gt_obj_trans[b]  # [seq, 3]
            
            # 转换为二进制接触标签
            lhand_contact = (lhand_contact_prob > contact_threshold).float()
            rhand_contact = (rhand_contact_prob > contact_threshold).float()
            lhand_contact[0] = 0
            rhand_contact[0] = 0
            
            # 找到接触变化点（从0到1的转换）
            lhand_start_contact = torch.zeros_like(lhand_contact)
            rhand_start_contact = torch.zeros_like(rhand_contact)
            
            # 检测接触开始
            for t in range(1, seq_len):
                if lhand_contact[t] > 0 and lhand_contact[t-1] == 0:
                    lhand_start_contact[t] = 1
                if rhand_contact[t] > 0 and rhand_contact[t-1] == 0:
                    rhand_start_contact[t] = 1
            
            # 找到所有接触段
            contact_segments = []
            current_contact = None
            segment_start = None
            
            for t in range(seq_len):
                # 检查是否有新的接触开始
                new_contact = None
                if lhand_start_contact[t] > 0:
                    new_contact = 'left'
                elif rhand_start_contact[t] > 0:
                    new_contact = 'right'
                
                # 检查当前接触手是否有接触
                has_contact = lhand_contact[t] > 0 if current_contact == 'left' else rhand_contact[t] > 0
                has_contact_another = lhand_contact[t] > 0 if current_contact == 'right' else rhand_contact[t] > 0
                
                if new_contact is not None:
                    # 结束之前的段
                    if current_contact is not None and segment_start is not None:
                        contact_segments.append({
                            'hand': current_contact,
                            'start': segment_start,
                            'end': t - 1
                        })
                    
                    # 开始新的段
                    current_contact = new_contact
                    segment_start = t
                elif not has_contact and has_contact_another:
                    # 结束之前的段
                    if current_contact is not None and segment_start is not None:
                        contact_segments.append({
                            'hand': current_contact,
                            'start': segment_start,
                            'end': t - 1
                        })
                        # 开始新的段
                    current_contact = 'left' if current_contact == 'right' else 'right'
                    segment_start = t
                elif not has_contact and current_contact is not None:
                    # 接触结束
                    contact_segments.append({
                        'hand': current_contact,
                        'start': segment_start,
                        'end': t - 1
                    })
                    current_contact = None
                    segment_start = None
            
            # 处理最后一个段
            if current_contact is not None and segment_start is not None:
                contact_segments.append({
                    'hand': current_contact,
                    'start': segment_start,
                    'end': seq_len - 1
                })
            
            # 按时间顺序排序接触段
            contact_segments.sort(key=lambda x: x['start'])
            
            # 跟踪当前物体位置（用于下一个接触段的参考）
            current_obj_position = None
            
            # 对每个接触段应用FK
            for segment_idx, segment in enumerate(contact_segments):
                hand_type = segment['hand']
                start_frame = segment['start']
                end_frame = segment['end']
                
                if hand_type == 'left':
                    hand_pos_segment = lhand_pos[start_frame:end_frame+1]  # [seg_len, 3]
                else:  # right
                    hand_pos_segment = rhand_pos[start_frame:end_frame+1]  # [seg_len, 3]
                
                # 计算接触首帧的方向向量和距离
                initial_hand_pos = hand_pos_segment[0]  # [3]
                initial_obj_rot_mat = obj_rot_mat[start_frame]  # [3, 3]
                
                # 决定用于计算的初始物体位置
                if segment_idx == 0:
                    # 第一个接触段使用真值位置
                    initial_obj_pos = gt_obj_pos[start_frame]  # [3]
                else:
                    # 后续接触段使用前一个接触段的最后预测位置
                    initial_obj_pos = current_obj_position  # [3]
                
                # 计算世界坐标系下手指向物体的向量
                hand_to_obj_world = initial_obj_pos - initial_hand_pos  # [3]
                initial_distance = torch.norm(hand_to_obj_world)  # scalar
                
                # 计算单位方向向量（避免距离被计算两次）
                if initial_distance > 1e-6:  # 避免除零
                    hand_to_obj_unit = hand_to_obj_world / initial_distance  # 归一化方向
                else:
                    # 如果距离太小，使用默认方向
                    hand_to_obj_unit = torch.tensor([0.0, 0.0, 1.0], device=hand_to_obj_world.device, dtype=hand_to_obj_world.dtype)
                    initial_distance = 0.1  # 设置最小距离
                
                # 转换到物体坐标系：^Ov_{HO} = ^WR_O^T * ^W(单位方向向量)
                initial_obj_rot_mat_inv = initial_obj_rot_mat.transpose(0, 1)  # [3, 3]
                obj_direction_initial = initial_obj_rot_mat_inv @ hand_to_obj_unit  # [3] 单位方向向量
                
                # 记录当前接触段的FK骨长和方向信息
                if hand_type == 'left':
                    fk_lhand_bone_length[b, start_frame:end_frame+1] = initial_distance
                    fk_lhand_direction[b, start_frame:end_frame+1] = obj_direction_initial.unsqueeze(0).repeat(end_frame-start_frame+1, 1)
                else:  # right
                    fk_rhand_bone_length[b, start_frame:end_frame+1] = initial_distance
                    fk_rhand_direction[b, start_frame:end_frame+1] = obj_direction_initial.unsqueeze(0).repeat(end_frame-start_frame+1, 1)
                
                # 对这个接触段的每一帧应用FK
                for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
                    current_hand_pos = hand_pos_segment[i]  # [3]
                    current_obj_rot_mat = obj_rot_mat[frame_idx]  # [3, 3]
                    
                    # FK公式: \hat p_O = \hat p_H + ^WR_O · ^Ov_{HO} · distance
                    # 将物体坐标系下的方向向量转换到世界坐标系
                    direction_world = current_obj_rot_mat @ obj_direction_initial  # [3]
                    
                    # 计算物体位置
                    predicted_obj_pos = current_hand_pos + direction_world * initial_distance  # [3]
                    
                    # 更新结果
                    computed_obj_trans[b, frame_idx] = predicted_obj_pos
                
                # 记录这个接触段的最后位置，用于后续接触段和非接触期间
                current_obj_position = computed_obj_trans[b, end_frame].clone()
                
                # 填充当前接触段之前到上一个接触段结束之间的空隙
                if segment_idx > 0:
                    prev_segment = contact_segments[segment_idx - 1]
                    prev_end_frame = prev_segment['end']
                    prev_last_position = computed_obj_trans[b, prev_end_frame].clone()
                    
                    # 填充空隙帧（保持上一个接触段的最后位置）
                    for gap_frame in range(prev_end_frame + 1, start_frame+1):
                        computed_obj_trans[b, gap_frame] = prev_last_position
            
            # 处理最后一个接触段之后的帧
            if contact_segments and current_obj_position is not None:
                last_segment = contact_segments[-1]
                last_end_frame = last_segment['end']
                
                # 最后接触段之后的所有帧保持最后接触位置
                for frame_idx in range(last_end_frame + 1, seq_len):
                    computed_obj_trans[b, frame_idx] = current_obj_position
        
        # 构建FK骨长和方向信息字典
        fk_bone_info = {
            'fk_lhand_bone_length': fk_lhand_bone_length,
            'fk_rhand_bone_length': fk_rhand_bone_length,
            'fk_lhand_direction': fk_lhand_direction,
            'fk_rhand_direction': fk_rhand_direction
        }
        
        return computed_obj_trans, fk_bone_info

    @staticmethod
    def _rot6d_delta(rot6d: torch.Tensor) -> torch.Tensor:
        bs, seq, _ = rot6d.shape
        rotm = rotation_6d_to_matrix(rot6d.reshape(-1, 6)).reshape(bs, seq, 3, 3)
        rel = torch.matmul(rotm[:, 1:].transpose(-1, -2), rotm[:, :-1])
        rel_axis = matrix_to_axis_angle(rel.reshape(-1, 3, 3)).reshape(bs, seq - 1, 3)
        rel_axis = torch.cat([torch.zeros(bs, 1, 3, device=rot6d.device, dtype=rot6d.dtype), rel_axis], dim=1)
        return rel_axis

    @staticmethod
    def _unit_vector(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    @staticmethod
    def _softplus_positive(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x) + 1e-6
    
    def _smooth_gating_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """
        对门控权重进行选择性时间平滑，防止位置突变（仅在推理时使用）
        
        平滑策略：
        - 训练时：不应用平滑，保持真实梯度信号
        - 推理时：选择性平滑
          * FK分支突增 → 需要平滑（防止位置跳跃）
          * IMU分支突增 → 不需要平滑（不会突变）
          * FK分支突减 → 不应平滑（物体需要能快速停止）
        
        Args:
            weights: [batch_size, seq_len, 3] 原始门控权重
                    dim=-1: [左手FK, 右手FK, IMU分支]
            
        Returns:
            smoothed_weights: [batch_size, seq_len, 3] 平滑后的门控权重
        """
        # 训练时不应用平滑，直接返回原始权重
        if self.training:
            return weights
            
        # 推理时才考虑平滑
        if not self.gating_smoothing_enabled:
            return weights
        
        bs, seq, num_branches = weights.shape
        device = weights.device
        
        # 分支索引定义
        # 0: 左手FK, 1: 右手FK, 2: IMU分支
        LHAND_FK, RHAND_FK, IMU_BRANCH = 0, 1, 2
        
        # 初始化平滑后的权重
        smoothed_weights = torch.zeros_like(weights)
        smoothed_weights[:, 0, :] = weights[:, 0, :]  # 第一帧保持不变
        
        # 逐帧应用选择性平滑
        for t in range(1, seq):
            current_weights = weights[:, t, :]  # [bs, 3]
            prev_smoothed = smoothed_weights[:, t-1, :]  # [bs, 3]
            
            # 确定主导分支（权重最大的分支）
            prev_dominant = torch.argmax(prev_smoothed, dim=-1)  # [bs]
            curr_dominant = torch.argmax(current_weights, dim=-1)  # [bs]
            
            # 初始化当前帧权重（默认不平滑）
            frame_weights = current_weights.clone()
            
            # 检查是否有任何样本需要平滑（优化：避免不必要的循环）
            transition_mask = prev_dominant != curr_dominant  # [bs] 布尔掩码
            
            if transition_mask.any():  # 只有当至少有一个样本发生转换时才处理
                # 逐批次处理（因为每个样本的主导分支转换可能不同）
                for b in range(bs):
                    if not transition_mask[b]:  # 该样本无转换，跳过
                        continue
                        
                    prev_dom = prev_dominant[b].item()
                    curr_dom = curr_dominant[b].item()
                    
                    # 判断是否需要平滑的转换类型
                    need_smoothing = False
                    
                    # 需要平滑的转换：
                    # IMU → FK分支 (2→0, 2→1)
                    # FK分支 → FK分支 (0→1, 1→0)
                    if (prev_dom == IMU_BRANCH and curr_dom in [LHAND_FK, RHAND_FK]) or \
                       (prev_dom in [LHAND_FK, RHAND_FK] and curr_dom in [LHAND_FK, RHAND_FK]):
                        need_smoothing = True
                    
                    # 不需要平滑的转换：
                    # FK分支 → IMU (0→2, 1→2) - 允许快速转到IMU，物体能快速停止
                    
                    if need_smoothing:
                        # 应用指数移动平均平滑
                        frame_weights[b, :] = (
                            self.gating_smoothing_alpha * prev_smoothed[b, :] + 
                            (1.0 - self.gating_smoothing_alpha) * current_weights[b, :]
                        )
                        
                        # 限制相邻帧的最大变化量
                        if self.gating_max_change > 0:
                            weight_change = frame_weights[b, :] - prev_smoothed[b, :]
                            weight_change_norm = torch.norm(weight_change)
                            
                            # 如果变化量超过阈值，则进行裁剪
                            if weight_change_norm > self.gating_max_change:
                                weight_change = weight_change * (self.gating_max_change / (weight_change_norm + 1e-8))
                                frame_weights[b, :] = prev_smoothed[b, :] + weight_change
                        
                        # 重新归一化以确保权重和为1
                        frame_weights[b, :] = torch.nn.functional.softmax(
                            torch.log(frame_weights[b, :] + 1e-8) * self.gating_temperature, 
                            dim=-1
                        )
            
            smoothed_weights[:, t, :] = frame_weights
        
        return smoothed_weights

    def _build_fk_inputs(self, obj_rot6d, hand_pos, hand_contact_scalar, obj_imu9, hand_imu9, obj_vel3, obj_rot_delta3):
        return torch.cat([obj_rot6d, hand_pos, hand_contact_scalar, obj_imu9, hand_imu9, obj_vel3, obj_rot_delta3], dim=2)

    def _build_imu_inputs(self, obj_imu9, obj_vel3, obj_rot_delta3, obj_rot6d):
        return torch.cat([obj_imu9, obj_vel3, obj_rot_delta3, obj_rot6d], dim=2)

    def _build_gating_inputs(self, contact_prob3, obj_vel3, obj_imu_acc3):
        return torch.cat([contact_prob3, obj_vel3, obj_imu_acc3], dim=2)

    def _compute_init_dir_len(self, hand_pos_0, obj_rotm_0, obj_pos_0):
        vec_world = obj_pos_0 - hand_pos_0  # [bs, 3]
        lb0 = vec_world.norm(dim=-1, keepdim=True)
        unit_world = self._unit_vector(vec_world)
        obj_Rt = obj_rotm_0.transpose(-1, -2)
        oe0 = torch.bmm(obj_Rt, unit_world.unsqueeze(-1)).squeeze(-1)
        return oe0, lb0

    def forward(
        self,
        hands_pos_feat: torch.Tensor,
        pred_hand_contact_prob: torch.Tensor,
        obj_rot: torch.Tensor,
        obj_trans: torch.Tensor,
        compressed_initial_state: torch.Tensor,
        obj_imu: torch.Tensor = None,
        human_imu: torch.Tensor = None,
        obj_vel_input: torch.Tensor = None,
        gt_hands_pos: torch.Tensor = None,
        use_gt_hands_for_obj: bool = False
    ):
        bs, seq, _ = hands_pos_feat.shape
        device = hands_pos_feat.device

        # 选择手部位置来源
        if use_gt_hands_for_obj and gt_hands_pos is not None:
            hand_positions = gt_hands_pos  # [bs, seq, 2, 3]
        else:
            hand_positions = hands_pos_feat.reshape(bs, seq, 2, 3)
        lhand_position = hand_positions[:, :, 0, :]
        rhand_position = hand_positions[:, :, 1, :]

        # 旋转矩阵与差分
        obj_rot_delta = self._rot6d_delta(obj_rot)
        obj_rotm = rotation_6d_to_matrix(obj_rot.reshape(-1, 6)).reshape(bs, seq, 3, 3)

        # 物体IMU拆分
        if obj_imu is None:
            obj_imu = torch.zeros(bs, seq, 9, device=device, dtype=hands_pos_feat.dtype)
        else:
            obj_imu = obj_imu.reshape(bs, seq, -1)
        obj_imu_acc = obj_imu[:, :, :3]

        # 人体IMU -> 取左右手
        if human_imu is None:
            human_imu_reshaped = torch.zeros(bs, seq, self.num_human_imus, self.imu_dim, device=device, dtype=hands_pos_feat.dtype)
        else:
            human_imu_reshaped = human_imu.reshape(bs, seq, self.num_human_imus, self.imu_dim)
        from configs.global_config import IMU_JOINT_NAMES
        l_idx = IMU_JOINT_NAMES.index('left_hand')
        r_idx = IMU_JOINT_NAMES.index('right_hand')
        lhand_imu9 = human_imu_reshaped[:, :, l_idx, :]
        rhand_imu9 = human_imu_reshaped[:, :, r_idx, :]

        # 物体速度
        if obj_vel_input is None:
            obj_vel_input = torch.zeros(bs, seq, 3, device=device, dtype=hands_pos_feat.dtype)
 
        # 手接触标量
        pL = pred_hand_contact_prob[:, :, 0:1]
        pR = pred_hand_contact_prob[:, :, 1:2]

        # FK分支输入
        fk_l_input = self._build_fk_inputs(obj_rot, lhand_position, pL, obj_imu, lhand_imu9, obj_vel_input, obj_rot_delta)
        fk_r_input = self._build_fk_inputs(obj_rot, rhand_position, pR, obj_imu, rhand_imu9, obj_vel_input, obj_rot_delta)

        # 初始状态
        l_fk_h0 = compute_rnn_initial_state(self.proj_lhand_fk_state, self.lhand_fk_head, compressed_initial_state, bs)
        r_fk_h0 = compute_rnn_initial_state(self.proj_rhand_fk_state, self.rhand_fk_head, compressed_initial_state, bs)
        # imu_h0 = compute_rnn_initial_state(self.proj_imu_state, self.imu_head, compressed_initial_state, bs)
        gate_h0 = compute_rnn_initial_state(self.proj_gate_state, self.gating_head, compressed_initial_state, bs)

        # 左右手方向+长度
        l_fk_out, _ = self.lhand_fk_head(fk_l_input, l_fk_h0)
        r_fk_out, _ = self.rhand_fk_head(fk_r_input, r_fk_h0)
        l_dir = self._unit_vector(l_fk_out[:, :, :3])
        r_dir = self._unit_vector(r_fk_out[:, :, :3])
        l_len = self._softplus_positive(l_fk_out[:, :, 3])
        r_len = self._softplus_positive(r_fk_out[:, :, 3])

        # FK到世界
        obj_rotm_flat = obj_rotm.reshape(bs * seq, 3, 3)
        l_dir_world = torch.bmm(obj_rotm_flat, l_dir.reshape(bs * seq, 3, 1)).reshape(bs, seq, 3)
        r_dir_world = torch.bmm(obj_rotm_flat, r_dir.reshape(bs * seq, 3, 1)).reshape(bs, seq, 3)
        l_pos_fk = lhand_position + l_dir_world * l_len.unsqueeze(-1)
        r_pos_fk = rhand_position + r_dir_world * r_len.unsqueeze(-1)

        # IMU分支：使用物体IMU加速度积分得到速度
        # # 对加速度进行时间积分得到速度
        # dt = 1.0 / FRAME_RATE
        # vel_from_acc_integration = torch.zeros_like(obj_imu_acc)
        # # 第一帧速度初始化为0
        # vel_from_acc_integration[:, 0, :] = 0.0
        # # 后续帧通过积分计算：v(t) = v(t-1) + a(t) * dt
        # for t in range(1, seq):
        #     vel_from_acc_integration[:, t, :] = vel_from_acc_integration[:, t-1, :] + obj_imu_acc[:, t, :] * dt
            
        # vel_used = vel_from_acc_integration  # [bs, seq, 3]
        vel_used = obj_vel_input  # [bs, seq, 3]

        # 门控
        gating_input = self._build_gating_inputs(pred_hand_contact_prob, obj_vel_input, obj_imu_acc)
        gate_logits, _ = self.gating_head(gating_input, gate_h0)
        prior_im = 1.0 - torch.maximum(pL.squeeze(-1), pR.squeeze(-1))
        prior = torch.stack([pL.squeeze(-1), pR.squeeze(-1), prior_im], dim=-1)
        eps = 1e-6
        gate_logits = gate_logits + self.gating_prior_beta * torch.log(prior + eps)
        weights_raw = torch.nn.functional.softmax(gate_logits / self.gating_temperature, dim=-1)
        
        # 应用权重平滑
        weights = self._smooth_gating_weights(weights_raw)

        # 逐帧融合：IMU速度积分引入上一帧锚点
        dt = 1.0 / FRAME_RATE
        fused_pos = torch.zeros(bs, seq, 3, device=device, dtype=hands_pos_feat.dtype)
        for t in range(seq):
            prev_pos = fused_pos[:, t - 1, :] if t > 0 else obj_trans[:, 0, :]
            pos_imu_integrated = prev_pos + vel_used[:, t, :] * dt
            fused_pos[:, t, :] = (
                weights[:, t, 0:1] * l_pos_fk[:, t, :] +
                weights[:, t, 1:2] * r_pos_fk[:, t, :] +
                weights[:, t, 2:3] * pos_imu_integrated
            )

        # 差分
        vel_from_pos = torch.zeros_like(fused_pos)
        acc_from_pos = torch.zeros_like(fused_pos)
        if seq > 1:
            vel_from_pos[:, 1:] = (fused_pos[:, 1:] - fused_pos[:, :-1]) * FRAME_RATE
        if seq > 2:
            acc_from_pos[:, 2:] = (fused_pos[:, 2:] - 2 * fused_pos[:, 1:-1] + fused_pos[:, :-2]) * (FRAME_RATE ** 2) / acc_scale

        # 首帧初始化参考
        obj_pos_0 = obj_trans[:, 0, :]
        obj_R_0 = obj_rotm[:, 0, :, :]
        l_hand_0 = lhand_position[:, 0, :]
        r_hand_0 = rhand_position[:, 0, :]
        l_oe0, l_lb0 = self._compute_init_dir_len(l_hand_0, obj_R_0, obj_pos_0)
        r_oe0, r_lb0 = self._compute_init_dir_len(r_hand_0, obj_R_0, obj_pos_0)

        # weights_watch = weights.detach().cpu().numpy()
        # pred_hand_contact_prob_watch = pred_hand_contact_prob.detach().cpu().numpy()
        # vel_used_watch = vel_used.detach().cpu().numpy()
        # obj_imu_watch = obj_imu.detach().cpu().numpy()


        return_dict = {
            'pred_obj_trans': fused_pos,
            'gating_weights': weights,
            'gating_weights_raw': weights_raw,  # 原始权重（平滑前）
            'pred_obj_vel_from_posdiff': vel_from_pos,
            'pred_obj_acc_from_posdiff': acc_from_pos,
            'obj_vel_input': obj_vel_input,  # 添加用于可视化的物体速度输入

            'pred_lhand_obj_direction': l_dir,
            'pred_rhand_obj_direction': r_dir,
            'pred_lhand_lb': l_len,
            'pred_rhand_lb': r_len,
            'pred_lhand_obj_trans': l_pos_fk,
            'pred_rhand_obj_trans': r_pos_fk,
            'pred_obj_trans_from_fk': (l_pos_fk + r_pos_fk) / 2.0,

            'init_lhand_oe_ho': l_oe0,
            'init_rhand_oe_ho': r_oe0,
            'init_lhand_lb': l_lb0.squeeze(-1),
            'init_rhand_lb': r_lb0.squeeze(-1),
            
            # 选择性平滑调试信息
            'gating_smoothing_applied': self.training == False and self.gating_smoothing_enabled,  # 是否应用了平滑
        }
        return return_dict
class TransPoseNet(torch.nn.Module):
    """
    适用于EgoIMU项目的TransPose网络架构，支持分阶段训练和模块化加载
    """
    def __init__(self, cfg, pretrained_modules=None, skip_modules=None):
        """
        初始化TransPoseNet
        
        Args:
            cfg: 配置对象
            pretrained_modules: dict, 预训练模块路径
                格式: {"velocity_contact": "path/to/velocity_contact.pt", 
                      "human_pose": "path/to/human_pose.pt", 
                      "object_trans": "path/to/object_trans.pt"}
            skip_modules: list, 跳过初始化的模块名称
                格式: ["velocity_contact", "human_pose", "object_trans"]
        """
        super().__init__()
        
        # 从配置中获取参数
        self.num_human_imus = cfg.num_human_imus if hasattr(cfg, 'num_human_imus') else 6
        self.imu_dim = cfg.imu_dim if hasattr(cfg, 'imu_dim') else 9
        self.joint_dim = cfg.joint_dim if hasattr(cfg, 'joint_dim') else 6
        self.num_joints = cfg.num_joints if hasattr(cfg, 'num_joints') else 22
        
        # 设置设备
        if hasattr(cfg, 'device'):
            self.device = torch.device(cfg.device)
        elif hasattr(cfg, 'gpus') and cfg.gpus:
            self.device = torch.device(f"cuda:{cfg.gpus[0]}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始状态输入维度
        n_motion_start = self.num_joints * self.joint_dim
        n_root_pos_start = 3
        n_obj_rot_start = 6
        n_obj_trans_start = 3
        n_lhand_pos_start = 3  # 左手初始位置
        n_rhand_pos_start = 3  # 右手初始位置
        n_initial_state = n_motion_start + n_root_pos_start + n_obj_rot_start + n_obj_trans_start + n_lhand_pos_start + n_rhand_pos_start
        
        # 压缩初始状态MLP
        self.initial_state_dim = INITIAL_STATE_DIM
        self.initial_state_compressor = torch.nn.Sequential(
            torch.nn.Linear(n_initial_state, self.initial_state_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.initial_state_dim * 2, self.initial_state_dim)
        ).to(self.device)
        
        # 初始化默认值
        if pretrained_modules is None:
            pretrained_modules = {}
        if skip_modules is None:
            skip_modules = []
        
        # 模块化初始化逻辑
        self._initialize_modules(cfg, pretrained_modules, skip_modules)
        
        print(f"TransPoseNet初始化完成:")
        print(f"  - velocity_contact_module: {'从预训练加载' if 'velocity_contact' in pretrained_modules else '新初始化' if 'velocity_contact' not in skip_modules else '跳过初始化'}")
        print(f"  - human_pose_module: {'从预训练加载' if 'human_pose' in pretrained_modules else '新初始化' if 'human_pose' not in skip_modules else '跳过初始化'}")
        print(f"  - object_trans_module: {'从预训练加载' if 'object_trans' in pretrained_modules else '新初始化' if 'object_trans' not in skip_modules else '跳过初始化'}")
    
    def _initialize_modules(self, cfg, pretrained_modules, skip_modules):
        """初始化或加载各个模块"""
        # 1. velocity_contact_module
        if 'velocity_contact' in skip_modules:
            print("跳过velocity_contact模块初始化")
            self.velocity_contact_module = None
        elif 'velocity_contact' in pretrained_modules:
            print(f"正在加载预训练的velocity_contact模块: {pretrained_modules['velocity_contact']}")
            self.velocity_contact_module = self._load_single_module(
                pretrained_modules['velocity_contact'], 'velocity_contact', cfg
            )
        else:
            print("正在初始化新的velocity_contact模块")
            self.velocity_contact_module = VelocityContactModule(cfg, self.device)
        
        # 2. human_pose_module
        if 'human_pose' in skip_modules:
            print("跳过human_pose模块初始化")
            self.human_pose_module = None
        elif 'human_pose' in pretrained_modules:
            print(f"正在加载预训练的human_pose模块: {pretrained_modules['human_pose']}")
            self.human_pose_module = self._load_single_module(
                pretrained_modules['human_pose'], 'human_pose', cfg
            )
        else:
            print("正在初始化新的human_pose模块")
            self.human_pose_module = HumanPoseModule(cfg, self.device)
        
        # 3. object_trans_module
        if 'object_trans' in skip_modules:
            print("跳过object_trans模块初始化")
            self.object_trans_module = None
        elif 'object_trans' in pretrained_modules:
            print(f"正在加载预训练的object_trans模块: {pretrained_modules['object_trans']}")
            self.object_trans_module = self._load_single_module(
                pretrained_modules['object_trans'], 'object_trans', cfg
            )
        else:
            print("正在初始化新的object_trans模块(统一融合版)")
            self.object_trans_module = UnifiedObjectTransModule(cfg, self.device)
    
    def _load_single_module(self, checkpoint_path, module_name, cfg):
        """加载单个模块的预训练权重"""
        try:
            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 验证模块名称
            if 'module_name' in checkpoint and checkpoint['module_name'] != module_name:
                print(f"警告: 检查点中的模块名称 '{checkpoint['module_name']}' 与请求的模块名称 '{module_name}' 不匹配")
            
            # 创建模块实例
            if module_name == 'velocity_contact':
                module = VelocityContactModule(cfg, self.device)
            elif module_name == 'human_pose':
                module = HumanPoseModule(cfg, self.device)
            elif module_name == 'object_trans':
                module = UnifiedObjectTransModule(cfg, self.device)
            else:
                raise ValueError(f"未知的模块名称: {module_name}")
            
            # 加载权重
            if 'module_state_dict' in checkpoint:
                module.load_state_dict(checkpoint['module_state_dict'])
            elif 'state_dict' in checkpoint:
                module.load_state_dict(checkpoint['state_dict'])
            else:
                # 尝试直接加载整个检查点作为state_dict
                module.load_state_dict(checkpoint)
            
            print(f"成功加载{module_name}模块，来自epoch {checkpoint.get('epoch', 'unknown')}")
            return module
            
        except Exception as e:
            print(f"加载{module_name}模块失败: {e}")
            print(f"将初始化新的{module_name}模块")
            # 如果加载失败，则初始化新模块
            if module_name == 'velocity_contact':
                return VelocityContactModule(cfg, self.device)
            elif module_name == 'human_pose':
                return HumanPoseModule(cfg, self.device)
            elif module_name == 'object_trans':
                return UnifiedObjectTransModule(cfg, self.device)
    
    def get_module_state_dict(self, module_name):
        """获取指定模块的状态字典"""
        if module_name == 'velocity_contact' and self.velocity_contact_module is not None:
            return self.velocity_contact_module.state_dict()
        elif module_name == 'human_pose' and self.human_pose_module is not None:
            return self.human_pose_module.state_dict()
        elif module_name == 'object_trans' and self.object_trans_module is not None:
            return self.object_trans_module.state_dict()
        else:
            return None
    
    def save_module(self, module_name, save_path, epoch, additional_info=None):
        """保存单个模块"""
        module_state_dict = self.get_module_state_dict(module_name)
        if module_state_dict is None:
            print(f"模块 {module_name} 不存在，无法保存")
            return False
        
        checkpoint_data = {
            'module_name': module_name,
            'module_state_dict': module_state_dict,
            'epoch': epoch,
        }
        
        if additional_info:
            checkpoint_data.update(additional_info)
        
        try:
            torch.save(checkpoint_data, save_path)
            print(f"成功保存{module_name}模块到: {save_path}")
            return True
        except Exception as e:
            print(f"保存{module_name}模块失败: {e}")
            return False
    
    def freeze_module(self, module_name):
        """冻结指定模块的参数"""
        if module_name == "velocity_contact":
            if self.velocity_contact_module:
                for param in self.velocity_contact_module.parameters():
                    param.requires_grad = False
        elif module_name == "human_pose":
            if self.human_pose_module:
                for param in self.human_pose_module.parameters():
                    param.requires_grad = False
        elif module_name == "object_trans":
            if self.object_trans_module:
                for param in self.object_trans_module.parameters():
                    param.requires_grad = False
        print(f"已冻结模块: {module_name}")
    
    def unfreeze_module(self, module_name):
        """解冻指定模块的参数"""
        if module_name == "velocity_contact":
            if self.velocity_contact_module:
                for param in self.velocity_contact_module.parameters():
                    param.requires_grad = True
        elif module_name == "human_pose":
            if self.human_pose_module:
                for param in self.human_pose_module.parameters():
                    param.requires_grad = True
        elif module_name == "object_trans":
            if self.object_trans_module:
                for param in self.object_trans_module.parameters():
                    param.requires_grad = True
        print(f"已解冻模块: {module_name}")
    
    def configure_training_modules(self, active_modules, frozen_modules=None):
        """配置训练模块：激活指定模块，冻结其他模块"""
        all_modules = ["velocity_contact", "human_pose", "object_trans"]
        
        # 首先冻结所有模块
        for module_name in all_modules:
            self.freeze_module(module_name)
        
        # 然后激活指定模块
        for module_name in active_modules:
            if module_name in all_modules:
                self.unfreeze_module(module_name)
        
        # 处理额外的冻结模块
        if frozen_modules:
            for module_name in frozen_modules:
                if module_name in all_modules:
                    self.freeze_module(module_name)
    
    def format_input(self, data_dict):
        """格式化输入数据"""
        human_imu = data_dict["human_imu"]
        obj_imu = data_dict.get("obj_imu", None)
        motion = data_dict["motion"]
        root_pos = data_dict["root_pos"]
        obj_rot = data_dict.get("obj_rot", None)
        obj_trans = data_dict.get("obj_trans", None)
        hands_pos = data_dict.get("gt_hands_pos", None)
        
        batch_size, seq_len = human_imu.shape[:2]
        
        # 处理IMU数据
        human_imu_flat_seq = human_imu.reshape(batch_size, seq_len, -1)
        obj_imu_flat_seq = obj_imu.reshape(batch_size, seq_len, -1)
        imu_data = torch.cat([human_imu_flat_seq, obj_imu_flat_seq], dim=2)
        
        # 处理第一帧状态信息
        motion_start = motion[:, 0].reshape(batch_size, -1)
        root_pos_start = root_pos[:, 0]
        obj_rot_start = obj_rot[:, 0]
        obj_trans_start = obj_trans[:, 0]
        lhand_pos_start = hands_pos[:, 0, 0, :]
        rhand_pos_start = hands_pos[:, 0, 1, :]
        
        initial_state_flat = torch.cat([
            motion_start,
            root_pos_start,
            obj_rot_start,
            obj_trans_start,
            lhand_pos_start,
            rhand_pos_start
        ], dim=1)
        
        return imu_data, initial_state_flat
    
    def forward(self, data_dict, use_object_data=None):
        """
        前向传播
        
        Args:
            data_dict: 包含输入数据的字典，可以包含'use_object_data'键
            use_object_data: 是否使用物体数据（用于分阶段训练），如果为None则从data_dict中获取，默认为True
        
        Returns:
            dict: 包含预测结果的字典
        """
        # 从data_dict中获取use_object_data，如果没有则使用参数值，如果参数也为None则默认为True
        if use_object_data is None:
            use_object_data = data_dict.get('use_object_data', True)
        # 格式化输入数据
        imu_data, initial_state_flat = self.format_input(data_dict)
        batch_size, seq_len, _ = imu_data.shape
        device = imu_data.device
        
        # 计算压缩后的初始状态
        compressed_initial_state = self.initial_state_compressor(initial_state_flat)
        
        # 分离人体和物体IMU数据
        human_imu_data = imu_data[:, :, :self.num_human_imus*self.imu_dim]
        obj_imu_data = imu_data[:, :, self.num_human_imus*self.imu_dim:]
        
        # 获取物体数据
        obj_rot = data_dict.get("obj_rot", torch.zeros((batch_size, seq_len, 6), device=device))
        obj_trans = data_dict.get("obj_trans", torch.zeros((batch_size, seq_len, 3), device=device))
        
        # 初始化结果字典
        results = {}
        
        # 模块1: 速度估计 + 手部接触预测
        if self.velocity_contact_module is not None:
            velocity_contact_outputs = self.velocity_contact_module(
                imu_data, compressed_initial_state, use_object_data
            )
            results.update(velocity_contact_outputs)
        else:
            # 如果模块不存在，提供默认值
            velocity_contact_outputs = {
                "pred_obj_vel": torch.zeros(batch_size, seq_len, 3, device=device),
                "pred_leaf_vel": torch.zeros(batch_size, seq_len, joint_set.n_leaf, 3, device=device),
                "pred_leaf_vel_flat": torch.zeros(batch_size, seq_len, joint_set.n_leaf * 3, device=device),
                "pred_hand_contact_prob": torch.zeros(batch_size, seq_len, 3, device=device)
            }
            results.update(velocity_contact_outputs)
        
        # 模块2: 人体姿态 + 足部接触 + 人体平移
        if self.human_pose_module is not None:
            human_pose_outputs = self.human_pose_module(
                human_imu_data, 
                velocity_contact_outputs["pred_leaf_vel_flat"], 
                compressed_initial_state
            )
            results.update(human_pose_outputs)
        else:
            # 如果模块不存在，提供默认值
            human_pose_outputs = {
                "pred_leaf_pos": torch.zeros(batch_size, seq_len, joint_set.n_leaf, 3, device=device),
                "pred_full_pos": torch.zeros(batch_size, seq_len, joint_set.n_full, 3, device=device),
                "motion": torch.zeros(batch_size, seq_len, self.num_joints * self.joint_dim, device=device),
                "tran_b2_vel": torch.zeros(batch_size, seq_len, 3, device=device),
                "contact_probability": torch.zeros(batch_size, seq_len, 2, device=device),
                "root_vel": torch.zeros(batch_size, seq_len, 3, device=device),
                "root_pos": torch.zeros(batch_size, seq_len, 3, device=device),
                "pred_hand_pos": torch.zeros(batch_size, seq_len, 2, 3, device=device),
                "hands_pos_feat": torch.zeros(batch_size, seq_len, 6, device=device)
            }
            results.update(human_pose_outputs)
        
        # 模块3: 物体位置估计（只有在使用物体数据时才执行）
        if use_object_data and self.object_trans_module is not None:
            # 选择物体速度输入：HOI用GT，OMOMO用VC预测（若可用）
            use_gt_hands_for_obj = bool(data_dict.get('use_gt_hands_for_obj', False))
            obj_vel_input = None
            if use_gt_hands_for_obj:
                obj_vel_input = data_dict.get('obj_vel', None)
            else:
                obj_vel_input = velocity_contact_outputs.get('pred_obj_vel', None)

            # HOI 阶段使用one-hot GT接触概率，否则用预测
            if use_gt_hands_for_obj:
                lhand_gt = data_dict.get('lhand_contact', None)
                rhand_gt = data_dict.get('rhand_contact', None)
                obj_gt = data_dict.get('obj_contact', None)
                if lhand_gt is not None and rhand_gt is not None and obj_gt is not None:
                    pred_hand_contact_prob_input = torch.stack([
                        lhand_gt.float().to(device),
                        rhand_gt.float().to(device),
                        obj_gt.float().to(device)
                    ], dim=2)
                else:
                    pred_hand_contact_prob_input = velocity_contact_outputs["pred_hand_contact_prob"]
            else:
                pred_hand_contact_prob_input = velocity_contact_outputs["pred_hand_contact_prob"]

            object_trans_outputs = self.object_trans_module(
                hands_pos_feat=human_pose_outputs["hands_pos_feat"],
                pred_hand_contact_prob=pred_hand_contact_prob_input,
                obj_rot=obj_rot,
                obj_trans=obj_trans,
                compressed_initial_state=compressed_initial_state,
                obj_imu=data_dict.get("obj_imu", None),
                human_imu=data_dict.get("human_imu", None),
                obj_vel_input=obj_vel_input,
                gt_hands_pos=data_dict.get("gt_hands_pos", None),
                use_gt_hands_for_obj=use_gt_hands_for_obj
            )
            results.update(object_trans_outputs)
        else:
            # 不使用物体数据或模块不存在时，返回零值
            object_trans_outputs = {
                "pred_lhand_obj_direction": torch.zeros(batch_size, seq_len, 3, device=device),
                "pred_rhand_obj_direction": torch.zeros(batch_size, seq_len, 3, device=device),
                "pred_obj_trans_from_fk": torch.zeros(batch_size, seq_len, 3, device=device),
                "pred_lhand_obj_trans": torch.zeros(batch_size, seq_len, 3, device=device),
                "pred_rhand_obj_trans": torch.zeros(batch_size, seq_len, 3, device=device),
            }
            results.update(object_trans_outputs)
        
        return results 
