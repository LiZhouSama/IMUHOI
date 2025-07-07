import sys
import os
sys.path.append("../")
import torch.nn
from torch.nn.functional import relu
import torch
import numpy as np
from human_body_prior.body_model.body_model import BodyModel
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d, so3_relative_angle
from utils.utils import global2local

from configs.global_config import FRAME_RATE, IMU_JOINTS_POS, IMU_JOINTS_ROT, IMU_JOINT_NAMES, joint_set



# 定义辅助函数
def lerp(val, low, high):
    """线性插值"""
    return low + (high - low) * val

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
        n_imu = n_human_imu + n_obj_imu
        
        # 手部接触网络相关索引
        self.lhand_imu_idx = IMU_JOINT_NAMES.index('left_hand')
        self.rhand_imu_idx = IMU_JOINT_NAMES.index('right_hand')
        
        # 速度估计网络
        n_velocity_output = 3 + joint_set.n_leaf * 3  # 物体速度 + 叶子节点速度
        self.velocity_net = RNN(n_imu, n_velocity_output, 256 * hidden_dim_multiplier)
        
        # 手部接触预测网络
        n_hand_contact_input = 3 * self.imu_dim + 2 * 3 + 3  # 双手IMU+物体IMU + 双手速度 + 物体速度
        self.hand_contact_net = RNN(n_hand_contact_input, 3, 128 * hidden_dim_multiplier)
        
        # 初始状态维度
        self.initial_state_dim = 64
        
        # 用于生成初始状态的线性层
        def create_state_projection(hidden_size, num_layers, bidirectional):
            num_directions = 2 if bidirectional else 1
            return torch.nn.Linear(self.initial_state_dim, 2 * num_layers * num_directions * hidden_size).to(device)
        
        self.proj_velocity_state = create_state_projection(self.velocity_net.n_hidden, self.velocity_net.n_rnn_layer, self.velocity_net.num_directions == 2)
        self.proj_hand_contact_state = create_state_projection(self.hand_contact_net.n_hidden, self.hand_contact_net.n_rnn_layer, self.hand_contact_net.num_directions == 2)
    
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
        
        # 生成初始状态
        def get_initial_rnn_state(proj_layer, rnn_module):
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
        
        # 速度估计
        velocity_initial_state = get_initial_rnn_state(self.proj_velocity_state, self.velocity_net)
        pred_velocity_output, _ = self.velocity_net(imu_data, velocity_initial_state)
        
        # 分离预测的速度
        pred_obj_vel = pred_velocity_output[:, :, :3]
        pred_leaf_vel = pred_velocity_output[:, :, 3:].reshape(batch_size, seq_len, joint_set.n_leaf, 3)
        pred_leaf_vel_flat = pred_leaf_vel.reshape(batch_size, seq_len, -1)
        
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
        hand_contact_initial_state = get_initial_rnn_state(self.proj_hand_contact_state, self.hand_contact_net)
        
        # 手部接触预测
        pred_hand_contact_prob_logits, _ = self.hand_contact_net(hand_contact_input, hand_contact_initial_state)
        pred_hand_contact_prob = torch.sigmoid(pred_hand_contact_prob_logits)
        
        return {
            "pred_obj_vel": pred_obj_vel,
            "pred_leaf_vel": pred_leaf_vel,
            "pred_leaf_vel_flat": pred_leaf_vel_flat,
            "pred_hand_contact_prob": pred_hand_contact_prob,
            "lhand_vel": lhand_vel,
            "rhand_vel": rhand_vel
        }


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
        self.initial_state_dim = 64
        
        # 用于生成初始状态的线性层
        def create_state_projection(hidden_size, num_layers, bidirectional):
            num_directions = 2 if bidirectional else 1
            return torch.nn.Linear(self.initial_state_dim, 2 * num_layers * num_directions * hidden_size).to(device)

        self.proj_s1_state = create_state_projection(self.pose_s1.n_hidden, self.pose_s1.n_rnn_layer, self.pose_s1.num_directions == 2)
        self.proj_s2_state = create_state_projection(self.pose_s2.n_hidden, self.pose_s2.n_rnn_layer, self.pose_s2.num_directions == 2)
        self.proj_s3_state = create_state_projection(self.pose_s3.n_hidden, self.pose_s3.n_rnn_layer, self.pose_s3.num_directions == 2)
        self.proj_contact_state = create_state_projection(self.contact_prob_net.n_hidden, self.contact_prob_net.n_rnn_layer, self.contact_prob_net.num_directions == 2)
        self.proj_trans_b2_state = create_state_projection(self.trans_b2.n_hidden, self.trans_b2.n_rnn_layer, self.trans_b2.num_directions == 2)
        
        # SMPL Body Model for FK
        self.body_model = BodyModel(bm_fname=cfg.bm_path, num_betas=16).to(device)
        for p in self.body_model.parameters():
            p.requires_grad = False
        
        # 注册buffer
        self.register_buffer('parents_tensor', self.body_model.kintree_table[0].long())
        self.register_buffer('imu_joints_pos', torch.tensor(IMU_JOINTS_POS, dtype=torch.long))
        self.register_buffer('imu_joints_rot', torch.tensor(IMU_JOINTS_ROT, dtype=torch.long))
        self.register_buffer('gravity_velocity', torch.tensor([0.0, 0.0, -0.018], dtype=torch.float32))
        
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
        
        # 生成初始状态
        def get_initial_rnn_state(proj_layer, rnn_module):
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

        # 级联姿态估计
        s1_initial_state = get_initial_rnn_state(self.proj_s1_state, self.pose_s1)
        s2_initial_state = get_initial_rnn_state(self.proj_s2_state, self.pose_s2)
        s3_initial_state = get_initial_rnn_state(self.proj_s3_state, self.pose_s3)
        contact_initial_state = get_initial_rnn_state(self.proj_contact_state, self.contact_prob_net)
        trans_b2_initial_state = get_initial_rnn_state(self.proj_trans_b2_state, self.trans_b2)

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


class ObjectTransModule(torch.nn.Module):
    """
    模块3: 物体平移估计
    输入: 手部位置, 接触概率, 物体速度
    输出: 物体平移
    """
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        hidden_dim_multiplier = cfg.hidden_dim_multiplier if hasattr(cfg, 'hidden_dim_multiplier') else 1
        
        # 物体平移预测网络
        n_obj_trans_input = (2 * 3) + 3 + 3  # 手部位置 + 接触概率 + 物体速度
        self.obj_trans_net = RNN(n_obj_trans_input, 3, 128 * hidden_dim_multiplier)
        
        # 初始状态维度
        self.initial_state_dim = 64
        
        # 用于生成初始状态的线性层
        def create_state_projection(hidden_size, num_layers, bidirectional):
            num_directions = 2 if bidirectional else 1
            return torch.nn.Linear(self.initial_state_dim, 2 * num_layers * num_directions * hidden_size).to(device)
        
        self.proj_obj_trans_state = create_state_projection(self.obj_trans_net.n_hidden, self.obj_trans_net.n_rnn_layer, self.obj_trans_net.num_directions == 2)
    
    def forward(self, hands_pos_feat, pred_hand_contact_prob, pred_obj_vel, compressed_initial_state):
        """
        前向传播
        
        Args:
            hands_pos_feat: [bs, seq, 2*3] 手部位置特征
            pred_hand_contact_prob: [bs, seq, 3] 手部接触概率
            pred_obj_vel: [bs, seq, 3] 物体速度
            compressed_initial_state: [bs, initial_state_dim] 压缩的初始状态
            
        Returns:
            dict: 包含物体平移预测的字典
        """
        batch_size, seq_len, _ = hands_pos_feat.shape
        device = hands_pos_feat.device
        
        # 生成初始状态
        def get_initial_rnn_state(proj_layer, rnn_module):
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
        
        # 物体平移预测
        obj_trans_initial_state = get_initial_rnn_state(self.proj_obj_trans_state, self.obj_trans_net)
        obj_trans_input = torch.cat([hands_pos_feat, pred_hand_contact_prob, pred_obj_vel], dim=2)
        pred_obj_trans, _ = self.obj_trans_net(obj_trans_input, obj_trans_initial_state)
        
        return {
            "pred_obj_trans": pred_obj_trans
        }


class TransPoseNet(torch.nn.Module):
    """
    适用于EgoIMU项目的TransPose网络架构，支持分阶段训练
    """
    def __init__(self, cfg):
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
        n_initial_state = n_motion_start + n_root_pos_start + n_obj_rot_start + n_obj_trans_start
        
        # 压缩初始状态MLP
        self.initial_state_dim = 64
        self.initial_state_compressor = torch.nn.Sequential(
            torch.nn.Linear(n_initial_state, self.initial_state_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.initial_state_dim * 2, self.initial_state_dim)
        ).to(self.device)
        
        # 三个子网络模块
        self.velocity_contact_module = VelocityContactModule(cfg, self.device)
        self.human_pose_module = HumanPoseModule(cfg, self.device)
        self.object_trans_module = ObjectTransModule(cfg, self.device)
        
    def freeze_module(self, module_name):
        """冻结指定模块的参数"""
        if module_name == "velocity_contact":
            for param in self.velocity_contact_module.parameters():
                param.requires_grad = False
        elif module_name == "human_pose":
            for param in self.human_pose_module.parameters():
                param.requires_grad = False
        elif module_name == "object_trans":
            for param in self.object_trans_module.parameters():
                param.requires_grad = False
        print(f"已冻结模块: {module_name}")
    
    def unfreeze_module(self, module_name):
        """解冻指定模块的参数"""
        if module_name == "velocity_contact":
            for param in self.velocity_contact_module.parameters():
                param.requires_grad = True
        elif module_name == "human_pose":
            for param in self.human_pose_module.parameters():
                param.requires_grad = True
        elif module_name == "object_trans":
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
        
        initial_state_flat = torch.cat([
            motion_start,
            root_pos_start,
            obj_rot_start,
            obj_trans_start
        ], dim=1)
        
        return imu_data, initial_state_flat
    
    def forward(self, data_dict, use_object_data=True):
        """
        前向传播
        
        Args:
            data_dict: 包含输入数据的字典
            use_object_data: 是否使用物体数据（用于分阶段训练）
        
        Returns:
            dict: 包含预测结果的字典
        """
        # 格式化输入数据
        imu_data, initial_state_flat = self.format_input(data_dict)
        batch_size, seq_len, _ = imu_data.shape
        device = imu_data.device
        
        # 计算压缩后的初始状态
        compressed_initial_state = self.initial_state_compressor(initial_state_flat)
        
        # 分离人体IMU数据
        human_imu_data = imu_data[:, :, :self.num_human_imus*self.imu_dim]
        
        # 模块1: 速度估计 + 手部接触预测
        velocity_contact_outputs = self.velocity_contact_module(
            imu_data, compressed_initial_state, use_object_data
        )
        
        # 模块2: 人体姿态 + 足部接触 + 人体平移
        human_pose_outputs = self.human_pose_module(
            human_imu_data, 
            velocity_contact_outputs["pred_leaf_vel_flat"], 
            compressed_initial_state
        )
        
        # 模块3: 物体平移估计（只有在使用物体数据时才执行）
        if use_object_data:
            object_trans_outputs = self.object_trans_module(
                human_pose_outputs["hands_pos_feat"],
                velocity_contact_outputs["pred_hand_contact_prob"],
                velocity_contact_outputs["pred_obj_vel"],
                compressed_initial_state
            )
        else:
            # 不使用物体数据时，返回零值
            object_trans_outputs = {
                "pred_obj_trans": torch.zeros(batch_size, seq_len, 3, device=device)
            }
        
        # 合并所有输出
        results = {
            **velocity_contact_outputs,
            **human_pose_outputs,
            **object_trans_outputs
        }
        
        return results 
