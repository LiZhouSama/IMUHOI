import sys
import os
sys.path.append("../")
import torch.nn
from torch.nn.functional import relu
import torch
import numpy as np
from human_body_prior.body_model.body_model import BodyModel
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d, so3_relative_angle, axis_angle_to_matrix
import pytorch3d.transforms as transforms
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
            "pred_hand_contact_prob": pred_hand_contact_prob
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
    模块3: 物体方向向量估计和位置重建
    输入: ^WR_O, 双手估计的接触label, 估计的手的位置, 物体的真实位置加噪声
    输出: ^Ov_{HO} (物体坐标系下的方向向量)
    """
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        self.imu_dim = cfg.imu_dim if hasattr(cfg, 'imu_dim') else 9
        self.num_human_imus = cfg.num_human_imus if hasattr(cfg, 'num_human_imus') else 6
        hidden_dim_multiplier = cfg.hidden_dim_multiplier if hasattr(cfg, 'hidden_dim_multiplier') else 1
        
        # 噪声参数
        self.obj_trans_noise_std = getattr(cfg, 'obj_trans_noise_std', 0.05)  # 物体位置噪声标准差
        
        # 分别为左右手创建^Ov_{HO}估计网络
        # 输入：^WR_O(6D旋转) + 手部位置(3) + 接触概率(3) + 物体位置加噪声(3)
        n_obj_direction_input = 6 + 3 + 3 + 3  # 15维
        
        # 左手^Ov_{HO}估计网络
        self.lhand_obj_direction_net = RNN(n_obj_direction_input, 3, 128 * hidden_dim_multiplier)  # 输出左手^Ov_{HO}，3D
        
        # 右手^Ov_{HO}估计网络  
        self.rhand_obj_direction_net = RNN(n_obj_direction_input, 3, 128 * hidden_dim_multiplier)  # 输出右手^Ov_{HO}，3D
        
        # 初始状态维度
        self.initial_state_dim = 64
        
        # 用于生成初始状态的线性层
        def create_state_projection(hidden_size, num_layers, bidirectional):
            num_directions = 2 if bidirectional else 1
            return torch.nn.Linear(self.initial_state_dim, 2 * num_layers * num_directions * hidden_size).to(device)
        
        self.proj_lhand_obj_direction_state = create_state_projection(self.lhand_obj_direction_net.n_hidden, self.lhand_obj_direction_net.n_rnn_layer, self.lhand_obj_direction_net.num_directions == 2)
        self.proj_rhand_obj_direction_state = create_state_projection(self.rhand_obj_direction_net.n_hidden, self.rhand_obj_direction_net.n_rnn_layer, self.rhand_obj_direction_net.num_directions == 2)
    
    def add_noise_to_obj_trans(self, obj_trans, noise_std):
        """
        为物体位置添加噪声
        
        Args:
            obj_trans: [bs, seq, 3] 物体位置
            noise_std: 噪声标准差
        
        Returns:
            [bs, seq, 3] 带噪声的物体位置
        """
        noise = torch.randn_like(obj_trans) * noise_std
        noisy_obj_trans = obj_trans + noise
        return noisy_obj_trans
    
    def compute_object_position_from_direction(self, hand_position, obj_rot_matrix, obj_direction, bone_length):
        """
        使用FK公式重建物体位置：\hat p_o = 估计的 p_H + ^WR_O · 估计的 ^Ov_{HO} · bone_length
        
        Args:
            hand_position: [bs, seq, 3] 手部位置
            obj_rot_matrix: [bs, seq, 3, 3] 物体旋转矩阵 ^WR_O
            obj_direction: [bs, seq, 3] 物体坐标系下的方向向量 ^Ov_{HO}
            bone_length: [bs, seq] 骨长（由加噪声的真值计算得出）
        
        Returns:
            computed_obj_trans: [bs, seq, 3] 重建的物体位置
        """
        # FK公式：\hat p_o = p_H + ^WR_O * ^Ov_{HO} * bone_length
        # 1. 将方向向量转换到世界坐标系：^WR_O * ^Ov_{HO}
        direction_world = torch.bmm(obj_rot_matrix.view(-1, 3, 3), obj_direction.view(-1, 3, 1)).squeeze(-1)  # [bs*seq, 3]
        direction_world = direction_world.view(hand_position.shape)  # [bs, seq, 3]
        
        # 2. 乘以骨长并加上手部位置
        computed_obj_trans = hand_position + direction_world * bone_length.unsqueeze(-1)  # [bs, seq, 3]
        
        return computed_obj_trans
    
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
        """
        batch_size, seq_len, _ = pred_hand_contact_prob.shape
        device = pred_hand_contact_prob.device
        
        # 初始化输出
        computed_obj_trans = gt_obj_trans.clone()  # 开始时使用真值位置作为基础
        
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
                
                # 转换到物体坐标系：^Ov_{HO} = ^WR_O^T * ^W(p_O - p_H)
                initial_obj_rot_mat_inv = initial_obj_rot_mat.transpose(0, 1)  # [3, 3]
                obj_direction_initial = initial_obj_rot_mat_inv @ hand_to_obj_world  # [3]
                
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
        
        return computed_obj_trans
    
    def forward(self, hands_pos_feat, pred_hand_contact_prob, obj_rot, obj_trans, compressed_initial_state):
        """
        前向传播
        
        Args:
            hands_pos_feat: [bs, seq, 2*3] 手部位置特征
            pred_hand_contact_prob: [bs, seq, 3] 手部接触概率
            obj_rot: [bs, seq, 6] 物体旋转（6D表示） - ^WR_O
            obj_trans: [bs, seq, 3] 物体位置真值
            compressed_initial_state: [bs, initial_state_dim] 压缩的初始状态
            
        Returns:
            dict: 包含物体方向向量和重建位置的字典
        """
        batch_size, seq_len, _ = hands_pos_feat.shape
        device = hands_pos_feat.device
        
        # 将6D旋转转换为旋转矩阵
        obj_rot_matrix = rotation_6d_to_matrix(obj_rot.view(-1, 6)).view(batch_size, seq_len, 3, 3)  # [bs, seq, 3, 3]
        
        # 为物体位置添加噪声
        obj_trans_noisy = self.add_noise_to_obj_trans(obj_trans, self.obj_trans_noise_std)  # [bs, seq, 3]
        
        # 分离手部位置
        hand_positions = hands_pos_feat.reshape(batch_size, seq_len, 2, 3)  # [bs, seq, 2, 3]
        lhand_position = hand_positions[:, :, 0, :]  # [bs, seq, 3]
        rhand_position = hand_positions[:, :, 1, :]  # [bs, seq, 3]
        
        # 计算加噪声的真值骨长：|加噪声的真值p_o - 真值p_H|
        lhand_bone_length_noisy = torch.norm(obj_trans_noisy - lhand_position, dim=-1)  # [bs, seq]
        rhand_bone_length_noisy = torch.norm(obj_trans_noisy - rhand_position, dim=-1)  # [bs, seq]
        
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
        
        # 准备RNN输入：^WR_O(6D旋转) + 手部位置(3) + 接触概率(3) + 物体位置加噪声(3)
        
        # 左手^Ov_{HO}估计
        lhand_obj_direction_initial_state = get_initial_rnn_state(self.proj_lhand_obj_direction_state, self.lhand_obj_direction_net)
        lhand_obj_direction_input = torch.cat([
            obj_rot,                    # [bs, seq, 6] - ^WR_O (6D旋转)
            lhand_position,             # [bs, seq, 3] - 左手位置
            pred_hand_contact_prob,     # [bs, seq, 3] - 接触概率
            obj_trans_noisy             # [bs, seq, 3] - 物体位置加噪声
        ], dim=2)  # [bs, seq, 15]
        
        pred_lhand_obj_direction, _ = self.lhand_obj_direction_net(lhand_obj_direction_input, lhand_obj_direction_initial_state)  # [bs, seq, 3]
        
        # 右手^Ov_{HO}估计
        rhand_obj_direction_initial_state = get_initial_rnn_state(self.proj_rhand_obj_direction_state, self.rhand_obj_direction_net)
        rhand_obj_direction_input = torch.cat([
            obj_rot,                    # [bs, seq, 6] - ^WR_O (6D旋转)
            rhand_position,             # [bs, seq, 3] - 右手位置
            pred_hand_contact_prob,     # [bs, seq, 3] - 接触概率
            obj_trans_noisy             # [bs, seq, 3] - 物体位置加噪声
        ], dim=2)  # [bs, seq, 15]
        
        pred_rhand_obj_direction, _ = self.rhand_obj_direction_net(rhand_obj_direction_input, rhand_obj_direction_initial_state)  # [bs, seq, 3]
        
        # 使用FK公式重建物体位置
        # 左手重建
        lhand_computed_obj_trans = self.compute_object_position_from_direction(
            lhand_position, obj_rot_matrix, pred_lhand_obj_direction, lhand_bone_length_noisy
        )  # [bs, seq, 3]
        
        # 右手重建
        rhand_computed_obj_trans = self.compute_object_position_from_direction(
            rhand_position, obj_rot_matrix, pred_rhand_obj_direction, rhand_bone_length_noisy
        )  # [bs, seq, 3]
        
        # 对左右手重建的物体位置取平均
        computed_obj_trans_avg = (lhand_computed_obj_trans + rhand_computed_obj_trans) / 2  # [bs, seq, 3]
        
        # 新增：基于接触的FK物体位置预测
        computed_obj_trans_contact_fk = self.predict_object_position_from_contact(
            pred_hand_contact_prob, hand_positions, obj_rot_matrix, obj_trans
        )  # [bs, seq, 3]
        
        return {
            "pred_lhand_obj_direction": pred_lhand_obj_direction,       # [bs, seq, 3] - 左手^Ov_{HO}
            "pred_rhand_obj_direction": pred_rhand_obj_direction,       # [bs, seq, 3] - 右手^Ov_{HO}
            "pred_obj_trans_from_fk": computed_obj_trans_avg,           # [bs, seq, 3] - FK重建的物体位置
            "pred_lhand_obj_trans": lhand_computed_obj_trans,           # [bs, seq, 3] - 左手单独重建
            "pred_rhand_obj_trans": rhand_computed_obj_trans,           # [bs, seq, 3] - 右手单独重建
            "pred_obj_trans_from_contact": computed_obj_trans_contact_fk,  # [bs, seq, 3] - 基于接触的FK物体位置
        }


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
        n_initial_state = n_motion_start + n_root_pos_start + n_obj_rot_start + n_obj_trans_start
        
        # 压缩初始状态MLP
        self.initial_state_dim = 64
        self.initial_state_compressor = torch.nn.Sequential(
            torch.nn.Linear(n_initial_state, self.initial_state_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.initial_state_dim * 2, self.initial_state_dim)
        ).to(self.device)
        
        # 传递噪声参数给配置
        if hasattr(cfg, 'noise_params'):
            cfg.init_rotation_noise_std = cfg.noise_params.get('init_rotation_noise_std', 0.1)
            cfg.bone_length_noise_std = cfg.noise_params.get('bone_length_noise_std', 0.05)
        
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
            print("正在初始化新的object_trans模块")
            self.object_trans_module = ObjectTransModule(cfg, self.device)
    
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
                module = ObjectTransModule(cfg, self.device)
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
                return ObjectTransModule(cfg, self.device)
    
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
        
        # 模块3: 物体方向向量估计（只有在使用物体数据时才执行）
        if use_object_data and self.object_trans_module is not None:
            object_trans_outputs = self.object_trans_module(
                human_pose_outputs["hands_pos_feat"],
                velocity_contact_outputs["pred_hand_contact_prob"],
                obj_rot,
                obj_trans,
                compressed_initial_state
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
