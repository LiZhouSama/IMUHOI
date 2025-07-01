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


class TransPoseNet(torch.nn.Module):
    """
    适用于EgoIMU项目的TransPose网络架构。
    用于基于IMU数据预测人体姿态、物体变换和人体平移。
    将人体IMU和物体IMU整合到同一个级联架构中。
    新增：将序列第一帧的归一化状态作为额外输入。
    """
    def __init__(self, cfg):
        """
        初始化TransPose网络

        Args:
            cfg: 配置对象，包含网络参数
                 需要包含: num_human_imus, imu_dim, hidden_dim_multiplier,
                          smpl_model_path, left_foot_idx, right_foot_idx,
                          gravity_velocity, vel_scale, prob_threshold_min,
                          prob_threshold_max, floor_y
        """
        super().__init__()
        
        # 从配置中获取参数
        self.num_human_imus = cfg.num_human_imus if hasattr(cfg, 'num_human_imus') else 6
        self.imu_dim = cfg.imu_dim if hasattr(cfg, 'imu_dim') else 9  # 每个IMU有9维数据(加速度3D + 旋转6D) -> 假设根旋转也是6D
        self.joint_dim = cfg.joint_dim if hasattr(cfg, 'joint_dim') else 6  # 使用6D旋转表示
        self.num_joints = cfg.num_joints if hasattr(cfg, 'num_joints') else 22 # SMPLH有22个身体关节
        hidden_dim_multiplier = cfg.hidden_dim_multiplier if hasattr(cfg, 'hidden_dim_multiplier') else 1
        self.device = cfg.device if hasattr(cfg, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # 计算IMU输入维度
        n_human_imu = self.num_human_imus * self.imu_dim
        n_obj_imu = 1 * self.imu_dim
        n_imu = n_human_imu + n_obj_imu
        
        # 初始状态输入维度 (来自第一帧的归一化数据)
        n_motion_start = self.num_joints * self.joint_dim # 人体动作
        n_root_pos_start = 3                             # 人体根位置
        n_obj_rot_start = 6                              # 物体旋转 (6D)
        n_obj_trans_start = 3                            # 物体平移
        n_initial_state = n_motion_start + n_root_pos_start + n_obj_rot_start + n_obj_trans_start

        # 压缩初始状态 MLP
        self.initial_state_dim = 64 # Compressed dimension
        self.initial_state_compressor = torch.nn.Sequential(
            torch.nn.Linear(n_initial_state, self.initial_state_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.initial_state_dim * 2, self.initial_state_dim)
        ).to(self.device)


        # --- 手部接触网络相关索引 ---
        self.lhand_imu_idx = IMU_JOINT_NAMES.index('left_hand') # 基于 global_config.py IMU_JOINT_NAMES
        self.rhand_imu_idx = IMU_JOINT_NAMES.index('right_hand') # 基于 global_config.py IMU_JOINT_NAMES
        # self.num_human_imus 维度中的索引

        # joint_set.full 中的手腕关节索引 (原始SMPL关节20和21)
        # joint_set.full = [1, ..., 21], 长度为21
        # 原始关节20 -> joint_set.full.index(20) = 19
        # 原始关节21 -> joint_set.full.index(21) = 20
        self.lhand_jnt_idx_in_full = joint_set.full.index(20)
        self.rhand_jnt_idx_in_full = joint_set.full.index(21)

        # joint_set.reduced 中的手腕关节索引 (原始SMPL关节20和21)
        # joint_set.reduced = [1, 2, ..., 19, 20, 21] (实际值)
        # 在这个 reduced 列表中，原始关节20是列表的第19个元素 (0-indexed)
        # 在这个 reduced 列表中，原始关节21是列表的第20个元素 (0-indexed)
        # self.lhand_jnt_idx_in_reduced = joint_set.reduced.index(20)
        # self.rhand_jnt_idx_in_reduced = joint_set.reduced.index(21)


        # --- RNN 模块定义 ---

        # --- 速度估计网络 (在手部接触网络之前) ---
        # 输入: 所有IMU数据 (human_imu + obj_imu)
        # 输出: 物体速度(3) + 叶子节点速度(n_leaf * 3)
        n_velocity_output = 3 + joint_set.n_leaf * 3  # 物体速度 + 叶子节点速度
        self.velocity_net = RNN(n_imu, n_velocity_output, 256 * hidden_dim_multiplier) # 速度估计网络

        # --- 手部接触预测网络 ---
        # 输入: 3*imu_dim (双手IMU+物体IMU) + 双手速度(2*3) + 物体速度(3)
        n_hand_contact_input = 3 * self.imu_dim + 2 * 3 + 3
        self.hand_contact_net = RNN(n_hand_contact_input, 3, 128 * hidden_dim_multiplier) # 输出3个接触概率
        
                # 姿态估计级联架构 (只使用人体IMU + 人体相关节点速度 + 手部接触概率)
        # S1输入: human_imu + 叶子节点速度（人体相关的速度）+ 手部接触概率(3)
        self.pose_s1 = RNN(n_human_imu + joint_set.n_leaf * 3 + 3, joint_set.n_leaf * 3, 256 * hidden_dim_multiplier)
        self.pose_s2 = RNN(joint_set.n_leaf * 3 + n_human_imu, joint_set.n_full * 3, 64 * hidden_dim_multiplier)
        self.pose_s3 = RNN(joint_set.n_full * 3 + n_human_imu, joint_set.n_reduced * 6, 128 * hidden_dim_multiplier)

        # 足部接触概率预测网络 (只使用人体IMU)
        self.contact_prob_net = RNN(joint_set.n_leaf * 3 + n_human_imu, 2, 64 * hidden_dim_multiplier)

        # 基于运动学的速度分支预测网络 (设置为单向, 只使用人体IMU)
        self.trans_b2 = RNN(joint_set.n_full * 3 + n_human_imu, 3, 256 * hidden_dim_multiplier, bidirectional=False)
        
        # --- 物体平移预测网络 ---
        # 输入: 2*3 (手部位置) + 3 (接触概率) + 3 (物体速度)
        n_obj_trans_input = (2 * 3) + 3 + 3
        self.obj_trans_net = RNN(n_obj_trans_input, 3, 128 * hidden_dim_multiplier) # 输出3维物体平移


        # --- 用于生成初始状态的线性层 ---
        # 每个RNN需要 h_0 和 c_0
        # h_0 shape: (num_layers * num_directions, batch_size, hidden_size)
        # c_0 shape: (num_layers * num_directions, batch_size, hidden_size)
        # 我们用一个线性层为 h_0 和 c_0 一起生成，然后 split
        def create_state_projection(hidden_size, num_layers, bidirectional):
            num_directions = 2 if bidirectional else 1
            return torch.nn.Linear(self.initial_state_dim, 2 * num_layers * num_directions * hidden_size).to(self.device)

        # Access hidden size directly from the RNN modules
        self.proj_velocity_state = create_state_projection(self.velocity_net.n_hidden, self.velocity_net.n_rnn_layer, self.velocity_net.num_directions == 2)
        self.proj_s1_state = create_state_projection(self.pose_s1.n_hidden, self.pose_s1.n_rnn_layer, self.pose_s1.num_directions == 2)
        self.proj_s2_state = create_state_projection(self.pose_s2.n_hidden, self.pose_s2.n_rnn_layer, self.pose_s2.num_directions == 2)
        self.proj_s3_state = create_state_projection(self.pose_s3.n_hidden, self.pose_s3.n_rnn_layer, self.pose_s3.num_directions == 2)
        self.proj_contact_state = create_state_projection(self.contact_prob_net.n_hidden, self.contact_prob_net.n_rnn_layer, self.contact_prob_net.num_directions == 2)
        self.proj_trans_b2_state = create_state_projection(self.trans_b2.n_hidden, self.trans_b2.n_rnn_layer, self.trans_b2.num_directions == 2)
        self.proj_hand_contact_state = create_state_projection(self.hand_contact_net.n_hidden, self.hand_contact_net.n_rnn_layer, self.hand_contact_net.num_directions == 2)
        self.proj_obj_trans_state = create_state_projection(self.obj_trans_net.n_hidden, self.obj_trans_net.n_rnn_layer, self.obj_trans_net.num_directions == 2)


        # 3. SMPL Body Model for FK
        self.body_model = BodyModel(bm_fname=cfg.bm_path, num_betas=16).to(cfg.device) # 假设cfg有device
        for p in self.body_model.parameters(): # Freeze SMPL model parameters
            p.requires_grad = False
        self.parents_tensor = self.body_model.kintree_table[0].long().to(cfg.device) # Store parents tensor
        self.imu_joints_pos = torch.tensor(IMU_JOINTS_POS, device=cfg.device) # Store IMU joint indices
        self.imu_joints_rot = torch.tensor(IMU_JOINTS_ROT, device=cfg.device) # Store IMU joint indices

        # 4. 常量
        # 确保 gravity_velocity 是一个 tensor 并移到正确的设备
        self.gravity_velocity = torch.tensor([0.0, 0.0, -0.018], dtype=torch.float32).to(cfg.device)
        self.vel_scale = cfg.vel_scale if hasattr(cfg, 'vel_scale') else 1.66 # 参考代码中的 vel_scale
        self.prob_threshold_min = cfg.prob_threshold_min if hasattr(cfg, 'prob_threshold_min') else 0.1 # 参考代码使用了 (0.5, 0.9)，这里用更宽松的默认值
        self.prob_threshold_max = cfg.prob_threshold_max if hasattr(cfg, 'prob_threshold_max') else 0.9
        self.floor_y = cfg.floor_y if hasattr(cfg, 'floor_y') else 0.0 # 地面高度
        
        # 物体静止逻辑参数
        self.obj_imu_change_threshold = cfg.obj_imu_change_threshold if hasattr(cfg, 'obj_imu_change_threshold') else 0.1
        self.obj_contact_threshold = cfg.obj_contact_threshold if hasattr(cfg, 'obj_contact_threshold') else 0.3
        self.obj_rot_weight = cfg.obj_rot_weight if hasattr(cfg, 'obj_rot_weight') else 0.5  # 旋转变化的权重
        
        # 手部接触相关参数（用于loss计算）
        self.hand_contact_distance = cfg.hand_contact_distance if hasattr(cfg, 'hand_contact_distance') else 0.1  # 10cm
        # # 5. Hand Refiner Module
        # self.hand_refiner = HandRefiner().to(cfg.device) # <<< 添加
        # 6. Indices for Refiner
        # SMPL joint indices
        self.left_foot_idx, self.right_foot_idx = 7, 8 
        self.wrist_l_idx, self.wrist_r_idx = 20, 21
        # ----------------------------------
        
        
    def format_input(self, data_dict):
        """
        格式化输入数据，包括IMU和第一帧的状态信息。

        Args:
            data_dict: 包含数据的字典，应包含 'human_imu', 'obj_imu' (可选),
                       'motion', 'root_pos', 'obj_rot' (可选), 'obj_trans' (可选)。
                       假设这些序列数据是归一化后的。

        Returns:
            tuple: (imu_data, initial_state_flat)
                   imu_data: 连接后的IMU数据 [batch_size, seq_len, n_imu]
                   initial_state_flat: 第一帧的状态信息 [batch_size, n_initial_state]
        """
        human_imu = data_dict["human_imu"]      # [bs, seq, num_imus, imu_dim]
        obj_imu = data_dict.get("obj_imu", None)  # [bs, seq, 1, imu_dim] or None
        motion = data_dict["motion"]            # [bs, seq, num_joints * joint_dim]
        root_pos = data_dict["root_pos"]        # [bs, seq, 3]
        obj_rot = data_dict.get("obj_rot", None)  # [bs, seq, 6] or None
        obj_trans = data_dict.get("obj_trans", None) # [bs, seq, 3] or None

        batch_size, seq_len = human_imu.shape[:2]

        # --- 处理 IMU 数据 ---
        human_imu_flat_seq = human_imu.reshape(batch_size, seq_len, -1) # [bs, seq, num_imus*imu_dim]
        obj_imu_flat_seq = obj_imu.reshape(batch_size, seq_len, -1) # [bs, seq, 1*imu_dim]
        imu_data = torch.cat([human_imu_flat_seq, obj_imu_flat_seq], dim=2) # [bs, seq, n_imu]

        # --- 处理第一帧状态信息 ---
        motion_start = motion[:, 0].reshape(batch_size, -1) # [bs, num_joints * joint_dim]
        root_pos_start = root_pos[:, 0]                   # [bs, 3]
        obj_rot_start = obj_rot[:, 0]                 # [bs, 6]
        obj_trans_start = obj_trans[:, 0]             # [bs, 3]

        initial_state_flat = torch.cat([
            motion_start,
            root_pos_start,
            obj_rot_start,
            obj_trans_start
        ], dim=1) # [bs, n_initial_state]

        return imu_data, initial_state_flat

    def _reduced_global_to_local_6d(self, imu_rotation_6d, reduced_pose_6d):
        """
        将预测的 reduced 全局旋转 + IMU 根旋转转换为完整的局部 6D 旋转表示 (motion)。

        Args:
            imu_rotation_6d: IMU 的全局 6D 旋转 [bs*seq, 6, 6] (来自 IMU)
            reduced_pose_6d: reduced 关节的预测全局 6D 旋转 [bs*seq, n_reduced*6]
        Returns:
            local_pose_6d: 完整的局部 6D 旋转 [bs*seq, num_joints*6]
        """
        bs_seq = reduced_pose_6d.shape[0] # Use reduced_pose_6d for bs_seq
        device = reduced_pose_6d.device

        # 1. 转换输入为旋转矩阵
        R_imu = rotation_6d_to_matrix(imu_rotation_6d) # [bs*seq, 6, 3, 3]
        R_reduced_pred = rotation_6d_to_matrix(reduced_pose_6d.reshape(bs_seq, joint_set.n_reduced, 6)) # [bs*seq, n_reduced, 3, 3]

        # 2. 构建完整的全局旋转矩阵
        R_global_full = torch.eye(3, device=device).repeat(bs_seq, self.num_joints, 1, 1) # [bs*seq, num_joints, 3, 3]

        # 3. 填充 R_global_full:
        R_global_full[:, joint_set.reduced] = R_reduced_pred
        R_global_full[:, IMU_JOINTS_ROT] = R_imu # 填充全局旋转真值

        # 4. 使用 global2local 转换为局部旋转矩阵
        # 确保 parents_tensor 在正确的设备上
        local_rotmats = global2local(R_global_full, self.parents_tensor.to(device)) # [bs*seq, num_joints, 3, 3]
        local_rotmats[:,0] = R_imu[:, -1] # 填充全局髋部的旋转真值
        local_rotmats[:, [20, 21, 7, 8]] = torch.eye(3, device=device).repeat(bs_seq, 4, 1, 1)

        # 5. 转换回 6D 表示
        local_pose_6d = matrix_to_rotation_6d(local_rotmats.reshape(-1, 3, 3)).reshape(bs_seq, self.num_joints * 6) # [bs*seq, num_joints*6]

        return local_pose_6d
        
    def _prob_to_weight(self, p):
        """
        将概率值转换为权重，用于融合两个平移分支。
        使用 clamp 将概率限制在阈值范围内，然后进行归一化。

        Args:
            p: 概率值 tensor [batch_size*seq_len]

        Returns:
            torch.Tensor: 权重 tensor [batch_size*seq_len, 1]
        """
        # 确保阈值是浮点数
        threshold_min = float(self.prob_threshold_min)
        threshold_max = float(self.prob_threshold_max)
        # 检查阈值是否有效
        if threshold_min >= threshold_max:
            # 如果阈值无效，可以返回一个默认权重，例如0.5，或者抛出错误
            # 这里我们选择返回0.5，并打印一个警告
            print(f"Warning: Invalid probability thresholds: min={threshold_min}, max={threshold_max}. Using default weight 0.5.")
            return torch.full_like(p, 0.5).unsqueeze(1)

        clamped_p = p.clamp(threshold_min, threshold_max)
        weight = (clamped_p - threshold_min) / (threshold_max - threshold_min)
        return weight.unsqueeze(1) # 返回 [batch_size*seq_len, 1]
    
    def compute_obj_imu_change(self, obj_imu):
        """
        计算物体IMU的变化量
        
        Args:
            obj_imu: [bs, seq, imu_dim] (前3维是加速度，后6维是旋转)
        
        Returns:
            imu_change: [bs, seq] 每帧的IMU变化标量
        """
        bs, seq = obj_imu.shape[:2]
        
        # 计算相邻帧的差异
        obj_imu_diff = obj_imu[:, 1:] - obj_imu[:, :-1]  # [bs, seq-1, imu_dim]
        
        # 分别处理加速度和旋转
        acc_diff = obj_imu_diff[:, :, :3]  # 加速度变化
        rot_diff = obj_imu_diff[:, :, 3:]  # 旋转变化
        
        # 计算变化量的幅度
        acc_change = torch.norm(acc_diff, dim=-1)  # [bs, seq-1]
        rot_change = torch.norm(rot_diff, dim=-1)  # [bs, seq-1]
        
        # 综合变化量（可以设置不同权重）
        total_change = acc_change + self.obj_rot_weight * rot_change  # 旋转权重可调
        
        # 第一帧设为0
        first_frame = torch.zeros(bs, 1, device=obj_imu.device)
        imu_change = torch.cat([first_frame, total_change], dim=1)  # [bs, seq]
        
        return imu_change
    
    
    def forward(self, data_dict):
        """
        前向传播

        Args:
            data_dict: 包含输入数据的字典

        Returns:
            dict: 包含预测结果的字典 (motion, obj_rot, root_pos)
        """
        # 1. 准备输入数据: IMU序列 和 第一帧状态
        imu_data, initial_state_flat = self.format_input(data_dict)
        batch_size, seq_len, _ = imu_data.shape
        device = imu_data.device
        human_imu_data = imu_data[:,:,:self.num_human_imus*self.imu_dim] # [bs, seq, 6*9]
        obj_imu_data = imu_data[:,:,self.num_human_imus*self.imu_dim:] # [bs, seq, 1*9]

        # 2. 计算压缩后的初始状态
        compressed_initial_state = self.initial_state_compressor(initial_state_flat) # [bs, initial_state_dim]

        # 3. 定义一个辅助函数来生成 RNN 的初始状态 (h_0, c_0)
        def get_initial_rnn_state(proj_layer, rnn_module):
            # Project the compressed state
            projected_state = proj_layer(compressed_initial_state) # [bs, 2 * num_layers * num_directions * hidden_size]
            # Reshape and split into h_0 and c_0
            num_layers = rnn_module.n_rnn_layer
            num_directions = rnn_module.num_directions
            hidden_size = rnn_module.n_hidden
            expected_shape = (batch_size, 2 * num_layers * num_directions, hidden_size)
            projected_state = projected_state.view(expected_shape)
            # Split into h_0 and c_0
            h_0_c_0 = torch.split(projected_state, num_layers * num_directions, dim=1)
            h_0 = h_0_c_0[0].permute(1, 0, 2).contiguous() # Shape: [num_layers * num_directions, bs, hidden_size]
            c_0 = h_0_c_0[1].permute(1, 0, 2).contiguous() # Shape: [num_layers * num_directions, bs, hidden_size]
            return (h_0, c_0)

        # --- 获取基本 RNNs 的初始状态 ---
        velocity_initial_state = get_initial_rnn_state(self.proj_velocity_state, self.velocity_net)
        s1_initial_state = get_initial_rnn_state(self.proj_s1_state, self.pose_s1)
        s2_initial_state = get_initial_rnn_state(self.proj_s2_state, self.pose_s2)
        s3_initial_state = get_initial_rnn_state(self.proj_s3_state, self.pose_s3)
        contact_initial_state = get_initial_rnn_state(self.proj_contact_state, self.contact_prob_net)
        trans_b2_initial_state = get_initial_rnn_state(self.proj_trans_b2_state, self.trans_b2)

        # --- 速度估计 (在所有其他网络之前) ---
        # 使用所有IMU数据进行速度估计
        pred_velocity_output, _ = self.velocity_net(imu_data, velocity_initial_state) # [bs, seq, n_velocity_output]
        
        # 分离预测的速度
        pred_obj_vel = pred_velocity_output[:, :, :3] # [bs, seq, 3] - 物体速度
        pred_leaf_vel = pred_velocity_output[:, :, 3:].reshape(batch_size, seq_len, joint_set.n_leaf, 3) # [bs, seq, n_leaf, 3] - 叶子节点速度
        pred_leaf_vel_flat = pred_leaf_vel.reshape(batch_size, seq_len, -1) # [bs, seq, n_leaf*3] - 展平的叶子节点速度
        
        # 提取双手速度 (叶子节点中的手腕速度)
        # joint_set.leaf = [7, 8, 12, 20, 21] 对应 [left_ankle, right_ankle, left_collar, left_wrist, right_wrist]
        # 手腕索引在 leaf 中的位置: left_wrist=3, right_wrist=4
        lhand_vel = pred_leaf_vel[:, :, 3, :] # [bs, seq, 3] - 左手速度
        rhand_vel = pred_leaf_vel[:, :, 4, :] # [bs, seq, 3] - 右手速度

        # --- 手部接触预测 ---
        # 提取双手IMU和物体IMU
        human_imu_reshaped = human_imu_data.reshape(batch_size, seq_len, self.num_human_imus, self.imu_dim) # [bs, seq, num_human_imus, imu_dim]
        lhand_imu = human_imu_reshaped[:, :, self.lhand_imu_idx, :] # [bs, seq, imu_dim]
        rhand_imu = human_imu_reshaped[:, :, self.rhand_imu_idx, :] # [bs, seq, imu_dim]
        
        # 构建手部接触网络的输入: IMU + 双手速度 + 物体速度
        imu_feat = torch.cat([lhand_imu, rhand_imu, obj_imu_data], dim=2) # [bs, seq, 3 * imu_dim]
        hand_vel_feat = torch.cat([lhand_vel, rhand_vel], dim=2) # [bs, seq, 2*3]
        hand_contact_input = torch.cat([imu_feat, hand_vel_feat, pred_obj_vel], dim=2) # [bs, seq, 3*imu_dim + 2*3 + 3]
        
        # 获取手部接触网络的初始状态
        hand_contact_initial_state = get_initial_rnn_state(self.proj_hand_contact_state, self.hand_contact_net)
        
        # 手部接触预测
        pred_hand_contact_prob_logits, _ = self.hand_contact_net(hand_contact_input, hand_contact_initial_state) # [bs, seq, 3]
        pred_hand_contact_prob = torch.sigmoid(pred_hand_contact_prob_logits) # [bs, seq, 3]

        # --- 级联网络处理 (使用人体IMU + 人体相关节点速度 + 手部接触概率) --- 
        # RNN 输入现在是 [bs, seq, n_human_imu + n_leaf*3 + 3]

        # 第一阶段：预测关键关节位置
        s1_input = torch.cat([human_imu_data, pred_leaf_vel_flat, pred_hand_contact_prob], dim=2) # [bs, seq, n_human_imu + n_leaf*3 + 3]
        s1_output, _ = self.pose_s1(s1_input, s1_initial_state) # [bs, seq, joint_set.n_leaf * 3]
        pred_leaf_pos = s1_output.reshape(batch_size, seq_len, joint_set.n_leaf, 3) # <<< 添加

        # 第二阶段：预测所有关节位置
        s2_input = torch.cat([s1_output, human_imu_data], dim=2) # [bs, seq, n_output_s1 + n_human_imu]
        s2_output, _ = self.pose_s2(s2_input, s2_initial_state) # [bs, seq, joint_set.n_full * 3]
        pred_full_pos = s2_output.reshape(batch_size, seq_len, joint_set.n_full, 3) # <<< 添加

        # 第三阶段：预测关节旋转
        s3_input = torch.cat([s2_output, human_imu_data], dim=2) # [bs, seq, n_output_s2 + n_human_imu]
        s3_output, _ = self.pose_s3(s3_input, s3_initial_state) # [bs, seq, n_output_s3]
        
        # S3 输出现在是 reduced 关节的全局 6D 旋转
        pred_reduced_global_pose_6d = s3_output # [bs, seq, n_reduced*6]

        human_imu_flat = human_imu_data.reshape(-1, self.num_human_imus, self.imu_dim) # [bs*seq, 6, 9]
        human_imu_rot_flat = human_imu_flat[:,:,3:9] # [bs*seq, 6, 6]
        # === 使用新函数计算最终的局部姿态 (motion) ===
        pred_motion_6d_flat = self._reduced_global_to_local_6d(
            human_imu_rot_flat, # [bs*seq, 6, 6]
            pred_reduced_global_pose_6d.reshape(-1, joint_set.n_reduced * 6) # [bs*seq, n_reduced*6]
        ) # Output: [bs*seq, num_joints*6]

        pred_motion = pred_motion_6d_flat.reshape(batch_size, seq_len, -1) # [bs, seq, 22, 6]
        # ===============================================

        # --- 人体平移估计 (只使用人体IMU) --- 
        # 1. 预测足部接触概率
        contact_prob_input = torch.cat([s1_output, human_imu_data], dim=2) # [bs, seq, n_output_s1 + n_human_imu]
        contact_probability, _ = self.contact_prob_net(contact_prob_input, contact_initial_state) # [bs, seq, 2]
        contact_probability = torch.sigmoid(contact_probability) # [bs, seq, 2]
        contact_probability_flat = contact_probability.reshape(-1, 2) # [bs*seq, 2]

        # 2. 预测速度分支 B2
        trans_b2_input = torch.cat([s2_output, human_imu_data], dim=2) # [bs, seq, n_output_s2 + n_human_imu]
        trans_b2_initial_state = get_initial_rnn_state(self.proj_trans_b2_state, self.trans_b2)
        velocity_b2, _ = self.trans_b2(trans_b2_input, trans_b2_initial_state) # [bs, seq, 3]
        velocity_b2_flat = velocity_b2.reshape(-1, 3) # [bs*seq, 3]

        # --- 后续处理 (保持不变，但使用序列化的数据) ---
        # 3. 准备位姿和根旋转矩阵
        pose_mat_flat = rotation_6d_to_matrix(pred_motion_6d_flat.reshape(-1, 22, 6)) # [bs*seq, num_joints, 3, 3]
        root_orient_mat_flat = pose_mat_flat[:, 0] # [bs*seq, 3, 3] # This now comes from pose_6d (derived from IMU root)
        pose_axis_angle_flat = matrix_to_axis_angle(pose_mat_flat).reshape(batch_size*seq_len, -1) # [bs*seq, num_joints*3]

        # 4. 计算关节全局位置 (Forward Kinematics)
        body_model_output = self.body_model(
            root_orient=pose_axis_angle_flat[:, :3],      # [bs*seq, 3]
            pose_body=pose_axis_angle_flat[:, 3:], # [bs*seq, (num_joints-1)*3]
        )
        j_flat = body_model_output.Jtr[:, :self.num_joints, :] # [bs*seq, num_joints, 3]

        # 5. 计算速度分支 B1
        j_seq = j_flat.reshape(batch_size, seq_len, self.num_joints, 3)
        left_foot_pos = j_seq[:, :, self.left_foot_idx, :]
        right_foot_pos = j_seq[:, :, self.right_foot_idx, :]
        # 计算位移：当前帧 - 上一帧 (注意符号，速度 = pos_prev - pos_curr / dt? 还是 pos_curr - pos_prev / dt?)
        # TransPose 原版计算的是 prev - curr, 表示速度是向前的
        prev_left_foot_pos = torch.roll(left_foot_pos, shifts=1, dims=1)
        prev_right_foot_pos = torch.roll(right_foot_pos, shifts=1, dims=1)
        prev_left_foot_pos[:, 0, :] = left_foot_pos[:, 0, :] # 第一帧速度为0
        prev_right_foot_pos[:, 0, :] = right_foot_pos[:, 0, :] # 第一帧速度为0
        lfoot_vel = (prev_left_foot_pos - left_foot_pos) # * 60.0 ? 如果速度以 m/s 为单位
        rfoot_vel = (prev_right_foot_pos - right_foot_pos) # * 60.0 ?
        lfoot_vel_flat = lfoot_vel.reshape(-1, 3) # [bs*seq, 3]
        rfoot_vel_flat = rfoot_vel.reshape(-1, 3) # [bs*seq, 3]
        
        contact_indices = contact_probability_flat.max(dim=1).indices # [bs*seq]
        chosen_foot_vel = torch.where(contact_indices.unsqueeze(1) == 0, rfoot_vel_flat, lfoot_vel_flat)
        tran_b1_vel = self.gravity_velocity.unsqueeze(0) + chosen_foot_vel # [bs*seq, 3]

        # 6. 计算速度分支 B2
        # velocity_b2_flat 是 RNN 输出，代表局部坐标系下的速度
        # 需要转换到世界坐标系
        tran_b2_vel = torch.bmm(root_orient_mat_flat, velocity_b2_flat.unsqueeze(-1)).squeeze(-1) # [bs*seq, 3]
        # tran_b2_vel = tran_b2_vel * self.vel_scale / 60.0 
        tran_b2_vel = tran_b2_vel * self.vel_scale  # 可能和监督方式有关，上面/60是Transpose Style

        # 7. 融合速度分支
        max_contact_prob = contact_probability_flat.max(dim=1).values # [bs*seq]
        weight = self._prob_to_weight(max_contact_prob) # [bs*seq, 1]
        velocity_flat = lerp(tran_b2_vel, tran_b1_vel, weight) # [bs*seq, 3]

        # 8. 防止地面穿透 (跳过)

        # 9. 计算最终平移 (通过累积速度)
        velocity = velocity_flat.reshape(batch_size, seq_len, 3) # [bs, seq, 3]
        # 假设 velocity 是 m/frame, 直接累积得到相对位移
        trans = torch.cumsum(velocity, dim=1) # [bs, seq, 3]
        # 如果需要绝对位置，需要加上初始位置 data_dict['root_pos'][:, 0]
        # trans = trans + data_dict['root_pos'][:, 0:1, :] # Add initial root position

        # --- 物体平移预测 (使用SMPL全局关节位置) ---
        # 现在我们有了完整的人体平移和姿态，可以通过SMPL前向运动学得到准确的全局关节位置
        
        # 1. 准备SMPL输入参数
        pred_root_orient_6d = pred_motion[:, :, :6]  # [bs, seq, 6]
        pred_body_pose_6d = pred_motion[:, :, 6:]    # [bs, seq, 21*6]
        
        # 转换为轴角表示
        pred_root_orient_6d_flat = pred_root_orient_6d.reshape(-1, 6)
        pred_body_pose_6d_flat = pred_body_pose_6d.reshape(-1, 21, 6)
        pred_root_orient_mat_flat = rotation_6d_to_matrix(pred_root_orient_6d_flat)
        pred_body_pose_mat_flat = rotation_6d_to_matrix(pred_body_pose_6d_flat.reshape(-1, 6)).reshape(-1, 21, 3, 3)
        pred_root_orient_axis_flat = matrix_to_axis_angle(pred_root_orient_mat_flat)
        pred_body_pose_axis_flat = matrix_to_axis_angle(pred_body_pose_mat_flat.reshape(-1, 3, 3)).reshape(-1, 21*3)
        pred_transl_flat = trans.reshape(-1, 3)
        
        # 2. SMPL前向运动学
        pred_pose_body_input = {
            'root_orient': pred_root_orient_axis_flat, 
            'pose_body': pred_body_pose_axis_flat, 
            'trans': pred_transl_flat
        }
        pred_smplh_out = self.body_model(**pred_pose_body_input)
        pred_joints_all = pred_smplh_out.Jtr.view(batch_size, seq_len, -1, 3)  # [bs, seq, num_joints, 3]
        
        # 3. 使用SMPL全局关节位置提取手部位置
        # 手腕关节在SMPL中的索引是20和21
        lhand_pos_global = pred_joints_all[:, :, self.wrist_l_idx, :] # [bs, seq, 3] (关节20)
        rhand_pos_global = pred_joints_all[:, :, self.wrist_r_idx, :] # [bs, seq, 3] (关节21)
        hands_pos_feat = torch.cat([lhand_pos_global, rhand_pos_global], dim=2) # [bs, seq, 2 * 3]
        
        # 4. 获取物体平移网络的初始状态
        obj_trans_initial_state = get_initial_rnn_state(self.proj_obj_trans_state, self.obj_trans_net)
        
        # 5. 物体平移预测 (使用手部位置、接触概率和物体速度)
        obj_trans_input = torch.cat([hands_pos_feat, pred_hand_contact_prob, pred_obj_vel], dim=2)
        pred_obj_trans, _ = self.obj_trans_net(obj_trans_input, obj_trans_initial_state) # [bs, seq, 3] - 物体位置

        # # --- 应用物体静止逻辑 ---
        # # 计算物体IMU变化量
        # obj_imu_change = self.compute_obj_imu_change(obj_imu_data) # [bs, seq]
        
        # # 物体静止条件：仅看IMU变化小
        # should_be_static = obj_imu_change < self.obj_imu_change_threshold # [bs, seq]
        
        # # 让静止时的位置保持与上一帧一致（逐帧处理以处理连续静止帧）
        # for t in range(1, seq_len):
        #     static_mask = should_be_static[:, t]  # [bs]
        #     if static_mask.any():
        #         pred_obj_trans[static_mask, t] = pred_obj_trans[static_mask, t-1]
        
        # --- 提取手部位置信息（用于loss计算） ---
        # 获取手部位置
        hand_positions = torch.stack([lhand_pos_global, rhand_pos_global], dim=2) # [bs, seq, 2, 3]
        
        # --- 重塑回结果格式 ---
        # 返回预测结果，保持原有的键名并添加手部位置信息
        results = {
            "motion": pred_motion,         # [bs, seq, num_joints*6] - 原始预测姿态
            "root_pos": trans,       # [bs, seq, 3] - 人体平移
            "pred_leaf_pos": pred_leaf_pos, # [bs, seq, n_leaf, 3]
            "pred_full_pos": pred_full_pos, # [bs, seq, n_full, 3]
            "root_vel": velocity,     # [bs, seq, 3] - 根关节速度
            "pred_hand_contact_prob": pred_hand_contact_prob, # [bs, seq, 3]
            "pred_obj_trans": pred_obj_trans, # [bs, seq, 3] - 物体位置（相对于初始位置的位移）
            "pred_hand_pos": hand_positions, # [bs, seq, 2, 3] - 手部位置（用于loss计算）
            "pred_obj_vel": pred_obj_vel, # [bs, seq, 3] - 预测的物体速度
            "pred_leaf_vel": pred_leaf_vel, # [bs, seq, n_leaf, 3] - 预测的叶子节点速度
        }
        
        return results 

# --- Hand Refiner Module --- # <<< 添加
class HandRefiner(torch.nn.Module):
    """
    MLP to refine hand/wrist rotations based on local features.
    Input: [B, 36] (L/R Wrist IMU Rot(6D) + L/R Elbow Pred Rot(6D) + L/R Wrist Pred Rot(6D))
    Output: [B, 12] (L/R Wrist 6D delta rotation)
    """
    def __init__(self, in_dim=36, hidden_dim=128, out_dim=12):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x): # x: [B, 36]
        return self.net(x) # Output: [B, 12] 