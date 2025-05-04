import sys
import os
sys.path.append("../")
import torch.nn
from torch.nn.functional import relu
import torch
import numpy as np
from human_body_prior.body_model.body_model import BodyModel
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d

from preprocess import IMU_JOINT_NAMES, IMU_JOINTS

class joint_set:
    # TransPose关节索引
    leaf = [7, 8, 12, 20, 21]
    full = list(range(1, 22))
    reduced = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    ignored = [0, 10, 11]

    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]
    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)


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



        # --- RNN 模块定义 ---
        # 姿态估计级联架构
        self.pose_s1 = RNN(n_imu, joint_set.n_leaf * 3, 256 * hidden_dim_multiplier)
        self.pose_s2 = RNN(joint_set.n_leaf * 3 + n_imu, joint_set.n_full * 3, 64 * hidden_dim_multiplier)
        self.pose_s3 = RNN(joint_set.n_full * 3 + n_imu, joint_set.n_reduced * 6 + 6, 128 * hidden_dim_multiplier)

        # 足部接触概率预测网络
        self.contact_prob_net = RNN(joint_set.n_leaf * 3 + n_imu, 2, 64 * hidden_dim_multiplier)

        # 基于运动学的速度分支预测网络 (设置为单向)
        self.trans_b2 = RNN(joint_set.n_full * 3 + n_imu, 3, 256 * hidden_dim_multiplier, bidirectional=False)

        # --- 用于生成初始状态的线性层 ---
        # 每个RNN需要 h_0 和 c_0
        # h_0 shape: (num_layers * num_directions, batch_size, hidden_size)
        # c_0 shape: (num_layers * num_directions, batch_size, hidden_size)
        # 我们用一个线性层为 h_0 和 c_0 一起生成，然后 split
        def create_state_projection(hidden_size, num_layers, bidirectional):
            num_directions = 2 if bidirectional else 1
            return torch.nn.Linear(self.initial_state_dim, 2 * num_layers * num_directions * hidden_size).to(self.device)

        # Access hidden size directly from the RNN modules
        self.proj_s1_state = create_state_projection(self.pose_s1.n_hidden, self.pose_s1.n_rnn_layer, self.pose_s1.num_directions == 2)
        self.proj_s2_state = create_state_projection(self.pose_s2.n_hidden, self.pose_s2.n_rnn_layer, self.pose_s2.num_directions == 2)
        self.proj_s3_state = create_state_projection(self.pose_s3.n_hidden, self.pose_s3.n_rnn_layer, self.pose_s3.num_directions == 2)
        self.proj_contact_state = create_state_projection(self.contact_prob_net.n_hidden, self.contact_prob_net.n_rnn_layer, self.contact_prob_net.num_directions == 2)
        self.proj_trans_b2_state = create_state_projection(self.trans_b2.n_hidden, self.trans_b2.n_rnn_layer, self.trans_b2.num_directions == 2)


        # 3. SMPL Body Model for FK
        self.body_model = BodyModel(bm_fname=cfg.bm_path, num_betas=16).to(cfg.device) # 假设cfg有device
        for p in self.body_model.parameters(): # Freeze SMPL model parameters
            p.requires_grad = False

        # 4. 常量
        self.left_foot_idx = 10 # e.g., 10 for SMPL
        self.right_foot_idx = 11 # e.g., 11 for SMPL
        # 确保 gravity_velocity 是一个 tensor 并移到正确的设备
        self.gravity_velocity = torch.tensor([0.0, 0.0, -0.018], dtype=torch.float32).to(cfg.device)
        self.vel_scale = cfg.vel_scale if hasattr(cfg, 'vel_scale') else 1.66 # 参考代码中的 vel_scale
        self.prob_threshold_min = cfg.prob_threshold_min if hasattr(cfg, 'prob_threshold_min') else 0.1 # 参考代码使用了 (0.5, 0.9)，这里用更宽松的默认值
        self.prob_threshold_max = cfg.prob_threshold_max if hasattr(cfg, 'prob_threshold_max') else 0.9
        self.floor_y = cfg.floor_y if hasattr(cfg, 'floor_y') else 0.0 # 地面高度
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
        device = human_imu.device

        # --- 处理 IMU 数据 ---
        human_imu_flat_seq = human_imu.reshape(batch_size, seq_len, -1) # [bs, seq, num_imus*imu_dim]

        if obj_imu is None:
            # 如果没有物体IMU，用零填充
            obj_imu_flat_seq = torch.zeros((batch_size, seq_len, 1 * self.imu_dim), device=device, dtype=human_imu.dtype)
            print("没有物体IMU，用零填充")
        else:
            obj_imu_flat_seq = obj_imu.reshape(batch_size, seq_len, -1) # [bs, seq, 1*imu_dim]

        imu_data = torch.cat([human_imu_flat_seq, obj_imu_flat_seq], dim=2) # [bs, seq, n_imu]

        # --- 处理第一帧状态信息 ---
        motion_start = motion[:, 0].reshape(batch_size, -1) # [bs, num_joints * joint_dim]
        root_pos_start = root_pos[:, 0]                   # [bs, 3]

        if obj_rot is None:
            obj_rot_start = torch.zeros((batch_size, 6), device=device, dtype=motion.dtype)
            print("没有物体旋转，用零填充")
        else:
            obj_rot_start = obj_rot[:, 0]                 # [bs, 6]

        if obj_trans is None:
            obj_trans_start = torch.zeros((batch_size, 3), device=device, dtype=motion.dtype)
            print("没有物体平移，用零填充")
        else:
            obj_trans_start = obj_trans[:, 0]             # [bs, 3]

        initial_state_flat = torch.cat([
            motion_start,
            root_pos_start,
            obj_rot_start,
            obj_trans_start
        ], dim=1) # [bs, n_initial_state]

        return imu_data, initial_state_flat
    
    def _reduced_to_pose_6d(self, root_rotation, reduced_pose):
        reduced_pose = rotation_6d_to_matrix(reduced_pose).reshape(-1, joint_set.n_reduced, 3, 3)
        pose_mat = torch.eye(3, device=reduced_pose.device).repeat(reduced_pose.shape[0], 22, 1, 1)
        pose_mat[:, joint_set.reduced] = reduced_pose
        pose_mat[:, joint_set.ignored] = torch.eye(3, device=pose_mat.device)
        pose_mat[:, 0] = rotation_6d_to_matrix(root_rotation)
        return matrix_to_rotation_6d(pose_mat)
        
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

        # 2. 计算压缩后的初始状态
        compressed_initial_state = self.initial_state_compressor(initial_state_flat) # [bs, initial_state_dim]

        # 3. 定义一个辅助函数来生成 RNN 的初始状态 (h_0, c_0)
        def get_initial_state(proj_layer, rnn_module):
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

        # --- 级联网络处理 --- 
        # RNN 输入现在是 [bs, seq, n_input]

        # 第一阶段：预测关键关节位置
        s1_input = imu_data # [bs, seq, n_imu]
        s1_initial_state = get_initial_state(self.proj_s1_state, self.pose_s1)
        s1_output, _ = self.pose_s1(s1_input, s1_initial_state) # [bs, seq, n_output_s1]

        # 第二阶段：预测所有关节位置
        s2_input = torch.cat([s1_output, imu_data], dim=2) # [bs, seq, n_output_s1 + n_imu]
        s2_initial_state = get_initial_state(self.proj_s2_state, self.pose_s2)
        s2_output, _ = self.pose_s2(s2_input, s2_initial_state) # [bs, seq, n_output_s2]

        # 第三阶段：预测关节旋转和物体旋转
        s3_input = torch.cat([s2_output, imu_data], dim=2) # [bs, seq, n_output_s2 + n_imu]
        s3_initial_state = get_initial_state(self.proj_s3_state, self.pose_s3)
        s3_output, _ = self.pose_s3(s3_input, s3_initial_state) # [bs, seq, n_output_s3]
        
        # 从S3输出中分离 reduced pose 和 object rotation
        # n_output_s3 = joint_set.n_reduced * 6 + 6
        reduced_pose_6d = s3_output[:, :, :-6] # [bs, seq, n_reduced*6]
        obj_rot_6d = s3_output[:, :, -6:] # [bs, seq, 6]

        # 从IMU提取根旋转(用于转换到完整姿态，但不是直接用作最终根旋转)
        root_idx = IMU_JOINT_NAMES.index('hip')
        imu_data_flat = imu_data.reshape(-1, imu_data.shape[-1]) # [bs*seq, n_imu]
        root_orient_6d_from_imu_flat = imu_data_flat[:, root_idx*self.imu_dim + 3 : root_idx*self.imu_dim + 9] # [bs*seq, 6]
        # 使用S3预测的根旋转, S3的输出包含了根旋转
        # _reduced_to_pose_6d 需要修改以接受S3预测的根旋转部分
        # pose_6d = self._reduced_to_pose_6d(root_orient_6d_from_imu_flat, reduced_pose_6d.reshape(-1, joint_set.n_reduced, 6))
        # 注意：pose_s3 的输出维度 n_output_s3 = joint_set.n_reduced * 6 + 6
        # 这意味着它预测了 reduced 关节的6D旋转 + 物体6D旋转
        # 它 *没有* 预测根关节的旋转。我们需要修改 _reduced_to_pose_6d 来使用 IMU 的根旋转。
        pose_6d = self._reduced_to_pose_6d(
            root_orient_6d_from_imu_flat, # 从IMU获取根旋转 [bs*seq, 6]
            reduced_pose_6d.reshape(-1, joint_set.n_reduced, 6) # 从S3获取肢体旋转 [bs*seq, n_reduced, 6]
        ).reshape(batch_size, seq_len, -1) # [bs, seq, num_joints*6]


        # --- 人体平移估计 --- 
        # 1. 预测足部接触概率
        contact_prob_input = torch.cat([s1_output, imu_data], dim=2) # [bs, seq, n_output_s1 + n_imu]
        contact_initial_state = get_initial_state(self.proj_contact_state, self.contact_prob_net)
        contact_probability, _ = self.contact_prob_net(contact_prob_input, contact_initial_state) # [bs, seq, 2]
        contact_probability = torch.sigmoid(contact_probability) # [bs, seq, 2]
        contact_probability_flat = contact_probability.reshape(-1, 2) # [bs*seq, 2]

        # 2. 预测速度分支 B2
        trans_b2_input = torch.cat([s2_output, imu_data], dim=2) # [bs, seq, n_output_s2 + n_imu]
        trans_b2_initial_state = get_initial_state(self.proj_trans_b2_state, self.trans_b2)
        velocity_b2, _ = self.trans_b2(trans_b2_input, trans_b2_initial_state) # [bs, seq, 3]
        velocity_b2_flat = velocity_b2.reshape(-1, 3) # [bs*seq, 3]

        # --- 后续处理 (保持不变，但使用序列化的数据) ---
        # 3. 准备位姿和根旋转矩阵
        pose_6d_flat = pose_6d.reshape(-1, self.num_joints * 6) # [bs*seq, num_joints*6]
        pose_mat_flat = rotation_6d_to_matrix(pose_6d_flat.reshape(-1, self.num_joints, 6)) # [bs*seq, num_joints, 3, 3]
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
        # 原 TransPose 做了 vel_scale / 60.0 的缩放，这里也加上
        tran_b2_vel = tran_b2_vel * self.vel_scale / 60.0 # ? (如果速度是m/frame, 乘以scale得到m/s? 还是说velocity_b2已经是m/s了?)
                                                    # 假设 velocity_b2 输出的是 m/frame (因为RNN在每帧上操作)
                                                    # 那么乘以 vel_scale (无单位?) / 60 (frames/s) 得到 m/s?
                                                    # 或者 velocity_b2 输出的是某种无单位的速度表示，乘以 scale/60 转为 m/frame?
                                                    # 暂时只乘以 vel_scale，单位问题需要后续确认

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

        # --- 重塑回结果格式 ---
        # pose_6d 已经是 [bs, seq, num_joints*6]
        obj_rot_6d = obj_rot_6d # [bs, seq, 6]
        
        # 返回预测结果
        results = {
            "motion": pose_6d,         # [bs, seq, num_joints*6]
            "obj_rot": obj_rot_6d,  # [bs, seq, 6]
            "root_pos": trans       # [bs, seq, 3] - 人体平移
        }
        
        return results 