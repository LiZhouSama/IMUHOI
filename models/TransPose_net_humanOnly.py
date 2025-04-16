import torch.nn
from torch.nn.functional import relu
import torch
import numpy as np

class RNN(torch.nn.Module):
    """
    RNN模块，包括线性输入层、RNN和线性输出层。
    """
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=True, dropout=0.2):
        super(RNN, self).__init__()
        self.rnn = torch.nn.LSTM(n_hidden, n_hidden, n_rnn_layer, bidirectional=bidirectional, batch_first=True)
        self.linear1 = torch.nn.Linear(n_input, n_hidden)
        self.linear2 = torch.nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, h=None):
        x, h = self.rnn(relu(self.linear1(self.dropout(x))).unsqueeze(1), h)
        return self.linear2(x.squeeze(1)), h


class TransPoseNet(torch.nn.Module):
    """
    适用于EgoIMU项目的TransPose网络架构。
    用于基于IMU数据预测人体姿态和物体变换。
    将人体IMU和物体IMU整合到同一个级联架构中。
    """
    def __init__(self, cfg):
        """
        初始化TransPose网络

        Args:
            cfg: 配置对象，包含网络参数
        """
        super().__init__()
        
        # 从配置中获取参数
        self.num_human_imus = cfg.num_human_imus if hasattr(cfg, 'num_human_imus') else 6
        self.imu_dim = cfg.imu_dim if hasattr(cfg, 'imu_dim') else 9  # 每个IMU有9维数据(加速度3D + 旋转6D)
        
        # 设置关节数量(使用SMPL模型的标准关节)
        self.num_joints = 22
        self.joint_dim = 6  # 使用6D旋转表示

        # 计算输入维度
        n_human_imu = self.num_human_imus * self.imu_dim  # 人体IMU
        n_imu = n_human_imu  # 总IMU输入
        
        # 定义网络级联架构
        # 第一阶段：预测关键关节位置（人体和物体）
        self.pose_s1 = RNN(n_imu, 5 * 3, 256 * cfg.hidden_dim_multiplier)  # 15个关键关节位置 + 物体位置
        
        # 第二阶段：预测所有关节位置（人体和物体）
        self.pose_s2 = RNN((5 * 3) + n_imu, self.num_joints * 3, 64 * cfg.hidden_dim_multiplier)
        
        # 第三阶段：预测关节旋转（人体和物体）
        self.pose_s3 = RNN((self.num_joints * 3) + n_imu, self.num_joints * self.joint_dim, 128 * cfg.hidden_dim_multiplier)  # 现在obj_rot是6D
        
        # # 根位置估计（两个分支）
        # self.trans_b1 = RNN((15 * 3 + 3) + n_imu, 3 + 3, 64 * cfg.hidden_dim_multiplier)  # 人体根位置 + 物体位置
        # self.trans_b2 = RNN((self.num_joints * 3 + 3) + n_imu, 3 + 3, 256 * cfg.hidden_dim_multiplier, bidirectional=False)
        
        # # 概率预测网络（用于加权两个平移分支）
        # self.contact_prob = RNN(n_imu, 2, 64 * cfg.hidden_dim_multiplier)  # 两个分支的权重
        
        
    def format_input(self, data_dict):
        """
        格式化输入数据

        Args:
            data_dict: 包含数据的字典，应包含'human_imu'和'obj_imu'

        Returns:
            torch.Tensor: 格式化后的输入tensor
        """
        human_imu = data_dict["human_imu"]  # [batch_size, seq_len, num_imus, imu_dim]
        
        batch_size, seq_len = human_imu.shape[:2]
        
        # 重塑IMU数据
        human_imu = human_imu.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, num_imus*imu_dim]
        
        return human_imu
        
    def forward(self, data_dict):
        """
        前向传播

        Args:
            data_dict: 包含输入数据的字典

        Returns:
            dict: 包含预测结果的字典
        """
        # 准备输入数据
        imu_data = self.format_input(data_dict)
        batch_size, seq_len, _ = imu_data.shape
        
        # 重塑为[batch_size*seq_len, input_dim]用于RNN处理
        imu_flat = imu_data.reshape(-1, imu_data.shape[-1])
        
        # 按照整合后的TransPose级联架构处理
        # 第一阶段：预测关键关节位置和物体位置
        leaf_joint_position, _ = self.pose_s1(imu_flat)  # [batch_size*seq_len, 15*3+3]
        
        # 第二阶段输入：第一阶段输出 + IMU数据
        s2_input = torch.cat([leaf_joint_position, imu_flat], dim=1)
        full_joint_position, _ = self.pose_s2(s2_input)  # [batch_size*seq_len, 24*3+3]
        
        # 第三阶段输入：第二阶段输出 + IMU数据
        s3_input = torch.cat([full_joint_position, imu_flat], dim=1)
        pose, _ = self.pose_s3(s3_input)  # [batch_size*seq_len, 24*6+6]
        
        # 重塑回[batch_size, seq_len, ...]
        pose = pose.reshape(batch_size, seq_len, -1)
        
        
        # 返回预测结果
        results = {
            "motion": pose,  # [batch_size, seq_len, 132]
        }
        
        return results 