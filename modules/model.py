import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from .diffusion import IMUCondGaussianDiffusion
from .transformer import (
    CondDiffusionTransformer, 
    IMUEncoder, 
    BPSEncoder,
    SinusoidalPosEmb
)

class IMUPoseGenerationModel(nn.Module):
    """
    基于IMU的人体姿态生成模型，使用单阶段扩散模型
    """
    def __init__(
        self,
        input_dim=144,           # 姿态参数维度 (SMPL)
        imu_dim=42,              # IMU数据维度 (7关节 x 6DoF)
        hidden_dim=512,          # 隐藏层维度
        time_dim=128,            # 时间编码维度
        num_layers=6,            # Transformer层数
        nhead=8,                 # 注意力头数
        dropout=0.1,             # Dropout率
        diffusion_steps=1000,    # 扩散步数
        diffusion_loss='l1',     # 扩散损失类型
        beta_schedule='cosine',  # Beta调度类型
        max_seq_len=120,         # 最大序列长度
        use_object=True,         # 是否使用物体信息
        obj_dim=3,               # 物体位置维度
        bps_dim=128              # BPS特征维度
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.imu_dim = imu_dim
        self.hidden_dim = hidden_dim
        self.use_object = use_object
        
        # IMU编码器
        self.imu_encoder = IMUEncoder(
            input_dim=imu_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            dropout=dropout
        )
        
        # 物体处理
        obj_feature_dim = 0
        if use_object:
            # BPS编码器
            self.bps_encoder = BPSEncoder(
                input_dim=3,
                hidden_dim=hidden_dim,
                output_dim=bps_dim,
                n_points=1024
            )
            obj_feature_dim = obj_dim + bps_dim
        else:
            self.bps_encoder = None
        
        # 条件维度 = IMU特征 + 物体特征
        cond_dim = hidden_dim + obj_feature_dim
        
        # Transformer模型
        self.transformer_model = CondDiffusionTransformer(
            input_dim=input_dim,
            cond_dim=cond_dim,
            time_dim=time_dim,
            ffn_dim=hidden_dim * 2,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
            output_dim=input_dim
        )
        
        # 扩散模型
        self.diffusion = IMUCondGaussianDiffusion(
            transformer_model=self.transformer_model,
            imu_encoder=self.imu_encoder,
            bps_encoder=self.bps_encoder,
            timesteps=diffusion_steps,
            loss_type=diffusion_loss,
            beta_schedule=beta_schedule
        )
    
    def forward(self, pose_params, cond_data, padding_mask=None):
        """
        模型前向传播，计算损失
        
        参数:
            pose_params: 人体姿态参数 [B, T, D]
            cond_data: 包含IMU和物体信息的字典
            padding_mask: 序列填充掩码 [B, T]
            
        返回:
            loss: 扩散损失
        """
        return self.diffusion(pose_params, cond_data, padding_mask)
    
    @torch.no_grad()
    def sample(self, cond_data, padding_mask=None):
        """
        从条件生成人体姿态
        
        参数:
            cond_data: 包含IMU和物体信息的字典
            padding_mask: 序列填充掩码
            
        返回:
            生成的人体姿态参数 [B, T, D]
        """
        return self.diffusion.sample(cond_data, padding_mask)
    
    def save(self, path):
        """保存模型到文件"""
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        """从文件加载模型"""
        self.load_state_dict(torch.load(path))
        
    @classmethod
    def from_config(cls, config):
        """从配置创建模型"""
        return cls(
            input_dim=config.get('input_dim', 144),
            imu_dim=config.get('imu_dim', 42),
            hidden_dim=config.get('hidden_dim', 512),
            time_dim=config.get('time_dim', 128),
            num_layers=config.get('num_layers', 6),
            nhead=config.get('nhead', 8),
            dropout=config.get('dropout', 0.1),
            diffusion_steps=config.get('diffusion_steps', 1000),
            diffusion_loss=config.get('diffusion_loss', 'l1'),
            beta_schedule=config.get('beta_schedule', 'cosine'),
            max_seq_len=config.get('max_seq_len', 120),
            use_object=config.get('use_object', True),
            obj_dim=config.get('obj_dim', 3),
            bps_dim=config.get('bps_dim', 128)
        ) 