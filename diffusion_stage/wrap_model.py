import copy

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import inspect
import diffusers
from timm.models.vision_transformer import Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class MotionDiffusion(nn.Module):
    def __init__(self, cfg, input_length, num_layers, use_upper=False, imu_input=True):
        super(MotionDiffusion, self).__init__()
        self.cfg = cfg
        self.scheduler = diffusers.DDIMScheduler(**cfg.scheduler.get("params", dict()))
        self.latent_dim = 512
        self.imu_input = imu_input
        
        # IMU输入编码器 - 将IMU数据(加速度和角速度)编码为潜在特征
        # 人体IMU数据形状为 [batch, seq, num_imus, 6]，物体IMU数据形状为 [batch, seq, 6]
        self.human_imu_encoder = nn.Sequential(
            nn.Linear(6, 16),  # 每个IMU点的特征扩展
            nn.SiLU(),
            nn.Linear(16, 32)
        )
        
        # 物体IMU编码器
        self.obj_imu_encoder = nn.Sequential(
            nn.Linear(6, 16),  # 物体IMU特征扩展
            nn.SiLU(),
            nn.Linear(16, 32)
        )
        
        # 时序编码器 - 用于人体IMU
        # 6个IMU x 32维特征 = 192维
        self.human_temporal_conv = nn.Conv1d(32 * 6, 384, kernel_size=3, padding=1)
        
        # 时序编码器 - 用于物体IMU
        self.obj_temporal_conv = nn.Conv1d(32, 128, kernel_size=3, padding=1)
        
        # 最终IMU特征编码器 - 结合人体和物体特征
        self.combined_feature_encoder = nn.Linear(384 + 128, self.latent_dim)
        
        # 解码器网络 - 生成人体姿态和物体变换
        self.denoiser = Denoiser(input_length, num_layers, self.latent_dim)
        
        # 超参数
        self.mask_training = cfg.mask_training
        self.mask_num = cfg.mask_num

    def diffusion_reverse(self, data=None):
        """
        反向扩散过程 - 从噪声生成姿态和物体变换
        
        Args:
            data: 包含以下键的字典:
                - human_imu: 人体IMU数据 [batch, seq, num_imus, 6]
                - obj_imu: 物体IMU数据 [batch, seq, 6]
                - bps_features: 可选的BPS特征 [batch, seq, N, 4]
        """
        if data is None:
            raise ValueError("输入数据不能为空")
            
        human_imu = data["human_imu"]
        obj_imu = data["obj_imu"]
        bps_features = data.get("bps_features", None)
        
        device = human_imu.device
        bs, seq = human_imu.shape[:2]
        
        # 编码人体IMU数据
        bs, seq, num_imus, _ = human_imu.shape
        human_imu_features = self.human_imu_encoder(human_imu.reshape(bs * seq * num_imus, -1))
        human_imu_features = human_imu_features.reshape(bs, seq, num_imus * 32)
        
        # 编码物体IMU数据
        obj_imu_features = self.obj_imu_encoder(obj_imu.reshape(bs * seq, -1))
        obj_imu_features = obj_imu_features.reshape(bs, seq, 32)
        
        # 应用时序卷积 - 人体IMU
        human_features = human_imu_features.transpose(1, 2)  # [bs, num_imus*32, seq]
        human_features = self.human_temporal_conv(human_features)  # [bs, 384, seq]
        human_features = human_features.transpose(1, 2)  # [bs, seq, 384]
        
        # 应用时序卷积 - 物体IMU
        obj_features = obj_imu_features.transpose(1, 2)  # [bs, 32, seq]
        obj_features = self.obj_temporal_conv(obj_features)  # [bs, 128, seq]
        obj_features = obj_features.transpose(1, 2)  # [bs, seq, 128]
        
        # 整合BPS特征 (如果有)
        if bps_features is not None:
            # TODO: 添加BPS特征处理逻辑
            pass
        
        # 组合人体和物体特征
        combined_features = torch.cat([human_features, obj_features], dim=-1)  # [bs, seq, 512]
        
        # 最终编码
        cond = self.combined_feature_encoder(combined_features)  # [bs, seq, latent_dim]
        
        # 初始化噪声 - 包括人体姿态和物体变换
        # 人体姿态: 132维 (22关节 x 6D旋转)
        # 物体变换: 12维 (3D平移 + 9D旋转矩阵)
        # 总共: 144维
        latents = torch.randn((bs, seq, 144)).to(device).float()
        latents = latents * self.cfg.init_noise_sigma
        
        # 设置时间步长
        self.scheduler.set_timesteps(self.cfg.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(device)
        extra_step_kwargs = {}
        
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.scheduler.eta
            
        # 反向扩散过程
        for i, t in enumerate(timesteps):
            x0_pred = self.denoiser(latents, t.expand(latents.shape[0], ), cond)
            latents = self.scheduler.step(x0_pred, timesteps[i], latents,
                                         **extra_step_kwargs).prev_sample
        
        # 分离人体姿态和物体变换
        motion_pred = latents[:, :, :132]  # [bs, seq, 132]
        obj_trans_pred = latents[:, :, 132:135]  # [bs, seq, 3]
        obj_rot_pred = latents[:, :, 135:144].reshape(bs, seq, 3, 3)  # [bs, seq, 3, 3]
        
        return {
            "motion": motion_pred,
            "obj_trans": obj_trans_pred,
            "obj_rot": obj_rot_pred
        }

    def forward(self, data=None):
        """
        前向传播 - 训练阶段
        
        Args:
            data: 包含以下键的字典:
                - motion: 人体姿态数据 [batch, seq, 132]
                - human_imu: 人体IMU数据 [batch, seq, num_imus, 6]
                - obj_imu: 物体IMU数据 [batch, seq, 6]
                - obj_trans: 物体平移 [batch, seq, 3]
                - obj_rot: 物体旋转矩阵 [batch, seq, 3, 3]
                - bps_features: 可选的BPS特征 [batch, seq, N, 4]
        
        Returns:
            motion_pred: 预测的人体姿态 [batch, seq, 132]
            obj_trans_pred: 预测的物体平移 [batch, seq, 3]
            obj_rot_pred: 预测的物体旋转 [batch, seq, 3, 3]
        """
        if data is None:
            raise ValueError("输入数据不能为空")
            
        motion = data["motion"]
        human_imu = data["human_imu"]
        obj_imu = data["obj_imu"]
        obj_trans = data["obj_trans"]
        obj_rot = data["obj_rot"]
        bps_features = data.get("bps_features", None)
        
        device = human_imu.device
        bs, seq = human_imu.shape[:2]
        
        # 准备目标数据 - 组合人体姿态和物体变换成单一目标
        # 展平旋转矩阵
        obj_rot_flat = obj_rot.reshape(bs, seq, 9)  # [bs, seq, 9]
        
        # 组合目标
        target = torch.cat([motion, obj_trans, obj_rot_flat], dim=-1)  # [bs, seq, 144]
        
        # 编码人体IMU数据
        bs, seq, num_imus, _ = human_imu.shape
        human_imu_features = self.human_imu_encoder(human_imu.reshape(bs * seq * num_imus, -1))
        human_imu_features = human_imu_features.reshape(bs, seq, num_imus * 32)
        
        # 编码物体IMU数据
        obj_imu_features = self.obj_imu_encoder(obj_imu.reshape(bs * seq, -1))
        obj_imu_features = obj_imu_features.reshape(bs, seq, 32)
        
        # 应用时序卷积 - 人体IMU
        human_features = human_imu_features.transpose(1, 2)  # [bs, num_imus*32, seq]
        human_features = self.human_temporal_conv(human_features)  # [bs, 384, seq]
        human_features = human_features.transpose(1, 2)  # [bs, seq, 384]
        
        # 应用时序卷积 - 物体IMU
        obj_features = obj_imu_features.transpose(1, 2)  # [bs, 32, seq]
        obj_features = self.obj_temporal_conv(obj_features)  # [bs, 128, seq]
        obj_features = obj_features.transpose(1, 2)  # [bs, seq, 128]
        
        # 整合BPS特征 (如果有)
        if bps_features is not None:
            # TODO: 添加BPS特征处理逻辑
            pass
        
        # 组合人体和物体特征
        combined_features = torch.cat([human_features, obj_features], dim=-1)  # [bs, seq, 512]
        
        # 最终编码
        cond = self.combined_feature_encoder(combined_features)  # [bs, seq, latent_dim]
        
        # 生成噪声
        noise = torch.randn_like(target).float()
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bs,)).to(device)
        timesteps = timesteps.long()
        
        # 添加噪声到目标
        noisy_target = self.scheduler.add_noise(target.clone(), noise, timesteps)
        
        # 去噪预测
        pred = self.denoiser(noisy_target, timesteps, cond)
        
        # 分离预测结果
        motion_pred = pred[:, :, :132]  # [bs, seq, 132]
        obj_trans_pred = pred[:, :, 132:135]  # [bs, seq, 3]
        obj_rot_pred = pred[:, :, 135:144].reshape(bs, seq, 3, 3)  # [bs, seq, 3, 3]
        
        return {
            "motion": motion_pred,
            "obj_trans": obj_trans_pred,
            "obj_rot": obj_rot_pred
        }


class Denoiser(nn.Module):
    def __init__(self, seq_len, num_layers, latent_dim):
        super(Denoiser, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.embed_timestep = TimestepEmbedder(self.latent_dim)
        self.sparse_up_conv = nn.Conv1d(self.seq_len, self.seq_len, 1)
        self.align_net = nn.Conv1d(self.seq_len, self.seq_len, 1)
        self.down_dim = nn.Linear(self.latent_dim + 144, self.latent_dim)  # 144 = 132(人体) + 12(物体)

        self.blocks = nn.ModuleList([
            DiTBlock(self.latent_dim, 8, mlp_ratio=4) for _ in range(num_layers)
        ])
        nn.init.normal_(self.embed_timestep.mlp[0].weight, std=0.02)
        nn.init.normal_(self.embed_timestep.mlp[2].weight, std=0.02)

        self.last = nn.Linear(self.latent_dim, 144)  # 144 = 132(人体) + 12(物体)

    def forward(self, noisy_latents, timesteps, cond):
        """
        解码器前向传播
        
        Args:
            noisy_latents: 噪声化的潜在特征 [batch, seq, 144]
            timesteps: 时间步 [batch]
            cond: 条件特征 [batch, seq, latent_dim]
        
        Returns:
            预测的去噪结果 [batch, seq, 144]
        """
        bs = cond.shape[0]
        timestep_emb = self.embed_timestep(timesteps)  # (batch, 1, latent_dim)

        cond_up = self.sparse_up_conv(cond)  # (bs, seq, latent_dim)
        noisy_latents = self.align_net(noisy_latents)  # (bs, seq, 144)
        
        input_all = torch.cat((cond_up, noisy_latents), dim=-1)  # (bs, seq, latent_dim + 144)
        input_all_512 = self.down_dim(input_all)  # (bs, seq, latent_dim)

        x = input_all_512
        for block in self.blocks:
            x = block(x, timestep_emb)
        x = self.last(x)  # (bs, seq, 144)
        
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb 