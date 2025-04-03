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
    def __init__(self, cfg, input_length, num_layers, imu_input=True):
        super(MotionDiffusion, self).__init__()
        self.cfg = cfg
        self.scheduler = diffusers.DDIMScheduler(**cfg.scheduler.get("params", dict()))
        self.latent_dim = cfg.model.d_model
        self.imu_input = imu_input
        self.num_layers = cfg.model.num_layers
        self.num_heads = cfg.model.n_heads
        
        # 设置掩码训练参数
        self.mask_training = cfg.get('mask_training', False)
        self.mask_num = cfg.get('mask_num', 2)  # 默认掩码2个IMU传感器
        
        # IMU输入编码器
        self.human_imu_encoder = nn.Sequential(
            nn.Linear(6, 16),
            nn.SiLU(),
            nn.Linear(16, 32)
        )
        self.obj_imu_encoder = nn.Sequential(
            nn.Linear(6, 16),
            nn.SiLU(),
            nn.Linear(16, 32)
        )
        
        # 假设有6个人体IMU + 1个物体IMU
        num_total_imus_features = 32 * 6 + 32 * 1 
        # 最终IMU特征编码器 - 将IMU特征映射到latent_dim
        self.combined_feature_encoder = nn.Linear(num_total_imus_features, self.latent_dim) 
        
        # 目标数据 (pose/object) 映射到 latent_dim
        self.target_feature_dim = 147 # 3(root_pos) + 132(motion) + 3(obj_trans) + 9(obj_rot_flat)
        self.input_projection = nn.Linear(self.target_feature_dim, self.latent_dim)

        # 解码器网络 - 生成人体姿态和物体变换
        self.denoiser = Denoiser(
            seq_len=input_length, 
            latent_dim=self.latent_dim, 
            num_layers=self.num_layers, 
            num_heads=self.num_heads
        )
        
        # 最终输出层 - 从 latent_dim 映射回目标维度
        self.output_projection = nn.Linear(self.latent_dim, self.target_feature_dim)

    def _encode_imu(self, human_imu, obj_imu):
        """Helper function to encode IMU data and apply masking."""
        bs, seq, num_imus, _ = human_imu.shape
        
        # 编码人体IMU数据
        human_imu_features = self.human_imu_encoder(human_imu.reshape(bs * seq * num_imus, -1))
        human_imu_features = human_imu_features.reshape(bs, seq, num_imus * 32)
        
        # 应用mask训练策略 (仅在训练时)
        if self.training and self.mask_training:
            human_imu_features_reshaped = human_imu_features.reshape(bs, seq, num_imus, 32)
            for i in range(bs):
                mask_index = torch.randint(0, num_imus, (self.mask_num,), device=human_imu.device)
                human_imu_features_reshaped[i, :, mask_index] = torch.ones_like(human_imu_features_reshaped[i, :, mask_index]) * 0.01
            human_imu_features = human_imu_features_reshaped.reshape(bs, seq, num_imus * 32)
        
        # 编码物体IMU数据
        obj_imu_features = self.obj_imu_encoder(obj_imu.reshape(bs * seq, -1))
        obj_imu_features = obj_imu_features.reshape(bs, seq, 32)
        
        # 组合人体和物体特征
        combined_features = torch.cat([human_imu_features, obj_imu_features], dim=-1)  # [bs, seq, 32*7]
        
        # 最终编码
        cond = self.combined_feature_encoder(combined_features) # [bs, seq, latent_dim]
        return cond

    def diffusion_reverse(self, data=None):
        if data is None:
            raise ValueError("输入数据不能为空")
            
        human_imu = data["human_imu"]
        obj_imu = data["obj_imu"]
        # bps_features = data.get("bps_features", None) # BPS 暂不处理
        
        device = human_imu.device
        bs, seq = human_imu.shape[:2]

        # 编码IMU条件
        cond = self._encode_imu(human_imu, obj_imu) # [bs, seq, latent_dim]

        # 初始化噪声 - 形状应为 [bs, seq, latent_dim]
        latents = torch.randn((bs, seq, self.latent_dim), device=device).float()
        latents = latents * self.scheduler.init_noise_sigma # 使用 scheduler 的 sigma

        # 设置时间步长
        self.scheduler.set_timesteps(self.cfg.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(device)
        
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.scheduler.eta
            
        # 反向扩散过程
        for t in timesteps:
            # 将 cond 添加到 timestep embedding，作为 DiTBlock 的条件输入 c
            model_output = self.denoiser(latents, t.expand(bs), cond) # Denoiser 输出 [bs, seq, latent_dim]
            # scheduler预测的是noise或者x0，具体取决于配置
            # 假设预测的是 x0 (epsilon prediction is common)
            latents = self.scheduler.step(model_output, t, latents, **extra_step_kwargs).prev_sample

        # 将最终的 latent 映射回目标维度
        final_output = self.output_projection(latents) # [bs, seq, 147]
        
        # 分离根关节位置、人体姿态和物体变换
        root_pos_pred = final_output[:, :, :3]
        motion_pred = final_output[:, :, 3:135]
        obj_trans_pred = final_output[:, :, 135:138]
        obj_rot_pred = final_output[:, :, 138:147].reshape(bs, seq, 3, 3)
        
        return {
            "root_pos": root_pos_pred,
            "motion": motion_pred,
            "obj_trans": obj_trans_pred,
            "obj_rot": obj_rot_pred
        }

    def forward(self, data=None):
        if data is None:
            raise ValueError("输入数据不能为空")
            
        root_pos = data["root_pos"]
        motion = data["motion"]
        human_imu = data["human_imu"]
        obj_imu = data["obj_imu"]
        obj_trans = data["obj_trans"]
        obj_rot = data["obj_rot"]
        # bps_features = data.get("bps_features", None) # BPS 暂不处理
        
        device = human_imu.device
        bs, seq = human_imu.shape[:2]
        
        # 准备目标数据 
        obj_rot_flat = obj_rot.reshape(bs, seq, 9)
        target = torch.cat([root_pos, motion, obj_trans, obj_rot_flat], dim=-1) # [bs, seq, 147]
        
        # 将目标数据映射到 latent_dim
        target_latents = self.input_projection(target) # [bs, seq, latent_dim]

        # 编码IMU条件 (与 reverse 过程相同)
        cond = self._encode_imu(human_imu, obj_imu) # [bs, seq, latent_dim]
        
        # 生成噪声
        noise = torch.randn_like(target_latents).float() # 噪声维度应与 target_latents 一致
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bs,), device=device)
        timesteps = timesteps.long()
        
        # 添加噪声到目标 latent
        noisy_target_latents = self.scheduler.add_noise(target_latents, noise, timesteps) # [bs, seq, latent_dim]
        
        # 模型预测噪声 (denoiser现在直接预测噪声或x0，取决于实现)
        # 假设 Denoiser 内部结构已调整为预测噪声 (或可以通过 scheduler 配置判断)
        # 如果 prediction_type 是 epsilon，denoiser 的输出就是 predicted_noise
        # 如果 prediction_type 是 sample，denoiser 输出 predicted_x0, 需要转换
        prediction_type = self.scheduler.config.prediction_type
        
        model_output = self.denoiser(noisy_target_latents, timesteps, cond) # [bs, seq, latent_dim]

        if prediction_type == "epsilon":
            predicted_noise = model_output
        elif prediction_type == "sample":
             # 如果模型预测 x_t-1，需要从模型输出计算出预测的噪声
             alpha_prod_t = self.scheduler.alphas_cumprod[timesteps].view(bs, 1, 1)
             beta_prod_t = 1 - alpha_prod_t
             predicted_noise = (target_latents - torch.sqrt(alpha_prod_t) * model_output) / torch.sqrt(beta_prod_t)
             # 注意：这里假设 model_output 是预测的 x0。如果模型预测的是其他形式，转换会不同。
             # 检查 scheduler 的 add_noise 实现和 step 实现来确认正确的转换方式。
             # 最简单的方式是确保 Denoiser 总是预测 epsilon。
             print("Warning: prediction_type is 'sample'. Ensure noise calculation from model_output is correct.")
        elif prediction_type == "v_prediction":
             # V-prediction 类型的转换
             alpha_prod_t = self.scheduler.alphas_cumprod[timesteps].view(bs, 1, 1)
             beta_prod_t = 1 - alpha_prod_t
             predicted_noise = torch.sqrt(alpha_prod_t) * noise - torch.sqrt(beta_prod_t) * target_latents # 这是v
             predicted_noise = torch.sqrt(beta_prod_t) * model_output + torch.sqrt(alpha_prod_t) * noise # 用v算出epsilon
             print("Warning: prediction_type is 'v_prediction'. Ensure noise calculation from model_output is correct.")
        else:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")

        # 返回预测的噪声和真实的噪声用于计算损失
        return predicted_noise, noise


class Denoiser(nn.Module):
    def __init__(self, seq_len, latent_dim, num_layers, num_heads, mlp_ratio=4.0):
        super(Denoiser, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embed_timestep = TimestepEmbedder(self.latent_dim)
        # 条件投影层 (可选，但有助于对齐维度)
        self.cond_projection = nn.Linear(self.latent_dim, self.latent_dim) 

        self.blocks = nn.ModuleList([
            DiTBlock(self.latent_dim, self.num_heads, mlp_ratio=mlp_ratio) for _ in range(self.num_layers)
        ])
        
        # 初始化 DiTBlock 内部的 adaLN_modulation 层权重
        for block in self.blocks:
             nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
             nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # 添加最终的 LayerNorm 和 线性层 (标准 DiT 结构)
        self.norm_final = nn.LayerNorm(self.latent_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation_final = nn.Sequential(
              nn.SiLU(),
              nn.Linear(self.latent_dim, 2 * self.latent_dim, bias=True) # Shift and Scale for final norm
        )
        nn.init.constant_(self.adaLN_modulation_final[-1].weight, 0) # Zero init for final modulation
        nn.init.constant_(self.adaLN_modulation_final[-1].bias, 0)


    def forward(self, x, timesteps, cond):
        """
        解码器前向传播 (符合 DiT 结构)
        
        Args:
            x: 噪声化的潜在特征 [batch, seq, latent_dim]
            timesteps: 时间步 [batch]
            cond: IMU 条件特征 [batch, seq, latent_dim]
        
        Returns:
            预测的噪声 epsilon (或预测的 x0，取决于训练目标) [batch, seq, latent_dim] 
        """
        bs = x.shape[0]
        # 1. 获取时间步嵌入
        timestep_emb = self.embed_timestep(timesteps) # [batch, latent_dim]
        
        # 2. 融合时间步嵌入和IMU条件
        #    简单相加，或者使用更复杂的融合方法
        #    对 cond 应用线性投影以匹配维度和提供灵活性
        projected_cond = self.cond_projection(cond) # [batch, seq, latent_dim]
        
        # 将 timestep_emb 扩展并加到 projected_cond 上
        # 注意：timestep_emb 是 [bs, latent_dim], projected_cond 是 [bs, seq, latent_dim]
        # 需要将 timestep_emb 扩展到 [bs, seq, latent_dim]
        combined_cond = projected_cond + timestep_emb.unsqueeze(1) # [bs, seq, latent_dim]

        # 3. 通过 DiT Blocks 传递
        #    注意：DiTBlock 需要的 c 是 [bs, latent_dim]，我们将使用序列上的平均条件
        #    或者，修改 DiTBlock 以接受 [bs, seq, latent_dim] 的条件 (更复杂)
        #    这里采用简单的方式：平均池化条件
        cond_pooled = torch.mean(combined_cond, dim=1) # [bs, latent_dim] 

        for block in self.blocks:
            x = block(x, cond_pooled) # 使用池化后的条件进行调制

        # 4. 应用最终的调制归一化
        shift, scale = self.adaLN_modulation_final(cond_pooled).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale) # [bs, seq, latent_dim]

        # Denoiser 的最终输出应该是预测的噪声或 x0
        # 如果目标是预测噪声 (epsilon)，则这里的 x 就是预测的噪声
        # 如果目标是预测 x0，则需要进一步处理，但这通常在 MotionDiffusion.forward 中完成
        # 我们假设 Denoiser 的目标是输出能用于计算 epsilon 的 latent state
        return x # 直接返回 latent state，让 MotionDiffusion.forward 处理


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