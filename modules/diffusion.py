import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction
from einops import rearrange, reduce
from tqdm.auto import tqdm

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    """从a中提取对应时间步t的值"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# 扩散过程的beta调度函数
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule，来自https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

# 用于时间步编码的正弦位置嵌入
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class GaussianDiffusion(nn.Module):
    """
    高斯扩散模型的基类
    """
    def __init__(
        self,
        denoise_fn,
        timesteps=1000,
        loss_type='l1',
        beta_schedule='cosine',
        p2_loss_weight_gamma=0.,
        p2_loss_weight_k=1
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        
        # 设置beta调度
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'未知的beta调度: {beta_schedule}')
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        
        # 保存timesteps
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        # 注册各种缓冲区
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # q(x_t | x_{t-1}) 计算所需的参数
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        # q(x_{t-1} | x_t, x_0) 计算所需的参数
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        
        # p2 权重重写
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)
    
    def predict_start_from_noise(self, x_t, t, noise):
        """从噪声预测初始状态"""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def q_posterior(self, x_start, x_t, t):
        """计算后验 q(x_{t-1} | x_t, x_0)"""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, x, t, cond, padding_mask=None, clip_denoised=True):
        """计算预测模型的均值和方差"""
        model_output = self.denoise_fn(x, t, cond, padding_mask)
        
        x_start = model_output  # 直接预测x0
        
        if clip_denoised:
            x_start = torch.clamp(x_start, -1., 1.)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        
        return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, x, t, cond, padding_mask=None, clip_denoised=True):
        """单步预测 p(x_{t-1} | x_t)"""
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, cond=cond, padding_mask=padding_mask, clip_denoised=clip_denoised
        )
        
        # 添加噪声
        noise = torch.randn_like(x)
        # t=0时不添加噪声
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, padding_mask=None):
        """完整的采样过程"""
        device = self.betas.device
        b = shape[0]
        
        # 从纯噪声开始
        img = torch.randn(shape, device=device)
        
        # 迭代去噪
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                cond,
                padding_mask
            )
            
        return img
    
    @torch.no_grad()
    def sample(self, cond, padding_mask=None):
        """生成采样"""
        batch_size = cond.shape[0]
        seq_len = cond.shape[1]
        
        # 确定模型输出维度
        dim = self.denoise_fn.output_dim
        shape = (batch_size, seq_len, dim)
        
        return self.p_sample_loop(shape, cond, padding_mask)
    
    def q_sample(self, x_start, t, noise=None):
        """向前扩散过程 q(x_t | x_0)"""
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def p_losses(self, x_start, t, cond, noise=None, padding_mask=None):
        """计算损失"""
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        # 添加噪声
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # 预测
        x_recon = self.denoise_fn(x_noisy, t, cond, padding_mask)
        
        # 计算损失
        if self.loss_type == 'l1':
            loss_fn = F.l1_loss
        elif self.loss_type == 'l2':
            loss_fn = F.mse_loss
        else:
            raise ValueError(f'未知的损失类型: {self.loss_type}')
        
        # 计算未加权损失
        if padding_mask is not None:
            # 逐元素损失
            element_wise_loss = loss_fn(x_recon, x_start, reduction='none')
            
            # 应用padding_mask（扩展维度以匹配loss的形状）
            masked_loss = element_wise_loss * padding_mask.unsqueeze(-1)
            
            # 提取对应时间步的p2权重并应用到损失上
            p2_weight = extract(self.p2_loss_weight, t, element_wise_loss.shape)
            weighted_loss = masked_loss * p2_weight
            
            # 对非padding部分取平均得到标量损失
            loss = torch.sum(weighted_loss) / torch.sum(padding_mask)
        else:
            # 逐元素损失
            element_wise_loss = loss_fn(x_recon, x_start, reduction='none')
            
            # 提取对应时间步的p2权重并应用到损失上
            p2_weight = extract(self.p2_loss_weight, t, element_wise_loss.shape)
            weighted_loss = element_wise_loss * p2_weight
            
            # 对所有元素取平均得到标量损失
            loss = weighted_loss.mean()
        
        return loss
    
    def forward(self, x, cond, padding_mask=None):
        """模型前向传播"""
        b, device = x.shape[0], x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, cond, padding_mask=padding_mask)


class IMUCondGaussianDiffusion(GaussianDiffusion):
    """
    条件高斯扩散模型，专门处理IMU条件和全身姿态生成
    """
    def __init__(
        self,
        transformer_model,
        imu_encoder,
        bps_encoder,
        timesteps=1000,
        loss_type='l1',
        beta_schedule='cosine',
        p2_loss_weight_gamma=0.,
        p2_loss_weight_k=1
    ):
        # 调用父类初始化，但暂时不传递denoise_fn
        super().__init__(
            None,  # 临时为None，稍后会设置
            timesteps=timesteps,
            loss_type=loss_type,
            beta_schedule=beta_schedule,
            p2_loss_weight_gamma=p2_loss_weight_gamma,
            p2_loss_weight_k=p2_loss_weight_k
        )
        
        # 设置模型组件
        self.transformer_model = transformer_model
        self.imu_encoder = imu_encoder
        self.bps_encoder = bps_encoder
        
        # 创建一个包装函数作为denoise_fn
        # 我们不能直接使用实例方法作为denoise_fn，因为它需要self参数
        def wrapped_denoise_fn(x, t, cond, padding_mask=None):
            return self._denoise_function(x, t, cond, padding_mask)
        
        # 将这个函数设置为denoise_fn
        self.denoise_fn = wrapped_denoise_fn
    
    def _denoise_function(self, x, t, cond, padding_mask=None):
        """
        自定义去噪函数的实际实现
        
        参数:
            x: 带噪声的人体姿态 [B, T, D]
            t: 时间步 [B]
            cond: 条件数据，包含IMU和物体信息 [B, T, C]
            padding_mask: 填充掩码 [B, T]
            
        返回:
            预测的人体姿态 [B, T, D]
        """
        # 从条件中分离IMU数据和物体数据
        imu_data = cond["imu_data"]  # [B, T, 42]
        
        # 编码IMU数据
        imu_encoded = self.imu_encoder(imu_data)  # [B, T, E_imu]
        
        # 处理物体数据(如果存在)
        obj_features = None
        if "obj_trans" in cond and self.bps_encoder is not None:
            obj_trans = cond["obj_trans"]  # [B, T, 3]
            obj_bps = cond["obj_bps"]  # [B, 3072]
            
            # 重塑BPS特征
            obj_bps = obj_bps.reshape(obj_bps.shape[0], -1, 3)  # [B, 1024, 3]
            
            # 编码BPS特征
            bps_encoded = self.bps_encoder(obj_bps)  # [B, E_bps]
            
            # 扩展到时间维度
            bps_encoded = bps_encoded.unsqueeze(1).repeat(1, x.shape[1], 1)  # [B, T, E_bps]
            
            # 合并物体位置和BPS特征
            obj_features = torch.cat([obj_trans, bps_encoded], dim=-1)  # [B, T, 3+E_bps]
        
        # 合并所有特征作为条件
        combined_features = imu_encoded
        if obj_features is not None:
            combined_features = torch.cat([combined_features, obj_features], dim=-1)
        
        # 通过Transformer预测去噪结果
        output = self.transformer_model(x, t, combined_features, padding_mask)
        
        return output
    
    @torch.no_grad()
    def sample(self, cond, padding_mask=None):
        """
        从条件生成人体姿态
        
        参数:
            cond: 条件数据，包含IMU和物体信息的字典
            padding_mask: 填充掩码
            
        返回:
            生成的人体姿态 [B, T, D]
        """
        batch_size = cond["imu_data"].shape[0]
        seq_len = cond["imu_data"].shape[1]
        
        # 确定输出维度
        dim = self.transformer_model.output_dim
        shape = (batch_size, seq_len, dim)
        
        return self.p_sample_loop(shape, cond, padding_mask) 