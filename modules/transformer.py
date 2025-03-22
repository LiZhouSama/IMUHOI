import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区而不是参数
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: 输入特征 [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1)]


# Transformer编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.relu
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: 输入序列 [batch_size, seq_len, d_model]
            src_mask: 注意力掩码 [seq_len, seq_len]
            src_key_padding_mask: 填充掩码 [batch_size, seq_len]
        """
        # 自注意力
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask, 
                             key_padding_mask=~src_key_padding_mask if src_key_padding_mask is not None else None)[0]
        src = src + self.dropout1(src2)
        
        # 前馈网络
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src


# Transformer编码器
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = nn.LayerNorm(encoder_layer.self_attn.embed_dim)
        
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        
        output = self.norm(output)
        return output


# IMU编码器
class IMUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.layers = nn.Sequential(*layers)
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: IMU数据 [batch_size, seq_len, input_dim]
        """
        x = self.input_projection(x)
        x = self.layers(x)
        x = self.output_projection(x)
        return x


# BPS编码器
class BPSEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=128, n_points=1024):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_points = n_points
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
        )
        
        self.process = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: 点云数据 [batch_size, n_points, 3]
        """
        # 投影每个点
        x = self.projection(x)  # [B, n_points, hidden_dim]
        
        # 全局max pooling
        x = torch.max(x, dim=1)[0]  # [B, hidden_dim]
        
        # 进一步处理
        x = self.process(x)
        
        # 输出投影
        x = self.output(x)
        
        return x


# 条件扩散Transformer模型
class CondDiffusionTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        cond_dim,
        time_dim=128,
        ffn_dim=2048,
        nhead=8,
        num_layers=6,
        dropout=0.1,
        max_seq_len=120,
        output_dim=144  # SMPL姿势参数维度
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        # 输入映射
        self.input_map = nn.Linear(input_dim, input_dim)
        
        # 条件映射
        self.cond_map = nn.Linear(cond_dim, input_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(input_dim, max_seq_len)
        
        # 创建Transformer编码器层
        encoder_layer = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout
        )
        
        # Transformer编码器
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        
        # 输出映射
        self.output_map = nn.Linear(input_dim, output_dim)
        
    def forward(self, x, t, cond, padding_mask=None):
        """
        Args:
            x: 输入特征 [batch_size, seq_len, input_dim]
            t: 时间步 [batch_size]
            cond: 条件特征 [batch_size, seq_len, cond_dim]
            padding_mask: 填充掩码 [batch_size, seq_len]
        """
        b, seq_len, device = x.shape[0], x.shape[1], x.device
        
        # 时间嵌入
        t_emb = t * 1000.0 / 1000  # 缩放到0-1000
        t_emb = rearrange(t, 'b -> b 1 1')  # [B, 1, 1]
        t_emb = t_emb.expand(b, seq_len, 1)  # [B, seq_len, 1]
        
        # 处理输入
        x = self.input_map(x)
        
        # 处理条件
        cond_feat = self.cond_map(cond)
        
        # 加入时间信息和条件
        x = x + cond_feat + t_emb
        
        # 加入位置编码
        x = self.pos_encoder(x)
        
        # 准备transformer的padding mask
        if padding_mask is not None:
            # 转换为Transformer所需的掩码格式
            src_key_padding_mask = padding_mask
        else:
            src_key_padding_mask = None
        
        # 通过Transformer
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # 输出映射
        x = self.output_map(x)
        
        return x 