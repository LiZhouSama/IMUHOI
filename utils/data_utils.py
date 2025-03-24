import torch
import numpy as np
import os
import random
from pathlib import Path

# 由于pytorch3d.transforms包可能不存在matrix_to_axis_angle等函数，我们自己实现
def matrix_to_axis_angle(matrix):
    """
    将旋转矩阵转换为轴角表示
    
    参数:
        matrix: 旋转矩阵 [B, ..., 3, 3]
        
    返回:
        axis_angles: 轴角表示 [B, ..., 3]
    """
    # 检查是否有pytorch3d
    try:
        from pytorch3d.transforms import matrix_to_axis_angle as p3d_matrix_to_axis_angle
        return p3d_matrix_to_axis_angle(matrix)
    except (ImportError, AttributeError):
        # 自己实现
        orig_shape = matrix.shape[:-2]
        matrix = matrix.reshape(-1, 3, 3)
        
        # 计算旋转角度
        cos_angle = (torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1) - 1) / 2
        cos_angle = torch.clamp(cos_angle, -1, 1)
        angle = torch.acos(cos_angle)
        
        # 计算旋转轴
        sin_angle = torch.sin(angle)
        
        # 避免除以零
        safe_sin = torch.where(sin_angle.abs() > 1e-6, sin_angle, torch.ones_like(sin_angle))
        
        x = (matrix[..., 2, 1] - matrix[..., 1, 2]) / (2 * safe_sin)
        y = (matrix[..., 0, 2] - matrix[..., 2, 0]) / (2 * safe_sin)
        z = (matrix[..., 1, 0] - matrix[..., 0, 1]) / (2 * safe_sin)
        
        axis = torch.stack([x, y, z], dim=-1)
        
        # 处理特殊情况：角度接近0或π
        # 对于角度接近0，轴不重要，我们设置为[0,0,0]
        # 对于角度接近π，需要特殊处理
        small_angle_mask = angle.abs() < 1e-6
        
        # 对于角度接近0的情况
        axis = torch.where(small_angle_mask.unsqueeze(-1).expand_as(axis),
                         torch.zeros_like(axis),
                         axis)
        
        # 对于角度接近π的情况，需要进一步处理...
        # 这里简化处理
        
        # 将轴和角度组合成轴角表示
        axis_angle = axis * angle.unsqueeze(-1)
        
        return axis_angle.reshape(*orig_shape, 3)

def axis_angle_to_matrix(axis_angle):
    """
    将轴角表示转换为旋转矩阵
    
    参数:
        axis_angle: 轴角表示 [B, ..., 3]
        
    返回:
        matrix: 旋转矩阵 [B, ..., 3, 3]
    """
    # 检查是否有pytorch3d
    try:
        from pytorch3d.transforms import axis_angle_to_matrix as p3d_axis_angle_to_matrix
        return p3d_axis_angle_to_matrix(axis_angle)
    except (ImportError, AttributeError):
        # 自己实现
        orig_shape = axis_angle.shape[:-1]
        axis_angle = axis_angle.reshape(-1, 3)
        
        # 计算角度
        angle = torch.norm(axis_angle, dim=1, keepdim=True)
        
        # 计算单位轴
        axis = axis_angle / (angle + 1e-8)
        
        # 创建skew-symmetric矩阵
        K = torch.zeros(axis.shape[0], 3, 3, device=axis_angle.device)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        
        # 计算旋转矩阵：R = I + sin(θ)*K + (1-cos(θ))*K^2
        I = torch.eye(3, device=axis_angle.device).unsqueeze(0).expand(axis.shape[0], -1, -1)
        K_sq = torch.bmm(K, K)
        
        sin_angle = torch.sin(angle).unsqueeze(-1)
        cos_angle = torch.cos(angle).unsqueeze(-1)
        
        R = I + sin_angle * K + (1 - cos_angle) * K_sq
        
        # 处理角度为0的特殊情况
        small_angle_mask = angle.abs().squeeze(-1) < 1e-6
        R[small_angle_mask] = torch.eye(3, device=axis_angle.device)
        
        return R.reshape(*orig_shape, 3, 3)

# 设置随机种子
def set_seed(seed):
    """设置随机种子以便结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 设置cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 计算均值和标准差
def compute_stats(data_tensors):
    """
    计算数据的均值和标准差，用于归一化
    
    参数:
        data_tensors: 数据张量列表
        
    返回:
        means: 每个数据的均值
        stds: 每个数据的标准差
    """
    means = {}
    stds = {}
    
    for key, tensor_list in data_tensors.items():
        if isinstance(tensor_list, list) and tensor_list:
            # 将所有张量连接起来
            all_data = torch.cat(tensor_list, dim=0)
            
            # 计算均值和标准差
            means[key] = torch.mean(all_data, dim=0)
            stds[key] = torch.std(all_data, dim=0)
            
            # 防止除零错误
            stds[key] = torch.clamp(stds[key], min=1e-6)
    
    return means, stds

# 归一化数据
def normalize_data(data_tensors, means, stds):
    """
    归一化数据
    
    参数:
        data_tensors: 数据张量字典
        means: 均值字典
        stds: 标准差字典
        
    返回:
        normalized_data: 归一化后的数据
    """
    normalized_data = {}
    
    for key, tensor in data_tensors.items():
        if key in means and key in stds:
            normalized_data[key] = (tensor - means[key]) / stds[key]
        else:
            normalized_data[key] = tensor
    
    return normalized_data

# 逆归一化数据
def denormalize_data(data_tensors, means, stds):
    """
    将归一化的数据转换回原始尺度
    
    参数:
        data_tensors: 归一化后的数据张量字典
        means: 均值字典
        stds: 标准差字典
        
    返回:
        denormalized_data: 逆归一化后的数据
    """
    denormalized_data = {}
    
    for key, tensor in data_tensors.items():
        if key in means and key in stds:
            denormalized_data[key] = tensor * stds[key] + means[key]
        else:
            denormalized_data[key] = tensor
    
    return denormalized_data

# 处理旋转矩阵和轴角表示之间的转换
def convert_rotation_matrix_to_axis_angle(rot_matrices):
    """
    将旋转矩阵转换为轴角表示
    
    参数:
        rot_matrices: 旋转矩阵 [B, ..., 3, 3]
        
    返回:
        axis_angles: 轴角表示 [B, ..., 3]
    """
    return matrix_to_axis_angle(rot_matrices)

def convert_axis_angle_to_matrix(axis_angles):
    """
    将轴角表示转换为旋转矩阵
    
    参数:
        axis_angles: 轴角表示 [B, ..., 3]
        
    返回:
        rot_matrices: 旋转矩阵 [B, ..., 3, 3]
    """
    return axis_angle_to_matrix(axis_angles)

# 加载数据集的.pt文件
def load_pt_files(data_dir, pattern="*.pt"):
    """
    加载目录中所有的.pt文件
    
    参数:
        data_dir: 数据目录
        pattern: 文件匹配模式
        
    返回:
        data: 加载的数据列表
    """
    data_dir = Path(data_dir)
    pt_files = list(data_dir.glob(pattern))
    
    data = []
    for pt_file in pt_files:
        try:
            data_item = torch.load(pt_file)
            data.append(data_item)
        except Exception as e:
            print(f"Error loading {pt_file}: {e}")
    
    return data

# 计算填充掩码
def create_padding_mask(lengths, max_len=None):
    """
    创建填充掩码
    
    参数:
        lengths: 每个序列的真实长度 [batch_size]
        max_len: 最大序列长度
        
    返回:
        padding_mask: 填充掩码 [batch_size, max_len]，True表示非填充位置
    """
    batch_size = len(lengths)
    if max_len is None:
        max_len = max(lengths)
    
    padding_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    
    for i, length in enumerate(lengths):
        padding_mask[i, :length] = True
    
    return padding_mask

# BPS计算工具
def compute_bps_features(points, basis, radius=0.3):
    """
    计算BPS特征
    
    参数:
        points: 点云 [B, N, 3]
        basis: BPS基点 [M, 3]
        radius: BPS半径
        
    返回:
        bps_features: BPS特征 [B, M, 3]
    """
    B, N, _ = points.shape
    M = basis.shape[0]
    
    # 将basis扩展到批量大小
    basis = basis.unsqueeze(0).expand(B, M, 3)
    
    # 初始化BPS特征
    bps_features = torch.zeros((B, M, 3), device=points.device)
    
    for b in range(B):
        # 为每个基点找到最近的点
        for m in range(M):
            # 计算基点到所有点的距离
            basis_point = basis[b, m, :]
            distances = torch.norm(points[b] - basis_point.unsqueeze(0), dim=1)
            
            # 找到最近的点
            min_idx = torch.argmin(distances)
            
            # 如果距离小于半径，则计算BPS特征
            if distances[min_idx] < radius:
                # 存储点到基点的偏移
                bps_features[b, m] = points[b, min_idx] - basis_point
            
    return bps_features