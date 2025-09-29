import numpy as np
import torch
import torch.nn.functional as F


def tensor2numpy(tensor):
    """将PyTorch张量转换为NumPy数组
    
    Args:
        tensor: PyTorch张量
        
    Returns:
        numpy_array: NumPy数组
    """
    return tensor.detach().cpu().numpy()

def _aa_to_R(a: torch.Tensor) -> torch.Tensor:
    """Axis-angle -> rotation matrix. a: [..., 3] -> [..., 3, 3]"""
    orig_shape = a.shape
    a = a.view(-1, 3)
    angle = torch.norm(a, dim=1, keepdim=True)
    # 避免除零
    axis = torch.where(angle > 1e-8, a / angle, torch.tensor([1.0, 0.0, 0.0], device=a.device, dtype=a.dtype).expand_as(a))
    x, y, z = axis[:, 0:1], axis[:, 1:2], axis[:, 2:3]
    c = torch.cos(angle)
    s = torch.sin(angle)
    C = 1 - c
    R = torch.stack([
        c + x * x * C, x * y * C - z * s, x * z * C + y * s,
        y * x * C + z * s, c + y * y * C, y * z * C - x * s,
        z * x * C - y * s, z * y * C + x * s, c + z * z * C
    ], dim=1).view(-1, 3, 3)
    return R.view(*orig_shape[:-1], 3, 3)


def _R_to_aa(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix -> axis-angle. R: [..., 3, 3] -> [..., 3]"""
    orig_shape = R.shape
    R = R.view(-1, 3, 3)
    # 角度
    trace = (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]).clamp(-1.0, 3.0)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    # 轴（避免小角度数值不稳）
    eps = 1e-8
    sin_theta = torch.sin(theta)
    rx = (R[:, 2, 1] - R[:, 1, 2]) / (2.0 * torch.where(sin_theta.abs() < eps, torch.ones_like(sin_theta), sin_theta))
    ry = (R[:, 0, 2] - R[:, 2, 0]) / (2.0 * torch.where(sin_theta.abs() < eps, torch.ones_like(sin_theta), sin_theta))
    rz = (R[:, 1, 0] - R[:, 0, 1]) / (2.0 * torch.where(sin_theta.abs() < eps, torch.ones_like(sin_theta), sin_theta))
    axis = torch.stack([rx, ry, rz], dim=1)
    aa = axis * theta.unsqueeze(1)
    small = theta.abs() < eps
    if small.any():
        aa[small] = 0.0
    return aa.view(*orig_shape[:-2], 3)


def _R_to_r6d(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix -> 6D representation (first two columns). R: [...,3,3] -> [...,6]"""
    orig_shape = R.shape
    R = R.view(-1, 3, 3)
    r6 = R[:, :, :2].transpose(1, 2).contiguous().view(-1, 6)
    return r6.view(*orig_shape[:-2], 6)

def calculate_mpjpe(rotmats, return_joints=False):
    """
    计算MPJPE (Mean Per Joint Position Error)
    
    Args:
        rotmats: 旋转矩阵 [batch_size, num_joints, 3, 3]
        return_joints: 是否返回关节位置
        
    Returns:
        如果return_joints=True，返回关节位置 [batch_size, num_joints, 3]
        否则返回MPJPE值 (float)
    """
    batch_size = rotmats.shape[0]
    num_joints = rotmats.shape[1]
    
    # 简化的骨架模型 - 实际应用中应该使用SMPL模型
    # 定义父关节索引，-1表示没有父关节
    parents = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15, 11, 17, 14, 19])
    
    # 定义骨骼长度 - 这是一个简化的骨架模型
    bone_lengths = torch.ones((num_joints), device=rotmats.device) * 0.1
    
    # 初始化关节位置 - 根关节在原点
    joints = torch.zeros((batch_size, num_joints, 3), device=rotmats.device)
    
    # 计算全局旋转
    global_rotmats = torch.zeros((batch_size, num_joints, 3, 3), device=rotmats.device)
    global_rotmats[:, 0] = rotmats[:, 0]
    
    # 设置骨骼方向 - 简化为沿着z轴的方向
    bone_dirs = torch.zeros((num_joints, 3), device=rotmats.device)
    bone_dirs[:, 2] = 1.0  # z方向
    
    # 前向运动学 - 从根关节开始计算每个关节的位置
    for i in range(1, num_joints):
        parent = parents[i]
        
        # 计算全局旋转
        global_rotmats[:, i] = torch.matmul(global_rotmats[:, parent], rotmats[:, i])
        
        # 计算关节位置 - 父关节位置 + 旋转后的骨骼向量
        parent_joint = joints[:, parent]
        bone_dir = bone_dirs[i]
        bone_vec = bone_dir * bone_lengths[i]
        
        joints[:, i] = parent_joint + torch.matmul(global_rotmats[:, parent], 
                                                  bone_vec.reshape(3, 1)).squeeze(-1)
    
    if return_joints:
        return joints
    
    # 计算MPJPE - 这里我们假设gt_joints也已经计算出来了
    # 在实际应用中，应该比较预测和真实的关节位置
    # mpjpe = torch.mean(torch.sqrt(torch.sum((joints - gt_joints) ** 2, dim=-1)))
    
    # 这里只是一个示例，所以我们返回0.0
    return 0.0


def normalize_vector(v):
    """
    归一化向量
    
    Args:
        v: 向量 [..., 3]
        
    Returns:
        归一化后的向量
    """
    batch = v.shape[:-1]
    v_mag = torch.sqrt(v.pow(2).sum(-1))
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(v.device)))
    v_mag = v_mag.view(*batch, 1)
    v = v / v_mag
    return v


def cross_product(u, v):
    """
    计算叉积
    
    Args:
        u: 第一个向量 [..., 3]
        v: 第二个向量 [..., 3]
        
    Returns:
        叉积结果
    """
    batch = u.shape[:-1]
    i = u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1]
    j = u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2]
    k = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
        
    out = torch.stack((i, j, k), dim=-1)
    
    return out 


def global2local(global_rotmats, parents):
    """
    将全局旋转矩阵转换为局部旋转矩阵。

    Args:
        global_rotmats: 全局旋转矩阵 [batch_size, num_joints, 3, 3]
        parents: 父关节索引数组 (NumPy array)，-1 表示根关节

    Returns:
        local_rotmats: 局部旋转矩阵 [batch_size, num_joints, 3, 3]
    """
    batch_size, num_joints, _, _ = global_rotmats.shape
    device = global_rotmats.device
    
    local_rotmats = torch.zeros_like(global_rotmats)
    
    # 根关节的局部旋转等于其全局旋转
    local_rotmats[:, 0] = global_rotmats[:, 0]
    
    # 遍历非根关节
    for i in range(1, num_joints):
        parent_idx = parents[i]
        
        # 获取父关节和当前关节的全局旋转
        R_global_parent = global_rotmats[:, parent_idx]
        R_global_current = global_rotmats[:, i]
        
        # 计算父关节全局旋转的逆（转置）
        R_global_parent_inv = R_global_parent.transpose(-1, -2)
        
        # 计算局部旋转: R_local = R_parent_inv * R_global
        R_local_current = torch.matmul(R_global_parent_inv, R_global_current)
        
        local_rotmats[:, i] = R_local_current
        
    return local_rotmats 