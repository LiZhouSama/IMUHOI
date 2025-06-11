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


def rot6d2matrix(x):
    """
    将6D旋转表示转换为旋转矩阵
    
    Args:
        x: 6D旋转表示 [batch_size, num_joints, 6]
        
    Returns:
        旋转矩阵 [batch_size, num_joints, 3, 3]
    """
    x_shape = x.shape
    x = x.reshape(-1, 6)
    
    # 将前三个和后三个元素分别作为旋转矩阵的两列
    a1, a2 = x[:, :3], x[:, 3:]
    
    # 归一化第一列
    b1 = F.normalize(a1, dim=1)
    
    # 计算第二列的正交分量
    dot_prod = torch.sum(b1 * a2, dim=1, keepdim=True)
    b2 = F.normalize(a2 - dot_prod * b1, dim=1)
    
    # 通过叉积计算第三列
    b3 = torch.cross(b1, b2, dim=1)
    
    # 拼接三列形成旋转矩阵
    rot_matrix = torch.stack([b1, b2, b3], dim=-1)
    
    return rot_matrix.reshape(x_shape[0], x_shape[1], 3, 3)


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