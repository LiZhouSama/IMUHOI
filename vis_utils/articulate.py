import os
import pickle
import torch
import numpy as np
import scipy.sparse as sp


def forward_kinematics_R(R_local, parent):
    """
    前向运动学，计算关节的全局旋转
    
    参数:
        R_local: 关节的局部旋转矩阵，形状 [batch_size, num_joint, 3, 3]
        parent: 父关节索引列表
    
    返回:
        R_global: 关节的全局旋转矩阵，形状 [batch_size, num_joint, 3, 3]
    """
    R_local = R_local.view(-1, len(parent), 3, 3)
    R_global = torch.zeros_like(R_local)
    
    # 根节点无需变换
    R_global[:, 0] = R_local[:, 0]
    
    # 按照树形结构计算全局旋转
    for i in range(1, len(parent)):
        if parent[i] is not None:
            R_global[:, i] = torch.matmul(R_global[:, parent[i]], R_local[:, i])
    
    return R_global


def inverse_kinematics_R(R_global, parent):
    """
    逆向运动学，从全局旋转计算局部旋转
    
    参数:
        R_global: 关节的全局旋转矩阵，形状 [batch_size, num_joint, 3, 3]
        parent: 父关节索引列表
    
    返回:
        R_local: 关节的局部旋转矩阵，形状 [batch_size, num_joint, 3, 3]
    """
    R_global = R_global.view(-1, len(parent), 3, 3)
    R_local = torch.zeros_like(R_global)
    
    # 根节点无需变换
    R_local[:, 0] = R_global[:, 0]
    
    # 按照树形结构计算局部旋转
    for i in range(1, len(parent)):
        if parent[i] is not None:
            R_local[:, i] = torch.matmul(R_global[:, parent[i]].transpose(1, 2), R_global[:, i])
    
    return R_local


def r6d_to_rotation_matrix(d6: torch.Tensor) -> torch.Tensor:
    """将6D旋转表示转换为旋转矩阵"""
    d6 = d6.view(-1, 6)
    
    # 前两列
    x_raw = d6[:, 0:3]
    y_raw = d6[:, 3:6]
    
    # 归一化x
    x = x_raw / torch.norm(x_raw, dim=1, keepdim=True)
    
    # 使y正交于x
    z = torch.cross(x, y_raw)
    z = z / torch.norm(z, dim=1, keepdim=True)
    
    # 使x正交于y和z
    y = torch.cross(z, x)
    
    # 组合结果
    result = torch.stack([x, y, z], dim=2)
    
    return result


class ParametricModel:
    """
    SMPL参数化人体模型
    """
    def __init__(self, official_model_file: str, device='cpu'):
        """
        初始化SMPL模型
        
        参数:
            official_model_file: SMPL模型文件路径
            device: 计算设备
        """
        with open(official_model_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            
        # 加载模型参数
        # 处理J_regressor，可能是稀疏矩阵或普通数组
        if 'J_regressor' in data:
            j_regressor = data['J_regressor']
            if hasattr(j_regressor, 'toarray'):
                # 如果是稀疏矩阵
                self._J_regressor = torch.from_numpy(j_regressor.toarray()).float().to(device)
            else:
                # 如果是普通数组
                self._J_regressor = torch.from_numpy(j_regressor).float().to(device)
        elif 'J_regressor_prior' in data:
            # SMPLX可能使用不同的键名
            j_regressor = data['J_regressor_prior']
            if hasattr(j_regressor, 'toarray'):
                self._J_regressor = torch.from_numpy(j_regressor.toarray()).float().to(device)
            else:
                self._J_regressor = torch.from_numpy(j_regressor).float().to(device)
        
        # 加载其他参数，对可能的键名做兼容处理
        if 'weights' in data:
            self._skinning_weights = torch.from_numpy(data['weights']).float().to(device)
        elif 'lbs_weights' in data:
            self._skinning_weights = torch.from_numpy(data['lbs_weights']).float().to(device)
            
        if 'posedirs' in data:
            self._posedirs = torch.from_numpy(data['posedirs']).float().to(device)
        elif 'pose_blendshapes' in data:
            self._posedirs = torch.from_numpy(data['pose_blendshapes']).float().to(device)
            
        if 'shapedirs' in data:
            self._shapedirs = torch.from_numpy(np.array(data['shapedirs'])).float().to(device)
        elif 'shape_blendshapes' in data:
            self._shapedirs = torch.from_numpy(np.array(data['shape_blendshapes'])).float().to(device)
            
        if 'v_template' in data:
            self._v_template = torch.from_numpy(data['v_template']).float().to(device)
        elif 'v_shaped' in data:
            self._v_template = torch.from_numpy(data['v_shaped']).float().to(device)
            
        if 'J' in data:
            self._J = torch.from_numpy(data['J']).float().to(device)
        elif 'joints' in data:
            self._J = torch.from_numpy(data['joints']).float().to(device)
            
        # 面片和父节点索引
        if 'f' in data:
            self.face = data['f']
        elif 'faces' in data:
            self.face = data['faces']
            
        if 'kintree_table' in data:
            self.parent = data['kintree_table'][0].tolist()
        else:
            # 默认的SMPL父节点结构
            self.parent = [None, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
            
        # 确保根节点的父节点为None    
        self.parent[0] = None
        self.device = device
    
    def forward_kinematics_R(self, R_local: torch.Tensor):
        """
        前向运动学，计算关节的全局旋转
        
        参数:
            R_local: 关节的局部旋转矩阵
            
        返回:
            R_global: 关节的全局旋转矩阵
        """
        return forward_kinematics_R(R_local, self.parent)
    
    def inverse_kinematics_R(self, R_global: torch.Tensor):
        """
        逆向运动学，从全局旋转计算局部旋转
        
        参数:
            R_global: 关节的全局旋转矩阵
            
        返回:
            R_local: 关节的局部旋转矩阵
        """
        return inverse_kinematics_R(R_global, self.parent)
    
    def forward_kinematics(self, pose: torch.Tensor, shape: torch.Tensor = None, tran: torch.Tensor = None, calc_mesh=False):
        """
        前向运动学，计算关节位置和网格顶点
        
        参数:
            pose: 姿态参数，形状为 [batch_size, num_joint, 3, 3]
            shape: 形状参数，形状为 [batch_size, 10]，默认为None
            tran: 平移参数，形状为 [batch_size, 3]，默认为None
            calc_mesh: 是否计算网格
            
        返回:
            J_global: 全局关节位置
            R_global: 全局关节旋转矩阵
            vertices: 网格顶点（如果calc_mesh=True）
        """
        batch_size = pose.shape[0]
        
        # 获取初始关节和顶点位置
        if shape is None:
            J, v = self._J - self._J[:1], self._v_template - self._J[:1]
            J = J.expand(batch_size, -1, -1)
            v = v.expand(batch_size, -1, -1)
        else:
            shape = shape.view(-1, 10)
            v = torch.tensordot(shape, self._shapedirs, dims=([1], [2])) + self._v_template
            J = torch.matmul(self._J_regressor, v)
            J, v = J - J[:, :1], v - J[:, :1]
        
        # 计算全局旋转
        R_global = self.forward_kinematics_R(pose)
        
        # 计算关节位置
        J_global = torch.zeros_like(J)
        J_global[:, 0] = J[:, 0]
        
        for i in range(1, len(self.parent)):
            if self.parent[i] is not None:
                J_global[:, i] = torch.matmul(R_global[:, self.parent[i]], J[:, i, :, None]).squeeze(-1) + J_global[:, self.parent[i]]
        
        # 添加平移
        if tran is not None:
            J_global = J_global + tran.view(-1, 1, 3)
        
        # 如果需要，计算网格顶点
        vertices = None
        if calc_mesh:
            # 计算姿态形变
            pose_feature = (R_global[:, 1:] - torch.eye(3, device=R_global.device)).view(batch_size, -1)
            pose_offsets = torch.matmul(pose_feature, self._posedirs.view(self._posedirs.shape[0], -1).transpose(0, 1)).view(batch_size, -1, 3)
            
            # 应用姿态形变
            v = v + pose_offsets
            
            # 初始化变形后的顶点
            vertices = torch.zeros_like(v)
            
            # 应用蒙皮权重
            for i in range(len(self.parent)):
                vertices = vertices + self._skinning_weights[:, i:i+1] * (
                    torch.matmul(R_global[:, i], v.transpose(1, 2)).transpose(1, 2) + J_global[:, i:i+1] - 
                    torch.matmul(R_global[:, i], J[:, i:i+1].transpose(1, 2)).transpose(1, 2)
                )
            
            # 添加平移
            if tran is not None:
                vertices = vertices + tran.view(-1, 1, 3)
        
        return J_global, R_global, vertices 