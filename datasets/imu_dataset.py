import os
import torch
import numpy as np
import glob
from torch.utils.data import Dataset
import pytorch3d.transforms as transforms

# IMU关节索引
IMU_JOINTS = [20, 21, 15, 7, 8, 0]  # 左手、右手、头部、左脚、右脚、髋部
HEAD_IDX = 2  # 头部IMU在IMU_JOINTS中的索引

class IMUDataset(Dataset):
    """处理IMU数据的数据集类"""
    
    def __init__(
        self,
        data_dir,           # 预处理数据目录
        window_size=120,    # 窗口大小
        stride=60,          # 窗口滑动步长
        normalize_imu=True, # 是否归一化IMU数据
        mode="train",       # 训练/测试模式
        device=None         # 设备
    ):
        super().__init__()
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.normalize_imu = normalize_imu
        self.mode = mode
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载所有pt文件
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        print(f"加载了 {len(self.file_paths)} 个序列文件")
        
        # 解析窗口
        self.windows = self._parse_windows()
        print(f"总共创建了 {len(self.windows)} 个窗口")
        
        # 用于BPS计算的点云基础
        self.setup_bps_basis()
        
    def _parse_windows(self):
        """解析所有序列，创建窗口索引"""
        windows = []
        
        for file_idx, file_path in enumerate(self.file_paths):
            # 加载序列数据
            try:
                data = torch.load(file_path)
                
                # 获取序列长度
                seq_len = data["position_global_full_gt_world"].shape[0]
                
                # 如果序列长度足够，创建窗口
                if seq_len >= self.window_size:
                    # 创建窗口索引
                    for start_idx in range(0, seq_len - self.window_size + 1, self.stride):
                        windows.append({
                            "file_idx": file_idx,
                            "start_idx": start_idx,
                            "end_idx": start_idx + self.window_size
                        })
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {e}")
                
        return windows
    
    def setup_bps_basis(self, n_bps_points=1024, radius=1.0):
        """设置BPS基础点云"""
        # 在单位球上均匀采样点
        u = torch.rand(n_bps_points) * 2 - 1  # [-1, 1]范围内的均匀分布
        v = torch.rand(n_bps_points) * 2 * np.pi  # [0, 2π]范围内的均匀分布
        
        # 计算球面上的坐标
        sqrt_1_u2 = torch.sqrt(1 - u * u)
        x = sqrt_1_u2 * torch.cos(v) * radius
        y = sqrt_1_u2 * torch.sin(v) * radius
        z = u * radius
        
        # 组合成点云 [n_bps_points, 3]
        self.bps_basis = torch.stack([x, y, z], dim=1).to(self.device)
        
    def compute_object_bps(self, obj_verts, obj_center):
        """
        计算物体的BPS表示
        
        参数:
            obj_verts: 物体顶点 [N, 3]
            obj_center: 物体中心 [3]
            
        返回:
            bps_features: BPS特征 [n_bps_points, 3]
        """
        # 将BPS基础点移动到物体中心
        bps_points = self.bps_basis + obj_center
        
        # 计算每个BPS点到最近物体顶点的向量
        dists = torch.cdist(bps_points, obj_verts)  # [n_bps_points, N]
        min_idxs = torch.argmin(dists, dim=1)       # [n_bps_points]
        
        # 计算BPS特征：从BPS点到最近物体顶点的向量
        nearest_verts = obj_verts[min_idxs]                # [n_bps_points, 3]
        bps_features = nearest_verts - bps_points           # [n_bps_points, 3]
        
        return bps_features
    
    def _normalize_imu_data(self, imu_data, positions=None, rotations=None):
        """
        归一化IMU数据
        
        参数:
            imu_data: IMU数据 [T, N*6]
            positions: 关节位置 [T, N, 3] (可选)
            rotations: 关节旋转 [T, N, 3, 3] (可选)
            
        返回:
            norm_imu_data: 归一化后的IMU数据 [T, N*6]
        """
        T, D = imu_data.shape
        N = D // 6
        
        # 重塑IMU数据为加速度和角速度分量
        imu_data = imu_data.reshape(T, N, 6)
        accel = imu_data[..., :3]  # [T, N, 3]
        gyro = imu_data[..., 3:]   # [T, N, 3]
        
        # 1. 相对于第一帧的归一化
        norm_accel = accel - accel[0:1]  # [T, N, 3]
        norm_gyro = gyro - gyro[0:1]     # [T, N, 3]
        
        # 2. 相对于头部IMU的归一化
        head_accel = norm_accel[:, HEAD_IDX:HEAD_IDX+1]  # [T, 1, 3]
        head_gyro = norm_gyro[:, HEAD_IDX:HEAD_IDX+1]    # [T, 1, 3]
        
        rel_accel = norm_accel - head_accel  # [T, N, 3]
        rel_gyro = norm_gyro - head_gyro     # [T, N, 3]
        
        # 如果提供了位置和旋转信息，也进行归一化
        if positions is not None and rotations is not None:
            # 相对于第一帧的归一化
            norm_pos = positions - positions[0:1]  # [T, N, 3]
            
            # 相对旋转：R_rel = R * R_0^(-1)
            R_0_inv = torch.inverse(rotations[0])  # [N, 3, 3]
            norm_rot = torch.matmul(rotations, R_0_inv.unsqueeze(0))  # [T, N, 3, 3]
            
            # 相对于头部的归一化
            head_pos = norm_pos[:, HEAD_IDX:HEAD_IDX+1]  # [T, 1, 3]
            head_rot = norm_rot[:, HEAD_IDX:HEAD_IDX+1]  # [T, 1, 3, 3]
            head_rot_inv = torch.inverse(head_rot)       # [T, 1, 3, 3]
            
            rel_pos = norm_pos - head_pos  # [T, N, 3]
            rel_rot = torch.matmul(head_rot_inv, norm_rot)  # [T, N, 3, 3]
            
            return torch.cat([rel_accel, rel_gyro], dim=-1).reshape(T, -1), rel_pos, rel_rot
        
        # 组合归一化后的IMU数据
        return torch.cat([rel_accel, rel_gyro], dim=-1).reshape(T, -1)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        """获取单个窗口数据"""
        window = self.windows[idx]
        file_idx = window["file_idx"]
        start_idx = window["start_idx"]
        end_idx = window["end_idx"]
        
        # 加载序列数据
        data = torch.load(self.file_paths[file_idx])
        
        # 提取窗口数据
        seq_name = data["seq_name"]
        gender = data["gender"]
        
        # 提取全局位置和IMU数据
        position_global = data["position_global_full_gt_world"][start_idx:end_idx]  # [T, 22, 3]
        
        # 提取IMU数据
        imu_data = data["imu_global_full_gt"]
        imu_positions = imu_data["positions"][start_idx:end_idx]  # [T, 6, 3]
        imu_rotations = imu_data["rotations"][start_idx:end_idx]  # [T, 6, 3, 3]
        imu_accelerations = imu_data["accelerations"][start_idx:end_idx]  # [T, 6, 3]
        
        # 计算角速度
        angular_velocities = torch.zeros_like(imu_positions)  # [T, 6, 3]
        if end_idx - start_idx > 1:
            # 使用有限差分近似角速度
            delta_t = 1.0 / data["framerate"]
            for t in range(1, end_idx - start_idx):
                rel_rot = torch.matmul(imu_rotations[t], torch.inverse(imu_rotations[t-1]))
                angular_velocities[t] = transforms.matrix_to_axis_angle(rel_rot) / delta_t
        
        # 合并IMU数据
        combined_imu = torch.cat([imu_accelerations, angular_velocities], dim=-1)  # [T, 6, 6]
        combined_imu = combined_imu.reshape(end_idx - start_idx, -1)  # [T, 36]
        
        # 归一化IMU数据
        if self.normalize_imu:
            combined_imu, norm_positions, norm_rotations = self._normalize_imu_data(
                combined_imu, imu_positions, imu_rotations
            )
        
        # 提取物体数据 (如果存在)
        has_object = "obj_trans" in data
        obj_data = {}
        
        if has_object:
            obj_trans = data["obj_trans"][start_idx:end_idx]  # [T, 3]
            obj_rot_mat = data["obj_rot"][start_idx:end_idx]  # [T, 3, 3]
            obj_scale = data["obj_scale"][start_idx:end_idx]  # [T]
            
            # 计算物体IMU数据 (加速度和角速度)
            obj_accel = torch.zeros_like(obj_trans)  # [T, 3]
            obj_angular_vel = torch.zeros_like(obj_trans)  # [T, 3]
            
            if end_idx - start_idx > 2:
                # 二阶差分计算加速度
                delta_t = 1.0 / data["framerate"]
                for t in range(1, end_idx - start_idx - 1):
                    obj_accel[t] = (obj_trans[t+1] - 2*obj_trans[t] + obj_trans[t-1]) / (delta_t**2)
                
                # 有限差分计算角速度
                for t in range(1, end_idx - start_idx):
                    rel_rot = torch.matmul(obj_rot_mat[t], torch.inverse(obj_rot_mat[t-1]))
                    obj_angular_vel[t] = transforms.matrix_to_axis_angle(rel_rot) / delta_t
            
            # 将物体IMU数据添加到合并的IMU数据中
            obj_imu = torch.cat([obj_accel, obj_angular_vel], dim=-1)  # [T, 6]
            combined_imu = torch.cat([combined_imu, obj_imu], dim=-1)  # [T, 42]
            
            # 计算物体中心
            obj_center = torch.mean(obj_trans, dim=0)  # [3]
            
            # TODO: 物体BPS表示的计算应根据实际物体网格数据计算
            # 这里暂时使用随机值填充
            obj_bps = torch.randn(1024, 3, device=self.device)  # [1024, 3]
            
            obj_data = {
                "obj_trans": obj_trans,
                "obj_rot_mat": obj_rot_mat,
                "obj_scale": obj_scale,
                "obj_bps": obj_bps.reshape(-1)  # [3072]
            }
        
        # 提取关节参数，用于SMPL姿态生成
        body_params = {
            "root_orient": data["body_parms_list"]["root_orient"][start_idx:end_idx],  # [T, 3]
            "pose_body": data["body_parms_list"]["pose_body"][start_idx:end_idx],  # [T, 63]
            "trans": data["body_parms_list"]["trans"][start_idx:end_idx],  # [T, 3]
            "betas": data["body_parms_list"]["betas"]  # [1, 16]
        }
        
        # 准备返回数据
        sample = {
            "seq_name": seq_name,
            "gender": gender,
            "imu_data": combined_imu,  # [T, 42] 包含6个人体IMU + 1个物体IMU
            "position_global": position_global,  # [T, 22, 3]
            "body_params": body_params,
            "window_info": {
                "file_idx": file_idx,
                "start_idx": start_idx,
                "end_idx": end_idx
            }
        }
        
        # 添加物体数据 (如果存在)
        if has_object:
            sample.update(obj_data)
        
        return sample 