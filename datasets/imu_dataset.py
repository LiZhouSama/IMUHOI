import os
import torch
import numpy as np
import glob
from torch.utils.data import Dataset
from utils.data_utils import convert_rotation_matrix_to_axis_angle

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
        device='cpu'        # 默认使用CPU - 数据加载器需要pin_memory
    ):
        super().__init__()
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.normalize_imu = normalize_imu
        self.mode = mode
        self.device = device  # 保持在CPU上用于数据加载
        
        # 加载所有pt文件
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        print(f"加载了 {len(self.file_paths)} 个序列文件")
        
        # 解析窗口
        self.windows = self._parse_windows()
        print(f"总共创建了 {len(self.windows)} 个窗口")
        
        # BPS目录
        self.bps_dir = os.path.join(os.path.dirname(data_dir), "bps_features")
        if not os.path.exists(self.bps_dir):
            print(f"警告：BPS目录 {self.bps_dir} 不存在，将使用默认BPS特征")
        
        # 预加载缓存
        self.data_cache = {}
        
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
    
    def _normalize_imu_first_frame(self, imu_data):
        """
        归一化IMU数据 - 只对第一帧进行归一化
        
        参数:
            imu_data: IMU加速度和角速度 [T, N*6]
            
        返回:
            norm_imu_data: 归一化后的IMU数据 [T, N*6]
        """
        T, D = imu_data.shape
        N = D // 6
        
        # 重塑IMU数据为加速度和角速度分量
        imu_data = imu_data.reshape(T, N, 6)
        accel = imu_data[..., :3]  # [T, N, 3]
        gyro = imu_data[..., 3:]   # [T, N, 3]
        
        # 相对于第一帧的归一化
        norm_accel = accel - accel[0:1]  # [T, N, 3]
        norm_gyro = gyro - gyro[0:1]     # [T, N, 3]
        
        # 重新组合
        return torch.cat([norm_accel, norm_gyro], dim=-1).reshape(T, -1)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        """获取单个窗口数据 - 保持在CPU上，避免pin_memory错误"""
        try:
            # 检查缓存
            if idx in self.data_cache:
                return self.data_cache[idx]
                
            window = self.windows[idx]
            file_idx = window["file_idx"]
            start_idx = window["start_idx"]
            end_idx = window["end_idx"]
            
            # 加载序列数据
            file_path = self.file_paths[file_idx]
            data = torch.load(file_path)
            
            # 提取窗口数据
            seq_name = data["seq_name"]
            gender = data["gender"]
            
            # 确保为字符串
            if not isinstance(seq_name, str):
                seq_name = str(seq_name)
            if not isinstance(gender, str):
                gender = str(gender)
                
            # 提取全局位置 - 保持在CPU上
            position_global = data["position_global_full_gt_world"][start_idx:end_idx]  # [T, 22, 3]
            
            # 提取IMU数据 - 保持在CPU上
            imu_data = data["imu_global_full_gt"]
            imu_accelerations = imu_data["accelerations"][start_idx:end_idx]  # [T, 6, 3]
            imu_angular_velocities = imu_data["angular_velocities"][start_idx:end_idx]  # [T, 6, 3]
            
            # 合并IMU数据 - 已经是归一化到头部坐标系的
            T = end_idx - start_idx
            combined_imu = torch.cat([imu_accelerations, imu_angular_velocities], dim=-1)  # [T, 6, 6]
            combined_imu = combined_imu.reshape(T, -1)  # [T, 36]
            
            # 对第一帧进行归一化处理（如果需要）
            if self.normalize_imu:
                combined_imu = self._normalize_imu_first_frame(combined_imu)
            
            # 提取物体数据 (如果存在) - 在CPU上
            has_object = "obj_trans" in data
            obj_data = {}
            
            if has_object:
                obj_trans = data["obj_trans"][start_idx:end_idx]  # [T, 3]
                obj_rot_mat = data["obj_rot"][start_idx:end_idx]  # [T, 3, 3]
                obj_scale = data["obj_scale"][start_idx:end_idx]  # [T]
                
                # 加载预计算的物体IMU数据
                obj_imu = data.get("obj_imu", None)
                if obj_imu:
                    obj_accel = obj_imu["accelerations"][start_idx:end_idx]  # [T, 3]
                    obj_angular_vel = obj_imu["angular_velocities"][start_idx:end_idx]  # [T, 3] 
                    
                    # 归一化处理 - 对第一帧
                    if self.normalize_imu:
                        obj_accel = obj_accel - obj_accel[0:1]
                        obj_angular_vel = obj_angular_vel - obj_angular_vel[0:1]
                else:
                    # 如果没有预计算的物体IMU数据，则使用零填充
                    obj_accel = torch.zeros_like(obj_trans)
                    obj_angular_vel = torch.zeros_like(obj_trans)
                
                # 将物体IMU数据添加到合并的IMU数据中
                obj_imu_combined = torch.cat([obj_accel, obj_angular_vel], dim=-1)  # [T, 6]
                combined_imu = torch.cat([combined_imu, obj_imu_combined], dim=-1)  # [T, 42]
                
                # 加载BPS特征
                bps_file = data.get("bps_file", None)
                if bps_file and os.path.exists(os.path.join(self.bps_dir, bps_file)):
                    # 加载预计算的BPS特征
                    bps_features = np.load(os.path.join(self.bps_dir, bps_file))
                    obj_bps = torch.from_numpy(bps_features[start_idx:end_idx]).float()
                else:
                    # 如果没有找到BPS文件，使用随机特征
                    print(f"警告：未找到BPS文件 {bps_file}，使用随机特征")
                    obj_bps = torch.randn(T, 1024, 3)
                
                # 展平BPS特征
                obj_bps = obj_bps.reshape(T, -1)  # [T, 3072]
                
                obj_data = {
                    "obj_trans": obj_trans,
                    "obj_rot_mat": obj_rot_mat,
                    "obj_scale": obj_scale,
                    "obj_bps": obj_bps
                }
            
            # 提取关节参数，用于SMPL姿态生成 - 保持在CPU上
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
                    "end_idx": end_idx,
                    "file_path": os.path.basename(file_path)
                }
            }
            
            # 添加物体数据 (如果存在)
            if has_object:
                sample.update(obj_data)
            
            # 缓存结果
            self.data_cache[idx] = sample
            
            return sample
        except Exception as e:
            print(f"获取数据项 {idx} 时出错: {e}")
            # 返回一个空的样本，避免崩溃 - 在CPU上
            return {
                "seq_name": "error",
                "gender": "error",
                "imu_data": torch.zeros(self.window_size, 36),
                "position_global": torch.zeros(self.window_size, 22, 3),
                "body_params": {
                    "root_orient": torch.zeros(self.window_size, 3),
                    "pose_body": torch.zeros(self.window_size, 63),
                    "trans": torch.zeros(self.window_size, 3),
                    "betas": torch.zeros(1, 16)
                },
                "window_info": {
                    "file_idx": -1,
                    "start_idx": -1,
                    "end_idx": -1,
                    "file_path": "error.pt"
                }
            } 