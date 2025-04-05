import glob
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pytorch3d.transforms as transforms

# IMU关节索引
IMU_JOINTS = [20, 21, 15, 7, 8, 0]  # 左手、右手、头部、左脚、右脚、髋部
HEAD_IDX = 2  # 头部在IMU_JOINTS中的索引

class IMUDataset(Dataset):
    def __init__(self, data_dir, window_size=60, window_stride=30, normalize=True, debug=False):
        """
        IMU数据集
        Args:
            data_dir: 数据目录
            window_size: 窗口大小
            window_stride: 窗口步长
            normalize: 是否对数据进行标准化
            debug: 是否在调试模式
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.window_stride = window_stride
        self.normalize = normalize
        self.debug = debug
        
        # 查找并解析数据
        self.sequence_files = glob.glob(os.path.join(data_dir, "*.pt"))
        print(f"找到{len(self.sequence_files)}个序列文件")
        
        # 解析窗口
        self.windows = self._parse_windows()
        print(f"创建了{len(self.windows)}个窗口")
        
        # 检查BPS文件夹
        self.bps_dir = os.path.join(os.path.dirname(data_dir), "bps_features")
        self.use_bps = os.path.exists(self.bps_dir)
        if self.use_bps:
            print(f"使用BPS特征从 {self.bps_dir}")
        else:
            print("未找到BPS特征文件夹")
            
        # 调试模式下只使用一小部分数据
        if debug:
            self.windows = self.windows[:100]
            print(f"调试模式：使用{len(self.windows)}个窗口")

    def _parse_windows(self):
        """解析数据窗口"""
        windows = []
        for file_path in tqdm(self.sequence_files, desc="解析窗口"):
            try:
                # 加载序列数据
                seq_data = torch.load(file_path)
                motion = seq_data["rotation_local_full_gt_list"]
                seq_len = motion.shape[0]
                
                if motion is None or motion.shape[0] - 1 < self.window_size:
                    continue
                
                seq_name = os.path.basename(file_path).replace(".pt", "")
                
                # 创建滑动窗口
                for start_idx in range(1, seq_len - self.window_size + 1, self.window_stride):  # 不要第一帧
                    end_idx = start_idx + self.window_size
                    windows.append({
                        "seq_name": seq_name,
                        "file_path": file_path,
                        "start_idx": start_idx,
                        "end_idx": end_idx
                    })
                    
            except Exception as e:
                print(f"无法处理文件 {file_path}: {e}")
                
        return windows

    def _imu_TN(self, imu_data):
        """
        对第一帧进行归一化 - 所有IMU数据相对于第一帧
        
        Args:
            imu_data: IMU数据 [T, num_imus, 6]，对于物体时，num_imus=1
            
        Returns:
            norm_imu: 归一化的IMU数据 [T, num_imus, 6]
        """
        # 分离加速度和角速度
        accel = imu_data[..., :3]  # [T, num_imus, 3]
        gyro = imu_data[..., 3:]  # [T, num_imus, 3]
        
        # 对IMU第一帧进行归一化
        norm_accel = accel - accel[0:1]  # 减去第一帧
        norm_gyro = gyro - gyro[0:1]  # 减去第一帧
        
        # 重新组合IMU
        norm_imu = torch.cat([norm_accel, norm_gyro], dim=-1)  # [T, num_imus, 6]
        
        return norm_imu
    
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        """
        获取单个数据样本
        
        Returns:
            数据字典，包含:
                - seq_name: 序列名称
                - body_parms_list: 身体参数列表
                - rotation_local_full_gt_list: 局部旋转姿态数据
                - head_global_trans: 头部全局变换矩阵数据
                - position_global_full_gt_world: 世界坐标下的全身位置数据
                - imu_global_full_gt: 全身IMU数据
                - framerate: 帧率
                - gender: 性别
                - obj_name: 物体名称
                - obj_scale: 物体缩放因子 [seq]
                - obj_trans: 物体平移 [seq, 3] - 已应用缩放
                - obj_rot: 物体旋转 [seq, 3, 3]
                - obj_com_pos: 物体质心位置 [seq, 3]
                - obj_imu: 物体IMU数据 [seq, 6]
        """
        window = self.windows[idx]
        file_path = window["file_path"]
        start_idx = window["start_idx"]
        end_idx = window["end_idx"]
        
        try:
            # 加载原始数据
            seq_data = torch.load(file_path)
            
            # 提取并切片数据
            root_pos = seq_data["position_global_full_gt_world"][start_idx:end_idx, 0, :]   # [seq, 3]
            motion = seq_data["rotation_local_full_gt_list"][start_idx:end_idx]
            human_imu_acc = seq_data["imu_global_full_gt"]["accelerations"][start_idx:end_idx]
            human_imu_gyro = seq_data["imu_global_full_gt"]["angular_velocities"][start_idx:end_idx]
            human_imu = torch.cat([human_imu_acc, human_imu_gyro], dim=-1)  # [seq, num_imus, 6]
            
            # 处理物体数据
            has_object = "obj_trans" in seq_data
            if has_object:
                obj_name = seq_data["obj_name"]
                obj_trans = seq_data["obj_trans"][start_idx:end_idx].squeeze(-1)  # [seq, 3]
                obj_rot = seq_data["obj_rot"][start_idx:end_idx]  # [seq, 3, 3]
                obj_imu_acc = seq_data["obj_imu"]["accelerations"][start_idx:end_idx]  # [seq, 1, 3]
                obj_imu_gyro = seq_data["obj_imu"]["angular_velocities"][start_idx:end_idx]  # [seq, 1, 3]
                
                # 获取物体缩放因子并应用于物体平移和加速度数据
                obj_scale = seq_data["obj_scale"][start_idx:end_idx]  # [seq]
                # 应用缩放到物体平移
                obj_trans = torch.mul(obj_trans, obj_scale.view(-1, 1))
                # 应用缩放到物体IMU加速度 (只对加速度部分应用缩放)
                obj_imu_acc = torch.mul(obj_imu_acc, obj_scale.view(-1, 1, 1))
                obj_imu = torch.cat([obj_imu_acc, obj_imu_gyro], dim=-1)  # [seq, 1, 6]
                
            else:
                # 如果没有物体数据，使用零填充
                seq_len = motion.shape[0]
                obj_trans = torch.zeros(seq_len, 3)
                obj_rot = torch.eye(3).unsqueeze(0).repeat(seq_len, 1, 1)
                obj_imu = torch.zeros(seq_len, 1, 6)
            
            # 对IMU数据进行归一化（如果需要）
            if self.normalize:
                # 对imu归一化(输入)
                norm_human_imu = self._imu_TN(human_imu)
                norm_obj_imu = self._imu_TN(obj_imu)
                # head_global_pos_start = seq_data["head_global_trans"][start_idx:start_idx+1, :3, 3]
                # head_global_rot_start = seq_data["head_global_trans"][start_idx:start_idx+1, :3, :3]
                # head_rot_invert = head_global_rot_start.swapaxes(-2,-1)
                # # 对motion归一化(输出)
                # norm_motion = motion.clone()
                # root_rot = transforms.rotation_6d_to_matrix(norm_motion[:, :6])
                # norm_root_rot = head_rot_invert @ root_rot
                # norm_motion[:, :6] = transforms.matrix_to_rotation_6d(norm_root_rot)
                # norm_root_pos = root_pos - head_global_pos_start
                # # 对obj归一化(输出)
                # norm_obj_trans = obj_trans - head_global_pos_start
                # norm_obj_rot = head_rot_invert @ obj_rot


            
            # 加载BPS特征（如果有）
            bps_features = None
            if self.use_bps and has_object:
                seq_name = window["seq_name"]
                bps_path = os.path.join(self.bps_dir, f"{seq_name}.npy")
                if os.path.exists(bps_path):
                    try:
                        bps_data = np.load(bps_path)
                        bps_features = torch.from_numpy(bps_data).float()
                        
                        # 切片BPS特征与窗口匹配
                        if bps_features.shape[0] > end_idx:
                            bps_features = bps_features[start_idx:end_idx]
                    except Exception as e:
                        if self.debug:
                            print(f"无法加载BPS特征 {bps_path}: {e}")
            
            # 构建结果字典
            # result = {
            #     "root_pos": norm_root_pos.float(),
            #     "motion": norm_motion.float(),  # [seq, 132]
            #     "head_global_trans": seq_data["head_global_trans"][start_idx:end_idx],    # [seq, 4, 4]
            #     "human_imu": norm_human_imu.float(),  # [seq, num_imus, 6]
            #     "obj_imu": norm_obj_imu.float(),  # [seq, 1, 6]
            #     "obj_trans": norm_obj_trans.float(),  # [seq, 3]
            #     "obj_rot": norm_obj_rot.float(),  # [seq, 3, 3]
            #     "obj_name": obj_name,
            #     "has_object": has_object,
            # }
            result = {
                "root_pos": root_pos.float(),
                "motion": motion.float(),  # [seq, 132]
                "human_imu": norm_human_imu.float(),  # [seq, num_imus, 6]
                "obj_imu": norm_obj_imu.float(),  # [seq, 1, 6]
                "obj_trans": obj_trans.float(),  # [seq, 3]
                "obj_rot": obj_rot.float(),  # [seq, 3, 3]
                "obj_name": obj_name,
                "has_object": has_object,
            }
            
            if bps_features is not None:
                result["bps_features"] = bps_features
                
            return result
            
        except Exception as e:
            print(f"加载数据失败 {file_path}: {e}")
            seq_len = self.window_size
            return {
                "motion": torch.randn(seq_len, 132),
                "head_global_trans": torch.randn(seq_len, 4, 4),
                "human_imu": torch.randn(seq_len, len(IMU_JOINTS), 6),
                "obj_imu": torch.randn(seq_len, 1, 6),
                "obj_trans": torch.randn(seq_len, 3),
                "obj_rot": torch.eye(3).unsqueeze(0).repeat(seq_len, 1, 1),
                "obj_name": None,
                "has_object": False
            }
