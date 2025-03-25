import glob
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# IMU关节索引
IMU_JOINTS = [20, 21, 15, 7, 8, 0]  # 左手、右手、头部、左脚、右脚、髋部
HEAD_IDX = 2  # 头部在IMU_JOINTS中的索引

class IMUDataset(Dataset):
    def __init__(self, data_dir, window_size=60, window_stride=30, normalize=True,
                 normalize_style="first_frame", debug=False):
        """
        IMU数据集
        Args:
            data_dir: 数据目录
            window_size: 窗口大小
            window_stride: 窗口步长
            normalize: 是否对数据进行标准化
            normalize_style: 标准化方式，可选"first_frame"或"window"
            debug: 是否在调试模式
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.window_stride = window_stride
        self.normalize = normalize
        self.normalize_style = normalize_style
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
                
                if motion is None or motion.shape[0] < self.window_size:
                    continue
                
                seq_name = os.path.basename(file_path).replace(".pt", "")
                
                # 创建滑动窗口
                for start_idx in range(0, seq_len - self.window_size + 1, self.window_stride):
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

    def _normalize_imu_first_frame(self, human_imu, obj_imu):
        """
        对第一帧进行归一化 - 所有IMU数据相对于第一帧
        
        Args:
            human_imu: 人体IMU数据 [T, num_imus, 6]
            obj_imu: 物体IMU数据 [T, 6]
            
        Returns:
            norm_human_imu: 归一化的人体IMU数据 [T, num_imus, 6]
            norm_obj_imu: 归一化的物体IMU数据 [T, 6]
        """
        # 分离加速度和角速度
        human_accel = human_imu[..., :3]  # [T, num_imus, 3]
        human_gyro = human_imu[..., 3:]  # [T, num_imus, 3]
        
        # 对人体IMU第一帧进行归一化
        norm_human_accel = human_accel - human_accel[0:1]  # 减去第一帧
        norm_human_gyro = human_gyro - human_gyro[0:1]  # 减去第一帧
        
        # 重新组合人体IMU
        norm_human_imu = torch.cat([norm_human_accel, norm_human_gyro], dim=-1)  # [T, num_imus, 6]
        
        # 分离物体加速度和角速度
        obj_accel = obj_imu[..., :3]  # [T, 3]
        obj_gyro = obj_imu[..., 3:]  # [T, 3]
        
        # 对物体IMU第一帧进行归一化
        norm_obj_accel = obj_accel - obj_accel[0:1]  # 减去第一帧
        norm_obj_gyro = obj_gyro - obj_gyro[0:1]  # 减去第一帧
        
        # 重新组合物体IMU
        norm_obj_imu = torch.cat([norm_obj_accel, norm_obj_gyro], dim=-1)  # [T, 6]
        
        return norm_human_imu, norm_obj_imu
    
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
                - head_global_trans_list: 头部全局平移数据
                - position_global_full_gt_world: 世界坐标下的全身位置数据
                - imu_global_full_gt: 全身IMU数据
                - framerate: 帧率
                - gender: 性别
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
            motion = seq_data["rotation_local_full_gt_list"][start_idx:end_idx]
            human_imu_acc = seq_data["imu_global_full_gt"]["accelerations"][start_idx:end_idx]
            human_imu_gyro = seq_data["imu_global_full_gt"]["angular_velocities"][start_idx:end_idx]
            human_imu = torch.cat([human_imu_acc, human_imu_gyro], dim=-1)  # [seq, 6]
            
            # 处理物体数据
            has_object = "obj_trans" in seq_data
            if has_object:
                obj_trans = seq_data["obj_trans"][start_idx:end_idx].squeeze(-1)  # [seq, 3]
                obj_rot = seq_data["obj_rot"][start_idx:end_idx]  # [seq, 3, 3]
                obj_imu_acc = seq_data["obj_imu"]["accelerations"][start_idx:end_idx].squeeze(-1)  # [seq, 3]
                obj_imu_gyro = seq_data["obj_imu"]["angular_velocities"][start_idx:end_idx]  # [seq, 3]
                obj_imu = torch.cat([obj_imu_acc, obj_imu_gyro], dim=-1)  # [seq, 6]
                
                # 获取物体缩放因子并应用于物体平移和加速度数据
                obj_scale = seq_data["obj_scale"][start_idx:end_idx]  # [seq]
                
                # 应用缩放到物体平移
                obj_trans = torch.mul(obj_trans, obj_scale.view(-1, 1))
                
                # 应用缩放到物体IMU加速度 (只对加速度部分应用缩放)
                obj_imu_acc = torch.mul(obj_imu_acc, obj_scale.view(-1, 1))
                obj_imu = torch.cat([obj_imu_acc, obj_imu_gyro], dim=-1)  # [seq, 6]
                
            else:
                # 如果没有物体数据，使用零填充
                seq_len = motion.shape[0]
                obj_trans = torch.zeros(seq_len, 3)
                obj_rot = torch.eye(3).unsqueeze(0).repeat(seq_len, 1, 1)
                obj_imu = torch.zeros(seq_len, 6)
            
            # 对IMU数据进行归一化（如果需要）
            if self.normalize and self.normalize_style == "first_frame":
                human_imu, obj_imu = self._normalize_imu_first_frame(human_imu, obj_imu)
            
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
            result = {
                "motion": motion.float(),  # [seq, 132]
                "imu": human_imu.float(),  # [seq, num_imus, 6]
                "obj_imu": obj_imu.float(),  # [seq, 6]
                "obj_trans": obj_trans.float(),  # [seq, 3]
                "obj_rot": obj_rot.float(),  # [seq, 3, 3]
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
                "imu": torch.randn(seq_len, len(IMU_JOINTS), 6),
                "obj_imu": torch.randn(seq_len, 6),
                "obj_trans": torch.randn(seq_len, 3),
                "obj_rot": torch.eye(3).unsqueeze(0).repeat(seq_len, 1, 1),
                "has_object": False
            }


class TestIMUDataset(Dataset):
    def __init__(self, all_info, filename_list, bps_dir=None, normalization=True):
        """
        测试数据集
        Args:
            all_info: 所有序列的完整信息
            filename_list: 文件名列表
            bps_dir: BPS特征目录
            normalization: 是否对数据进行标准化
        """
        self.filename_list = filename_list
        self.all_info = all_info
        self.normalization = normalization
        self.use_bps = bps_dir is not None and os.path.exists(bps_dir)
        self.bps_dir = bps_dir
        
        # 提取数据
        self.motions = []
        self.human_imus = []
        self.body_params = []
        self.obj_data = []
        
        for i in all_info:
            # 人体姿态数据
            self.motions.append(i["rotation_local_full_gt_list"])
            self.body_params.append(i["body_parms_list"])
            
            # 处理人体IMU数据
            imu = i["imu_global_full_gt"]
            acc = imu["accelerations"]  # [T, num_imus, 3]
            gyro = imu["angular_velocities"]  # [T, num_imus, 3]
            
            # 合并加速度和角速度 [T, num_imus, 6]
            combined_human_imu = torch.cat([acc, gyro], dim=-1)
            self.human_imus.append(combined_human_imu)
            
            # 提取物体数据(如果存在)
            has_object = "obj_trans" in i
            obj_info = {}
            
            if has_object:
                obj_trans = i["obj_trans"]  # [T, 3]
                obj_rot = i["obj_rot"]  # [T, 3, 3]
                obj_scale = i["obj_scale"]  # [T] 或标量
                
                # 提取物体IMU数据(如果存在)
                if "obj_imu" in i:
                    obj_imu = i["obj_imu"]
                    obj_accel = obj_imu["accelerations"]  # [T, 3]
                    obj_angular_vel = obj_imu["angular_velocities"]  # [T, 3]
                    combined_obj_imu = torch.cat([obj_accel, obj_angular_vel], dim=-1)  # [T, 6]
                else:
                    # 如果没有物体IMU数据，使用零填充
                    T = combined_human_imu.shape[0]
                    combined_obj_imu = torch.zeros(T, 6)
                
                obj_info = {
                    "obj_trans": obj_trans,
                    "obj_rot": obj_rot,
                    "obj_scale": obj_scale,
                    "obj_imu": combined_obj_imu,
                    "has_object": True
                }
            else:
                # 如果没有物体数据，创建零填充
                T = combined_human_imu.shape[0]
                obj_info = {
                    "obj_trans": torch.zeros(T, 3),
                    "obj_rot": torch.eye(3).unsqueeze(0).repeat(T, 1, 1),
                    "obj_scale": torch.ones(T),
                    "obj_imu": torch.zeros(T, 6),
                    "has_object": False
                }
            
            self.obj_data.append(obj_info)

    def __len__(self):
        return len(self.motions)

    def _normalize_imu_first_frame(self, human_imu, obj_imu):
        """
        对第一帧进行归一化 - 所有IMU数据相对于第一帧
        
        Args:
            human_imu: 人体IMU数据 [T, num_imus, 6]
            obj_imu: 物体IMU数据 [T, 6]
            
        Returns:
            norm_human_imu: 归一化的人体IMU数据 [T, num_imus, 6]
            norm_obj_imu: 归一化的物体IMU数据 [T, 6]
        """
        # 分离加速度和角速度
        human_accel = human_imu[..., :3]  # [T, num_imus, 3]
        human_gyro = human_imu[..., 3:]  # [T, num_imus, 3]
        
        # 对人体IMU第一帧进行归一化
        norm_human_accel = human_accel - human_accel[0:1]  # 减去第一帧
        norm_human_gyro = human_gyro - human_gyro[0:1]  # 减去第一帧
        
        # 重新组合人体IMU
        norm_human_imu = torch.cat([norm_human_accel, norm_human_gyro], dim=-1)  # [T, num_imus, 6]
        
        # 分离物体加速度和角速度
        obj_accel = obj_imu[..., :3]  # [T, 3]
        obj_gyro = obj_imu[..., 3:]  # [T, 3]
        
        # 对物体IMU第一帧进行归一化
        norm_obj_accel = obj_accel - obj_accel[0:1]  # 减去第一帧
        norm_obj_gyro = obj_gyro - obj_gyro[0:1]  # 减去第一帧
        
        # 重新组合物体IMU
        norm_obj_imu = torch.cat([norm_obj_accel, norm_obj_gyro], dim=-1)  # [T, 6]
        
        return norm_human_imu, norm_obj_imu

    def __getitem__(self, idx):
        """
        获取单个测试样本
        """
        motion = self.motions[idx]  # [T, 132]
        human_imu = self.human_imus[idx]  # [T, num_imus, 6]
        body_param = self.body_params[idx]  # {root_orient:[N+1,3], pose_body:[N+1,63], trans:[N+1,3]}
        obj_info = self.obj_data[idx]
        filename = self.filename_list[idx]
        
        # 提取物体数据
        obj_trans = obj_info["obj_trans"].float()  # [T, 3]
        obj_rot = obj_info["obj_rot"].float()  # [T, 3, 3]
        obj_scale = obj_info["obj_scale"].float()  # [T] 或标量 
        obj_imu = obj_info["obj_imu"].float()  # [T, 6]
        has_object = obj_info["has_object"]
        
        # 对第一帧进行归一化（如果需要）
        if self.normalization:
            human_imu, obj_imu = self._normalize_imu_first_frame(human_imu, obj_imu)
        
        # 加载BPS特征（如果有）
        bps_features = None
        if self.use_bps and has_object and 'bps_file' in self.all_info[idx]:
            bps_path = os.path.join(self.bps_dir, self.all_info[idx]['bps_file'])
            if os.path.exists(bps_path):
                try:
                    bps_features = np.load(bps_path)
                    bps_features = torch.from_numpy(bps_features).float()
                except Exception as e:
                    print(f"无法加载BPS特征 {bps_path}: {e}")
        
        # 返回结果
        result = {
            "motion": motion,  # [T, 132]
            "human_imu": human_imu,  # [T, num_imus, 6]
            "obj_trans": obj_trans,  # [T, 3]
            "obj_rot": obj_rot,  # [T, 3, 3]
            "obj_scale": obj_scale,  # [T] 或标量
            "obj_imu": obj_imu,  # [T, 6]
            "body_param": body_param,
            "has_object": has_object,
            "bps_features": bps_features,  # 可能为None
            "filename": filename
        }
        
        return result