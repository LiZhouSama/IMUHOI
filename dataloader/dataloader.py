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
            imu_data: IMU数据 [T, num_imus, 9]，包含加速度(3维)和6D旋转表示(6维)
            
        Returns:
            norm_imu: 归一化的IMU数据 [T, num_imus, 9]
        """
        if imu_data is None or imu_data.nelement() == 0:
            return imu_data # 如果输入为空则返回
        
        # 分离加速度和方向
        accel = imu_data[..., :3]  # [T, num_imus, 3]
        rot6d = imu_data[..., 3:]  # [T, num_imus, 6]
        
        T, num_imus, _ = accel.shape
        
        # 对加速度第一帧进行归一化
        norm_accel = accel - accel[0:1]  # 减去第一帧 [T, num_imus, 3]
        
        # 对6D旋转表示进行归一化
        norm_rot6d = torch.zeros_like(rot6d)  # [T, num_imus, 6]
        
        # 遍历每个IMU传感器
        for i in range(num_imus):
            # 将6D旋转表示转换回旋转矩阵
            rot_matrices = transforms.rotation_6d_to_matrix(rot6d[:, i])  # [T, 3, 3]
            
            # 获取第一帧的旋转矩阵并计算其逆
            first_orient = rot_matrices[0]  # [3, 3]
            first_orient_inv = torch.inverse(first_orient)  # [3, 3]
            
            # 计算每一帧相对于第一帧的旋转
            rel_rotations = torch.matmul(first_orient_inv.unsqueeze(0), rot_matrices)  # [T, 3, 3]
            
            # 将结果转换为6D旋转表示
            norm_rot6d[:, i, :] = transforms.matrix_to_rotation_6d(rel_rotations)  # [T, 6]
        
        # 重新组合IMU数据
        norm_imu = torch.cat([norm_accel, norm_rot6d], dim=-1)  # [T, num_imus, 9]
        
        return norm_imu
    
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        """
        获取单个数据样本
        
        Returns:
            数据字典，包含:
                - seq_name: 序列名称
                - root_pos: 根关节位置 [seq, 3]
                - motion: 局部旋转姿态数据 (6D) [seq, 132]
                - human_imu: 归一化后的人体IMU数据 [seq, num_imus, 9]，包含加速度(3D)和6D旋转表示(6D)
                - obj_imu: 归一化后的物体IMU数据 [seq, 1, 9]，包含加速度(3D)和6D旋转表示(6D)
                - obj_trans: 物体平移 [seq, 3] (未应用缩放)
                - obj_rot: 物体旋转 [seq, 3, 3]
                - obj_scale: 物体缩放因子 [seq]
                - obj_name: 物体名称
                - has_object: 是否有物体
                - (可选) bps_features: BPS特征
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
            human_imu_ori = seq_data["imu_global_full_gt"]["orientations"][start_idx:end_idx]
            seq_len = motion.shape[0]

            # 将旋转矩阵转换为6D旋转表示
            human_imu_ori_flat = human_imu_ori.reshape(seq_len, -1, 3, 3)
            human_imu_ori_6d = transforms.matrix_to_rotation_6d(human_imu_ori_flat)
            human_imu = torch.cat([human_imu_acc, human_imu_ori_6d], dim=-1)  # [T, num_imus, 9]
            
            # 处理物体数据
            has_object = "obj_trans" in seq_data and seq_data["obj_trans"] is not None
            obj_name = None # Default value
            obj_trans = torch.zeros(motion.shape[0], 3) # Default value
            obj_rot = torch.eye(3).unsqueeze(0).repeat(motion.shape[0], 1, 1) # Default value
            obj_scale = torch.ones(motion.shape[0]) # Default value
            obj_imu = torch.zeros(motion.shape[0], 1, 9) # Default value - 现在是9D (3D加速度 + 6D旋转)

            if has_object:
                obj_name = seq_data.get("obj_name", "unknown_object") # Use get for safety
                # --- 加载原始 trans, rot, scale ---
                obj_trans = seq_data["obj_trans"][start_idx:end_idx].squeeze(-1)  # [seq, 3] (保持未缩放)
                obj_rot = seq_data["obj_rot"][start_idx:end_idx]  # [seq, 3, 3]
                obj_scale = seq_data["obj_scale"][start_idx:end_idx]  # [seq] (单独加载)
                # --- 结束修改 ---

                obj_imu_acc = seq_data.get("obj_imu", {}).get("accelerations", torch.zeros(motion.shape[0], 1, 3))[start_idx:end_idx] # [seq, 1, 3]
                obj_imu_ori = seq_data.get("obj_imu", {}).get("orientations", torch.zeros(motion.shape[0], 1, 3, 3))[start_idx:end_idx] # [seq, 1, 3, 3]
                
                # 将旋转矩阵转换为6D旋转表示
                obj_imu_ori_6d = transforms.matrix_to_rotation_6d(obj_imu_ori)
                # 将加速度(3D)和6D旋转表示(6D)拼接
                obj_imu = torch.cat([obj_imu_acc, obj_imu_ori_6d], dim=-1)  # [seq, 1, 9]
                
            else:
                # 如果没有物体数据，使用上面定义的默认值
                pass
            
            # 对IMU数据进行归一化（如果需要）
            norm_human_imu = human_imu.float()
            norm_obj_imu = obj_imu.float()
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
                         # 切片 BPS 特征
                         # 注意：原始 OMOMO BPS 是按窗口保存的，这里假设是按完整序列保存
                         # 如果 BPS 文件已经是窗口化的，可能不需要切片
                         if bps_features.shape[0] == seq_data["rotation_local_full_gt_list"].shape[0]: # 假设 BPS 与完整序列对齐
                             bps_features = bps_features[start_idx:end_idx]
                         elif bps_features.shape[0] != motion.shape[0]: # 如果长度不匹配窗口
                             print(f"警告：BPS 特征长度 ({bps_features.shape[0]}) 与窗口 ({motion.shape[0]}) 不匹配，路径：{bps_path}")
                             bps_features = None # 忽略不匹配的 BPS
                     except Exception as e:
                         if self.debug:
                             print(f"无法加载或切片 BPS 特征 {bps_path}: {e}")
                         bps_features = None # 失败时置空


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
                "human_imu": norm_human_imu.float(),  # [seq, num_imus, 9] - 现在是9D (3D加速度 + 6D旋转)
                "obj_imu": norm_obj_imu.float(),  # [seq, 1, 9] - 现在是9D (3D加速度 + 6D旋转)
                "obj_trans": obj_trans.float(),  # [seq, 3] (未缩放)
                "obj_rot": transforms.matrix_to_rotation_6d(obj_rot).float(),  # [seq, 6]
                "obj_scale": obj_scale.float(), # [seq] (单独返回)
                "obj_name": obj_name,
                "has_object": has_object,
            }
            
            if bps_features is not None:
                result["bps_features"] = bps_features
                
            return result
            
        except Exception as e:
            print(f"加载数据失败 {file_path} at index {idx}: {e}")
            import traceback
            traceback.print_exc() # 打印详细错误
            seq_len = self.window_size
            # 返回包含正确键但值为默认值的字典，以避免后续代码出错
            return {
                "root_pos": torch.zeros(seq_len, 3),
                "motion": torch.zeros(seq_len, 132),
                "human_imu": torch.zeros(seq_len, len(IMU_JOINTS), 9),  # 现在是9D (3D加速度 + 6D旋转)
                "obj_imu": torch.zeros(seq_len, 1, 9),  # 现在是9D (3D加速度 + 6D旋转)
                "obj_trans": torch.zeros(seq_len, 3),
                "obj_rot": transforms.matrix_to_rotation_6d(torch.eye(3).unsqueeze(0).repeat(seq_len, 1, 1)).float(),  # [seq, 6] - 6D旋转表示
                "obj_scale": torch.ones(seq_len),
                "obj_name": None,
                "has_object": False
            }
