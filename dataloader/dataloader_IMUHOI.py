import glob
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pytorch3d.transforms as transforms
import random # Import random for shuffling sequence info in debug mode

from configs.global_config_IMUHOI import FRAME_RATE, _SENSOR_POS_INDICES, _SENSOR_ROT_INDICES, _VEL_SELECTION_INDICES,\
     _REDUCED_INDICES, _IGNORED_INDICES, _SENSOR_NAMES, _SENSOR_VEL_NAMES, _REDUCED_POSE_NAMES

# --- IMU 计算函数 ---
def _central_diff(tensor: torch.Tensor, dt: float) -> torch.Tensor:
    """使用中心差分计算一阶导数，支持任意最后一维。"""
    if tensor.shape[0] <= 1:
        return torch.zeros_like(tensor)
    vel = torch.zeros_like(tensor)
    vel[1:-1] = (tensor[2:] - tensor[:-2]) / (2.0 * dt)
    vel[0] = (tensor[1] - tensor[0]) / dt
    vel[-1] = (tensor[-1] - tensor[-2]) / dt
    return vel


def _smooth_acceleration(position: torch.Tensor, fps: float, smooth_n: int = 4) -> torch.Tensor:
    """
    对位置序列计算平滑的加速度，参考 TransPose 的平滑策略。
    
    Args:
        position: 位置序列 [T, ..., 3] 或 [T, 3]
        fps: 帧率
        smooth_n: 平滑窗口大小（默认4）
    
    Returns:
        平滑后的加速度 [T, ..., 3] 或 [T, 3]
    """
    if position.shape[0] <= 2:
        return torch.zeros_like(position)
    
    # 保存原始形状
    original_shape = position.shape
    # 重塑为 [T, -1] 以便处理任意维度
    pos_flat = position.reshape(position.shape[0], -1)
    
    mid = smooth_n // 2
    dt = 1.0 / fps
    fps_squared = fps * fps  # 对应参考代码中的 3600 (60^2)
    
    # 初始化加速度：使用标准的二阶差分
    acc = torch.stack([
        (pos_flat[i] + pos_flat[i + 2] - 2 * pos_flat[i + 1]) * fps_squared 
        for i in range(0, pos_flat.shape[0] - 2)
    ])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    
    # 对中间部分使用更大的平滑窗口
    if mid != 0 and acc.shape[0] > smooth_n * 2:
        acc[smooth_n:-smooth_n] = torch.stack([
            (pos_flat[i] + pos_flat[i + smooth_n * 2] - 2 * pos_flat[i + smooth_n]) * fps_squared / (smooth_n ** 2)
            for i in range(0, pos_flat.shape[0] - smooth_n * 2)
        ])
    
    # 恢复原始形状
    acc = acc.reshape(original_shape)
    return acc




def _build_human_imu_root(position_global: torch.Tensor,
                     rotation_global: torch.Tensor,
                     imu_joints_pos,
                     imu_joints_rot,
                     fps: float) -> torch.Tensor:
    """利用关节位置和姿态构建 DynaIP 风格的 IMU 特征。"""
    dt = 1.0 / fps
    pos = position_global[:, imu_joints_pos, :]
    rot = rotation_global[:, imu_joints_rot, :, :]
    vel = _central_diff(pos, dt)
    # 使用平滑加速度函数代替二次差分
    acc = _smooth_acceleration(pos, fps, smooth_n=4)
    root_imu_ori = rot[:, 0]
    imu_acc_norm = torch.cat((acc[:, :1], acc[:, 1:] - acc[:, :1]), dim=1).bmm(root_imu_ori)
    imu_ori_norm = torch.cat((rot[:, :1], rot[:, :1].transpose(2, 3).matmul(rot[:, 1:])), dim=1)
    imu_ori_norm_6d = transforms.matrix_to_rotation_6d(imu_ori_norm)
    imu = torch.cat((imu_acc_norm, imu_ori_norm_6d), dim=-1)
    return imu  # [T, 6, 9]


def _compute_joint_velocity_root(rotation_global: torch.Tensor,
                                 position_global: torch.Tensor,
                                 fps: float):
    """计算根坐标系下的关节速度和位置。"""
    pos = position_global.clone()
    root_pos = position_global[:, :1, :].clone()
    pos[:, :, 0] = pos[:, :, 0] - root_pos[:, :, 0]
    pos[:, :, 2] = pos[:, :, 2] - root_pos[:, :, 2]

    vel_world = _central_diff(pos, 1.0 / fps)
    root_vel = vel_world[:, :1]
    rel_vel = torch.cat((root_vel, vel_world[:, 1:] - root_vel), dim=1)

    root_R = rotation_global[:, 0]
    vel_root = rel_vel.bmm(root_R)
    return vel_root

def _build_object_imu(obj_rot: torch.Tensor, obj_pos: torch.Tensor, fps: float) -> torch.Tensor:
    """计算物体的 IMU 特征（姿态 + 加速度）。"""
    if obj_rot.shape[-1] != 6:
        ori_obj_6d = transforms.matrix_to_rotation_6d(obj_rot.reshape(-1, 3, 3))
    else:
        ori_obj_6d = obj_rot
    vel_obj = _central_diff(obj_pos, 1.0 / fps)
    # 使用平滑加速度函数代替二次差分
    acc_obj = _smooth_acceleration(obj_pos, fps, smooth_n=4)
    return torch.cat([acc_obj, ori_obj_6d], dim=-1)   # [T, 9]

# --- 结束 IMU 计算函数 ---

def find_contact_segments(contact_mask):
    """找到连续的接触段"""
    segments = []
    contact_indices = torch.where(contact_mask)[0]
    
    if len(contact_indices) == 0:
        return segments
    
    start = contact_indices[0].item()
    end = start
    
    for i in range(1, len(contact_indices)):
        if contact_indices[i] == contact_indices[i-1] + 1:
            end = contact_indices[i].item()
        else:
            segments.append((start, end))
            start = contact_indices[i].item()
            end = start
    
    segments.append((start, end))
    return segments

def compute_obj_direction_supervision(position_global, obj_trans, obj_rot, 
                                    lhand_contact, rhand_contact):
    """
    计算物体坐标系下的方向向量监督标签
    Returns:
        dict: 包含物体坐标系下的方向向量
    """
    device = position_global.device
    seq_len = obj_trans.shape[0]
    
    # 将6D旋转表示转换为旋转矩阵
    obj_rot_mat = transforms.rotation_6d_to_matrix(obj_rot)  # [seq, 3, 3]
    
    # 初始化结果
    result = {
        "lhand_obj_direction": torch.zeros(seq_len, 3, device=device),
        "rhand_obj_direction": torch.zeros(seq_len, 3, device=device),
    }
    
    # 提取手腕位置
    wrist_pos = {
        'left': position_global[:, 20, :],   # [seq, 3] - 左手腕位置（关节20）
        'right': position_global[:, 21, :]  # [seq, 3] - 右手腕位置（关节21）
    }
    
    contacts = {'left': lhand_contact, 'right': rhand_contact}
    
    for hand_name in ['left', 'right']:
        contact_mask = contacts[hand_name]
        
        if not contact_mask.any():
            continue
        
        # 找到接触段
        contact_segments = find_contact_segments(contact_mask)
        
        for seg_start, seg_end in contact_segments:
            # 获取接触段的索引
            seg_indices = torch.arange(seg_start, seg_end + 1, device=obj_trans.device)
            
            # 批量获取接触段的数据
            wrist_pos_seg = wrist_pos[hand_name][seg_indices]  # [seg_len, 3]
            obj_trans_seg = obj_trans[seg_indices]  # [seg_len, 3]
            obj_rot_mat_seg = obj_rot_mat[seg_indices]  # [seg_len, 3, 3]
            
            # 1. 计算世界坐标系下手腕指向物体的向量
            v_HO_world = obj_trans_seg - wrist_pos_seg  # [seg_len, 3]
            
            # 2. 归一化得到世界坐标系下的单位向量
            v_HO_world_unit = v_HO_world / (torch.norm(v_HO_world, dim=1, keepdim=True) + 1e-8)  # [seg_len, 3]
            
            # 3. 转换到物体坐标系：^Ov_{HO} = ^WR_O^T * ^Wv_{HO}
            obj_rot_mat_inv_seg = obj_rot_mat_seg.transpose(-1, -2)  # [seg_len, 3, 3]
            v_HO_obj = torch.bmm(obj_rot_mat_inv_seg, v_HO_world_unit.unsqueeze(-1)).squeeze(-1)  # [seg_len, 3]
            
            # 4. 存储结果
            if hand_name == 'left':
                result["lhand_obj_direction"][seg_indices] = v_HO_obj
            else:
                result["rhand_obj_direction"][seg_indices] = v_HO_obj
    
    return result

class IMUDataset(Dataset):
    def __init__(self, data_dir, window_size=60, debug=False, min_obj_contact_frames=10, full_sequence=False):
        """
        IMU数据集 - 每个epoch为每个序列随机采样一个窗口
        Args:
            data_dir: 数据目录，可以是字符串（单个目录）或列表（多个目录）
            window_size: 窗口大小
            debug: 是否在调试模式
            min_obj_contact_frames: 序列中至少需要的物体接触帧数，少于此值的序列将被过滤掉
        """
        # 支持单个目录或多个目录
        if isinstance(data_dir, str):
            self.data_dirs = [data_dir]
        elif isinstance(data_dir, list):
            self.data_dirs = data_dir
        else:
            raise ValueError("data_dir must be a string or a list of strings")
        
        self.window_size = window_size
        self.debug = debug
        self.min_obj_contact_frames = min_obj_contact_frames
        self.full_sequence = full_sequence

        # 查找所有目录中的序列文件
        self.sequence_files = []
        for data_dir_path in self.data_dirs:
            if os.path.exists(data_dir_path):
                dir_files = glob.glob(os.path.join(data_dir_path, "*.pt"))
                self.sequence_files.extend(dir_files)
                print(f"从目录 {data_dir_path} 找到 {len(dir_files)} 个序列文件")
            else:
                print(f"警告: 目录 {data_dir_path} 不存在，跳过")
        
        print(f"总共找到 {len(self.sequence_files)} 个序列文件")

        # 初始化用于存储预加载数据和序列信息的容器
        self.loaded_data = {}
        self.sequence_info = [] # Store sequence metadata: {'file_path': ..., 'seq_name': ..., 'seq_len': ...}

        # 执行加载、共享和序列信息收集
        self._load_share_and_collect_info()
        print(f"预加载并收集信息完成，共找到{len(self.sequence_info)}个有效序列")
        if self.full_sequence:
            print(f"过滤条件：整段模式启用，序列长度 >= 1，物体接触帧数 >= {self.min_obj_contact_frames}")
        else:
            print(f"过滤条件：序列长度 >= {self.window_size + 1}，物体接触帧数 >= {self.min_obj_contact_frames}")

        # 检查BPS文件夹 - 对于多个数据目录，检查第一个目录的BPS文件夹
        self.bps_dir = os.path.join(os.path.dirname(self.data_dirs[0]), "bps_features")
        self.use_bps = os.path.exists(self.bps_dir)
        if self.use_bps:
            print(f"使用BPS特征从 {self.bps_dir}")
        else:
            print("未找到BPS特征文件夹")

        # 调试模式下只使用一小部分序列
        if debug and len(self.sequence_info) > 100:
            # 只缩减 sequence_info 列表
            random.shuffle(self.sequence_info) # Shuffle before taking a subset
            self.sequence_info = self.sequence_info[:100]
            print(f"调试模式：使用{len(self.sequence_info)}个序列")
        elif len(self.sequence_info) == 0:
             print("警告：没有找到有效的序列，数据集为空。请检查数据和窗口参数。")

    def _load_share_and_collect_info(self):
        """加载所有序列数据，将其Tensor移动到共享内存，并收集序列信息"""
        print("开始预加载、共享内存处理和序列信息收集...")
        for file_path in tqdm(self.sequence_files, desc="预加载和收集信息"):
            try:
                # 加载序列数据
                seq_data = torch.load(file_path)
                seq_name = os.path.basename(file_path).replace(".pt", "")

                # 检查基本数据是否存在
                if seq_data is None or "rotation_local_full_gt_list" not in seq_data or seq_data["rotation_local_full_gt_list"] is None:
                    print(f"警告：跳过文件 {file_path}，缺少必要的 'rotation_local_full_gt_list' 数据。")
                    continue

                motion = seq_data["rotation_local_full_gt_list"]
                seq_len = motion.shape[0]

                # 检查序列长度是否足够
                if self.full_sequence:
                    # 整段模式：仅需至少1帧即可
                    if seq_len < 1:
                        continue
                else:
                    # 窗口模式：需要能采样到一个窗口（从索引1开始）
                    if seq_len < self.window_size + 1:
                        # print(f"调试：跳过文件 {file_path}，序列长度 {seq_len} 不足以创建大小为 {self.window_size} 的窗口。")
                        continue

                # 检查物体接触帧数是否足够
                if "obj_contact" in seq_data and seq_data["obj_contact"] is not None:
                    obj_contact_frames = seq_data["obj_contact"]
                    if isinstance(obj_contact_frames, torch.Tensor):
                        contact_count = obj_contact_frames.sum().item()
                    else:
                        contact_count = np.sum(obj_contact_frames)
                    
                    if contact_count < self.min_obj_contact_frames:
                        # print(f"跳过文件 {file_path}，物体接触帧数 {contact_count} 少于最小要求 {self.min_obj_contact_frames}")
                        continue

                # 将所有Tensor移动到共享内存
                for key, value in seq_data.items():
                    if isinstance(value, torch.Tensor):
                        value.share_memory_()

                # 存储预加载的数据 (使用 file_path 作为 key)
                self.loaded_data[file_path] = seq_data

                # -- 收集序列信息 --
                self.sequence_info.append({
                    "file_path": file_path,
                    "seq_name": seq_name,
                    "seq_len": seq_len
                })

            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                import traceback
                traceback.print_exc() # 打印更详细的错误追踪

    def cleanup(self):
        """清理共享内存和缓存数据"""
        print(f"清理IMUDataset，释放{len(self.loaded_data)}个序列的共享内存...")
        self.loaded_data.clear()
        self.sequence_info.clear()
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        print("IMUDataset清理完成")

    def __del__(self):
        """析构函数，确保在对象被删除时清理共享内存"""
        if hasattr(self, 'loaded_data') and self.loaded_data:
            self.cleanup()

    def __len__(self):
        # 返回独立序列的数量
        return len(self.sequence_info)

    def __getitem__(self, idx):
        """
        获取单个数据样本 (从指定序列中随机采样一个窗口)

        Args:
            idx: sequence_info 列表中的索引

        Returns:
            数据字典
        """
        # 1. 获取序列信息
        try:
            seq_info = self.sequence_info[idx]
            file_path = seq_info["file_path"]
            seq_name = seq_info["seq_name"]
            seq_len = seq_info["seq_len"]
        except IndexError:
             print(f"错误：索引 {idx} 超出 sequence_info 范围 (大小: {len(self.sequence_info)})")
             raise IndexError(f"索引 {idx} 超出范围")


        # 2. 选择切片范围
        if self.full_sequence:
            # 整段模式：不裁剪，直接使用全长度
            start_idx = 0
            end_idx = seq_len
        else:
            # 随机生成 start_idx（窗口模式）
            # 有效 start_idx 范围：[1, seq_len - window_size] (包含两端)
            max_start_idx = seq_len - self.window_size
            if max_start_idx < 1:
                # 这种情况理论上在 __init__ 中被过滤掉了，但为了安全起见
                print(f"错误：序列 {seq_name} (长度 {seq_len}) 过短，无法采样窗口大小 {self.window_size}")
                raise ValueError(f"序列 {seq_name} 长度不足")
            start_idx = torch.randint(1, max_start_idx + 1, (1,)).item()
            # start_idx = 1
            end_idx = start_idx + self.window_size # 切片时使用 end_idx

        # 3. 从预加载数据中获取序列数据
        try:
             seq_data = self.loaded_data[file_path]
        except KeyError:
             print(f"错误：无法在预加载数据中找到文件路径 {file_path} 对应的键。序列索引: {idx}")
             raise KeyError(f"无法找到文件 {file_path}")

        # 4. 切片和处理数据
        try:
            # --- 从预加载的 seq_data 中取出窗口数据 ---
            # 注意：无需再次 torch.load(file_path)

            trans = seq_data["trans"][start_idx:end_idx, :]   # [seq, 3]
            if trans.dtype == np.float32:
                trans = torch.from_numpy(trans).float()
            
            # 获取实际的序列长度（考虑full_sequence模式）
            actual_seq_len = end_idx - start_idx
            
            pose = transforms.matrix_to_axis_angle(transforms.rotation_6d_to_matrix(
                seq_data["rotation_local_full_gt_list"].reshape(seq_len, -1, 6)[start_idx:end_idx])).reshape(actual_seq_len, -1) # [seq, 66]
            position_global_full = seq_data["position_global_full_gt_world"][start_idx:end_idx]  # [seq, J, 3]
            rotation_global_full = seq_data["rotation_global"][start_idx:end_idx]  # [seq, J, 3, 3]
            ori_glb_reduced = rotation_global_full[:, _REDUCED_INDICES] # [seq, n_reduced, 3, 3]
            ori_root_reduced = rotation_global_full[:, :1].transpose(-1, -2).matmul(ori_glb_reduced)

            # --- DynaIP 风格的人体 IMU 与速度处理 ---
            human_imu = _build_human_imu_root(position_global_full, rotation_global_full, _SENSOR_POS_INDICES, _SENSOR_ROT_INDICES, FRAME_RATE) # [seq, num_imus, 9]
            joint_vel_root = _compute_joint_velocity_root(rotation_global_full, position_global_full, FRAME_RATE)
            root_vel = _central_diff(position_global_full[:, 0, :], 1.0 / FRAME_RATE)
            sensor_vel_root = joint_vel_root[:, _VEL_SELECTION_INDICES, :] # [seq, 6, 3]
            sensor_vel_glb = _central_diff(position_global_full[:, _SENSOR_POS_INDICES, :], 1.0 / FRAME_RATE)

            # --- 物体数据准备 ---
            has_object = "obj_trans" in seq_data and seq_data["obj_trans"] is not None
            obj_name = ""
            device = position_global_full.device
            dtype = position_global_full.dtype
            imu_dim = human_imu.shape[-1]
            obj_trans = torch.zeros(actual_seq_len, 3, device=device, dtype=dtype)
            obj_rot = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(actual_seq_len, 1, 1)
            obj_rot_6d = transforms.matrix_to_rotation_6d(obj_rot.reshape(-1, 3, 3)).reshape(actual_seq_len, 6)
            obj_scale = torch.ones(actual_seq_len, device=device, dtype=dtype)
            obj_imu = torch.zeros(actual_seq_len, imu_dim, device=device, dtype=dtype)
            obj_vel = torch.zeros(actual_seq_len, 3, device=device, dtype=dtype)

            if has_object:
                obj_name = seq_data.get("obj_name", "unknown_object")
                obj_trans = seq_data["obj_trans"][start_idx:end_idx].squeeze(-1).to(device=device, dtype=dtype)
                obj_rot = seq_data["obj_rot"][start_idx:end_idx].to(device=device, dtype=dtype)
                obj_rot_6d = transforms.matrix_to_rotation_6d(obj_rot.reshape(-1, 3, 3)).reshape(actual_seq_len, 6)
                obj_scale = seq_data["obj_scale"][start_idx:end_idx].to(device=device, dtype=dtype)
                obj_imu = _build_object_imu(obj_rot_6d, obj_trans, FRAME_RATE) # [seq, 9]
                obj_vel = _central_diff(obj_trans, 1.0 / FRAME_RATE)

            # --- 提前初始化所有接触窗口变量 ---
            lfoot_contact_window = seq_data.get("lfoot_contact", torch.zeros(seq_len, dtype=torch.float))[start_idx:end_idx]
            rfoot_contact_window = seq_data.get("rfoot_contact", torch.zeros(seq_len, dtype=torch.float))[start_idx:end_idx]
            lhand_contact_window = seq_data.get("lhand_contact", torch.zeros(seq_len, dtype=torch.bool))[start_idx:end_idx]
            rhand_contact_window = seq_data.get("rhand_contact", torch.zeros(seq_len, dtype=torch.bool))[start_idx:end_idx]
            obj_contact_window = seq_data.get("obj_contact", torch.zeros(seq_len, dtype=torch.bool))[start_idx:end_idx]

            # --- 虚拟关节监督 ---
            lhand_obj_direction = torch.zeros(actual_seq_len, 3, device=device, dtype=dtype)
            rhand_obj_direction = torch.zeros(actual_seq_len, 3, device=device, dtype=dtype)

            if has_object:
                try:
                    virtual_joint_data = compute_obj_direction_supervision(
                        position_global_full, obj_trans, obj_rot_6d,
                        lhand_contact_window, rhand_contact_window
                    )

                    lhand_obj_direction = virtual_joint_data["lhand_obj_direction"]
                    rhand_obj_direction = virtual_joint_data["rhand_obj_direction"]

                except Exception as e:
                    print(f"计算虚拟关节数据时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    lhand_obj_direction = torch.zeros(actual_seq_len, 3, device=device, dtype=dtype)
                    rhand_obj_direction = torch.zeros(actual_seq_len, 3, device=device, dtype=dtype)

            result = {
                "human_imu": human_imu.float(),  # [seq, num_imus, 9]
                "obj_imu": obj_imu.float(),  # [seq, 9]
                "seq_name": seq_name,
                "seq_file": os.path.basename(file_path),
                "window_start": int(start_idx),
                "window_end": int(end_idx),

                "trans": trans.float(),
                "ori_root_reduced": ori_root_reduced.float(),
                "has_object": has_object,
                "obj_trans": obj_trans.float(),
                "obj_rot": obj_rot_6d.float(),  # 使用6D表示而不是矩阵
                "obj_scale": obj_scale.float(),
                "obj_name": obj_name,

                "root_vel": root_vel.float(),
                "obj_vel": obj_vel.float(),
                "sensor_vel_root": sensor_vel_root.float(),
                "sensor_vel_glb": sensor_vel_glb.float(),

                "lhand_contact": lhand_contact_window.bool(),
                "rhand_contact": rhand_contact_window.bool(),
                "obj_contact": obj_contact_window.bool(),
                "lfoot_contact": lfoot_contact_window.float(),
                "rfoot_contact": rfoot_contact_window.float(),
                "lhand_obj_direction": lhand_obj_direction.float(),
                "rhand_obj_direction": rhand_obj_direction.float(),

                "position_global": position_global_full.float(),
                "rotation_global": rotation_global_full.float(),
                "pose": pose.float(),
            }

            return result

        except Exception as e:
            print(f"处理预加载数据时出错，文件: {file_path}, 窗口索引: {idx}, Start: {start_idx}, End: {end_idx}: {e}")
            import traceback
            traceback.print_exc() # 打印详细错误
            # 抛出异常让DataLoader跳过该样本
            raise RuntimeError(f"处理样本{idx}时出错: {e}")


