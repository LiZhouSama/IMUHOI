import glob
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pytorch3d.transforms as transforms
import random # Import random for shuffling sequence info in debug mode

from configs.global_config import FRAME_RATE, IMU_JOINTS_POS, IMU_JOINTS_ROT, IMU_JOINT_NAMES, acc_scale

# --- IMU 计算函数 ---
def _syn_acc_optimized(v, smooth_n=4):
    """使用优化的张量运算从位置生成加速度
    
    参数:
        v: 位置数据 [T, 3] 或 [T, N, 3]
        smooth_n: 平滑窗口大小
    
    返回:
        acc: 加速度数据，与输入相同形状
    """
    # 获取维度信息
    orig_shape = v.shape
    if len(orig_shape) == 3:
        # 如果是 [T, N, 3]，展平为 [T, N*3]
        T, N, D = orig_shape
        v_flat = v.reshape(T, -1)
    else:
        # 如果是 [T, 3]
        v_flat = v
    
    # 构建基础差分计算
    T = v_flat.shape[0]
    acc = torch.zeros_like(v_flat)
    
    # 使用张量索引实现二阶差分
    if T > 2:
        acc[1:-1] = (v_flat[:-2] + v_flat[2:] - 2 * v_flat[1:-1]) * FRAME_RATE ** 2
    
    # 应用平滑
    mid = smooth_n // 2
    if mid != 0 and T > smooth_n * 2:
        # 使用张量索引实现平滑计算
        smooth_range = slice(smooth_n, -smooth_n)
        acc[smooth_range] = (v_flat[:-smooth_n*2] + v_flat[smooth_n*2:] - 2 * v_flat[smooth_n:-smooth_n]) * FRAME_RATE ** 2 / smooth_n ** 2
    
    # 恢复原始形状
    return acc.reshape(orig_shape)

def compute_imu_data(position_global, rotation_global, imu_joints_pos, imu_joints_rot, smooth_n=4):
    """
    计算特定关节的IMU数据（加速度和方向）
    
    参数:
        position_global: 全局关节位置 [T, J, 3]
        rotation_global: 全局关节旋转 [T, J, 3, 3]
        imu_joints: IMU关节索引列表
        smooth_n: 平滑窗口大小
    
    返回:
        IMU数据字典，包含加速度和方向信息
    """
    device = position_global.device
    
    # 提取指定关节的位置和旋转
    imu_positions = position_global[:, imu_joints_pos, :]  # [T, num_imus, 3]
    imu_orientations = rotation_global[:, imu_joints_rot, :, :]  # [T, num_imus, 3, 3]
    
    T = imu_positions.shape[0]
    num_imus = len(imu_joints_pos)
    
    # 并行计算所有IMU关节的加速度
    imu_accelerations = _syn_acc_optimized(imu_positions, smooth_n)
   
    # 返回IMU数据字典，包含加速度和方向
    imu_data = {
        'accelerations': imu_accelerations,  # [T, num_imus, 3]
        'orientations': imu_orientations   # [T, num_imus, 3, 3]
    }
    
    return imu_data

def compute_object_imu(obj_trans, obj_rot_mat, smooth_n=4):
    """
    计算物体的IMU数据（加速度和方向）
    
    参数:
        obj_trans: 物体位置 [T, 3]
        obj_rot_mat: 物体旋转矩阵 [T, 3, 3]
        smooth_n: 平滑窗口大小
        
    返回:
        物体IMU数据字典
    """
    T = obj_trans.shape[0]
    obj_trans = obj_trans.reshape(T, 3)
    device = obj_trans.device
    
    # 计算物体加速度
    obj_accel = _syn_acc_optimized(obj_trans, smooth_n=4)  # [T, 3]

    # 返回物体IMU数据
    return {
        'accelerations': obj_accel.reshape(T, 1, 3),  # [T, 1, 3]
        'orientations': obj_rot_mat.reshape(T, 1, 3, 3)  # [T, 1, 3, 3]
    }
# --- 结束 IMU 计算函数 ---

class IMUDataset(Dataset):
    def __init__(self, data_dir, window_size=60, normalize=True, debug=False, min_obj_contact_frames=10):
        """
        IMU数据集 - 每个epoch为每个序列随机采样一个窗口
        Args:
            data_dir: 数据目录，可以是字符串（单个目录）或列表（多个目录）
            window_size: 窗口大小
            normalize: 是否对数据进行标准化
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
        self.normalize = normalize
        self.debug = debug
        self.min_obj_contact_frames = min_obj_contact_frames

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

                # 检查序列长度是否足够创建至少一个窗口 (从索引1开始，长度为window_size)
                # 需要 seq_len >= window_size + 1 (因为最大 start_idx 是 seq_len - window_size)
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
             # 返回错误字典
             return 


        # 2. 随机生成 start_idx
        # 有效 start_idx 范围：[1, seq_len - window_size] (包含两端)
        max_start_idx = seq_len - self.window_size
        if max_start_idx < 1:
            # 这种情况理论上在 __init__ 中被过滤掉了，但为了安全起见
            print(f"错误：序列 {seq_name} (长度 {seq_len}) 过短，无法采样窗口大小 {self.window_size}")
            return 
        start_idx = torch.randint(1, max_start_idx + 1, (1,)).item()
        # start_idx = 1
        end_idx = start_idx + self.window_size # 切片时使用 end_idx

        # 3. 从预加载数据中获取序列数据
        try:
             seq_data = self.loaded_data[file_path]
        except KeyError:
             print(f"错误：无法在预加载数据中找到文件路径 {file_path} 对应的键。序列索引: {idx}")
             return 

        # 4. 切片和处理数据
        try:
            # --- 从预加载的 seq_data 中提取和切片数据 ---
            # 注意：不再需要 torch.load(file_path)

            # 提取并切片数据 (确保键存在)
            root_pos = seq_data["position_global_full_gt_world"][start_idx:end_idx, 0, :]   # [seq, 3]
            motion = seq_data["rotation_local_full_gt_list"][start_idx:end_idx]
            # human_imu_acc = seq_data["imu_global_full_gt"]["accelerations"][start_idx:end_idx]
            # human_imu_ori = seq_data["imu_global_full_gt"]["orientations"][start_idx:end_idx]

            # --- 动态计算人类 IMU 数据 ---
            position_global_full = seq_data["position_global_full_gt_world"][start_idx:end_idx] # [seq, J, 3]
            rotation_global_full = seq_data["rotation_global"][start_idx:end_idx] # [seq, J, 3, 3]
            human_imu_data = compute_imu_data(position_global_full, rotation_global_full, IMU_JOINTS_POS, IMU_JOINTS_ROT)
            human_imu_acc = human_imu_data["accelerations"] # [seq, num_imus, 3]
            human_imu_ori = human_imu_data["orientations"] # [seq, num_imus, 3, 3]
            
            # 提取IMU关节的全局位置和旋转（用于可视化）
            imu_global_positions = position_global_full[:, IMU_JOINTS_POS, :]  # [seq, num_imus, 3]
            imu_global_rotations = rotation_global_full[:, IMU_JOINTS_ROT, :, :]  # [seq, num_imus, 3, 3]
            # --- 结束动态计算 ---

            # 将旋转矩阵转换为6D旋转表示
            human_imu_ori_flat = human_imu_ori.reshape(self.window_size, -1, 3, 3)
            human_imu_ori_6d = transforms.matrix_to_rotation_6d(human_imu_ori_flat)
            human_imu = torch.cat([human_imu_acc, human_imu_ori_6d], dim=-1)  # [T, num_imus, 9]

            # 处理物体数据
            has_object = "obj_trans" in seq_data and seq_data["obj_trans"] is not None
            obj_name = '' # Default value
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

                # --- 动态计算物体 IMU 数据 ---
                obj_imu_data = compute_object_imu(obj_trans, obj_rot)
                obj_imu_acc = obj_imu_data["accelerations"] # [seq, 1, 3]
                obj_imu_ori = obj_imu_data["orientations"] # [seq, 1, 3, 3]
                # --- 结束动态计算 ---

                # 将旋转矩阵转换为6D旋转表示
                obj_imu_ori_6d = transforms.matrix_to_rotation_6d(obj_imu_ori)
                # 将加速度(3D)和6D旋转表示(6D)拼接
                obj_imu = torch.cat([obj_imu_acc, obj_imu_ori_6d], dim=-1)  # [seq, 1, 9]

            else:
                # 如果没有物体数据，使用上面定义的默认值
                pass

            # 对数据进行归一化（如果需要）
            norm_human_imu = human_imu.float()
            norm_root_pos = root_pos.float()
            norm_motion = motion.float()
            norm_obj_imu = obj_imu.float()
            norm_obj_trans = obj_trans.float()
            norm_obj_rot = transforms.matrix_to_rotation_6d(obj_rot).float()

            if self.normalize:  
                root_imu_acc_start = human_imu_acc[0, -1] # [3]
                root_imu_ori_start = human_imu_ori_flat[0, -1] # [3, 3]
                root_imu_ori_start_inv = torch.inverse(root_imu_ori_start)
                # norm_human_imu_acc = root_imu_ori_start_inv @ (human_imu_acc - root_imu_acc_start).unsqueeze(-1)
                norm_human_imu_acc = root_imu_ori_start_inv @ human_imu_acc .unsqueeze(-1)
                norm_human_imu_acc = norm_human_imu_acc.squeeze(-1) / acc_scale
                norm_human_imu_ori = root_imu_ori_start_inv @ human_imu_ori_flat
                norm_human_imu = torch.cat([norm_human_imu_acc, transforms.matrix_to_rotation_6d(norm_human_imu_ori)], dim=-1)
                if has_object:
                    # norm_obj_acc = root_imu_ori_start_inv @ (obj_imu_acc - root_imu_acc_start).unsqueeze(-1)
                    norm_obj_acc = root_imu_ori_start_inv @ obj_imu_acc.unsqueeze(-1)
                    norm_obj_acc = norm_obj_acc.squeeze(-1) / acc_scale
                    norm_obj_ori = root_imu_ori_start_inv @ obj_imu_ori
                    norm_obj_imu = torch.cat([norm_obj_acc, transforms.matrix_to_rotation_6d(norm_obj_ori)], dim=-1)

                # norm_human_imu_acc = torch.cat((human_imu_acc[:, :5] - human_imu_acc[:, 5:], human_imu_acc[:, 5:]), dim=1).bmm(human_imu_ori_flat[:, -1]) / acc_scale
                # norm_human_imu_ori = torch.cat((human_imu_ori_flat[:, 5:].transpose(2, 3).matmul(human_imu_ori_flat[:, :5]), human_imu_ori_flat[:, 5:]), dim=1)
                # norm_human_imu = torch.cat((norm_human_imu_acc, transforms.matrix_to_rotation_6d(norm_human_imu_ori)), dim=-1)
                # # 物体是不是不需要归一化到根坐标系下？
                # norm_obj_imu_acc = obj_imu_acc.bmm(human_imu_ori_flat[:, -1]) / acc_scale
                # norm_obj_imu_ori = human_imu_ori_flat[:, 5:].transpose(2, 3).matmul(obj_imu_ori)
                # norm_obj_imu = torch.cat((norm_obj_imu_acc, transforms.matrix_to_rotation_6d(norm_obj_imu_ori)), dim=-1)
                # # norm_obj_imu_acc = obj_imu_acc/acc_scale
                # # norm_obj_imu_ori = obj_imu_ori
                # # norm_obj_imu = torch.cat((norm_obj_imu_acc, transforms.matrix_to_rotation_6d(norm_obj_imu_ori)), dim=-1)

                # 对motion归一化(输出)
                root_global_pos_start = root_pos[0]
                norm_motion = motion.clone()
                root_rot = transforms.rotation_6d_to_matrix(norm_motion[:, :6]) # 每一帧
                root_global_rot_start = root_rot[0]
                root_start_rot_invert = root_global_rot_start.swapaxes(-2,-1)
                norm_root_rot = root_start_rot_invert @ root_rot
                norm_motion[:, :6] = transforms.matrix_to_rotation_6d(norm_root_rot)
                norm_root_pos = root_start_rot_invert @ (root_pos - root_global_pos_start).unsqueeze(-1)    # 相当于初始位置为0
                norm_root_pos = norm_root_pos.squeeze(-1)
                if has_object:
                    # 对obj归一化(输出)
                    norm_obj_trans = root_start_rot_invert @ (obj_trans - root_global_pos_start).unsqueeze(-1)
                    norm_obj_trans = norm_obj_trans.squeeze(-1)
                    norm_obj_rot = root_start_rot_invert @ obj_rot
                    norm_obj_rot = transforms.matrix_to_rotation_6d(norm_obj_rot)

                # norm_human_imu = human_imu.clone()
                # norm_obj_imu = obj_imu.clone()
                # norm_motion = motion.clone()
                # norm_root_pos = root_pos.clone()
                # norm_obj_trans = obj_trans.clone()
                # norm_obj_rot = transforms.matrix_to_rotation_6d(obj_rot)


            # --- 计算速度 ---
            # 1. 根关节速度
            if norm_root_pos.shape[0] > 1:
                root_vel = (norm_root_pos[1:] - norm_root_pos[:-1]) * FRAME_RATE  # 转换为米/秒
                root_vel = torch.cat([torch.zeros_like(root_vel[:1]), root_vel], dim=0)
            else:
                root_vel = torch.zeros_like(norm_root_pos)
            
            # 2. 物体速度
            if has_object and norm_obj_trans.shape[0] > 1:
                obj_vel = (norm_obj_trans[1:] - norm_obj_trans[:-1]) * FRAME_RATE  # 转换为米/秒
                obj_vel = torch.cat([torch.zeros_like(obj_vel[:1]), obj_vel], dim=0)
            else:
                obj_vel = torch.zeros(norm_root_pos.shape[0], 3)
                
            # 3. 叶子节点速度 (从position_global_full计算)
            from configs.global_config import joint_set
            leaf_indices = joint_set.leaf  # [7, 8, 12, 20, 21]
            
            # 从全局关节位置中提取叶子节点位置
            leaf_positions_global = position_global_full[:, leaf_indices, :]  # [seq, n_leaf, 3]
            
            # 计算叶子节点的全局速度（相邻帧差分）
            if leaf_positions_global.shape[0] > 1:
                leaf_vel_global = (leaf_positions_global[1:] - leaf_positions_global[:-1]) * FRAME_RATE  # 转换为米/秒
                # 第一帧速度设为0
                leaf_vel_global = torch.cat([torch.zeros_like(leaf_vel_global[:1]), leaf_vel_global], dim=0)  # [seq, n_leaf, 3]
            else:
                leaf_vel_global = torch.zeros_like(leaf_positions_global)  # [seq, n_leaf, 3]
            
            # 归一化叶子节点速度：使用root_start_rot_invert左乘
            # root_start_rot_invert: [3, 3], leaf_vel_global: [seq, n_leaf, 3]
            leaf_vel = torch.einsum('ij,tnj->tni', root_start_rot_invert, leaf_vel_global)  # [seq, n_leaf, 3]
            # --- 结束计算速度 ---

            # 构建结果字典
            # --- 加载足部接触标签 ---
            lfoot_contact_window = seq_data.get("lfoot_contact", torch.zeros(seq_len, dtype=torch.float))[start_idx:end_idx]
            rfoot_contact_window = seq_data.get("rfoot_contact", torch.zeros(seq_len, dtype=torch.float))[start_idx:end_idx]

            # --- 直接使用预处理计算好的接触标签 ---
            lhand_contact_window = seq_data.get("lhand_contact", torch.zeros(seq_len, dtype=torch.bool))[start_idx:end_idx]
            rhand_contact_window = seq_data.get("rhand_contact", torch.zeros(seq_len, dtype=torch.bool))[start_idx:end_idx]
            obj_contact_window = seq_data.get("obj_contact", torch.zeros(seq_len, dtype=torch.bool))[start_idx:end_idx]

            result = {
                "root_pos": norm_root_pos.float(),
                "motion": norm_motion.float(),  # [seq, 132]
                "root_pos_start": root_global_pos_start.float(),  # [3]
                "root_rot_start": root_global_rot_start.float(),  # [3, 3]
                "human_imu": norm_human_imu.float(),  # [seq, num_imus, 9] - 现在是9D (3D加速度 + 6D旋转)
                "obj_imu": norm_obj_imu.float() ,  # [seq, 1, 9] - 现在是9D (3D加速度 + 6D旋转)
                "obj_trans": norm_obj_trans.float() ,  # [seq, 3] (未缩放)
                "obj_rot": norm_obj_rot.float() ,  # [seq, 6]
                "obj_scale": obj_scale.float() , # [seq] (单独返回)
                "obj_name": obj_name,
                "has_object": has_object,
                "root_vel": root_vel.float(), # [seq, 3] - 单位: m/s
                "obj_vel": obj_vel.float(), # [seq, 3] - 物体速度 (m/s)
                "leaf_vel": leaf_vel.float(), # [seq, n_leaf, 3] - 叶子节点速度 (m/s)
                "lhand_contact": lhand_contact_window.bool(), # [seq]
                "rhand_contact": rhand_contact_window.bool(), # [seq]
                "obj_contact": obj_contact_window.bool(), # [seq]
                "lfoot_contact": lfoot_contact_window.float(), # [seq] - 左脚接触标签
                "rfoot_contact": rfoot_contact_window.float(), # [seq] - 右脚接触标签
            }

            return result

        except Exception as e:
            print(f"处理预加载数据时出错，文件: {file_path}, 窗口索引: {idx}, Start: {start_idx}, End: {end_idx}: {e}")
            import traceback
            traceback.print_exc() # 打印详细错误
            # 返回包含正确键但值为默认值的字典，以避免后续代码出错
            return 
