"""
OMOMO dataset adapter for GlobalPose training pipeline.
Converts OMOMO data format to match GP's expected input/target structure.

支持的数据组织形式：
- 单个.pt文件：包含一个或多个序列
- 文件夹路径：加载文件夹中所有 pt 文件

支持的数据格式：

1. GP格式（已经包含IMU数据）：
   必须包含 'aS', 'wS', 'RIS', 'tran', 'pose'
   可选键: 'name', 'shape', 'obj_imu', 'obj_trans', 'RIM', 'RSB'

   a. 聚合格式（多序列）：
      - 各键为列表，data['aS'][i] 是第 i 个序列
      - data['aS']: List[Tensor[T_i, 6, 3]]

   b. 单序列格式：
      - 各键直接是张量，data['aS']: Tensor[T, 6, 3]

2. Raw格式（从关节数据合成IMU）：
   必需键: 'rotation_global', 'position_global_full_gt_world'
   推荐键: 'rotation_local_full_gt_list' (若缺失则从全局旋转推导)
   可选键: 'shape', 'obj_rot', 'obj_trans'

   a. 聚合格式（多序列）：
      - data['rotation_global']: Tensor[N, T, 22, 3, 3]

   b. 单序列格式：
      - data['rotation_global']: Tensor[T, 22, 3, 3]

   物体数据可直接放在顶层或 object 子字典中。
"""
from __future__ import annotations

import os
import glob
from typing import Dict, List, Optional, Tuple

import torch
import articulate as art
from articulate.utils.torch import RNNWithInitDataset

# Constants matching GP training
V_IMU = (1961, 5424, 1176, 4662, 411, 3021)
J_REDUCE = (1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19)
J_IGNORE = (0, 7, 8, 10, 11, 20, 21, 22, 23)
J_CONTACT = (0, 10, 11, 22, 23)

# DIP-IMU 顺序：0=LHeel, 1=RHeel, 2=LAnkle, 3=RAnkle, 4=Head, 5=Root
_SENSOR_ROT_INDICES = [18, 19, 4, 5, 15, 0]
_SENSOR_POS_INDICES = [20, 21, 7, 8, 15, 0]

def _ensure_matrix_pose(pose: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle pose representation to rotation matrices."""
    if pose.ndim == 3 and pose.shape[-1] == 3:
        aa_pose = pose
    elif pose.ndim == 2 and pose.shape[1] == 72:
        aa_pose = pose.view(-1, 24, 3)
    else:
        raise ValueError(f"Unsupported pose tensor shape: {pose.shape}")
    return art.math.axis_angle_to_rotation_matrix(aa_pose).view(-1, 24, 3, 3)


def _finite_difference(x: torch.Tensor, dt: float) -> torch.Tensor:
    """Compute velocity via finite difference."""
    vel = torch.zeros_like(x)
    if x.shape[0] > 1:
        vel[1:] = (x[1:] - x[:-1]) / dt
    return vel


def _central_diff(a: torch.Tensor, dt: float) -> torch.Tensor:
    """Central difference for smoother velocity estimation."""
    if a.shape[0] <= 1:
        return torch.zeros_like(a)
    vel = torch.zeros_like(a)
    vel[1:-1] = (a[2:] - a[:-2]) / (2.0 * dt)
    vel[0] = (a[1] - a[0]) / dt
    vel[-1] = (a[-1] - a[-2]) / dt
    return vel


def _synthesize_imu_from_joints(
    rotation_global: torch.Tensor,
    position_global: torch.Tensor,
    fps: float,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从关节旋转和位置合成IMU数据 (参考 dataset_trans_obj.py)
    
    Args:
        rotation_global: [T, 22/24, 3, 3] 全局关节旋转
        position_global: [T, 22/24, 3] 全局关节位置
        fps: 帧率
        device: 设备
        
    Returns:
        aS: [T, 6, 3] 传感器加速度
        wS: [T, 6, 3] 传感器角速度
        RIS: [T, 6, 3, 3] 传感器旋转
    """
    T = rotation_global.shape[0]
    dt = 1.0 / fps
    
    # 选择6个IMU传感器的旋转和位置
    sel_R = rotation_global[:, _SENSOR_ROT_INDICES]  # [T, 6, 3, 3]
    sel_pos = position_global[:, _SENSOR_POS_INDICES]  # [T, 6, 3]
    # 计算加速度：位置 -> 速度 -> 加速度
    vel = _central_diff(sel_pos, dt)  # [T, 6, 3]
    acc = _central_diff(vel, dt)  # [T, 6, 3]
    
    # 计算角速度（从连续帧的旋转矩阵）
    w = torch.zeros(T, 6, 3, device=device)
    if T > 1:
        # R(t+1) = R(t) * exp(w*dt)
        # => exp(w*dt) = R(t)^T * R(t+1)
        dR = sel_R[:-1].transpose(-1, -2).matmul(sel_R[1:])  # [T-1, 6, 3, 3]
        # 转换为轴角然后除以dt得到角速度
        w_aa = art.math.rotation_matrix_to_axis_angle(dR.reshape(-1, 3, 3)).view(T-1, 6, 3)
        w[1:] = w_aa / dt
    
    # 将加速度转换到传感器局部坐标系（IMU数据已去除重力，无需再次补偿）
    aS = sel_R.transpose(-1, -2).matmul(acc.unsqueeze(-1)).squeeze(-1)  # [T, 6, 3]
    
    # 将角速度转换到传感器局部坐标系
    # wS = R^T * w
    wS = sel_R.transpose(-1, -2).matmul(w.unsqueeze(-1)).squeeze(-1)  # [T, 6, 3]
    
    # RIS 就是传感器的全局旋转
    RIS = sel_R  # [T, 6, 3, 3]
    
    return aS, wS, RIS

def _convert_raw_to_gp_format(
    raw_record: Dict[str, torch.Tensor],
    fps: float,
    device: torch.device
) -> Optional[Dict[str, torch.Tensor]]:
    """
    将raw格式数据转换为GP格式 (支持 dataset_trans_obj.py 的数据格式)
    
    Args:
        raw_record: 包含 rotation_global, position_global_full_gt_world 等的字典
        fps: 帧率
        device: 设备
        
    Returns:
        GP格式的record，包含 aS, wS, RIS, tran, pose 等
    """
    # 检查必需的键
    if 'rotation_global' not in raw_record or 'position_global_full_gt_world' not in raw_record:
        return None
    
    rotation_global = raw_record['rotation_global'].to(device).float()  # [T, 22/24, 3, 3]
    position_global = raw_record['position_global_full_gt_world'].to(device).float()  # [T, 22/24, 3]
    
    # 处理22关节到24关节的padding
    if rotation_global.shape[1] == 22:
        T = rotation_global.shape[0]
        rot_padded = torch.zeros(T, 24, 3, 3, dtype=rotation_global.dtype, device=device)
        rot_padded[:, :22] = rotation_global
        # 填充单位矩阵
        rot_padded[:, 22:] = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)
        rotation_global = rot_padded
        
        pos_padded = torch.zeros(T, 24, 3, dtype=position_global.dtype, device=device)
        pos_padded[:, :22] = position_global
        position_global = pos_padded
    
    # 合成IMU数据
    aS, wS, RIS = _synthesize_imu_from_joints(rotation_global, position_global, fps, device)
    
    # 提取根节点平移
    tran = position_global[:, 0, :].clone()  # [T, 3]
    
    # 提取局部姿态
    if 'rotation_local_full_gt_list' in raw_record:
        pose_local = raw_record['rotation_local_full_gt_list'].to(device).float()  # [T, 132]
        # 转换为轴角
        pose_mat = art.math.r6d_to_rotation_matrix(pose_local.reshape(pose_local.shape[0], -1, 6)).view(pose_local.shape[0], -1, 3, 3)
        pose = art.math.rotation_matrix_to_axis_angle(pose_mat).view(pose_local.shape[0], -1, 3)
        # Padding到24关节
        if pose.shape[1] == 22:
            pose_padded = torch.zeros(pose.shape[0], 24, 3, device=device)
            pose_padded[:, :22] = pose
            pose = pose_padded
    else:
        # 如果没有局部姿态，从全局旋转推导
        # 注意：这是一个简化方法，相对于根节点而不是父关节
        # 可能不够准确，建议在数据中提供 rotation_local_full_gt_list
        print("警告: 未找到 rotation_local_full_gt_list，将从全局旋转推导局部姿态（可能不准确）")
        root_rot_inv = rotation_global[:, 0].transpose(-1, -2).unsqueeze(1)  # [T, 1, 3, 3]
        pose_local = root_rot_inv.matmul(rotation_global)  # [T, 24, 3, 3]
        pose = art.math.rotation_matrix_to_axis_angle(pose_local.reshape(-1, 3, 3)).view(pose_local.shape[0], 24, 3)
    
    # 构建GP格式的record
    gp_record = {
        'aS': aS,
        'wS': wS,
        'RIS': RIS,
        'tran': tran,
        'pose': pose,
        'name': raw_record.get('name', 'unknown'),
        'RIM': torch.eye(3).repeat(6, 1, 1),
        'RSB': torch.eye(3).repeat(6, 1, 1),
    }
    
    # 可选字段
    if 'shape' in raw_record:
        gp_record['shape'] = raw_record['shape'].to(device)
    
    # 物体数据（支持多种格式）
    obj_rot = None
    obj_trans = None
    
    # 尝试从不同位置获取物体数据
    if 'obj_rot' in raw_record and 'obj_trans' in raw_record:
        obj_rot = raw_record['obj_rot']
        obj_trans = raw_record['obj_trans']
    elif 'object' in raw_record:
        if 'rot' in raw_record['object']:
            obj_rot = raw_record['object']['rot']
        if 'position' in raw_record['object']:
            obj_trans = raw_record['object']['position']
    
    if obj_rot is not None and obj_trans is not None:
        obj_rot = obj_rot.to(device).float()
        obj_trans = obj_trans.to(device).float()
        
        # 合成物体IMU
        if obj_rot.dim() == 2 and obj_rot.shape[-1] == 6:
            # 6D旋转表示
            obj_rot_mat = art.math.r6d_to_rotation_matrix(obj_rot)  # [T, 3, 3]
        elif obj_rot.dim() == 3 and obj_rot.shape[-2:] == (3, 3):
            # 已经是旋转矩阵
            obj_rot_mat = obj_rot  # [T, 3, 3]
        else:
            print(f"警告: 不支持的物体旋转格式 {obj_rot.shape}，跳过物体数据")
            obj_rot_mat = None
        
        if obj_rot_mat is not None:
            obj_orientation = obj_rot_mat.reshape(obj_rot_mat.shape[0], 9)  # [T, 9]
            gp_record['obj_imu'] = obj_orientation
            gp_record['obj_trans'] = obj_trans
    
    return gp_record
def build_omomo_sequence_targets(
    record: Dict[str, torch.Tensor],
    body_model: art.ParametricModel,
    dt: float = 1.0 / 30.0,
    device: Optional[torch.device] = None
) -> Optional[Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]]:
    """
    Convert OMOMO record to GP-style inputs and targets.
    
    Args:
        record: Dict with keys 'aS', 'wS', 'RIS', 'tran', 'pose', optionally 'obj_imu', 'obj_trans'
        body_model: Articulate SMPL model
        dt: Time step in seconds
        device: Target device
        
    Returns:
        (inputs, targets, meta, obj_data) or None if invalid
    """
    device = device or torch.device('cpu')
    
    # Required keys for human
    required = ['aS', 'wS', 'RIS', 'tran', 'pose']
    for key in required:
        if key not in record:
            return None
    
    aS = record['aS'].to(device)  # [T, 6, 3] sensor frame acceleration
    wS = record['wS'].to(device)  # [T, 6, 3] sensor frame angular velocity
    RIS = record['RIS'].to(device)  # [T, 6, 3, 3] sensor to inertial rotation
    tran = record['tran'].to(device)  # [T, 3] root translation
    pose_tensor = record['pose']
    
    pose_local = _ensure_matrix_pose(pose_tensor.to(device))  # [T, 24, 3, 3]
    seq_len = pose_local.shape[0]
    
    if seq_len < 4:
        return None
    
    # Use identity for RIM and RSB if not provided (OMOMO data is already aligned)
    RIM = record.get('RIM', torch.eye(3, device=device).repeat(6, 1, 1))
    RSB = record.get('RSB', torch.eye(3, device=device).repeat(6, 1, 1))
    
    # Transform to model frame (M)
    if RIM.dim() == 2:  # [3, 3]
        RIM = RIM.unsqueeze(0).expand(6, 3, 3)
    if RSB.dim() == 2:
        RSB = RSB.unsqueeze(0).expand(6, 3, 3)
    
    rim_t = RIM.transpose(-1, -2).unsqueeze(0)  # [1, 6, 3, 3]  # eye
    rot_im_to_model = torch.matmul(rim_t, RIS)  # [T, 6, 3, 3] # RIS
    RMB = torch.matmul(rot_im_to_model, RSB.unsqueeze(0))  # [T, 6, 3, 3] # RIS
    
    aM = torch.matmul(rot_im_to_model, aS.unsqueeze(-1)).squeeze(-1) # a
    wM = torch.matmul(rot_im_to_model, wS.unsqueeze(-1)).squeeze(-1) # w    
    
    # Root frame transformation
    root_rot = RMB[:, 5]  # [T, 3, 3] pelvis rotation
    aRB = torch.matmul(aM, root_rot)  # [T, 6, 3]
    wRB = torch.matmul(wM, root_rot)  # [T, 6, 3]
    RRB = torch.matmul(root_rot.transpose(1, 2).unsqueeze(1), RMB[:, :5])  # [T, 5, 3, 3]
    gR0 = -root_rot[:, 1] 
    
    # Construct network inputs (matching GP format)
    inputs = torch.cat([
        aRB.reshape(seq_len, -1),  # 18
        wRB.reshape(seq_len, -1),  # 18
        RRB.reshape(seq_len, -1),  # 45
        gR0  # 3
    ], dim=-1)  # [T, 84]
    
    # Shape parameters
    shape_params = record.get('shape', None)
    if shape_params is not None:
        if shape_params.ndim == 1:
            shape_params = shape_params.unsqueeze(0)
        shape_params = shape_params.to(device).expand(seq_len, -1)
    
    # Forward kinematics
    pose_global, joints_global, verts_global = body_model.forward_kinematics(
        pose_local, shape_params, tran, calc_mesh=True
    )
    
    root_global = pose_global[:, 0]  # [T, 3, 3]
    # PL targets: 5 vertices relative to pelvis in root frame
    pRB = torch.matmul(
        (verts_global[:, :5] - verts_global[:, 5:6]).view(seq_len, 5, 3),
        root_global
    )  # [T, 5, 3]
    gR = -root_global[:,1]  # [T, 3]
    
    # IK targets: canonical joint positions
    pose_canon = pose_local.clone()
    identity = torch.eye(3, device=device)
    for j in J_IGNORE:
        pose_canon[:, j] = identity
    
    _, joints_canon = body_model.forward_kinematics(pose_canon, shape_params, None, calc_mesh=False)
    pRJ = joints_canon[:, 1:]  # [T, 23, 3]
    
    # RR targets: reduced joint rotations in 6D
    root_inv = root_global.transpose(1, 2)
    global_canon = torch.matmul(root_inv.unsqueeze(1), pose_global)
    for j in J_IGNORE:
        global_canon[:, j] = identity
    rrj = art.math.rotation_matrix_to_r6d(
        global_canon[:, J_REDUCE].reshape(-1, 3, 3)
    ).view(seq_len, -1)  # [T, 90]
    
    # VR targets: root velocity and contact
    root_velocity_world = _finite_difference(tran, dt)
    vRR_V = root_velocity_world[:, 1]  # [T] vertical component
    vRR_H = torch.matmul(root_inv, root_velocity_world.unsqueeze(-1)).squeeze(-1)  # [T, 3]
    
    # Contact/stationary detection
    contact_positions = joints_global[:, J_CONTACT]  # [T, 5, 3]
    contact_velocity = _finite_difference(contact_positions, dt)
    speed = contact_velocity.norm(dim=-1)  # [T, 5]
    ground = contact_positions.min(dim=1, keepdim=True).values[:, :, 1]  # [T, 1]
    height = contact_positions[:, :, 1]  # [T, 5]
    near_ground = (height - ground) < 0.06
    stationary = (speed < 0.35) & near_ground
    stationary_prob = stationary.float()  # [T, 5]
    
    # Assemble targets (matching GP order)
    targets = torch.cat([
        pRB.view(seq_len, -1),  # 15 (PL position)
        gR,  # 3 (PL orientation)
        pRJ.view(seq_len, -1),  # 69 (IK position)
        gR,  # 3 (IK orientation)
        rrj,  # 90 (RR rotation)
        torch.cat([vRR_V.unsqueeze(-1), vRR_H], dim=-1),  # 4 (VR velocity)
        stationary_prob  # 5 (VR contact)
    ], dim=-1)  # [T, 189]
    
    if torch.isnan(inputs).any() or torch.isnan(targets).any():
        return None
    
    meta = {
        'tran': tran,
        'root_rot': root_global,
        'pose': pose_local,
        'name': record.get('name', 'unknown')
    }
    
    # Object data (optional)
    obj_data = None
    if 'obj_imu' in record and 'obj_trans' in record:
        obj_imu = record['obj_imu'].to(device)  # [T, 9]
        obj_trans = record['obj_trans'].to(device)  # [T, 3]
        obj_vel = _finite_difference(obj_trans, dt)  # [T, 3]
        
        # Transform obj velocity to root frame for consistency
        obj_vel_root = torch.matmul(root_inv, obj_vel.unsqueeze(-1)).squeeze(-1)
        
        obj_data = {
            'obj_imu': obj_imu,
            'obj_trans': obj_trans,
            'obj_vel': obj_vel,
            'obj_vel_root': obj_vel_root
        }
    
    return inputs.float(), targets.float(), meta, obj_data
class OMOMOGlobalPoseDataset(RNNWithInitDataset):
    """
    OMOMO dataset adapter that follows GP's RNNWithInitDataset structure.
    
    Args:
        data_file: 数据文件路径或包含.pt文件的文件夹路径
                  - 如果是文件夹，将加载文件夹中所有.pt文件
                  - 如果是文件，仅加载该文件
        body_model: SMPL身体模型（可选）
        sequence_len: RNN序列长度
        drop_last: 是否丢弃最后不完整的序列
        min_seq_len: 最小序列长度
        fps: 帧率
        device: 计算设备
    """
    
    def __init__(
        self,
        data_file: str,
        body_model: Optional[art.ParametricModel] = None,
        sequence_len: int = 240,
        drop_last: bool = False,
        min_seq_len: int = 60,
        fps: int = 30,
        device: Optional[torch.device] = None
    ):
        self.body_model = body_model or art.ParametricModel(
            'models/SMPL_male.pkl', vert_mask=V_IMU
        )
        self.dt = 1.0 / float(fps)
        self.sequence_sources: List[Tuple[str, int]] = []
        device = device or torch.device('cpu')
        
        print(f"Loading OMOMO data: {data_file}")
        
        # 检查路径是否存在
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"路径不存在: {data_file}")
        
        # 确定是文件还是文件夹
        data_files: List[str] = []
        if os.path.isdir(data_file):
            # 文件夹：加载所有.pt文件
            pt_files = glob.glob(os.path.join(data_file, '*.pt'))
            if not pt_files:
                raise FileNotFoundError(f"文件夹中未找到.pt文件: {data_file}")
            data_files = sorted(pt_files)
            print(f"找到 {len(data_files)} 个.pt文件")
        else:
            # 单个文件
            data_files = [data_file]
        
        data_tensors: List[torch.Tensor] = []
        target_tensors: List[torch.Tensor] = []
        self.meta_list: List[Dict] = []
        self.obj_data_list: List[Optional[Dict]] = []
        
        # 处理所有文件
        for file_idx, pt_file in enumerate(data_files):
            # print(f"\n处理文件 [{file_idx+1}/{len(data_files)}]: {os.path.basename(pt_file)}")
            
            try:
                # 加载数据
                data = torch.load(pt_file, map_location=device)
                
                # 检测数据格式
                is_raw_format = ('rotation_global' in data and 'position_global_full_gt_world' in data)
                
                # 判断是聚合格式（多序列）还是单序列格式
                is_aggregated = False
                num_sequences = 1
                
                if is_raw_format:
                    # Raw格式：检查rotation_global的维度
                    rot_shape = data['rotation_global'].shape
                    if len(rot_shape) == 5:
                        # [N, T, 22, 3, 3] - 聚合格式
                        is_aggregated = True
                        num_sequences = rot_shape[0]
                        print(f"  检测到Raw聚合格式数据，包含 {num_sequences} 个序列，将进行IMU合成")
                    elif len(rot_shape) == 4:
                        # [T, 22, 3, 3] - 单序列
                        is_aggregated = False
                    else:
                        raise ValueError(f"不支持的rotation_global形状: {rot_shape}")
                else:
                    raise ValueError("未知的数据格式，需要包含 (aS, wS, RIS) 或 (rotation_global, position_global_full_gt_world)")
                
            except Exception as e:
                print(f"  加载文件失败: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # 处理序列
            valid_count = 0
            for idx in range(num_sequences):
                try:
                    # Raw格式
                    if is_aggregated:
                        # Raw聚合格式：提取第idx个序列
                        raw_record = {
                            'rotation_global': data['rotation_global'][idx],
                            'position_global_full_gt_world': data['position_global_full_gt_world'][idx],
                        }
                        
                        # Optional fields
                        if 'rotation_local_full_gt_list' in data:
                            raw_record['rotation_local_full_gt_list'] = data['rotation_local_full_gt_list'][idx]
                        if 'shape' in data:
                            raw_record['shape'] = data['shape'][idx] if idx < len(data['shape']) else None
                        
                        # 物体数据（支持多种格式）
                        if 'obj_rot' in data and 'obj_trans' in data:
                            raw_record['obj_rot'] = data['obj_rot'][idx]
                            raw_record['obj_trans'] = data['obj_trans'][idx]
                        elif 'object' in data:
                            obj_dict = data['object'] if isinstance(data['object'], dict) else {}
                            raw_record['object'] = {}
                            if 'rot' in obj_dict:
                                raw_record['object']['rot'] = obj_dict['rot'][idx]
                            if 'position' in obj_dict:
                                raw_record['object']['position'] = obj_dict['position'][idx]
                    else:
                        # Raw单序列格式：直接使用
                        raw_record = {
                            'rotation_global': data['rotation_global'],
                            'position_global_full_gt_world': data['position_global_full_gt_world'],
                        }
                        
                        # Optional fields
                        if 'rotation_local_full_gt_list' in data:
                            raw_record['rotation_local_full_gt_list'] = data['rotation_local_full_gt_list']
                        if 'shape' in data:
                            raw_record['shape'] = data['shape']
                        
                        # 物体数据
                        if 'obj_rot' in data and 'obj_trans' in data:
                            raw_record['obj_rot'] = data['obj_rot']
                            raw_record['obj_trans'] = data['obj_trans']
                        elif 'object' in data:
                            obj_dict = data['object'] if isinstance(data['object'], dict) else {}
                            if 'rot' in obj_dict and 'position' in obj_dict:
                                raw_record['object'] = {
                                    'rot': obj_dict['rot'],
                                    'position': obj_dict['position']
                                }
                    
                    # 转换为GP格式
                    self.record = _convert_raw_to_gp_format(raw_record, fps, device)
                    if self.record is None:
                        print(f"  警告: 序列转换失败，跳过")
                        continue
                    
                    # Build targets
                    result = build_omomo_sequence_targets(self.record, self.body_model, self.dt, device)
                    if result is None:
                        continue
                    
                    inputs, targets, meta, obj_data = result
                    
                    if inputs.shape[0] < min_seq_len:
                        # print(f"  警告: 序列长度 {inputs.shape[0]} < 最小长度 {min_seq_len}，跳过")
                        continue
                    
                    data_tensors.append(inputs)
                    target_tensors.append(targets)
                    self.meta_list.append(meta)
                    self.obj_data_list.append(obj_data)
                    self.sequence_sources.append((pt_file, idx))
                    valid_count += 1
                    
                except Exception as e:
                    print(f"  处理序列时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        if not data_tensors:
            raise RuntimeError("未找到有效序列")
        print(f"\n成功处理 {len(data_tensors)} 个序列（来自 {len(data_files)} 个文件）")
        effective_split_size = sequence_len
        if sequence_len and sequence_len > 0:
            chunk_size = sequence_len
            def _compute_slices(length: int) -> List[Tuple[int, int]]:
                if chunk_size <= 0:
                    return [(0, length)]
                if drop_last:
                    if length < chunk_size:
                        return []
                    slices: List[Tuple[int, int]] = []
                    start = 0
                    while start + chunk_size <= length:
                        slices.append((start, start + chunk_size))
                        start += chunk_size
                    if start < length:
                        slices.append((length - chunk_size, length))
                    return slices
                else:
                    slices: List[Tuple[int, int]] = []
                    start = 0
                    while start < length:
                        end = min(start + chunk_size, length)
                        slices.append((start, end))
                        start += chunk_size
                    return slices
            split_inputs: List[torch.Tensor] = []
            split_targets: List[torch.Tensor] = []
            split_meta: List[Dict[str, torch.Tensor]] = []
            split_obj: List[Optional[Dict[str, torch.Tensor]]] = []
            split_sources: List[Tuple] = []
            for inputs, targets, meta, obj, src in zip(
                data_tensors, target_tensors, self.meta_list, self.obj_data_list, self.sequence_sources
            ):
                slices = _compute_slices(inputs.shape[0])
                if not slices:
                    continue
                for start, end in slices:
                    split_inputs.append(inputs[start:end])
                    split_targets.append(targets[start:end])
                    split_meta.append({
                        'tran': meta['tran'][start:end],
                        'root_rot': meta['root_rot'][start:end],
                        'pose': meta['pose'][start:end],
                        'name': meta.get('name', 'unknown'),
                    })
                    if obj is not None:
                        split_obj.append({k: v[start:end] for k, v in obj.items()})
                    else:
                        split_obj.append(None)
                    split_sources.append((*src, start, end))
            data_tensors = split_inputs
            target_tensors = split_targets
            self.meta_list = split_meta
            self.obj_data_list = split_obj
            self.sequence_sources = split_sources
            effective_split_size = -1
        print(f"拆分后共有 {len(data_tensors)} 个训练片段")
        # Initialize parent RNNWithInitDataset
        super().__init__(
            data_tensors,
            target_tensors,
            split_size=effective_split_size,
            device=device,
            drop_last=drop_last
        )
        self.num_sequences = len(self.data)
        self.has_object_data = any(obj is not None for obj in self.obj_data_list)
        print(f"Dataset has object data: {self.has_object_data}")
    
    def get_sequence_meta(self, idx: int) -> Dict:
        """Get metadata for a sequence."""
        if idx < len(self.meta_list):
            return self.meta_list[idx]
        return {}
    def __getitem__(self, idx: int):
        """
        Override to also return样本索引用于物体分支构建。
        RNNWithInitDataset 的默认实现返回 ((data, init), label)。
        这里附加原始索引，方便上层在 batch 内定位对应的 object 数据。
        """
        (data, init), label = super().__getitem__(idx)
        return (data, init), label, idx
    
    def get_object_data(self, idx: int) -> Optional[Dict]:
        """Get object data for a sequence."""
        if idx < len(self.obj_data_list):
            return self.obj_data_list[idx]
        return None
    
    def stats(self) -> Dict[str, float]:
        """Return dataset statistics."""
        lengths = [d.shape[0] for d in self.data]
        return {
            'num_sequences': float(self.num_sequences),
            'mean_length': float(sum(lengths) / len(lengths)),
            'min_length': float(min(lengths)),
            'max_length': float(max(lengths)),
        }
