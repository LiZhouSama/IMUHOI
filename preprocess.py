import argparse
import os
import numpy as np
import torch
import joblib
from tqdm import tqdm
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.rotation_tools import local2global_pose
import multiprocessing as mp
from functools import partial
import pytorch3d.transforms as transforms
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
import trimesh
import glob

# 导入BPS相关库，如果不存在需要安装
try:
    from bps_torch.bps import bps_torch
    from bps_torch.tools import sample_sphere_uniform
except ImportError:
    print("警告: bps_torch库未安装，请安装后再运行")

# IMU关节索引，可以根据需要修改
IMU_JOINTS = [20, 21, 7, 8, 0, 15]  # 左手、右手、左脚、右脚、髋部、头部
IMU_JOINT_NAMES = ['left_hand', 'right_hand', 'left_foot', 'right_foot', 'hip', 'head']
HEAD_IDX = 5  # 头部在IMU_JOINTS列表中的索引
FRAME_RATE = 60  # 帧率

# 生成加速度数据，使用二阶差分计算
def _syn_acc(v, smooth_n=4):
    """从位置生成加速度"""
    mid = smooth_n // 2
    # 基础二阶差分计算
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * FRAME_RATE ** 2 for i in range(0, v.shape[0] - 2)])
    # 增加边界填充
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    
    # 平滑处理
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * FRAME_RATE ** 2 / smooth_n ** 2
                for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc

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

def compute_imu_data(position_global, rotation_global, imu_joints, smooth_n=4):
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
    imu_positions = position_global[:, imu_joints, :]  # [T, num_imus, 3]
    imu_orientations = rotation_global[:, imu_joints, :, :]  # [T, num_imus, 3, 3]
    
    T = imu_positions.shape[0]
    num_imus = len(imu_joints)
    
    
    # 并行计算所有IMU关节的加速度
    imu_accelerations = _syn_acc_optimized(imu_positions, smooth_n)
   
    # 返回IMU数据字典，包含加速度和方向
    imu_data = {
        'accelerations': imu_accelerations,  # [T, num_imus, 3]
        'orientations': imu_orientations   # [T, num_imus, 3, 3]
    }
    
    return imu_data

def normalize_to_head_frame(imu_data, head_imu_data):
    """
    将IMU数据归一化到头部坐标系
    
    参数:
        imu_data: 包含加速度和方向的字典
            - accelerations: [T, num_imus, 3]
            - orientations: [T, num_imus, 3, 3]
    
    返回:
        归一化后的IMU数据字典
    """
    # 复制数据以避免修改原始数据
    norm_imu_data = {
        'accelerations': imu_data['accelerations'].clone(),
        'orientations': imu_data['orientations'].clone()
    }
    
    # 获取头部IMU数据
    head_accel = head_imu_data[0]  # [T, 1, 3]
    head_orient = head_imu_data[1]  # [T, 1, 3, 3]
    
    # 所有IMU数据相对于头部IMU
    norm_imu_data['accelerations'] = norm_imu_data['accelerations'] - head_accel
    norm_imu_data['orientations'] = torch.matmul(torch.inverse(head_orient), norm_imu_data['orientations'])
    
    return norm_imu_data

def prep_bps_data(n_bps_points=1024, radius=1.0, device='cuda'):
    """
    准备BPS基础点云
    
    参数:
        n_bps_points: BPS点的数量
        radius: 球体半径
        device: 计算设备
        
    返回:
        BPS基础点云 [n_bps_points, 3]
    """
    # 使用bps_torch库中的函数采样单位球面
    bps_points = sample_sphere_uniform(n_points=n_bps_points, radius=radius)
    return bps_points.to(device)

def compute_object_geo_bps(obj_verts, basis_points, device):
    """
    计算物体几何的BPS特征
    
    参数:
        obj_verts: 变换后的物体顶点 [T, N_v, 3] 或 [N_v, 3]
        basis_points: BPS基点 [T, N_b, 3] 或 [N_b, 3]
        device: 计算设备
        
    返回:
        BPS特征 [T, N_b, 4] (距离+单位向量)
    """
    # print(f"计算BPS特征 - 输入形状: obj_verts={obj_verts.shape}, basis_points={basis_points.shape}")
    
    # 确保输入在同一设备上
    obj_verts = obj_verts.to(device)
    basis_points = basis_points.to(device)
    
    # 检查维度并调整
    if obj_verts.dim() == 2:
        # 单个时间步骤
        obj_verts = obj_verts.unsqueeze(0)  # [1, N_v, 3]
    
    if basis_points.dim() == 2:
        basis_points = basis_points.unsqueeze(0)  # [1, N_b, 3]
    
    # 确保时间维度一致
    T = max(obj_verts.shape[0], basis_points.shape[0])
    
    if obj_verts.shape[0] == 1 and T > 1:
        obj_verts = obj_verts.repeat(T, 1, 1)
    
    if basis_points.shape[0] == 1 and T > 1:
        basis_points = basis_points.repeat(T, 1, 1)
    
    # print(f"调整后的形状: obj_verts={obj_verts.shape}, basis_points={basis_points.shape}")
    
    # 初始化结果
    N_b = basis_points.shape[1]
    bps_features = torch.zeros(T, N_b, 4, device=device)
    
    # 为每个时间步骤单独计算BPS特征
    for t in range(T):
        try:
            # 将计算拆分为批次以避免内存问题
            batch_size = 1000
            for b_start in range(0, N_b, batch_size):
                b_end = min(b_start + batch_size, N_b)
                basis_batch = basis_points[t, b_start:b_end]
                
                # 计算从每个基点到每个顶点的距离
                diff = basis_batch.unsqueeze(1) - obj_verts[t].unsqueeze(0)  # [batch, N_v, 3]
                dist = torch.norm(diff, dim=2)  # [batch, N_v]
                
                # 找到每个基点的最近顶点
                min_dist, min_idx = torch.min(dist, dim=1)  # [batch]
                
                # 直接使用索引来获取最近点的差异向量，而不是使用gather
                nearest_vectors = torch.zeros((min_idx.shape[0], 3), device=device)
                for i in range(min_idx.shape[0]):
                    nearest_vectors[i] = diff[i, min_idx[i]]
                
                # 计算单位向量 (如果距离不为零)
                unit_vectors = torch.zeros_like(nearest_vectors)
                non_zero_mask = min_dist > 1e-6
                if non_zero_mask.any():
                    # 对非零距离计算单位向量
                    for i in range(min_idx.shape[0]):
                        if min_dist[i] > 1e-6:
                            unit_vectors[i] = nearest_vectors[i] / min_dist[i]
                
                # 存储特征 [距离, 单位向量]
                bps_features[t, b_start:b_end, 0] = min_dist
                bps_features[t, b_start:b_end, 1:] = unit_vectors
            
            # print(f"时间步骤 {t}: BPS计算完成，特征形状={bps_features[t].shape}")
            
            # 检查无效值
            if torch.isnan(bps_features[t]).any() or torch.isinf(bps_features[t]).any():
                print(f"警告: 时间步骤 {t} 中存在NaN或Inf值")
                # 替换无效值
                bps_features[t] = torch.nan_to_num(bps_features[t], nan=0.0, posinf=1e6, neginf=-1e6)
        
        except Exception as e:
            print(f"计算时间步骤 {t} 的BPS特征时出错: {e}")
            import traceback
            traceback.print_exc()
            # 对失败的时间步使用零值
            bps_features[t] = torch.zeros(N_b, 4, device=device)
    
    return bps_features

def apply_transformation_to_obj(obj_verts, obj_scale, obj_rot, obj_trans):
    """
    应用变换到物体顶点
    
    参数:
        obj_verts: 原始物体顶点 [N, 3] 或 [1, N, 3]
        obj_scale: 缩放因子 [T] 或标量
        obj_rot: 旋转矩阵 [T, 3, 3]
        obj_trans: 平移向量 [T, 3]
        
    返回:
        变换后的物体顶点 [T, N, 3]
    """
    T = obj_rot.shape[0]
    device = obj_rot.device
    
    # 确保obj_verts有正确的维度
    if obj_verts.dim() == 2:
        # 从[N, 3]转换为[1, N, 3]
        obj_verts = obj_verts.unsqueeze(0)  
    
    # 复制到T个时间步
    if obj_verts.shape[0] == 1 and T > 1:
        obj_verts = obj_verts.repeat(T, 1, 1)  # [T, N, 3]
    
    N = obj_verts.shape[1]  # 顶点数量
    
    # 应用缩放 - 确保广播维度正确
    if isinstance(obj_scale, torch.Tensor) and obj_scale.dim() > 0:
        # [T] -> [T, 1, 1] 以便正确广播到 [T, N, 3]
        obj_scale_expanded = obj_scale.view(T, 1, 1)
        scaled_verts = obj_verts * obj_scale_expanded
    else:
        scaled_verts = obj_verts * obj_scale
    
    # 创建输出张量
    rotated_verts = torch.zeros_like(scaled_verts)
    
    # 对每个时间步单独应用旋转
    for t in range(T):
        # [N, 3] @ [3, 3] -> [N, 3]
        rotated_verts[t] = scaled_verts[t] @ obj_rot[t].transpose(0, 1)
    
    # 应用平移 - 确保广播维度正确
    # [T, 3] -> [T, 1, 3] 以便正确广播到 [T, N, 3]
    obj_trans_expanded = obj_trans.view(T, 1, 3)
    transformed_verts = rotated_verts + obj_trans_expanded
    
    return transformed_verts

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
    obj_accel = _syn_acc(obj_trans, smooth_n=4)  # [T, 3]
    # # 计算角速度 - 使用矩阵运算
    # obj_angular_vel = torch.zeros_like(obj_trans)
    # delta_t = 1.0 / FRAME_RATE  # 假设60帧每秒
    # if T > 1:
    #     # 计算相对旋转
    #     rot_t = obj_rot_mat[1:]  # [T-1, 3, 3]
    #     rot_tm1 = obj_rot_mat[:-1]  # [T-1, 3, 3]
    #     rot_tm1_inv = torch.inverse(rot_tm1)  # [T-1, 3, 3]
    #     rel_rot = torch.bmm(rot_t, rot_tm1_inv)  # [T-1, 3, 3]
    #     axis_angle = transforms.matrix_to_axis_angle(rel_rot)  # [T-1, 3]
    #     obj_angular_vel[1:] = axis_angle / delta_t  # [T-1, 3]
    # 返回物体IMU数据
    return {
        'accelerations': obj_accel.reshape(T, 1, 3),  # [T, 1, 3]
        'orientations': obj_rot_mat.reshape(T, 1, 3, 3)  # [T, 1, 3, 3]
    }

def process_sequence(seq_data, seq_key, save_dir, bm, device='cuda', bps_dir=None, bps_points=None, obj_mesh_dir=None):
    """处理单个序列并保存为pt文件"""
    
    # 提取序列数据
    seq_name = seq_data['seq_name']
    bdata_poses = np.concatenate([seq_data['root_orient'].reshape(-1, 3), 
                                 seq_data['pose_body'].reshape(-1, 63)], axis=1)
    bdata_trans = seq_data['trans']
    subject_gender = seq_data['gender']
    
    # 确保为字符串类型
    if not isinstance(seq_name, str):
        seq_name = str(seq_name)
    if not isinstance(subject_gender, str):
        subject_gender = str(subject_gender)
    
    # 构建body参数字典
    body_parms = {
        "root_orient": torch.tensor(seq_data['root_orient'], device=device).float(),
        "pose_body": torch.tensor(seq_data['pose_body'], device=device).float(),
        "trans": torch.tensor(seq_data['trans'], device=device).float(),
        "betas": torch.tensor(seq_data['betas'], device=device).float() if 'betas' in seq_data else None
    }
    
    # 使用SMPL模型获取全局姿态
    body_pose_world = bm(
        **{k: v for k, v in body_parms.items() 
           if k in ["pose_body", "root_orient", "trans"] and v is not None}
    )
    
    # 计算局部旋转的6D表示 - 使用pytorch3d
    local_rot_mat = transforms.axis_angle_to_matrix(
        torch.tensor(bdata_poses, device=device).float().reshape(-1, 3)
    ).reshape(bdata_poses.shape[0], -1, 3, 3)
    
    # 计算6D表示
    rotation_local_6d = transforms.matrix_to_rotation_6d(local_rot_mat)
    rotation_local_full_gt_list = rotation_local_6d.reshape(bdata_poses.shape[0], -1)
    
    # 只使用前22个关节的层次结构
    kintree_table = bm.kintree_table[0].long()[:22]  # 只取前22个关节的父子关系
    kintree_table[0] = -1
    
    # 计算全局旋转矩阵
    rotation_global_matrot = local2global_pose(
        local_rot_mat.reshape(local_rot_mat.shape[0], -1, 9), kintree_table
    )
    
    # 重塑全局旋转矩阵为标准形状 [T, J, 3, 3]
    rotation_global_matrot_reshaped = rotation_global_matrot.reshape(rotation_global_matrot.shape[0], -1, 3, 3)
    
    # 提取头部全局旋转
    head_rotation_global_matrot = rotation_global_matrot[:, [15], :, :]
    
    # 获取全局关节位置
    position_global_full_gt_world = body_pose_world.Jtr[:, :22, :].cpu()
    
    # 计算头部全局变换矩阵
    position_head_world = position_global_full_gt_world[:, 15, :].to(device)
    head_global_trans = torch.eye(4, device=device).repeat(position_head_world.shape[0], 1, 1)
    head_global_trans[:, :3, :3] = head_rotation_global_matrot.squeeze()
    head_global_trans[:, :3, 3] = position_head_world
    
    # 计算IMU数据 (现在是加速度和方向)
    imu_global_full_gt = compute_imu_data(
        position_global_full_gt_world.to(device), 
        rotation_global_matrot_reshaped.to(device),
        IMU_JOINTS,
        smooth_n=4
    )
    # 归一化IMU数据到头部坐标系
    # head_accel = imu_global_full_gt['accelerations'][:, HEAD_IDX:HEAD_IDX+1]  # [T, 1, 3]
    # head_ori = imu_global_full_gt['orientations'][:, HEAD_IDX:HEAD_IDX+1]  # [T, 1, 3, 3]
    # imu_global_exp_head_gt = {k: v[:, :HEAD_IDX] for k, v in imu_global_full_gt.items()}
    # norm_imu_global_full_gt = normalize_to_head_frame(imu_global_exp_head_gt, head_imu_data=(head_accel, head_ori))
    # norm_imu_acc = torch.cat([norm_imu_global_full_gt['accelerations'], head_accel], dim=1).bmm(head_ori[:, -1])
    # norm_imu_ori = torch.cat([norm_imu_global_full_gt['orientations'], head_ori], dim=1)
    # processed_imu_global_full_gt = {
    #     'accelerations': norm_imu_acc,
    #     'orientations': norm_imu_ori
    # }

    # --- 或者，如果不进行归一化，直接使用全局数据 ---
    processed_imu_global_full_gt = imu_global_full_gt # 使用未归一化的数据
    # -------------------------------------------------

    # 提取物体相关信息(如果存在)
    obj_data = {}
    transformed_verts = None # 初始化以备后用
    if 'obj_scale' in seq_data:
        # 将物体数据转移到设备
        obj_scale = torch.tensor(seq_data['obj_scale'], device=device).float()
        if seq_data['obj_trans'].shape[-1] > 1:
            obj_trans = torch.tensor(seq_data['obj_trans'][:, :, 0], device=device).float()
        else:
            obj_trans = torch.tensor(seq_data['obj_trans'], device=device).float()
        obj_rot = torch.tensor(seq_data['obj_rot'], device=device).float()
        obj_com_pos = torch.tensor(seq_data['obj_com_pos'], device=device).float()
        T = obj_trans.shape[0] # 获取时间步长

        # 计算物体的IMU数据 (现在是加速度和方向)
        obj_imu_data = compute_object_imu(obj_trans, obj_rot)
        # norm_obj_imu_data = normalize_to_head_frame(obj_imu_data, head_imu_data=(head_accel, head_ori))
        # norm_obj_imu_data['accelerations'] = norm_obj_imu_data['accelerations'].bmm(head_ori[:, -1])
        # processed_obj_imu_data = norm_obj_imu_data
        # --- 或者，如果不进行归一化，直接使用全局数据 ---
        processed_obj_imu_data = obj_imu_data # 使用未归一化的数据
        # -------------------------------------------------

        # 提取物体名称
        object_name = seq_name.split("_")[1] if "_" in seq_name else "unknown"

        # --- 添加手部-物体接触计算 ---
        lhand_contact = torch.zeros(T, dtype=torch.bool, device=device)
        rhand_contact = torch.zeros(T, dtype=torch.bool, device=device)
        obj_contact = torch.zeros(T, dtype=torch.bool, device=device)

        if obj_mesh_dir:
            try:
                # 加载物体网格
                obj_mesh_path = os.path.join(obj_mesh_dir, f"{object_name}.obj")
                if not os.path.exists(obj_mesh_path):
                    obj_mesh_path = os.path.join(obj_mesh_dir, f"{object_name}_cleaned_simplified.obj")

                if os.path.exists(obj_mesh_path):
                    # 加载网格
                    mesh = trimesh.load_mesh(obj_mesh_path)
                    obj_mesh_verts = torch.tensor(mesh.vertices, device=device).float() #物体顶点维度: torch.Size([N_v, 3])

                    # 确保物体顶点是二维数组 [N_v, 3]
                    if obj_mesh_verts.dim() > 2:
                        obj_mesh_verts = obj_mesh_verts.reshape(-1, 3)

                    # 应用变换到物体网格
                    transformed_verts = apply_transformation_to_obj(
                        obj_mesh_verts, obj_scale, obj_rot, obj_trans
                    ) # [T, N_v, 3]

                    # 提取手部关节位置 (使用索引 20 和 21)
                    lhand_jnt = position_global_full_gt_world[:, 20, :].to(device) # [T, 3]
                    rhand_jnt = position_global_full_gt_world[:, 21, :].to(device) # [T, 3]

                    num_obj_verts = transformed_verts.shape[1]

                    # 计算距离 (可以使用更高效的 cdist)
                    # [T, 1, 3] vs [T, N_v, 3] -> [T, 1, N_v] -> [T, N_v]
                    lhand2obj_dist = torch.cdist(lhand_jnt.unsqueeze(1), transformed_verts).squeeze(1)
                    rhand2obj_dist = torch.cdist(rhand_jnt.unsqueeze(1), transformed_verts).squeeze(1)

                    # 找出最小距离
                    lhand2obj_dist_min = torch.min(lhand2obj_dist, dim=1)[0] # [T]
                    rhand2obj_dist_min = torch.min(rhand2obj_dist, dim=1)[0] # [T]

                    # 设置接触阈值 (参考 evaluation_metrics.py, use_joints24=False)
                    contact_threh = 0.10

                    # 判断接触状态
                    lhand_contact = (lhand2obj_dist_min < contact_threh)
                    rhand_contact = (rhand2obj_dist_min < contact_threh)
                    obj_contact = lhand_contact | rhand_contact

                else:
                    print(f"警告: 未找到物体网格文件以计算接触: {obj_mesh_path}")
            except Exception as e:
                print(f"警告: 计算手部-物体接触时出错: {e}")
                import traceback
                traceback.print_exc()
        # --- 手部-物体接触计算结束 ---

        # 构建物体数据字典
        obj_data = {
            "obj_name": object_name,
            "obj_scale": obj_scale.cpu(),
            "obj_trans": obj_trans.cpu(),
            "obj_rot": obj_rot.cpu(),
            "obj_com_pos": obj_com_pos.cpu(),
            "obj_imu": {
                "accelerations": processed_obj_imu_data["accelerations"].cpu(),    # [T, 1, 3]
                "orientations": processed_obj_imu_data["orientations"].cpu()  # [T, 1, 3, 3]
            },
            # "bps_file": f"{seq_name}_{seq_key}.npy"  # 存储BPS文件路径的相对引用
            # --- 添加接触信息到obj_data ---
            "lhand_contact": lhand_contact.cpu(), # [T] bool
            "rhand_contact": rhand_contact.cpu(), # [T] bool
            "obj_contact": obj_contact.cpu() # [T] bool
            # -----------------------------
        }
    
    # 组装输出数据
    data = {
        "seq_name": seq_name,
        "body_parms_list": {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in body_parms.items()},
        "rotation_local_full_gt_list": rotation_local_full_gt_list.cpu(),
        "head_global_trans": head_global_trans.cpu(),
        "position_global_full_gt_world": position_global_full_gt_world.float(),
        "imu_global_full_gt": {
            "accelerations": processed_imu_global_full_gt["accelerations"].cpu(),    # [T, num_imus, 3]
            "orientations": processed_imu_global_full_gt["orientations"].cpu()  # [T, num_imus, 3, 3]
        },
        "framerate": FRAME_RATE,
        "gender": subject_gender
    }
    
    # 添加物体数据(如果存在)
    if obj_data:
        data.update(obj_data)
    
    # 保存处理后的数据
    torch.save(data, os.path.join(save_dir, f"{seq_key}.pt"))
    return 1

def preprocess_amass(args):
    """
    处理AMASS数据集，提取姿态、平移和IMU数据
    
    参数:
        args: 命令行参数，包含数据路径和保存目录
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"处理AMASS数据集，使用设备: {device}")
    
    # 加载SMPL模型
    bm_fname_male = os.path.join(args.support_dir, f"smplh/male/model.npz")
    body_model = BodyModel(
        bm_fname=bm_fname_male,
        num_betas=16,
    ).to(device)
    
    # 创建保存目录
    amass_dir = args.save_dir_train
    os.makedirs(amass_dir, exist_ok=True)
    
    # 定义AMASS数据集路径
    amass_dirs = []
    if hasattr(args, "amass_dirs") and args.amass_dirs:
        amass_dirs = args.amass_dirs
    else:
        amass_dirs = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh', 'Transitions_mocap', 'SSM_synced', 'CMU',
              'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BMLmovi', 'EKUT', 'TCD_handMocap', 'ACCAD',
              'BioMotionLab_NTroje', 'BMLhandball', 'MPI_Limits', 'DFaust67']
    
    # 创建单独的聚合数据文件
    sequence_index = 0
    amass_summary = {'datasets': [], 'total_sequences': 0, 'total_frames': 0}
    
    # 遍历所有数据集目录
    for ds_name in amass_dirs:
        print(f'读取AMASS子数据集: {ds_name}')
        ds_path = os.path.join(args.amass_dir, ds_name)
        ds_summary = {'name': ds_name, 'sequences': 0, 'frames': 0}
        
        if not os.path.exists(ds_path):
            print(f"警告: 找不到数据集 {ds_path}")
            continue
        
        # 处理每个数据集中的每个文件
        for npz_fname in tqdm(glob.glob(os.path.join(ds_path, '*/*_poses.npz'))):
            try:
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 加载单个文件
                cdata = np.load(npz_fname)
                
                # 根据帧率调整采样
                framerate = int(cdata['mocap_framerate'])
                if framerate == 120: 
                    step = 2
                elif framerate == 60 or framerate == 59: 
                    step = 1
                else: 
                    continue  # 跳过其他帧率的数据
                
                # 获取姿势、平移和体形参数
                poses_data = cdata['poses'][::step].astype(np.float32)
                trans_data = cdata['trans'][::step].astype(np.float32)
                betas_data = cdata['betas'][:10].astype(np.float32)
                
                # 检查序列长度
                seq_len = poses_data.shape[0]
                if seq_len <= 60:
                    print(f"\t丢弃太短的序列: {npz_fname}，长度为 {seq_len}")
                    continue
                
                # 转换为PyTorch张量
                pose = torch.tensor(poses_data, device=device).view(-1, 52, 3)
                tran = torch.tensor(trans_data, device=device)
                shape = torch.tensor(betas_data, device=device).unsqueeze(0)  # 添加批次维度
                
                # 只使用身体的22个关节，不包括手指
                pose = pose[:, :22].clone()  # 只使用躯干
                
                # # 对齐AMASS全局坐标系
                # amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]], device=device)
                # tran = torch.matmul(amass_rot, tran.unsqueeze(-1)).squeeze(-1)
                
                # # 对根关节方向进行转换
                # root_orient = pose[:, 0]
                # root_rotmat = transforms.axis_angle_to_matrix(root_orient)
                # aligned_root_rotmat = torch.matmul(amass_rot, root_rotmat)
                # pose[:, 0] = transforms.matrix_to_axis_angle(aligned_root_rotmat)
                
                # 将轴角转换为旋转矩阵
                pose_rotmat = transforms.axis_angle_to_matrix(pose.reshape(-1, 3)).reshape(seq_len, 22, 3, 3)
                
                # 通过body model计算全局关节位置和旋转
                body_output = body_model(
                    pose_body=pose[:, 1:22].reshape(-1, 63),  # 身体姿态
                    root_orient=pose[:, 0],                  # 根关节方向
                    trans=tran                               # 平移
                )
                
                # 获取关节位置和全局旋转
                joints = body_output.Jtr[:, :22, :]  # 关节位置
                
                # 计算全局旋转矩阵
                kintree_table = body_model.kintree_table[0].long()[:22]
                global_rotmat = local2global_pose(
                    pose_rotmat.reshape(seq_len, -1, 9),  # 重塑为 [seq_len, joints, 9]
                    kintree_table
                ).reshape(seq_len, 22, 3, 3)
                
                # 计算IMU数据 (加速度和方向)
                imu_data = compute_imu_data(
                    joints.cpu(),                 # 关节位置
                    global_rotmat.cpu(),          # 全局旋转
                    IMU_JOINTS,                  # IMU关节索引
                    smooth_n=4                    # 平滑窗口大小
                )

                # 计算头部全局变换矩阵
                position_head_world = joints[:, 15, :].to(device)
                head_global_trans = torch.eye(4, device=device).repeat(position_head_world.shape[0], 1, 1)
                head_global_trans[:, :3, :3] = global_rotmat[:, 15, :, :].squeeze()
                head_global_trans[:, :3, 3] = position_head_world
                
                # 创建和保存单个序列数据
                seq_data = {
                    "seq_name": f"amass_{ds_name}_{sequence_index}",
                    "rotation_local_full_gt_list": transforms.matrix_to_rotation_6d(pose_rotmat.cpu()).reshape(seq_len, -1),
                    "position_global_full_gt_world": joints.cpu(),  # 使用实际计算的关节位置
                    "head_global_trans": head_global_trans.cpu(),
                    "imu_global_full_gt": {
                        "accelerations": imu_data['accelerations'],
                        "orientations": imu_data['orientations']
                    },
                    "framerate": FRAME_RATE,
                    "gender": "neutral"
                }
                
                # 保存这个序列
                seq_file = os.path.join(amass_dir, f"seq_{sequence_index}.pt")
                torch.save(seq_data, seq_file)
                
                # 更新统计信息
                sequence_index += 1
                ds_summary['sequences'] += 1
                ds_summary['frames'] += seq_len
                amass_summary['total_sequences'] += 1
                amass_summary['total_frames'] += seq_len
                
            except Exception as e:
                print(f"处理文件时出错 {npz_fname}: {e}")
                import traceback
                traceback.print_exc()
        
        # 添加数据集摘要
        amass_summary['datasets'].append(ds_summary)
        print(f"完成处理数据集 {ds_name}: {ds_summary['sequences']} 个序列, {ds_summary['frames']} 帧")
    
    print(f"AMASS数据处理完成，共处理了 {amass_summary['total_sequences']} 个序列")
    print(f"预处理后的AMASS数据集保存在 {amass_dir}")

def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置SMPL模型
    print("加载SMPL模型...")
    bm_fname_male = os.path.join(args.support_dir, f"smplh/male/model.npz")
    
    num_betas = 16  # 身体参数数量
    bm_male = BodyModel(
        bm_fname=bm_fname_male,
        num_betas=num_betas,
    ).to(device)

    # 创建输出目录
    os.makedirs(args.save_dir_train, exist_ok=True)
    os.makedirs(args.save_dir_test, exist_ok=True)
    
    # # 创建BPS目录
    # bps_dir = os.path.join(args.save_dir, "bps_features")
    # os.makedirs(bps_dir, exist_ok=True)
    # 准备BPS基础点云
    # print("准备BPS基础点云...")
    # bps_points = prep_bps_data(n_bps_points=args.n_bps_points, radius=args.bps_radius, device=device)

    # 处理AMASS数据集
    if args.process_amass:
        preprocess_amass(args)
    
    # 加载数据集
    else:
        print(f"正在加载数据集：{args.data_path_train}")
        data_dict_train = joblib.load(args.data_path_train)
        print(f"数据集加载完成，共有{len(data_dict_train)}个序列")
        
        print(f"正在加载数据集：{args.data_path_test}")
        data_dict_test = joblib.load(args.data_path_test)
        print(f"数据集加载完成，共有{len(data_dict_test)}个序列")
        
        # 处理所有序列
        print("开始处理序列...")
        
        for seq_key in tqdm(data_dict_train, desc="处理序列"):
            process_sequence(data_dict_train[seq_key], seq_key, args.save_dir_train, bm_male, device=device, obj_mesh_dir=args.obj_mesh_dir)
        
        print(f"所有序列处理完成，结果保存在：{args.save_dir_train}")

        for seq_key in tqdm(data_dict_test, desc="处理序列"):
            process_sequence(data_dict_test[seq_key], seq_key, args.save_dir_test, bm_male, device=device, obj_mesh_dir=args.obj_mesh_dir)
        
        print(f"所有序列处理完成，结果保存在：{args.save_dir_test}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理人体动作数据集")
    parser.add_argument("--data_path_train", type=str, default="dataset/train_diffusion_manip_seq_joints24.p",
                        help="输入数据集路径(.p文件)")
    parser.add_argument("--data_path_test", type=str, default="dataset/test_diffusion_manip_seq_joints24.p",
                        help="输入数据集路径(.p文件)")
    parser.add_argument("--save_dir_train", type=str, default="processed_data_0429/train",
                        help="输出数据保存目录")
    parser.add_argument("--save_dir_test", type=str, default="processed_data_0429/test",
                        help="输出数据保存目录")
    parser.add_argument("--support_dir", type=str, default="body_models",
                        help="SMPL模型目录")
    parser.add_argument("--obj_mesh_dir", type=str, default="dataset/captured_objects",
                        help="物体网格目录")
    parser.add_argument("--smooth_n", type=int, default=4,
                        help="IMU加速度平滑窗口大小")
    # parser.add_argument("--n_bps_points", type=int, default=1024,
    #                     help="BPS点云中的点数量")
    # parser.add_argument("--bps_radius", type=float, default=1.0,
    #                     help="BPS点云的球体半径")
    parser.add_argument("--process_amass", action="store_true",
                        help="是否处理AMASS数据集")
    parser.add_argument("--amass_dir", type=str, default="/mnt/d/a_WORK/Projects/PhD/datasets/AMASS_SMPL_H",
                        help="AMASS数据集路径")
    parser.add_argument("--amass_dirs", type=str, nargs="+",
                        help="AMASS子数据集列表，例如CMU BMLmovi")
    
    args = parser.parse_args()
    main(args)