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

# 导入BPS相关库，如果不存在需要安装
try:
    from bps_torch.bps import bps_torch
    from bps_torch.tools import sample_sphere_uniform
except ImportError:
    print("警告: bps_torch库未安装，请安装后再运行")

# IMU关节索引，可以根据需要修改
IMU_JOINTS = [20, 21, 15, 7, 8, 0]  # 左手、右手、头部、左脚、右脚、髋部
IMU_JOINT_NAMES = ['left_hand', 'right_hand', 'head', 'left_foot', 'right_foot', 'hip']
HEAD_IDX = 2  # 头部在IMU_JOINTS列表中的索引
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

def compute_imu_data(position_global, rotation_global, imu_joints, smooth_n=4):
    """
    计算特定关节的IMU数据（只包括加速度和角速度），使用矩阵运算优化
    
    参数:
        position_global: 全局关节位置 [T, J, 3]
        rotation_global: 全局关节旋转 [T, J, 3, 3]
        imu_joints: IMU关节索引列表
        smooth_n: 平滑窗口大小
    
    返回:
        IMU数据字典，仅包含加速度和角速度信息
    """
    device = position_global.device
    
    # 提取指定关节的位置和旋转
    imu_positions = position_global[:, imu_joints, :]  # [T, num_imus, 3]
    imu_rotations = rotation_global[:, imu_joints, :, :]  # [T, num_imus, 3, 3]
    
    T = imu_positions.shape[0]
    num_imus = len(imu_joints)
    
    
    # 并行计算所有IMU关节的加速度
    imu_accelerations = torch.zeros_like(imu_positions)
    for i in range(num_imus):
        imu_accelerations[:, i, :] = _syn_acc(imu_positions[:, i, :], smooth_n=4)
    
    # 计算角速度 - 使用矩阵运算优化
    angular_velocities = torch.zeros_like(imu_positions)
    delta_t = 1.0 / FRAME_RATE  # 假设60帧每秒
    
    if T > 1:
        # 构建t和t-1时刻的旋转矩阵
        rot_t = imu_rotations[1:].reshape(-1, 3, 3)  # [(T-1)*num_imus, 3, 3]
        rot_tm1 = imu_rotations[:-1].reshape(-1, 3, 3)  # [(T-1)*num_imus, 3, 3]
        
        # 计算相对旋转: R_rel = R_t * R_(t-1)^(-1)
        rot_tm1_inv = torch.inverse(rot_tm1)  # [(T-1)*num_imus, 3, 3]
        rel_rot = torch.bmm(rot_t, rot_tm1_inv)  # [(T-1)*num_imus, 3, 3]
        
        # 将旋转矩阵转换为轴角表示
        axis_angle = transforms.matrix_to_axis_angle(rel_rot)  # [(T-1)*num_imus, 3]
        
        # 角速度 = 轴角 / 时间间隔
        angular_vel = axis_angle / delta_t  # [(T-1)*num_imus, 3]
        
        # 重新整形为 [T-1, num_imus, 3]
        angular_vel = angular_vel.reshape(T-1, num_imus, 3)
        
        # 将计算结果填入输出张量的对应位置
        angular_velocities[1:] = angular_vel
    
    # 返回IMU数据字典，只包含加速度和角速度
    imu_data = {
        'accelerations': imu_accelerations,  # [T, num_imus, 3]
        'angular_velocities': angular_velocities  # [T, num_imus, 3]
    }
    
    return imu_data

def normalize_to_head_frame(imu_data):
    """
    将IMU数据归一化到头部坐标系
    
    参数:
        imu_data: 包含加速度和角速度的字典
            - accelerations: [T, num_imus, 3]
            - angular_velocities: [T, num_imus, 3]
    
    返回:
        归一化后的IMU数据字典
    """
    # 复制数据以避免修改原始数据
    norm_imu_data = {
        'accelerations': imu_data['accelerations'].clone(),
        'angular_velocities': imu_data['angular_velocities'].clone()
    }
    
    # 获取头部IMU数据
    head_accel = norm_imu_data['accelerations'][:, HEAD_IDX:HEAD_IDX+1]  # [T, 1, 3]
    head_gyro = norm_imu_data['angular_velocities'][:, HEAD_IDX:HEAD_IDX+1]  # [T, 1, 3]
    
    # 所有IMU数据相对于头部IMU
    norm_imu_data['accelerations'] = norm_imu_data['accelerations'] - head_accel
    norm_imu_data['angular_velocities'] = norm_imu_data['angular_velocities'] - head_gyro
    
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
    计算物体的IMU数据（加速度和角速度），使用矩阵运算优化
    
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
    
    # 计算角速度 - 使用矩阵运算
    obj_angular_vel = torch.zeros_like(obj_trans)
    delta_t = 1.0 / FRAME_RATE  # 假设60帧每秒
    
    if T > 1:
        # 计算相对旋转
        rot_t = obj_rot_mat[1:]  # [T-1, 3, 3]
        rot_tm1 = obj_rot_mat[:-1]  # [T-1, 3, 3]
        rot_tm1_inv = torch.inverse(rot_tm1)  # [T-1, 3, 3]
        rel_rot = torch.bmm(rot_t, rot_tm1_inv)  # [T-1, 3, 3]
        
        # 将旋转矩阵转换为轴角表示
        axis_angle = transforms.matrix_to_axis_angle(rel_rot)  # [T-1, 3]
        
        # 角速度 = 轴角 / 时间间隔
        obj_angular_vel[1:] = axis_angle / delta_t  # [T-1, 3]
    
    # 返回物体IMU数据
    return {
        'accelerations': obj_accel,  # [T, 3]
        'angular_velocities': obj_angular_vel  # [T, 3]
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
    rotation_local_full_gt_list = rotation_local_6d[1:].reshape(bdata_poses.shape[0] - 1, -1)
    
    # 只使用前22个关节的层次结构
    kintree_table = bm.kintree_table[0].long()[:22]  # 只取前22个关节的父子关系
    
    # 计算全局旋转矩阵
    rotation_global_matrot = local2global_pose(
        local_rot_mat.reshape(local_rot_mat.shape[0], -1, 9)[:, :22], kintree_table
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
    head_global_trans_list = head_global_trans[1:]
    
    # 计算IMU数据
    imu_global_full_gt = compute_imu_data(
        position_global_full_gt_world.to(device), 
        rotation_global_matrot_reshaped.to(device),
        IMU_JOINTS,
        smooth_n=4
    )
    
    # 归一化IMU数据到头部坐标系
    norm_imu_global_full_gt = normalize_to_head_frame(imu_global_full_gt)
    
    # 提取物体相关信息(如果存在)
    obj_data = {}
    if 'obj_scale' in seq_data:
        # 将物体数据转移到设备
        obj_scale = torch.tensor(seq_data['obj_scale'], device=device).float()
        if seq_data['obj_trans'].shape[-1] > 1:
            obj_trans = torch.tensor(seq_data['obj_trans'][:, :, 0], device=device).float()
        else:
            obj_trans = torch.tensor(seq_data['obj_trans'], device=device).float()
        obj_rot = torch.tensor(seq_data['obj_rot'], device=device).float()
        obj_com_pos = torch.tensor(seq_data['obj_com_pos'], device=device).float()
        
        # 计算物体的IMU数据
        obj_imu_data = compute_object_imu(obj_trans, obj_rot)
        
        # # 提取物体名称
        # object_name = seq_name.split("_")[1] if "_" in seq_name else "unknown"
        
        # # 计算BPS特征 (如果提供了物体网格目录)
        # bps_save_path = os.path.join(bps_dir, f"{seq_name}_{seq_key}.npy")
        # if obj_mesh_dir and not os.path.exists(bps_save_path):
        #     try:
        #         # 加载物体网格
        #         obj_mesh_path = os.path.join(obj_mesh_dir, f"{object_name}.obj")
        #         if not os.path.exists(obj_mesh_path):
        #             obj_mesh_path = os.path.join(obj_mesh_dir, f"{object_name}_cleaned_simplified.obj")
                
        #         if os.path.exists(obj_mesh_path):
        #             # 加载网格
        #             mesh = trimesh.load_mesh(obj_mesh_path)
        #             obj_mesh_verts = torch.tensor(mesh.vertices, device=device).float() #物体顶点维度: torch.Size([17996, 3]), 
        #             # 物体变换维度: 缩放=torch.Size([T]), 旋转=torch.Size([T, 3, 3]), 平移=torch.Size([T, 3, 1])
                    
        #             # 确保物体顶点是二维数组 [N, 3]
        #             if obj_mesh_verts.dim() > 2:
        #                 obj_mesh_verts = obj_mesh_verts.reshape(-1, 3)
                    
        #             # 确保变换维度匹配
        #             T = obj_trans.shape[0]
                    
        #             # 应用变换到物体网格
        #             transformed_verts = apply_transformation_to_obj(
        #                 obj_mesh_verts, obj_scale, obj_rot, obj_trans
        #             ) # [T, 17996, 3]
                    
        #             # 计算物体中心
        #             center_verts = transformed_verts.mean(dim=1)  # [T, 3] [T, 3]
                    
        #             # 准备自定义基点 (将BPS点移动到物体中心)
        #             custom_basis = bps_points.unsqueeze(0).repeat(T, 1, 1)
                    
        #             # 应用物体中心偏移 (确保广播维度正确)
        #             custom_basis = custom_basis + center_verts.unsqueeze(1) # [T, 1024, 3]
                    
        #             # 计算BPS特征
        #             bps_features = compute_object_geo_bps(transformed_verts, custom_basis, device)
                    
        #             # 确保特征形状一致
        #             if bps_features.shape[0] != T:
        #                 print(f"警告: BPS特征时间步数与原始数据不一致，调整形状 {bps_features.shape[0]} -> {T}")
        #                 if bps_features.shape[0] > T:
        #                     bps_features = bps_features[:T]
        #                 else:
        #                     # 填充缺失的时间步
        #                     pad = torch.zeros(T - bps_features.shape[0], bps_features.shape[1], bps_features.shape[2], device=device)
        #                     bps_features = torch.cat([bps_features, pad], dim=0)
                    
        #             # 保存BPS特征
        #             np.save(bps_save_path, bps_features.cpu().numpy())
        #             print(f"已保存BPS特征到: {bps_save_path}")
        #         else:
        #             print(f"警告: 未找到物体网格文件: {obj_mesh_path}")
        #     except Exception as e:
        #         print(f"警告: 计算BPS特征时出错: {e}")
        #         import traceback
        #         traceback.print_exc()
        
        # 构建物体数据字典
        obj_data = {
            "obj_scale": obj_scale.cpu(),
            "obj_trans": obj_trans.cpu(),
            "obj_rot": obj_rot.cpu(),
            "obj_com_pos": obj_com_pos.cpu(),
            "obj_imu": {
                "accelerations": obj_imu_data["accelerations"].cpu(),
                "angular_velocities": obj_imu_data["angular_velocities"].cpu()
            },
            # "bps_file": f"{seq_name}_{seq_key}.npy"  # 存储BPS文件路径的相对引用
        }
    
    # 组装输出数据
    data = {
        "seq_name": seq_name,
        "body_parms_list": {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in body_parms.items()},
        "rotation_local_full_gt_list": rotation_local_full_gt_list.cpu(),
        "head_global_trans_list": head_global_trans_list.cpu(),
        "position_global_full_gt_world": position_global_full_gt_world[1:].float(),
        "imu_global_full_gt": {
            "accelerations": norm_imu_global_full_gt["accelerations"].cpu(),
            "angular_velocities": norm_imu_global_full_gt["angular_velocities"].cpu()
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

def main(args):
    # 创建输出目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # # 创建BPS目录
    # bps_dir = os.path.join(args.save_dir, "bps_features")
    # os.makedirs(bps_dir, exist_ok=True)
    # 准备BPS基础点云
    # print("准备BPS基础点云...")
    # bps_points = prep_bps_data(n_bps_points=args.n_bps_points, radius=args.bps_radius, device=device)
    
    # 加载数据集
    print(f"正在加载数据集：{args.data_path}")
    data_dict = joblib.load(args.data_path)
    print(f"数据集加载完成，共有{len(data_dict)}个序列")
    
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
    
    
    # 处理所有序列
    print("开始处理序列...")
    
    for seq_key in tqdm(data_dict, desc="处理序列"):
        process_sequence(data_dict[seq_key], seq_key, args.save_dir, bm_male, device=device, obj_mesh_dir=args.obj_mesh_dir)
    
    print(f"所有序列处理完成，结果保存在：{args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理人体动作数据集")
    parser.add_argument("--data_path", type=str, default="dataset/train_diffusion_manip_seq_joints24.p",
                        help="输入数据集路径(.p文件)")
    parser.add_argument("--save_dir", type=str, default="processed_data_0324/train",
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
    
    args = parser.parse_args()
    main(args)