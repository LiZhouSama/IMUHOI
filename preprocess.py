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
# 设置接触阈值 (参考 evaluation_metrics.py, use_joints24=False)
contact_threh = 0.25


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

def apply_transformation_to_obj_geometry(obj_mesh_verts, obj_rot, obj_trans, scale=None, device='cpu'):
    """
    应用变换到物体顶点 (遵循 hand_foot_dataset.py 的逻辑: Rotate -> Scale -> Translate)

    参数:
        obj_mesh_verts: 物体顶点 [Nv, 3] (torch tensor on device)
        obj_rot: 旋转矩阵 [T, 3, 3] (torch tensor on device)
        obj_trans: 平移向量 [T, 3] (torch tensor on device)
        scale: 缩放因子 [T] (torch tensor on device)
        device: 计算设备

    返回:
        transformed_obj_verts: 变换后的顶点 [T, Nv, 3] (torch tensor on device)
    """
    try:
        # 确保输入在正确的设备上且为 float 类型
        obj_mesh_verts = obj_mesh_verts.float().to(device) # Nv X 3
        seq_rot_mat = obj_rot.float().to(device) # T X 3 X 3
        seq_trans = obj_trans.float().to(device) # T X 3
        if scale is not None:
            seq_scale = scale.float().to(device) # T
        else:
            seq_scale = None

        T = seq_trans.shape[0]
        ori_obj_verts = obj_mesh_verts[None].repeat(T, 1, 1) # T X Nv X 3

        # --- 遵循参考代码的顺序：Rotate -> Scale -> Translate ---
        
        # 1. 旋转 (Rotate)
        # Transpose vertices for matmul: T X 3 X Nv
        verts_rotated = torch.bmm(seq_rot_mat, ori_obj_verts.transpose(1, 2))
        # Result shape: T X 3 X Nv

        # 2. 缩放 (Scale)
        # 准备 scale tensor for broadcasting: T -> T X 1 X 1
        if seq_scale is not None:
            scale_factor = seq_scale.unsqueeze(-1).unsqueeze(-1)
            verts_scaled = scale_factor * verts_rotated
        else:
            verts_scaled = verts_rotated # No scaling
        # Result shape: T X 3 X Nv

        # 3. 平移 (Translate)
        # 准备 translation tensor for broadcasting: T X 3 -> T X 3 X 1
        trans_vector = seq_trans.unsqueeze(-1)
        verts_translated = verts_scaled + trans_vector
        # Result shape: T X 3 X Nv

        # 4. Transpose back to T X Nv X 3
        transformed_obj_verts = verts_translated.transpose(1, 2)

    except Exception as e:
        print(f"应用变换到物体几何体失败: {e}")
        import traceback
        traceback.print_exc()
        # 返回设备上的虚拟数据
        transformed_obj_verts = torch.zeros((obj_trans.shape[0] if obj_trans is not None else 1, 1, 3), device=device)

    return transformed_obj_verts


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
    


    # -------------------------------------------------

    # 提取物体相关信息(如果存在)
    obj_data = {}
    transformed_verts = None # 初始化以备后用
    if 'obj_scale' in seq_data:
        # 将物体数据转移到设备
        obj_scale = torch.tensor(seq_data['obj_scale'], device=device).float()
        
        # --- 更健壮地加载和处理 obj_trans --- 
        raw_obj_trans = torch.tensor(seq_data['obj_trans'], device=device).float()
        T_check = raw_obj_trans.shape[0]
        if raw_obj_trans.shape == (T_check, 3):
            obj_trans = raw_obj_trans
        elif raw_obj_trans.shape == (T_check, 3, 1):
            obj_trans = raw_obj_trans.squeeze(-1) # -> [T, 3]
        elif raw_obj_trans.shape == (T_check, 1, 3):
             obj_trans = raw_obj_trans.squeeze(1)  # -> [T, 3]
        elif raw_obj_trans.numel() == T_check * 3:
             print(f"警告: seq {seq_key} 的 obj_trans 形状为 {raw_obj_trans.shape}, 将尝试重塑为 ({T_check}, 3)")
             try:
                 obj_trans = raw_obj_trans.reshape(T_check, 3)
             except Exception as e:
                 raise ValueError(f"无法将 seq {seq_key} 的 obj_trans 形状 {raw_obj_trans.shape} 重塑为 ({T_check}, 3): {e}")
        else:
             # 之前导致错误的情况 (例如 [T, 185, X]) 会在这里触发
             raise ValueError(f"seq {seq_key} 的 obj_trans 形状无法处理: {raw_obj_trans.shape}. 预期形状类似于 ({T_check}, 3), ({T_check}, 3, 1), 或 ({T_check}, 1, 3).")
        # --- obj_trans 处理结束 ---

        # 确保 obj_trans 现在是 [T, 3]
        if obj_trans.shape != (T_check, 3):
            raise ValueError(f"处理后 seq {seq_key} 的 obj_trans 形状为 {obj_trans.shape}, 预期为 ({T_check}, 3)")

        obj_rot = torch.tensor(seq_data['obj_rot'], device=device).float()
        obj_com_pos = torch.tensor(seq_data['obj_com_pos'], device=device).float()
        T = obj_trans.shape[0] # 获取时间步长

        # 提取物体名称
        object_name = seq_name.split("_")[1] if "_" in seq_name else "unknown"

        # --- 计算手部-物体接触 (基于距离) ---
        lhand_contact = torch.zeros(T, dtype=torch.bool, device=device)
        rhand_contact = torch.zeros(T, dtype=torch.bool, device=device)

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
                    transformed_verts = apply_transformation_to_obj_geometry(
                        obj_mesh_verts, obj_rot, obj_trans, scale=obj_scale, device=device
                    ) # [T, N_v, 3]

                    # 提取手部关节位置 (使用索引 20 和 21)
                    lhand_jnt = position_global_full_gt_world[:, 20, :].to(device) # [T, 3]
                    rhand_jnt = position_global_full_gt_world[:, 21, :].to(device) # [T, 3]

                    # 计算距离 (可以使用更高效的 cdist)
                    lhand2obj_dist = torch.cdist(lhand_jnt.unsqueeze(1), transformed_verts).squeeze(1)
                    rhand2obj_dist = torch.cdist(rhand_jnt.unsqueeze(1), transformed_verts).squeeze(1)

                    # 找出最小距离
                    lhand2obj_dist_min = torch.min(lhand2obj_dist, dim=1)[0] # [T]
                    rhand2obj_dist_min = torch.min(rhand2obj_dist, dim=1)[0] # [T]

                    # 判断接触状态 (使用 contact_threh)
                    lhand_contact = (lhand2obj_dist_min < contact_threh)
                    rhand_contact = (rhand2obj_dist_min < contact_threh)

                else:
                    print(f"警告: 未找到物体网格文件以计算手部接触: {obj_mesh_path}")
            except Exception as e:
                print(f"警告: 计算手部-物体接触时出错: {e}")
                import traceback
                traceback.print_exc()
        # --- 手部距离接触计算结束 ---

        # --- 计算物体接触 (基于运动) ---
        # 定义物体运动检测阈值
        trans_change_threshold = 0.005 # 平移变化阈值 (米) - 例如 1cm
        rot_change_threshold = 0.01   # 旋转矩阵差异阈值 (弗罗贝尼乌斯范数) - 需要调整
        obj_contact = torch.zeros(T, dtype=torch.bool, device=device)

        if T > 1: # 至少需要两帧才能计算变化
            for t in range(1, T):
                trans_diff = torch.norm(obj_trans[t] - obj_trans[t-1])
                rot_diff = torch.norm(obj_rot[t] - obj_rot[t-1])
                if trans_diff > trans_change_threshold or rot_diff > rot_change_threshold:
                    obj_contact[t] = True
        # --- 物体运动接触计算结束 ---

        # 构建物体数据字典
        obj_data = {
            "obj_name": object_name,
            "obj_scale": obj_scale.cpu(),
            "obj_trans": obj_trans.cpu(),
            "obj_rot": obj_rot.cpu(),
            "obj_com_pos": obj_com_pos.cpu(),
            "lhand_contact": lhand_contact.cpu(), # 基于距离
            "rhand_contact": rhand_contact.cpu(), # 基于距离
            "obj_contact": obj_contact.cpu()      # 基于运动
        }
    
    # 组装输出数据
    data = {
        "seq_name": seq_name,
        "body_parms_list": {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in body_parms.items()},
        "rotation_local_full_gt_list": rotation_local_full_gt_list.cpu(),
        "head_global_trans": head_global_trans.cpu(),
        "position_global_full_gt_world": position_global_full_gt_world.float(),
        "gender": subject_gender,
        "rotation_global": rotation_global_matrot_reshaped.cpu()
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
                    # "imu_global_full_gt": {
                    #     "accelerations": imu_data['accelerations'],
                    #     "orientations": imu_data['orientations']
                    # },
                    "gender": "neutral",
                    "rotation_global": global_rotmat.cpu()
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
    parser.add_argument("--save_dir_train", type=str, default="processed_data_0506/train",
                        help="输出数据保存目录")
    parser.add_argument("--save_dir_test", type=str, default="processed_data_0506/test",
                        help="输出数据保存目录")
    parser.add_argument("--support_dir", type=str, default="body_models",
                        help="SMPL模型目录")
    parser.add_argument("--obj_mesh_dir", type=str, default="dataset/captured_objects",
                        help="物体网格目录")
    parser.add_argument("--process_amass", action="store_true",
                        help="是否处理AMASS数据集")
    parser.add_argument("--amass_dir", type=str, default="/mnt/d/a_WORK/Projects/PhD/datasets/AMASS_SMPL_H",
                        help="AMASS数据集路径")
    parser.add_argument("--amass_dirs", type=str, nargs="+",
                        help="AMASS子数据集列表，例如CMU BMLmovi")
    
    args = parser.parse_args()
    main(args)