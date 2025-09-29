import torch
import os
import numpy as np
from numpy import array
import random
import argparse
import yaml
import re
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.viewer import Viewer
from aitviewer.scene.camera import Camera
from moderngl_window.context.base import KeyModifiers
from transforms3d.axangles import mat2axangle
import trimesh
# from configs.global_config import FRAME_RATE

from human_body_prior.body_model.body_model import BodyModel
from easydict import EasyDict as edict

####DataLoader需要自行实现######
from torch.utils.data import DataLoader
from dataloader.dataloader import IMUDataset # 从 eval.py 引入

# 导入模型相关 - 根据需要选择正确的模型加载方式
from models.TransPose_net import TransPoseNet # 明确使用 TransPose

import imgui


# --- 定义 Z-up 到 Y-up 的旋转矩阵 ---
R_yup = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, -1.0, 0.0]], dtype=torch.float32)

# R_yup = torch.tensor([[1.0, 0.0, 0.0],
#                       [0.0, 1.0, 0.0],
#                       [0.0, 0.0, 1.0]], dtype=torch.float32)


def rotation_6d_to_matrix(rots_6d: torch.Tensor) -> torch.Tensor:
    """将 6D 旋转表示转换为旋转矩阵（不依赖 pytorch3d）。

    参考 Zhou 等人的表示法：用两个 3D 向量经正交化得到旋转矩阵的前两列，第三列为叉乘。

    支持形状 [..., 6]，返回 [..., 3, 3]。
    """
    a1 = rots_6d[..., 0:3]
    a2 = rots_6d[..., 3:6]

    b1 = torch.nn.functional.normalize(a1, dim=-1)
    a2_proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2 - a2_proj, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    rot_mat = torch.stack([b1, b2, b3], dim=-1)
    return rot_mat


def matrix_to_axis_angle(rot_mats: torch.Tensor) -> torch.Tensor:
    """将旋转矩阵转换为轴角 3 向量（使用 transforms3d），返回与 pytorch3d 相同的 3 向量格式。

    输入形状 [..., 3, 3]，输出形状 [..., 3]，其中向量方向是旋转轴，范数为旋转角。
    """
    device = rot_mats.device
    dtype = rot_mats.dtype
    prefix_shape = rot_mats.shape[:-2]

    rot_np = rot_mats.detach().cpu().numpy().reshape(-1, 3, 3)
    aa_list = []
    for i in range(rot_np.shape[0]):
        axis, angle = mat2axangle(rot_np[i])
        aa_list.append(np.asarray(axis, dtype=np.float64) * float(angle))
    aa_np = np.stack(aa_list, axis=0).reshape(*prefix_shape, 3)
    aa = torch.from_numpy(aa_np).to(device=device, dtype=dtype)
    return aa


def load_smpl_model(smpl_model_path, device):
    """加载 SMPL 模型 using human_body_prior"""
    print(f"Loading SMPL model from: {smpl_model_path}")
    if not os.path.exists(smpl_model_path):
        print(f"Error: SMPL model path not found: {smpl_model_path}")
        raise FileNotFoundError(f"SMPL model not found at {smpl_model_path}")
    smpl_model = BodyModel(
        bm_fname=smpl_model_path,
        num_betas=16,
        model_type='smplh' # 明确使用 smplh
    ).to(device)
    return smpl_model

def apply_transformation_to_obj_geometry(obj_mesh_path, obj_rot, obj_trans, scale=None, device='cpu'):
    """
    应用变换到物体网格 (遵循 hand_foot_dataset.py 的逻辑: Rotate -> Scale -> Translate)

    参数:
        obj_mesh_path: 物体网格路径
        obj_rot: 旋转矩阵 [T, 3, 3] (torch tensor on device)
        obj_trans: 平移向量 [T, 3] (torch tensor on device)
        scale: 缩放因子 [T] (torch tensor on device)

    返回:
        transformed_obj_verts: 变换后的顶点 [T, Nv, 3] (torch tensor on device)
        obj_mesh_faces: 物体网格的面 [Nf, 3] (numpy array)
    """
    try:
        mesh = trimesh.load_mesh(obj_mesh_path)
        obj_mesh_verts_np = np.asarray(mesh.vertices) # Nv X 3
        obj_mesh_faces = np.asarray(mesh.faces) # Nf X 3

        # 确保输入在正确的设备上且为 float 类型
        obj_mesh_verts = torch.from_numpy(obj_mesh_verts_np).float().to(device) # Nv X 3
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
        verts_rotated = torch.bmm(seq_rot_mat, ori_obj_verts.transpose(1, 2)) # T X 3 X Nv

        # 2. 缩放 (Scale)
        if seq_scale is not None:
            scale_factor = seq_scale.unsqueeze(-1).unsqueeze(-1) # T X 1 X 1
            verts_scaled = scale_factor * verts_rotated
        else:
            verts_scaled = verts_rotated # No scaling
        # Result shape: T X 3 X Nv

        # 3. 平移 (Translate)
        trans_vector = seq_trans.unsqueeze(-1) # T X 3 X 1
        verts_translated = verts_scaled + trans_vector # T X 3 X Nv

        # 4. Transpose back to T X Nv X 3
        transformed_obj_verts = verts_translated.transpose(1, 2)

    except Exception as e:
        print(f"应用变换到物体几何体失败 for {obj_mesh_path}: {e}")
        import traceback
        traceback.print_exc()
        # Return dummy data on the correct device
        transformed_obj_verts = torch.zeros((obj_trans.shape[0] if obj_trans is not None else 1, 1, 3), device=device)
        obj_mesh_faces = np.zeros((1, 3), dtype=np.int64)

    return transformed_obj_verts, obj_mesh_faces


def merge_two_parts(verts_list, faces_list, device='cpu'):
    """ 合并两个网格部分 """
    verts_num = 0
    merged_verts_list = []
    merged_faces_list = []
    for p_idx in range(len(verts_list)):
        part_verts = verts_list[p_idx].to(device) # T X Nv X 3
        part_faces = torch.from_numpy(faces_list[p_idx]).long().to(device) # Nf X 3

        merged_verts_list.append(part_verts)
        merged_faces_list.append(part_faces + verts_num)
        verts_num += part_verts.shape[1]

    merged_verts = torch.cat(merged_verts_list, dim=1)
    merged_faces = torch.cat(merged_faces_list, dim=0).cpu().numpy()
    return merged_verts, merged_faces

def load_object_geometry(obj_name, obj_rot, obj_trans, obj_scale=None, obj_geo_root='./dataset/captured_objects', device='cpu'):
    """ 加载物体几何体并应用变换 (OMOMO 方式) """
    if obj_name is None:
        print("Warning: Object name is None, cannot load geometry.")
        return torch.zeros((1, 1, 3), device=device), np.zeros((1, 3), dtype=np.int64)

    # Ensure transformations are tensors on the correct device
    obj_rot = torch.as_tensor(obj_rot, dtype=torch.float32, device=device)
    obj_trans = torch.as_tensor(obj_trans, dtype=torch.float32, device=device)
    if obj_scale is not None:
        obj_scale = torch.as_tensor(obj_scale, dtype=torch.float32, device=device)


    obj_mesh_path = os.path.join(obj_geo_root, f"{obj_name}_cleaned_simplified.obj")

    if not os.path.exists(obj_mesh_path):
        print(f"Warning: Cannot find object geometry file: {obj_mesh_path}")
        return torch.zeros((obj_trans.shape[0] if obj_trans is not None else 1, 1, 3), device=device), np.zeros((1, 3), dtype=np.int64)
    obj_mesh_verts, obj_mesh_faces = apply_transformation_to_obj_geometry(obj_mesh_path, obj_rot, obj_trans, scale=obj_scale, device=device)

    return obj_mesh_verts, obj_mesh_faces


def generate_mesh_data(batch, model, smpl_model, device, obj_geo_root, show_objects=True, vis_gt_only=False):
    """
    生成GT和预测的人体和物体mesh数据
    
    返回:
        tuple: (gt_human_verts, pred_human_verts, gt_obj_verts, pred_obj_verts, faces_human, faces_obj)
        其中每个verts为 [T, Nv, 3]，faces为 [Nf, 3]
    """
    with torch.no_grad():
        bs = 0
        # --- 1. 准备数据 ---
        gt_root_pos = batch["root_pos"].to(device)         # [bs, T, 3]
        gt_motion = batch["motion"].to(device)           # [bs, T, 132]
        human_imu = batch["human_imu"].to(device)        # [bs, T, num_imus, 9/12]
        root_global_pos_start = batch["root_pos_start"].to(device)  # [bs, 3]
        root_global_rot_start = batch["root_rot_start"].to(device)  # [bs, 3, 3]
        obj_imu = batch.get("obj_imu", None)             # [bs, T, 1, 9/12] or None
        gt_obj_trans = batch.get("obj_trans", None)      # [bs, T, 3] or None
        gt_obj_rot_6d = batch.get("obj_rot", None)       # [bs, T, 6] or None
        obj_name = batch.get("obj_name", [None])[0]      # 物体名称 (取列表第一个)
        gt_obj_scale = batch.get("obj_scale", None)      # [bs, T] or [bs, T, 1]? Check dataloader

        if obj_imu is not None: obj_imu = obj_imu.to(device)
        if gt_obj_trans is not None: gt_obj_trans = gt_obj_trans.to(device)
        if gt_obj_rot_6d is not None: gt_obj_rot_6d = gt_obj_rot_6d.to(device)
        if gt_obj_scale is not None: gt_obj_scale = gt_obj_scale.to(device)

        # 仅处理批次中的第一个序列 (bs=0)
        T = gt_motion.shape[1]
        gt_root_pos_seq = gt_root_pos[bs]           # [T, 3]
        gt_motion_seq = gt_motion[bs]             # [T, 132]
        root_global_rot_start = root_global_rot_start[bs]  # [3, 3]
        root_global_pos_start = root_global_pos_start[bs]  # [3]

        # --- 2. 获取真值 SMPL ---
        gt_rot_matrices = rotation_6d_to_matrix(gt_motion_seq.reshape(T, 22, 6)) # [T, 22, 3, 3]
        gt_root_orient_mat_norm = gt_rot_matrices[:, 0]                         # [T, 3, 3]
        gt_pose_body_mat = gt_rot_matrices[:, 1:].reshape(T * 21, 3, 3)    # [T*21, 3, 3]
        gt_pose_body_axis = matrix_to_axis_angle(gt_pose_body_mat).reshape(T, -1) # [T, 63]

        # Denormalization
        gt_root_orient_mat_denorm = root_global_rot_start @ gt_root_orient_mat_norm
        gt_root_orient_axis_denorm = matrix_to_axis_angle(gt_root_orient_mat_denorm).reshape(T, 3)
        gt_root_pos_seq_denorm = (root_global_rot_start @ gt_root_pos_seq.unsqueeze(-1)).squeeze(-1) + root_global_pos_start

        gt_smplh_input = {
            'root_orient': gt_root_orient_axis_denorm,
            'pose_body': gt_pose_body_axis,
            'trans': gt_root_pos_seq_denorm
        }
        body_pose_gt = smpl_model(**gt_smplh_input)
        verts_gt_seq = body_pose_gt.v                          # [T, Nv, 3]
        faces_gt_np = smpl_model.f.cpu().numpy() if isinstance(smpl_model.f, torch.Tensor) else smpl_model.f

        # --- 3. 模型预测 ---
        pred_motion_seq = None
        pred_obj_rot_6d_seq = None
        pred_obj_trans_seq = None # 现在模型会预测物体平移
        pred_root_pos_seq = None

        # 只有在非仅真值模式下才执行模型推理
        if not vis_gt_only:
            model_input = {
                    "human_imu": human_imu,
                    "motion": gt_motion,             # 新增
                    "root_pos": gt_root_pos,           # 新增
                }
            
            has_object_data_for_model = obj_imu is not None
            if has_object_data_for_model:
                model_input["obj_imu"] = obj_imu # [bs, T, 1, dim]
                model_input["obj_rot"] = gt_obj_rot_6d # [bs, T, 6]
                model_input["obj_trans"] = gt_obj_trans # [bs, T, 3]
            
            # 添加GT手部位置（如果可用）
            position_global_norm = batch.get("position_global_norm", None)
            if position_global_norm is not None and position_global_norm.shape[1] == T:
                try:
                    wrist_l_idx, wrist_r_idx = 20, 21
                    pos = position_global_norm.to(device)
                    lhand_pos = pos[:, :, wrist_l_idx, :]
                    rhand_pos = pos[:, :, wrist_r_idx, :]
                    gt_hands_pos = torch.stack([lhand_pos, rhand_pos], dim=2)  # [bs, seq, 2, 3]
                    model_input["gt_hands_pos"] = gt_hands_pos
                except Exception:
                    # 如果提取失败，创建零值
                    model_input["gt_hands_pos"] = torch.zeros((1, T, 2, 3), device=device, dtype=gt_root_pos.dtype)
            else:
                # 如果没有position_global_norm，创建零值
                model_input["gt_hands_pos"] = torch.zeros((1, T, 2, 3), device=device, dtype=gt_root_pos.dtype)

            try:
                pred_dict = model(model_input)
                pred_motion = pred_dict.get("motion") # [bs, T, 132]
                pred_obj_rot = pred_dict.get("obj_rot") # [bs, T, 6] (TransPose 输出 6D)
                pred_obj_trans = pred_dict.get("pred_obj_trans") # 默认使用融合后的物体位置
                pred_root_pos = pred_dict.get("root_pos") # [bs, T, 3]
                pred_obj_vel_batch = pred_dict.get("pred_obj_vel", None)  # [bs, T, 3]

                if pred_motion is not None:
                    pred_motion_seq = pred_motion[bs] # [T, 132]
                else:
                    print("Warning: Model did not output 'motion'")

                if pred_root_pos is not None:
                    pred_root_pos_seq = pred_root_pos[bs] # [T, 3]
                else:
                    print("Warning: Model did not output 'root_pos'")

                if pred_obj_rot is not None:
                    pred_obj_rot_6d_seq = pred_obj_rot[bs] # [T, 6]

                # 新增：获取预测的物体平移
                if pred_obj_trans is not None:
                    pred_obj_trans_seq = pred_obj_trans[bs] # [T, 3]
                elif has_object_data_for_model:
                    print("Warning: Model did not output 'obj_trans', even with object IMU input")

                # 物体速度（用于IMU积分）
                pred_obj_vel_seq = None
                if pred_obj_vel_batch is not None:
                    pred_obj_vel_seq = pred_obj_vel_batch[bs]

            except Exception as e:
                print(f"Model inference failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("GT-only mode: Skip model inference")

        # --- 4. 获取预测 SMPL (使用预测 motion + 真值 trans) ---
        verts_pred_seq = None
        if pred_motion_seq is not None and not vis_gt_only:
            pred_rot_matrices = rotation_6d_to_matrix(pred_motion_seq.reshape(T, 22, 6)) # [T, 22, 3, 3]
            pred_root_orient_mat_norm = pred_rot_matrices[:, 0]                         # [T, 3, 3]
            pred_pose_body_mat = pred_rot_matrices[:, 1:].reshape(T * 21, 3, 3)    # [T*21, 3, 3]
            pred_pose_body_axis = matrix_to_axis_angle(pred_pose_body_mat).reshape(T, -1) # [T, 63]

            # Denormalization
            pred_root_orient_mat_denorm = root_global_rot_start @ pred_root_orient_mat_norm
            pred_root_orient_axis_denorm = matrix_to_axis_angle(pred_root_orient_mat_denorm).reshape(T, 3)
            pred_root_pos_seq_denorm = (root_global_rot_start @ pred_root_pos_seq.unsqueeze(-1)).squeeze(-1) + root_global_pos_start

            pred_smplh_input = {
                'root_orient': pred_root_orient_axis_denorm,
                'pose_body': pred_pose_body_axis,
                'trans': pred_root_pos_seq_denorm
            }
            body_pose_pred = smpl_model(**pred_smplh_input)
            verts_pred_seq = body_pose_pred.v # [T, Nv, 3]

        # --- 5. 获取物体几何体 ---
        gt_obj_verts_seq = None
        pred_obj_verts_seq = None
        obj_faces_np = None
        has_object_gt = gt_obj_trans is not None and gt_obj_rot_6d is not None and obj_name is not None

        if show_objects and has_object_gt:
            gt_obj_trans_seq = gt_obj_trans[bs]     # [T, 3]
            gt_obj_rot_6d_seq = gt_obj_rot_6d[bs]   # [T, 6]
            gt_obj_rot_mat_seq = rotation_6d_to_matrix(gt_obj_rot_6d_seq) # [T, 3, 3]
            gt_obj_scale_seq = gt_obj_scale[bs] if gt_obj_scale is not None else None # [T] or [T, 1]?

            # Denormalization
            gt_obj_rot_mat_seq_denorm = root_global_rot_start @ gt_obj_rot_mat_seq
            gt_obj_trans_seq_denorm = (root_global_rot_start @ gt_obj_trans_seq.unsqueeze(-1)).squeeze(-1) + root_global_pos_start

            # 获取真值物体
            gt_obj_verts_seq, obj_faces_np = load_object_geometry(
                obj_name, gt_obj_rot_mat_seq_denorm, gt_obj_trans_seq_denorm, gt_obj_scale_seq, device=device
            )

            # 获取预测物体 (使用真值旋转 + 预测平移) - 只在非仅真值模式下执行
            if not vis_gt_only:
                if pred_obj_trans_seq is not None:
                    # 对预测的物体平移进行反归一化
                    pred_obj_trans_seq_denorm = (root_global_rot_start @ pred_obj_trans_seq.unsqueeze(-1)).squeeze(-1) + root_global_pos_start
                    
                    # 使用真值旋转 + 预测平移
                    pred_obj_verts_seq, _ = load_object_geometry(
                        obj_name, 
                        gt_obj_rot_mat_seq_denorm, # 使用真值旋转
                        pred_obj_trans_seq_denorm, # 使用预测平移
                        gt_obj_scale_seq, 
                        device=device
                    )
                else:
                    # If no predicted translation, fallback to using GT translation
                    print("Warning: No predicted object translation, using GT translation for visualization")
                    pred_obj_verts_seq, _ = load_object_geometry(
                        obj_name, 
                        gt_obj_rot_mat_seq_denorm, # 使用真值旋转
                        gt_obj_trans_seq_denorm,   # 使用真值平移
                        gt_obj_scale_seq, 
                        device=device
                    )

        return verts_gt_seq, verts_pred_seq, gt_obj_verts_seq, pred_obj_verts_seq, faces_gt_np, obj_faces_np


def visualize_mesh_data(viewer, gt_human_verts, pred_human_verts, gt_obj_verts, pred_obj_verts, faces_human, faces_obj, obj_name=None, vis_gt_only=False):
    """
    将生成的mesh数据添加到aitviewer场景中
    
    参数:
        viewer: aitviewer viewer实例
        gt_human_verts: GT人体顶点 [T, Nv, 3] or None
        pred_human_verts: 预测人体顶点 [T, Nv, 3] or None  
        gt_obj_verts: GT物体顶点 [T, Nv, 3] or None
        pred_obj_verts: 预测物体顶点 [T, Nv, 3] or None
        faces_human: 人体面 [Nf, 3] or None
        faces_obj: 物体面 [Nf, 3] or None
        obj_name: 物体名称 (用于命名)
        vis_gt_only: 是否仅显示GT
    """
    # --- 场景清理 ---
    try:
        # Use collect_nodes to get all nodes currently managed by the scene
        # We filter based on name to identify previously added GT/Pred meshes
        nodes_to_remove = [
            node for node in viewer.scene.collect_nodes()
            if hasattr(node, 'name') and node.name is not None and 
               (node.name.startswith("GT-") or 
                node.name.startswith("Pred-") or
                node.name.startswith("FK-"))
        ]

        # Call viewer.scene.remove() for each identified node
        removed_count = 0
        if nodes_to_remove:
            for node_to_remove in nodes_to_remove:
                try:
                    viewer.scene.remove(node_to_remove)
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing node '{node_to_remove.name}' from scene: {e}")

    except AttributeError as e:
        print(f"Error accessing scene nodes or methods: {e}")
    except Exception as e:
        print(f"Error during scene clearing: {e}")

    # --- 添加到 aitviewer 场景 ---
    global R_yup # 使用全局定义的 Y-up 旋转
    device = gt_human_verts.device if gt_human_verts is not None else torch.device('cpu')
    
    # Define a visual offset for predicted elements
    pred_offset = torch.tensor([0.0, 0.0, 0.0], device=device)

    # 添加真值人体 (绿色, 不偏移)
    if gt_human_verts is not None and faces_human is not None:
        verts_gt_yup = torch.matmul(gt_human_verts, R_yup.T.to(device))
        gt_human_mesh = Meshes(
            verts_gt_yup.cpu().numpy(), faces_human,
            name="GT-Human", color=(0.1, 0.8, 0.3, 0.8), gui_affine=False, is_selectable=False
        )
        viewer.scene.add(gt_human_mesh)

    # 添加预测人体 (红色, 偏移) - 只在非仅真值模式下执行
    if pred_human_verts is not None and faces_human is not None and not vis_gt_only:
        verts_pred_shifted = pred_human_verts + pred_offset
        verts_pred_yup = torch.matmul(verts_pred_shifted, R_yup.T.to(device))
        pred_human_mesh = Meshes(
            verts_pred_yup.cpu().numpy(), faces_human,
            name="Pred-Human", color=(0.9, 0.2, 0.2, 0.8), gui_affine=False, is_selectable=False
        )
        viewer.scene.add(pred_human_mesh)

    # 添加真值物体 (绿色, 不偏移)
    if gt_obj_verts is not None and faces_obj is not None:
        gt_obj_verts_yup = torch.matmul(gt_obj_verts, R_yup.T.to(device))
        obj_display_name = f"GT-{obj_name}" if obj_name else "GT-Object"
        gt_obj_mesh = Meshes(
            gt_obj_verts_yup.cpu().numpy(), faces_obj,
            name=obj_display_name, color=(0.1, 0.8, 0.3, 0.8), gui_affine=False, is_selectable=False
        )
        viewer.scene.add(gt_obj_mesh)

    # 添加预测物体 (红色, 偏移) - 只在非仅真值模式下执行
    if pred_obj_verts is not None and faces_obj is not None and not vis_gt_only:
        pred_obj_verts_shifted = pred_obj_verts + pred_offset
        pred_obj_verts_yup = torch.matmul(pred_obj_verts_shifted, R_yup.T.to(device))
        obj_display_name = f"Pred-{obj_name}" if obj_name else "Pred-Object"
        pred_obj_mesh = Meshes(
            pred_obj_verts_yup.cpu().numpy(), faces_obj,
            name=obj_display_name, color=(0.9, 0.2, 0.2, 0.8), gui_affine=False, is_selectable=False
        )
        viewer.scene.add(pred_obj_mesh)


def visualize_batch_data(viewer, batch, model, smpl_model, device, obj_geo_root, show_objects=True, vis_gt_only=False):
    """ 在 aitviewer 场景中可视化单个批次的数据 (真值和预测) """
    # 生成mesh数据
    gt_human_verts, pred_human_verts, gt_obj_verts, pred_obj_verts, faces_human, faces_obj = generate_mesh_data(
        batch, model, smpl_model, device, obj_geo_root, show_objects, vis_gt_only
    )
    
    # 获取物体名称
    obj_name = batch.get("obj_name", [None])[0]
    
    # 可视化mesh数据
    visualize_mesh_data(viewer, gt_human_verts, pred_human_verts, gt_obj_verts, pred_obj_verts, 
                       faces_human, faces_obj, obj_name, vis_gt_only)


# === 自定义 Viewer 类 ===

class InteractiveViewer(Viewer):
    def __init__(self, data_list, model, smpl_model, device, obj_geo_root, show_objects=True, vis_gt_only=False, **kwargs):
        super().__init__(**kwargs)
        self.data_list = data_list # 直接使用加载到内存的列表
        self.current_index = 0
        self.model = model
        self.smpl_model = smpl_model
        self.device = device
        self.show_objects = show_objects
        self.vis_gt_only = vis_gt_only
        self.obj_geo_root = obj_geo_root
        

        # 设置初始相机位置 (可选)
        # self.scene.camera.position = np.array([0.0, 1.0, 3.0])
        # self.scene.camera.target = np.array([0.5, 0.8, 0.0]) # 对准偏移后的中间区域

        # 初始可视化
        self.visualize_current_sequence()

    def visualize_current_sequence(self):
        if not self.data_list:
            print("Error: Data list is empty.")
            return
        if 0 <= self.current_index < len(self.data_list):
            entry = self.data_list[self.current_index]
            batch = entry["batch"] if isinstance(entry, dict) and "batch" in entry else entry
            mode_str = " (GT only)" if self.vis_gt_only else " (GT+Pred)"
            seq_file_name = ""
            if isinstance(entry, dict):
                seq_file_name = entry.get("seq_file_name") or os.path.basename(entry.get("seq_file_path", ""))
            if seq_file_name:
                print(f"Visualizing sequence: {seq_file_name}{mode_str}")
            else:
                print(f"Visualizing sequence index: {self.current_index}{mode_str}")
            try:
                visualize_batch_data(self, batch, self.model, self.smpl_model, self.device, self.obj_geo_root, self.show_objects, self.vis_gt_only)
                title_base = (
                    f"Sequence: {seq_file_name}" if seq_file_name else f"Sequence Index: {self.current_index}/{len(self.data_list)-1}"
                )
                self.title = f"{title_base}{mode_str} (q/e:±1, Ctrl+q/e:±10, Alt+q/e:±50)"
            except Exception as e:
                 print(f"Error visualizing sequence {self.current_index}: {e}")
                 import traceback
                 traceback.print_exc()
                 self.title = f"Error visualizing index: {self.current_index}"
        else:
            print("Index out of bounds.")

    def gui_scene(self):
        """重写GUI场景方法"""
        # 调用父类的GUI场景方法
        super().gui_scene()
    

    # --- Rename to key_event and adjust logic --- 
    # def key_press_event(self, key, scancode: int, mods: KeyModifiers): # Old name and signature
    def key_event(self, key, action, modifiers):
        # --- Call Parent First --- 
        # Important: Call super first to allow base class and ImGui to process event
        super().key_event(key, action, modifiers)

        # --- Check if ImGui wants keyboard input --- 
        # If ImGui is active and wants keyboard input, don't process our keys
        io = imgui.get_io()
        if self.render_gui and (io.want_capture_keyboard or io.want_text_input):
             return # Let ImGui handle it

        # --- Check for Key PRESS action --- 
        is_press = action == self.wnd.keys.ACTION_PRESS

        if is_press:
            # Check for modifier keys
            ctrl_pressed = modifiers.ctrl
            alt_pressed = modifiers.alt
            
            # Compare using self.wnd.keys
            if key == self.wnd.keys.Q:
                if alt_pressed:
                    # Alt + Q: 后退50个index
                    step = 50
                    new_index = max(0, self.current_index - step)
                    if new_index != self.current_index:
                        self.current_index = new_index
                        self.visualize_current_sequence()
                        self.scene.current_frame_id = 0
                        print(f"Jump back 50 sequences to index: {self.current_index}")
                    else:
                        print("Already at the first sequence.")
                elif ctrl_pressed:
                    # Ctrl + Q: 后退10个index
                    step = 10
                    new_index = max(0, self.current_index - step)
                    if new_index != self.current_index:
                        self.current_index = new_index
                        self.visualize_current_sequence()
                        self.scene.current_frame_id = 0
                        print(f"Jump back 10 sequences to index: {self.current_index}")
                    else:
                        print("Already at the first sequence.")
                else:
                    # Q: 后退1个index
                    if self.current_index > 0:
                        self.current_index -= 1
                        self.visualize_current_sequence()
                        self.scene.current_frame_id = 0 # Reset scene frame id
                    else:
                        print("Already at the first sequence.")
            elif key == self.wnd.keys.E:
                if alt_pressed:
                    # Alt + E: 前进50个index
                    step = 50
                    new_index = min(len(self.data_list) - 1, self.current_index + step)
                    if new_index != self.current_index:
                        self.current_index = new_index
                        self.visualize_current_sequence()
                        self.scene.current_frame_id = 0
                        print(f"Jump forward 50 sequences to index: {self.current_index}")
                    else:
                        print("Already at the last sequence.")
                elif ctrl_pressed:
                    # Ctrl + E: 前进10个index
                    step = 10
                    new_index = min(len(self.data_list) - 1, self.current_index + step)
                    if new_index != self.current_index:
                        self.current_index = new_index
                        self.visualize_current_sequence()
                        self.scene.current_frame_id = 0
                        print(f"Jump forward 10 sequences to index: {self.current_index}")
                    else:
                        print("Already at the last sequence.")
                else:
                    # E: 前进1个index
                    if self.current_index < len(self.data_list) - 1:
                        self.current_index += 1
                        self.visualize_current_sequence()
                        self.scene.current_frame_id = 0 # Reset scene frame id
                    else:
                        print("Already at the last sequence.")
            

# === 主函数 ===

def main():
    parser = argparse.ArgumentParser(description='Interactive EgoMotion Visualization Tool')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the trained TransPose model checkpoint. Overrides config if provided.')
    parser.add_argument('--smpl_model_path', type=str, default=None, help='Path to the SMPLH model file. Overrides config if provided.')
    parser.add_argument('--test_data_dir', type=str, default=None, help='Path to the test dataset directory. Overrides config if provided.')
    parser.add_argument('--obj_geo_root', type=str, default='./dataset/captured_objects', help='Path to the object geometry root directory.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader (should be 1 for sequential vis).')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers.')
    parser.add_argument('--no_objects', action='store_true', help='Do not load or visualize objects.')
    parser.add_argument('--vis_gt_only', action='store_true', help='Only visualize ground truth, skip model inference and prediction visualization.')
    parser.add_argument('--limit_sequences', type=int, default=None, help='Limit the number of sequences to load for visualization.')
    args = parser.parse_args()

    if args.batch_size != 1:
        print("Warning: Setting batch_size to 1 for interactive visualization.")
        args.batch_size = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load SMPL Model ---
    smpl_model = load_smpl_model(args.smpl_model_path, device)

    # --- Load Trained Model ---
    model_path = args.model_path
    if not model_path:
        print("Error: No model path provided in config or via --model_path.")
        # return
    model = load_model(model_path, device)
    model.eval()

    # --- Load Test Dataset ---
    test_data_dir = args.test_data_dir
    print(f"Loading test dataset from: {test_data_dir}")

    test_dataset = IMUDataset(
        data_dir=test_data_dir,
        window_size=test_window_size,
        normalize=config.test.get('normalize', True),
        debug=config.get('debug', False),
        full_sequence=args.full_sequence
    )

    # 确保序列按 pt 文件名进行自然排序（如 1.pt, 2.pt, 10.pt）
    def _natural_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
    try:
        test_dataset.sequence_info.sort(
            key=lambda info: _natural_key(os.path.basename(info.get('file_path', '')))
        )
    except Exception as _e:
        print(f"Warning: failed to sort sequence_info: {_e}")

    if len(test_dataset) == 0:
         print("Error: Test dataset is empty.")
         return

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, # Should be 1
        shuffle=False, # IMPORTANT: Keep order for navigation
        num_workers=args.num_workers, # Set workers based on args/config
        pin_memory=True,
        drop_last=False
    )

    print(f"Loading data into memory (limit={args.limit_sequences})...")
    data_list = []
    for i, batch in enumerate(test_loader):
        if args.limit_sequences is not None and i >= args.limit_sequences:
            print(f"Stopped loading after {args.limit_sequences} sequences.")
            break
        # 记录该 batch 对应的 pt 文件路径与文件名
        try:
            seq_info_i = test_dataset.sequence_info[i]
            file_path_i = seq_info_i.get('file_path', '')
            file_name_i = os.path.basename(file_path_i) if file_path_i else ''
        except Exception:
            file_path_i, file_name_i = '', ''
        data_list.append({
            'batch': batch,
            'seq_file_path': file_path_i,
            'seq_file_name': file_name_i,
        })
        if i % 50 == 0 and i > 0:
            print(f"  Loaded {i+1} sequences...")
    print(f"Finished loading {len(data_list)} sequences.")

    if not data_list:
        print("Error: No data loaded into the list.")
        return

    # --- Initialize and Run Viewer ---
    print("Initializing Interactive Viewer...")
    if args.vis_gt_only:
        print("GT-only mode: Will only show GT data, skip model inference")
    viewer_instance = InteractiveViewer(
        data_list=data_list,
        model=model,
        smpl_model=smpl_model,
        device=device,
        obj_geo_root=args.obj_geo_root,
        show_objects=(not args.no_objects),
        vis_gt_only=args.vis_gt_only,
        window_size=(1920, 1080) # Example window size
        # Add other Viewer kwargs if needed (e.g., fps)
    )
    print("Viewer Initialized. Navigation controls:")
    print("  q/e: Previous/Next 1 sequence")
    print("  Ctrl+q/e: Previous/Next 10 sequences")
    print("  Alt+q/e: Previous/Next 50 sequences")
    print("Other standard aitviewer controls should also work (e.g., mouse drag to rotate, scroll to zoom).")
    viewer_instance.run()

if __name__ == "__main__":
    main() 