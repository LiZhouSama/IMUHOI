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
import pytorch3d.transforms as transforms
import trimesh
from configs.global_config import FRAME_RATE

from human_body_prior.body_model.body_model import BodyModel
from easydict import EasyDict as edict

from torch.utils.data import DataLoader
from dataloader.dataloader import IMUDataset # 从 eval.py 引入

# 导入模型相关 - 根据需要选择正确的模型加载方式
# from models.DiT_model import MotionDiffusion # 如果要用 DiT
from models.TransPose_net import TransPoseNet # 明确使用 TransPose

import imgui
from aitviewer.renderables.spheres import Spheres


# --- 定义 Z-up 到 Y-up 的旋转矩阵 ---
R_yup = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, -1.0, 0.0]], dtype=torch.float32)

# R_yup = torch.tensor([[1.0, 0.0, 0.0],
#                       [0.0, 1.0, 0.0],
#                       [0.0, 0.0, 1.0]], dtype=torch.float32)

# === 辅助函数 (来自 eval.py 和 vis.py) ===

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = edict(config)
    return config

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

def load_object_geometry(obj_name, obj_rot, obj_trans, obj_scale=None, obj_bottom_trans=None, obj_bottom_rot=None, obj_geo_root='./dataset/captured_objects', device='cpu'):
    """ 加载物体几何体并应用变换 (OMOMO 方式) """
    if obj_name is None:
        print("Warning: Object name is None, cannot load geometry.")
        return torch.zeros((1, 1, 3), device=device), np.zeros((1, 3), dtype=np.int64)

    # Ensure transformations are tensors on the correct device
    obj_rot = torch.as_tensor(obj_rot, dtype=torch.float32, device=device)
    obj_trans = torch.as_tensor(obj_trans, dtype=torch.float32, device=device)
    if obj_scale is not None:
        obj_scale = torch.as_tensor(obj_scale, dtype=torch.float32, device=device)
    if obj_bottom_rot is not None:
        obj_bottom_rot = torch.as_tensor(obj_bottom_rot, dtype=torch.float32, device=device)
    if obj_bottom_trans is not None:
        obj_bottom_trans = torch.as_tensor(obj_bottom_trans, dtype=torch.float32, device=device)


    obj_mesh_path = os.path.join(obj_geo_root, f"{obj_name}_cleaned_simplified.obj")
    two_parts = obj_name in ["vacuum", "mop"] and obj_bottom_trans is not None and obj_bottom_rot is not None

    if two_parts:
        top_obj_mesh_path = os.path.join(obj_geo_root, f"{obj_name}_cleaned_simplified_top.obj")
        bottom_obj_mesh_path = os.path.join(obj_geo_root, f"{obj_name}_cleaned_simplified_bottom.obj")

        if not os.path.exists(top_obj_mesh_path) or not os.path.exists(bottom_obj_mesh_path):
             print(f"Warning: Cannot find two-part geometry files for object {obj_name}. Will try to load the complete file.")
             two_parts = False
             obj_mesh_path = os.path.join(obj_geo_root, f"{obj_name}_cleaned_simplified.obj") # Fallback
        else:
            top_obj_mesh_verts, top_obj_mesh_faces = apply_transformation_to_obj_geometry(top_obj_mesh_path, obj_rot, obj_trans, scale=obj_scale, device=device)
            # Assume bottom uses the same scale, pass bottom transforms
            bottom_obj_mesh_verts, bottom_obj_mesh_faces = apply_transformation_to_obj_geometry(bottom_obj_mesh_path, obj_bottom_rot, obj_bottom_trans, scale=obj_scale, device=device)
            obj_mesh_verts, obj_mesh_faces = merge_two_parts([top_obj_mesh_verts, bottom_obj_mesh_verts], [top_obj_mesh_faces, bottom_obj_mesh_faces], device=device)

    if not two_parts:
        if not os.path.exists(obj_mesh_path):
             print(f"Warning: Cannot find object geometry file: {obj_mesh_path}")
             return torch.zeros((obj_trans.shape[0] if obj_trans is not None else 1, 1, 3), device=device), np.zeros((1, 3), dtype=np.int64)
        obj_mesh_verts, obj_mesh_faces = apply_transformation_to_obj_geometry(obj_mesh_path, obj_rot, obj_trans, scale=obj_scale, device=device)

    return obj_mesh_verts, obj_mesh_faces


def compute_virtual_bone_info(wrist_pos, obj_trans, obj_rot_mat):
    """
    计算虚拟骨长和方向（轴角表示）
    
    Args:
        wrist_pos: [T, 3] - 手腕位置
        obj_trans: [T, 3] - 物体位置  
        obj_rot_mat: [T, 3, 3] - 物体旋转矩阵
    
    Returns:
        bone_length: [T] - 虚拟骨长
        obj_direction_axis_angle: [T, 3] - 物体坐标系下方向的轴角表示
    """
    # 1. 计算世界坐标系下的向量
    v_HO_world = obj_trans - wrist_pos  # [T, 3]
    
    # 2. 计算骨长
    bone_length = torch.norm(v_HO_world, dim=1)  # [T]
    
    # 3. 归一化得到世界坐标系下的单位向量
    v_HO_world_unit = v_HO_world / (bone_length.unsqueeze(-1) + 1e-8)  # [T, 3]
    
    # 4. 转换到物体坐标系：^Ov_{HO} = ^WR_O^T * ^Wv_{HO}
    obj_rot_inv = obj_rot_mat.transpose(-1, -2)  # [T, 3, 3]
    obj_direction = torch.bmm(obj_rot_inv, v_HO_world_unit.unsqueeze(-1)).squeeze(-1)  # [T, 3]
    
    # 5. 将方向向量转换为轴角表示
    # 这里我们直接将物体坐标系下的方向向量作为轴角表示
    # 因为这个向量本身就包含了方向和幅度信息
    obj_direction_axis_angle = obj_direction  # [T, 3]
    
    return bone_length, obj_direction_axis_angle

def visualize_batch_data(viewer, batch, model, smpl_model, device, obj_geo_root, show_objects=True, vis_gt_only=False, show_foot_contact=False, use_fk=False):
    """ 在 aitviewer 场景中可视化单个批次的数据 (真值和预测) """
    # --- Revised Clearing Logic (Attempt 5 - Using Scene.remove) ---
    try:
        # Use collect_nodes to get all nodes currently managed by the scene
        # We filter based on name to identify previously added GT/Pred meshes and all contact indicators
        nodes_to_remove = [
            node for node in viewer.scene.collect_nodes()
            if hasattr(node, 'name') and node.name is not None and 
               (node.name.startswith("GT-") or 
                node.name.startswith("Pred-") or
                node.name.startswith("FK-") or
                node.name == "GT-LHandContact" or  # 明确手部接触名称
                node.name == "GT-RHandContact" or  # 明确手部接触名称
                node.name == "ObjContactIndicator" or # 物体运动指示器名称
                node.name == "Pred-LHandContact" or # 预测左手接触
                node.name == "Pred-RHandContact" or # 预测右手接触
                node.name == "Pred-ObjContactIndicator" or # 预测物体接触
                node.name == "GT-LFootContact" or  # 真值左脚接触
                node.name == "GT-RFootContact" or  # 真值右脚接触
                node.name == "Pred-LFootContact" or # 预测左脚接触
                node.name == "Pred-RFootContact") # 预测右脚接触
        ]

        # Call viewer.scene.remove() for each identified node
        removed_count = 0
        if nodes_to_remove:
            # print(f"Attempting to remove {len(nodes_to_remove)} nodes from the scene.")
            for node_to_remove in nodes_to_remove:
                try:
                    # print(f"  Removing: {node_to_remove.name}")
                    viewer.scene.remove(node_to_remove)
                    removed_count += 1
                except Exception as e:
                    # This might happen if the node was already removed or detached somehow
                    print(f"Error removing node '{node_to_remove.name}' from scene: {e}")
            # print(f"Successfully removed {removed_count} nodes.")
        # else:
            # print("No old GT/Pred nodes found to remove.")

    except AttributeError as e:
        print(f"Error accessing scene nodes or methods (maybe collect_nodes or remove doesn't exist?): {e}")
    except Exception as e:
        print(f"Error during scene clearing: {e}")
    # --- End Revised Clearing Logic ---

    with torch.no_grad():
        bs = 0
        fk_bone_info_seq = None
        # --- 1. 准备数据 ---
        gt_root_pos = batch["root_pos"].to(device)         # [bs, T, 3]
        gt_motion = batch["motion"].to(device)           # [bs, T, 132]
        human_imu = batch["human_imu"].to(device)        # [bs, T, num_imus, 9/12]
        # head_global_rot_start = batch["head_global_trans_start"][..., :3, :3].to(device)  # [bs, 1, 3, 3]
        # head_global_pos_start = batch["head_global_trans_start"][..., :3, 3].to(device)  # [bs, 1, 3]
        root_global_pos_start = batch["root_pos_start"].to(device)  # [bs, 3]
        root_global_rot_start = batch["root_rot_start"].to(device)  # [bs, 3, 3]
        obj_imu = batch.get("obj_imu", None)             # [bs, T, 1, 9/12] or None
        gt_obj_trans = batch.get("obj_trans", None)      # [bs, T, 3] or None
        gt_obj_rot_6d = batch.get("obj_rot", None)       # [bs, T, 6] or None
        obj_name = batch.get("obj_name", [None])[0]      # 物体名称 (取列表第一个)
        gt_obj_scale = batch.get("obj_scale", None)      # [bs, T] or [bs, T, 1]? Check dataloader
        gt_obj_bottom_trans = batch.get("obj_bottom_trans", None) # [bs, T, 3] or None
        gt_obj_bottom_rot = batch.get("obj_bottom_rot", None)     # [bs, T, 3, 3] or None

        # 获取用于可视化的接触标志 (由dataloader预处理)
        lhand_contact_viz_seq = batch.get("lhand_contact") [bs].to(device)
        rhand_contact_viz_seq = batch.get("rhand_contact")[bs].to(device)
        obj_contact_viz_seq = batch.get("obj_contact")[bs].to(device)

        if obj_imu is not None: obj_imu = obj_imu.to(device)
        if gt_obj_trans is not None: gt_obj_trans = gt_obj_trans.to(device)
        if gt_obj_rot_6d is not None: gt_obj_rot_6d = gt_obj_rot_6d.to(device)
        if gt_obj_scale is not None: gt_obj_scale = gt_obj_scale.to(device)
        if gt_obj_bottom_trans is not None: gt_obj_bottom_trans = gt_obj_bottom_trans.to(device)
        if gt_obj_bottom_rot is not None: gt_obj_bottom_rot = gt_obj_bottom_rot.to(device)

        # 仅处理批次中的第一个序列 (bs=0)
        T = gt_motion.shape[1]
        gt_root_pos_seq = gt_root_pos[bs]           # [T, 3]
        gt_motion_seq = gt_motion[bs]             # [T, 132]
        # head_global_rot_start = head_global_rot_start[bs]  # [1, 3, 3]
        # head_global_pos_start = head_global_pos_start[bs]  # [1, 3]
        root_global_rot_start = root_global_rot_start[bs]  # [3, 3]
        root_global_pos_start = root_global_pos_start[bs]  # [3]

        # --- 2. 获取真值 SMPL ---
        gt_rot_matrices = transforms.rotation_6d_to_matrix(gt_motion_seq.reshape(T, 22, 6)) # [T, 22, 3, 3]
        gt_root_orient_mat_norm = gt_rot_matrices[:, 0]                         # [T, 3, 3]
        gt_pose_body_mat = gt_rot_matrices[:, 1:].reshape(T * 21, 3, 3)    # [T*21, 3, 3]
        # gt_root_orient_axis = transforms.matrix_to_axis_angle(gt_root_orient_mat_norm) # [T, 3]
        gt_pose_body_axis = transforms.matrix_to_axis_angle(gt_pose_body_mat).reshape(T, -1) # [T, 63]

        # Denormalization
        # gt_root_orient_mat = head_global_rot_start @ gt_root_orient_mat_norm
        # gt_root_orient_axis = transforms.matrix_to_axis_angle(gt_root_orient_mat).reshape(T, 3)
        # gt_root_pos_seq = (head_global_rot_start @ gt_root_pos_seq.unsqueeze(-1)).squeeze(-1) + head_global_pos_start

        gt_root_orient_mat_denorm = root_global_rot_start @ gt_root_orient_mat_norm
        gt_root_orient_axis_denorm = transforms.matrix_to_axis_angle(gt_root_orient_mat_denorm).reshape(T, 3)
        gt_root_pos_seq_denorm = (root_global_rot_start @ gt_root_pos_seq.unsqueeze(-1)).squeeze(-1) + root_global_pos_start
        # gt_root_orient_axis = transforms.matrix_to_axis_angle(gt_root_orient_mat_norm)

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
        pred_lhand_contact_labels_seq = None
        pred_rhand_contact_labels_seq = None
        pred_obj_contact_labels_seq = None

        # --- Define a visual offset for predicted elements ---
        pred_offset = torch.tensor([0.0, 0.0, 0.0], device=device)

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

                # --- Get predicted contact probabilities ---
                pred_hand_contact_prob_batch = pred_dict.get("pred_hand_contact_prob") # [bs, T, 3]
                if pred_hand_contact_prob_batch is not None:
                    pred_hand_contact_prob_seq = pred_hand_contact_prob_batch[bs].to(device) # [T, 3]
                    # Convert probabilities to 0/1 labels
                    pred_contact_labels = (pred_hand_contact_prob_seq > 0.5).bool()
                    pred_lhand_contact_labels_seq = pred_contact_labels[:, 0]
                    pred_rhand_contact_labels_seq = pred_contact_labels[:, 1]
                    pred_obj_contact_labels_seq = pred_contact_labels[:, 2]
                else:
                    print("Warning: Model did not output 'pred_hand_contact_prob'.")
                
                # --- Get predicted foot contact probabilities ---
                pred_foot_contact_prob_batch = pred_dict.get("contact_probability") # [bs, T, 2]
                pred_lfoot_contact_labels_seq = None
                pred_rfoot_contact_labels_seq = None
                if pred_foot_contact_prob_batch is not None:
                    pred_foot_contact_prob_seq = pred_foot_contact_prob_batch[bs].to(device) # [T, 2]
                    # Convert probabilities to 0/1 labels
                    pred_foot_contact_labels = (pred_foot_contact_prob_seq > 0.5).bool()
                    pred_lfoot_contact_labels_seq = pred_foot_contact_labels[:, 0]
                    pred_rfoot_contact_labels_seq = pred_foot_contact_labels[:, 1]
                else:
                    print("Warning: Model did not output 'contact_probability' for foot contact.")
                # --- End predicted contact probabilities ---

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
            pred_rot_matrices = transforms.rotation_6d_to_matrix(pred_motion_seq.reshape(T, 22, 6)) # [T, 22, 3, 3]
            pred_root_orient_mat_norm = pred_rot_matrices[:, 0]                         # [T, 3, 3]
            pred_pose_body_mat = pred_rot_matrices[:, 1:].reshape(T * 21, 3, 3)    # [T*21, 3, 3]
            pred_pose_body_axis = transforms.matrix_to_axis_angle(pred_pose_body_mat).reshape(T, -1) # [T, 63]

            # Denormalization
            # pred_root_orient_mat = head_global_rot_start @ pred_root_orient_mat_norm
            # pred_root_orient_axis = transforms.matrix_to_axis_angle(pred_root_orient_mat).reshape(T, 3)
            # pred_root_pos_seq = (head_global_rot_start @ pred_root_pos_seq.unsqueeze(-1)).squeeze(-1) + head_global_pos_start

            pred_root_orient_mat_denorm = root_global_rot_start @ pred_root_orient_mat_norm
            pred_root_orient_axis_denorm = transforms.matrix_to_axis_angle(pred_root_orient_mat_denorm).reshape(T, 3)
            pred_root_pos_seq_denorm = (root_global_rot_start @ pred_root_pos_seq.unsqueeze(-1)).squeeze(-1) + root_global_pos_start
            # pred_root_orient_axis = transforms.matrix_to_axis_angle(pred_root_orient_mat_norm)

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
        pred_obj_verts_seq_fk = None
        obj_faces_np = None
        has_object_gt = gt_obj_trans is not None and gt_obj_rot_6d is not None and obj_name is not None

        if show_objects and has_object_gt:
            gt_obj_trans_seq = gt_obj_trans[bs]     # [T, 3]
            gt_obj_rot_6d_seq = gt_obj_rot_6d[bs]   # [T, 6]
            gt_obj_rot_mat_seq = transforms.rotation_6d_to_matrix(gt_obj_rot_6d_seq) # [T, 3, 3]
            gt_obj_scale_seq = gt_obj_scale[bs] if gt_obj_scale is not None else None # [T] or [T, 1]?
            # Handle bottom parts if they exist
            gt_obj_bottom_trans_seq = gt_obj_bottom_trans[bs] if gt_obj_bottom_trans is not None else None
            gt_obj_bottom_rot_seq = gt_obj_bottom_rot[bs] if gt_obj_bottom_rot is not None else None

            # Denormalization
            # gt_obj_rot_mat_seq = head_global_rot_start @ gt_obj_rot_mat_seq
            # gt_obj_trans_seq = (head_global_rot_start @ gt_obj_trans_seq.unsqueeze(-1)).squeeze(-1) + head_global_pos_start
            
            gt_obj_rot_mat_seq_denorm = root_global_rot_start @ gt_obj_rot_mat_seq
            gt_obj_trans_seq_denorm = (root_global_rot_start @ gt_obj_trans_seq.unsqueeze(-1)).squeeze(-1) + root_global_pos_start
            # gt_obj_rot_mat_seq = gt_obj_rot_mat_seq

            # 获取真值物体
            gt_obj_verts_seq, obj_faces_np = load_object_geometry(
                obj_name, gt_obj_rot_mat_seq_denorm, gt_obj_trans_seq_denorm, gt_obj_scale_seq, device=device
            )

            # 在渲染预测物体之前，若开启 use_fk，则单独计算FK路径（不覆盖fusion），用于黄色mesh可视化
            pred_obj_trans_fk_seq = None
            if not vis_gt_only and ('pred_hand_contact_prob' in pred_dict):
                try:
                    # 手部位置 [bs, T, 2, 3]
                    pred_hand_positions = pred_dict.get('pred_hand_pos', None)
                    if pred_hand_positions is None:
                        hands_pos_feat = pred_dict.get('hands_pos_feat', None)
                        if hands_pos_feat is not None:
                            pred_hand_positions = hands_pos_feat.reshape(1, T, 2, 3)

                    # pred_hand_positions_watch = pred_hand_positions.detach().cpu().numpy()

                    # 物体旋转矩阵 [bs, T, 3, 3]
                    obj_rot_mat_seq_b = gt_obj_rot_mat_seq.unsqueeze(0)
                    # 仅取当前样本的接触概率，保持 batch 维一致
                    pred_hand_contact_prob_batched = pred_dict['pred_hand_contact_prob'][bs:bs+1]
                    pred_obj_trans_fk_seq_b, _fk_info_seq = model.object_trans_module.predict_object_position_from_contact(
                        pred_hand_contact_prob=pred_hand_contact_prob_batched,
                        pred_hand_positions=pred_hand_positions,
                        obj_rot_matrix=obj_rot_mat_seq_b,
                        gt_obj_trans=gt_obj_trans
                    )
                    pred_obj_trans_fk_seq = pred_obj_trans_fk_seq_b[0]
                    fk_bone_info_seq = _fk_info_seq
                except Exception as _e:
                    print(f"FK computation failed: {_e}")

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

                # 计算并渲染基于FK的物体mesh（黄色）；若无法得到FK平移，则回退到GT平移以保证可见
                try:
                    if pred_obj_trans_fk_seq is not None:
                        pred_obj_trans_fk_seq_denorm = (root_global_rot_start @ pred_obj_trans_fk_seq.unsqueeze(-1)).squeeze(-1) + root_global_pos_start
                    else:
                        pred_obj_trans_fk_seq_denorm = gt_obj_trans_seq_denorm
                    pred_obj_verts_seq_fk, _ = load_object_geometry(
                        obj_name,
                        gt_obj_rot_mat_seq_denorm,
                        pred_obj_trans_fk_seq_denorm,
                        gt_obj_scale_seq,
                        device=device
                    )
                except Exception as _e:
                    print(f"FK object mesh generation failed: {_e}")

                # 计算并渲染基于IMU积分的物体mesh（蓝色）
                pred_obj_verts_seq_imu = None
                if pred_obj_vel_seq is not None:
                    try:
                        dt = 1.0 / float(FRAME_RATE)
                        disp_imu = torch.cumsum(pred_obj_vel_seq * dt, dim=0)  # [T, 3]
                        init_pos = gt_obj_trans[bs, 0, :]  # [3]
                        pred_obj_trans_imu_seq = disp_imu + init_pos.unsqueeze(0)  # [T, 3]
                        # 去规范化到世界系（与其他路径一致）
                        pred_obj_trans_imu_denorm = (root_global_rot_start @ pred_obj_trans_imu_seq.unsqueeze(-1)).squeeze(-1) + root_global_pos_start
                        # 构建蓝色mesh（同真值旋转）
                        pred_obj_verts_seq_imu, _ = load_object_geometry(
                            obj_name,
                            gt_obj_rot_mat_seq_denorm,
                            pred_obj_trans_imu_denorm,
                            gt_obj_scale_seq,
                            device=device
                        )
                    except Exception as _e:
                        print(f"IMU-only object mesh generation failed: {_e}")

            # --- 计算并打印物体平移误差（整段，去规范化，单位mm） ---
            try:
                def _mean_err_mm(pred_denorm: torch.Tensor, gt_denorm: torch.Tensor):
                    if pred_denorm is None:
                        return None
                    if pred_denorm.shape != gt_denorm.shape:
                        return None
                    return (pred_denorm - gt_denorm).norm(dim=-1).mean().item() * 1000.0

                fusion_err_mm = None
                fk_err_mm = None
                imu_err_mm = None

                if 'pred_obj_trans_seq_denorm' in locals():
                    fusion_err_mm = _mean_err_mm(pred_obj_trans_seq_denorm, gt_obj_trans_seq_denorm)
                if 'pred_obj_trans_fk_seq_denorm' in locals() and pred_obj_trans_fk_seq is not None:
                    fk_err_mm = _mean_err_mm(pred_obj_trans_fk_seq_denorm, gt_obj_trans_seq_denorm)
                if 'pred_obj_trans_imu_denorm' in locals():
                    imu_err_mm = _mean_err_mm(pred_obj_trans_imu_denorm, gt_obj_trans_seq_denorm)

                def _fmt(v):
                    return f"{v:.2f}" if v is not None else "N/A"

                print(f"Object translation errors (mm) - {obj_name}: Fusion={_fmt(fusion_err_mm)}, FK={_fmt(fk_err_mm)}, IMU={_fmt(imu_err_mm)}")
            except Exception as _e:
                print(f"Warning: Failed to compute object translation errors: {_e}")

        # --- 6. 添加到 aitviewer 场景 ---
        global R_yup # 使用全局定义的 Y-up 旋转

        # 添加真值人体 (绿色, 不偏移)
        if verts_gt_seq is not None:
            verts_gt_yup = torch.matmul(verts_gt_seq, R_yup.T.to(device))
            gt_human_mesh = Meshes(
                verts_gt_yup.cpu().numpy(), faces_gt_np,
                name="GT-Human", color=(0.1, 0.8, 0.3, 0.8), gui_affine=False, is_selectable=False
            )
            viewer.scene.add(gt_human_mesh)

        # 添加预测人体 (红色, 偏移 x=1.0) - 只在非仅真值模式下执行
        if verts_pred_seq is not None and not vis_gt_only:
            verts_pred_shifted = verts_pred_seq + pred_offset # 使用定义的偏移
            verts_pred_yup = torch.matmul(verts_pred_shifted, R_yup.T.to(device))
            pred_human_mesh = Meshes(
                verts_pred_yup.cpu().numpy(), faces_gt_np,
                name="Pred-Human", color=(0.9, 0.2, 0.2, 0.8), gui_affine=False, is_selectable=False
            )
            viewer.scene.add(pred_human_mesh)

        # 添加真值物体 (绿色, 不偏移)
        if gt_obj_verts_seq is not None and obj_faces_np is not None:
            gt_obj_verts_yup = torch.matmul(gt_obj_verts_seq, R_yup.T.to(device))
            gt_obj_mesh = Meshes(
                gt_obj_verts_yup.cpu().numpy(), obj_faces_np,
                name=f"GT-{obj_name}", color=(0.1, 0.8, 0.3, 0.8), gui_affine=False, is_selectable=False
            )
            viewer.scene.add(gt_obj_mesh)

        # 添加预测物体 (红色, 偏移 x=1.0) - 只在非仅真值模式下执行
        if pred_obj_verts_seq is not None and obj_faces_np is not None and not vis_gt_only:
            pred_obj_verts_shifted = pred_obj_verts_seq + pred_offset # 使用定义的偏移
            pred_obj_verts_yup = torch.matmul(pred_obj_verts_shifted, R_yup.T.to(device))
            pred_obj_mesh = Meshes(
                pred_obj_verts_yup.cpu().numpy(), obj_faces_np,
                name=f"Pred-{obj_name}", color=(0.9, 0.2, 0.2, 0.8), gui_affine=False, is_selectable=False
            )
            viewer.scene.add(pred_obj_mesh)

        # 添加IMU积分物体 (蓝色, 偏移 x=1.0) - 只在非仅真值模式下执行
        if 'pred_obj_verts_seq_imu' in locals() and pred_obj_verts_seq_imu is not None and obj_faces_np is not None and not vis_gt_only:
            pred_obj_verts_imu_shifted = pred_obj_verts_seq_imu + pred_offset
            pred_obj_verts_imu_yup = torch.matmul(pred_obj_verts_imu_shifted, R_yup.T.to(device))
            pred_obj_mesh_imu = Meshes(
                pred_obj_verts_imu_yup.cpu().numpy(), obj_faces_np,
                name=f"Pred-IMU-{obj_name}", color=(0.2, 0.2, 0.9, 0.8), gui_affine=False, is_selectable=False
            )
            viewer.scene.add(pred_obj_mesh_imu)

        # 添加FK物体 (黄色, 偏移 x=1.0) - 只在非仅真值模式下执行
        if 'pred_obj_verts_seq_fk' in locals() and pred_obj_verts_seq_fk is not None and obj_faces_np is not None and not vis_gt_only:
            pred_obj_verts_fk_shifted = pred_obj_verts_seq_fk + pred_offset
            pred_obj_verts_fk_yup = torch.matmul(pred_obj_verts_fk_shifted, R_yup.T.to(device))
            fk_obj_mesh = Meshes(
                pred_obj_verts_fk_yup.cpu().numpy(), obj_faces_np,
                name=f"FK-{obj_name}", color=(1.0, 1.0, 0.0, 0.8), gui_affine=False, is_selectable=False
            )
            viewer.scene.add(fk_obj_mesh)

        lhand_contact_seq = batch["lhand_contact"][bs] # [T]
        rhand_contact_seq = batch["rhand_contact"][bs] # [T]
        obj_contact_seq = batch["obj_contact"][bs] # [T]
        
        # --- 获取足部接触真值数据 ---
        lfoot_contact_seq = batch.get("lfoot_contact", None)
        rfoot_contact_seq = batch.get("rfoot_contact", None)
        if lfoot_contact_seq is not None:
            lfoot_contact_seq = lfoot_contact_seq[bs] # [T]
        if rfoot_contact_seq is not None:
            rfoot_contact_seq = rfoot_contact_seq[bs] # [T]
        # --- 获取 GT 关节点位置 ---
        Jtr_gt_seq = body_pose_gt.Jtr # [T, J, 3]

        # --- 获取 Pred 关节点位置 (如果预测了 pose) ---
        Jtr_pred_seq = None
        if verts_pred_seq is not None and not vis_gt_only: # 仅当有预测姿态且非仅真值模式时才计算
            Jtr_pred_seq = body_pose_pred.Jtr # [T, J, 3]


        # --- 定义手腕关节点索引 (请根据你的模型确认) ---
        lhand_idx = 20
        rhand_idx = 21
        
        # --- 定义足踝关节点索引 ---
        lfoot_idx = 7  # 左脚踝
        rfoot_idx = 8  # 右脚踝

        # --- 应用新的接触可视化逻辑 ---
        contact_radius = 0.03 # 可调整

        # --- 取消手部接触小球（GT） ---

        # --- 取消 HOI 物体接触指示小球可视化 ---
        # disabled as per request
        # --- 结束物体接触可视化 ---
        
        # --- 可视化预测的接触标签 - 只在非仅真值模式下执行 ---
        # --- 取消手部接触小球（Pred） ---
        
        # --- 足部接触可视化 (只在启用时执行) ---
        if show_foot_contact:
            foot_contact_radius = 0.04  # 稍大一些的半径来区分手部接触
            
            # --- 可视化真值左脚接触 (紫色立方体) ---
            if lfoot_contact_seq is not None:
                gt_lfoot_contact_points_list = []
                for t in range(T):
                    if lfoot_contact_seq[t] > 0.5:  # 处理可能的浮点数标签
                        gt_lfoot_contact_points_list.append(Jtr_gt_seq[t, lfoot_idx])
                
                if gt_lfoot_contact_points_list:
                    gt_lfoot_contact_points = torch.stack(gt_lfoot_contact_points_list, dim=0)
                    if gt_lfoot_contact_points.numel() > 0:
                        gt_lfoot_points_yup_tensor = torch.matmul(gt_lfoot_contact_points, R_yup.T.to(device))
                        gt_lfoot_points_yup_np = gt_lfoot_points_yup_tensor.cpu().numpy()
                        gt_lfoot_spheres = Spheres(
                            positions=gt_lfoot_points_yup_np,
                            radius=foot_contact_radius,
                            name="GT-LFootContact",
                            color=(0.5, 0.0, 0.5, 0.8),  # 紫色
                            gui_affine=False,
                            is_selectable=False
                        )
                        viewer.scene.add(gt_lfoot_spheres)
            
            # --- 可视化真值右脚接触 (橙色立方体) ---
            if rfoot_contact_seq is not None:
                gt_rfoot_contact_points_list = []
                for t in range(T):
                    if rfoot_contact_seq[t] > 0.5:  # 处理可能的浮点数标签
                        gt_rfoot_contact_points_list.append(Jtr_gt_seq[t, rfoot_idx])
                
                if gt_rfoot_contact_points_list:
                    gt_rfoot_contact_points = torch.stack(gt_rfoot_contact_points_list, dim=0)
                    if gt_rfoot_contact_points.numel() > 0:
                        gt_rfoot_points_yup_tensor = torch.matmul(gt_rfoot_contact_points, R_yup.T.to(device))
                        gt_rfoot_points_yup_np = gt_rfoot_points_yup_tensor.cpu().numpy()
                        gt_rfoot_spheres = Spheres(
                            positions=gt_rfoot_points_yup_np,
                            radius=foot_contact_radius,
                            name="GT-RFootContact",
                            color=(1.0, 0.5, 0.0, 0.8),  # 橙色
                            gui_affine=False,
                            is_selectable=False
                        )
                        viewer.scene.add(gt_rfoot_spheres)
            
            # --- 可视化预测的足部接触 (只在非仅真值模式下) ---
            if not vis_gt_only and Jtr_pred_seq is not None:
                pred_foot_contact_radius = 0.035  # 稍小一些来区分真值和预测
                
                # 预测左脚接触
                if pred_lfoot_contact_labels_seq is not None:
                    pred_lfoot_contact_points_list = []
                    for t in range(T):
                        if pred_lfoot_contact_labels_seq[t]:
                            point_on_pred_human = Jtr_pred_seq[t, lfoot_idx]
                            pred_lfoot_contact_points_list.append(point_on_pred_human + pred_offset)
                    
                    if pred_lfoot_contact_points_list:
                        pred_lfoot_contact_points = torch.stack(pred_lfoot_contact_points_list, dim=0)
                        if pred_lfoot_contact_points.numel() > 0:
                            pred_lfoot_points_yup_tensor = torch.matmul(pred_lfoot_contact_points, R_yup.T.to(device))
                            pred_lfoot_points_yup_np = pred_lfoot_points_yup_tensor.cpu().numpy()
                            pred_lfoot_spheres = Spheres(
                                positions=pred_lfoot_points_yup_np,
                                radius=pred_foot_contact_radius,
                                name="Pred-LFootContact",
                                color=(0.8, 0.3, 0.8, 0.8),  # 浅紫色
                                gui_affine=False,
                                is_selectable=False
                            )
                            viewer.scene.add(pred_lfoot_spheres)
                
                # 预测右脚接触
                if pred_rfoot_contact_labels_seq is not None:
                    pred_rfoot_contact_points_list = []
                    for t in range(T):
                        if pred_rfoot_contact_labels_seq[t]:
                            point_on_pred_human = Jtr_pred_seq[t, rfoot_idx]
                            pred_rfoot_contact_points_list.append(point_on_pred_human + pred_offset)
                    
                    if pred_rfoot_contact_points_list:
                        pred_rfoot_contact_points = torch.stack(pred_rfoot_contact_points_list, dim=0)
                        if pred_rfoot_contact_points.numel() > 0:
                            pred_rfoot_points_yup_tensor = torch.matmul(pred_rfoot_contact_points, R_yup.T.to(device))
                            pred_rfoot_points_yup_np = pred_rfoot_points_yup_tensor.cpu().numpy()
                            pred_rfoot_spheres = Spheres(
                                positions=pred_rfoot_points_yup_np,
                                radius=pred_foot_contact_radius,
                                name="Pred-RFootContact",
                                color=(1.0, 0.7, 0.3, 0.8),  # 浅橙色
                                gui_affine=False,
                                is_selectable=False
                            )
                            viewer.scene.add(pred_rfoot_spheres)
        # --- 结束足部接触可视化 ---

        # --- 计算并存储虚拟骨长和方向信息 ---
        viewer.virtual_bone_info = {}  # 初始化存储字典
        
        if has_object_gt and gt_obj_trans_seq_denorm is not None and gt_obj_rot_mat_seq_denorm is not None:
            # 计算左手虚拟骨长和方向
            lhand_pos_seq = Jtr_gt_seq[:, lhand_idx, :]  # [T, 3]
            lhand_bone_length, lhand_direction_axis_angle = compute_virtual_bone_info(
                lhand_pos_seq, gt_obj_trans_seq_denorm, gt_obj_rot_mat_seq_denorm
            )
            
            # 计算右手虚拟骨长和方向  
            rhand_pos_seq = Jtr_gt_seq[:, rhand_idx, :]  # [T, 3]
            rhand_bone_length, rhand_direction_axis_angle = compute_virtual_bone_info(
                rhand_pos_seq, gt_obj_trans_seq_denorm, gt_obj_rot_mat_seq_denorm
            )
            
            # 从模型预测中获取门控权重、物体速度输入和预测的虚拟骨长方向（如果有预测结果）
            gating_weights = None
            obj_vel_input = None
            pred_lhand_bone_length = None
            pred_rhand_bone_length = None
            pred_lhand_direction = None
            pred_rhand_direction = None
            
            if not vis_gt_only and 'gating_weights' in pred_dict:
                gating_weights = pred_dict['gating_weights'][bs].cpu().numpy()  # [T, 3]
            if not vis_gt_only and 'obj_vel_input' in pred_dict:
                obj_vel_input = pred_dict['obj_vel_input'][bs].cpu().numpy()  # [T, 3]
            
            # 虚拟骨骼信息改为 Fusion：不再优先使用 FK 结果
            using_fk_data = False

            # 如果未使用按需FK，则退回融合方案的骨长/方向（若有）
            if not using_fk_data and not vis_gt_only and 'pred_lhand_lb' in pred_dict:
                pred_lhand_bone_length = pred_dict['pred_lhand_lb'][bs].cpu().numpy()  # [T] 融合方案骨长
            if not using_fk_data and not vis_gt_only and 'pred_rhand_lb' in pred_dict:
                pred_rhand_bone_length = pred_dict['pred_rhand_lb'][bs].cpu().numpy()  # [T] 融合方案骨长
            if not using_fk_data and not vis_gt_only and 'pred_lhand_obj_direction' in pred_dict:
                pred_lhand_direction = pred_dict['pred_lhand_obj_direction'][bs].cpu().numpy()  # [T, 3] 融合方案方向
            if not using_fk_data and not vis_gt_only and 'pred_rhand_obj_direction' in pred_dict:
                pred_rhand_direction = pred_dict['pred_rhand_obj_direction'][bs].cpu().numpy()  # [T, 3] 融合方案方向
            
            
            # 存储到viewer中供GUI显示使用
            viewer.virtual_bone_info = {
                'has_data': True,
                # GT数据
                'gt_lhand_bone_length': lhand_bone_length.cpu().numpy(),  # [T] - 真值左手骨长
                'gt_rhand_bone_length': rhand_bone_length.cpu().numpy(),  # [T] - 真值右手骨长  
                'gt_lhand_direction_axis_angle': lhand_direction_axis_angle.cpu().numpy(),  # [T, 3] - 真值左手方向
                'gt_rhand_direction_axis_angle': rhand_direction_axis_angle.cpu().numpy(),  # [T, 3] - 真值右手方向
                # 预测数据
                'pred_lhand_bone_length': pred_lhand_bone_length,  # [T] or None - 预测左手骨长
                'pred_rhand_bone_length': pred_rhand_bone_length,  # [T] or None - 预测右手骨长
                'pred_lhand_direction': pred_lhand_direction,  # [T, 3] or None - 预测左手方向  
                'pred_rhand_direction': pred_rhand_direction,  # [T, 3] or None - 预测右手方向
                # FK（仅用于显示，不参与融合）
                'fk_lhand_bone_length': (fk_bone_info_seq['fk_lhand_bone_length'][0].cpu().numpy() if fk_bone_info_seq is not None else None),
                'fk_rhand_bone_length': (fk_bone_info_seq['fk_rhand_bone_length'][0].cpu().numpy() if fk_bone_info_seq is not None else None),
                # 方案标识
                'using_fk_data': using_fk_data,  # 是否在可视化时动态使用了FK方案
                'prediction_method': 'Fusion Scheme',  # 预测方法名称固定为 Fusion
                # 其他信息
                'gating_weights': gating_weights,  # [T, 3] - 门控权重
                'obj_vel_input': obj_vel_input,  # [T, 3] - 物体速度输入
                'num_frames': T
            }
        else:
            viewer.virtual_bone_info = {'has_data': False}


# === 自定义 Viewer 类 ===

class InteractiveViewer(Viewer):
    def __init__(self, data_list, model, smpl_model, config, device, obj_geo_root, show_objects=True, vis_gt_only=False, show_foot_contact=False, use_fk=False, **kwargs):
        super().__init__(**kwargs)
        self.data_list = data_list # 直接使用加载到内存的列表
        self.current_index = 0
        self.model = model
        self.smpl_model = smpl_model
        self.config = config
        self.device = device
        self.show_objects = show_objects
        self.vis_gt_only = vis_gt_only
        self.show_foot_contact = show_foot_contact
        self.obj_geo_root = obj_geo_root
        self.use_fk = use_fk
        
        # 初始化虚拟骨长信息
        self.virtual_bone_info = {'has_data': False}

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
                visualize_batch_data(self, batch, self.model, self.smpl_model, self.device, self.obj_geo_root, self.show_objects, self.vis_gt_only, self.show_foot_contact, self.use_fk)
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
        """重写GUI场景方法，添加虚拟骨长信息显示"""
        # 调用父类的GUI场景方法
        super().gui_scene()
        
        # 添加虚拟骨长信息窗口
        if self.virtual_bone_info.get('has_data', False):
            self.render_virtual_bone_info_window()
    
    def render_virtual_bone_info_window(self):
        """渲染虚拟骨长信息窗口"""
        # 获取当前帧ID
        current_frame = self.scene.current_frame_id
        num_frames = self.virtual_bone_info.get('num_frames', 0)
        
        # 确保帧ID在有效范围内
        if not (0 <= current_frame < num_frames):
            return
        
        # Create virtual bone info window - make it wider for comparison
        imgui.set_next_window_size(500, 600)  # 设置窗口大小：宽度500，高度600
        
        # 获取预测方法信息
        prediction_method = self.virtual_bone_info.get('prediction_method', 'Unknown Scheme')
        using_fk_data = self.virtual_bone_info.get('using_fk_data', False)
        
        imgui.begin(f"Virtual Bone GT vs Pred Comparison [{prediction_method}]", True)
        
        # Get current frame GT data
        gt_lhand_bone_length = self.virtual_bone_info['gt_lhand_bone_length'][current_frame]
        gt_rhand_bone_length = self.virtual_bone_info['gt_rhand_bone_length'][current_frame]
        gt_lhand_direction = self.virtual_bone_info['gt_lhand_direction_axis_angle'][current_frame]
        gt_rhand_direction = self.virtual_bone_info['gt_rhand_direction_axis_angle'][current_frame]
        
        # Get current frame prediction data (if available)
        pred_lhand_bone_length = self.virtual_bone_info.get('pred_lhand_bone_length', None)
        pred_rhand_bone_length = self.virtual_bone_info.get('pred_rhand_bone_length', None)
        pred_lhand_direction = self.virtual_bone_info.get('pred_lhand_direction', None)
        pred_rhand_direction = self.virtual_bone_info.get('pred_rhand_direction', None)
        
        # Display current frame info
        imgui.text(f"Frame: {current_frame}/{num_frames-1}")
        
        # Display prediction method info
        imgui.text(f"Prediction Method: {prediction_method}")
        if using_fk_data:
            imgui.text_colored("• Shows FK scheme initial distance and direction", 0.2, 0.8, 0.2, 1.0)
            imgui.text("• FK: Based on geometric constraints of contact segment first frame")
        else:
            imgui.text_colored("• Shows fusion scheme predicted bone length and direction", 0.8, 0.6, 0.2, 1.0)  
            imgui.text("• Fusion: Network predicted time-varying bone length and direction")
        imgui.separator()
        
        # === Left Hand Comparison ===
        imgui.text("Left Hand Virtual Bone:")
        imgui.columns(3, "LeftHandColumns")
        imgui.text("GT")
        imgui.next_column()
        imgui.text("Pred (Fusion)")
        imgui.next_column()
        imgui.text("Error")
        imgui.next_column()
        imgui.separator()
        
        # Left hand bone length comparison
        imgui.text(f"{gt_lhand_bone_length:.4f}")
        imgui.next_column()
        if pred_lhand_bone_length is not None:
            pred_len = pred_lhand_bone_length[current_frame]
            length_error = abs(pred_len - gt_lhand_bone_length)
            imgui.text(f"{pred_len:.4f}")
            imgui.next_column()
            imgui.text(f"{length_error:.4f}")
        else:
            imgui.text("N/A")
            imgui.next_column()
            imgui.text("N/A")
        imgui.next_column()

        # Left hand FK length (display only)
        fk_lhand_bone_length = self.virtual_bone_info.get('fk_lhand_bone_length', None)
        if fk_lhand_bone_length is not None:
            imgui.text_colored(f"FK Len: {fk_lhand_bone_length[current_frame]:.4f}", 1.0, 1.0, 0.0, 1.0)
        else:
            imgui.text_colored("FK Len: N/A", 1.0, 1.0, 0.0, 1.0)
        imgui.separator()
        
        # Left hand direction comparison
        imgui.text(f"[{gt_lhand_direction[0]:.3f}, {gt_lhand_direction[1]:.3f}, {gt_lhand_direction[2]:.3f}]")
        imgui.next_column()
        if pred_lhand_direction is not None:
            pred_dir = pred_lhand_direction[current_frame]
            # Normalize both vectors to unit vectors
            gt_norm = (gt_lhand_direction[0]**2 + gt_lhand_direction[1]**2 + gt_lhand_direction[2]**2)**0.5
            pred_norm = (pred_dir[0]**2 + pred_dir[1]**2 + pred_dir[2]**2)**0.5
            
            if gt_norm > 1e-8 and pred_norm > 1e-8:
                gt_unit = gt_lhand_direction / gt_norm
                pred_unit = pred_dir / pred_norm
                # Calculate angle between unit vectors: Δθ = arccos(u1 · u2)
                dot_product = gt_unit[0] * pred_unit[0] + gt_unit[1] * pred_unit[1] + gt_unit[2] * pred_unit[2]
                dot_product = max(-1.0, min(1.0, dot_product))  # Clamp to [-1, 1] for numerical stability
                angle_error_rad = __import__('math').acos(abs(dot_product))  # Use abs to get smallest angle
                angle_error_deg = angle_error_rad * 180.0 / 3.14159265359
            else:
                angle_error_deg = float('nan')
            
            imgui.text(f"[{pred_dir[0]:.3f}, {pred_dir[1]:.3f}, {pred_dir[2]:.3f}]")
            imgui.next_column()
            imgui.text(f"{angle_error_deg:.2f}°")
        else:
            imgui.text("N/A")
            imgui.next_column()
            imgui.text("N/A")
        imgui.next_column()
        
        imgui.columns(1)  # Reset to single column
        imgui.separator()
        
        # === Right Hand Comparison ===
        imgui.text("Right Hand Virtual Bone:")
        imgui.columns(3, "RightHandColumns")
        imgui.text("GT")
        imgui.next_column()
        imgui.text("Pred (Fusion)")
        imgui.next_column()
        imgui.text("Error")
        imgui.next_column()
        imgui.separator()
        
        # Right hand bone length comparison
        imgui.text(f"{gt_rhand_bone_length:.4f}")
        imgui.next_column()
        if pred_rhand_bone_length is not None:
            pred_len = pred_rhand_bone_length[current_frame]
            length_error = abs(pred_len - gt_rhand_bone_length)
            imgui.text(f"{pred_len:.4f}")
            imgui.next_column()
            imgui.text(f"{length_error:.4f}")
        else:
            imgui.text("N/A")
            imgui.next_column()
            imgui.text("N/A")
        imgui.next_column()

        # Right hand FK length (display only)
        fk_rhand_bone_length = self.virtual_bone_info.get('fk_rhand_bone_length', None)
        if fk_rhand_bone_length is not None:
            imgui.text_colored(f"FK Len: {fk_rhand_bone_length[current_frame]:.4f}", 1.0, 1.0, 0.0, 1.0)
        else:
            imgui.text_colored("FK Len: N/A", 1.0, 1.0, 0.0, 1.0)
        imgui.separator()
        
        # Right hand direction comparison
        imgui.text(f"[{gt_rhand_direction[0]:.3f}, {gt_rhand_direction[1]:.3f}, {gt_rhand_direction[2]:.3f}]")
        imgui.next_column()
        if pred_rhand_direction is not None:
            pred_dir = pred_rhand_direction[current_frame]
            # Normalize both vectors to unit vectors
            gt_norm = (gt_rhand_direction[0]**2 + gt_rhand_direction[1]**2 + gt_rhand_direction[2]**2)**0.5
            pred_norm = (pred_dir[0]**2 + pred_dir[1]**2 + pred_dir[2]**2)**0.5
            
            if gt_norm > 1e-8 and pred_norm > 1e-8:
                gt_unit = gt_rhand_direction / gt_norm
                pred_unit = pred_dir / pred_norm
                # Calculate angle between unit vectors: Δθ = arccos(u1 · u2)
                dot_product = gt_unit[0] * pred_unit[0] + gt_unit[1] * pred_unit[1] + gt_unit[2] * pred_unit[2]
                dot_product = max(-1.0, min(1.0, dot_product))  # Clamp to [-1, 1] for numerical stability
                angle_error_rad = __import__('math').acos(abs(dot_product))  # Use abs to get smallest angle
                angle_error_deg = angle_error_rad * 180.0 / 3.14159265359
            else:
                angle_error_deg = float('nan')
            
            imgui.text(f"[{pred_dir[0]:.3f}, {pred_dir[1]:.3f}, {pred_dir[2]:.3f}]")
            imgui.next_column()
            imgui.text(f"{angle_error_deg:.2f}°")
        else:
            imgui.text("N/A")
            imgui.next_column()
            imgui.text("N/A")
        imgui.next_column()
        
        imgui.columns(1)  # Reset to single column
        imgui.separator()
        
        # Display gating weights if available
        gating_weights = self.virtual_bone_info.get('gating_weights', None)
        if gating_weights is not None:
            weights = gating_weights[current_frame]  # [3]
            imgui.text("Gating Weights:")
            imgui.text(f"  L-Hand: {weights[0]:.3f}")
            imgui.text(f"  R-Hand: {weights[1]:.3f}")
            imgui.text(f"  IMU:    {weights[2]:.3f}")
            imgui.separator()
        
        # Display object velocity input if available
        obj_vel_input = self.virtual_bone_info.get('obj_vel_input', None)
        if obj_vel_input is not None:
            velocity = obj_vel_input[current_frame]  # [3]
            imgui.text("Object Velocity Input:")
            imgui.text(f"  X: {velocity[0]:.4f}")
            imgui.text(f"  Y: {velocity[1]:.4f}")
            imgui.text(f"  Z: {velocity[2]:.4f}")
            imgui.text(f"  Magnitude: {(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)**0.5:.4f}")
            imgui.separator()
        
        # Display additional info
        imgui.text("Description:")
        imgui.text("Length: Wrist to Object Distance (m)")
        imgui.text("Direction: Unit Vector in Object Frame")
        imgui.text("Error: Absolute Difference / Vector Angle Error (degrees)")
        if using_fk_data:
            imgui.text("FK Scheme: Fixed initial geometric constraints per contact segment")
        else:
            imgui.text("Fusion Scheme: Network time-varying prediction + gating fusion")
            imgui.text("Gating: L/R-Hand FK vs IMU Integration")
        
        # Add total progress info
        progress = (current_frame + 1) / num_frames * 100
        imgui.text(f"Progress: {progress:.1f}%")
        
        imgui.end()

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
    parser.add_argument('--config', type=str, default='configs/TransPose_train.yaml', help='Path to the main configuration file (used for model, dataset params).')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the trained TransPose model checkpoint. Overrides config if provided.')
    parser.add_argument('--smpl_model_path', type=str, default=None, help='Path to the SMPLH model file. Overrides config if provided.')
    parser.add_argument('--test_data_dir', type=str, default=None, help='Path to the test dataset directory. Overrides config if provided.')
    parser.add_argument('--obj_geo_root', type=str, default='./dataset/captured_objects', help='Path to the object geometry root directory.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader (should be 1 for sequential vis).')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers.')
    parser.add_argument('--no_objects', action='store_true', help='Do not load or visualize objects.')
    parser.add_argument('--vis_gt_only', action='store_true', help='Only visualize ground truth, skip model inference and prediction visualization.')
    parser.add_argument('--show_foot_contact', action='store_true', help='Visualize foot-ground contact indicators.')
    parser.add_argument('--use_fk', action='store_true', help='Use on-demand FK for object translation and virtual bone info in visualization.')
    parser.add_argument('--limit_sequences', type=int, default=None, help='Limit the number of sequences to load for visualization.')
    parser.add_argument('--full_sequence', action='store_true', help='Use full-length sequences (no fixed window slicing).')
    args = parser.parse_args()

    if args.batch_size != 1:
        print("Warning: Setting batch_size to 1 for interactive visualization.")
        args.batch_size = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Override config with command line args
    if args.model_path: config.model_path = args.model_path
    if args.smpl_model_path: config.bm_path = args.smpl_model_path
    if args.test_data_dir: config.test.data_path = args.test_data_dir
    if args.num_workers is not None: config.num_workers = args.num_workers
    # Ensure test config exists or copy from train
    if 'test' not in config: config.test = config.train.copy()
    config.test.batch_size = args.batch_size # Force batch size 1

    # --- Load SMPL Model ---
    smpl_model_path = config.get('bm_path', 'body_models/smplh/neutral/model.npz')
    smpl_model = load_smpl_model(smpl_model_path, device)

    # --- Load Trained Model ---
    model_path = config.get('model_path', None)
    if not model_path:
        print("Error: No model path provided in config or via --model_path.")
        # return
    # 使用配置中的 pretrained_modules 加载模块（与 eval.py 保持一致）
    staged_cfg = config.get('staged_training', {}) if hasattr(config, 'get') else config.staged_training
    modular_cfg = staged_cfg.get('modular_training', {}) if staged_cfg else {}
    use_modular = bool(modular_cfg.get('enabled', False))
    pretrained_modules = modular_cfg.get('pretrained_modules', {}) if use_modular else {}
    if use_modular and pretrained_modules:
        print("Loading TransPose model with pretrained modules for visualization:")
        for k, v in pretrained_modules.items():
            print(f"  - {k}: {v}")
        model = TransPoseNet(config, pretrained_modules=pretrained_modules, skip_modules=[]).to(device)
    else:
        print("Warning: No pretrained_modules in config; initializing a fresh TransPoseNet for visualization.")
        model = TransPoseNet(config).to(device)
    model.eval()

    # --- Load Test Dataset ---
    # 优先使用 OMOMO 数据集配置
    test_data_dir = None
    try:
        if 'datasets' in config and 'omomo' in config.datasets and 'test_path' in config.datasets.omomo:
            test_data_dir = config.datasets.omomo.test_path
    except Exception:
        pass
    if args.test_data_dir:
        test_data_dir = args.test_data_dir
    if test_data_dir is None and hasattr(config, 'test') and 'data_path' in config.test:
        test_data_dir = config.test.data_path
    if not test_data_dir or not os.path.exists(test_data_dir):
        print(f"Error: Test dataset path not found or invalid: {test_data_dir}")
        return
    print(f"Loading test dataset from: {test_data_dir}")

    # Use test window size from config, default if not present
    test_window_size = config.test.get('window', config.train.get('window', 60))

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
        batch_size=config.test.batch_size, # Should be 1
        shuffle=False, # IMPORTANT: Keep order for navigation
        num_workers=config.get('num_workers', 0), # Set workers based on args/config
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
        config=config,
        device=device,
        obj_geo_root=args.obj_geo_root,
        show_objects=(not args.no_objects),
        vis_gt_only=args.vis_gt_only,
        show_foot_contact=args.show_foot_contact,
        use_fk=args.use_fk,
        window_size=(1920, 1080) # Example window size
        # Add other Viewer kwargs if needed (e.g., fps)
    )
    print("Viewer Initialized. Navigation controls:")
    print("  q/e: Previous/Next 1 sequence")
    print("  Ctrl+q/e: Previous/Next 10 sequences")
    print("  Alt+q/e: Previous/Next 50 sequences")
    if args.show_foot_contact:
        print("Foot contact visualization enabled:")
        print("  GT Left Foot: Purple spheres")
        print("  GT Right Foot: Orange spheres")
        print("  Pred Left Foot: Light Purple spheres")
        print("  Pred Right Foot: Light Orange spheres")
    print("Other standard aitviewer controls should also work (e.g., mouse drag to rotate, scroll to zoom).")
    viewer_instance.run()


if __name__ == "__main__":
    main() 