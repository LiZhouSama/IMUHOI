import torch
import os
import numpy as np
import random
import argparse
import yaml
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.viewer import Viewer
from aitviewer.scene.camera import Camera
from moderngl_window.context.base import KeyModifiers
import pytorch3d.transforms as transforms
import trimesh

from human_body_prior.body_model.body_model import BodyModel
from easydict import EasyDict as edict

from torch.utils.data import DataLoader
from dataloader.dataloader import IMUDataset # 从 eval.py 引入

# 导入模型相关 - 根据需要选择正确的模型加载方式
# from models.DiT_model import MotionDiffusion # 如果要用 DiT
from models.TransPose_net import TransPoseNet # 明确使用 TransPose
from models.do_train_imu_TransPose import load_transpose_model # 或者使用这个加载函数

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
        print("警告: 物体名称为 None，无法加载几何体。")
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
             print(f"警告: 找不到物体 {obj_name} 的两部分几何文件。将尝试加载整体文件。")
             two_parts = False
             obj_mesh_path = os.path.join(obj_geo_root, f"{obj_name}_cleaned_simplified.obj") # Fallback
        else:
            top_obj_mesh_verts, top_obj_mesh_faces = apply_transformation_to_obj_geometry(top_obj_mesh_path, obj_rot, obj_trans, scale=obj_scale, device=device)
            # Assume bottom uses the same scale, pass bottom transforms
            bottom_obj_mesh_verts, bottom_obj_mesh_faces = apply_transformation_to_obj_geometry(bottom_obj_mesh_path, obj_bottom_rot, obj_bottom_trans, scale=obj_scale, device=device)
            obj_mesh_verts, obj_mesh_faces = merge_two_parts([top_obj_mesh_verts, bottom_obj_mesh_verts], [top_obj_mesh_faces, bottom_obj_mesh_faces], device=device)

    if not two_parts:
        if not os.path.exists(obj_mesh_path):
             print(f"警告: 找不到物体几何文件: {obj_mesh_path}")
             return torch.zeros((obj_trans.shape[0] if obj_trans is not None else 1, 1, 3), device=device), np.zeros((1, 3), dtype=np.int64)
        obj_mesh_verts, obj_mesh_faces = apply_transformation_to_obj_geometry(obj_mesh_path, obj_rot, obj_trans, scale=obj_scale, device=device)

    return obj_mesh_verts, obj_mesh_faces


def visualize_batch_data(viewer, batch, model, smpl_model, device, obj_geo_root, show_objects=True):
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
                node.name == "GT-LHandContact" or  # 明确手部接触名称
                node.name == "GT-RHandContact" or  # 明确手部接触名称
                node.name == "ObjContactIndicator" or # 物体运动指示器名称
                node.name == "Pred-LHandContact" or # 预测左手接触
                node.name == "Pred-RHandContact" or # 预测右手接触
                node.name == "Pred-ObjContactIndicator") # 预测物体接触
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

        gt_root_orient_mat = root_global_rot_start @ gt_root_orient_mat_norm
        gt_root_orient_axis = transforms.matrix_to_axis_angle(gt_root_orient_mat).reshape(T, 3)
        gt_root_pos_seq = (root_global_rot_start @ gt_root_pos_seq.unsqueeze(-1)).squeeze(-1) + root_global_pos_start
        # gt_root_orient_axis = transforms.matrix_to_axis_angle(gt_root_orient_mat_norm)

        gt_smplh_input = {
            'root_orient': gt_root_orient_axis,
            'pose_body': gt_pose_body_axis,
            'trans': gt_root_pos_seq
        }
        body_pose_gt = smpl_model(**gt_smplh_input)
        verts_gt_seq = body_pose_gt.v                          # [T, Nv, 3]
        faces_gt_np = smpl_model.f.cpu().numpy() if isinstance(smpl_model.f, torch.Tensor) else smpl_model.f

        # --- 3. 模型预测 ---
        pred_motion_seq = None
        pred_obj_rot_6d_seq = None
        pred_obj_trans_seq = None # 现在模型会预测物体平移
        pred_root_pos_seq = None

        # --- Define a visual offset for predicted elements ---
        pred_offset = torch.tensor([0.0, 0.0, 0.0], device=device)

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

        try:
            pred_dict = model(model_input)
            pred_motion = pred_dict.get("motion") # [bs, T, 132]
            pred_obj_rot = pred_dict.get("obj_rot") # [bs, T, 6] (TransPose 输出 6D)
            pred_obj_trans = pred_dict.get("pred_obj_trans") # [bs, T, 3] (新增：预测物体平移)
            pred_root_pos = pred_dict.get("root_pos") # [bs, T, 3]

            # --- Get predicted contact probabilities ---
            pred_hand_contact_prob_batch = pred_dict.get("pred_hand_contact_prob") # [bs, T, 3]
            pred_lhand_contact_labels_seq = None
            pred_rhand_contact_labels_seq = None
            pred_obj_contact_labels_seq = None
            if pred_hand_contact_prob_batch is not None:
                pred_hand_contact_prob_seq = pred_hand_contact_prob_batch[bs].to(device) # [T, 3]
                # Convert probabilities to 0/1 labels
                pred_contact_labels = (pred_hand_contact_prob_seq > 0.5).bool()
                pred_lhand_contact_labels_seq = pred_contact_labels[:, 0]
                pred_rhand_contact_labels_seq = pred_contact_labels[:, 1]
                pred_obj_contact_labels_seq = pred_contact_labels[:, 2]
            else:
                print("Warning: Model did not output 'pred_hand_contact_prob'.")
            # --- End predicted contact probabilities ---

            if pred_motion is not None:
                pred_motion_seq = pred_motion[bs] # [T, 132]
            else:
                print("警告: 模型未输出 'motion'")

            if pred_root_pos is not None:
                pred_root_pos_seq = pred_root_pos[bs] # [T, 3]
            else:
                print("警告: 模型未输出 'root_pos'")

            if pred_obj_rot is not None:
                pred_obj_rot_6d_seq = pred_obj_rot[bs] # [T, 6]
            elif has_object_data_for_model:
                print("警告: 模型未输出 'obj_rot'，即使有物体 IMU 输入")

            # 新增：获取预测的物体平移
            if pred_obj_trans is not None:
                pred_obj_trans_seq = pred_obj_trans[bs] # [T, 3]
            elif has_object_data_for_model:
                print("警告: 模型未输出 'obj_trans'，即使有物体 IMU 输入")

        except Exception as e:
            print(f"模型推理失败: {e}")
            import traceback
            traceback.print_exc()

        # --- 4. 获取预测 SMPL (使用预测 motion + 真值 trans) ---
        verts_pred_seq = None
        if pred_motion_seq is not None:
            pred_rot_matrices = transforms.rotation_6d_to_matrix(pred_motion_seq.reshape(T, 22, 6)) # [T, 22, 3, 3]
            pred_root_orient_mat_norm = pred_rot_matrices[:, 0]                         # [T, 3, 3]
            pred_pose_body_mat = pred_rot_matrices[:, 1:].reshape(T * 21, 3, 3)    # [T*21, 3, 3]
            pred_pose_body_axis = transforms.matrix_to_axis_angle(pred_pose_body_mat).reshape(T, -1) # [T, 63]

            # Denormalization
            # pred_root_orient_mat = head_global_rot_start @ pred_root_orient_mat_norm
            # pred_root_orient_axis = transforms.matrix_to_axis_angle(pred_root_orient_mat).reshape(T, 3)
            # pred_root_pos_seq = (head_global_rot_start @ pred_root_pos_seq.unsqueeze(-1)).squeeze(-1) + head_global_pos_start

            pred_root_orient_mat = root_global_rot_start @ pred_root_orient_mat_norm
            pred_root_orient_axis = transforms.matrix_to_axis_angle(pred_root_orient_mat).reshape(T, 3)
            pred_root_pos_seq = (root_global_rot_start @ pred_root_pos_seq.unsqueeze(-1)).squeeze(-1) + root_global_pos_start

            pred_smplh_input = {
                'root_orient': pred_root_orient_axis,
                'pose_body': pred_pose_body_axis,
                'trans': pred_root_pos_seq
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
            gt_obj_rot_mat_seq = transforms.rotation_6d_to_matrix(gt_obj_rot_6d_seq) # [T, 3, 3]
            gt_obj_scale_seq = gt_obj_scale[bs] if gt_obj_scale is not None else None # [T] or [T, 1]?
            # Handle bottom parts if they exist
            gt_obj_bottom_trans_seq = gt_obj_bottom_trans[bs] if gt_obj_bottom_trans is not None else None
            gt_obj_bottom_rot_seq = gt_obj_bottom_rot[bs] if gt_obj_bottom_rot is not None else None

            # Denormalization
            # gt_obj_rot_mat_seq = head_global_rot_start @ gt_obj_rot_mat_seq
            # gt_obj_trans_seq = (head_global_rot_start @ gt_obj_trans_seq.unsqueeze(-1)).squeeze(-1) + head_global_pos_start
            
            gt_obj_rot_mat_seq = root_global_rot_start @ gt_obj_rot_mat_seq
            gt_obj_trans_seq = (root_global_rot_start @ gt_obj_trans_seq.unsqueeze(-1)).squeeze(-1) + root_global_pos_start
            # gt_obj_rot_mat_seq = gt_obj_rot_mat_seq

            # 获取真值物体
            gt_obj_verts_seq, obj_faces_np = load_object_geometry(
                obj_name, gt_obj_rot_mat_seq, gt_obj_trans_seq, gt_obj_scale_seq, device=device
            )

            # 获取预测物体 (使用真值旋转 + 预测平移)
            if pred_obj_trans_seq is not None:
                # 对预测的物体平移进行反归一化
                pred_obj_trans_seq_denorm = (root_global_rot_start @ pred_obj_trans_seq.unsqueeze(-1)).squeeze(-1) + root_global_pos_start
                
                # 使用真值旋转 + 预测平移
                pred_obj_verts_seq, _ = load_object_geometry(
                    obj_name, 
                    gt_obj_rot_mat_seq, # 使用真值旋转
                    pred_obj_trans_seq_denorm, # 使用预测平移
                    gt_obj_scale_seq, 
                    device=device
                )
            else:
                # 如果没有预测平移，回退到使用真值平移
                print("警告: 没有预测的物体平移，使用真值平移进行可视化")
                pred_obj_verts_seq, _ = load_object_geometry(
                    obj_name, 
                    gt_obj_rot_mat_seq, # 使用真值旋转
                    gt_obj_trans_seq,   # 使用真值平移
                    gt_obj_scale_seq, 
                    device=device
                )

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

        # 添加预测人体 (红色, 偏移 x=1.0)
        if verts_pred_seq is not None:
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

        # 添加预测物体 (红色, 偏移 x=1.0)
        if pred_obj_verts_seq is not None and obj_faces_np is not None:
            pred_obj_verts_shifted = pred_obj_verts_seq + pred_offset # 使用定义的偏移
            pred_obj_verts_yup = torch.matmul(pred_obj_verts_shifted, R_yup.T.to(device))
            pred_obj_mesh = Meshes(
                pred_obj_verts_yup.cpu().numpy(), obj_faces_np,
                name=f"Pred-{obj_name}", color=(0.9, 0.2, 0.2, 0.8), gui_affine=False, is_selectable=False
            )
            viewer.scene.add(pred_obj_mesh)

        lhand_contact_seq = batch["lhand_contact"][bs] # [T]
        rhand_contact_seq = batch["rhand_contact"][bs] # [T]
        obj_contact_seq = batch["obj_contact"][bs] # [T]
        # --- 获取 GT 关节点位置 ---
        Jtr_gt_seq = body_pose_gt.Jtr # [T, J, 3]

        # --- 获取 Pred 关节点位置 (如果预测了 pose) ---
        Jtr_pred_seq = None
        if verts_pred_seq is not None: # 仅当有预测姿态时才计算
            Jtr_pred_seq = body_pose_pred.Jtr # [T, J, 3]


        # --- 定义手腕关节点索引 (请根据你的模型确认) ---
        lhand_idx = 20
        rhand_idx = 21

        # --- 应用新的接触可视化逻辑 ---
        contact_radius = 0.03 # 可调整

        # --- 可视化左手接触 (红球) ---
        gt_lhand_contact_points_list = []
        for t in range(T): # 遍历每一帧
            if lhand_contact_viz_seq[t]: # 使用预处理的标志
                gt_lhand_contact_points_list.append(Jtr_gt_seq[t, lhand_idx])
        
        if gt_lhand_contact_points_list:
            gt_lhand_contact_points = torch.stack(gt_lhand_contact_points_list, dim=0)
            if gt_lhand_contact_points.numel() > 0:
                gt_lhand_points_yup_tensor = torch.matmul(gt_lhand_contact_points, R_yup.T.to(device))
                gt_lhand_points_yup_np = gt_lhand_points_yup_tensor.cpu().numpy()
                gt_lhand_spheres = Spheres(
                    positions=gt_lhand_points_yup_np,
                    radius=contact_radius,
                    name="GT-LHandContact",
                    color=(1.0, 0.0, 0.0, 0.8), # 红色
                    gui_affine=False,
                    is_selectable=False
                )
                viewer.scene.add(gt_lhand_spheres)

        # --- 可视化右手接触 (蓝球) ---
        gt_rhand_contact_points_list = []
        for t in range(T):
             if rhand_contact_viz_seq[t]: # 使用预处理的标志
                 gt_rhand_contact_points_list.append(Jtr_gt_seq[t, rhand_idx])

        if gt_rhand_contact_points_list:
             gt_rhand_contact_points = torch.stack(gt_rhand_contact_points_list, dim=0)
             if gt_rhand_contact_points.numel() > 0:
                gt_rhand_points_yup_tensor = torch.matmul(gt_rhand_contact_points, R_yup.T.to(device))
                gt_rhand_points_yup_np = gt_rhand_points_yup_tensor.cpu().numpy()
                gt_rhand_spheres = Spheres(
                    positions=gt_rhand_points_yup_np,
                    radius=contact_radius,
                    name="GT-RHandContact", # 确保名称一致
                    color=(0.0, 0.0, 1.0, 0.8), # 蓝色
                    gui_affine=False,
                    is_selectable=False
                )
                viewer.scene.add(gt_rhand_spheres)

        # --- 可视化物体移动指示 (黄球) ---
        # 确保 gt_obj_trans_seq 可用
        if gt_obj_trans_seq is not None:
            obj_indicator_points_list = []
            for t in range(T):
                if obj_contact_viz_seq[t]: # 使用预处理的标志
                    obj_indicator_points_list.append(gt_obj_trans_seq[t])
            
            if obj_indicator_points_list:
                contact_positions = torch.stack(obj_indicator_points_list, dim=0)
                contact_positions_yup_tensor = torch.matmul(contact_positions, R_yup.T.to(device))
                contact_positions_yup_np = contact_positions_yup_tensor.cpu().numpy()

                contact_indicator_radius = 0.04 # 可调整大小
                contact_indicator_color = (1.0, 1.0, 0.0, 0.8) # 黄色

                obj_contact_spheres = Spheres(
                    positions=contact_positions_yup_np,
                    radius=contact_indicator_radius,
                    name="ObjContactIndicator", # 用于场景清理
                    color=contact_indicator_color,
                    gui_affine=False,
                    is_selectable=False
                )
                viewer.scene.add(obj_contact_spheres)
        # --- 结束物体接触可视化 ---
        
        # --- 可视化预测的接触标签 ---
        contact_radius_pred = 0.03 # Can be same or different from GT

        # 预测左手接触
        if pred_lhand_contact_labels_seq is not None and Jtr_pred_seq is not None:
            pred_lhand_contact_points_list = []
            for t in range(T):
                if pred_lhand_contact_labels_seq[t]:
                    point_on_pred_human = Jtr_pred_seq[t, lhand_idx]
                    pred_lhand_contact_points_list.append(point_on_pred_human + pred_offset)
            
            if pred_lhand_contact_points_list:
                pred_lhand_contact_points = torch.stack(pred_lhand_contact_points_list, dim=0)
                if pred_lhand_contact_points.numel() > 0:
                    pred_lhand_points_yup_tensor = torch.matmul(pred_lhand_contact_points, R_yup.T.to(device))
                    pred_lhand_points_yup_np = pred_lhand_points_yup_tensor.cpu().numpy()
                    pred_lhand_spheres = Spheres(
                        positions=pred_lhand_points_yup_np,
                        radius=contact_radius_pred,
                        name="Pred-LHandContact",
                        color=(1.0, 0.0, 0.0, 0.8), # 红色
                        gui_affine=False,
                        is_selectable=False
                    )
                    viewer.scene.add(pred_lhand_spheres)

        # 预测右手接触
        if pred_rhand_contact_labels_seq is not None and Jtr_pred_seq is not None:
            pred_rhand_contact_points_list = []
            for t in range(T):
                if pred_rhand_contact_labels_seq[t]:
                    point_on_pred_human = Jtr_pred_seq[t, rhand_idx]
                    pred_rhand_contact_points_list.append(point_on_pred_human + pred_offset)
            
            if pred_rhand_contact_points_list:
                pred_rhand_contact_points = torch.stack(pred_rhand_contact_points_list, dim=0)
                if pred_rhand_contact_points.numel() > 0:
                    pred_rhand_points_yup_tensor = torch.matmul(pred_rhand_contact_points, R_yup.T.to(device))
                    pred_rhand_points_yup_np = pred_rhand_points_yup_tensor.cpu().numpy()
                    pred_rhand_spheres = Spheres(
                        positions=pred_rhand_points_yup_np,
                        radius=contact_radius_pred,
                        name="Pred-RHandContact",
                        color=(0.0, 0.0, 1.0, 0.8), # 蓝色
                        gui_affine=False,
                        is_selectable=False
                    )
                    viewer.scene.add(pred_rhand_spheres)
        # 预测物体移动指示
        if pred_obj_contact_labels_seq is not None: # 检查是否有预测的物体接触标签
            # 优先使用预测的物体平移，如果没有则使用真值
            obj_trans_for_pred_contact = pred_obj_trans_seq_denorm if pred_obj_trans_seq is not None else gt_obj_trans_seq
            
            if obj_trans_for_pred_contact is not None:
                pred_obj_indicator_points_list = []
                for t in range(T):
                    if pred_obj_contact_labels_seq[t]:
                        # 使用预测的物体平移位置 + 预测偏移
                        pred_obj_indicator_points_list.append(obj_trans_for_pred_contact[t] + pred_offset)
                
                if pred_obj_indicator_points_list:
                    pred_obj_contact_positions = torch.stack(pred_obj_indicator_points_list, dim=0)
                    if pred_obj_contact_positions.numel() > 0:
                        pred_obj_contact_positions_yup_tensor = torch.matmul(pred_obj_contact_positions, R_yup.T.to(device))
                        pred_obj_contact_positions_yup_np = pred_obj_contact_positions_yup_tensor.cpu().numpy()
                        pred_obj_contact_spheres = Spheres(
                            positions=pred_obj_contact_positions_yup_np,
                            radius=contact_radius_pred, # Can use same or different radius
                            name="Pred-ObjContactIndicator",
                            color=(1.0, 1.0, 0.0, 0.8), # 黄色
                            gui_affine=False,
                            is_selectable=False
                        )
                        viewer.scene.add(pred_obj_contact_spheres)
        # --- 结束预测接触可视化 ---


# === 自定义 Viewer 类 ===

class InteractiveViewer(Viewer):
    def __init__(self, data_list, model, smpl_model, config, device, obj_geo_root, show_objects=True, **kwargs):
        super().__init__(**kwargs)
        self.data_list = data_list # 直接使用加载到内存的列表
        self.current_index = 0
        self.model = model
        self.smpl_model = smpl_model
        self.config = config
        self.device = device
        self.show_objects = show_objects
        self.obj_geo_root = obj_geo_root

        # 设置初始相机位置 (可选)
        # self.scene.camera.position = np.array([0.0, 1.0, 3.0])
        # self.scene.camera.target = np.array([0.5, 0.8, 0.0]) # 对准偏移后的中间区域

        # 初始可视化
        self.visualize_current_sequence()

    def visualize_current_sequence(self):
        if not self.data_list:
            print("错误：数据列表为空。")
            return
        if 0 <= self.current_index < len(self.data_list):
            batch = self.data_list[self.current_index]
            print(f"Visualizing sequence index: {self.current_index}")
            try:
                visualize_batch_data(self, batch, self.model, self.smpl_model, self.device, self.obj_geo_root, self.show_objects)
                self.title = f"Sequence Index: {self.current_index}/{len(self.data_list)-1} (q/e:±1, Ctrl+q/e:±10, Alt+q/e:±50)"
            except Exception as e:
                 print(f"Error visualizing sequence {self.current_index}: {e}")
                 import traceback
                 traceback.print_exc()
                 self.title = f"Error visualizing index: {self.current_index}"
        else:
            print("Index out of bounds.")

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
                        print(f"后退50个序列到索引: {self.current_index}")
                    else:
                        print("已经在最前面的序列。")
                elif ctrl_pressed:
                    # Ctrl + Q: 后退10个index
                    step = 10
                    new_index = max(0, self.current_index - step)
                    if new_index != self.current_index:
                        self.current_index = new_index
                        self.visualize_current_sequence()
                        self.scene.current_frame_id = 0
                        print(f"后退10个序列到索引: {self.current_index}")
                    else:
                        print("已经在最前面的序列。")
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
                        print(f"前进50个序列到索引: {self.current_index}")
                    else:
                        print("已经在最后面的序列。")
                elif ctrl_pressed:
                    # Ctrl + E: 前进10个index
                    step = 10
                    new_index = min(len(self.data_list) - 1, self.current_index + step)
                    if new_index != self.current_index:
                        self.current_index = new_index
                        self.visualize_current_sequence()
                        self.scene.current_frame_id = 0
                        print(f"前进10个序列到索引: {self.current_index}")
                    else:
                        print("已经在最后面的序列。")
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
    parser.add_argument('--limit_sequences', type=int, default=None, help='Limit the number of sequences to load for visualization.')
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
        return
    model = load_transpose_model(config, model_path)
    model = model.to(device)
    model.eval()

    # --- Load Test Dataset ---
    test_data_dir = config.test.get('data_path', None)
    if not test_data_dir or not os.path.exists(test_data_dir):
        print(f"Error: Test dataset path not found or invalid: {test_data_dir}")
        return
    print(f"Loading test dataset from: {test_data_dir}")

    # Use test window size from config, default if not present
    test_window_size = config.test.get('window', config.train.get('window', 60))

    test_dataset = IMUDataset(
        data_dir=test_data_dir,
        window_size=test_window_size,
        window_stride=config.test.get('window_stride', test_window_size), # Use stride from config
        normalize=config.test.get('normalize', True),
        debug=config.get('debug', False)
    )

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
        data_list.append(batch)
        if i % 50 == 0 and i > 0:
            print(f"  Loaded {i+1} sequences...")
    print(f"Finished loading {len(data_list)} sequences.")

    if not data_list:
        print("Error: No data loaded into the list.")
        return

    # --- Initialize and Run Viewer ---
    print("Initializing Interactive Viewer...")
    viewer_instance = InteractiveViewer(
        data_list=data_list,
        model=model,
        smpl_model=smpl_model,
        config=config,
        device=device,
        obj_geo_root=args.obj_geo_root,
        show_objects=(not args.no_objects),
        window_size=(1920, 1080) # Example window size
        # Add other Viewer kwargs if needed (e.g., fps)
    )
    print("Viewer Initialized. Navigation controls:")
    print("  q/e: 前进/后退 1个序列")
    print("  Ctrl+q/e: 前进/后退 10个序列")
    print("  Alt+q/e: 前进/后退 50个序列")
    print("Other standard aitviewer controls should also work (e.g., mouse drag to rotate, scroll to zoom).")
    viewer_instance.run()


if __name__ == "__main__":
    main() 