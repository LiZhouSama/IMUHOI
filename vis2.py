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
from dataloader.dataloader import IMUDataset

# 导入不同版本的模型
from models.TransPose_net import TransPoseNet as OriginalTransPoseNet
from models.TransPose_net_simpleObjT import TransPoseNet as SimpleObjTTransPoseNet

import imgui
from aitviewer.renderables.spheres import Spheres


# --- 定义 Z-up 到 Y-up 的旋转矩阵 ---
R_yup = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, -1.0, 0.0]], dtype=torch.float32)

# === 辅助函数 ===

class SafeDataBatch:
    """安全的数据批次处理类，确保所有数据都在正确的设备上"""
    def __init__(self, batch, device):
        self.device = device
        self.data = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                self.data[key] = value.to(device)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                self.data[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
            else:
                self.data[key] = value
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def __getitem__(self, key):
        return self.data[key]
    
    def keys(self):
        return self.data.keys()
    
    def items(self):
        return self.data.items()

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
        model_type='smplh'
    ).to(device)
    return smpl_model

def create_modular_original_model(config, device):
    """创建原始版本的模块化TransPose模型"""
    print("Creating original TransPose model with pretrained modules...")
    
    # 获取预训练模块路径
    pretrained_modules = {}
    if ('staged_training' in config and 
        'modular_training' in config.staged_training and 
        'pretrained_modules' in config.staged_training.modular_training):
        pretrained_modules = config.staged_training.modular_training.pretrained_modules
    
    print(f"Original model pretrained modules: {pretrained_modules}")
    
    # 使用训练脚本中的加载函数创建原始模型
    try:
        from models.do_train_imu_TransPose import load_transpose_model
        # 如果有模型路径配置，使用标准加载方式
        if hasattr(config, 'model_path') and config.model_path:
            model = load_transpose_model(config, config.model_path)
        else:
            # 否则创建模块化模型
            model = OriginalTransPoseNet(
                cfg=config,
                pretrained_modules=pretrained_modules,
                skip_modules=[]
            ).to(device)
    except Exception as e:
        print(f"Failed to create original model using load function: {e}")
        # 回退到直接创建
        model = OriginalTransPoseNet(
            cfg=config,
            pretrained_modules=pretrained_modules,
            skip_modules=[]
        ).to(device)
    
    model.eval()
    return model

def create_modular_simple_objt_model(config, device, simple_objt_module_path=None):
    """创建SimpleObjT版本的模块化TransPose模型（专注于物体位移估计）"""
    print("Creating SimpleObjT TransPose model for object translation estimation...")
    
    # 获取预训练模块路径
    pretrained_modules = {}
    if simple_objt_module_path and os.path.exists(simple_objt_module_path):
        # 使用提供的SimpleObjT物体位移模块
        pretrained_modules['object_trans'] = simple_objt_module_path
        print(f"Using SimpleObjT object_trans module from: {simple_objt_module_path}")
    else:
        # 如果没有提供专用的SimpleObjT模块，跳过其他模块，只训练object_trans
        print("No SimpleObjT module provided, will use fresh object_trans module")
    
    print(f"SimpleObjT model pretrained modules: {pretrained_modules}")
    
    # 创建SimpleObjT模型（只包含object_trans模块）
    model = SimpleObjTTransPoseNet(
        cfg=config,
        pretrained_modules=pretrained_modules,
        skip_modules=['velocity_contact', 'human_pose']  # 跳过人体相关模块
    ).to(device)
    
    model.eval()
    return model

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
    """ 加载物体几何体并应用变换 (采用vis.py的实现) """
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

def visualize_comparative_results(viewer, batch, original_model, simple_objt_model, smpl_model, device, obj_geo_root, show_objects=True):
    """
    可视化对比结果：真值、原模型预测、SimpleObjT物体预测
    """
    # 清除之前的渲染对象
    try:
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
                node.name == "Pred-ObjContactIndicator" or # 预测物体接触
                node.name == "GT-LFootContact" or  # 真值左脚接触
                node.name == "GT-RFootContact" or  # 真值右脚接触
                node.name == "Pred-LFootContact" or # 预测左脚接触
                node.name == "Pred-RFootContact") or # 预测右脚接触
                node.name.startswith("Original-") or
                node.name.startswith("SimpleObjT-")
        ]

        for node_to_remove in nodes_to_remove:
            try:
                viewer.scene.remove(node_to_remove)
            except Exception as e:
                print(f"Error removing node '{node_to_remove.name}' from scene: {e}")

    except Exception as e:
        print(f"Error during scene clearing: {e}")

    # 使用SafeDataBatch确保数据在正确设备上
    safe_batch = SafeDataBatch(batch, device)

    with torch.no_grad():
        bs = 0
        # --- 1. 准备数据（严格按照vis.py的逻辑）---
        gt_root_pos = safe_batch["root_pos"]         # [bs, T, 3]
        gt_motion = safe_batch["motion"]           # [bs, T, 132]
        human_imu = safe_batch["human_imu"]        # [bs, T, num_imus, 9/12]
        root_global_pos_start = safe_batch["root_pos_start"]  # [bs, 3]
        root_global_rot_start = safe_batch["root_rot_start"]  # [bs, 3, 3]
        obj_imu = safe_batch.get("obj_imu", None)             # [bs, T, 1, 9/12] or None
        gt_obj_trans = safe_batch.get("obj_trans", None)      # [bs, T, 3] or None
        gt_obj_rot_6d = safe_batch.get("obj_rot", None)       # [bs, T, 6] or None
        obj_name = safe_batch.get("obj_name", [None])[0]      # 物体名称 (取列表第一个)
        gt_obj_scale = safe_batch.get("obj_scale", None)      # [bs, T] or [bs, T, 1]? Check dataloader
        gt_obj_bottom_trans = safe_batch.get("obj_bottom_trans", None) # [bs, T, 3] or None
        gt_obj_bottom_rot = safe_batch.get("obj_bottom_rot", None)     # [bs, T, 3, 3] or None

        # 仅处理批次中的第一个序列 (bs=0)
        T = gt_motion.shape[1]
        gt_root_pos_seq = gt_root_pos[bs]           # [T, 3]
        gt_motion_seq = gt_motion[bs]             # [T, 132]
        root_global_rot_start = root_global_rot_start[bs]  # [3, 3]
        root_global_pos_start = root_global_pos_start[bs]  # [3]

        seq_len = T

                # === 2. 获取真值 SMPL（严格按照vis.py逻辑）===
        print("Visualizing Ground Truth...")
        
        gt_rot_matrices = transforms.rotation_6d_to_matrix(gt_motion_seq.reshape(T, 22, 6)) # [T, 22, 3, 3]
        gt_root_orient_mat_norm = gt_rot_matrices[:, 0]                         # [T, 3, 3]
        gt_pose_body_mat = gt_rot_matrices[:, 1:].reshape(T * 21, 3, 3)    # [T*21, 3, 3]
        gt_pose_body_axis = transforms.matrix_to_axis_angle(gt_pose_body_mat).reshape(T, -1) # [T, 63]

        # Denormalization（关键步骤）
        gt_root_orient_mat = root_global_rot_start @ gt_root_orient_mat_norm
        gt_root_orient_axis = transforms.matrix_to_axis_angle(gt_root_orient_mat).reshape(T, 3)
        gt_root_pos_seq = (root_global_rot_start @ gt_root_pos_seq.unsqueeze(-1)).squeeze(-1) + root_global_pos_start

        gt_smplh_input = {
            'root_orient': gt_root_orient_axis,
            'pose_body': gt_pose_body_axis,
            'trans': gt_root_pos_seq
        }
        body_pose_gt = smpl_model(**gt_smplh_input)
        verts_gt_seq = body_pose_gt.v                          # [T, Nv, 3]
        faces_gt_np = smpl_model.f.cpu().numpy() if isinstance(smpl_model.f, torch.Tensor) else smpl_model.f

        # === 3. 模型预测（预定义变量）===
        pred_motion_seq = None
        pred_obj_rot_6d_seq = None
        pred_obj_trans_seq = None 
        pred_root_pos_seq = None

        # --- Define a visual offset for predicted elements ---
        pred_offset = torch.tensor([0.0, 0.0, 0.0], device=device)

        # === 4. 获取物体几何体（严格按照vis.py逻辑）===
        gt_obj_verts_seq = None
        pred_obj_verts_seq = None
        obj_faces_np = None
        has_object_gt = gt_obj_trans is not None and gt_obj_rot_6d is not None and obj_name is not None

        if show_objects and has_object_gt:
            gt_obj_trans_seq = gt_obj_trans[bs]     # [T, 3]
            gt_obj_rot_6d_seq = gt_obj_rot_6d[bs]   # [T, 6]
            gt_obj_rot_mat_seq = transforms.rotation_6d_to_matrix(gt_obj_rot_6d_seq) # [T, 3, 3]
            gt_obj_scale_seq = gt_obj_scale[bs] if gt_obj_scale is not None else None 
            # Handle bottom parts if they exist
            gt_obj_bottom_trans_seq = gt_obj_bottom_trans[bs] if gt_obj_bottom_trans is not None else None
            gt_obj_bottom_rot_seq = gt_obj_bottom_rot[bs] if gt_obj_bottom_rot is not None else None

            # Denormalization（关键步骤）
            gt_obj_rot_mat_seq = root_global_rot_start @ gt_obj_rot_mat_seq
            gt_obj_trans_seq = (root_global_rot_start @ gt_obj_trans_seq.unsqueeze(-1)).squeeze(-1) + root_global_pos_start

            # 获取真值物体
            gt_obj_verts_seq, obj_faces_np = load_object_geometry(
                obj_name, gt_obj_rot_mat_seq, gt_obj_trans_seq, gt_obj_scale_seq, device=device
            )

        # === 5. 模型预测（严格按照vis.py逻辑）===
        print("Running inference with Original Model...")
        if original_model is not None:
            model_input = {
                "human_imu": human_imu,
                "motion": gt_motion,             
                "root_pos": gt_root_pos,           
            }
            
            has_object_data_for_model = obj_imu is not None
            if has_object_data_for_model:
                model_input["obj_imu"] = obj_imu 
                model_input["obj_rot"] = gt_obj_rot_6d 
                model_input["obj_trans"] = gt_obj_trans 

            try:
                pred_dict = original_model(model_input)
                pred_motion = pred_dict.get("motion") 
                pred_obj_rot = pred_dict.get("obj_rot") 
                pred_obj_trans = pred_dict.get("pred_obj_trans_from_contact") 
                pred_root_pos = pred_dict.get("root_pos") 

                if pred_motion is not None:
                    pred_motion_seq = pred_motion[bs] 

                if pred_root_pos is not None:
                    pred_root_pos_seq = pred_root_pos[bs] 

                if pred_obj_rot is not None:
                    pred_obj_rot_6d_seq = pred_obj_rot[bs] 

                if pred_obj_trans is not None:
                    pred_obj_trans_seq = pred_obj_trans[bs] 

            except Exception as e:
                print(f"模型推理失败: {e}")
                import traceback
                traceback.print_exc()

        # === 6. 获取预测 SMPL（严格按照vis.py逻辑）===
        verts_pred_seq = None
        if pred_motion_seq is not None:
            pred_rot_matrices = transforms.rotation_6d_to_matrix(pred_motion_seq.reshape(T, 22, 6)) 
            pred_root_orient_mat_norm = pred_rot_matrices[:, 0]                         
            pred_pose_body_mat = pred_rot_matrices[:, 1:].reshape(T * 21, 3, 3)    
            pred_pose_body_axis = transforms.matrix_to_axis_angle(pred_pose_body_mat).reshape(T, -1) 

            # Denormalization（关键步骤）
            pred_root_orient_mat = root_global_rot_start @ pred_root_orient_mat_norm
            pred_root_orient_axis = transforms.matrix_to_axis_angle(pred_root_orient_mat).reshape(T, 3)
            pred_root_pos_seq = (root_global_rot_start @ pred_root_pos_seq.unsqueeze(-1)).squeeze(-1) + root_global_pos_start

            pred_smplh_input = {
                'root_orient': pred_root_orient_axis,
                'pose_body': pred_pose_body_axis,
                'trans': pred_root_pos_seq
            }
            body_pose_pred = smpl_model(**pred_smplh_input)
            verts_pred_seq = body_pose_pred.v 

            # 获取预测物体（使用真值旋转 + 预测平移）
            if show_objects and has_object_gt and pred_obj_trans_seq is not None:
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

        # === 7. 添加到 aitviewer 场景（严格按照vis.py逻辑）===
        global R_yup # 使用全局定义的 Y-up 旋转

        # 添加真值人体 (绿色, 不偏移)
        if verts_gt_seq is not None:
            verts_gt_yup = torch.matmul(verts_gt_seq, R_yup.T.to(device))
            gt_human_mesh = Meshes(
                verts_gt_yup.cpu().numpy(), faces_gt_np,
                name="GT-Human", color=(0.1, 0.8, 0.3, 0.8), gui_affine=False, is_selectable=False
            )
            viewer.scene.add(gt_human_mesh)

        # 添加预测人体 (红色, 使用pred_offset偏移)
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

        # 添加预测物体 (红色, 使用pred_offset偏移)
        if pred_obj_verts_seq is not None and obj_faces_np is not None:
            pred_obj_verts_shifted = pred_obj_verts_seq + pred_offset # 使用定义的偏移
            pred_obj_verts_yup = torch.matmul(pred_obj_verts_shifted, R_yup.T.to(device))
            pred_obj_mesh = Meshes(
                pred_obj_verts_yup.cpu().numpy(), obj_faces_np,
                name=f"Pred-{obj_name}", color=(0.9, 0.2, 0.2, 0.8), gui_affine=False, is_selectable=False
            )
            viewer.scene.add(pred_obj_mesh)

        # === SimpleObjT模型物体预测（简化版本，专注于位移）===
        if simple_objt_model is not None and show_objects and obj_name and gt_obj_rot_6d is not None:
            try:
                simple_objt_input = {
                    "obj_imu": obj_imu,
                    "obj_trans": gt_obj_trans,
                }
                # print(obj_imu.detach().cpu().numpy())
                simple_objt_pred = simple_objt_model(simple_objt_input)
                
                if "pred_obj_trans" in simple_objt_pred:
                    pred_obj_trans_simple = simple_objt_pred["pred_obj_trans"][bs]
                    
                    # 反归一化预测的物体位移
                    pred_obj_trans_simple_denorm = (root_global_rot_start @ pred_obj_trans_simple.unsqueeze(-1)).squeeze(-1) + root_global_pos_start
                    # print(pred_obj_trans_simple_denorm.detach().cpu().numpy())

                    # 偏移显示（左移）
                    simple_pred_offset = torch.tensor([-0.0, 0.0, 0.0], device=device)
                    
                    # 加载SimpleObjT物体几何体
                    simple_obj_verts, _ = load_object_geometry(
                        obj_name, 
                        gt_obj_rot_mat_seq, # 使用真值旋转
                        pred_obj_trans_simple_denorm, # 使用SimpleObjT预测平移
                        gt_obj_scale_seq, 
                        device=device
                    )
                    
                    if simple_obj_verts is not None and simple_obj_verts.numel() > 0:
                        simple_obj_verts_shifted = simple_obj_verts + simple_pred_offset
                        simple_obj_verts_yup = torch.matmul(simple_obj_verts_shifted, R_yup.T.to(device))
                        simple_obj_mesh = Meshes(
                            simple_obj_verts_yup.cpu().numpy(), obj_faces_np,
                            name="SimpleObjT-Object", color=(0.0, 0.0, 1.0, 0.8), gui_affine=False, is_selectable=False
                        )
                        viewer.scene.add(simple_obj_mesh)
                        print(f"SimpleObjT predicted object translation at first frame: {pred_obj_trans_simple[0].cpu().numpy()}")
                        
            except Exception as e:
                print(f"Error during SimpleObjT model inference: {e}")
                import traceback
                traceback.print_exc()

# === 自定义 Viewer 类 ===

class ModularComparativeViewer(Viewer):
    def __init__(self, data_list, original_model, simple_objt_model, smpl_model, config, device, obj_geo_root, show_objects=True, **kwargs):
        super().__init__(**kwargs)
        self.data_list = data_list
        self.current_index = 0
        self.original_model = original_model
        self.simple_objt_model = simple_objt_model
        self.smpl_model = smpl_model
        self.config = config
        self.device = device
        self.show_objects = show_objects
        self.obj_geo_root = obj_geo_root

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
                visualize_comparative_results(
                    self, batch, self.original_model, self.simple_objt_model, 
                    self.smpl_model, self.device, self.obj_geo_root, self.show_objects
                )
                self.title = f"Sequence Index: {self.current_index}/{len(self.data_list)-1} | 绿色:真值 红色:原模型 红色左移:SimpleObjT (q/e:±1, Ctrl+q/e:±10, Alt+q/e:±50)"
            except Exception as e:
                 print(f"Error visualizing sequence {self.current_index}: {e}")
                 import traceback
                 traceback.print_exc()
                 self.title = f"Error visualizing index: {self.current_index}"
                 
                 # 尝试仅显示真值数据
                 try:
                     print("Attempting to show ground truth only...")
                     # 创建安全批次
                     safe_batch = SafeDataBatch(batch, self.device)
                     # 只显示人体真值
                     gt_motion = safe_batch["motion"][0]
                     gt_root_pos = safe_batch["root_pos"][0]
                     print(f"GT data shapes: motion={gt_motion.shape}, root_pos={gt_root_pos.shape}")
                 except Exception as e2:
                     print(f"Failed to show ground truth: {e2}")
        else:
            print("Index out of bounds.")

    def key_event(self, key, action, modifiers):
        # 调用父类方法
        super().key_event(key, action, modifiers)

        # 检查ImGui是否需要键盘输入
        io = imgui.get_io()
        if self.render_gui and (io.want_capture_keyboard or io.want_text_input):
             return

        # 检查按键事件
        is_press = action == self.wnd.keys.ACTION_PRESS

        if is_press:
            # 检查修饰键
            ctrl_pressed = modifiers.ctrl
            alt_pressed = modifiers.alt
            
            if key == self.wnd.keys.Q:
                if alt_pressed:
                    step = 50
                    new_index = max(0, self.current_index - step)
                    if new_index != self.current_index:
                        self.current_index = new_index
                        self.visualize_current_sequence()
                        print(f"Jumped backward by {step} to index: {self.current_index}")
                elif ctrl_pressed:
                    step = 10
                    new_index = max(0, self.current_index - step)
                    if new_index != self.current_index:
                        self.current_index = new_index
                        self.visualize_current_sequence()
                        print(f"Stepped backward by {step} to index: {self.current_index}")
                else:
                    if self.current_index > 0:
                        self.current_index -= 1
                        self.visualize_current_sequence()
                        print(f"Stepped backward to index: {self.current_index}")

            elif key == self.wnd.keys.E:
                if alt_pressed:
                    step = 50
                    new_index = min(len(self.data_list) - 1, self.current_index + step)
                    if new_index != self.current_index:
                        self.current_index = new_index
                        self.visualize_current_sequence()
                        print(f"Jumped forward by {step} to index: {self.current_index}")
                elif ctrl_pressed:
                    step = 10
                    new_index = min(len(self.data_list) - 1, self.current_index + step)
                    if new_index != self.current_index:
                        self.current_index = new_index
                        self.visualize_current_sequence()
                        print(f"Stepped forward by {step} to index: {self.current_index}")
                else:
                    if self.current_index < len(self.data_list) - 1:
                        self.current_index += 1
                        self.visualize_current_sequence()
                        print(f"Stepped forward to index: {self.current_index}")

def main():
    parser = argparse.ArgumentParser(description='Modular Comparative EgoMotion Visualization Tool (with SimpleObjT Translation Module)')
    parser.add_argument('--config', type=str, default='configs/TransPose_train.yaml', help='Path to the main configuration file.')
    parser.add_argument('--simple_objt_module_path', type=str, default=None, help='Path to the SimpleObjT object_trans module for object translation estimation (optional).')
    parser.add_argument('--smpl_model_path', type=str, default=None, help='Path to the SMPLH model file.')
    parser.add_argument('--test_data_dir', type=str, default=None, help='Path to the test dataset directory.')
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

    # 覆盖配置
    if args.smpl_model_path: config.bm_path = args.smpl_model_path
    if args.test_data_dir: config.test.data_path = args.test_data_dir
    if args.num_workers is not None: config.num_workers = args.num_workers
    
    # 确保测试配置存在
    if 'test' not in config: config.test = config.train.copy()
    config.test.batch_size = args.batch_size

    # 加载SMPL模型
    smpl_model_path = config.get('bm_path', 'body_models/smplh/neutral/model.npz')
    smpl_model = load_smpl_model(smpl_model_path, device)

    # 创建模块化的原始模型
    print("Creating modular original model...")
    original_model = create_modular_original_model(config, device)

    # 创建模块化的SimpleObjT模型
    print("Creating modular SimpleObjT model...")
    simple_objt_model = create_modular_simple_objt_model(config, device, args.simple_objt_module_path)

    # 加载测试数据集
    test_data_dir = config.test.data_path
    if not test_data_dir or not os.path.exists(test_data_dir):
        print(f"Error: Test dataset path not found or invalid: {test_data_dir}")
        return
    print(f"Loading test dataset from: {test_data_dir}")

    test_window_size = config.test.get('window', config.train.get('window', 60))

    test_dataset = IMUDataset(
        data_dir=test_data_dir,
        window_size=test_window_size,
        normalize=config.test.get('normalize', True),
        debug=config.get('debug', False)
    )

    if len(test_dataset) == 0:
         print("Error: Test dataset is empty.")
         return

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.get('num_workers', 0),
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

    # 初始化并运行查看器
    print("Initializing Modular Comparative Viewer...")
    print("显示内容（严格按照vis.py逻辑）：")
    print("- 绿色: 真值（人体+物体）")
    print("- 红色: 原模型预测（人体+物体，无偏移）") 
    print("- 红色左移: SimpleObjT物体位移预测（仅物体位移，左移2米）")
    print("")
    print("注意：")
    print("- 已修复反归一化逻辑，确保与vis.py完全一致")
    print("- R_yup变换在最后阶段应用，符合vis.py逻辑")
    print("- 如果模型推理失败，将只显示真值数据")
    print("- 如果物体几何文件不存在，将跳过物体可视化")
    print("- 使用命名格式: {obj_name}_cleaned_simplified.obj")
    
    viewer_instance = ModularComparativeViewer(
        data_list=data_list,
        original_model=original_model,
        simple_objt_model=simple_objt_model,
        smpl_model=smpl_model,
        config=config,
        device=device,
        obj_geo_root=args.obj_geo_root,
        show_objects=(not args.no_objects),
        window_size=(1920, 1080)
    )
    
    print("Viewer Initialized. Navigation controls:")
    print("  q/e: 前进/后退 1个序列")
    print("  Ctrl+q/e: 前进/后退 10个序列")
    print("  Alt+q/e: 前进/后退 50个序列")
    
    viewer_instance.run()


if __name__ == "__main__":
    main() 