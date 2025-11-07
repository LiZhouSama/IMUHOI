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
from configs.global_config_IMUHOI import FRAME_RATE

from human_body_prior.body_model.body_model import BodyModel
from easydict import EasyDict as edict

from torch.utils.data import DataLoader
from dataloader.dataloader_IMUHOI import IMUDataset
from process.preprocess import load_object_geometry

from configs.global_config_IMUHOI import (
    FRAME_RATE,
    _SENSOR_NAMES,
    _SENSOR_VEL_NAMES,
    _REDUCED_POSE_NAMES,
    _REDUCED_INDICES,
    _IGNORED_INDICES,
    _SENSOR_ROT_INDICES,
    _SENSOR_POS_INDICES,
    _VEL_SELECTION_INDICES,
)

# 导入模型相关 - 根据需要选择正确的模型加载方式
# from models.DiT_model import MotionDiffusion # 如果要用 DiT
from models.IMUHOI_stage_net_noTrans import TransPoseNet
from models.do_train_IMUHOI_noTrans import build_model_input_dict

import imgui
from aitviewer.renderables.spheres import Spheres

R_yup = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)

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


def visualize_batch_data(viewer, batch, model, smpl_model, device, obj_geo_root, show_objects=True, vis_gt_only=False, show_foot_contact=False, show_hands_contact=False, use_fk=False, compare_3=False, pred_offset_np=None):
    """在 aitviewer 场景中可视化单个批次的数据 (真值和预测)
    
    Args:
        pred_offset_np: numpy array [3] or None, 预测mesh的偏移向量
    """
    try:
        nodes_to_remove = [
            node for node in viewer.scene.collect_nodes()
            if hasattr(node, "name")
            and node.name is not None
            and (
                node.name.startswith("GT-")
                or node.name.startswith("Pred-")
                or node.name.startswith("FK-")
                or node.name == "GT-LHandContact"
                or node.name == "GT-RHandContact"
                or node.name == "ObjContactIndicator"
                or node.name == "Pred-LHandContact"
                or node.name == "Pred-RHandContact"
                or node.name == "Pred-ObjContactIndicator"
                or node.name == "GT-LFootContact"
                or node.name == "GT-RFootContact"
                or node.name == "Pred-LFootContact"
                or node.name == "Pred-RFootContact"
            )
        ]
        for node in nodes_to_remove:
            try:
                viewer.scene.remove(node)
            except Exception as exc:
                print(f"Error removing node '{node.name}': {exc}")
    except AttributeError as exc:
        print(f"Error accessing scene nodes: {exc}")
    except Exception as exc:
        print(f"Error during scene clearing: {exc}")

    with torch.no_grad():
        config = getattr(viewer, "config", None)
        if config is None:
            config = edict({})
        stage_info = getattr(viewer, "stage_info", {"use_object_data": True})

        batch_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_device[key] = value.to(device)
            else:
                batch_device[key] = value

        def _get_tensor(key):
            value = batch_device.get(key)
            if isinstance(value, torch.Tensor):
                return value
            return None

        human_imu_batch = _get_tensor("human_imu")
        if human_imu_batch is None:
            print("Error: Batch missing 'human_imu'.")
            return
        if human_imu_batch.dim() == 3:
            human_imu_batch = human_imu_batch.unsqueeze(0)
        batch_size, T = human_imu_batch.shape[:2]
        if batch_size == 0:
            print("Warning: empty batch encountered during visualization.")
            return
        bs = 0
        human_imu_seq = human_imu_batch[bs]

        pose_batch = _get_tensor("pose")
        trans_batch = _get_tensor("trans")
        position_global_batch = _get_tensor("position_global")
        rotation_global_batch = _get_tensor("rotation_global")
        obj_trans_batch = _get_tensor("obj_trans")
        obj_rot_batch = _get_tensor("obj_rot")
        obj_scale_batch = _get_tensor("obj_scale")
        obj_vel_batch = _get_tensor("obj_vel")
        obj_imu_batch = _get_tensor("obj_imu")
        lfoot_contact_batch = _get_tensor("lfoot_contact")
        rfoot_contact_batch = _get_tensor("rfoot_contact")
        lhand_contact_batch = _get_tensor("lhand_contact")
        rhand_contact_batch = _get_tensor("rhand_contact")
        obj_contact_batch = _get_tensor("obj_contact")

        def _get_bool_sequence(tensor):
            if tensor is None:
                return None
            seq = tensor[bs]
            if seq.dtype != torch.bool:
                seq = seq > 0.5
            return seq

        lfoot_contact_seq = lfoot_contact_batch[bs] if lfoot_contact_batch is not None else None
        rfoot_contact_seq = rfoot_contact_batch[bs] if rfoot_contact_batch is not None else None
        lhand_contact_seq = _get_bool_sequence(lhand_contact_batch)
        rhand_contact_seq = _get_bool_sequence(rhand_contact_batch)
        obj_contact_seq = _get_bool_sequence(obj_contact_batch)

        has_object_value = batch_device.get("has_object")

        def _extract_bool_flag(value):
            if value is None:
                return False
            if isinstance(value, torch.Tensor):
                if value.dim() == 0:
                    return bool(value.item())
                return bool(value[bs].item())
            if isinstance(value, (list, tuple)):
                return bool(value[bs])
            return bool(value)

        has_object_bool = _extract_bool_flag(has_object_value)
        if obj_trans_batch is not None and has_object_value is None:
            has_object_bool = True

        obj_name = "object"
        if "obj_name" in batch_device:
            obj_name_raw = batch_device["obj_name"]
            if isinstance(obj_name_raw, (list, tuple)):
                obj_name_candidate = obj_name_raw[bs]
            else:
                obj_name_candidate = obj_name_raw
            if isinstance(obj_name_candidate, torch.Tensor):
                obj_name_candidate = obj_name_candidate.item()
            if isinstance(obj_name_candidate, bytes):
                obj_name_candidate = obj_name_candidate.decode("utf-8")
            if obj_name_candidate:
                obj_name = str(obj_name_candidate)

        faces_attr = getattr(smpl_model, "f", None)
        if isinstance(faces_attr, torch.Tensor):
            faces_gt_np = faces_attr.detach().cpu().numpy()
        elif faces_attr is not None:
            faces_gt_np = faces_attr
        else:
            faces_gt_np = smpl_model.faces_tensor.detach().cpu().numpy()

        verts_gt_seq = None
        Jtr_gt_seq = None
        if pose_batch is not None and trans_batch is not None:
            try:
                gt_pose_seq = pose_batch[bs]
                gt_trans_seq = trans_batch[bs]
                if gt_pose_seq.dim() == 2 and gt_trans_seq.dim() == 2:
                    root_orient = gt_pose_seq[:, :3]
                    pose_body = gt_pose_seq[:, 3:]
                    if pose_body.shape[1] < 63:
                        pose_body_padded = torch.zeros(T, 63, device=device, dtype=pose_body.dtype)
                        pose_body_padded[:, :pose_body.shape[1]] = pose_body
                        pose_body = pose_body_padded
                    elif pose_body.shape[1] > 63:
                        pose_body = pose_body[:, :63]
                    body_pose_gt = smpl_model(
                        pose_body=pose_body,
                        root_orient=root_orient,
                        trans=gt_trans_seq
                    )
                    verts_gt_seq = body_pose_gt.v
                    Jtr_gt_seq = body_pose_gt.Jtr
            except Exception as exc:
                print(f"Failed to build GT SMPL mesh: {exc}")

        pred_dict = None
        model_input = None
        fk_bone_info_seq = None
        compute_fk_flag = bool(use_fk or compare_3)
        if not vis_gt_only:
            try:
                model_input = build_model_input_dict(batch_device, stage_info, config, device, add_noise=False)
                use_object_data = stage_info.get("use_object_data", True)
                pred_dict = model(
                    model_input,
                    use_object_data=use_object_data,
                    compute_fk=compute_fk_flag
                )
            except Exception as exc:
                print(f"Model inference failed: {exc}")
                pred_dict = None

        verts_pred_seq = None
        Jtr_pred_seq = None
        pred_root_trans_seq = gt_trans_seq
        pred_obj_trans_seq = None
        pred_obj_trans_fk_seq = None
        pred_obj_trans_imu_seq = None
        pred_obj_vel_seq = None
        pred_hand_contact_prob_seq = None
        pred_lhand_contact_labels_seq = None
        pred_rhand_contact_labels_seq = None
        pred_obj_contact_labels_seq = None
        pred_lfoot_contact_labels_seq = None
        pred_rfoot_contact_labels_seq = None
        gating_weights_seq = None
        obj_vel_input_np = None
        pred_lhand_bone_length_np = None
        pred_rhand_bone_length_np = None
        pred_lhand_direction_np = None
        pred_rhand_direction_np = None

        if pred_dict is not None:
            if "root_trans_pred" in pred_dict:
                pred_root_trans_seq = pred_dict["root_trans_pred"][bs].to(device)
            p_pred_seq = pred_dict.get("p_pred")
            if p_pred_seq is not None:
                p_pred_seq = p_pred_seq[bs].to(device)
            if "pred_obj_trans" in pred_dict:
                pred_obj_trans_seq = pred_dict["pred_obj_trans"][bs].to(device)
            if compute_fk_flag and "pred_obj_trans_fk" in pred_dict:
                pred_obj_trans_fk_seq = pred_dict["pred_obj_trans_fk"][bs].to(device)
            if "pred_obj_vel" in pred_dict:
                pred_obj_vel_seq = pred_dict["pred_obj_vel"][bs].to(device)
            if "pred_hand_contact_prob" in pred_dict:
                pred_hand_contact_prob_seq = pred_dict["pred_hand_contact_prob"][bs].to(device)
                pred_lhand_contact_labels_seq = pred_hand_contact_prob_seq[:, 0] > 0.5
                pred_rhand_contact_labels_seq = pred_hand_contact_prob_seq[:, 1] > 0.5
                pred_obj_contact_labels_seq = pred_hand_contact_prob_seq[:, 2] > 0.5
            if "contact_pred" in pred_dict:
                contact_probs = torch.sigmoid(pred_dict["contact_pred"][bs].to(device))
                pred_lfoot_contact_labels_seq = contact_probs[:, 0] > 0.5
                pred_rfoot_contact_labels_seq = contact_probs[:, 1] > 0.5
            if "gating_weights" in pred_dict:
                gating_weights_seq = pred_dict["gating_weights"][bs].detach().cpu().numpy()
            if "obj_vel_input" in pred_dict:
                obj_vel_input_np = pred_dict["obj_vel_input"][bs].detach().cpu().numpy()
            if "pred_lhand_lb" in pred_dict:
                pred_lhand_bone_length_np = pred_dict["pred_lhand_lb"][bs].detach().cpu().numpy()
            if "pred_rhand_lb" in pred_dict:
                pred_rhand_bone_length_np = pred_dict["pred_rhand_lb"][bs].detach().cpu().numpy()
            if "pred_lhand_obj_direction" in pred_dict:
                pred_lhand_direction_np = pred_dict["pred_lhand_obj_direction"][bs].detach().cpu().numpy()
            if "pred_rhand_obj_direction" in pred_dict:
                pred_rhand_direction_np = pred_dict["pred_rhand_obj_direction"][bs].detach().cpu().numpy()
            if compute_fk_flag:
                fk_bone_info_seq = {}
                for key in ["fk_lhand_bone_length", "fk_rhand_bone_length", "fk_lhand_direction", "fk_rhand_direction"]:
                    if key in pred_dict:
                        fk_bone_info_seq[key] = pred_dict[key][bs].detach().cpu()
                if not fk_bone_info_seq:
                    fk_bone_info_seq = None

            if (
                p_pred_seq is not None
                and pred_root_trans_seq is not None
                and getattr(model, "human_pose_module", None) is not None
            ):
                try:
                    reduced_pose = p_pred_seq.view(T, len(_REDUCED_POSE_NAMES), 6)
                    orientation_6d = human_imu_batch[bs, :, :, -6:]
                    orientation_mat = transforms.rotation_6d_to_matrix(
                        orientation_6d.reshape(-1, 6)
                    ).reshape(T, human_imu_batch.shape[2], 3, 3)
                    orientation_subset = orientation_mat[:, :len(_SENSOR_ROT_INDICES), :, :]
                    human_module = model.human_pose_module
                    full_glb = human_module._reduced_glb_6d_to_full_glb_mat(
                        reduced_pose,
                        orientation_subset.reshape(T, len(_SENSOR_ROT_INDICES), 3, 3)
                    )
                    parents = human_module.smpl_parents.tolist()
                    local_rot = human_module._global2local(full_glb, parents)
                    pose_axis = transforms.matrix_to_axis_angle(
                        local_rot.reshape(T * full_glb.shape[1], 3, 3)
                    ).reshape(T, full_glb.shape[1], 3)
                    root_axis = pose_axis[:, 0, :]
                    pose_body_axis = pose_axis[:, 1:22, :].reshape(T, -1)
                    smpl_pred = smpl_model(
                        pose_body=pose_body_axis,
                        root_orient=root_axis,
                        trans=pred_root_trans_seq
                    )
                    verts_pred_seq = smpl_pred.v
                    Jtr_pred_seq = smpl_pred.Jtr
                except Exception as exc:
                    print(f"Predicted SMPL reconstruction failed: {exc}")

        # 使用传入的偏移参数或默认值
        if pred_offset_np is not None:
            pred_offset = torch.tensor(pred_offset_np, device=device, dtype=torch.float32)
        else:
            pred_offset = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)

        if compare_3 and pred_obj_vel_seq is not None and model_input is not None:
            try:
                obj_trans_init = model_input["obj_trans_init"][bs].to(device)
                delta = pred_obj_vel_seq * (1.0 / FRAME_RATE)
                disp = torch.zeros_like(delta)
                if T > 1:
                    cumulative = torch.cumsum(delta, dim=0)
                    disp[1:] = cumulative[:-1]
                pred_obj_trans_imu_seq = obj_trans_init.unsqueeze(0) + disp
            except Exception as exc:
                print(f"Failed to integrate object velocity: {exc}")
                pred_obj_trans_imu_seq = None

        gt_obj_verts_seq = None
        pred_obj_verts_seq_mesh = None
        pred_obj_verts_seq_fk = None
        pred_obj_verts_seq_imu = None
        obj_faces_np = None
        gt_obj_rot_mat_seq = None
        gt_obj_trans_seq = None
        gt_obj_scale_seq = None

        if (
            show_objects
            and has_object_bool
            and obj_trans_batch is not None
            and obj_rot_batch is not None
        ):
            gt_obj_trans_seq = obj_trans_batch[bs]
            gt_obj_rot_6d_seq = obj_rot_batch[bs]
            gt_obj_rot_mat_seq = transforms.rotation_6d_to_matrix(gt_obj_rot_6d_seq)
            if obj_scale_batch is not None:
                gt_obj_scale_seq = obj_scale_batch[bs]
            try:
                gt_obj_verts_seq, obj_faces_np = load_object_geometry(
                    obj_name,
                    gt_obj_rot_mat_seq,
                    gt_obj_trans_seq,
                    gt_obj_scale_seq,
                    device=device,
                    obj_geo_root=obj_geo_root
                )
            except Exception as exc:
                print(f"Failed to load GT object geometry: {exc}")
                gt_obj_verts_seq = None
            if not vis_gt_only and pred_obj_trans_seq is not None and obj_faces_np is not None:
                try:
                    pred_obj_verts_seq_mesh, _ = load_object_geometry(
                        obj_name,
                        gt_obj_rot_mat_seq,
                        pred_obj_trans_seq,
                        gt_obj_scale_seq,
                        device=device,
                        obj_geo_root=obj_geo_root
                    )
                except Exception as exc:
                    print(f"Failed to load predicted object geometry: {exc}")
            if not vis_gt_only and compare_3 and pred_obj_trans_fk_seq is not None and obj_faces_np is not None:
                try:
                    pred_obj_verts_seq_fk, _ = load_object_geometry(
                        obj_name,
                        gt_obj_rot_mat_seq,
                        pred_obj_trans_fk_seq,
                        gt_obj_scale_seq,
                        device=device,
                        obj_geo_root=obj_geo_root
                    )
                except Exception as exc:
                    print(f"Failed to load FK object geometry: {exc}")
            if not vis_gt_only and compare_3 and pred_obj_trans_imu_seq is not None and obj_faces_np is not None:
                try:
                    pred_obj_verts_seq_imu, _ = load_object_geometry(
                        obj_name,
                        gt_obj_rot_mat_seq,
                        pred_obj_trans_imu_seq,
                        gt_obj_scale_seq,
                        device=device,
                        obj_geo_root=obj_geo_root
                    )
                except Exception as exc:
                    print(f"Failed to load IMU object geometry: {exc}")

        if (
            not vis_gt_only
            and show_objects
            and has_object_bool
            and gt_obj_trans_seq is not None
        ):
            try:
                def _mean_err_mm(pred_seq):
                    if pred_seq is None or pred_seq.shape != gt_obj_trans_seq.shape:
                        return None
                    return (pred_seq - gt_obj_trans_seq).norm(dim=-1).mean().item() * 1000.0

                fusion_err_mm = _mean_err_mm(pred_obj_trans_seq)
                fk_err_mm = _mean_err_mm(pred_obj_trans_fk_seq) if compare_3 else None
                imu_err_mm = _mean_err_mm(pred_obj_trans_imu_seq) if compare_3 else None

                def _fmt(val):
                    return f"{val:.2f}" if val is not None else "N/A"

                print(
                    f"Object translation errors (mm) - {obj_name}: "
                    f"Fusion={_fmt(fusion_err_mm)}, FK={_fmt(fk_err_mm)}, IMU={_fmt(imu_err_mm)}"
                )
            except Exception as exc:
                print(f"Warning: Failed to compute object translation errors: {exc}")

        if verts_gt_seq is not None:
            verts_gt_yup = torch.matmul(verts_gt_seq, R_yup.T.to(device))
            gt_human_mesh = Meshes(
                verts_gt_yup.detach().cpu().numpy(),
                faces_gt_np,
                name="GT-Human",
                color=(0.1, 0.8, 0.3, 0.8),
                gui_affine=False,
                is_selectable=False
            )
            viewer.scene.add(gt_human_mesh)

        if verts_pred_seq is not None and not vis_gt_only:
            verts_pred_shifted = verts_pred_seq + pred_offset
            verts_pred_yup = torch.matmul(verts_pred_shifted, R_yup.T.to(device))
            pred_human_mesh = Meshes(
                verts_pred_yup.detach().cpu().numpy(),
                faces_gt_np,
                name="Pred-Human",
                color=(0.9, 0.2, 0.2, 0.8),
                gui_affine=False,
                is_selectable=False
            )
            viewer.scene.add(pred_human_mesh)

        if gt_obj_verts_seq is not None and obj_faces_np is not None:
            gt_obj_verts_yup = torch.matmul(gt_obj_verts_seq, R_yup.T.to(device))
            gt_obj_mesh = Meshes(
                gt_obj_verts_yup.detach().cpu().numpy(),
                obj_faces_np,
                name=f"GT-{obj_name}",
                color=(0.1, 0.8, 0.3, 0.8),
                gui_affine=False,
                is_selectable=False
            )
            viewer.scene.add(gt_obj_mesh)

        if pred_obj_verts_seq_mesh is not None and obj_faces_np is not None and not vis_gt_only:
            pred_obj_verts_shifted = pred_obj_verts_seq_mesh + pred_offset
            pred_obj_verts_yup = torch.matmul(pred_obj_verts_shifted, R_yup.T.to(device))
            pred_obj_mesh = Meshes(
                pred_obj_verts_yup.detach().cpu().numpy(),
                obj_faces_np,
                name=f"Pred-{obj_name}",
                color=(0.9, 0.2, 0.2, 0.8),
                gui_affine=False,
                is_selectable=False
            )
            viewer.scene.add(pred_obj_mesh)

        if compare_3 and pred_obj_verts_seq_imu is not None and obj_faces_np is not None and not vis_gt_only:
            obj_verts_imu_shifted = pred_obj_verts_seq_imu + pred_offset
            obj_verts_imu_yup = torch.matmul(obj_verts_imu_shifted, R_yup.T.to(device))
            imu_mesh = Meshes(
                obj_verts_imu_yup.detach().cpu().numpy(),
                obj_faces_np,
                name=f"Pred-IMU-{obj_name}",
                color=(0.2, 0.2, 0.9, 0.8),
                gui_affine=False,
                is_selectable=False
            )
            viewer.scene.add(imu_mesh)

        if compare_3 and pred_obj_verts_seq_fk is not None and obj_faces_np is not None and not vis_gt_only:
            obj_verts_fk_shifted = pred_obj_verts_seq_fk + pred_offset
            obj_verts_fk_yup = torch.matmul(obj_verts_fk_shifted, R_yup.T.to(device))
            fk_mesh = Meshes(
                obj_verts_fk_yup.detach().cpu().numpy(),
                obj_faces_np,
                name=f"FK-{obj_name}",
                color=(1.0, 1.0, 0.0, 0.8),
                gui_affine=False,
                is_selectable=False
            )
            viewer.scene.add(fk_mesh)

        lhand_idx = 20
        rhand_idx = 21
        lfoot_idx = 7
        rfoot_idx = 8

        # --- 可视化手部接触（红色和蓝色小球）---
        if Jtr_gt_seq is not None and show_hands_contact:
            contact_radius = 0.03  # 手部接触半径
            
            # --- 可视化 GT 左手接触 (红色) ---
            if lhand_contact_seq is not None:
                gt_lhand_contact_points_list = []
                for t in range(T):
                    if lhand_contact_seq[t]:
                        gt_lhand_contact_points_list.append(Jtr_gt_seq[t, lhand_idx])
                
                if gt_lhand_contact_points_list:
                    gt_lhand_contact_points = torch.stack(gt_lhand_contact_points_list, dim=0)
                    if gt_lhand_contact_points.numel() > 0:
                        gt_lhand_points_yup = torch.matmul(gt_lhand_contact_points, R_yup.T.to(device))
                        gt_lhand_spheres = Spheres(
                            positions=gt_lhand_points_yup.detach().cpu().numpy(),
                            radius=contact_radius,
                            name="GT-LHandContact",
                            color=(1.0, 0.0, 0.0, 0.8),  # 红色
                            gui_affine=False,
                            is_selectable=False
                        )
                        viewer.scene.add(gt_lhand_spheres)
            
            # --- 可视化 GT 右手接触 (蓝色) ---
            if rhand_contact_seq is not None:
                gt_rhand_contact_points_list = []
                for t in range(T):
                    if rhand_contact_seq[t]:
                        gt_rhand_contact_points_list.append(Jtr_gt_seq[t, rhand_idx])
                
                if gt_rhand_contact_points_list:
                    gt_rhand_contact_points = torch.stack(gt_rhand_contact_points_list, dim=0)
                    if gt_rhand_contact_points.numel() > 0:
                        gt_rhand_points_yup = torch.matmul(gt_rhand_contact_points, R_yup.T.to(device))
                        gt_rhand_spheres = Spheres(
                            positions=gt_rhand_points_yup.detach().cpu().numpy(),
                            radius=contact_radius,
                            name="GT-RHandContact",
                            color=(0.0, 0.0, 1.0, 0.8),  # 蓝色
                            gui_affine=False,
                            is_selectable=False
                        )
                        viewer.scene.add(gt_rhand_spheres)
            
            # --- 可视化物体移动指示 (黄色) ---
            if obj_contact_seq is not None and gt_obj_trans_seq is not None:
                obj_indicator_points_list = []
                for t in range(T):
                    if obj_contact_seq[t]:
                        obj_indicator_points_list.append(gt_obj_trans_seq[t])
                
                if obj_indicator_points_list:
                    contact_positions = torch.stack(obj_indicator_points_list, dim=0)
                    contact_positions_yup = torch.matmul(contact_positions, R_yup.T.to(device))
                    obj_contact_spheres = Spheres(
                        positions=contact_positions_yup.detach().cpu().numpy(),
                        radius=0.04,  # 稍大的半径
                        name="ObjContactIndicator",
                        color=(1.0, 1.0, 0.0, 0.8),  # 黄色
                        gui_affine=False,
                        is_selectable=False
                    )
                    viewer.scene.add(obj_contact_spheres)
        
        # --- 可视化预测的手部接触 (只在非仅真值模式下) ---
        if not vis_gt_only and Jtr_pred_seq is not None and show_hands_contact:
            contact_radius_pred = 0.03
            
            # --- 预测左手接触 (红色) ---
            if pred_lhand_contact_labels_seq is not None:
                pred_lhand_contact_points_list = []
                for t in range(T):
                    if pred_lhand_contact_labels_seq[t]:
                        point_on_pred_human = Jtr_pred_seq[t, lhand_idx]
                        pred_lhand_contact_points_list.append(point_on_pred_human + pred_offset)
                
                if pred_lhand_contact_points_list:
                    pred_lhand_contact_points = torch.stack(pred_lhand_contact_points_list, dim=0)
                    if pred_lhand_contact_points.numel() > 0:
                        pred_lhand_points_yup = torch.matmul(pred_lhand_contact_points, R_yup.T.to(device))
                        pred_lhand_spheres = Spheres(
                            positions=pred_lhand_points_yup.detach().cpu().numpy(),
                            radius=contact_radius_pred,
                            name="Pred-LHandContact",
                            color=(1.0, 0.0, 0.0, 0.8),  # 红色
                            gui_affine=False,
                            is_selectable=False
                        )
                        viewer.scene.add(pred_lhand_spheres)
            
            # --- 预测右手接触 (蓝色) ---
            if pred_rhand_contact_labels_seq is not None:
                pred_rhand_contact_points_list = []
                for t in range(T):
                    if pred_rhand_contact_labels_seq[t]:
                        point_on_pred_human = Jtr_pred_seq[t, rhand_idx]
                        pred_rhand_contact_points_list.append(point_on_pred_human + pred_offset)
                
                if pred_rhand_contact_points_list:
                    pred_rhand_contact_points = torch.stack(pred_rhand_contact_points_list, dim=0)
                    if pred_rhand_contact_points.numel() > 0:
                        pred_rhand_points_yup = torch.matmul(pred_rhand_contact_points, R_yup.T.to(device))
                        pred_rhand_spheres = Spheres(
                            positions=pred_rhand_points_yup.detach().cpu().numpy(),
                            radius=contact_radius_pred,
                            name="Pred-RHandContact",
                            color=(0.0, 0.0, 1.0, 0.8),  # 蓝色
                            gui_affine=False,
                            is_selectable=False
                        )
                        viewer.scene.add(pred_rhand_spheres)
            
            # --- 预测物体移动指示 (黄色) ---
            if pred_obj_contact_labels_seq is not None:
                # 优先使用预测的物体平移，如果没有则使用真值
                obj_trans_for_pred_contact = pred_obj_trans_seq if pred_obj_trans_seq is not None else gt_obj_trans_seq
                
                if obj_trans_for_pred_contact is not None:
                    pred_obj_indicator_points_list = []
                    for t in range(T):
                        if pred_obj_contact_labels_seq[t]:
                            pred_obj_indicator_points_list.append(obj_trans_for_pred_contact[t] + pred_offset)
                    
                    if pred_obj_indicator_points_list:
                        pred_obj_contact_positions = torch.stack(pred_obj_indicator_points_list, dim=0)
                        if pred_obj_contact_positions.numel() > 0:
                            pred_obj_contact_positions_yup = torch.matmul(pred_obj_contact_positions, R_yup.T.to(device))
                            pred_obj_contact_spheres = Spheres(
                                positions=pred_obj_contact_positions_yup.detach().cpu().numpy(),
                                radius=contact_radius_pred,
                                name="Pred-ObjContactIndicator",
                                color=(1.0, 1.0, 0.0, 0.8),  # 黄色
                                gui_affine=False,
                                is_selectable=False
                            )
                            viewer.scene.add(pred_obj_contact_spheres)

        if show_foot_contact and Jtr_gt_seq is not None:
            foot_contact_radius = 0.04
            if lfoot_contact_seq is not None:
                gt_l_points = [Jtr_gt_seq[t, lfoot_idx] for t in range(T) if lfoot_contact_seq[t] > 0.5]
                if gt_l_points:
                    gt_l_points_tensor = torch.stack(gt_l_points, dim=0)
                    gt_l_points_yup = torch.matmul(gt_l_points_tensor, R_yup.T.to(device))
                    gt_l_spheres = Spheres(
                        positions=gt_l_points_yup.detach().cpu().numpy(),
                        radius=foot_contact_radius,
                        name="GT-LFootContact",
                        color=(0.5, 0.0, 0.5, 0.8),
                        gui_affine=False,
                        is_selectable=False
                    )
                    viewer.scene.add(gt_l_spheres)
            if rfoot_contact_seq is not None:
                gt_r_points = [Jtr_gt_seq[t, rfoot_idx] for t in range(T) if rfoot_contact_seq[t] > 0.5]
                if gt_r_points:
                    gt_r_points_tensor = torch.stack(gt_r_points, dim=0)
                    gt_r_points_yup = torch.matmul(gt_r_points_tensor, R_yup.T.to(device))
                    gt_r_spheres = Spheres(
                        positions=gt_r_points_yup.detach().cpu().numpy(),
                        radius=foot_contact_radius,
                        name="GT-RFootContact",
                        color=(1.0, 0.5, 0.0, 0.8),
                        gui_affine=False,
                        is_selectable=False
                    )
                    viewer.scene.add(gt_r_spheres)
            if not vis_gt_only and Jtr_pred_seq is not None:
                pred_foot_contact_radius = 0.035
                if pred_lfoot_contact_labels_seq is not None:
                    pred_l_points = [Jtr_pred_seq[t, lfoot_idx] + pred_offset for t in range(T) if bool(pred_lfoot_contact_labels_seq[t])]
                    if pred_l_points:
                        pred_l_points_tensor = torch.stack(pred_l_points, dim=0)
                        pred_l_points_yup = torch.matmul(pred_l_points_tensor, R_yup.T.to(device))
                        pred_l_spheres = Spheres(
                            positions=pred_l_points_yup.detach().cpu().numpy(),
                            radius=pred_foot_contact_radius,
                            name="Pred-LFootContact",
                            color=(0.8, 0.3, 0.8, 0.8),
                            gui_affine=False,
                            is_selectable=False
                        )
                        viewer.scene.add(pred_l_spheres)
                if pred_rfoot_contact_labels_seq is not None:
                    pred_r_points = [Jtr_pred_seq[t, rfoot_idx] + pred_offset for t in range(T) if bool(pred_rfoot_contact_labels_seq[t])]
                    if pred_r_points:
                        pred_r_points_tensor = torch.stack(pred_r_points, dim=0)
                        pred_r_points_yup = torch.matmul(pred_r_points_tensor, R_yup.T.to(device))
                        pred_r_spheres = Spheres(
                            positions=pred_r_points_yup.detach().cpu().numpy(),
                            radius=pred_foot_contact_radius,
                            name="Pred-RFootContact",
                            color=(1.0, 0.7, 0.3, 0.8),
                            gui_affine=False,
                            is_selectable=False
                        )
                        viewer.scene.add(pred_r_spheres)

        viewer.virtual_bone_info = {'has_data': False}
        if (
            Jtr_gt_seq is not None
            and gt_obj_trans_seq is not None
            and gt_obj_rot_mat_seq is not None
        ):
            try:
                lhand_pos_seq = Jtr_gt_seq[:, lhand_idx, :]
                rhand_pos_seq = Jtr_gt_seq[:, rhand_idx, :]
                lhand_bone_length, lhand_direction_axis_angle = compute_virtual_bone_info(
                    lhand_pos_seq, gt_obj_trans_seq, gt_obj_rot_mat_seq
                )
                rhand_bone_length, rhand_direction_axis_angle = compute_virtual_bone_info(
                    rhand_pos_seq, gt_obj_trans_seq, gt_obj_rot_mat_seq
                )

                fk_lhand_length_np = None
                fk_rhand_length_np = None
                if fk_bone_info_seq is not None:
                    fk_l = fk_bone_info_seq.get("fk_lhand_bone_length")
                    fk_r = fk_bone_info_seq.get("fk_rhand_bone_length")
                    if isinstance(fk_l, torch.Tensor):
                        fk_lhand_length_np = fk_l.detach().numpy()
                    elif fk_l is not None:
                        fk_lhand_length_np = fk_l
                    if isinstance(fk_r, torch.Tensor):
                        fk_rhand_length_np = fk_r.detach().numpy()
                    elif fk_r is not None:
                        fk_rhand_length_np = fk_r

                viewer.virtual_bone_info = {
                    'has_data': True,
                    'gt_lhand_bone_length': lhand_bone_length.detach().cpu().numpy(),
                    'gt_rhand_bone_length': rhand_bone_length.detach().cpu().numpy(),
                    'gt_lhand_direction_axis_angle': lhand_direction_axis_angle.detach().cpu().numpy(),
                    'gt_rhand_direction_axis_angle': rhand_direction_axis_angle.detach().cpu().numpy(),
                    'pred_lhand_bone_length': pred_lhand_bone_length_np,
                    'pred_rhand_bone_length': pred_rhand_bone_length_np,
                    'pred_lhand_direction': pred_lhand_direction_np,
                    'pred_rhand_direction': pred_rhand_direction_np,
                    'fk_lhand_bone_length': fk_lhand_length_np,
                    'fk_rhand_bone_length': fk_rhand_length_np,
                    'using_fk_data': False,
                    'prediction_method': 'Fusion Scheme',
                    'gating_weights': gating_weights_seq,
                    'obj_vel_input': obj_vel_input_np,
                    'num_frames': T
                }
            except Exception as exc:
                print(f"Failed to compute virtual bone info: {exc}")


# === 自定义 Viewer 类 ===

class InteractiveViewer(Viewer):
    def __init__(self, data_list, model, smpl_model, config, device, obj_geo_root, show_objects=True, vis_gt_only=False, show_foot_contact=False, show_hands_contact=False, use_fk=False, compare_3=False, pred_offset=None, **kwargs):
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
        self.show_hands_contact = show_hands_contact
        self.obj_geo_root = obj_geo_root
        self.use_fk = use_fk
        self.compare_3 = compare_3
        self.pred_offset = pred_offset  # 预测mesh的偏移向量 [3]
        self.stage_info = {"use_object_data": True}
        
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
                visualize_batch_data(
                    self,
                    batch,
                    self.model,
                    self.smpl_model,
                    self.device,
                    self.obj_geo_root,
                    self.show_objects,
                    self.vis_gt_only,
                    self.show_foot_contact,
                    self.show_hands_contact,
                    self.use_fk,
                    self.compare_3,
                    self.pred_offset,
                )
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
    parser.add_argument('--config', type=str, default='configs/IMUHOI_train_noTrans.yaml', help='Path to the main configuration file (used for model, dataset params).')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the trained TransPose model checkpoint. Overrides config if provided.')
    parser.add_argument('--smpl_model_path', type=str, default=None, help='Path to the SMPLH model file. Overrides config if provided.')
    parser.add_argument('--test_data_dir', type=str, default="process/processed_data_BEHAVE/test", help='Path to the test dataset directory. Overrides config if provided.')
    parser.add_argument('--obj_geo_root', type=str, default='./datasets/BEHAVE/objects', help='Path to the object geometry root directory.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers.')
    parser.add_argument('--no_objects', action='store_true', help='Do not load or visualize objects.')
    parser.add_argument('--vis_gt_only', action='store_true', help='Only visualize ground truth, skip model inference and prediction visualization.')
    parser.add_argument('--show_foot_contact', action='store_true', help='Visualize foot-ground contact indicators.')
    parser.add_argument('--show_hands_contact', action='store_true', help='Visualize hand-object contact indicators.')
    parser.add_argument('--use_fk', action='store_true', help='Use on-demand FK for object translation and virtual bone info in visualization.')
    parser.add_argument('--compare_3', action='store_true', help='Enable comparison between fused, IMU-integrated, and FK object meshes.')
    parser.add_argument('--limit_sequences', type=int, default=None, help='Limit the number of sequences to load for visualization.')
    parser.add_argument('--pred_offset', type=float, nargs=3, default=[2.0, 0.0, 0.0], metavar=('X', 'Y', 'Z'), help='Offset vector [X Y Z] for predicted meshes (default: 0.0 0.0 0.0)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Override config with command line args
    if args.model_path: config.model_path = args.model_path
    if args.smpl_model_path: config.body_model_path = args.smpl_model_path
    if args.test_data_dir: config.test.data_path = args.test_data_dir
    if args.num_workers is not None: config.num_workers = args.num_workers
    config.device = str(device)

    # --- Load SMPL Model ---
    smpl_model_path = config.get('body_model_path', 'datasets/smpl_models/smplh/neutral/model.npz')
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
        debug=config.get('debug', False),
        full_sequence=True
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
        batch_size=1, # Should be 1
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
    
    # 将偏移参数转换为numpy数组
    pred_offset_np = np.array(args.pred_offset, dtype=np.float32)
    print(f"Prediction mesh offset: [{pred_offset_np[0]:.2f}, {pred_offset_np[1]:.2f}, {pred_offset_np[2]:.2f}]")
    
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
        show_hands_contact=args.show_hands_contact,
        use_fk=args.use_fk,
        compare_3=args.compare_3,
        pred_offset=pred_offset_np,
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
    if args.compare_3 and not args.vis_gt_only:
        print("Compare-3 mode: displaying fused (red), IMU-integrated (blue), and FK (yellow) object meshes.")
    print("Other standard aitviewer controls should also work (e.g., mouse drag to rotate, scroll to zoom).")
    viewer_instance.run()


if __name__ == "__main__":
    main() 
