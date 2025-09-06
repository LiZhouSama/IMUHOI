import os
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from models.DiT_model import MotionDiffusion
from dataloader.dataloader import IMUDataset
from easydict import EasyDict as edict
from human_body_prior.body_model.body_model import BodyModel
import pytorch3d.transforms as transforms
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score
from configs.global_config import FRAME_RATE

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = edict(config)
    return config

def load_smpl_model(smpl_model_path, device):
    """加载 SMPL 模型 using human_body_prior"""
    smpl_model = BodyModel(
        bm_fname=smpl_model_path,
        num_betas=16
    ).to(device)
    return smpl_model

def load_model(config, device):
    """加载训练好的模型（使用配置中的 pretrained_modules 进行模块化加载）"""
    from models.TransPose_net import TransPoseNet
    # 读取配置中的预训练模块
    staged_cfg = config.get('staged_training', {}) if hasattr(config, 'get') else config.staged_training
    modular_cfg = staged_cfg.get('modular_training', {}) if staged_cfg else {}
    use_modular = bool(modular_cfg.get('enabled', False))
    pretrained_modules = modular_cfg.get('pretrained_modules', {}) if use_modular else {}

    if use_modular and pretrained_modules:
        print("Loading TransPose model with pretrained modules:")
        for k, v in pretrained_modules.items():
            print(f"  - {k}: {v}")
        model = TransPoseNet(config, pretrained_modules=pretrained_modules, skip_modules=[]).to(device)
    else:
        print("Warning: No pretrained_modules provided in config; initializing a fresh TransPoseNet.")
        model = TransPoseNet(config).to(device)

    model.eval()
    return model

def evaluate_model(model, smpl_model, data_loader, device, evaluate_objects=True):
    """评估模型性能，计算 MPJPE, MPJRE (角度), Object Trans Error, Contact F1, Jitter (适配 SMPLH)"""
    metrics = {
        'mpjpe': [], 'mpjre_angle': [], 'jitter': [],
        'obj_trans_err_fusion': [], 'obj_trans_err_fk': [], 'obj_trans_err_imu': [],  # 分别跟踪融合、FK、IMU方案的物体位置误差
        'hoi_err_fusion': [], 'hoi_err_fk': [], 'hoi_err_imu': [],                    # 分别跟踪融合、FK、IMU方案的HOI误差
        'contact_f1_lhand': [], 'contact_f1_rhand': [], 'contact_f1_obj': []
    }
    num_batches = 0
    num_eval_joints = 22
    num_body_joints = 21

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # --- Ground Truth Data ---
            gt_root_pos = batch["root_pos"].to(device)
            gt_motion = batch["motion"].to(device)
            gt_human_imu = batch["human_imu"].to(device)
            # 不需要反归一化，移除相关检查

            gt_obj_imu = batch.get("obj_imu", None)
            gt_obj_trans = batch.get("obj_trans", None)
            gt_obj_rot_6d = batch.get("obj_rot", None)
            
            # 获取接触标志
            gt_lhand_contact = batch.get("lhand_contact", None)  # [bs, T]
            gt_rhand_contact = batch.get("rhand_contact", None)  # [bs, T]
            gt_obj_contact = batch.get("obj_contact", None)      # [bs, T]

            if gt_obj_imu is not None: gt_obj_imu = gt_obj_imu.to(device)
            if gt_obj_trans is not None: gt_obj_trans = gt_obj_trans.to(device)
            if gt_obj_rot_6d is not None: gt_obj_rot_6d = gt_obj_rot_6d.to(device)
            if gt_lhand_contact is not None: gt_lhand_contact = gt_lhand_contact.to(device)
            if gt_rhand_contact is not None: gt_rhand_contact = gt_rhand_contact.to(device)
            if gt_obj_contact is not None: gt_obj_contact = gt_obj_contact.to(device)

            bs, seq_len, motion_dim = gt_motion.shape
            if motion_dim != 132:
                 print(f"Warning: Batch {batch_idx}: Expected motion dimension 132 for SMPLH (22*6D), got {motion_dim}. Skipping batch.")
                 continue

            # Determine if the batch contains object data
            has_object = gt_obj_imu is not None and gt_obj_trans is not None and gt_obj_rot_6d is not None

            # --- Model Prediction ---
            # Prepare input dictionary
            model_input = {
                "human_imu": gt_human_imu,
                "motion": gt_motion,             # 用于状态初始化
                "root_pos": gt_root_pos,         # 用于状态初始化
            }
            
            # 添加物体相关输入（如果有）
            if has_object:
                model_input["obj_imu"] = gt_obj_imu        # [bs, T, 1, dim]
                model_input["obj_rot"] = gt_obj_rot_6d     # [bs, T, 6]
                model_input["obj_trans"] = gt_obj_trans    # [bs, T, 3]
            else:
                # 为没有物体数据的情况提供默认值
                device_model = gt_human_imu.device
                model_input["obj_imu"] = torch.zeros(bs, seq_len, 1, gt_human_imu.shape[-1], device=device_model)
                model_input["obj_rot"] = torch.zeros(bs, seq_len, 6, device=device_model)
                model_input["obj_trans"] = torch.zeros(bs, seq_len, 3, device=device_model)
            
            # 添加GT手部位置（如果可用）
            position_global_norm = batch.get("position_global_norm", None)
            if position_global_norm is not None and position_global_norm.shape[1] == seq_len:
                try:
                    wrist_l_idx, wrist_r_idx = 20, 21
                    pos = position_global_norm.to(device)
                    lhand_pos = pos[:, :, wrist_l_idx, :]
                    rhand_pos = pos[:, :, wrist_r_idx, :]
                    gt_hands_pos = torch.stack([lhand_pos, rhand_pos], dim=2)  # [bs, seq, 2, 3]
                    model_input["gt_hands_pos"] = gt_hands_pos
                except Exception:
                    # 如果提取失败，创建零值
                    model_input["gt_hands_pos"] = torch.zeros((bs, seq_len, 2, 3), device=device, dtype=gt_root_pos.dtype)
            else:
                # 如果没有position_global_norm，创建零值
                model_input["gt_hands_pos"] = torch.zeros((bs, seq_len, 2, 3), device=device, dtype=gt_root_pos.dtype)

            try:
                if hasattr(model, 'diffusion_reverse'):
                    pred_dict = model.diffusion_reverse(model_input)
                else:
                    model_input["use_object_data"] = has_object
                    pred_dict = model(model_input)
            except Exception as e:
                 print(f"Error during model inference in batch {batch_idx}: {e}")
                 continue

            # --- Extract Predictions (Normalized) ---
            pred_root_pos_norm = pred_dict.get("root_pos", None)
            pred_motion_norm = pred_dict.get("motion", None)
            pred_obj_trans_fusion = pred_dict.get("pred_obj_trans", None)  # 融合方案的物体位置
            pred_obj_trans_fk = None  # 直接FK方案的物体位置（按需计算）
            pred_hand_contact_prob = pred_dict.get("pred_hand_contact_prob", None)  # [bs, T, 3]
            pred_obj_vel = pred_dict.get("pred_obj_vel", None)  # 纯IMU方案的物体速度 [bs, T, 3]

            if pred_motion_norm is None:
                print(f"Warning: Batch {batch_idx}: Model did not output 'motion'. Skipping batch.")
                continue
            if pred_root_pos_norm is None:
                print(f"Warning: Batch {batch_idx}: Model did not output 'root_pos'. Using GT root position for evaluation.")


            # --- 直接使用归一化数据进行评估 ---
            pred_root_orient_6d_norm = pred_motion_norm[:, :, :6]
            pred_root_orient_mat_norm = transforms.rotation_6d_to_matrix(pred_root_orient_6d_norm)
            pred_root_orient_axis = transforms.matrix_to_axis_angle(pred_root_orient_mat_norm).reshape(bs * seq_len, 3)
            
            if pred_root_pos_norm is not None:
                pred_transl = pred_root_pos_norm.reshape(bs * seq_len, 3)
            else:
                pred_transl = gt_root_pos.reshape(bs*seq_len, 3)

            # Body pose is local, no denormalization needed
            pred_body_pose_6d = pred_motion_norm[:, :, 6:].reshape(bs * seq_len, 21, 6)
            pred_body_pose_mat = transforms.rotation_6d_to_matrix(pred_body_pose_6d.reshape(-1, 6)).reshape(bs * seq_len, 21, 3, 3)
            pred_body_pose_axis = transforms.matrix_to_axis_angle(pred_body_pose_mat.reshape(-1, 3, 3)).reshape(bs * seq_len, 21 * 3)

            # --- 准备GT数据进行SMPL和指标计算 (直接使用归一化数据) ---
            gt_root_orient_6d = gt_motion[:, :, :6]  # [bs, T, 6]
            gt_body_pose_6d = gt_motion[:, :, 6:].reshape(bs * seq_len, 21, 6)
            gt_root_orient_mat = transforms.rotation_6d_to_matrix(gt_root_orient_6d)
            gt_body_pose_mat = transforms.rotation_6d_to_matrix(gt_body_pose_6d.reshape(-1, 6)).reshape(bs * seq_len, 21, 3, 3)
            gt_root_orient_axis = transforms.matrix_to_axis_angle(gt_root_orient_mat).reshape(bs * seq_len, 3)
            gt_body_pose_axis = transforms.matrix_to_axis_angle(gt_body_pose_mat.reshape(-1, 3, 3)).reshape(bs * seq_len, 21 * 3)
            gt_transl = gt_root_pos.reshape(bs * seq_len, 3)

            # --- Get SMPL Joints ---
            pred_pose_body_input = {'root_orient': pred_root_orient_axis, 'pose_body': pred_body_pose_axis, 'trans': pred_transl}
            gt_pose_body_input = {'root_orient': gt_root_orient_axis, 'pose_body': gt_body_pose_axis, 'trans': gt_transl}

            try:
                pred_smplh_out = smpl_model(**pred_pose_body_input)
                gt_smplh_out = smpl_model(**gt_pose_body_input)
            except Exception as e:
                print(f"Error during SMPL forward pass in batch {batch_idx}: {e}")
                continue

            pred_joints_all = pred_smplh_out.Jtr.view(bs, seq_len, -1, 3)
            gt_joints_all = gt_smplh_out.Jtr.view(bs, seq_len, -1, 3)

            # --- Calculate Metrics ---
            # Select joints for evaluation (first 22 for SMPLH compatibility with SMPL eval)
            pred_joints_eval = pred_joints_all[:, :, :num_eval_joints, :]
            gt_joints_eval = gt_joints_all[:, :, :num_eval_joints, :]

            # MPJPE (Mean Per Joint Position Error - Absolute)
            # Calculate Euclidean distance between absolute predicted and GT joint positions
            # Shape of norm: [bs, seq_len, num_eval_joints]
            joint_distances = torch.linalg.norm(pred_joints_eval - gt_joints_eval, dim=-1)
            # Mean over batch, sequence length, and joints
            mpjpe = joint_distances.mean()
            metrics['mpjpe'].append(mpjpe.item() * 1000) # Convert meters to mm

            # MPJRE (Mean Per Joint Rotation Error - Angle in Degrees)
            # Reshape matrices for batch processing: [bs, seq, num_body_joints, 3, 3]
            pred_body_pose_mat_rs = pred_body_pose_mat.view(bs, seq_len, 21, 3, 3)
            gt_body_pose_mat_rs = gt_body_pose_mat.view(bs, seq_len, 21, 3, 3)
            # Calculate relative rotation: R_rel = R_gt^T @ R_pred
            # Transpose gt: [bs, seq, num_body_joints, 3, 3]
            rel_rot_mat = torch.matmul(gt_body_pose_mat_rs.transpose(-1, -2), pred_body_pose_mat_rs)
            # Calculate trace: sum of diagonal elements
            trace = torch.einsum('...ii->...', rel_rot_mat) # Shape: [bs, seq, num_body_joints]
            # Calculate cos(theta) = (trace - 1) / 2
            cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0) # Clamp for numerical stability
            # Calculate angle in radians: theta = arccos(cos_theta)
            angle_rad = torch.acos(cos_theta)
            # Convert angle to degrees
            angle_deg = angle_rad * (180.0 / np.pi)
            # Calculate the mean angle error over batch, sequence, and joints
            mpjre_angle = angle_deg.mean()
            metrics['mpjre_angle'].append(mpjre_angle.item()) # Store mean angle in degrees

            # Object Translation Error - 融合方案 (直接使用归一化数据)
            if evaluate_objects and has_object and pred_obj_trans_fusion is not None:
                # 直接计算归一化空间中的平移误差
                obj_trans_err_fusion = torch.linalg.norm(pred_obj_trans_fusion - gt_obj_trans, dim=-1).mean()
                metrics['obj_trans_err_fusion'].append(obj_trans_err_fusion.item() * 1000) # Convert to mm scale
            elif evaluate_objects and has_object:
                print(f"Warning: Batch {batch_idx}: Missing fusion object translation for error calculation.")
                metrics['obj_trans_err_fusion'].append(float('nan'))
            elif evaluate_objects and not has_object:
                metrics['obj_trans_err_fusion'].append(float('nan'))

            # 按需计算FK方案的物体位置
            if evaluate_objects and has_object and pred_hand_contact_prob is not None:
                # 获取手部位置 [bs, seq, 2, 3]
                pred_hand_positions = pred_dict.get("pred_hand_pos", None)
                if pred_hand_positions is None:
                    hands_pos_feat = pred_dict.get("hands_pos_feat", None)
                    if hands_pos_feat is not None:
                        try:
                            pred_hand_positions = hands_pos_feat.reshape(bs, seq_len, 2, 3)
                        except Exception:
                            pred_hand_positions = None
                # 物体旋转矩阵 [bs, seq, 3, 3]
                obj_rot_matrix = transforms.rotation_6d_to_matrix(gt_obj_rot_6d) if gt_obj_rot_6d is not None else None
                try:
                    if pred_hand_positions is not None and obj_rot_matrix is not None and hasattr(model, 'object_trans_module'):
                        pred_obj_trans_fk, _fk_info = model.object_trans_module.predict_object_position_from_contact(
                            pred_hand_contact_prob=pred_hand_contact_prob,
                            pred_hand_positions=pred_hand_positions,
                            obj_rot_matrix=obj_rot_matrix,
                            gt_obj_trans=gt_obj_trans
                        )
                except Exception as _e:
                    pred_obj_trans_fk = None

            # 纯IMU方案：由速度积分得到物体位置
            pred_obj_trans_imu = None
            if evaluate_objects and has_object and pred_obj_vel is not None:
                try:
                    dt = 1.0 / float(FRAME_RATE)
                    # 以GT首帧为锚点做绝对位置积分
                    init_pos = gt_obj_trans[:, 0, :]  # [bs, 3]
                    disp = torch.cumsum(pred_obj_vel * dt, dim=1)  # [bs, T, 3]
                    pred_obj_trans_imu = disp + init_pos.unsqueeze(1)
                except Exception:
                    pred_obj_trans_imu = None

            # Object Translation Error - FK方案 (直接使用归一化数据)
            if evaluate_objects and has_object and pred_obj_trans_fk is not None:
                # 直接计算归一化空间中的平移误差
                obj_trans_err_fk = torch.linalg.norm(pred_obj_trans_fk - gt_obj_trans, dim=-1).mean()
                metrics['obj_trans_err_fk'].append(obj_trans_err_fk.item() * 1000) # Convert to mm scale
            elif evaluate_objects and has_object:
                print(f"Warning: Batch {batch_idx}: Missing FK object translation for error calculation.")
                metrics['obj_trans_err_fk'].append(float('nan'))
            elif evaluate_objects and not has_object:
                metrics['obj_trans_err_fk'].append(float('nan'))

            # Object Translation Error - IMU方案 (速度积分)
            if evaluate_objects and has_object and pred_obj_trans_imu is not None:
                obj_trans_err_imu = torch.linalg.norm(pred_obj_trans_imu - gt_obj_trans, dim=-1).mean()
                metrics['obj_trans_err_imu'].append(obj_trans_err_imu.item() * 1000)  # mm
            elif evaluate_objects and has_object:
                print(f"Warning: Batch {batch_idx}: Missing IMU-integrated object translation for error calculation.")
                metrics['obj_trans_err_imu'].append(float('nan'))
            elif evaluate_objects and not has_object:
                metrics['obj_trans_err_imu'].append(float('nan'))

            # Object-Hand Relative Position Error - 融合方案 (在真值交互帧下)
            def compute_hoi_error(pred_obj_trans, method_name):
                """计算HOI误差的通用函数"""
                if evaluate_objects and has_object and pred_obj_trans is not None:
                    # 提取手腕关节位置 (SMPL joint indices: 20=left wrist, 21=right wrist)
                    wrist_l_idx, wrist_r_idx = 20, 21
                    
                    # 预测的手腕位置
                    pred_lhand_pos = pred_joints_all[:, :, wrist_l_idx, :]  # [bs, seq, 3]
                    pred_rhand_pos = pred_joints_all[:, :, wrist_r_idx, :]  # [bs, seq, 3]
                    
                    # 真值的手腕位置
                    gt_lhand_pos = gt_joints_all[:, :, wrist_l_idx, :]      # [bs, seq, 3]
                    gt_rhand_pos = gt_joints_all[:, :, wrist_r_idx, :]      # [bs, seq, 3]
                    
                    relative_errors = []
                    
                    # 计算左手交互帧的相对位置误差
                    if gt_lhand_contact is not None:
                        lhand_contact_mask = gt_lhand_contact.bool()  # [bs, seq]
                        if lhand_contact_mask.any():
                            # 在交互帧中计算物体相对于左手的位置
                            gt_obj_rel_lhand = gt_obj_trans - gt_lhand_pos        # [bs, seq, 3]
                            pred_obj_rel_lhand = pred_obj_trans - pred_lhand_pos  # [bs, seq, 3]
                            
                            # 只在真值交互帧计算误差
                            valid_frames = lhand_contact_mask.unsqueeze(-1).expand_as(gt_obj_rel_lhand)  # [bs, seq, 3]
                            if valid_frames.any():
                                gt_rel_valid = gt_obj_rel_lhand[valid_frames].view(-1, 3)  # 重塑为 [N, 3]
                                pred_rel_valid = pred_obj_rel_lhand[valid_frames].view(-1, 3)  # 重塑为 [N, 3]
                                lhand_rel_error = torch.linalg.norm(pred_rel_valid - gt_rel_valid, dim=-1)  # [N]
                                relative_errors.append(lhand_rel_error)
                    
                    # 计算右手交互帧的相对位置误差
                    if gt_rhand_contact is not None:
                        rhand_contact_mask = gt_rhand_contact.bool()  # [bs, seq]
                        if rhand_contact_mask.any():
                            # 在交互帧中计算物体相对于右手的位置
                            gt_obj_rel_rhand = gt_obj_trans - gt_rhand_pos        # [bs, seq, 3]
                            pred_obj_rel_rhand = pred_obj_trans - pred_rhand_pos  # [bs, seq, 3]
                            
                            # 只在真值交互帧计算误差
                            valid_frames = rhand_contact_mask.unsqueeze(-1).expand_as(gt_obj_rel_rhand)  # [bs, seq, 3]
                            if valid_frames.any():
                                gt_rel_valid = gt_obj_rel_rhand[valid_frames].view(-1, 3)  # 重塑为 [N, 3]
                                pred_rel_valid = pred_obj_rel_rhand[valid_frames].view(-1, 3)  # 重塑为 [N, 3]
                                rhand_rel_error = torch.linalg.norm(pred_rel_valid - gt_rel_valid, dim=-1)  # [N]
                                relative_errors.append(rhand_rel_error)
                    
                    # 合并所有相对误差
                    if relative_errors:
                        all_relative_errors = torch.cat(relative_errors, dim=0)
                        hoi_err = all_relative_errors.mean()
                        return hoi_err.item() * 1000  # Convert to mm scale
                    else:
                        # 没有交互帧
                        return float('nan')
                elif evaluate_objects and has_object:
                    print(f"Warning: Batch {batch_idx}: Missing {method_name} object translation for HOI error calculation.")
                    return float('nan')
                elif evaluate_objects and not has_object:
                    return float('nan')
                else:
                    return float('nan')

            # 计算融合方案的HOI误差
            hoi_err_fusion = compute_hoi_error(pred_obj_trans_fusion, "fusion")
            metrics['hoi_err_fusion'].append(hoi_err_fusion)

            # 计算FK方案的HOI误差
            hoi_err_fk = compute_hoi_error(pred_obj_trans_fk, "FK")
            metrics['hoi_err_fk'].append(hoi_err_fk)

            # 计算IMU方案的HOI误差
            hoi_err_imu = compute_hoi_error(pred_obj_trans_imu, "IMU")
            metrics['hoi_err_imu'].append(hoi_err_imu)

            # Contact Prediction Evaluation
            if pred_hand_contact_prob is not None:
                # Convert probabilities to binary predictions (threshold = 0.5)
                pred_contacts = (pred_hand_contact_prob > 0.5).float()  # [bs, T, 3]
                # pred_contacts = torch.ones_like(pred_contacts)  # 测试默认预测接触的baseline
                pred_lhand_contact = pred_contacts[:, :, 0]  # [bs, T]
                pred_rhand_contact = pred_contacts[:, :, 1]  # [bs, T] 
                pred_obj_contact = pred_contacts[:, :, 2]    # [bs, T]
                
                # Calculate F1 scores for each contact type
                if gt_lhand_contact is not None:
                    gt_lhand_flat = gt_lhand_contact.cpu().numpy().flatten()
                    pred_lhand_flat = pred_lhand_contact.cpu().numpy().flatten()
                    if len(np.unique(gt_lhand_flat)) > 1:  # 确保有正负样本
                        lhand_f1 = f1_score(gt_lhand_flat, pred_lhand_flat, average='binary')
                        metrics['contact_f1_lhand'].append(lhand_f1)
                    else:
                        metrics['contact_f1_lhand'].append(float('nan'))
                else:
                    metrics['contact_f1_lhand'].append(float('nan'))
                    
                if gt_rhand_contact is not None:
                    gt_rhand_flat = gt_rhand_contact.cpu().numpy().flatten()
                    pred_rhand_flat = pred_rhand_contact.cpu().numpy().flatten()
                    if len(np.unique(gt_rhand_flat)) > 1:  # 确保有正负样本
                        rhand_f1 = f1_score(gt_rhand_flat, pred_rhand_flat, average='binary')
                        metrics['contact_f1_rhand'].append(rhand_f1)
                    else:
                        metrics['contact_f1_rhand'].append(float('nan'))
                else:
                    metrics['contact_f1_rhand'].append(float('nan'))
                    
                if gt_obj_contact is not None:
                    gt_obj_flat = gt_obj_contact.cpu().numpy().flatten()
                    pred_obj_flat = pred_obj_contact.cpu().numpy().flatten()
                    if len(np.unique(gt_obj_flat)) > 1:  # 确保有正负样本
                        obj_f1 = f1_score(gt_obj_flat, pred_obj_flat, average='binary')
                        metrics['contact_f1_obj'].append(obj_f1)
                    else:
                        metrics['contact_f1_obj'].append(float('nan'))
                else:
                    metrics['contact_f1_obj'].append(float('nan'))
            else:
                # 如果没有预测接触，则添加NaN
                metrics['contact_f1_lhand'].append(float('nan'))
                metrics['contact_f1_rhand'].append(float('nan'))
                metrics['contact_f1_obj'].append(float('nan'))


            # Jitter (Mean acceleration magnitude of joints)
            if seq_len >= 3:
                # Use world frame joints for jitter calculation
                pred_accel = pred_joints_eval[:, 2:] - 2 * pred_joints_eval[:, 1:-1] + pred_joints_eval[:, :-2]
                jitter = torch.linalg.norm(pred_accel, dim=-1).mean()
                metrics['jitter'].append(jitter.item() * 1000) # Convert m/frame^2 to mm/frame^2
            else:
                 metrics['jitter'].append(float('nan'))

            num_batches += 1
            if batch_idx % 50 == 0:
                print(f"Processed batch {batch_idx + 1} / {len(data_loader)}")

    avg_metrics = {}
    print("\nCalculating average metrics...")
    for key, values in metrics.items():
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            avg_metrics[key] = np.mean(valid_values)
            unit = ""
            if 'mpjpe' in key or 'obj_trans_err' in key or 'hoi_err' in key: unit = "(mm)"
            elif key == 'mpjre_angle': unit = "(deg)"
            elif key == 'jitter': unit = "(mm/frame^2)"
            elif 'contact_f1' in key: unit = "(F1)"
            print(f"  {key} {unit}: {avg_metrics[key]:.4f} (from {len(valid_values)}/{len(values)} valid samples)")
        else:
            avg_metrics[key] = float('nan')
            print(f"  {key}: NaN (no valid samples)")

    return avg_metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate EgoMotion Model')
    parser.add_argument('--config', type=str, default='configs/TransPose_train.yaml', help='Path to the configuration file.')
    parser.add_argument('--smpl_model_path', type=str, default=None, help='Path to the SMPL model file (e.g., SMPLH neutral). Overrides config if provided.')
    parser.add_argument('--test_data_dir', type=str, default=None, help='Path to the test dataset directory. Overrides config if provided.')
    parser.add_argument('--batch_size', type=int, default=None, help='Test batch size. Overrides config if provided.')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of dataloader workers. Overrides config if provided.')
    parser.add_argument('--no_eval_objects', action='store_true', help='Do not evaluate object pose errors.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Override config values with command line arguments if provided
    if args.smpl_model_path: config.bm_path = args.smpl_model_path
    if args.test_data_dir: config.test.data_path = args.test_data_dir
    if args.batch_size: config.test.batch_size = args.batch_size
    if args.num_workers: config.num_workers = args.num_workers

    # Validate paths from config
    smpl_model_path = config.get('bm_path', 'body_models/smplh/neutral/model.npz')
    if not os.path.exists(smpl_model_path):
       print(f"Error: SMPL model path not found: {smpl_model_path}")
       print("Please provide the correct path in the config (smpl_model_path) or via --smpl_model_path.")
       return
    print(f"Loading SMPL model from: {smpl_model_path}")
    smpl_model = load_smpl_model(smpl_model_path, device)
    model = load_model(config, device)

    test_data_dir = config.datasets.omomo.get('test_path', None)
    if test_data_dir is None or not os.path.exists(test_data_dir):
        print(f"Error: Test dataset path not found or invalid: {test_data_dir}")
        print("Please provide the correct path in the config (test.data_path) or via --test_data_dir.")
        return
    print(f"Loading test dataset from: {test_data_dir}")

    # Use test window size from config
    test_window_size = config.test.get('window', config.train.get('window', 60))
    test_dataset = IMUDataset(
            data_dir=test_data_dir,
            window_size=test_window_size,
            normalize=config.test.get('normalize', True),
            debug=config.get('debug', False)
        )

    if len(test_dataset) == 0:
         print("Error: Test dataset is empty. Check data path and dataset parameters.")
         return

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test.get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=False
    )

    print(f"\nDataset size: {len(test_dataset)}, Loader size: {len(test_loader)}")
    print(f"Batch size: {config.test.get('batch_size', 32)}, Num workers: {config.get('num_workers', 4)}")
    print(f"Evaluating objects: {not args.no_eval_objects}")

    print("\nStarting model evaluation...")
    results = evaluate_model(model, smpl_model, test_loader, device, evaluate_objects=(not args.no_eval_objects))

    print("\n--- Evaluation Results ---")
    print(f"MPJPE (mm):                                   {results.get('mpjpe', 'N/A'):.4f}")
    print(f"MPJRE (deg):                                  {results.get('mpjre_angle', 'N/A'):.4f}")
    print(f"Jitter (mm/frame^2):                          {results.get('jitter', 'N/A'):.4f}")
    if not args.no_eval_objects:
        print(f"\n--- Object Position Errors (方案对比) ---")
        print(f"Obj Trans Error - Fusion (mm):               {results.get('obj_trans_err_fusion', 'N/A'):.4f}")
        print(f"Obj Trans Error - FK Only (mm):              {results.get('obj_trans_err_fk', 'N/A'):.4f}")
        print(f"Obj Trans Error - IMU Only (mm):             {results.get('obj_trans_err_imu', 'N/A'):.4f}")
        
        print(f"\n--- HOI Errors (方案对比) ---")
        print(f"HOI Error - Fusion (mm):                     {results.get('hoi_err_fusion', 'N/A'):.4f}")
        print(f"HOI Error - FK Only (mm):                    {results.get('hoi_err_fk', 'N/A'):.4f}")
        print(f"HOI Error - IMU Only (mm):                   {results.get('hoi_err_imu', 'N/A'):.4f}")
        
        # 计算改进量
        fusion_obj_err = results.get('obj_trans_err_fusion', float('nan'))
        fk_obj_err = results.get('obj_trans_err_fk', float('nan'))
        imu_obj_err = results.get('obj_trans_err_imu', float('nan'))
        fusion_hoi_err = results.get('hoi_err_fusion', float('nan'))
        fk_hoi_err = results.get('hoi_err_fk', float('nan'))
        imu_hoi_err = results.get('hoi_err_imu', float('nan'))
        
        if not (np.isnan(fusion_obj_err) or np.isnan(fk_obj_err)):
            obj_improvement = ((fk_obj_err - fusion_obj_err) / fk_obj_err) * 100
            print(f"Obj Trans Improvement (融合 vs FK):           {obj_improvement:+.2f}%")
        if not (np.isnan(fusion_obj_err) or np.isnan(imu_obj_err)):
            obj_improvement_imu = ((imu_obj_err - fusion_obj_err) / imu_obj_err) * 100
            print(f"Obj Trans Improvement (融合 vs IMU):          {obj_improvement_imu:+.2f}%")
        
        if not (np.isnan(fusion_hoi_err) or np.isnan(fk_hoi_err)):
            hoi_improvement = ((fk_hoi_err - fusion_hoi_err) / fk_hoi_err) * 100
            print(f"HOI Improvement (融合 vs FK):                 {hoi_improvement:+.2f}%")
        if not (np.isnan(fusion_hoi_err) or np.isnan(imu_hoi_err)):
            hoi_improvement_imu = ((imu_hoi_err - fusion_hoi_err) / imu_hoi_err) * 100
            print(f"HOI Improvement (融合 vs IMU):                {hoi_improvement_imu:+.2f}%")
    else:
         print("Object metrics skipped.")
    
    print(f"\n--- Contact Prediction ---")
    print(f"LHand Contact F1:                             {results.get('contact_f1_lhand', 'N/A'):.4f}")
    print(f"RHand Contact F1:                             {results.get('contact_f1_rhand', 'N/A'):.4f}")
    print(f"Obj Contact F1:                               {results.get('contact_f1_obj', 'N/A'):.4f}")
    print("--------------------------")

if __name__ == "__main__":
    main()