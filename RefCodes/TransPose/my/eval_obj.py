"""
Evaluation utilities for the model_obj (TransPose-with-object) model.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pytorch3d import transforms

if __package__ is None or __package__ == '':
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _PARENT_DIR = os.path.dirname(_THIS_DIR)
    if _PARENT_DIR not in sys.path:
        sys.path.insert(0, _PARENT_DIR)

from my.dataset_trans_obj import TransPoseObjectDataset, collate_transpose
from my.model_obj import TransPoseWithObject




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TransPose-with-object model.")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/transpose_object_omomo/best_val.pth',
                        help='Path to the trained model checkpoint (.pth).')
    parser.add_argument('--data_root', type=str, default='../../process/',
                        help='Root directory containing evaluation datasets.')
    parser.add_argument('--datasets', nargs='+', default=['processed_split_data_OMOMO'],
                        help='Dataset names to evaluate.')
    parser.add_argument('--subset', type=str, default='test',
                        help='Data split to evaluate.')
    parser.add_argument('--seq_len', type=int, default=120,
                        help='Sequence length used during preprocessing.')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Frame rate used during preprocessing.')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (defaults to cuda if available).')
    return parser.parse_args()


def _compute_metrics(pred: Dict[str, torch.Tensor], gt: Dict[str, torch.Tensor], 
                     model: TransPoseWithObject, imu: torch.Tensor) -> Dict[str, float]:
    """
    计算指标：
    - MPJPE (cm): Mean Per Joint Position Error（相对根关节归一化）
    - MPJRE (deg): Mean Per Joint Rotation Error
    - Obj Trans Error (cm): 物体位移误差
    - HOI Error (cm): 手-物体交互误差（如果有接触标签）
    - Jitter (mm/frame^2): 关节加速度抖动
    """
    metrics = {}
    from human_body_prior.body_model.body_model import BodyModel
    body_model = BodyModel(bm_fname='../../smpl_models/smplh/male/model.npz', num_betas=16)
    
    # 从 IMU 中提取 root rotation: [B, T, 72] -> [B, T, 9] -> [B, T, 3, 3]
    B, T = imu.shape[:2]
    root_rotation = imu[:, :, -9:].reshape(B, T, 3, 3)
    pose_6d_gt = gt['pose'].reshape(B * T, -1, 6)
    pose_mat_gt = transforms.rotation_6d_to_matrix(pose_6d_gt)
    pose_aa_gt = transforms.matrix_to_axis_angle(pose_mat_gt)
    body_model = body_model.to(imu.device)
    body_out_gt = body_model(
                    pose_body=pose_aa_gt[:, 1:22].reshape(B*T, 63).to(imu.device),
                    root_orient=pose_aa_gt[:, 0].reshape(B*T, 3).to(imu.device)
                )
    joints_gt = body_out_gt.Jtr[:, :24, :].to(imu.device).view(B, T, 24, 3)

    pred_pose_6d = pred['reduced_pose'].reshape(B * T, -1, 6)  # [B*T, n_reduced*6]
    root_rot_flat = root_rotation.reshape(B * T, 3, 3)
    pose_mat_pred = model.base._reduced_glb_6d_to_full_local_mat(root_rot_flat, pred_pose_6d)
    pose_aa_pred = transforms.matrix_to_axis_angle(pose_mat_pred)
    body_out_pred = body_model(
                    pose_body=pose_aa_pred[:, 1:22].reshape(B*T, 63).to(imu.device),
                    root_orient=pose_aa_pred[:, 0].reshape(B*T, 3).to(imu.device)
                )
    joints_pred = body_out_pred.Jtr[:, :24, :].to(imu.device).view(B, T, 24, 3)

    # 1. MPJPE: 使用 full_pos (已经是相对根关节的位置)
    if 'full_pos' in pred and 'full_pos' in gt:
        # full_pos: [B, T, n_joints, 3]，已经是相对根关节坐标
        # 取前22个关节（SMPL格式）
        mpjpe = torch.linalg.norm(joints_gt - joints_pred, dim=-1).mean().item() * 100.0  # 转换为cm
        metrics['mpjpe'] = mpjpe    
    
    # 2. MPJRE: 计算关节旋转误差（度）
    if 'reduced_pose' in pred and 'reduced_pose' in gt:
        pose_6d_pred = transforms.matrix_to_rotation_6d(pose_mat_pred.reshape(-1, 3, 3)).reshape(B,T, -1, 6)
        pose_6d_gt = transforms.matrix_to_rotation_6d(pose_mat_gt.reshape(-1, 3, 3)).reshape(B,T, -1, 6)
        # # 只评估前22个关节（排除根关节，从索引1开始）
        pred_local_eval = pose_6d_pred[:, :, 1:22, :]
        gt_local_eval = pose_6d_gt[:, :, 1:22, :]
        
        # # 计算相对旋转: R_rel = R_gt^T @ R_pred
        # rel_rot = torch.matmul(gt_local_eval.transpose(-1, -2), pred_local_eval)
        
        # # 从旋转矩阵的trace计算旋转角度
        # trace = torch.einsum("...ii->...", rel_rot)
        # angles = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
        # mpjre = angles.mean().item() * (180.0 / np.pi)  # 转换为度
        mpjre = torch.mean(torch.absolute(pred_local_eval-gt_local_eval)) * 57.2958
        metrics['mpjre'] = mpjre
    
    # 3. Obj Trans Error: 物体位移误差
    if 'obj_position' in pred and 'obj_position' in gt:
        obj_diff = pred['obj_position'] - gt['obj_position']
        obj_trans_err = torch.linalg.norm(obj_diff, dim=-1).mean().item() * 100.0  # cm
        metrics['obj_trans_err'] = obj_trans_err
    else:
        metrics['obj_trans_err'] = float('nan')
    
    # 4. HOI Error: 手-物体交互误差
    # 需要手腕关节（索引20, 21在SMPL中）和物体位置
    if 'obj_position' in pred and 'obj_position' in gt and 'full_pos' in pred and 'full_pos' in gt:
        # 手腕索引（SMPL ordering）
        wrist_indices = [20, 21]  # left, right wrist
        
        # 从 full_pos 提取手腕位置
        # full_pos: [B, T, n_joints, 3]
        pred_wrists = joints_pred[:, :, wrist_indices, :]  # [B, T, 2, 3]
        gt_wrists = joints_gt[:, :, wrist_indices, :]
        
        # 计算相对关系误差
        # pred_obj_pos: [B, T, 3] -> [B, T, 1, 3]
        rel_pred = pred['obj_position'].unsqueeze(2) - pred_wrists
        rel_gt = gt['obj_position'].unsqueeze(2) - gt_wrists
        
        hoi_diff = torch.linalg.norm(rel_pred - rel_gt, dim=-1)  # [B, T, 2]
        
        # 如果有接触标签，只计算接触时的误差
        if 'lhand_contact' in gt or 'rhand_contact' in gt:
            collected = []
            if 'lhand_contact' in gt:
                l_mask = gt['lhand_contact'].to(hoi_diff.device).bool()
                if l_mask.any():
                    collected.append(hoi_diff[:, :, 0][l_mask])
            if 'rhand_contact' in gt:
                r_mask = gt['rhand_contact'].to(hoi_diff.device).bool()
                if r_mask.any():
                    collected.append(hoi_diff[:, :, 1][r_mask])
            
            if collected:
                hoi_err = torch.cat(collected).mean().item() * 100.0
            else:
                hoi_err = float('nan')
        else:
            # 没有接触标签，计算所有帧的平均
            hoi_err = hoi_diff.mean().item() * 100.0
        
        metrics['hoi_err'] = hoi_err
    else:
        metrics['hoi_err'] = float('nan')
    
    # 5. Jitter: 关节加速度抖动（mm/frame^2）
    if 'full_pos' in pred:
        # 使用相对根关节的关节位置
        pred_joints_eval = joints_pred[:, :, :22, :]  # [B, T, 22, 3]
        
        B, T = pred_joints_eval.shape[:2]
        if T >= 3:
            # 计算二阶差分（加速度）
            acc = (
                pred_joints_eval[:, 2:, :, :]
                - 2.0 * pred_joints_eval[:, 1:-1, :, :]
                + pred_joints_eval[:, :-2, :, :]
            )
            jitter = torch.linalg.norm(acc, dim=-1).mean().item() * 1000.0  # mm
            metrics['jitter'] = jitter
        else:
            metrics['jitter'] = float('nan')
    else:
        metrics['jitter'] = float('nan')
    
    return metrics


def evaluate() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device is not None else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"[Eval] Using device: {device}")

    dataset = TransPoseObjectDataset(
        datasets=args.datasets,
        seq_len=args.seq_len,
        data_root=args.data_root,
        subset=args.subset,
        fps_default=args.fps,
        trim_frames=0,
        random_sample=False,
        use_full_sequence=True,
    )
    if len(dataset) == 0:
        raise RuntimeError("No sequences available for evaluation.")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_transpose,
    )

    model = TransPoseWithObject(fps=args.fps).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    metrics_list: Dict[str, list] = {
        'mpjpe': [],
        'mpjre': [],
        'obj_trans_err': [],
        'hoi_err': [],
        'jitter': [],
    }
    
    iterator = tqdm(dataloader, desc="Evaluating", leave=True)

    for batch_idx, batch in enumerate(iterator):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(
            human_imu=batch['imu'],
            object_imu=batch['obj_imu'],
            obj_pos_init=batch['obj_pos_init'],
            fps=batch['fps'],
        )
        gt = {
            'leaf_pos': batch['leaf_pos'],
            'full_pos': batch['full_pos'],
            'reduced_pose': batch['reduced_pose'],  # 添加旋转数据用于MPJRE计算
            'obj_position': batch['obj_position'],
            'obj_velocity': batch['obj_velocity'],
            'pose': batch['pose'],
        }
        
        # 传递接触标签（如果存在）
        if 'lhand_contact' in batch:
            gt['lhand_contact'] = batch['lhand_contact']
        if 'rhand_contact' in batch:
            gt['rhand_contact'] = batch['rhand_contact']
        
        metrics = _compute_metrics(outputs, gt, model, batch['imu'])
        for key, value in metrics.items():
            if key in metrics_list:
                metrics_list[key].append(value)
        
        if (batch_idx + 1) % 10 == 0:
            iterator.set_postfix({'processed': f'{batch_idx + 1}/{len(dataloader)}'})

    # 计算平均值（忽略 nan），处理 torch.Tensor 在 GPU 上不能直接用 np.isnan/v.numpy 问题
    averaged = {}
    for key, values in metrics_list.items():
        # 保证全部在 CPU 并为 float
        valid = []
        for v in values:
            if isinstance(v, torch.Tensor):
                vitem = v.detach().cpu().item()
            else:
                vitem = float(v)
            if not np.isnan(vitem):
                valid.append(vitem)
        averaged[key] = float(np.mean(valid)) if valid else float('nan')
    
    print("\n--- Evaluation Results ---")
    print(f"MPJPE (cm):             {averaged.get('mpjpe', float('nan')):.4f}")
    print(f"MPJRE (deg):            {averaged.get('mpjre', float('nan')):.4f}")
    print(f"Jitter (mm/frame^2):    {averaged.get('jitter', float('nan')):.4f}")
    print(f"Obj Trans Error (cm):   {averaged.get('obj_trans_err', float('nan')):.4f}")
    print(f"HOI Error (cm):         {averaged.get('hoi_err', float('nan')):.4f}")


if __name__ == '__main__':
    evaluate()

