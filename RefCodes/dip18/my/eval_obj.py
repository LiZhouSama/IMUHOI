"""
评估脚本 - DIPModelWithObject (重构版)

使用简化的RNN架构（贴近原始DIP实现）:
- 架构: Linear -> ReLU -> Dropout -> BiLSTM -> Linear
- 支持从checkpoint自动加载模型配置
- 支持命令行指定模型超参数
"""
import argparse
import os
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import torch
import pytorch3d.transforms as transforms
from human_body_prior.body_model.body_model import BodyModel

from my.dataset_omomo_dip import OMOMODIPDataset, collate_fn_omomo_dip
from my.model_dip_obj import DIPModelWithObject

  
class FullSequenceLoader:
    """完整序列加载器（无滑动窗口采样）- 使用 OMOMODIPDataset"""

    def __init__(self, dataset: OMOMODIPDataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset.sequences)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # 遍历所有序列，每次返回完整序列
        for i in range(len(self.dataset.sequences)):
            sample = self.dataset[i]
            yield self._sample_to_batch(sample)

    @staticmethod
    def _sample_to_batch(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """将 OMOMODIPDataset 的 sample 转换为 batch 格式 [1, T, ...]"""
        return {
            "human_imu": sample["human_imu"].unsqueeze(0).contiguous(),
            "object_imu": sample["object_imu"].unsqueeze(0).contiguous(),
            "human_pose": sample["human_pose"].unsqueeze(0).contiguous(),
            "human_velocity": sample["human_velocity"].unsqueeze(0).contiguous(),
            "object_velocity": sample["object_velocity"].unsqueeze(0).contiguous(),
            "object_position": sample["object_position"].unsqueeze(0).contiguous(),
            "object_init_pos": sample["object_init_pos"].unsqueeze(0).contiguous(),
        }


@torch.no_grad()
def evaluate_model(
    model: DIPModelWithObject,
    data_loader: Iterable[Dict[str, torch.Tensor]],
    body_model: BodyModel,
    device: torch.device,
    evaluate_objects: bool = True,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    评估 DIPModelWithObject 并计算指标
    
    Returns:
        包含 MPJPE (cm), MPJRE (deg), Jitter (mm/frame^2), 
        Object Trans Error (cm), HOI Error (cm) 的字典
    """
    if body_model is None:
        raise RuntimeError("需要 SMPL body model 进行评估")
    
    if hasattr(data_loader, "__len__") and len(data_loader) == 0:
        raise ValueError("data_loader 为空")
    
    model.eval()
    
    metrics = {
        "mpjpe": [],
        "mpjre": [],
        "jitter": [],
        "obj_trans_err": [],
        "hoi_err": [],
    }
    
    # SMPL父关节索引（24个关节）
    smpl_parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    wrist_indices = [20, 21]  # SMPL中的左右手腕索引
    
    def _compute_pose_and_joints(pose_matrices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将9D旋转矩阵表示转换为局部旋转矩阵和关节位置
        Args:
            pose_matrices: [B, T, J*9] 全局姿态（9D旋转矩阵表示）
        Returns:
            local_pose: [B, T, 24, 3, 3] 局部旋转矩阵
            joints: [B, T, 24, 3] 关节位置（相对根节点）
        """
        B, T, feat = pose_matrices.shape
        J = feat // 9  # Number of joints in the pose representation
        
        # Reshape to [B, T, J, 3, 3]
        pose_mats = pose_matrices.reshape(B, T, J, 3, 3)
        
        # These are already local rotations relative to root
        # So we can directly use them
        local_mats = pose_mats
        
        BT = B * T
        
        # Pad to 24 joints if necessary (model might predict fewer joints)
        if J < 24:
            padded_mats = torch.zeros(B, T, 24, 3, 3, device=pose_mats.device, dtype=pose_mats.dtype)
            # Copy available joints
            padded_mats[:, :, :J] = pose_mats
            # Set remaining joints to identity
            padded_mats[:, :, J:] = torch.eye(3, device=pose_mats.device, dtype=pose_mats.dtype)
            local_mats = padded_mats
        
        # Convert to axis-angle for SMPL
        local_mats_flat = local_mats.reshape(BT * 24, 3, 3)
        axis_angle_flat = transforms.matrix_to_axis_angle(local_mats_flat)
        axis_angle = axis_angle_flat.reshape(BT, 24, 3)
        
        # Pass through SMPL to get joint positions
        body_out = body_model(
            pose_body=axis_angle[:, 1:22].reshape(BT, 63),
            root_orient=axis_angle[:, 0].reshape(BT, 3),
            trans=torch.zeros(BT, 3, device=device, dtype=pose_matrices.dtype),
        )
        
        joints = body_out.Jtr[:, :24, :].reshape(B, T, 24, 3)
        local_pose = local_mats.reshape(B, T, 24, 3, 3)
        
        return local_pose, joints
    
    def _compute_hoi_error(
        pred_obj_pos: torch.Tensor,
        gt_obj_pos: torch.Tensor,
        pred_joints: torch.Tensor,
        gt_joints: torch.Tensor,
        batch_dict: Dict[str, torch.Tensor],
    ) -> float:
        """计算HOI误差（基于手腕-物体相对位置）"""
        pred_hands = pred_joints[:, :, wrist_indices, :]  # [B, T, 2, 3]
        gt_hands = gt_joints[:, :, wrist_indices, :]      # [B, T, 2, 3]
        
        # 计算相对位置误差
        rel_pred = pred_obj_pos.unsqueeze(2) - pred_hands  # [B, T, 2, 3]
        rel_gt = gt_obj_pos.unsqueeze(2) - gt_hands        # [B, T, 2, 3]
        diff = torch.linalg.norm(rel_pred - rel_gt, dim=-1)  # [B, T, 2]
        
        # 如果有接触标签，只计算接触时刻的误差
        collected = []
        lhand_contact = batch_dict.get("lhand_contact")
        rhand_contact = batch_dict.get("rhand_contact")
        
        if lhand_contact is not None:
            l_mask = lhand_contact.to(diff.device).bool()
            if l_mask.any():
                collected.append(diff[:, :, 0][l_mask])
        if rhand_contact is not None:
            r_mask = rhand_contact.to(diff.device).bool()
            if r_mask.any():
                collected.append(diff[:, :, 1][r_mask])
        
        if collected:
            values = torch.cat(collected)
        else:
            # 如果没有接触标签，使用所有帧
            values = diff.reshape(-1)
        
        if values.numel() == 0:
            return float("nan")
        return values.mean().item() * 100.0  # 转为cm
    
    total_batches = len(data_loader) if hasattr(data_loader, "__len__") else None
    
    for batch_idx, batch in enumerate(data_loader):
        batch = {key: value.to(device) for key, value in batch.items()}
        
        human_imu = batch["human_imu"]  # [B, T, F_h]
        object_imu = batch["object_imu"]  # [B, T, F_o]
        obj_init_pos = batch["object_init_pos"]  # [B, 3]
        
        B, T = human_imu.shape[0], human_imu.shape[1]
        
        # 模型推理
        human_pose_pred, obj_vel_pred, obj_pos_pred, _, _ = model(
            human_imu,
            object_imu,
            obj_init_pos,
        )
        
        # Ground truth
        human_pose_gt = batch["human_pose"]  # [B, T, N*6]
        obj_pos_gt = batch["object_position"]  # [B, T, 3]
        
        # 通过body model计算局部旋转和关节位置
        local_pose_pred, joints_pred_rel = _compute_pose_and_joints(human_pose_pred)
        local_pose_gt, joints_gt_rel = _compute_pose_and_joints(human_pose_gt)
        
        # MPJPE (cm) - 使用前22个关节（不包括手部）
        pred_eval_root_normalized = joints_pred_rel[:, :, :22, :]
        gt_eval_root_normalized = joints_gt_rel[:, :, :22, :]
        mpjpe_val = torch.linalg.norm(pred_eval_root_normalized - gt_eval_root_normalized, dim=-1).mean().item() * 100.0
        metrics["mpjpe"].append(mpjpe_val)
        
        # MPJRE (deg) - 使用局部旋转矩阵（1:22为body joints）
        pred_local = local_pose_pred[:, :, 1:22, :, :]
        gt_local = local_pose_gt[:, :, 1:22, :, :]
        pred_body_6d = transforms.matrix_to_rotation_6d(pred_local.reshape(-1, 3, 3)).reshape(B, T, -1, 6)
        gt_body_6d = transforms.matrix_to_rotation_6d(gt_local.reshape(-1, 3, 3)).reshape(B, T,-1, 6)
        rot_error_ = torch.mean(torch.absolute(gt_body_6d-pred_body_6d)) * 57.2958
        mpjre_val = rot_error_.item()
        # # 计算旋转误差：使用Frobenius范数
        # # R_error = R_pred^T @ R_gt，如果相同则为I
        # BT_joints = B * T * 21
        # pred_flat = pred_local.reshape(BT_joints, 3, 3)
        # gt_flat = gt_local.reshape(BT_joints, 3, 3)
        
        # # 计算相对旋转矩阵
        # R_error = torch.bmm(pred_flat.transpose(-1, -2), gt_flat)
        
        # # 从旋转矩阵提取角度：trace(R) = 1 + 2*cos(theta)
        # trace = R_error[:, 0, 0] + R_error[:, 1, 1] + R_error[:, 2, 2]
        # angle_rad = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))
        # angle_deg = angle_rad * 180.0 / np.pi
        # mpjre_val = angle_deg.mean().item()



        metrics["mpjre"].append(mpjre_val)
        
        # Jitter (mm/frame^2) - 使用关节加速度
        if T >= 3:
            pred_joints_eval = joints_pred_rel[:, :, :22, :]  # [B, T, 22, 3]
            acc = (
                pred_joints_eval[:, 2:, :, :]
                - 2.0 * pred_joints_eval[:, 1:-1, :, :]
                + pred_joints_eval[:, :-2, :, :]
            )
            jitter_val = torch.linalg.norm(acc, dim=-1).mean().item() * 1000.0
            metrics["jitter"].append(jitter_val)
        else:
            metrics["jitter"].append(float("nan"))
        
        # Object translation error (cm)
        if evaluate_objects:
            obj_err = torch.linalg.norm(obj_pos_pred - obj_pos_gt, dim=-1).mean().item() * 100.0
            metrics["obj_trans_err"].append(obj_err)
            
            # HOI Error (cm) - 使用真实手腕位置
            hoi_err = _compute_hoi_error(
                obj_pos_pred, obj_pos_gt, 
                joints_pred_rel, joints_gt_rel,
                batch
            )
            metrics["hoi_err"].append(hoi_err)
        else:
            metrics["obj_trans_err"].append(float("nan"))
            metrics["hoi_err"].append(float("nan"))
        
        if verbose and total_batches is not None and (batch_idx + 1) % 10 == 0:
            print(f"[Eval] 已处理 {batch_idx + 1}/{total_batches} 个序列")
    
    # 计算平均值
    averaged = {}
    for key, values in metrics.items():
        valid = [v for v in values if not np.isnan(v)]
        averaged[key] = float(np.mean(valid)) if valid else float("nan")
    
    return averaged


def _load_checkpoint(
    model: DIPModelWithObject,
    checkpoint_path: str,
    device: torch.device,
) -> Dict:
    """
    加载模型权重和配置
    
    Returns:
        包含config等信息的字典（如果有）
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state = checkpoint
    config_dict = None
    
    if isinstance(checkpoint, dict):
        # 提取配置信息
        if "config" in checkpoint:
            config_dict = checkpoint["config"]
        
        # 提取模型权重
        if "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    
    if missing:
        print(f"[Warning] 缺失的键: {sorted(missing)}")
    if unexpected:
        print(f"[Warning] 意外的键: {sorted(unexpected)}")
    
    return {"config": config_dict}


def _prepare_body_model(smplh_path: str, device: torch.device) -> BodyModel:
    """准备SMPL body model"""
    if not os.path.exists(smplh_path):
        raise FileNotFoundError(f"SMPL-H model 不存在: {smplh_path}")
    
    bm = BodyModel(bm_fname=smplh_path, num_betas=16).to(device)
    bm.eval()
    return bm


def _build_sequence_loader(args: argparse.Namespace) -> FullSequenceLoader:
    """构建完整序列加载器 - 使用 OMOMODIPDataset"""
    
    print("[Eval] 使用 OMOMODIPDataset")
    dataset = OMOMODIPDataset(
        dataset_names=args.datasets,
        data_root=args.data_root,
        subset=args.subset,
        seq_len=args.seq_len,
        random_sample=False,
        use_full_sequence=True,
        fps=args.fps,
        trim_frames=args.trim_frames,
        imu_noise_std=args.imu_noise,
        normalize=args.normalize_data,
        data_stats=None,
    )
    
    if len(dataset.sequences) == 0:
        raise RuntimeError("没有可用于评估的序列")
    
    return FullSequenceLoader(dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估 DIPModelWithObject (重构版 - 简化RNN架构) 在数据集上的表现"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/dip_obj_imhd/model_best.pt",
        help="模型checkpoint路径 (.pt)",
    )
    parser.add_argument(
        "--smplh_path",
        type=str,
        default="../../smpl_models/smplh/male/model.npz",
        help="SMPL-H body model 路径",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../../process",
        help="数据集根目录",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["processed_seg_data_IMHD"],
        help="要评估的数据集名称（例如：processed_seg_data_BEHAVE processed_seg_data_IMHD）",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="test",
        help="数据子集 (train/test/debug)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=120,
        help="序列长度（用于固定窗口采样，评估时通常使用完整序列）",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="帧率",
    )
    parser.add_argument(
        "--trim_frames",
        type=int,
        default=6,
        help="从序列开始和结束修剪的帧数",
    )
    parser.add_argument(
        "--imu_noise",
        type=float,
        default=0.0,
        help="IMU高斯噪声标准差 (默认: 0.0，评估时通常不添加噪声)",
    )
    parser.add_argument(
        "--normalize-data",
        default=True,
        help="是否对数据进行归一化（应与训练时设置保持一致）",
    )
    parser.add_argument(
        "--no-object-metrics",
        action="store_true",
        help="跳过物体平移和HOI评估",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="禁用进度信息",
    )
    
    # 模型架构参数（应与训练时保持一致）
    parser.add_argument(
        "--rnn-hidden-size",
        type=int,
        default=512,
        help="RNN隐藏层大小（默认: 512）",
    )
    parser.add_argument(
        "--rnn-layers",
        type=int,
        default=2,
        help="RNN层数（默认: 2）",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        default=True,
        help="使用双向LSTM（原始DIP使用双向）",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout率（默认: 0.2）",
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] 使用设备: {device}")
    
    # 显示配置
    print(f"[Eval] 数据归一化: {'启用' if args.normalize_data else '禁用'}")
    print(f"[Eval] 修剪帧数: {args.trim_frames}")
    
    # 加载数据集
    data_loader = _build_sequence_loader(args)
    print(f"[Eval] 加载了 {len(data_loader)} 个完整序列，子集='{args.subset}'")
    print(f"[Eval] IMU噪声标准差: {args.imu_noise}")
    
    # 获取数据维度
    dataset = data_loader.dataset
    human_input_dim = dataset.human_input_dim
    human_output_dim = dataset.human_pose_dim
    object_input_dim = dataset.object_input_dim
    object_velocity_dim = dataset.object_velocity_dim
    
    print(f"[Eval] 人体输入维度: {human_input_dim}")
    print(f"[Eval] 人体输出维度: {human_output_dim}")
    print(f"[Eval] 物体输入维度: {object_input_dim}")
    print(f"[Eval] 物体速度维度: {object_velocity_dim}")
    
    # 先尝试加载checkpoint以获取配置信息
    print(f"[Eval] 检查checkpoint: {args.checkpoint}")
    dt = 1.0 / args.fps
    
    # 如果checkpoint中有配置，优先使用
    try:
        ckpt_info = torch.load(args.checkpoint, map_location="cpu")
        saved_config = ckpt_info.get("config", {}) if isinstance(ckpt_info, dict) else {}
    except:
        saved_config = {}
    
    # 确定模型超参数（优先级：保存的配置 > 命令行参数）
    rnn_hidden = saved_config.get("rnn_hidden_size", args.rnn_hidden_size)
    rnn_layers = saved_config.get("rnn_layers", args.rnn_layers)
    bidirectional = saved_config.get("rnn_bidirectional", args.bidirectional)
    dropout = saved_config.get("dropout", args.dropout)
    
    print(f"[Eval] 模型配置:")
    print(f"  - RNN hidden size: {rnn_hidden}")
    print(f"  - RNN layers: {rnn_layers}")
    print(f"  - Bidirectional: {bidirectional}")
    print(f"  - Dropout: {dropout}")
    
    # 构建模型（使用简化的RNN架构）
    model = DIPModelWithObject(
        human_input_size=human_input_dim,
        human_output_size=human_output_dim,
        object_input_size=object_input_dim,
        object_velocity_size=object_velocity_dim,
        n_hidden=rnn_hidden,
        n_rnn_layer=rnn_layers,
        bidirectional=bidirectional,
        dropout=dropout,
        dt=dt,
        integrate_position=True,
    ).to(device)
    
    # 加载checkpoint权重
    ckpt_result = _load_checkpoint(model, args.checkpoint, device)
    print(f"[Eval] 加载模型权重: {args.checkpoint}")
    
    # 打印模型参数量
    num_params = model.count_parameters()
    print(f"[Eval] 模型参数量: {num_params:,}")
    
    # 加载SMPL body model
    body_model = _prepare_body_model(args.smplh_path, device)
    print(f"[Eval] 加载SMPL body model: {args.smplh_path}")
    
    # 评估
    metrics = evaluate_model(
        model,
        data_loader,
        body_model,
        device,
        evaluate_objects=not args.no_object_metrics,
        verbose=not args.quiet,
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"模型架构: 简化RNN (Linear->ReLU->Dropout->LSTM->Linear)")
    print(f"  - Hidden: {rnn_hidden}, Layers: {rnn_layers}")
    print(f"  - Bidirectional: {bidirectional}, Dropout: {dropout}")
    print(f"数据集: {args.datasets}, 子集: {args.subset}")
    print(f"序列数量: {len(data_loader)}")
    print("-" * 60)
    print(f"MPJPE (cm):             {metrics.get('mpjpe', float('nan')):.4f}")
    print(f"MPJRE (deg):            {metrics.get('mpjre', float('nan')):.4f}")
    print(f"Jitter (mm/frame^2):    {metrics.get('jitter', float('nan')):.4f}")
    
    if not args.no_object_metrics:
        print(f"Obj Trans Error (cm):   {metrics.get('obj_trans_err', float('nan')):.4f}")
        print(f"HOI Error (cm):         {metrics.get('hoi_err', float('nan')):.4f}")
    else:
        print("跳过物体指标评估")
    print("=" * 60)


if __name__ == "__main__":
    main()

