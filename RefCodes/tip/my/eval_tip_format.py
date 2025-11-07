"""
Evaluation script for TIP-format trained models.
Uses TF_RNN_Past_State model and computes comprehensive metrics.
"""
import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import pytorch3d.transforms as transforms
from human_body_prior.body_model.body_model import BodyModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_transformer_with_state import TF_RNN_Past_State
from my.dataset_omomo_tip import OMOMODatasetWithObject  # Use old dataset for GT data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TIP-format model on OMOMO sequences.")
    parser.add_argument("--data_dirs", type=str, nargs="+", 
                       default=['../../process/processed_data_OMOMO/test'],
                       help="Evaluation sequence directories.")
    parser.add_argument("--weights", type=str, 
                       default='checkpoints/tip_omomo_format/latest.pt',
                       help="Checkpoint path (.pt).")
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--root_supervision", type=str, default="vel", choices=["vel", "pos"])
    parser.add_argument("--imu_noise_std", type=float, default=0.0, 
                       help="Gaussian IMU noise at inference (default 0.0 for clean eval).")
    parser.add_argument("--eval_contacts", action="store_true", 
                       help="Load contact signals for HOI error computation.")
    parser.add_argument(
        "--smplh_path",
        type=str,
        default='../../smpl_models/smplh/male/model.npz',
        help="Path to SMPLH body model.",
    )
    parser.add_argument("--with_acc_sum", action="store_true",
                       help="Use accelerometer sum features (must match training).")
    parser.add_argument("--use_object_imu", action="store_true",
                       help="Include object IMU (must match training).")
    
    # Model architecture (must match training)
    parser.add_argument("--rnn_nhid", type=int, default=512)
    parser.add_argument("--tf_nhid", type=int, default=1024)
    parser.add_argument("--tf_in_dim", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--tf_layers", type=int, default=4)
    parser.add_argument("--past_dropout", type=float, default=0.8)
    
    return parser.parse_args()


def _add_imu_noise(tensor: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return tensor
    return tensor + torch.randn_like(tensor) * std


def _integrate_velocity(vel: torch.Tensor, init: torch.Tensor, fps: float) -> torch.Tensor:
    """Integrate velocity to get position."""
    if vel.ndim != 2 or vel.shape[-1] != 3:
        raise ValueError("Expected velocity tensor with shape [T, 3].")
    dt = 1.0 / float(fps)
    increments = vel * dt
    return torch.cumsum(increments, dim=0) + init.unsqueeze(0)


def _prepare_model(
    dataset: OMOMODatasetWithObject, 
    args: argparse.Namespace, 
    device: torch.device
) -> TF_RNN_Past_State:
    """Load TIP-format model (TF_RNN_Past_State)."""
    # Calculate input/output dimensions
    num_imus_total = dataset.num_imus_total
    state_dim = dataset.state_dim
    
    input_channels = num_imus_total * 9
    if args.with_acc_sum:
        input_channels += num_imus_total * 3
    
    print(f"[Model] Input channels: {input_channels}")
    print(f"[Model] State dim: {state_dim}")
    
    model = TF_RNN_Past_State(
        input_size_imu=input_channels,
        size_s=state_dim,
        rnn_hid_size=args.rnn_nhid,
        tf_hid_size=args.tf_nhid,
        tf_in_dim=args.tf_in_dim,
        n_heads=args.n_heads,
        tf_layers=args.tf_layers,
        dropout=0.0,  # No dropout at inference
        in_dropout=0.0,
        past_state_dropout=0.0,  # No dropout at inference
        with_rnn=True,
        with_acc_sum=args.with_acc_sum
    )
    
    # Load weights
    state_dict = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    
    print(f"[Model] Loaded weights from {args.weights}")
    
    return model


def _prepare_body_model(args: argparse.Namespace, device: torch.device) -> BodyModel:
    """Load SMPLH body model for FK evaluation."""
    if not os.path.exists(args.smplh_path):
        print(f"[Warning] SMPLH model not found at {args.smplh_path}")
        print("[Warning] MPJPE and related metrics will not be computed.")
        return None
    
    bm = BodyModel(bm_fname=args.smplh_path, num_betas=16).to(device)
    bm.eval()
    return bm


def evaluate_sequences(
    model: TF_RNN_Past_State,
    dataset: OMOMODatasetWithObject,
    device: torch.device,
    args: argparse.Namespace,
    bm: BodyModel,
) -> Dict[str, List[float]]:
    """Evaluate model on all sequences and collect metrics."""
    metrics: Dict[str, List[float]] = {
        "mpjpe": [],
        "mpjre": [],
        "jitter": [],
        "obj_trans_err": [],
        "hoi_err": [],
    }

    human_root_slice = slice(
        dataset.human_out_rot_dim, 
        dataset.human_out_rot_dim + dataset.root_pos_dim
    )
    tip_18_from_22_idx = dataset.tip_18_from_22_idx
    wrist_indices = [20, 21]  # left, right wrist in SMPL-H ordering

    print(f"\n[Eval] Evaluating {len(dataset)} sequences...")

    for idx in range(len(dataset)):
        sample = dataset[idx]
        meta = sample.get("metadata", {})
        seq_name = meta.get("name", f"seq_{idx}")
        fps = float(meta.get("frame_rate", args.fps))

        imu = sample["imu"].unsqueeze(0).to(device)  # [1, T, C]
        state_hist = sample["state_hist"].unsqueeze(0).to(device)  # [1, T, D]
        target_state = sample["state_target"].to(device)  # [T, D]

        # Add optional IMU noise
        imu_in = _add_imu_noise(imu, args.imu_noise_std)

        # Predict
        with torch.no_grad():
            pred_state = model(imu_in, state_hist).squeeze(0)  # [T, D]

        # Extract predictions
        obj_vel_pred = pred_state[:, -3:]
        root_pred = pred_state[:, human_root_slice]

        # Compute metrics if GT data is available
        if bm is not None and "gt_motion" in sample and sample["gt_motion"] is not None:
            gt_motion = sample["gt_motion"].to(device)  # [L, 22*6]
            L = gt_motion.shape[0]
            motion22_gt = gt_motion.reshape(L, 22, 6)

            # Reconstruct 22-joint motion from 18-joint prediction
            human_rot6d_pred = pred_state[:, : dataset.human_out_rot_dim].reshape(
                L, dataset.human_joint_num_for_output, 6
            )
            motion22_pred = motion22_gt.clone()
            for k, joint in enumerate(tip_18_from_22_idx):
                motion22_pred[:, joint, :] = human_rot6d_pred[:, k, :]

            # Convert to axis-angle for body model
            root6d_pred = motion22_pred[:, 0, :]
            pose6d_pred = motion22_pred[:, 1:, :].reshape(L, 21, 6)
            root_axis_pred = transforms.matrix_to_axis_angle(
                transforms.rotation_6d_to_matrix(root6d_pred)
            )
            pose_axis_pred = transforms.matrix_to_axis_angle(
                transforms.rotation_6d_to_matrix(pose6d_pred.reshape(-1, 6))
            ).reshape(L, 21 * 3)

            root6d_gt = motion22_gt[:, 0, :]
            pose6d_gt = motion22_gt[:, 1:, :].reshape(L, 21, 6)
            root_axis_gt = transforms.matrix_to_axis_angle(
                transforms.rotation_6d_to_matrix(root6d_gt)
            )
            pose_axis_gt = transforms.matrix_to_axis_angle(
                transforms.rotation_6d_to_matrix(pose6d_gt.reshape(-1, 6))
            ).reshape(L, 21 * 3)

            # Integrate root velocity to get position
            if dataset.root_supervision == "vel":
                root_pos_pred = _integrate_velocity(
                    root_pred, torch.zeros(3, device=device), fps
                )
            else:
                root_pos_pred = root_pred

            # Forward kinematics
            pred_out = bm(
                root_orient=root_axis_pred, 
                pose_body=pose_axis_pred, 
                trans=root_pos_pred
            )
            gt_out = bm(
                root_orient=root_axis_gt,
                pose_body=pose_axis_gt,
                trans=torch.zeros_like(root_pos_pred),
            )

            pred_joints = pred_out.Jtr[:, :22, :]  # [L, 22, 3]
            gt_joints = gt_out.Jtr[:, :22, :]  # [L, 22, 3]

            # === Metric 1: MPJPE (cm) ===
            # Mean Per Joint Position Error in root coordinate
            mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1).mean().item() * 100.0
            metrics["mpjpe"].append(mpjpe)

            # === Metric 2: MPJRE (deg) ===
            # Mean Per Joint Rotation Error using 6D representation
            pred_body_6d = motion22_pred[:, 1:22, :].reshape(L, 21, 6)
            gt_body_6d = motion22_gt[:, 1:22, :].reshape(L, 21, 6)
            rot_error = torch.mean(torch.abs(gt_body_6d - pred_body_6d)) * 57.2958  # rad to deg
            metrics["mpjre"].append(rot_error.item())

            # === Metric 3: Jitter (mm/frame^2) ===
            # Acceleration magnitude of predicted joints
            if L >= 3:
                acc = pred_joints[2:] - 2 * pred_joints[1:-1] + pred_joints[:-2]
                jitter = torch.linalg.norm(acc, dim=-1).mean().item() * 1000.0
                metrics["jitter"].append(jitter)

            # === Metric 4 & 5: Object and HOI errors ===
            if "obj_pos_gt" in sample and sample["obj_pos_gt"] is not None:
                obj_pos_gt = sample["obj_pos_gt"].to(device)  # [L, 3]
                obj_pos_init = sample["obj_pos_init"].to(device)  # [3]
                obj_pos_pred = _integrate_velocity(obj_vel_pred, obj_pos_init, fps)

                # === Metric 4: Object Translation Error (cm) ===
                obj_err = torch.linalg.norm(obj_pos_pred - obj_pos_gt, dim=-1).mean().item() * 100.0
                metrics["obj_trans_err"].append(obj_err)

                # === Metric 5: HOI Error (cm) ===
                # Hand-object relative position error
                pred_hands = pred_joints[:, wrist_indices, :]  # [L, 2, 3]
                gt_hands = gt_joints[:, wrist_indices, :]  # [L, 2, 3]

                # Relative positions: object - hand
                rel_pred = obj_pos_pred.unsqueeze(1) - pred_hands  # [L, 2, 3]
                rel_gt = obj_pos_gt.unsqueeze(1) - gt_hands  # [L, 2, 3]
                diff = torch.linalg.norm(rel_pred - rel_gt, dim=-1)  # [L, 2]

                # If contact labels available, only compute error during contact
                collected = []
                if "contacts" in sample and sample["contacts"] is not None:
                    contacts = sample["contacts"].to(device)  # [L, 2] - [lhand, rhand]
                    for hand_idx in range(2):
                        contact_mask = contacts[:, hand_idx].bool()
                        if contact_mask.any():
                            collected.append(diff[contact_mask, hand_idx])

                if collected:
                    hoi_err = torch.cat(collected).mean().item() * 100.0
                else:
                    # No contact labels or no contacts - compute average over all frames
                    hoi_err = diff.mean().item() * 100.0

                metrics["hoi_err"].append(hoi_err)

        if (idx + 1) % 10 == 0 or idx == len(dataset) - 1:
            print(f"  Processed {idx + 1}/{len(dataset)} sequences")

    return metrics


def summarize_metrics(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    """Compute mean of all collected metrics."""
    summary: Dict[str, float] = {}
    for key, values in metrics.items():
        if values:
            summary[key] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    return summary


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")
    print(f"[Eval] Weights: {args.weights}")

    # Load dataset with full sequence mode for evaluation
    dataset = OMOMODatasetWithObject(
        data_dirs=args.data_dirs,
        seq_len=args.seq_len,
        frame_rate=args.fps,
        use_object_imu=args.use_object_imu,
        root_supervision=args.root_supervision,
        return_gt_pos=True,
        return_gt_motion=True,
        return_contacts=args.eval_contacts,
        use_full_sequence=True,  # Evaluate on full sequences
        random_sample=False,
    )

    if len(dataset) == 0:
        raise RuntimeError("No sequences found for evaluation.")

    print(f"[Eval] Loaded {len(dataset)} sequences")

    # Load model
    model = _prepare_model(dataset, args, device)

    # Load body model (optional)
    bm = _prepare_body_model(args, device)

    # Evaluate
    metrics = evaluate_sequences(model, dataset, device, args, bm)
    summary = summarize_metrics(metrics)

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    if "mpjpe" in summary:
        print(f"MPJPE (cm):               {summary['mpjpe']:.4f} ± {summary.get('mpjpe_std', 0.0):.4f}")
    if "mpjre" in summary:
        print(f"MPJRE (deg):              {summary['mpjre']:.4f} ± {summary.get('mpjre_std', 0.0):.4f}")
    if "jitter" in summary:
        print(f"Jitter (mm/frame²):       {summary['jitter']:.4f} ± {summary.get('jitter_std', 0.0):.4f}")
    if "obj_trans_err" in summary:
        print(f"Obj Trans Error (cm):     {summary['obj_trans_err']:.4f} ± {summary.get('obj_trans_err_std', 0.0):.4f}")
    if "hoi_err" in summary:
        print(f"HOI Error (cm):           {summary['hoi_err']:.4f} ± {summary.get('hoi_err_std', 0.0):.4f}")
    
    print("="*80)
    
    # Also print per-metric sample counts
    print(f"\nMetric sample counts:")
    for key, values in metrics.items():
        print(f"  {key}: {len(values)} samples")


if __name__ == "__main__":
    main()


