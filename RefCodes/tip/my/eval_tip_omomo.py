import argparse
import os
from typing import Dict, List

import numpy as np
import torch
import pytorch3d.transforms as transforms
from human_body_prior.body_model.body_model import BodyModel

from my.dataset_omomo_tip import OMOMODatasetWithObject
from my.model_tip_with_object import TIPWithObject, TIPWithObjectConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TIP (with object) on OMOMO sequences.")
    parser.add_argument("--data_dirs", type=str, nargs="+", default='../../process/processed_data_OMOMO/test', help="Evaluation sequence directories.")
    parser.add_argument("--weights", type=str, default='output/tip_omomo_obj/best.pt', help="Checkpoint path (.pt).")
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--root_supervision", type=str, default="vel", choices=["vel", "pos"])
    parser.add_argument("--imu_noise_std", type=float, default=0.1, help="Gaussian IMU noise used at inference.")
    parser.add_argument("--eval_contacts", action="store_true", help="Load contact signals (for downstream use).")
    parser.add_argument(
        "--smplh_path",
        type=str,
        default='../../smpl_models/smplh/male/model.npz',
        help="Path to SMPLH body model.",
    )
    return parser.parse_args()


def _add_imu_noise(tensor: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return tensor
    return tensor + torch.randn_like(tensor) * std


def _integrate_velocity(vel: torch.Tensor, init: torch.Tensor, fps: float) -> torch.Tensor:
    if vel.ndim != 2 or vel.shape[-1] != 3:
        raise ValueError("Expected velocity tensor with shape [T, 3].")
    dt = 1.0 / float(fps)
    increments = vel * dt
    return torch.cumsum(increments, dim=0) + init.unsqueeze(0)


def _prepare_model(dataset: OMOMODatasetWithObject, args: argparse.Namespace, device: torch.device) -> TIPWithObject:
    cfg = TIPWithObjectConfig(
        num_imus_total=dataset.num_imus_total,
        state_dim=dataset.state_dim,
    )
    model = cfg.build().to(device)
    state_dict = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _prepare_body_model(args: argparse.Namespace, device: torch.device) -> BodyModel:
    bm = BodyModel(bm_fname=args.smplh_path, num_betas=16).to(device)
    bm.eval()
    return bm


def evaluate_sequences(
    model: TIPWithObject,
    dataset: OMOMODatasetWithObject,
    device: torch.device,
    args: argparse.Namespace,
    bm: BodyModel,
) -> Dict[str, List[float]]:
    metrics: Dict[str, List[float]] = {
        "mpjpe": [],
        "mpjre": [],
        "jitter": [],
        "obj_trans_err": [],
        "hoi_err": [],
    }

    human_root_slice = slice(dataset.human_out_rot_dim, dataset.human_out_rot_dim + dataset.root_pos_dim)
    tip_18_from_22_idx = dataset.tip_18_from_22_idx
    wrist_indices = [20, 21]  # left, right wrist in SMPL ordering

    for idx in range(len(dataset)):
        sample = dataset[idx]
        meta = sample.get("metadata", {})
        fps = float(meta.get("frame_rate", args.fps))

        imu = sample["imu"].to(device)
        state_hist = sample["state_hist"].to(device)
        target_state = sample["state_target"].to(device)

        imu_in = _add_imu_noise(imu, args.imu_noise_std)

        with torch.no_grad():
            pred_state = model.predict(imu_in, state_hist, keep_batch=False)

        obj_vel_pred = pred_state[:, -3:]
        root_pred = pred_state[:, human_root_slice]

        if bm is not None and "gt_motion" in sample and sample["gt_motion"] is not None:
            gt_motion = sample["gt_motion"].to(device)  # [L, 22*6]
            L = gt_motion.shape[0]
            motion22_gt = gt_motion.reshape(L, 22, 6)

            human_rot6d_pred = pred_state[:, : dataset.human_out_rot_dim].reshape(
                L, dataset.human_joint_num_for_output, 6
            )
            motion22_pred = motion22_gt.clone()
            for k, joint in enumerate(tip_18_from_22_idx):
                motion22_pred[:, joint, :] = human_rot6d_pred[:, k, :]

            root6d_pred = motion22_pred[:, 0, :]
            pose6d_pred = motion22_pred[:, 1:, :].reshape(L, 21, 6)
            root_axis_pred = transforms.matrix_to_axis_angle(transforms.rotation_6d_to_matrix(root6d_pred))
            pose_axis_pred = transforms.matrix_to_axis_angle(
                transforms.rotation_6d_to_matrix(pose6d_pred.reshape(-1, 6))
            ).reshape(L, 21 * 3)

            root6d_gt = motion22_gt[:, 0, :]
            pose6d_gt = motion22_gt[:, 1:, :].reshape(L, 21, 6)
            root_axis_gt = transforms.matrix_to_axis_angle(transforms.rotation_6d_to_matrix(root6d_gt))
            pose_axis_gt = transforms.matrix_to_axis_angle(
                transforms.rotation_6d_to_matrix(pose6d_gt.reshape(-1, 6))
            ).reshape(L, 21 * 3)

            if dataset.root_supervision == "vel":
                root_pos_pred = _integrate_velocity(root_pred, torch.zeros(3, device=device), fps)
            else:
                root_pos_pred = root_pred

            pred_out = bm(root_orient=root_axis_pred, pose_body=pose_axis_pred, trans=root_pos_pred)
            gt_out = bm(
                root_orient=root_axis_gt,
                pose_body=pose_axis_gt,
                trans=torch.zeros_like(root_pos_pred),
            )

            pred_joints = pred_out.Jtr[:, :22, :]
            gt_joints = gt_out.Jtr[:, :22, :]

            # MPJPE (cm) - 归一化到各自根关节坐标系
            mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1).mean().item() * 100.0
            metrics["mpjpe"].append(mpjpe)

            # MPJRE (deg) - 使用6D表示的绝对差异（与eval_obj.py一致）
            pred_body_6d = motion22_pred[:, 1:22, :].reshape(L, 21, 6)
            gt_body_6d = motion22_gt[:, 1:22, :].reshape(L, 21, 6)
            rot_error = torch.mean(torch.abs(gt_body_6d - pred_body_6d)) * 57.2958  # rad to deg
            metrics["mpjre"].append(rot_error.item())

            # Jitter (mm/frame^2)
            if L >= 3:
                acc = pred_joints[2:] - 2 * pred_joints[1:-1] + pred_joints[:-2]
                metrics["jitter"].append(torch.linalg.norm(acc, dim=-1).mean().item() * 1000.0)
            
            # Object translation error (cm) & HOI Error (cm)
            if "obj_pos_gt" in sample and sample["obj_pos_gt"] is not None:
                obj_pos_gt = sample["obj_pos_gt"].to(device)
                obj_pos_init = sample["obj_pos_init"].to(device)
                obj_pos_pred = _integrate_velocity(obj_vel_pred, obj_pos_init, fps)
                
                # Object translation error (cm)
                obj_err = torch.linalg.norm(obj_pos_pred - obj_pos_gt, dim=-1).mean().item() * 100.0
                metrics["obj_trans_err"].append(obj_err)
                
                # HOI Error (cm) - 手腕与物体的相对位置误差
                pred_hands = pred_joints[:, wrist_indices, :]  # [L, 2, 3]
                gt_hands = gt_joints[:, wrist_indices, :]  # [L, 2, 3]
                
                rel_pred = obj_pos_pred.unsqueeze(1) - pred_hands  # [L, 2, 3]
                rel_gt = obj_pos_gt.unsqueeze(1) - gt_hands  # [L, 2, 3]
                diff = torch.linalg.norm(rel_pred - rel_gt, dim=-1)  # [L, 2]
                
                # 如果有手部接触标签，只计算接触时的误差
                collected = []
                if "lhand_contact" in sample and sample["lhand_contact"] is not None:
                    lhand_contact = sample["lhand_contact"].to(device)
                    l_mask = lhand_contact.bool()
                    if l_mask.any():
                        collected.append(diff[:, 0][l_mask])
                if "rhand_contact" in sample and sample["rhand_contact"] is not None:
                    rhand_contact = sample["rhand_contact"].to(device)
                    r_mask = rhand_contact.bool()
                    if r_mask.any():
                        collected.append(diff[:, 1][r_mask])
                
                if collected:
                    hoi_err = torch.cat(collected).mean().item() * 100.0
                else:
                    # 如果没有接触标签，计算所有帧的平均误差
                    hoi_err = diff.mean().item() * 100.0
                
                metrics["hoi_err"].append(hoi_err)

    return metrics


def summarize_metrics(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for key, values in metrics.items():
        if values:
            summary[key] = float(np.mean(values))
    return summary


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    dataset = OMOMODatasetWithObject(
        data_dirs=args.data_dirs,
        seq_len=args.seq_len,
        frame_rate=args.fps,
        use_object_imu=True,
        root_supervision=args.root_supervision,
        return_gt_pos=True,
        return_gt_motion=True,
        return_contacts=args.eval_contacts,
        use_full_sequence=True,
        random_sample=False,
    )

    if len(dataset) == 0:
        raise RuntimeError("No sequences found for evaluation.")

    model = _prepare_model(dataset, args, device)
    bm = _prepare_body_model(args, device)

    metrics = evaluate_sequences(model, dataset, device, args, bm)
    summary = summarize_metrics(metrics)

    print("\n--- Evaluation Results ---")
    if "mpjpe" in summary:
        print(f"MPJPE (cm):             {summary['mpjpe']:.4f}")
    if "mpjre" in summary:
        print(f"MPJRE (deg):            {summary['mpjre']:.4f}")
    if "jitter" in summary:
        print(f"Jitter (mm/frame^2):    {summary['jitter']:.4f}")
    if "obj_trans_err" in summary:
        print(f"Obj Trans Error (cm):   {summary['obj_trans_err']:.4f}")
    if "hoi_err" in summary:
        print(f"HOI Error (cm):         {summary['hoi_err']:.4f}")


if __name__ == "__main__":
    main()
