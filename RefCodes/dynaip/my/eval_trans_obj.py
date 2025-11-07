import argparse
import os
from typing import Dict, Iterable, Iterator

import numpy as np
import torch
import utils.config as cfg
from pytorch3d import transforms

from my.dataset_trans_obj import MotionDatasetWithObjectAndTrans, _VEL_SELECTION_INDICES, _REDUCED_POSE_NAMES
from my.model_trans_obj import PoserWithObjectAndTrans


class FullSequenceLoader:
    """Iterate through prepared sequences without sliding-window sampling."""

    def __init__(self, dataset: MotionDatasetWithObjectAndTrans):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset.sequences)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for seq in self.dataset.sequences:
            yield self._sequence_to_batch(seq)

    @staticmethod
    def _sequence_to_batch(seq: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        batch = {
            "imu": seq["imu"].unsqueeze(0),
            "velocity": seq["velocity"].unsqueeze(0),
            "ori_glb_reduced": seq["ori_glb_reduced"].unsqueeze(0),
            "v_init": seq["velocity"][0].unsqueeze(0),
            "p_init": seq["ori_glb_reduced"][0].unsqueeze(0),
            "obj_imu": seq["obj_imu"].unsqueeze(0),
            "obj_vel": seq["obj_vel"].unsqueeze(0),
            "obj_pos": seq["obj_pos"].unsqueeze(0),
            "obj_v_init": seq["obj_vel"][0].unsqueeze(0),
            "obj_p_init": seq["obj_pos"][0].unsqueeze(0),
            "foot_contact": seq["foot_contact"].unsqueeze(0),
            "root_velocity": seq["root_velocity"].unsqueeze(0),
        }
        for key in ("lhand_contact", "rhand_contact", "obj_contact"):
            if key in seq:
                batch[key] = seq[key].unsqueeze(0)
        return {k: v.contiguous().float() for k, v in batch.items()}


@torch.no_grad()
def evaluate_trans_obj_model(
    model: PoserWithObjectAndTrans,
    data_loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    evaluate_objects: bool = True,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate ``PoserWithObjectAndTrans`` predictions and report core metrics.

    Returns:
        Dictionary with averaged MPJPE (cm), MPJRE (deg), object translation error (cm),
        HOI error (cm) and jitter (cm/frame^2). Metrics with no valid samples are ``nan``.
    """
    if model.body_model is None:
        raise RuntimeError(
            "Body model is required for evaluation. "
            "Instantiate PoserWithObjectAndTrans with a valid SMPL body model path."
        )
    if hasattr(data_loader, "__len__") and len(data_loader) == 0:
        raise ValueError("The provided data_loader is empty.")

    model.eval()
    fps = float(getattr(model, "fps", 30.0))

    metrics = {
        "mpjpe": [],
        "mpjre": [],
        "root_trans_err": [],
        "obj_trans_err": [],
        "hoi_err": [],
        "jitter": [],
    }

    def _compute_pose_and_joints(
        glb_pose_tensor: torch.Tensor, orient_tensor: torch.Tensor
    ):
        """Convert reduced 6D pose predictions into SMPL local rotations and joints."""
        device_tensor = glb_pose_tensor.device
        
        # 处理不同的输入形状
        if glb_pose_tensor.dim() == 4:
            # [B, T, 10, 6]
            B, T = glb_pose_tensor.shape[0], glb_pose_tensor.shape[1]
            BT = B * T
            pose6d = glb_pose_tensor.reshape(BT, len(_REDUCED_POSE_NAMES), 6)
        elif glb_pose_tensor.dim() == 3:
            # [B, T, 60]
            B, T = glb_pose_tensor.shape[0], glb_pose_tensor.shape[1]
            BT = B * T
            pose6d = glb_pose_tensor.reshape(BT, len(_REDUCED_POSE_NAMES), 6)
        else:
            raise ValueError(f"Unexpected glb_pose_tensor shape: {glb_pose_tensor.shape}")
        
        orient_bt = orient_tensor.reshape(BT, 6, 3, 3).clone()

        # 使用新的方法转换为全局姿态矩阵
        glb_pose_smpl = model._reduced_glb_6d_to_full_glb_mat(pose6d, orient_bt)
        local_pose_bt = model._global2local(glb_pose_smpl, model.smpl_parents.tolist())
        local_pose = local_pose_bt.view(B, T, 24, 3, 3)

        pose_aa = transforms.matrix_to_axis_angle(local_pose_bt.detach().cpu()).to(
            device_tensor
        )
        body_out = model.body_model(
            pose_body=pose_aa[:, 1:22].reshape(BT, 63),
            root_orient=pose_aa[:, 0].reshape(BT, 3),
            trans=torch.zeros(BT, 3, device=device_tensor, dtype=glb_pose_tensor.dtype),
        )
        joints = body_out.Jtr[:, :24, :].to(device_tensor).view(B, T, 24, 3)
        return local_pose, joints

    def _compute_root_translation(root_velocity: torch.Tensor) -> torch.Tensor:
        """Integrate root velocity (world, m/s) to translation (m)."""
        return torch.cumsum(root_velocity / fps, dim=1)

    def _compute_object_position(
        obj_vel: torch.Tensor, obj_pos_init: torch.Tensor
    ) -> torch.Tensor:
        """Integrate predicted object velocity to absolute position (m)."""
        disp = torch.cumsum(obj_vel * (1.0 / fps), dim=1)
        return obj_pos_init.unsqueeze(1) + disp

    def _compute_hoi_error(
        pred_obj_pos: torch.Tensor,
        gt_obj_pos: torch.Tensor,
        pred_joints: torch.Tensor,
        gt_joints: torch.Tensor,
        batch_dict: Dict[str, torch.Tensor],
    ) -> float:
        """Compute HOI error conditioned on available contact labels."""
        wrist_indices = [20, 21]  # left, right wrist in SMPL ordering
        pred_hands = pred_joints[:, :, wrist_indices, :]
        gt_hands = gt_joints[:, :, wrist_indices, :]

        rel_pred = pred_obj_pos.unsqueeze(2) - pred_hands
        rel_gt = gt_obj_pos.unsqueeze(2) - gt_hands
        diff = torch.linalg.norm(rel_pred - rel_gt, dim=-1)

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
            values = diff.reshape(-1)

        if values.numel() == 0:
            return float("nan")
        return values.mean().item() * 100.0

    total_batches = len(data_loader) if hasattr(data_loader, "__len__") else None
    for batch_idx, batch in enumerate(data_loader):
        batch = {key: value.to(device) for key, value in batch.items()}

        imu = batch["imu"]  # [B, T, 6, 12]
        B, T = imu.shape[0], imu.shape[1]
        orientation = imu[:, :, :, :9].contiguous().view(B, T, 6, 3, 3)

        obj_imu = batch.get(
            "obj_imu", torch.zeros(B, T, 12, device=device, dtype=imu.dtype)
        )
        obj_v_init = batch.get(
            "obj_v_init", torch.zeros(B, 3, device=device, dtype=imu.dtype)
        )

        _, glb_p_pred, obj_v_pred, _, _, _, root_trans_pred = model(
            imu,
            batch["v_init"],
            batch["p_init"],
            obj_imu,
            obj_v_init,
        )

        local_pose_pred, joints_pred_rel = _compute_pose_and_joints(
            glb_p_pred, orientation.clone()
        )
        local_pose_gt, joints_gt_rel = _compute_pose_and_joints(
            batch["ori_glb_reduced"], orientation.clone()
        )

        root_trans_gt = _compute_root_translation(batch["root_velocity"])
        pred_joints = joints_pred_rel + root_trans_pred.unsqueeze(2)
        gt_joints = joints_gt_rel + root_trans_gt.unsqueeze(2)

        # MPJPE: 归一化到各自根关节坐标系（去除全局位移影响）
        pred_eval_root_normalized = joints_pred_rel[:, :, :22, :]  # 已经是相对根关节
        gt_eval_root_normalized = joints_gt_rel[:, :, :22, :]      # 已经是相对根关节
        mpjpe_val = torch.linalg.norm(pred_eval_root_normalized - gt_eval_root_normalized, dim=-1).mean().item() * 100.0
        metrics["mpjpe"].append(mpjpe_val)

        pred_local = local_pose_pred[:, :, 1:22, :, :]
        gt_local = local_pose_gt[:, :, 1:22, :, :]
        rel_rot = torch.matmul(gt_local.transpose(-1, -2), pred_local)
        trace = torch.einsum("...ii->...", rel_rot)
        angles = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
        mpjre_val = angles.mean().item() * (180.0 / np.pi)
        metrics["mpjre"].append(mpjre_val)

        # 计算 root translation error (cm)
        root_trans_error = torch.linalg.norm(
            root_trans_pred - root_trans_gt, dim=-1
        ).mean().item() * 100.0
        metrics["root_trans_err"].append(root_trans_error)

        if evaluate_objects and "obj_pos" in batch:
            gt_obj_pos = batch["obj_pos"]
            pred_obj_pos = _compute_object_position(obj_v_pred, gt_obj_pos[:, 0, :])
            obj_err = (
                torch.linalg.norm(pred_obj_pos - gt_obj_pos, dim=-1).mean().item()
                * 100.0
            )
            metrics["obj_trans_err"].append(obj_err)
            hoi_err = _compute_hoi_error(
                pred_obj_pos, gt_obj_pos, pred_joints, gt_joints, batch
            )
            metrics["hoi_err"].append(hoi_err)
        else:
            metrics["obj_trans_err"].append(float("nan"))
            metrics["hoi_err"].append(float("nan"))

        # Jitter: 使用全局坐标系的关节位置（包含根位移）
        if T >= 3:
            pred_joints_eval = pred_joints[:, :, :22, :]  # [B, T, 22, 3]
            acc = (
                pred_joints_eval[:, 2:, :, :]
                - 2.0 * pred_joints_eval[:, 1:-1, :, :]
                + pred_joints_eval[:, :-2, :, :]
            )
            jitter_val = torch.linalg.norm(acc, dim=-1).mean().item() * 1000.0
            metrics["jitter"].append(jitter_val)
        else:
            metrics["jitter"].append(float("nan"))

        if verbose and total_batches is not None and (batch_idx + 1) % 10 == 0:
            print(f"[Eval] Processed {batch_idx + 1}/{total_batches} sequences")

    averaged = {}
    for key, values in metrics.items():
        valid = [v for v in values if not np.isnan(v)]
        averaged[key] = float(np.mean(valid)) if valid else float("nan")
    return averaged


def _load_checkpoint(
    model: PoserWithObjectAndTrans, checkpoint: str, device: torch.device
) -> None:
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Warning] Missing keys: {sorted(missing)}")
    if unexpected:
        print(f"[Warning] Unexpected keys: {sorted(unexpected)}")


def _build_sequence_loader(
    args: argparse.Namespace, device: torch.device
) -> FullSequenceLoader:
    dataset = MotionDatasetWithObjectAndTrans(
        datasets=args.datasets,
        seq_len=args.seq_len,
        data_root=args.data_root,
        device=device,
        subset=args.subset,
        random_sample=False,
        use_full_sequence=True,
        fps=args.fps,
        imu_noise_std=args.imu_noise,
    )
    if len(dataset.sequences) == 0:
        raise RuntimeError("No full sequences available for evaluation.")
    return FullSequenceLoader(dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PoserWithObjectAndTrans on prepared DynaIP data."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weights/trans_obj/best_val.pth",
        help="Path to model checkpoint (.pth).",
    )
    parser.add_argument(
        "--body_model_path",
        type=str,
        default="../../smpl_models/smplh/male/model.npz",
        help="Path to SMPL/SMPLH body model (e.g., body_models/smplh/male/model.npz).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default='../../',
        help="Root directory of the prepared DynaIP dataset.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=['processed_data_OMOMO'],
        help="Dataset subset names to evaluate (e.g., BEHAVE IMHD).",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="test",
        help="Data split to evaluate (default: test).",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=120,
        help="Sequence length used during preprocessing (not for slicing).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate used when preparing the dataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run evaluation on (e.g., cuda:0). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--no_object_metrics",
        action="store_true",
        help="Skip object translation and HOI evaluation.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable sequence-level progress messages.",
    )
    parser.add_argument(
        "--imu_noise",
        type=float,
        default=0.1,
        help="IMU Gaussian noise std for evaluation (default: 0.1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Eval] Using device: {device}")

    model = PoserWithObjectAndTrans(
        body_model_path=args.body_model_path,
        fps=args.fps,
    ).to(device)
    _load_checkpoint(model, args.checkpoint, device)

    data_loader = _build_sequence_loader(args, device)
    print(f"[Eval] Loaded {len(data_loader)} full sequences from subset='{args.subset}'.")
    print(f"[Eval] IMU noise std: {args.imu_noise}")

    metrics = evaluate_trans_obj_model(
        model,
        data_loader,
        device,
        evaluate_objects=not args.no_object_metrics,
        verbose=not args.quiet,
    )

    print("\n--- Evaluation Results ---")
    print(f"MPJPE (cm):             {metrics.get('mpjpe', float('nan')):.4f}")
    print(f"MPJRE (deg):            {metrics.get('mpjre', float('nan')):.4f}")
    print(f"Root Trans Error (cm):  {metrics.get('root_trans_err', float('nan')):.4f}")
    print(f"Jitter (mm/frame^2):    {metrics.get('jitter', float('nan')):.4f}")
    if not args.no_object_metrics:
        print(f"Obj Trans Error (cm):   {metrics.get('obj_trans_err', float('nan')):.4f}")
        print(f"HOI Error (cm):         {metrics.get('hoi_err', float('nan')):.4f}")
    else:
        print("Object metrics skipped.")


if __name__ == "__main__":
    main()

