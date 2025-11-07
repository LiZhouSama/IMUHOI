import argparse
import os
from typing import Dict, Iterable, Iterator

import numpy as np
import torch
from pytorch3d import transforms

from my_simple.dataset_obj import MotionDatasetWithObjectAndTrans
from my_simple.model_obj import PoserWithObject


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
            "pose": seq["pose"].unsqueeze(0),
            "obj_imu": seq["obj_imu"].unsqueeze(0),
            "obj_vel": seq["obj_vel"].unsqueeze(0),
            "obj_pos": seq["obj_pos"].unsqueeze(0),
        }
        return {k: v.contiguous().float() for k, v in batch.items()}


@torch.no_grad()
def evaluate_obj_model(
    model: PoserWithObject,
    data_loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    evaluate_objects: bool = True,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate ``PoserWithObject`` predictions and report core metrics.

    Returns:
        Dictionary with averaged MPJPE (cm), MPJRE (deg), object translation error (cm),
        HOI error (cm) and jitter (cm/frame^2). Metrics with no valid samples are ``nan``.
    """
    if hasattr(data_loader, "__len__") and len(data_loader) == 0:
        raise ValueError("The provided data_loader is empty.")

    model.eval()
    fps = float(getattr(model, "fps", 30.0))

    metrics = {
        "mpjpe": [],
        "mpjre": [],
        "obj_trans_err": [],
        "hoi_err": [],
        "jitter": [],
    }

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
        pose_gt_aa = transforms.rotation_6d_to_matrix(batch["pose"].reshape(B, T, 22, 6))
        pose_gt_aa = transforms.matrix_to_axis_angle(pose_gt_aa)
        
        obj_imu = batch.get(
            "obj_imu", torch.zeros(B, T, 12, device=device, dtype=imu.dtype)
        )
        p_pred, obj_v_pred = model(imu,obj_imu)
        pose_pred_aa = transforms.matrix_to_axis_angle(p_pred.reshape(B, T, 22, 3, 3))


        body_out_gt = model.body_model(
            pose_body=pose_gt_aa[:, :, 1:22].reshape(B*T, 63),
            root_orient=pose_gt_aa[:, :, 0].reshape(B*T, 3),
        )
        joints_gt = body_out_gt.Jtr[:, :24, :].to(device).view(B, T, 24, 3)

        body_out_pred = model.body_model(
            pose_body=pose_pred_aa[:, :, 1:22].reshape(B*T, 63),
            root_orient=pose_pred_aa[:, :, 0].reshape(B*T, 3),
        )
        joints_pred = body_out_pred.Jtr[:, :24, :].to(device).view(B, T, 24, 3)

        # MPJPE: 归一化到各自根关节坐标系（去除全局位移影响）
        mpjpe_val = torch.linalg.norm(joints_pred - joints_gt, dim=-1).mean().item() * 100.0
        metrics["mpjpe"].append(mpjpe_val)

        pred_body_6d = transforms.matrix_to_rotation_6d(p_pred.reshape(-1, 3, 3)).reshape(B, T, -1, 6)
        gt_body_6d = batch["pose"].reshape(B, T, -1, 6)
        rot_error_ = torch.mean(torch.absolute(gt_body_6d-pred_body_6d)) * 57.2958
        metrics["mpjre"].append(rot_error_.item())

        if evaluate_objects and "obj_pos" in batch:
            gt_obj_pos = batch["obj_pos"]
            pred_obj_pos = _compute_object_position(obj_v_pred, gt_obj_pos[:, 0, :])
            obj_err = (
                torch.linalg.norm(pred_obj_pos - gt_obj_pos, dim=-1).mean().item()
                * 100.0
            )
            metrics["obj_trans_err"].append(obj_err)
            hoi_err = _compute_hoi_error(
                pred_obj_pos, gt_obj_pos, joints_pred, joints_gt, batch
            )
            metrics["hoi_err"].append(hoi_err)
        else:
            metrics["obj_trans_err"].append(float("nan"))
            metrics["hoi_err"].append(float("nan"))

        # Jitter: 使用全局坐标系的关节位置（包含根位移）
        if T >= 3:
            pred_joints_eval = joints_pred[:, :, :22, :]  # [B, T, 22, 3]
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
    model: PoserWithObject, checkpoint: str, device: torch.device
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
        default="checkpoints/dip_obj_omomo/best_val.pth",
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
        default='../../process',
        help="Root directory of the prepared DynaIP dataset.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=['processed_split_data_OMOMO'],
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
        default=0,
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

    model = PoserWithObject(
        body_model_path=args.body_model_path,
        fps=args.fps,
    ).to(device)
    _load_checkpoint(model, args.checkpoint, device)

    data_loader = _build_sequence_loader(args, device)
    print(f"[Eval] Loaded {len(data_loader)} full sequences from subset='{args.subset}'.")
    print(f"[Eval] IMU noise std: {args.imu_noise}")

    metrics = evaluate_obj_model(
        model,
        data_loader,
        device,
        evaluate_objects=not args.no_object_metrics,
        verbose=not args.quiet,
    )

    print("\n--- Evaluation Results ---")
    print(f"MPJPE (cm):             {metrics.get('mpjpe', float('nan')):.4f}")
    print(f"MPJRE (deg):            {metrics.get('mpjre', float('nan')):.4f}")
    print(f"Jitter (mm/frame^2):    {metrics.get('jitter', float('nan')):.4f}")
    if not args.no_object_metrics:
        print(f"Obj Trans Error (cm):   {metrics.get('obj_trans_err', float('nan')):.4f}")
        print(f"HOI Error (cm):         {metrics.get('hoi_err', float('nan')):.4f}")
    else:
        print("Object metrics skipped.")


if __name__ == "__main__":
    main()

