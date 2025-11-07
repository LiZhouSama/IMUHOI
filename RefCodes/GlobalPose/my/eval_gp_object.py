"""
Evaluation script for GPNet with object tracking (GT root translation).
Root translation is taken from ground truth rather than model estimates.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import articulate as art

# Ensure local package imports work when run from CLI
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from my.omomo_gp_dataset import OMOMOGlobalPoseDataset  # noqa: E402
from my.models.gpnet_object import GPNetWithObject  # noqa: E402


# Joint indices reused from dataset definition
J_REDUCE = (1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19)
WRIST_INDICES = (20, 21)

SCRIPT_DIR = Path(__file__).resolve().parent
GLOBALPOSE_DIR = Path(ROOT_DIR)
REFCODES_DIR = GLOBALPOSE_DIR.parent
PROJECT_ROOT = REFCODES_DIR.parent

DEFAULT_DATASET_CONFIG: Dict[str, Dict[str, Path]] = {
    "processed_seg_data_IMHD": {
        "data_dir": PROJECT_ROOT / "process" / "processed_seg_data_IMHD" / "test",
        "checkpoint": SCRIPT_DIR / "checkpoints" / "gp_object_IMHD" / "best.pt",
    },
    "processed_seg_data_BEHAVE": {
        "data_dir": PROJECT_ROOT / "process" / "processed_seg_data_BEHAVE" / "test",
        "checkpoint": SCRIPT_DIR / "checkpoints" / "gp_object_BEHAVE" / "best.pt",
    },
    "processed_split_data_OMOMO": {
        "data_dir": PROJECT_ROOT / "process" / "processed_split_data_OMOMO" / "test",
        "checkpoint": SCRIPT_DIR / "checkpoints" / "gp_object_OMOMO" / "best.pt",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GPNet with object tracking on OMOMO dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=sorted(DEFAULT_DATASET_CONFIG.keys()),
        help="Evaluate only the specified dataset key. Default: run all configured datasets sequentially.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path. Defaults to dataset-specific configuration.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Override dataset directory or file. Defaults to dataset-specific configuration.",
    )
    parser.add_argument(
        "--sequence_len",
        type=int,
        default=0,
        help="Split size for sequences. Use 0 for full sequences (recommended).",
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=60,
        help="Minimum sequence length for evaluation filtering.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Dataset frame rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device (e.g., cuda:0). Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Limit number of sequences for quick evaluation.",
    )
    parser.add_argument(
        "--no_object_metrics",
        action="store_true",
        help="Skip object translation and HOI metrics.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-sequence progress logs.",
    )
    return parser.parse_args()


def _select_path(override: Optional[str], default_path: Path) -> Path:
    path = Path(override).expanduser() if override else default_path
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path


def _build_dataset_runs(args: argparse.Namespace) -> List[Tuple[str, Dict[str, Path]]]:
    runs: List[Tuple[str, Dict[str, Path]]] = []
    if args.dataset:
        runs.append((args.dataset, dict(DEFAULT_DATASET_CONFIG[args.dataset])))
        return runs
    if args.data_file or args.checkpoint:
        if not args.data_file or not args.checkpoint:
            raise ValueError(
                "Both --data_file and --checkpoint must be provided when overriding without --dataset."
            )
        runs.append(
            (
                "custom",
                {
                    "data_dir": Path(args.data_file).expanduser(),
                    "checkpoint": Path(args.checkpoint).expanduser(),
                },
            )
        )
        return runs
    for name, cfg in DEFAULT_DATASET_CONFIG.items():
        runs.append((name, dict(cfg)))
    return runs


def _load_checkpoint(model: GPNetWithObject, checkpoint: Path, device: torch.device) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Eval] Warning: missing keys in checkpoint: {sorted(missing)}")
    if unexpected:
        print(f"[Eval] Warning: unexpected keys in checkpoint: {sorted(unexpected)}")


def _integrate_velocity(
    velocity: torch.Tensor, init_pos: torch.Tensor, dt: float
) -> torch.Tensor:
    """Integrate velocity (T, 3) to positions with known starting point."""
    T = velocity.shape[0]
    pos = torch.zeros_like(velocity)
    pos[0] = init_pos
    for t in range(1, T):
        pos[t] = pos[t - 1] + velocity[t - 1] * dt
    return pos


def _build_object_input(
    human_pred: torch.Tensor, obj_data: Dict[str, torch.Tensor], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct object VR inputs and targets given human predictions.
    Returns (input_seq[T, 252], vel_target[T, 3]).
    """
    T = human_pred.shape[0]
    x_vr = torch.zeros(T, 243, device=device, dtype=human_pred.dtype)

    rr_pred = human_pred[:, 90:180]
    ik_pred = human_pred[:, 18:90]

    x_vr[:, :90] = rr_pred  # reduced rotations
    x_vr[:, 135:204] = ik_pred[:, :69]  # joint positions
    x_vr[:, 240:243] = ik_pred[:, 69:72]  # gravity direction

    obj_imu = obj_data["obj_imu"].to(device, dtype=human_pred.dtype)
    obj_input = torch.cat([x_vr, obj_imu], dim=-1)
    obj_vel_root = obj_data["obj_vel_root"].to(device, dtype=human_pred.dtype)
    return obj_input, obj_vel_root


@torch.no_grad()
def evaluate_model(
    model: GPNetWithObject,
    dataset: OMOMOGlobalPoseDataset,
    body_model: art.ParametricModel,
    device: torch.device,
    fps: float,
    evaluate_object: bool,
    verbose: bool,
    max_sequences: Optional[int] = None,
) -> Dict[str, float]:
    dt = 1.0 / fps
    metrics: Dict[str, List[float]] = {
        "mpjpe": [],
        "mpjre": [],
        "root_trans_err": [],
        "obj_trans_err": [],
        "hoi_err": [],
        "jitter": [],
    }

    total = len(dataset) if max_sequences is None else min(len(dataset), max_sequences)
    for idx in range(total):
        try:
            sample = dataset[idx]
            if isinstance(sample, tuple) and len(sample) == 3:
                (seq_input, init_vec), _, seq_idx = sample
            else:
                (seq_input, init_vec), _ = sample
                seq_idx = idx

            seq_input = seq_input.to(device=device, dtype=torch.float32)
            init_vec = init_vec.to(device=device, dtype=torch.float32)
            dtype = seq_input.dtype

            T = seq_input.shape[0]
            if T == 0:
                continue

            meta = dataset.get_sequence_meta(seq_idx)
            tran_gt = meta["tran"].to(device=device, dtype=dtype)
            root_rot_gt = meta["root_rot"].to(device=device, dtype=dtype)
            pose_gt = meta["pose"].to(device=device, dtype=dtype)

            root_meta = [{'tran': tran_gt, 'root_rot': root_rot_gt}]
            human_pred_list, _ = model([(seq_input, init_vec)], x_object=None, fast=True, root_meta=root_meta)
            human_stage_pred = human_pred_list[0]

            aRB = seq_input[:, :18].view(T, 6, 3)
            wRB = seq_input[:, 18:36].view(T, 6, 3)
            RRB = seq_input[:, 36:81].view(T, 5, 3, 3)

            root_rot = root_rot_gt
            root_rot_t = root_rot.transpose(1, 2)
            aM = torch.einsum("tsc,tcd->tsd", aRB, root_rot_t)
            wM = torch.einsum("tsc,tcd->tsd", wRB, root_rot_t)

            RMB = torch.zeros(T, 6, 3, 3, device=device, dtype=dtype)
            RMB[:, :5] = torch.matmul(root_rot.unsqueeze(1), RRB)
            RMB[:, 5] = root_rot

            model.human.rnn_initialize(pose_gt[0])
            pose_frames: List[torch.Tensor] = []
            for t in range(T):
                pose_t, _ = model.human.forward_frame(aM[t], wM[t], RMB[t])
                pose_frames.append(pose_t.float())

            pose_pred = torch.stack(pose_frames, dim=0).to(device=device, dtype=dtype)
            tran_pred = tran_gt.clone()

            pose_gt_seq = pose_gt
            pred_sel = pose_pred[:, 1:23].reshape(-1, 3, 3)
            gt_sel = pose_gt_seq[:, 1:23].reshape(-1, 3, 3)
            # pred_r6d = art.math.rotation_matrix_to_r6d(pred_sel)
            # gt_r6d = art.math.rotation_matrix_to_r6d(gt_sel)
            try:
                pred_aa = art.math.rotation_matrix_to_axis_angle(pred_sel)  
                gt_aa = art.math.rotation_matrix_to_axis_angle(gt_sel)  
                mpjre = torch.mean(torch.absolute(pred_aa-gt_aa)) * 57.2958
                metrics["mpjre"].append(mpjre.item())
            except Exception as e:
                if verbose:
                    print(f"[Eval] Warning: MPJRE calculation failed for sequence {seq_idx}: {e}")
                metrics["mpjre"].append(float("nan"))

            try:
                root_trans_err = torch.linalg.norm(tran_pred - tran_gt, dim=-1).mean().item() * 100.0
                metrics["root_trans_err"].append(root_trans_err)
            except Exception as e:
                if verbose:
                    print(f"[Eval] Warning: Root trans error calculation failed for sequence {seq_idx}: {e}")
                metrics["root_trans_err"].append(float("nan"))

            pose_pred_cpu = pose_pred.cpu()
            tran_pred_cpu = tran_pred.cpu()
            pose_gt_cpu = pose_gt_seq.cpu()
            tran_gt_cpu = tran_gt.cpu()
            _, joints_pred= body_model.forward_kinematics(
                pose_pred_cpu, tran=tran_pred_cpu, calc_mesh=False
            )
            _, joints_gt= body_model.forward_kinematics(
                pose_gt_cpu, tran=tran_gt_cpu, calc_mesh=False
            )
            joints_pred = joints_pred.to(device=device, dtype=dtype)
            joints_gt = joints_gt.to(device=device, dtype=dtype)

            joints_pred_rel = joints_pred[:, 1:24, :] - joints_pred[:, :1, :]
            joints_gt_rel = joints_gt[:, 1:24, :] - joints_gt[:, :1, :]
            try:
                mpjpe = torch.linalg.norm(joints_pred_rel - joints_gt_rel, dim=-1).mean().item() * 100.0
                metrics["mpjpe"].append(mpjpe)
            except Exception as e:
                if verbose:
                    print(f"[Eval] Warning: MPJPE calculation failed for sequence {seq_idx}: {e}")
                metrics["mpjpe"].append(float("nan"))

            obj_data = dataset.get_object_data(seq_idx)
            obj_trans_pred = None
            if evaluate_object and obj_data is not None:
                try:
                    obj_input, obj_vel_target = _build_object_input(human_stage_pred, obj_data, device)
                    if obj_input is not None and obj_vel_target is not None:
                        init_obj = torch.zeros(3, device=device, dtype=obj_input.dtype)
                        obj_pred_list = model.object_vr([(obj_input, init_obj)])
                        obj_vel_root_pred = obj_pred_list[0]

                        root_rot_pred = pose_pred[:, 0]
                        obj_vel_world = torch.matmul(
                            root_rot_pred, obj_vel_root_pred.unsqueeze(-1)
                        ).squeeze(-1)
                        obj_trans_gt = obj_data["obj_trans"].to(device=device, dtype=dtype)
                        obj_offset = obj_trans_gt[0]
                        obj_trans_pred = (
                            _integrate_velocity(obj_vel_world, torch.zeros_like(obj_offset), dt)
                            + obj_offset
                        )

                        obj_err = (
                            torch.linalg.norm(obj_trans_pred - obj_trans_gt, dim=-1).mean().item() * 100.0
                        )
                        metrics["obj_trans_err"].append(obj_err)

                        wrist_pred = joints_pred[:, WRIST_INDICES, :]
                        wrist_gt = joints_gt[:, WRIST_INDICES, :]
                        pred_dist = torch.linalg.norm(
                            wrist_pred - obj_trans_pred.unsqueeze(1), dim=-1
                        )
                        gt_dist = torch.linalg.norm(
                            wrist_gt - obj_trans_gt.unsqueeze(1), dim=-1
                        )
                        hoi_err = (pred_dist - gt_dist).abs().mean().item() * 100.0
                        metrics["hoi_err"].append(hoi_err)
                    else:
                        metrics["obj_trans_err"].append(float("nan"))
                        metrics["hoi_err"].append(float("nan"))
                except Exception as e:
                    if verbose:
                        print(f"[Eval] Warning: Object metrics calculation failed for sequence {seq_idx}: {e}")
                    metrics["obj_trans_err"].append(float("nan"))
                    metrics["hoi_err"].append(float("nan"))
            else:
                metrics["obj_trans_err"].append(float("nan"))
                metrics["hoi_err"].append(float("nan"))

            try:
                if joints_pred_rel.shape[0] >= 3:
                    acc = (
                        joints_pred_rel[2:, :22, :]
                        - 2.0 * joints_pred_rel[1:-1, :22, :]
                        + joints_pred_rel[:-2, :22, :]
                    )
                    jitter = torch.linalg.norm(acc, dim=-1).mean().item() * 1000.0
                    metrics["jitter"].append(jitter)
                else:
                    metrics["jitter"].append(float("nan"))
            except Exception as e:
                if verbose:
                    print(f"[Eval] Warning: Jitter calculation failed for sequence {seq_idx}: {e}")
                metrics["jitter"].append(float("nan"))
        except Exception as e:
            if verbose:
                print(f"[Eval] Error processing sequence {idx}: {e}")
            continue


    averaged: Dict[str, float] = {}
    for key, values in metrics.items():
        valid = [float(v) for v in values if not math.isnan(v)]
        averaged[key] = np.mean(valid) if valid else float("nan")
    return averaged


def main() -> None:
    args = parse_args()
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Eval] Using device: {device}")
    try:
        dataset_runs = _build_dataset_runs(args)
    except ValueError as exc:
        print(f"[Eval] Argument error: {exc}")
        return

    all_stats: Dict[str, Dict[str, float]] = {}
    for dataset_name, dataset_cfg in dataset_runs:
        dataset_override = args.data_file if args.dataset else None
        checkpoint_override = args.checkpoint if args.dataset else None

        data_path = _select_path(dataset_override, dataset_cfg["data_dir"])
        checkpoint_path = _select_path(checkpoint_override, dataset_cfg["checkpoint"])

        print(f"\n=== Evaluating dataset: {dataset_name} ===")
        print(f"[Eval] Data path: {data_path}")
        print(f"[Eval] Checkpoint: {checkpoint_path}")

        if not data_path.exists():
            print(f"[Eval] Skipping '{dataset_name}' because data path does not exist.")
            continue
        if not checkpoint_path.exists():
            print(f"[Eval] Skipping '{dataset_name}' because checkpoint does not exist.")
            continue

        model = GPNetWithObject(pretrained_path="", dt=1.0 / args.fps).to(device)
        _load_checkpoint(model, checkpoint_path, device)
        model.eval()

        dataset = OMOMOGlobalPoseDataset(
            str(data_path),
            sequence_len=args.sequence_len,
            drop_last=False,
            min_seq_len=args.min_seq_len,
            fps=int(args.fps),
            device=torch.device("cpu"),
        )
        print(f"[Eval] Loaded {len(dataset)} sequences from '{data_path}'.")

        stats = evaluate_model(
            model,
            dataset,
            dataset.body_model,
            device=device,
            fps=args.fps,
            evaluate_object=not args.no_object_metrics,
            verbose=not args.quiet,
            max_sequences=args.max_sequences,
        )

        print("\n--- Evaluation Results ---")
        def _format_metric(value: float, name: str) -> str:
            if math.isnan(value):
                return f"{name}: {'N/A (no valid samples)':>25}"
            return f"{name}: {value:>25.4f}"
        
        print(_format_metric(stats['mpjpe'], "MPJPE (cm)"))
        print(_format_metric(stats['mpjre'], "MPJRE (deg)"))
        print(_format_metric(stats['root_trans_err'], "Root Trans Error (cm)"))
        print(_format_metric(stats['jitter'], "Jitter (mm/frame^2)"))
        if not args.no_object_metrics:
            print(_format_metric(stats['obj_trans_err'], "Obj Trans Error (cm)"))
            print(_format_metric(stats['hoi_err'], "HOI Error (cm)"))
        else:
            print("Object metrics skipped.")

        all_stats[dataset_name] = stats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
