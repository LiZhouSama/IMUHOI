import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from my.omomo_gp_dataset import OMOMOGlobalPoseDataset
from my.models.gpnet_object import GPNetWithObject


SCRIPT_DIR = Path(__file__).resolve().parent
GLOBALPOSE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = GLOBALPOSE_DIR

DEFAULT_DATASET_CONFIG: Dict[str, Path] = {
    "processed_seg_data_IMHD": SCRIPT_DIR / "checkpoints" / "gp_object_IMHD" / "best.pt",
    "processed_seg_data_BEHAVE": SCRIPT_DIR / "checkpoints" / "gp_object_BEHAVE" / "best.pt",
    "processed_split_data_OMOMO": SCRIPT_DIR / "checkpoints" / "gp_object_OMOMO" / "best.pt",
}

WRIST_INDICES = (20, 21)


def _resolve_path(base: Path, path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (base / path_str).resolve()
    return path


def _load_checkpoint(model: GPNetWithObject, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Warn] Missing keys in checkpoint {checkpoint_path.name}: {sorted(missing)}")
    if unexpected:
        print(f"[Warn] Unexpected keys in checkpoint {checkpoint_path.name}: {sorted(unexpected)}")


def _build_sequence_name(dataset: OMOMOGlobalPoseDataset, seq_idx: int, meta: Dict) -> str:
    name = meta.get("name")
    if name and isinstance(name, str) and name.strip() and name != "unknown":
        return name
    if seq_idx < len(dataset.sequence_sources):
        source_info = dataset.sequence_sources[seq_idx]
        source_path = source_info[0]
        subseq_idx: Optional[int] = None
        if len(source_info) > 1 and isinstance(source_info[1], (int, torch.Tensor)):
            subseq_idx = int(source_info[1])
        base = Path(source_path).stem
        if subseq_idx is None or (subseq_idx == 0 and "_seg" in base):
            return base
        return f"{base}_{subseq_idx:03d}"
    return f"sequence_{seq_idx:05d}"


def generate_dataset_predictions(
    dataset_name: str,
    checkpoint: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, Dict[str, List[List[float]]]], int]:
    data_dir = _resolve_path(PROJECT_ROOT, os.path.join(args.data_root, dataset_name, args.subset))
    if not data_dir.exists():
        print(f"[WARN] Data directory missing for {dataset_name}: {data_dir}")
        return {}, 0

    dataset = OMOMOGlobalPoseDataset(
        str(data_dir),
        sequence_len=0,
        drop_last=False,
        min_seq_len=args.min_seq_len,
        fps=int(args.fps),
        device=torch.device("cpu"),
    )
    if len(dataset) == 0:
        print(f"[WARN] Dataset '{dataset_name}' has no valid sequences.")
        return {}, 0

    model = GPNetWithObject(pretrained_path="", dt=1.0 / args.fps).to(device)
    _load_checkpoint(model, checkpoint, device)
    model.eval()

    body_model = dataset.body_model
    predictions: Dict[str, Dict[str, List[List[float]]]] = {}

    total = len(dataset)
    for sample_idx in range(total):
        sample = dataset[sample_idx]
        if isinstance(sample, tuple) and len(sample) == 3:
            (seq_input, init_vec), _, seq_idx = sample
        else:
            (seq_input, init_vec), _ = sample
            seq_idx = sample_idx

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

        root_meta = [{"tran": tran_gt, "root_rot": root_rot_gt}]
        human_pred_list, _ = model([(seq_input, init_vec)], x_object=None, fast=True, root_meta=root_meta)
        human_stage_pred = human_pred_list[0]

        aRB = seq_input[:, :18].view(T, 6, 3)
        wRB = seq_input[:, 18:36].view(T, 6, 3)
        RRB = seq_input[:, 36:81].view(T, 5, 3, 3)

        root_rot = root_rot_gt
        root_rot_t = root_rot.transpose(1, 2)
        aM = torch.einsum("tsc, tcd -> tsd", aRB, root_rot_t)
        wM = torch.einsum("tsc, tcd -> tsd", wRB, root_rot_t)

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

        pose_pred_cpu = pose_pred.cpu()
        tran_pred_cpu = tran_pred.cpu()

        _, joints_pred = body_model.forward_kinematics(pose_pred_cpu, tran=tran_pred_cpu, calc_mesh=False)
        joints_pred = joints_pred.to(device=device, dtype=dtype)

        hand_positions = joints_pred[:, WRIST_INDICES, :].detach().cpu().tolist()

        seq_name = _build_sequence_name(dataset, seq_idx, meta)
        predictions[seq_name] = {
            "pred_hand_glb_pos": hand_positions,
        }

    return predictions, len(predictions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GlobalPose hand position JSON per dataset.")
    parser.add_argument(
        "--data_root",
        type=str,
        default="../../process",
        help="Root directory containing processed datasets.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="my/globalpose_hand_predictions",
        help="Directory to store generated JSON files.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="test",
        help="Dataset subset to process.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate used for the datasets.",
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=60,
        help="Minimum sequence length to keep.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device identifier (e.g., cuda:0). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASET_CONFIG.keys()),
        help="Dataset names to process.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Optional list of checkpoint paths matching --datasets order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Info] Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.checkpoints is not None and len(args.checkpoints) != len(args.datasets):
        raise ValueError("When specifying --checkpoints, its length must match --datasets.")

    dataset_ckpt_pairs: List[Tuple[str, Path]] = []
    if args.checkpoints is None:
        for dataset in args.datasets:
            if dataset not in DEFAULT_DATASET_CONFIG:
                raise KeyError(f"No default checkpoint for dataset '{dataset}'. Provide via --checkpoints.")
            dataset_ckpt_pairs.append((dataset, DEFAULT_DATASET_CONFIG[dataset]))
    else:
        dataset_ckpt_pairs = [
            (dset, _resolve_path(SCRIPT_DIR, ckpt))
            for dset, ckpt in zip(args.datasets, args.checkpoints)
        ]

    for dataset_name, checkpoint in dataset_ckpt_pairs:
        print(f"\n[Info] Processing dataset '{dataset_name}'")
        predictions, seq_count = generate_dataset_predictions(
            dataset_name,
            checkpoint,
            args,
            device,
        )
        if not predictions:
            print(f"[Info] No predictions generated for '{dataset_name}'.")
            continue

        output_path = Path(args.output_dir) / f"{dataset_name}_hand_predictions.json"
        payload = {
            "dataset": dataset_name,
            "checkpoint": str(checkpoint),
            "subset": args.subset,
            "fps": args.fps,
            "num_sequences": seq_count,
            "predictions": predictions,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        print(f"[Info] Saved {seq_count} sequences to {output_path}")


if __name__ == "__main__":
    main()
