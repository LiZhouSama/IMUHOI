import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
from pytorch3d import transforms

from my.dataset_trans_obj import (
    MotionDatasetWithObjectAndTrans,
    _REDUCED_POSE_NAMES,
)
from my.eval_obj import FullSequenceLoader
from my.model_obj import PoserWithObject


DEFAULT_DATASET_CONFIG: Dict[str, str] = {
    "processed_seg_data_BEHAVE": "checkpoints/dynaip_obj_behave/best_val.pth",
    "processed_seg_data_IMHD": "checkpoints/dynaip_obj_imhd/best_val.pth",
    "processed_split_data_OMOMO": "checkpoints/dynaip_obj_omomo/best_val.pth",
}


def _load_checkpoint(model: PoserWithObject, checkpoint: str, device: torch.device) -> None:
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


def _ensure_body_model_device(model: PoserWithObject, device: torch.device) -> None:
    if model.body_model is None:
        raise RuntimeError("Body model is required to generate hand positions. "
                           "Instantiate PoserWithObject with a valid body_model_path.")
    if model.body_model_device != device:
        model.body_model = model.body_model.to(device)
        model.body_model_device = device


def _compute_joints(
    model: PoserWithObject,
    pose_pred: torch.Tensor,
    orientation: torch.Tensor,
) -> torch.Tensor:
    """
    Convert reduced 6D pose predictions to 24 SMPL joint positions.
    pose_pred: [B, T, len(_REDUCED_POSE_NAMES) * 6]
    orientation: [B, T, 6, 3, 3]
    Returns: joints [B, T, 24, 3]
    """
    device = pose_pred.device
    B, T = pose_pred.shape[0], pose_pred.shape[1]
    BT = B * T

    pose6d = pose_pred.reshape(BT, len(_REDUCED_POSE_NAMES), 6)
    orient_bt = orientation.reshape(BT, 6, 3, 3).clone()

    glb_pose_smpl = model._reduced_glb_6d_to_full_glb_mat(pose6d, orient_bt)
    local_pose_bt = model._global2local(glb_pose_smpl, model.smpl_parents.tolist())

    pose_aa = transforms.matrix_to_axis_angle(local_pose_bt.detach().cpu()).to(
        device=device, dtype=pose_pred.dtype
    )

    _ensure_body_model_device(model, device)
    body_out = model.body_model(
        pose_body=pose_aa[:, 1:22].reshape(BT, 63),
        root_orient=pose_aa[:, 0].reshape(BT, 3),
        trans=torch.zeros(BT, 3, device=device, dtype=pose_pred.dtype),
    )
    joints = body_out.Jtr[:, :24, :].to(device).view(B, T, 24, 3)
    return joints


def _integrate_root_velocity(root_velocity: torch.Tensor, fps: float) -> torch.Tensor:
    root_disp = torch.cumsum(root_velocity / fps, dim=1)
    if root_disp.shape[1] > 0:
        root_disp[:, 0] = torch.zeros_like(root_disp[:, 0])
    return root_disp


def generate_dataset_predictions(
    dataset_name: str,
    checkpoint: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, Dict[str, List[List[float]]]], int]:
    dataset = MotionDatasetWithObjectAndTrans(
        datasets=[dataset_name],
        seq_len=args.seq_len,
        data_root=args.data_root,
        device=device,
        subset=args.subset,
        random_sample=False,
        use_full_sequence=True,
        fps=args.fps,
        imu_noise_std=0.0,
    )
    if len(dataset.sequences) == 0:
        print(f"[WARN] Dataset '{dataset_name}' has no sequences for subset '{args.subset}'. Skipping.")
        return {}, 0

    loader = FullSequenceLoader(dataset)

    model = PoserWithObject(
        body_model_path=args.body_model_path,
        fps=args.fps,
    ).to(device)
    _load_checkpoint(model, checkpoint, device)
    model.eval()

    predictions: Dict[str, Dict[str, List[List[float]]]] = {}
    for idx, seq in enumerate(dataset.sequences):
        seq_name = seq.get("seq_name", f"{dataset_name}_{idx:04d}")
        batch = FullSequenceLoader._sequence_to_batch(seq)
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        imu = batch["imu"]  # [1, T, 6, 12]
        orientation = imu[:, :, :, :9].contiguous().view(imu.shape[0], imu.shape[1], 6, 3, 3)
        obj_imu = batch.get("obj_imu", torch.zeros(imu.shape[0], imu.shape[1], 12, device=device, dtype=imu.dtype))
        obj_v_init = batch.get("obj_v_init", torch.zeros(imu.shape[0], 3, device=device, dtype=imu.dtype))

        v_pred, p_pred, obj_v_pred = model(
            imu,
            batch["v_init"],
            batch["p_init"],
            obj_imu,
            obj_v_init,
        )

        joints = _compute_joints(model, p_pred, orientation)
        joints = joints + batch['trans'].unsqueeze(2)

        wrist_indices = (20, 21)
        hand_positions = joints[0, :, wrist_indices, :].detach().cpu().tolist()

        predictions[seq_name] = {
            "pred_hand_glb_pos": hand_positions,
        }

    return predictions, len(dataset.sequences)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DynaIP hand position JSON files per dataset.")
    parser.add_argument(
        "--data_root",
        type=str,
        default="../../process",
        help="Root directory that contains processed dataset folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="my/dynaip_hand_predictions",
        help="Directory to store generated JSON files.",
    )
    parser.add_argument(
        "--body_model_path",
        type=str,
        default="../../datasets/smpl_models/smplh/male/model.npz",
        help="Path to SMPL/SMPLH body model file.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="test",
        help="Dataset subset to process (e.g., train/val/test).",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=120,
        help="Sequence length used during preprocessing (kept for compatibility).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate assumed during preprocessing.",
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
        help="Dataset folders to process.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Optional list of checkpoint paths matching --datasets order. "
             "If omitted, defaults from DEFAULT_DATASET_CONFIG are used.",
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
        raise ValueError("Number of checkpoints must match number of datasets when provided.")

    dataset_ckpt_pairs: List[Tuple[str, str]] = []
    if args.checkpoints is None:
        for dataset in args.datasets:
            if dataset not in DEFAULT_DATASET_CONFIG:
                raise KeyError(f"No default checkpoint configured for dataset '{dataset}'. "
                               "Provide --checkpoints explicitly.")
            dataset_ckpt_pairs.append((dataset, DEFAULT_DATASET_CONFIG[dataset]))
    else:
        dataset_ckpt_pairs = list(zip(args.datasets, args.checkpoints))

    summary: Dict[str, Dict] = {}
    for dataset_name, checkpoint in dataset_ckpt_pairs:
        print(f"[Info] Processing dataset '{dataset_name}' with checkpoint '{checkpoint}'")
        predictions, seq_count = generate_dataset_predictions(
            dataset_name,
            checkpoint,
            args,
            device,
        )
        if not predictions:
            continue

        output_path = os.path.join(
            args.output_dir,
            f"{dataset_name}_hand_predictions.json",
        )
        payload = {
            "dataset": dataset_name,
            "checkpoint": checkpoint,
            "subset": args.subset,
            "fps": args.fps,
            "num_sequences": seq_count,
            "predictions": predictions,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        print(f"[Info] Saved {len(predictions)} sequences to {output_path}")
        summary[dataset_name] = {
            "output": output_path,
            "sequences": seq_count,
        }

    if summary:
        print("\n[Summary]")
        for dataset_name, item in summary.items():
            print(f"  - {dataset_name}: {item['sequences']} sequences -> {item['output']}")
    else:
        print("[Info] No predictions generated.")


if __name__ == "__main__":
    main()
