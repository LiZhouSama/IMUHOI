import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
from pytorch3d import transforms

from my_simple.dataset_obj import MotionDatasetWithObjectAndTrans
from my_simple.model_obj import PoserWithObject


DEFAULT_DATASET_CONFIG: Dict[str, str] = {
    "processed_seg_data_BEHAVE": "checkpoints/dip_obj_behave/best_val.pth",
    "processed_seg_data_IMHD": "checkpoints/dip_obj_imhd/best_val.pth",
    "processed_split_data_OMOMO": "checkpoints/dip_obj_omomo/best_val.pth",
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
        raise RuntimeError("Body model is required. Provide a valid --body_model_path.")
    if model.body_model_device != device:
        model.body_model = model.body_model.to(device)
        model.body_model_device = device


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
        trim_frames=args.trim_frames,
    )
    if len(dataset.sequences) == 0:
        print(f"[WARN] Dataset '{dataset_name}' has no sequences for subset '{args.subset}'. Skipping.")
        return {}, 0

    model = PoserWithObject(
        body_model_path=args.body_model_path,
        fps=args.fps,
    ).to(device)
    _load_checkpoint(model, checkpoint, device)
    _ensure_body_model_device(model, device)
    model.eval()

    predictions: Dict[str, Dict[str, List[List[float]]]] = {}
    for idx, seq in enumerate(dataset.sequences):
        seq_name = seq.get("seq_name", f"{dataset_name}_{idx:04d}")

        imu = seq["imu"].unsqueeze(0).to(device)
        obj_imu = seq["obj_imu"].unsqueeze(0).to(device)
        trans_seq = seq["trans"].to(device)

        with torch.no_grad():
            p_pred_raw, _ = model(imu, obj_imu)

        B, T = p_pred_raw.shape[0], p_pred_raw.shape[1]
        pose_mat = p_pred_raw.view(B, T, 22, 3, 3)
        pose_aa = transforms.matrix_to_axis_angle(pose_mat)

        pose_body = pose_aa[:, :, 1:22].reshape(B * T, 63)
        root_orient = pose_aa[:, :, 0].reshape(B * T, 3)

        body_out = model.body_model(
            pose_body=pose_body,
            root_orient=root_orient,
        )
        joints = body_out.Jtr[:, :24, :].to(device).view(B, T, 24, 3)

        trans_expanded = trans_seq.view(1, T, 1, 3)
        joints_global = joints + trans_expanded

        hand_indices = [20, 21]
        pred_hands = joints_global[0, :, hand_indices, :].detach().cpu().tolist()

        predictions[seq_name] = {
            "pred_hand_glb_pos": pred_hands,
        }

    return predictions, len(dataset.sequences)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DIP hand position JSON per dataset.")
    parser.add_argument(
        "--data_root",
        type=str,
        default="../../process",
        help="Root directory containing processed dataset folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="my/dip_hand_predictions",
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
        help="Sequence length used during preprocessing (for compatibility).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate assumed during preprocessing.",
    )
    parser.add_argument(
        "--trim_frames",
        type=int,
        default=6,
        help="Number of frames trimmed at the beginning/end during preprocessing.",
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
                raise KeyError(
                    f"No default checkpoint configured for dataset '{dataset}'. "
                    "Provide --checkpoints explicitly."
                )
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
            "trim_frames": args.trim_frames,
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
