import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
from pytorch3d import transforms

from my.dataset_trans_obj import TransPoseObjectDataset
from my.model_obj import TransPoseWithObject


DEFAULT_DATASET_CONFIG: Dict[str, str] = {
    "processed_seg_data_BEHAVE": "checkpoints/transpose_object_behave/best_val.pth",
    "processed_seg_data_IMHD": "checkpoints/transpose_object_imhd/best_val.pth",
    "processed_split_data_OMOMO": "checkpoints/transpose_object_omomo/best_val.pth",
}


def _load_checkpoint(model: TransPoseWithObject, checkpoint: str, device: torch.device) -> None:
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)


def generate_dataset_predictions(
    dataset_name: str,
    checkpoint: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, Dict[str, List[List[float]]]], int]:
    dataset = TransPoseObjectDataset(
        datasets=[dataset_name],
        seq_len=args.seq_len,
        data_root=args.data_root,
        subset=args.subset,
        fps_default=args.fps,
        trim_frames=args.trim_frames,
        random_sample=False,
        use_full_sequence=True,
        device=None,
        pin_memory=False,
    )
    if len(dataset) == 0:
        print(f"[WARN] Dataset '{dataset_name}' has no sequences for subset '{args.subset}'. Skipping.")
        return {}, 0

    model = TransPoseWithObject(fps=args.fps).to(device)
    _load_checkpoint(model, checkpoint, device)
    model.eval()

    from human_body_prior.body_model.body_model import BodyModel
    body_model = BodyModel(bm_fname=args.body_model_path, num_betas=16).to(device)
    body_model.eval()
    for param in body_model.parameters():
        param.requires_grad_(False)

    predictions: Dict[str, Dict[str, List[List[float]]]] = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        seq_name = sample.get("seq_name", f"{dataset_name}_{idx:04d}")

        imu_seq = sample["imu"]
        imu = imu_seq.unsqueeze(0).to(device)

        obj_imu_seq = sample.get("obj_imu")
        if obj_imu_seq is None:
            obj_imu_seq = torch.zeros(imu_seq.shape[0], 12, dtype=imu_seq.dtype, device=imu_seq.device)
        obj_imu = obj_imu_seq.unsqueeze(0).to(device)

        obj_pos_init_seq = sample.get("obj_pos_init", torch.zeros(3, dtype=imu_seq.dtype))
        if isinstance(obj_pos_init_seq, torch.Tensor):
            obj_pos_init = obj_pos_init_seq.to(device).unsqueeze(0)
        else:
            obj_pos_init = torch.tensor(obj_pos_init_seq, device=device, dtype=imu_seq.dtype).unsqueeze(0)
        fps_tensor = sample.get("fps")
        if isinstance(fps_tensor, torch.Tensor):
            fps_tensor = fps_tensor.unsqueeze(0).to(device)
        else:
            fps_tensor = torch.full((1, imu_seq.shape[0]), float(args.fps), device=device, dtype=imu_seq.dtype)

        with torch.no_grad():
            outputs = model(
                human_imu=imu,
                object_imu=obj_imu,
                obj_pos_init=obj_pos_init,
                fps=fps_tensor,
            )

        reduced_pose = outputs["reduced_pose"]
        B, T = reduced_pose.shape[0], reduced_pose.shape[1]
        root_rotation = imu[:, :, -9:].reshape(B, T, 3, 3)
        pose6d = reduced_pose.reshape(B * T, -1, 6)
        root_flat = root_rotation.reshape(B * T, 3, 3)

        pose_mat = model.base._reduced_glb_6d_to_full_local_mat(root_flat, pose6d)
        pose_aa = transforms.matrix_to_axis_angle(pose_mat)

        body_out = body_model(
            pose_body=pose_aa[:, 1:22].reshape(B * T, 63),
            root_orient=pose_aa[:, 0].reshape(B * T, 3),
        )
        joints = body_out.Jtr[:, :24, :].to(device).view(B, T, 24, 3)

        root_position = sample.get("root_position_world")
        if root_position is None:
            root_position = torch.zeros(T, 3, device=device)
        else:
            if isinstance(root_position, torch.Tensor):
                root_position = root_position.to(device)
            else:
                root_position = torch.tensor(root_position, device=device, dtype=joints.dtype)
        joints_world = joints + root_position.view(1, T, 1, 3)

        hand_indices = [20, 21]
        hand_positions = joints_world[0, :, hand_indices, :].detach().cpu().tolist()

        predictions[seq_name] = {
            "pred_hand_glb_pos": hand_positions,
        }

    return predictions, len(dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TransPose hand position JSON per dataset.")
    parser.add_argument(
        "--data_root",
        type=str,
        default="../../process",
        help="Root directory containing processed dataset folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="my/transpose_hand_predictions",
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
