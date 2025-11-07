import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from configs.global_config_IMUHOI import FRAME_RATE
from dataloader.dataloader_IMUHOI import IMUDataset
from models.IMUHOI_stage_net_noTrans_external_hand import TransPoseNetExternalHand
from models.do_train_IMUHOI_noTrans import build_model_input_dict


def load_config(path: str) -> edict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return edict(data)


class ExternalHandProvider:
    """
    Loads external hand global positions either from a directory of per-sequence JSON files
    or from a single aggregate JSON produced by generate_hand_json.py.
    """

    def __init__(
        self,
        base_path: str,
        key: str = "pred_hand_glb_pos",
        scale: float = 1.0,
        extension: str = ".json",
    ):
        self.base_path = base_path
        self.key = key
        self.scale = scale
        self.extension = extension
        self.is_directory = os.path.isdir(base_path)
        self.cache: Dict[str, torch.Tensor] = {}
        self.aggregate_data: Dict[str, torch.Tensor] = {}

        if not self.is_directory:
            if not os.path.isfile(base_path):
                raise FileNotFoundError(f"Hand position path not found: {base_path}")

            with open(base_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            predictions = payload.get("predictions", payload)
            if not isinstance(predictions, dict):
                raise ValueError(
                    "Aggregate hand JSON must contain a 'predictions' dict mapping "
                    "sequence names to data dictionaries."
                )

            for seq_name, seq_dict in predictions.items():
                if not isinstance(seq_dict, dict):
                    continue
                if self.key not in seq_dict:
                    continue
                tensor = torch.as_tensor(seq_dict[self.key], dtype=torch.float32) * float(self.scale)
                if tensor.ndim != 3 or tensor.shape[1:] != (2, 3):
                    raise ValueError(
                        f"Sequence '{seq_name}' hand data must have shape [T, 2, 3], "
                        f"got {tuple(tensor.shape)}"
                    )
                self.aggregate_data[seq_name] = tensor

    def _file_path(self, seq_name: str) -> str:
        filename = f"{seq_name}{self.extension}" if not seq_name.endswith(self.extension) else seq_name
        return os.path.join(self.base_path, filename)

    def get(self, seq_name: str) -> torch.Tensor:
        if seq_name in self.cache:
            return self.cache[seq_name]

        if self.is_directory:
            path = self._file_path(seq_name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Hand position file not found: {path}")

            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            if self.key not in payload:
                raise KeyError(f"Key '{self.key}' not found in {path}")

            tensor = torch.as_tensor(payload[self.key], dtype=torch.float32)
            if tensor.ndim != 3 or tensor.shape[1:] != (2, 3):
                raise ValueError(
                    f"Hand data for sequence '{seq_name}' must have shape [T, 2, 3], "
                    f"got {tuple(tensor.shape)} from {path}"
                )
            tensor = tensor * float(self.scale)
        else:
            if seq_name not in self.aggregate_data:
                sample_keys = ", ".join(list(self.aggregate_data.keys())[:5])
                raise KeyError(
                    f"Sequence '{seq_name}' not found in aggregate hand JSON. "
                    f"Sample keys: {sample_keys}"
                )
            tensor = self.aggregate_data[seq_name].clone()

        self.cache[seq_name] = tensor
        return tensor


def stack_external_hands(
    batch: Dict[str, Any],
    hand_provider: ExternalHandProvider,
    seq_len: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Fetch external hand predictions for the current batch and stack them into a tensor with
    shape [B, T, 2, 3]. Returns None if any sample cannot be loaded.
    """
    hands = torch.zeros(batch_size, seq_len, 2, 3, dtype=dtype, device=device)

    seq_names = batch.get("seq_name")
    starts = batch.get("window_start")
    ends = batch.get("window_end")

    def _get_value(container, idx):
        if isinstance(container, list):
            return container[idx]
        if isinstance(container, tuple):
            return container[idx]
        if isinstance(container, torch.Tensor):
            item = container[idx]
            if item.dim() == 0:
                return item.item()
            return item
        return container

    for i in range(batch_size):
        seq_name = _get_value(seq_names, i)
        if isinstance(seq_name, torch.Tensor):
            seq_name = seq_name.item()
        if not isinstance(seq_name, str):
            raise ValueError(f"Unexpected type for seq_name: {type(seq_name)}")

        start_idx = int(_get_value(starts, i))
        end_idx = int(_get_value(ends, i))

        try:
            full_hands = hand_provider.get(seq_name)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"[WARN] Skip batch because external hand data unavailable for {seq_name}: {exc}")
            return None

        if end_idx > full_hands.shape[0]:
            print(
                f"[WARN] Skip batch because external hand data for {seq_name} "
                f"has length {full_hands.shape[0]}, but requires up to index {end_idx}"
            )
            return None

        window = full_hands[start_idx:end_idx]
        if window.shape[0] != seq_len:
            print(
                f"[WARN] Skip batch because external hand window length ({window.shape[0]}) "
                f"does not match sequence length ({seq_len}) for {seq_name}"
            )
            return None

        hands[i] = window.to(device=device, dtype=dtype)

    return hands


def trim_batch_sequences(batch: Dict[str, Any], trim_frames: int) -> Dict[str, Any]:
    if trim_frames <= 0:
        return batch

    human_imu = batch.get("human_imu")
    if not isinstance(human_imu, torch.Tensor) or human_imu.dim() < 2:
        raise ValueError("Batch is missing 'human_imu' tensor required for trimming.")

    original_len = human_imu.shape[1]
    if original_len <= 2 * trim_frames:
        raise ValueError(
            f"Cannot trim {trim_frames} frames from sequences of length {original_len}."
        )

    trimmed_batch: Dict[str, Any] = {}
    time_slice = slice(trim_frames, -trim_frames if trim_frames > 0 else None)
    trimmed_len = original_len - 2 * trim_frames

    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.shape[1] == original_len:
            trimmed_batch[key] = value[:, time_slice, ...].contiguous()
        else:
            trimmed_batch[key] = value

    def _to_int(x: Any, default: int) -> int:
        if x is None:
            return default
        if isinstance(x, torch.Tensor):
            return int(x.item())
        return int(x)

    old_start = _to_int(batch.get("window_start"), 0)
    old_end = _to_int(batch.get("window_end"), original_len)
    new_start = max(0, old_start - trim_frames)
    clipped_end = min(old_end, original_len - trim_frames)
    new_end = max(new_start, clipped_end - trim_frames)
    new_end = min(new_end, trimmed_len)

    if "window_start" in batch:
        trimmed_batch["window_start"] = new_start
    if "window_end" in batch:
        trimmed_batch["window_end"] = new_end

    return trimmed_batch


def compute_obj_error(pred: Optional[torch.Tensor], gt: Optional[torch.Tensor]) -> float:
    if pred is None or gt is None:
        return math.nan
    if pred.shape != gt.shape:
        return math.nan
    diff = torch.linalg.norm(pred - gt, dim=-1).mean()
    return float(diff.item() * 100.0)


def compute_hoi_error(
    pred_obj_trans: Optional[torch.Tensor],
    pred_hand_positions: Optional[torch.Tensor],
    gt_obj_trans: Optional[torch.Tensor],
    gt_joints: Optional[torch.Tensor],
    gt_lhand_contact: Optional[torch.Tensor],
    gt_rhand_contact: Optional[torch.Tensor],
) -> float:
    if (
        pred_obj_trans is None
        or pred_hand_positions is None
        or gt_obj_trans is None
        or gt_joints is None
    ):
        return math.nan

    wrist_l_idx, wrist_r_idx = 20, 21

    pred_lhand_pos = pred_hand_positions[:, 0, :]
    pred_rhand_pos = pred_hand_positions[:, 1, :]
    gt_lhand_pos = gt_joints[:, wrist_l_idx, :]
    gt_rhand_pos = gt_joints[:, wrist_r_idx, :]

    rel_errors: List[torch.Tensor] = []

    if gt_lhand_contact is not None and gt_lhand_contact.any():
        mask = gt_lhand_contact.bool()
        rel_gt = (gt_obj_trans - gt_lhand_pos)[mask]
        rel_pred = (pred_obj_trans - pred_lhand_pos)[mask]
        if rel_gt.numel() > 0 and rel_pred.shape == rel_gt.shape:
            rel_errors.append(torch.linalg.norm(rel_pred - rel_gt, dim=-1))

    if gt_rhand_contact is not None and gt_rhand_contact.any():
        mask = gt_rhand_contact.bool()
        rel_gt = (gt_obj_trans - gt_rhand_pos)[mask]
        rel_pred = (pred_obj_trans - pred_rhand_pos)[mask]
        if rel_gt.numel() > 0 and rel_pred.shape == rel_gt.shape:
            rel_errors.append(torch.linalg.norm(rel_pred - rel_gt, dim=-1))

    if not rel_errors:
        return math.nan

    error_cat = torch.cat(rel_errors, dim=0)
    return float(error_cat.mean().item() * 100.0)


def load_model(
    config: edict,
    device: torch.device,
    velocity_ckpt: Optional[str],
    object_ckpt: Optional[str],
) -> TransPoseNetExternalHand:
    pretrained_modules: Dict[str, str] = {}
    if velocity_ckpt:
        pretrained_modules["velocity_contact"] = velocity_ckpt
    if object_ckpt:
        pretrained_modules["object_trans"] = object_ckpt

    model = TransPoseNetExternalHand(config, pretrained_modules=pretrained_modules)
    model = model.to(device)
    model.eval()
    return model


def evaluate(
    model: TransPoseNetExternalHand,
    data_loader: DataLoader,
    config: edict,
    hand_provider: ExternalHandProvider,
    device: torch.device,
    compute_fk: bool = True,
    trim_frames: int = 0,
) -> Dict[str, float]:
    metrics: Dict[str, List[float]] = {
        "obj_trans_err_fusion": [],
        "obj_trans_err_fk": [],
        "obj_trans_err_imu": [],
        "hoi_err_fusion": [],
        "hoi_err_fk": [],
        "hoi_err_imu": [],
    }

    stage_info = {"use_object_data": True}

    processed_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch_device = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_device[key] = value.to(device)
                else:
                    batch_device[key] = value

            if trim_frames > 0:
                try:
                    batch_device = trim_batch_sequences(batch_device, trim_frames)
                except ValueError as exc:
                    print(f"[WARN] Skipping batch {batch_idx} due to trim setting: {exc}")
                    continue

            try:
                data_dict = build_model_input_dict(
                    batch_device,
                    stage_info,
                    config,
                    device,
                    add_noise=False,
                )
            except Exception as exc:
                print(f"[WARN] Failed to build model input for batch {batch_idx}: {exc}")
                continue

            human_imu = data_dict["human_imu"]
            batch_size, seq_len = human_imu.shape[:2]

            external_hands = stack_external_hands(
                batch_device,
                hand_provider,
                seq_len,
                batch_size,
                dtype=human_imu.dtype,
                device=human_imu.device,
            )
            if external_hands is None:
                continue

            try:
                pred_dict = model(
                    data_dict,
                    external_hand_glb_pos=external_hands,
                    use_object_data=stage_info.get("use_object_data", True),
                    compute_fk=compute_fk,
                )
            except Exception as exc:
                print(f"[WARN] Model forward failed on batch {batch_idx}: {exc}")
                continue

            pred_obj_trans = pred_dict.get("pred_obj_trans")
            pred_obj_trans_fk = pred_dict.get("pred_obj_trans_fk")
            pred_obj_vel = pred_dict.get("pred_obj_vel")

            obj_trans_init = data_dict.get("obj_trans_init")

            gt_obj_trans = batch_device.get("obj_trans")
            gt_joints_all = batch_device.get("position_global")
            gt_lhand_contact = batch_device.get("lhand_contact")
            gt_rhand_contact = batch_device.get("rhand_contact")
            has_object_batch = batch_device.get("has_object")

            if not isinstance(has_object_batch, torch.Tensor):
                has_object_batch = torch.ones(batch_size, dtype=torch.bool, device=device)

            for sample_idx in range(batch_size):
                if not has_object_batch[sample_idx].bool().item():
                    continue

                seq_pred_obj = pred_obj_trans[sample_idx] if pred_obj_trans is not None else None
                seq_pred_fk = pred_obj_trans_fk[sample_idx] if pred_obj_trans_fk is not None else None
                seq_gt_obj = gt_obj_trans[sample_idx] if isinstance(gt_obj_trans, torch.Tensor) else None
                seq_gt_joints = gt_joints_all[sample_idx] if isinstance(gt_joints_all, torch.Tensor) else None
                seq_gt_lhand_contact = (
                    gt_lhand_contact[sample_idx] if isinstance(gt_lhand_contact, torch.Tensor) else None
                )
                seq_gt_rhand_contact = (
                    gt_rhand_contact[sample_idx] if isinstance(gt_rhand_contact, torch.Tensor) else None
                )
                seq_hand_pos = external_hands[sample_idx]

                # Velocity-based translation (imu branch)
                seq_pred_imu = None
                if pred_obj_vel is not None:
                    seq_vel = pred_obj_vel[sample_idx]
                    dt = 1.0 / float(FRAME_RATE)
                    disp = torch.cumsum(seq_vel * dt, dim=0)
                    if disp.shape[0] > 0:
                        zero_row = torch.zeros(1, 3, device=disp.device, dtype=disp.dtype)
                        disp = torch.cat([zero_row, disp[:-1]], dim=0)

                    if isinstance(obj_trans_init, torch.Tensor):
                        init_pos = obj_trans_init[sample_idx]
                    elif seq_gt_obj is not None:
                        init_pos = seq_gt_obj[0]
                    else:
                        init_pos = torch.zeros(3, device=disp.device, dtype=disp.dtype)
                    seq_pred_imu = init_pos.unsqueeze(0) + disp

                metrics["obj_trans_err_fusion"].append(compute_obj_error(seq_pred_obj, seq_gt_obj))
                metrics["obj_trans_err_fk"].append(compute_obj_error(seq_pred_fk, seq_gt_obj))
                metrics["obj_trans_err_imu"].append(compute_obj_error(seq_pred_imu, seq_gt_obj))

                metrics["hoi_err_fusion"].append(
                    compute_hoi_error(
                        seq_pred_obj,
                        seq_hand_pos,
                        seq_gt_obj,
                        seq_gt_joints,
                        seq_gt_lhand_contact,
                        seq_gt_rhand_contact,
                    )
                )
                metrics["hoi_err_fk"].append(
                    compute_hoi_error(
                        seq_pred_fk,
                        seq_hand_pos,
                        seq_gt_obj,
                        seq_gt_joints,
                        seq_gt_lhand_contact,
                        seq_gt_rhand_contact,
                    )
                )
                metrics["hoi_err_imu"].append(
                    compute_hoi_error(
                        seq_pred_imu,
                        seq_hand_pos,
                        seq_gt_obj,
                        seq_gt_joints,
                        seq_gt_lhand_contact,
                        seq_gt_rhand_contact,
                    )
                )

            processed_batches += 1
            if processed_batches % 50 == 0:
                print(f"Processed {processed_batches} batches")

    results: Dict[str, float] = {}
    for key, values in metrics.items():
        valid = [v for v in values if not math.isnan(v)]
        results[key] = float(np.mean(valid)) if valid else math.nan
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate TransPoseNet (noTrans) with external hand positions.")
    parser.add_argument("--config", type=str, default="configs/IMUHOI_train_noTrans.yaml", help="Config file path")
    parser.add_argument("--test_data_dir", type=str, default="process/processed_split_data_OMOMO/test")
    parser.add_argument(
        "--hand_pos_dir",
        type=str,
        default="RefCodes/GlobalPose/my/globalpose_hand_predictions/processed_split_data_OMOMO_hand_predictions.json",
        help="Directory of per-sequence hand JSON files or a single aggregate JSON file.",
    )
    parser.add_argument("--hand_pos_key", type=str, default="pred_hand_glb_pos", help="JSON key for hand positions")
    parser.add_argument("--hand_pos_scale", type=float, default=1.0, help="Scale factor applied to loaded hand data")
    parser.add_argument("--hand_pos_ext", type=str, default=".json", help="File extension for hand position files")
    parser.add_argument("--velocity_contact_ckpt", type=str, default='outputs/IMUHOI_noTrans/transpose_noTrans_omomo/modules/velocity_contact_best.pt', help="Checkpoint for velocity/contact module")
    parser.add_argument("--object_trans_ckpt", type=str, default='outputs/IMUHOI_noTrans/transpose_noTrans_omomo/modules/object_trans_best.pt', help="Checkpoint for object translation module")
    parser.add_argument("--num_workers", type=int, default=12, help="Dataloader worker count")
    parser.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size (recommend 1)")
    parser.add_argument("--no_fk", action="store_true", help="Disable FK branch computation in object module")
    parser.add_argument(
        "--trim_frames",
        type=int,
        default=6,
        help="Number of frames to trim from the beginning and end of each sequence.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = load_config(args.config)
    if not hasattr(config, "debug"):
        config.debug = False
    config.device = str(device)

    if args.velocity_contact_ckpt and not os.path.exists(args.velocity_contact_ckpt):
        raise FileNotFoundError(f"Velocity/contact checkpoint not found: {args.velocity_contact_ckpt}")
    if args.object_trans_ckpt and not os.path.exists(args.object_trans_ckpt):
        raise FileNotFoundError(f"Object translation checkpoint not found: {args.object_trans_ckpt}")

    if not (os.path.isdir(args.hand_pos_dir) or os.path.isfile(args.hand_pos_dir)):
        raise FileNotFoundError(f"Hand position path not found: {args.hand_pos_dir}")

    if not os.path.exists(args.test_data_dir):
        raise FileNotFoundError(f"Test dataset directory not found: {args.test_data_dir}")

    test_window = config.test.get("window", config.train.get("window", 60))
    test_dataset = IMUDataset(
        data_dir=args.test_data_dir,
        window_size=test_window,
        full_sequence=True,
    )

    if len(test_dataset) == 0:
        raise RuntimeError("Test dataset is empty. Please check preprocessing.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Dataset size: {len(test_dataset)} | Batches: {len(test_loader)}")

    model = load_model(
        config,
        device,
        args.velocity_contact_ckpt,
        args.object_trans_ckpt,
    )

    hand_provider = ExternalHandProvider(
        args.hand_pos_dir,
        key=args.hand_pos_key,
        scale=args.hand_pos_scale,
        extension=args.hand_pos_ext,
    )

    results = evaluate(
        model,
        test_loader,
        config,
        hand_provider,
        device,
        compute_fk=not args.no_fk,
        trim_frames=max(0, args.trim_frames),
    )

    def _fmt(key: str) -> str:
        val = results.get(key, math.nan)
        return f"{val:.4f}" if not math.isnan(val) else "NaN"

    print("\n--- Evaluation (External Hand) ---")
    print(f"Obj Trans Fusion (cm): {_fmt('obj_trans_err_fusion')}")
    print(f"Obj Trans FK     (cm): {_fmt('obj_trans_err_fk')}")
    print(f"Obj Trans IMU    (cm): {_fmt('obj_trans_err_imu')}")
    print(f"HOI Fusion       (cm): {_fmt('hoi_err_fusion')}")
    print(f"HOI FK           (cm): {_fmt('hoi_err_fk')}")
    print(f"HOI IMU          (cm): {_fmt('hoi_err_imu')}")


if __name__ == "__main__":
    main()
