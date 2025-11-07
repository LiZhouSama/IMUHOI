import os
import argparse
import copy
from pathlib import Path
from typing import Dict, Optional

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from sklearn.metrics import f1_score
import pytorch3d.transforms as t3d

from dataloader.dataloader_IMUHOI import IMUDataset
from models.IMUHOI_stage_net import TransPoseNet
from models.do_train_IMUHOI import build_model_input_dict
from human_body_prior.body_model.body_model import BodyModel
from configs.global_config_IMUHOI import (
    FRAME_RATE,
    _REDUCED_POSE_NAMES,
    _SENSOR_ROT_INDICES,
)

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESS_ROOT = PROJECT_ROOT / "process"
OUTPUT_IMUHOI_ROOT = PROJECT_ROOT / "outputs" / "IMUHOI"

DEFAULT_DATASET_CONFIG = {
    "processed_seg_data_BEHAVE": {
        "data_dir": PROCESS_ROOT / "processed_seg_data_BEHAVE" / "test",
        # "model_path": OUTPUT_IMUHOI_ROOT / "transpose_10311104_BEHAVE" / "stage_best_joint_training.pt",
        "modules": {
            "velocity_contact": OUTPUT_IMUHOI_ROOT
            / "transpose_10311104_BEHAVE"
            / "modules"
            / "velocity_contact_best.pt",
            "human_pose": OUTPUT_IMUHOI_ROOT
            / "transpose_10311104_BEHAVE"
            / "modules"
            / "human_pose_best.pt",
            "object_trans": OUTPUT_IMUHOI_ROOT
            / "transpose_10311104_BEHAVE"
            / "modules"
            / "object_trans_best.pt",
        },
    },
    "processed_seg_data_IMHD": {
        "data_dir": PROCESS_ROOT / "processed_seg_data_IMHD" / "test",
        # "model_path": OUTPUT_IMUHOI_ROOT / "transpose_10310305_IMHD" / "stage_best_joint_training.pt",
        "modules": {
            "velocity_contact": OUTPUT_IMUHOI_ROOT
            / "transpose_10310305_IMHD"
            / "modules"
            / "velocity_contact_best.pt",
            "human_pose": OUTPUT_IMUHOI_ROOT
            / "transpose_10310305_IMHD"
            / "modules"
            / "human_pose_best.pt",
            "object_trans": OUTPUT_IMUHOI_ROOT
            / "transpose_10310305_IMHD"
            / "modules"
            / "object_trans_best.pt",
        },
    },
    "processed_split_data_OMOMO": {
        "data_dir": PROCESS_ROOT / "processed_split_data_OMOMO" / "test",
        # "model_path": OUTPUT_IMUHOI_ROOT / "transpose_10302129_OMOMO" / "stage_best_joint_training.pt",
        "modules": {
            "velocity_contact": OUTPUT_IMUHOI_ROOT
            / "transpose_10302129_OMOMO"
            / "modules"
            / "velocity_contact_best.pt",
            "human_pose": OUTPUT_IMUHOI_ROOT
            / "transpose_10302129_OMOMO"
            / "modules"
            / "human_pose_best.pt",
            "object_trans": OUTPUT_IMUHOI_ROOT
            / "transpose_10302129_OMOMO"
            / "modules"
            / "object_trans_best.pt",
        },
    },
}


def load_config(config_path: str) -> edict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return edict(cfg)


def load_smpl_model(smpl_path: str, device: torch.device) -> BodyModel:
    if not os.path.exists(smpl_path):
        raise FileNotFoundError(f"SMPL model not found at {smpl_path}")
    print(f"Loading SMPL model from: {smpl_path}")
    smpl_model = BodyModel(bm_fname=smpl_path, num_betas=16, model_type="smplh")
    return smpl_model.to(device)


def load_model(config: edict, device: torch.device) -> TransPoseNet:
    staged_cfg = getattr(config, "staged_training", {})
    modular_cfg = staged_cfg.get("modular_training", {}) if staged_cfg else {}
    use_modular = bool(modular_cfg.get("enabled", False))
    pretrained_modules = modular_cfg.get("pretrained_modules", {}) if use_modular else {}
    skip_modules = modular_cfg.get("skip_modules", []) if use_modular else []

    if use_modular and pretrained_modules:
        print("Building TransPoseNet with pretrained modules:")
        for name, path in pretrained_modules.items():
            print(f"  - {name}: {path}")
        model = TransPoseNet(config, pretrained_modules=pretrained_modules, skip_modules=skip_modules)
    else:
        print("Building TransPoseNet without modular weights (fresh init).")
        model = TransPoseNet(config)

    model = model.to(device)

    model.eval()
    return model


def _select_path(override: Optional[str], default_path: Path) -> Path:
    path = Path(override).expanduser() if override else default_path
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path


def _build_dataset_runs(args: argparse.Namespace):
    runs = []
    if args.dataset:
        runs.append((args.dataset, copy.deepcopy(DEFAULT_DATASET_CONFIG[args.dataset])))
        return runs
    if args.test_data_dir:
        run_cfg = {"data_dir": Path(args.test_data_dir).expanduser()}
        runs.append(("custom", run_cfg))
        return runs
    for name, cfg in DEFAULT_DATASET_CONFIG.items():
        runs.append((name, copy.deepcopy(cfg)))
    return runs


def _apply_module_overrides(config: edict, modules_override: Dict[str, Path]) -> None:
    if not modules_override:
        return
    staged_cfg = getattr(config, "staged_training", None)
    if not staged_cfg:
        return
    if isinstance(staged_cfg, dict):
        modular_cfg = staged_cfg.get("modular_training")
    else:
        modular_cfg = getattr(staged_cfg, "modular_training", None)
    if not modular_cfg:
        return

    pretrained_modules = dict(modular_cfg.get("pretrained_modules", {}))
    print("Using dataset-specific pretrained modules:")
    for module_name, module_path in modules_override.items():
        pretrained_modules[module_name] = str(module_path)
        print(f"  - {module_name}: {module_path}")
    modular_cfg["pretrained_modules"] = pretrained_modules

    if isinstance(staged_cfg, dict):
        staged_cfg["modular_training"] = modular_cfg
    else:
        staged_cfg.modular_training = modular_cfg
    config.staged_training = staged_cfg


def evaluate_model(
    model: TransPoseNet,
    smpl_model: BodyModel,
    data_loader: DataLoader,
    config: edict,
    device: torch.device,
    evaluate_objects: bool = True,
    compare_three: bool = True,
):
    metrics = {
        "mpjpe": [],
        "mpjre_angle": [],
        "root_trans_err": [],
        "jitter": [],
        "obj_trans_err_fusion": [],
        "obj_trans_err_fk": [],
        "obj_trans_err_imu": [],
        "hoi_err_fusion": [],
        "hoi_err_fk": [],
        "hoi_err_imu": [],
        "contact_f1_lhand": [],
        "contact_f1_rhand": [],
        "contact_f1_obj": [],
    }

    stage_info = {"use_object_data": True}
    num_eval_joints = 22

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Move tensors to device
            batch_device = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_device[key] = value.to(device)
                else:
                    batch_device[key] = value

            try:
                data_dict = build_model_input_dict(batch_device, stage_info, config, device, add_noise=False)
            except Exception as exc:
                print(f"Failed to build model input (batch {batch_idx}): {exc}")
                continue

            compute_fk = evaluate_objects and compare_three
            try:
                pred_dict = model(
                    data_dict,
                    use_object_data=stage_info.get("use_object_data", True),
                    compute_fk=compute_fk,
                )
            except Exception as exc:
                print(f"Model inference failed on batch {batch_idx}: {exc}")
                continue

            human_imu = data_dict["human_imu"]  # [B, T, num_imu, 9]
            batch_size, seq_len = human_imu.shape[0], human_imu.shape[1]

            for sample_idx in range(batch_size):
                human_imu_seq = human_imu[sample_idx]
                T = human_imu_seq.shape[0]

                pose_batch = batch_device.get("pose")
                trans_batch = batch_device.get("trans")
                if pose_batch is None or trans_batch is None:
                    print("Warning: Missing GT pose/trans in batch; skipping sample.")
                    continue

                gt_pose_seq = pose_batch[sample_idx]  # [T, 66] expected
                gt_trans_seq = trans_batch[sample_idx]  # [T, 3]

                if gt_pose_seq.shape[-1] < 6:
                    print("Warning: GT pose dimension too small; skipping sample.")
                    continue

                gt_root_axis = gt_pose_seq[:, :3]
                gt_body_axis = gt_pose_seq[:, 3:3 + 63]
                if gt_body_axis.shape[-1] < 63:
                    pad = torch.zeros(T, 63 - gt_body_axis.shape[-1], device=device, dtype=gt_body_axis.dtype)
                    gt_body_axis = torch.cat([gt_body_axis, pad], dim=-1)
                else:
                    gt_body_axis = gt_body_axis[:, :63]
                gt_body_axis = gt_body_axis.view(T, 21, 3)

                try:
                    gt_smpl_out = smpl_model(
                        root_orient=gt_root_axis,
                        pose_body=gt_body_axis.reshape(T, -1),
                        trans=gt_trans_seq,
                    )
                except Exception as exc:
                    print(f"SMPL forward failed for GT (batch {batch_idx}, sample {sample_idx}): {exc}")
                    continue

                gt_joints_all = gt_smpl_out.Jtr  # [T, J, 3]

                # Predicted outputs
                pred_root_trans_all = pred_dict.get("root_trans_pred")
                pred_root_trans_seq = (
                    pred_root_trans_all[sample_idx]
                    if pred_root_trans_all is not None
                    else None
                )

                p_pred_seq = pred_dict.get("p_pred")
                if p_pred_seq is None:
                    print("Warning: Model did not output reduced pose; skipping sample.")
                    continue

                p_pred_seq = p_pred_seq[sample_idx]
                reduced_pose = p_pred_seq.view(T, len(_REDUCED_POSE_NAMES), 6)

                orientation_6d = human_imu_seq[:, :, -6:]
                orientation_mat = t3d.rotation_6d_to_matrix(orientation_6d.reshape(-1, 6)).reshape(
                    T, human_imu_seq.shape[1], 3, 3
                )
                orientation_subset = orientation_mat[:, : len(_SENSOR_ROT_INDICES), :, :]

                human_module = model.human_pose_module
                if human_module is None:
                    print("Warning: human_pose_module is None; skipping sample.")
                    continue

                full_glb = human_module._reduced_glb_6d_to_full_glb_mat(
                    reduced_pose,
                    orientation_subset.reshape(T, len(_SENSOR_ROT_INDICES), 3, 3),
                )
                parents = human_module.smpl_parents.tolist()
                local_rot = human_module._global2local(full_glb, parents)
                pose_axis_full = t3d.matrix_to_axis_angle(local_rot.reshape(-1, 3, 3)).reshape(
                    T, full_glb.shape[1], 3
                )
                pred_root_axis = pose_axis_full[:, 0, :]
                pred_body_axis = pose_axis_full[:, 1:22, :]

                try:
                    pred_smpl_out = smpl_model(
                        root_orient=pred_root_axis,
                        pose_body=pred_body_axis.reshape(T, -1),
                        trans=pred_root_trans_seq
                        if pred_root_trans_seq is not None
                        else gt_trans_seq,
                    )
                except Exception as exc:
                    print(f"SMPL forward failed for prediction (batch {batch_idx}, sample {sample_idx}): {exc}")
                    continue

                pred_joints_all = pred_smpl_out.Jtr

                # --- MPJPE ---
                pred_joints_eval = pred_joints_all[:, :num_eval_joints, :]
                gt_joints_eval = gt_joints_all[:, :num_eval_joints, :]
                pred_joints_rel = pred_joints_eval - pred_joints_eval[:, 0:1, :]
                gt_joints_rel = gt_joints_eval - gt_joints_eval[:, 0:1, :]
                joint_distances = torch.linalg.norm(pred_joints_rel - gt_joints_rel, dim=-1)
                metrics["mpjpe"].append(joint_distances.mean().item() * 100.0)

                # --- MPJRE (deg) ---
                # pred_body_mat = t3d.axis_angle_to_matrix(pred_body_axis.reshape(-1, 3)).reshape(T, 21, 3, 3)
                # gt_body_mat = t3d.axis_angle_to_matrix(gt_body_axis.reshape(-1, 3)).reshape(T, 21, 3, 3)
                # rel_rot = torch.matmul(gt_body_mat.transpose(-1, -2), pred_body_mat)
                # trace = torch.einsum("...ii->...", rel_rot)
                # cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
                # angle_rad = torch.acos(cos_theta)
                # metrics["mpjre_angle"].append((angle_rad.mean().item() * 180.0) / np.pi)
                # pred_body_6d = t3d.matrix_to_rotation_6d(pred_body_mat)
                # gt_body_6d = t3d.matrix_to_rotation_6d(gt_body_mat)
                rot_error_ = torch.mean(torch.absolute(gt_body_axis-pred_body_axis)) * 57.2958
                metrics["mpjre_angle"].append(rot_error_.item())

                if pred_root_trans_seq is not None:
                    root_err = torch.linalg.norm(
                        pred_root_trans_seq - gt_trans_seq, dim=-1
                    ).mean().item() * 100.0
                    metrics["root_trans_err"].append(root_err)
                else:
                    metrics["root_trans_err"].append(float("nan"))

                # --- Jitter ---
                if T >= 3:
                    pred_joints_global = pred_joints_eval
                    accel = (
                        pred_joints_global[2:]
                        - 2 * pred_joints_global[1:-1]
                        + pred_joints_global[:-2]
                    )
                    jitter = torch.linalg.norm(accel, dim=-1).mean().item() * 1000.0
                    metrics["jitter"].append(jitter)
                else:
                    metrics["jitter"].append(float("nan"))

                # Prepare GT contacts and object data
                has_object_mask = data_dict.get("has_object")
                has_object_sample = (
                    bool(has_object_mask[sample_idx].item()) if isinstance(has_object_mask, torch.Tensor) else True
                )

                gt_obj_trans = batch_device.get("obj_trans")
                gt_obj_rot = batch_device.get("obj_rot")
                gt_obj_trans_seq = gt_obj_trans[sample_idx] if gt_obj_trans is not None else None
                gt_obj_rot_seq = gt_obj_rot[sample_idx] if gt_obj_rot is not None else None

                gt_lhand_contact = batch_device.get("lhand_contact")
                gt_rhand_contact = batch_device.get("rhand_contact")
                gt_obj_contact = batch_device.get("obj_contact")
                gt_lhand_contact_seq = (
                    gt_lhand_contact[sample_idx].bool() if gt_lhand_contact is not None else None
                )
                gt_rhand_contact_seq = (
                    gt_rhand_contact[sample_idx].bool() if gt_rhand_contact is not None else None
                )
                gt_obj_contact_seq = (
                    gt_obj_contact[sample_idx].bool() if gt_obj_contact is not None else None
                )

                # Object predictions
                pred_obj_trans_fusion = pred_dict.get("pred_obj_trans")
                pred_obj_trans_fusion = (
                    pred_obj_trans_fusion[sample_idx] if pred_obj_trans_fusion is not None else None
                )

                pred_obj_trans_fk = pred_dict.get("pred_obj_trans_fk")
                pred_obj_trans_fk = pred_obj_trans_fk[sample_idx] if pred_obj_trans_fk is not None else None

                pred_obj_vel = pred_dict.get("pred_obj_vel")
                pred_obj_vel = pred_obj_vel[sample_idx] if pred_obj_vel is not None else None

                pred_obj_trans_imu = None
                if evaluate_objects and has_object_sample and pred_obj_vel is not None:
                    dt = 1.0 / float(FRAME_RATE)
                    vel_scaled = pred_obj_vel * dt
                    disp = torch.cumsum(vel_scaled, dim=0)
                    if disp.shape[0] > 0:
                        zero_row = torch.zeros(1, 3, device=device, dtype=disp.dtype)
                        disp = torch.cat([zero_row, disp[:-1]], dim=0)
                    obj_trans_init = data_dict.get("obj_trans_init")
                    if isinstance(obj_trans_init, torch.Tensor):
                        init_pos = obj_trans_init[sample_idx]
                    else:
                        init_pos = gt_obj_trans_seq[0] if gt_obj_trans_seq is not None else torch.zeros(3, device=device)
                    pred_obj_trans_imu = init_pos.unsqueeze(0) + disp

                def _append_obj_error(name: str, pred_trans):
                    if evaluate_objects and has_object_sample and gt_obj_trans_seq is not None and pred_trans is not None:
                        err = (
                            torch.linalg.norm(pred_trans - gt_obj_trans_seq, dim=-1)
                            .mean()
                            .item()
                            * 100.0
                        )
                        metrics[name].append(err)
                    elif evaluate_objects and has_object_sample:
                        metrics[name].append(float("nan"))
                    else:
                        metrics[name].append(float("nan"))

                _append_obj_error("obj_trans_err_fusion", pred_obj_trans_fusion)
                _append_obj_error("obj_trans_err_fk", pred_obj_trans_fk)
                _append_obj_error("obj_trans_err_imu", pred_obj_trans_imu)

                def compute_hoi_error(pred_obj_trans_seq):
                    if not (evaluate_objects and has_object_sample):
                        return float("nan")
                    if pred_obj_trans_seq is None or gt_obj_trans_seq is None:
                        return float("nan")

                    wrist_l_idx, wrist_r_idx, root_idx = 20, 21, 0
                    pred_lhand_pos = pred_joints_all[:, wrist_l_idx, :]
                    pred_rhand_pos = pred_joints_all[:, wrist_r_idx, :]
                    pred_root_pos = pred_joints_all[:, root_idx, :]

                    gt_lhand_pos = gt_joints_all[:, wrist_l_idx, :]
                    gt_rhand_pos = gt_joints_all[:, wrist_r_idx, :]
                    gt_root_pos = gt_joints_all[:, root_idx, :]

                    rel_errors = []

                    if gt_lhand_contact_seq is not None and gt_lhand_contact_seq.any():
                        mask = gt_lhand_contact_seq
                        rel_gt = (gt_obj_trans_seq - gt_lhand_pos)[mask]
                        rel_pred = (pred_obj_trans_seq - pred_lhand_pos)[mask]
                        rel_errors.append(torch.linalg.norm(rel_pred - rel_gt, dim=-1))

                    if gt_rhand_contact_seq is not None and gt_rhand_contact_seq.any():
                        mask = gt_rhand_contact_seq
                        rel_gt = (gt_obj_trans_seq - gt_rhand_pos)[mask]
                        rel_pred = (pred_obj_trans_seq - pred_rhand_pos)[mask]
                        rel_errors.append(torch.linalg.norm(rel_pred - rel_gt, dim=-1))

                    if gt_lhand_contact_seq is not None and gt_rhand_contact_seq is not None:
                        non_inter_mask = (~gt_lhand_contact_seq) & (~gt_rhand_contact_seq)
                        if non_inter_mask.any():
                            rel_gt = (gt_obj_trans_seq - gt_root_pos)[non_inter_mask]
                            rel_pred = (pred_obj_trans_seq - pred_root_pos)[non_inter_mask]
                            rel_errors.append(torch.linalg.norm(rel_pred - rel_gt, dim=-1))

                    if rel_errors:
                        return torch.cat(rel_errors, dim=0).mean().item() * 100.0
                    return float("nan")

                metrics["hoi_err_fusion"].append(compute_hoi_error(pred_obj_trans_fusion))
                metrics["hoi_err_fk"].append(compute_hoi_error(pred_obj_trans_fk))
                metrics["hoi_err_imu"].append(compute_hoi_error(pred_obj_trans_imu))

                # Contact metrics
                pred_hand_contact_prob = pred_dict.get("pred_hand_contact_prob")
                if pred_hand_contact_prob is not None:
                    pred_hand_contact_prob = pred_hand_contact_prob[sample_idx]
                    pred_lhand = (pred_hand_contact_prob[:, 0] > 0.5).long().cpu().numpy()
                    pred_rhand = (pred_hand_contact_prob[:, 1] > 0.5).long().cpu().numpy()
                    pred_objc = (pred_hand_contact_prob[:, 2] > 0.5).long().cpu().numpy()

                    if gt_lhand_contact_seq is not None:
                        gt_lhand = gt_lhand_contact_seq.cpu().numpy().astype(int)
                        metrics["contact_f1_lhand"].append(
                            f1_score(gt_lhand, pred_lhand) if np.unique(gt_lhand).size > 1 else float("nan")
                        )
                    else:
                        metrics["contact_f1_lhand"].append(float("nan"))

                    if gt_rhand_contact_seq is not None:
                        gt_rhand = gt_rhand_contact_seq.cpu().numpy().astype(int)
                        metrics["contact_f1_rhand"].append(
                            f1_score(gt_rhand, pred_rhand) if np.unique(gt_rhand).size > 1 else float("nan")
                        )
                    else:
                        metrics["contact_f1_rhand"].append(float("nan"))

                    if gt_obj_contact_seq is not None:
                        gt_objc = gt_obj_contact_seq.cpu().numpy().astype(int)
                        metrics["contact_f1_obj"].append(
                            f1_score(gt_objc, pred_objc) if np.unique(gt_objc).size > 1 else float("nan")
                        )
                    else:
                        metrics["contact_f1_obj"].append(float("nan"))
                else:
                    metrics["contact_f1_lhand"].append(float("nan"))
                    metrics["contact_f1_rhand"].append(float("nan"))
                    metrics["contact_f1_obj"].append(float("nan"))

            if batch_idx % 50 == 0:
                print(f"Processed {batch_idx + 1}/{len(data_loader)} batches")

    avg_metrics = {}
    for key, values in metrics.items():
        valid = [v for v in values if not np.isnan(v)]
        avg_metrics[key] = float(np.mean(valid)) if valid else float("nan")

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate IMUHOI TransPose Model")
    parser.add_argument("--config", type=str, default="configs/IMUHOI_train.yaml", help="Path to config file")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=sorted(DEFAULT_DATASET_CONFIG.keys()),
        help="Evaluate only the specified dataset key. Default: run all configured datasets sequentially.",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Override model checkpoint path")
    parser.add_argument("--smpl_model_path", type=str, default=None, help="Override SMPL model path")
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default=None,
        help="Override test dataset directory. Defaults to dataset-specific configuration.",
    )
    parser.add_argument("--num_workers", type=int, default=None, help="Number of DataLoader workers")
    parser.add_argument("--no_eval_objects", action="store_true", help="Skip object-related metrics")
    parser.add_argument("--compare_3", action="store_true", help="Compute FK/IMU comparison metrics")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        dataset_runs = _build_dataset_runs(args)
    except Exception as exc:
        print(f"[Eval] Argument error: {exc}")
        return

    all_results = {}
    for dataset_name, dataset_cfg in dataset_runs:
        print(f"\n=== Evaluating dataset: {dataset_name} ===")
        config = load_config(args.config)
        if args.model_path:
            config.model_path = args.model_path
        elif "model_path" in dataset_cfg and dataset_cfg.get("model_path") is not None:
            config.model_path = str(dataset_cfg["model_path"])

        if args.num_workers is not None:
            config.num_workers = args.num_workers
        if not hasattr(config, "debug"):
            config.debug = False

        modules_override = dataset_cfg.get("modules")
        if modules_override:
            _apply_module_overrides(config, modules_override)

        if args.smpl_model_path:
            config.body_model_path = args.smpl_model_path

        smpl_model_path = config.get("body_model_path", "body_models/smplh/neutral/model.npz")
        try:
            smpl_model = load_smpl_model(smpl_model_path, device)
        except FileNotFoundError as exc:
            print(f"[Eval] Skipping '{dataset_name}': {exc}")
            continue

        model = load_model(config, device)

        data_dir_default = dataset_cfg.get("data_dir")
        if data_dir_default is None and not args.test_data_dir:
            print(f"[Eval] Skipping '{dataset_name}' (no dataset directory configured).")
            continue

        base_data_path = data_dir_default
        if base_data_path is None and args.test_data_dir:
            base_data_path = Path(args.test_data_dir).expanduser()

        data_override = args.test_data_dir if args.dataset else None
        data_path = _select_path(data_override, base_data_path)

        if not data_path.exists():
            print(f"[Eval] Skipping '{dataset_name}' (data not found at {data_path}).")
            continue

        print(f"Loading test dataset from: {data_path}")

        test_window = config.test.get("window", config.train.get("window", 60))
        test_dataset = IMUDataset(
            data_dir=str(data_path),
            window_size=test_window,
            debug=config.get("debug", False),
            full_sequence=True,
        )

        if len(test_dataset) == 0:
            print(f"[Eval] Skipping '{dataset_name}' (dataset is empty).")
            continue

        batch_size = 1
        num_workers = config.get("num_workers", 0)

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        print(f"Dataset size: {len(test_dataset)} | Loader batches: {len(test_loader)}")
        print(f"Batch size: {batch_size} | Num workers: {num_workers}")
        print(f"Evaluate objects: {not args.no_eval_objects}")

        results = evaluate_model(
            model,
            smpl_model,
            test_loader,
            config,
            device,
            evaluate_objects=not args.no_eval_objects,
            compare_three=args.compare_3,
        )

        all_results[dataset_name] = results

        def _fmt(key: str) -> str:
            val = results.get(key, float("nan"))
            return f"{val:.4f}" if not np.isnan(val) else "NaN"

        print("\n--- Evaluation Results ---")
        print(f"MPJPE (cm):                     {_fmt('mpjpe')}")
        print(f"MPJRE (deg):                    {_fmt('mpjre_angle')}")
        print(f"Root Trans Error (cm):          {_fmt('root_trans_err')}")
        print(f"Jitter (mm/frame^2):            {_fmt('jitter')}")

        if not args.no_eval_objects:
            print("\n--- Object Translation Errors ---")
            print(f"Fusion (cm):                    {_fmt('obj_trans_err_fusion')}")
            print(f"FK (cm):                        {_fmt('obj_trans_err_fk')}")
            print(f"IMU (cm):                       {_fmt('obj_trans_err_imu')}")

            print("\n--- HOI Errors ---")
            print(f"Fusion (cm):                    {_fmt('hoi_err_fusion')}")
            print(f"FK (cm):                        {_fmt('hoi_err_fk')}")
            print(f"IMU (cm):                       {_fmt('hoi_err_imu')}")
        else:
            print("Object-related metrics skipped.")

        print("\n--- Contact Prediction F1 ---")
        print(f"Left Hand:                      {_fmt('contact_f1_lhand')}")
        print(f"Right Hand:                     {_fmt('contact_f1_rhand')}")
        print(f"Object:                         {_fmt('contact_f1_obj')}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
