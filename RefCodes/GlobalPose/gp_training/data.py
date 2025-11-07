"""Utilities for preparing GlobalPose training sequences."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import articulate as art

from articulate.utils.torch import RNNWithInitDataset

_TORCH_DEVICE = torch.device("cpu")

# Indices reused from GPNet so we keep them in one place to avoid importing the heavy module at import time.
V_IMU = (1961, 5424, 1176, 4662, 411, 3021)
J_REDUCE = (1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19)
J_IGNORE = (0, 7, 8, 10, 11, 20, 21, 22, 23)
J_CONTACT = (0, 10, 11, 22, 23)
DT = 1.0 / 60.0
_GRAVITY = torch.tensor([0.0, -9.8, 0.0])


def _ensure_matrix_pose(pose: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle pose representation to rotation matrices."""
    if pose.ndim == 3 and pose.shape[-1] == 3:
        aa_pose = pose
    elif pose.ndim == 2 and pose.shape[1] == 72:
        aa_pose = pose.view(-1, 24, 3)
    else:
        raise ValueError(f"Unsupported pose tensor shape: {pose.shape}")
    return art.math.axis_angle_to_rotation_matrix(aa_pose).view(-1, 24, 3, 3)


def _expand_shape(shape: Optional[torch.Tensor], length: int, device: torch.device) -> Optional[torch.Tensor]:
    if shape is None:
        return None
    if shape.ndim == 1:
        shape = shape.unsqueeze(0)
    if shape.shape[-1] != 10:
        raise ValueError("Shape parameter must have length 10")
    return shape.to(device).expand(length, -1)


def _finite_difference(x: torch.Tensor, dt: float) -> torch.Tensor:
    vel = torch.zeros_like(x)
    if x.shape[0] > 1:
        vel[1:] = (x[1:] - x[:-1]) / dt
    return vel


@dataclass
class SequenceBuildResult:
    inputs: torch.Tensor
    targets: torch.Tensor
    meta: Dict[str, torch.Tensor]


def build_sequence_targets(record: Dict[str, torch.Tensor], body_model: art.ParametricModel,
                           device: Optional[torch.device] = None) -> Optional[SequenceBuildResult]:
    """Convert a raw GlobalPose record to network inputs and supervision targets."""
    device = device or _TORCH_DEVICE

    required = ["aS", "wS", "RIS", "RIM", "RSB"]
    for key in required:
        if key not in record:
            raise KeyError(f"Missing key '{key}' in record")

    aS = record["aS"].to(device)
    wS = record["wS"].to(device)
    RIS = record["RIS"].to(device)
    RIM = record["RIM"].to(device)
    RSB = record["RSB"].to(device)
    tran = record.get("tran", torch.zeros(aS.shape[0], 3, device=device)).to(device)
    pose_tensor = record.get("pose")
    if pose_tensor is None:
        for alt_key in ("AMASS_pose", "DIP_pose"):
            if alt_key in record:
                pose_tensor = record[alt_key]
                break
    if pose_tensor is None:
        raise KeyError('pose')
    pose_local = _ensure_matrix_pose(pose_tensor.to(device))
    seq_len = pose_local.shape[0]

    if seq_len < 4:
        return None

    g_vec = _GRAVITY.to(device)
    rim_t = RIM.transpose(-1, -2).unsqueeze(0)
    rot_im_to_model = torch.matmul(rim_t, RIS)
    RMB = torch.matmul(rot_im_to_model, RSB.unsqueeze(0))

    aM = torch.matmul(rot_im_to_model, aS.unsqueeze(-1)).squeeze(-1) + g_vec
    wM = torch.matmul(rot_im_to_model, wS.unsqueeze(-1)).squeeze(-1)

    root_rot = RMB[:, 5]
    aRB = torch.matmul(aM, root_rot)
    wRB = torch.matmul(wM, root_rot)
    RRB = torch.matmul(root_rot.transpose(1, 2).unsqueeze(1), RMB[:, :5])
    gR0 = -root_rot[:, 1]

    inputs = torch.cat([
        aRB.reshape(seq_len, -1),
        wRB.reshape(seq_len, -1),
        RRB.reshape(seq_len, -1),
        gR0
    ], dim=-1)

    shape_params = _expand_shape(record.get("shape"), seq_len, device)
    pose_global, joints_global, verts_global = body_model.forward_kinematics(
        pose_local, shape_params, tran, calc_mesh=True)

    root_global = pose_global[:, 0]
    pRB = torch.matmul((verts_global[:, :5] - verts_global[:, 5:6]).view(seq_len, 5, 3), root_global)
    gR = -root_global[:, 1]

    pose_canon = pose_local.clone()
    identity = torch.eye(3, device=device)
    for j in J_IGNORE:
        pose_canon[:, j] = identity

    _, joints_canon = body_model.forward_kinematics(pose_canon, shape_params, None, calc_mesh=False)
    pRJ = joints_canon[:, 1:]

    root_inv = root_global.transpose(1, 2)
    global_canon = torch.matmul(root_inv.unsqueeze(1), pose_global)
    for j in J_IGNORE:
        global_canon[:, j] = identity
    rrj = art.math.rotation_matrix_to_r6d(global_canon[:, J_REDUCE].reshape(-1, 3, 3)).view(seq_len, -1)

    root_velocity_world = _finite_difference(tran, DT)
    vRR_V = root_velocity_world[:, 1]
    vRR_H = torch.matmul(root_inv, root_velocity_world.unsqueeze(-1)).squeeze(-1)

    contact_positions = joints_global[:, J_CONTACT]
    contact_velocity = _finite_difference(contact_positions, DT)
    speed = contact_velocity.norm(dim=-1)
    ground = contact_positions.min(dim=1, keepdim=True).values[:, :, 1]
    height = contact_positions[:, :, 1]
    near_ground = (height - ground) < 0.06
    stationary = (speed < 0.35) & near_ground
    stationary_prob = stationary.float()

    targets = torch.cat([
        pRB.view(seq_len, -1),
        gR,
        pRJ.view(seq_len, -1),
        gR,
        rrj,
        torch.cat([vRR_V.unsqueeze(-1), vRR_H], dim=-1),
        stationary_prob
    ], dim=-1)

    if torch.isnan(inputs).any() or torch.isnan(targets).any():
        return None

    return SequenceBuildResult(inputs=inputs.float(), targets=targets.float(),
                               meta={"tran": tran, "root_rot": root_global})


class GlobalPoseDataset(RNNWithInitDataset):
    """Dataset that prepares sequences for training GPNet."""

    def __init__(self, data_files: Sequence[str], body_model: Optional[art.ParametricModel] = None,
                 sequence_len: int = 240, drop_last: bool = False,
                 min_seq_len: int = 60, device: Optional[torch.device] = None):
        self.body_model = body_model or art.ParametricModel('models/SMPL_male.pkl', vert_mask=V_IMU)
        self.sequence_sources: List[Tuple[str, int]] = []
        data_tensors: List[torch.Tensor] = []
        target_tensors: List[torch.Tensor] = []
        device = device or _TORCH_DEVICE

        for file_path in data_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(file_path)
            records = torch.load(file_path, map_location=device)
            keys = [k for k in records.keys() if isinstance(records[k], (list, tuple))]
            num_sequences = len(records[keys[0]]) if keys else 0
            for idx in range(num_sequences):
                record = {k: records[k][idx] for k in keys if len(records[k]) > idx}
                result = build_sequence_targets(record, self.body_model, device=device)
                if result is None:
                    continue
                if result.inputs.shape[0] < min_seq_len:
                    continue
                data_tensors.append(result.inputs)
                target_tensors.append(result.targets)
                self.sequence_sources.append((file_path, idx))

        if not data_tensors:
            raise RuntimeError("No valid sequences found for training")

        super().__init__(data_tensors, target_tensors, split_size=sequence_len, device=device, drop_last=drop_last)
        self.num_sequences = len(data_tensors)

    def stats(self) -> Dict[str, float]:
        lengths = [d.shape[0] for d in self.data]
        return {
            "num_sequences": float(self.num_sequences),
            "mean_length": float(sum(lengths) / len(lengths)),
            "min_length": float(min(lengths)),
            "max_length": float(max(lengths)),
        }
