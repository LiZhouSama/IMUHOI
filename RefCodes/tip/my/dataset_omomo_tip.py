import glob
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _rotmat_to_6d(rot_mats: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to their first two columns (6D representation)."""
    if rot_mats.shape[-2:] != (3, 3):
        raise ValueError(f"Expect rotation matrices of shape (*,3,3), got {rot_mats.shape}")
    first_two = rot_mats[..., :3, :2]
    return first_two.reshape(*first_two.shape[:-2], 6)


def _second_diff_acc(positions: torch.Tensor, frame_rate: float) -> torch.Tensor:
    """Approximate accelerations via a second-order finite difference."""
    T = positions.shape[0]
    acc = torch.zeros_like(positions)
    if T > 2:
        acc[2:] = (positions[:-2] + positions[2:] - 2.0 * positions[1:-1]) * (frame_rate ** 2)
        acc[1] = (positions[1] - positions[0]) * (frame_rate ** 2)
    return acc


def _ensure_tensor(data: Any) -> torch.Tensor:
    """Convert numpy arrays or lists into float tensors."""
    if torch.is_tensor(data):
        return data.float()
    return torch.from_numpy(np.asarray(data)).float()


def _normalize_by_root(vecs: torch.Tensor, root_rot0: torch.Tensor) -> torch.Tensor:
    """Transform vectors into the first-frame root coordinate system."""
    R_inv = root_rot0.transpose(0, 1)
    return vecs @ R_inv


def _apply_rot_to_rotmats(rotmats: torch.Tensor, root_rot0: torch.Tensor) -> torch.Tensor:
    """Left-multiply rotation matrices by the inverse of the first-frame root rotation."""
    R_inv = root_rot0.transpose(0, 1)
    return torch.einsum("ij,tjk->tik", R_inv, rotmats)


@dataclass
class SequenceWindow:
    """Container that stores per-sequence tensors for efficient window slicing."""

    name: str
    imu: torch.Tensor
    state_hist: torch.Tensor
    state_target: torch.Tensor
    root_target: torch.Tensor
    obj_pos_full: torch.Tensor
    gt_pos: Optional[torch.Tensor]
    gt_motion: Optional[torch.Tensor]
    contacts: Optional[torch.Tensor]
    metadata: Dict[str, Any]

    @property
    def length(self) -> int:
        return int(self.imu.shape[0])

    def slice(self, start: int, length: Optional[int] = None) -> Dict[str, Any]:
        end = self.length if length is None else min(start + length, self.length)
        if start < 0 or start >= self.length or end <= start:
            raise IndexError(f"Invalid slice [{start}, {end}) for sequence '{self.name}' (length={self.length})")

        window = {
            "imu": self.imu[start:end].clone(),
            "state_hist": self.state_hist[start:end].clone(),
            "state_target": self.state_target[start:end].clone(),
            "root_target": self.root_target[start:end].clone(),
            "obj_pos_gt": self.obj_pos_full[start + 1 : end + 1].clone(),
            "obj_pos_init": self.obj_pos_full[start].clone(),
            "metadata": dict(self.metadata),
        }

        if self.gt_pos is not None:
            window["gt_pos"] = self.gt_pos[start:end].clone()
        if self.gt_motion is not None:
            window["gt_motion"] = self.gt_motion[start:end].clone()
        if self.contacts is not None:
            window["contacts"] = self.contacts[start:end].clone()
        return window


class OMOMODatasetWithObject(Dataset):
    """
    Adapter that follows the DynaIP dataset structure but keeps TIP-specific supervision.

    Each sample returns a dictionary with:
        - imu: [L, (N_human + N_obj) * 9] accelerations + 6D rotations per sensor
        - state_hist/state_target: [L, state_dim] with human rot6d + root supervision + object velocity
        - obj_pos_gt / obj_pos_init: object translation in the root frame
        - optional GT human positions, local pose, and contact flags
    """

    def __init__(
        self,
        data_dirs: Sequence[str],
        seq_len: int = 60,
        frame_rate: float = 30.0,
        use_object_imu: bool = True,
        human_joint_num_for_output: int = 18,
        human_joint_indices_for_imu: Optional[Sequence[int]] = None,
        acc_scale: float = 1.0,
        root_supervision: str = "vel",
        return_gt_pos: bool = False,
        return_gt_motion: bool = False,
        return_contacts: bool = False,
        use_full_sequence: bool = False,
        random_sample: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        self.data_dirs = list(data_dirs)
        self.seq_len = int(seq_len)
        self.frame_rate = float(frame_rate)
        self.use_object_imu = bool(use_object_imu)
        self.acc_scale = float(acc_scale)
        self.use_full_sequence = bool(use_full_sequence)
        self.random_sample = bool(random_sample)

        if root_supervision not in ("vel", "pos"):
            raise ValueError("root_supervision must be 'vel' or 'pos'")
        self.root_supervision = root_supervision
        self.return_gt_pos = bool(return_gt_pos)
        self.return_gt_motion = bool(return_gt_motion)
        self.return_contacts = bool(return_contacts)

        if human_joint_indices_for_imu is None:
            human_joint_indices_for_imu = [0, 20, 21, 7, 8, 15]
        self.imu_joint_idx = list(human_joint_indices_for_imu)
        self.num_human_imus = len(self.imu_joint_idx)
        self.num_obj_imus = 1 if self.use_object_imu else 0
        self.num_imus_total = self.num_human_imus + self.num_obj_imus

        self.tip_18_from_22_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19]
        self.human_joint_num_for_output = human_joint_num_for_output
        self.human_out_rot_dim = self.human_joint_num_for_output * 6
        self.root_pos_dim = 3
        self.obj_pos_dim = 3
        self.state_dim = self.human_out_rot_dim + self.root_pos_dim + self.obj_pos_dim
        self.imu_feat_per_sensor = 9
        self.input_imu_dim = self.num_imus_total * self.imu_feat_per_sensor

        self.files = self._gather_files()
        self.sequences: List[SequenceWindow] = []
        self.sample_index: List[Tuple[int, int]] = []

        if not self.files:
            print(f"[OMOMODatasetWithObject] Warning: no .pt files found in {self.data_dirs}")
        else:
            self._build_sequences()
            self._build_index()

    # ------------------------------------------------------------------ #
    # Sequence construction
    # ------------------------------------------------------------------ #
    def _gather_files(self) -> List[str]:
        files: List[str] = []
        for data_dir in self.data_dirs:
            if os.path.exists(data_dir):
                files.extend(sorted(glob.glob(os.path.join(data_dir, "*.pt"))))
        return files

    def _build_sequences(self) -> None:
        for path in self.files:
            try:
                raw = torch.load(path, map_location="cpu")
            except Exception as exc:  # noqa: BLE001
                print(f"[OMOMODatasetWithObject] Skip {path}: {exc}")
                continue

            try:
                sequence = self._convert_raw_sequence(raw, path)
            except Exception as exc:  # noqa: BLE001
                print(f"[OMOMODatasetWithObject] Failed to parse {path}: {exc}")
                continue

            if sequence.length <= 0:
                continue
            self.sequences.append(sequence)

    def _build_index(self) -> None:
        self.sample_index.clear()
        for seq_idx, seq in enumerate(self.sequences):
            if self.use_full_sequence:
                # 使用完整序列模式：每个序列只返回1次完整数据
                self.sample_index.append((seq_idx, 0))
            elif self.random_sample:
                # 随机采样模式：每个序列算1个样本（但每次随机起点）
                self.sample_index.append((seq_idx, 0))
            else:
                # 滑动窗口模式：生成所有可能的窗口
                if seq.length <= self.seq_len:
                    self.sample_index.append((seq_idx, 0))
                else:
                    max_start = seq.length - self.seq_len
                    for start in range(max_start + 1):
                        self.sample_index.append((seq_idx, start))

        if not self.sample_index:
            print("[OMOMODatasetWithObject] Warning: no valid windows constructed.")

    def _convert_raw_sequence(self, raw: Dict[str, Any], path: str) -> SequenceWindow:
        pos_global = _ensure_tensor(raw["position_global_full_gt_world"])
        rot_global = _ensure_tensor(raw["rotation_global"])
        motion_local = _ensure_tensor(raw["rotation_local_full_gt_list"])

        T = pos_global.shape[0]
        fps = self.frame_rate

        root_pos0 = pos_global[0, 0]
        root_rot0 = rot_global[0, 0]

        imu_list: List[torch.Tensor] = []

        for joint_idx in self.imu_joint_idx:
            pj = pos_global[:, joint_idx]
            Rj = rot_global[:, joint_idx]
            acc = _second_diff_acc(pj, fps)
            acc_n = _normalize_by_root(acc, root_rot0) / self.acc_scale
            Rj_n = _apply_rot_to_rotmats(Rj, root_rot0)
            ori6d = _rotmat_to_6d(Rj_n)
            imu_list.append(torch.cat([acc_n, ori6d], dim=-1))

        has_object = "obj_trans" in raw and raw["obj_trans"] is not None
        obj_trans = _ensure_tensor(raw["obj_trans"]) if has_object else torch.zeros(T, 3)
        obj_rot = raw.get("obj_rot", None)

        if self.use_object_imu:
            if has_object:
                obj_acc = _second_diff_acc(obj_trans, fps)
                obj_acc_n = _normalize_by_root(obj_acc, root_rot0) / self.acc_scale
                if obj_rot is not None:
                    obj_rot_tensor = _ensure_tensor(obj_rot)
                    if obj_rot_tensor.dim() == 3 and obj_rot_tensor.shape[-1] == 6:
                        obj_ori6d = obj_rot_tensor
                    elif obj_rot_tensor.dim() == 4 and obj_rot_tensor.shape[-2:] == (3, 3):
                        obj_ori6d = _rotmat_to_6d(_apply_rot_to_rotmats(obj_rot_tensor, root_rot0))
                    else:
                        obj_ori6d = torch.zeros(T, 6)
                else:
                    obj_ori6d = torch.zeros(T, 6)
                imu_obj = torch.cat([obj_acc_n, obj_ori6d], dim=-1)
            else:
                imu_obj = torch.zeros(T, 9)
            imu_list.append(imu_obj)

        imu_full = torch.cat(imu_list, dim=-1)

        if motion_local.size(-1) == 22 * 6 and self.human_joint_num_for_output == 18:
            motion_chunks = motion_local.view(T, 22, 6)
            human_rot6d = motion_chunks[:, self.tip_18_from_22_idx, :].reshape(T, -1)
        else:
            human_rot6d = motion_local[:, : self.human_out_rot_dim]

        root_pos_n = _normalize_by_root(pos_global[:, 0] - root_pos0, root_rot0)
        if self.root_supervision == "vel":
            root_vel = torch.zeros_like(root_pos_n)
            if root_pos_n.shape[0] > 1:
                root_vel[1:] = (root_pos_n[1:] - root_pos_n[:-1]) * fps
            root_target = root_vel
        else:
            root_target = root_pos_n

        obj_pos_n = _normalize_by_root(obj_trans - root_pos0, root_rot0) if has_object else torch.zeros(T, 3)
        obj_vel = torch.zeros_like(obj_pos_n)
        if obj_pos_n.shape[0] > 1:
            obj_vel[1:] = (obj_pos_n[1:] - obj_pos_n[:-1]) * fps

        state_all = torch.cat([human_rot6d, root_target, obj_vel], dim=-1)

        imu_hist = imu_full[:-1]
        state_hist = state_all[:-1]
        state_target = state_all[1:]
        root_hist = root_target[:-1]
        obj_pos_full = obj_pos_n.clone()

        gt_pos = None
        if self.return_gt_pos and "position_global_full_gt_world" in raw:
            gt_pos_full = pos_global[:, self.tip_18_from_22_idx]
            gt_pos = _normalize_by_root(gt_pos_full - root_pos0.unsqueeze(0), root_rot0)[1:].clone()

        gt_motion = None
        if self.return_gt_motion:
            gt_motion = motion_local[1:].clone()

        contacts = None
        if self.return_contacts:
            lh = raw.get("lhand_contact", None)
            rh = raw.get("rhand_contact", None)
            if lh is not None and rh is not None:
                lh_tensor = _ensure_tensor(lh)[1:].bool()
                rh_tensor = _ensure_tensor(rh)[1:].bool()
                contacts = torch.stack([lh_tensor, rh_tensor], dim=-1)

        metadata = {
            "name": os.path.basename(path),
            "frame_rate": fps,
            "obj_name": raw.get("obj_name", None),
            "obj_scale": raw.get("obj_scale", None),
            "root_supervision": self.root_supervision,
        }

        return SequenceWindow(
            name=os.path.basename(path),
            imu=imu_hist.float(),
            state_hist=state_hist.float(),
            state_target=state_target.float(),
            root_target=root_hist.float(),
            obj_pos_full=obj_pos_full.float(),
            gt_pos=gt_pos.float() if gt_pos is not None else None,
            gt_motion=gt_motion.float() if gt_motion is not None else None,
            contacts=contacts,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # Dataset API
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not self.sample_index:
            raise IndexError("Empty dataset.")
        seq_idx, start = self.sample_index[index]
        sequence = self.sequences[seq_idx]

        if self.use_full_sequence:
            # 返回完整序列
            return sequence.slice(0, None)

        if self.random_sample:
            # 随机采样模式：每次随机选择起点
            if sequence.length > self.seq_len:
                start = torch.randint(0, sequence.length - self.seq_len + 1, (1,)).item()
            else:
                start = 0
        # 否则使用 _build_index 中预先计算的 start 位置

        return sequence.slice(start, self.seq_len)


def collate_tip_with_object(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack tensor fields across the batch and keep non-tensor metadata as lists."""
    if len(batch) == 0:
        return {}

    collated: Dict[str, Any] = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            # 检查是否需要padding（序列长度不同）
            if values[0].dim() > 0 and any(v.shape[0] != values[0].shape[0] for v in values):
                # 找到最大序列长度
                max_len = max(v.shape[0] for v in values)
                padded_values = []
                lengths = []
                
                for v in values:
                    seq_len = v.shape[0]
                    lengths.append(seq_len)
                    if seq_len < max_len:
                        # padding到最大长度
                        pad_size = max_len - seq_len
                        pad_shape = (pad_size,) + v.shape[1:]
                        padding = torch.zeros(pad_shape, dtype=v.dtype, device=v.device)
                        padded_v = torch.cat([v, padding], dim=0)
                        padded_values.append(padded_v)
                    else:
                        padded_values.append(v)
                
                collated[key] = torch.stack(padded_values, dim=0)
                # 保存长度信息，方便后续使用
                if "seq_lengths" not in collated:
                    collated["seq_lengths"] = torch.tensor(lengths, dtype=torch.long)
            else:
                collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = values
    return collated


__all__ = [
    "OMOMODatasetWithObject",
    "collate_tip_with_object",
    "_rotmat_to_6d",
    "_second_diff_acc",
]
