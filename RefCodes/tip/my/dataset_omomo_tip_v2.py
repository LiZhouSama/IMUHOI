"""
Dataset adapter that transforms OMOMO data to match TIP's original data format.
Returns tuple (x_imu, x_s, y_s_n) instead of dictionaries.
"""
import glob
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _rotmat_to_2axis(rot_mats: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to 2-axis representation (first two columns).
    This matches TIP's batch_to_rot_mat_2axis format.
    """
    if rot_mats.shape[-2:] != (3, 3):
        raise ValueError(f"Expect rotation matrices of shape (*,3,3), got {rot_mats.shape}")
    first_two = rot_mats[..., :3, :2]
    return first_two.reshape(*first_two.shape[:-2], 6)


def _second_diff_acc(positions: torch.Tensor, frame_rate: float) -> torch.Tensor:
    """Approximate accelerations via second-order finite difference."""
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


class OMOMODatasetTIPFormat(Dataset):
    """
    Dataset that loads OMOMO data and returns it in TIP's original format:
    - Returns tuple (x_imu, x_s, y_s_n) matching training_data_loader.py
    - Uses 2-axis rotation representation
    - Supports object IMU as additional sensor
    - State: [18*6 (rot 2axis), 3 (root_vel), 3 (obj_vel)] = 129 dims
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
        with_acc_sum: bool = False,
        random_sample: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        self.data_dirs = list(data_dirs)
        self.seq_len = int(seq_len)
        self.frame_rate = float(frame_rate)
        self.use_object_imu = bool(use_object_imu)
        self.acc_scale = float(acc_scale)
        self.with_acc_sum = bool(with_acc_sum)
        self.random_sample = bool(random_sample)

        if human_joint_indices_for_imu is None:
            human_joint_indices_for_imu = [0, 20, 21, 7, 8, 15]  # TIP's default: root, wrists, knees, neck
        self.imu_joint_idx = list(human_joint_indices_for_imu)
        self.num_human_imus = len(self.imu_joint_idx)
        self.num_obj_imus = 1 if self.use_object_imu else 0
        self.num_imus_total = self.num_human_imus + self.num_obj_imus

        self.tip_18_from_22_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19]
        self.human_joint_num_for_output = human_joint_num_for_output
        self.human_out_rot_dim = self.human_joint_num_for_output * 6
        self.root_vel_dim = 3
        self.obj_vel_dim = 3
        # State dimension: rot2axis + root_vel + obj_vel
        self.state_dim = self.human_out_rot_dim + self.root_vel_dim + self.obj_vel_dim

        self.imu_feat_per_sensor = 9  # 3 acc + 6 ori(2axis)
        self.input_imu_dim = self.num_imus_total * self.imu_feat_per_sensor
        if self.with_acc_sum:
            self.input_imu_dim += self.num_imus_total * 3  # add acc_sum

        self.files = self._gather_files()
        # Pre-load and process all sequences into memory (like TIP's preprocessed data)
        self.IMU: List[torch.Tensor] = []
        self.IMU_sum: List[torch.Tensor] = []
        self.S: List[torch.Tensor] = []
        self.sample_ranges: List[Tuple[int, int]] = []  # (seq_idx, max_start_idx)

        if not self.files:
            print(f"[OMOMODatasetTIPFormat] Warning: no .pt files found in {self.data_dirs}")
        else:
            self._load_all_sequences()
            self._build_sample_ranges()

    def _gather_files(self) -> List[str]:
        files: List[str] = []
        for data_dir in self.data_dirs:
            if os.path.exists(data_dir):
                files.extend(sorted(glob.glob(os.path.join(data_dir, "*.pt"))))
        return files

    def _load_all_sequences(self) -> None:
        """Load all sequences into memory and convert to TIP format."""
        for path in self.files:
            try:
                raw = torch.load(path, map_location="cpu")
            except Exception as exc:
                print(f"[OMOMODatasetTIPFormat] Skip {path}: {exc}")
                continue

            try:
                imu, imu_sum, state = self._convert_sequence_to_tip_format(raw)
            except Exception as exc:
                print(f"[OMOMODatasetTIPFormat] Failed to parse {path}: {exc}")
                continue

            if imu.shape[0] <= self.seq_len:
                continue

            self.IMU.append(imu)
            if self.with_acc_sum:
                self.IMU_sum.append(imu_sum)
            self.S.append(state)

        print(f"[OMOMODatasetTIPFormat] Loaded {len(self.IMU)} sequences")

    def _convert_sequence_to_tip_format(
        self, raw: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert raw OMOMO sequence to TIP format.
        Returns:
            imu: [T, num_imus * 9] - accelerations + 6D rotations
            imu_sum: [T, num_imus * 3] - cumulative accelerations
            state: [T+1, state_dim] - rot 2axis + root_vel + obj_vel
        """
        pos_global = _ensure_tensor(raw["position_global_full_gt_world"])
        rot_global = _ensure_tensor(raw["rotation_global"])
        motion_local = _ensure_tensor(raw["rotation_local_full_gt_list"])

        T = pos_global.shape[0]
        fps = self.frame_rate

        root_pos0 = pos_global[0, 0]
        root_rot0 = rot_global[0, 0]

        # ============ Build IMU features ============
        imu_list: List[torch.Tensor] = []

        for joint_idx in self.imu_joint_idx:
            pj = pos_global[:, joint_idx]
            Rj = rot_global[:, joint_idx]
            acc = _second_diff_acc(pj, fps)
            acc_n = _normalize_by_root(acc, root_rot0) / self.acc_scale
            Rj_n = _apply_rot_to_rotmats(Rj, root_rot0)
            ori_2axis = _rotmat_to_2axis(Rj_n)
            imu_list.append(torch.cat([acc_n, ori_2axis], dim=-1))

        has_object = "obj_trans" in raw and raw["obj_trans"] is not None
        obj_trans = _ensure_tensor(raw["obj_trans"]) if has_object else torch.zeros(T, 3)

        if self.use_object_imu:
            if has_object:
                obj_acc = _second_diff_acc(obj_trans, fps)
                obj_acc_n = _normalize_by_root(obj_acc, root_rot0) / self.acc_scale
                obj_rot = raw.get("obj_rot", None)
                if obj_rot is not None:
                    obj_rot_tensor = _ensure_tensor(obj_rot)
                    if obj_rot_tensor.dim() == 3 and obj_rot_tensor.shape[-1] == 6:
                        obj_ori_2axis = obj_rot_tensor
                    elif obj_rot_tensor.dim() == 4 and obj_rot_tensor.shape[-2:] == (3, 3):
                        obj_ori_2axis = _rotmat_to_2axis(_apply_rot_to_rotmats(obj_rot_tensor, root_rot0))
                    else:
                        obj_ori_2axis = torch.zeros(T, 6)
                else:
                    obj_ori_2axis = torch.zeros(T, 6)
                imu_obj = torch.cat([obj_acc_n, obj_ori_2axis], dim=-1)
            else:
                imu_obj = torch.zeros(T, 9)
            imu_list.append(imu_obj)

        imu_full = torch.cat(imu_list, dim=-1)  # [T, num_imus * 9]

        # ============ Build cumulative acceleration (if needed) ============
        imu_sum = torch.zeros(T, self.num_imus_total * 3)
        if self.with_acc_sum:
            acc_only = torch.zeros(T, self.num_imus_total * 3)
            for i in range(self.num_imus_total):
                acc_only[:, i*3:(i+1)*3] = imu_full[:, i*9:i*9+3]
            
            # Cumulative sum with window (matching TIP's ACC_SUM_WIN_LEN)
            win_len = 40
            b = torch.cumsum(acc_only, dim=0)
            if T > win_len:
                b[win_len:, :] = b[win_len:, :] - b[:-win_len, :]
            imu_sum = b / 15.0  # TIP's ACC_SUM_DOWN_SCALE

        # ============ Build state representation ============
        # Convert motion_local to 2-axis representation
        if motion_local.size(-1) == 22 * 6 and self.human_joint_num_for_output == 18:
            motion_chunks = motion_local.view(T, 22, 6)
            human_rot_2axis = motion_chunks[:, self.tip_18_from_22_idx, :].reshape(T, -1)
        else:
            human_rot_2axis = motion_local[:, : self.human_out_rot_dim]

        # Root velocity
        root_pos_n = _normalize_by_root(pos_global[:, 0] - root_pos0, root_rot0)
        root_vel = torch.zeros_like(root_pos_n)
        if root_pos_n.shape[0] > 1:
            root_vel[1:] = (root_pos_n[1:] - root_pos_n[:-1]) * fps

        # Object velocity
        obj_pos_n = _normalize_by_root(obj_trans - root_pos0, root_rot0) if has_object else torch.zeros(T, 3)
        obj_vel = torch.zeros_like(obj_pos_n)
        if obj_pos_n.shape[0] > 1:
            obj_vel[1:] = (obj_pos_n[1:] - obj_pos_n[:-1]) * fps

        # Concatenate state: [rot_2axis, root_vel, obj_vel]
        state_all = torch.cat([human_rot_2axis, root_vel, obj_vel], dim=-1)  # [T, state_dim]

        return imu_full.float(), imu_sum.float(), state_all.float()

    def _build_sample_ranges(self) -> None:
        """Build sample ranges for each sequence."""
        self.sample_ranges.clear()
        for seq_idx, seq_S in enumerate(self.S):
            T = seq_S.shape[0]
            if T <= self.seq_len:
                continue
            max_start = T - self.seq_len - 1  # -1 because we need T+1 for state
            self.sample_ranges.append((seq_idx, max_start))

        total_samples = sum(max_start + 1 for _, max_start in self.sample_ranges)
        print(f"[OMOMODatasetTIPFormat] Built {total_samples} potential samples from {len(self.sample_ranges)} sequences")

    def __len__(self) -> int:
        if self.random_sample:
            # In random sample mode, each sequence contributes 1 sample
            return len(self.sample_ranges)
        else:
            # In exhaustive mode, count all possible windows
            return sum(max_start + 1 for _, max_start in self.sample_ranges)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_imu: [seq_len, input_imu_dim]
            x_s: [seq_len, state_dim] - history state
            y_s_n: [seq_len, state_dim] - next state (target)
        """
        if not self.sample_ranges:
            raise IndexError("Empty dataset.")

        if self.random_sample:
            # Random sampling: pick a random sequence and random start
            seq_idx, max_start = self.sample_ranges[index % len(self.sample_ranges)]
            if max_start > 0:
                start = torch.randint(0, max_start + 1, (1,)).item()
            else:
                start = 0
        else:
            # Exhaustive sampling: map index to (seq_idx, start)
            cumsum = 0
            for seq_idx, max_start in self.sample_ranges:
                num_samples = max_start + 1
                if index < cumsum + num_samples:
                    start = index - cumsum
                    break
                cumsum += num_samples
            else:
                raise IndexError(f"Index {index} out of range")

        # Extract window
        imu = self.IMU[seq_idx][start : start + self.seq_len]
        state = self.S[seq_idx][start : start + self.seq_len + 1]

        # Split into input and target
        x_s = state[:-1]  # [seq_len, state_dim]
        y_s_n = state[1:]  # [seq_len, state_dim]

        # Concatenate acc_sum if needed
        if self.with_acc_sum:
            imu_sum = self.IMU_sum[seq_idx][start : start + self.seq_len]
            x_imu = torch.cat([imu, imu_sum], dim=-1)
        else:
            x_imu = imu

        return x_imu, x_s, y_s_n


__all__ = ["OMOMODatasetTIPFormat"]


