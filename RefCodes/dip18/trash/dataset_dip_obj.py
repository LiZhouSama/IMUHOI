import glob
import os
from bisect import bisect_right
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from my.dataset_dynaip_trans_obj import load_dynaip_sequence


def _finite_difference(sequence: torch.Tensor, dt: float) -> torch.Tensor:
    if sequence.shape[0] <= 1:
        return torch.zeros_like(sequence)
    diff = torch.zeros_like(sequence)
    diff[1:] = (sequence[1:] - sequence[:-1]) / dt
    diff[0] = diff[1]
    return diff


def _apply_zero_offset(position: torch.Tensor) -> torch.Tensor:
    if position.numel() == 0:
        return position
    offset = position[0].clone()
    return position - offset.unsqueeze(0)


class DIPObjectDataset(Dataset):
    """
    Loads raw motion sequences (same format as DynaIP) and prepares inputs/targets for DIP-style training.
    """

    def __init__(
        self,
        dataset_names: List[str],
        seq_len: int,
        data_root: str,
        subset: str = "train",
        random_sample: bool = True,
        use_full_sequence: bool = False,
        fps_override: Optional[float] = None,
        trim_frames: int = 6,
        imu_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.dataset_names = dataset_names
        self.seq_len = seq_len
        self.data_root = data_root
        self.subset = subset
        self.random_sample = random_sample
        self.use_full_sequence = use_full_sequence
        self.fps_override = fps_override
        self.trim_frames = trim_frames
        self.imu_noise_std = imu_noise_std

        self.sequences: List[Dict[str, torch.Tensor]] = []
        self.sequence_lengths: List[int] = []
        self.cumulative_sizes: List[int] = []
        self.total_samples: int = 0

        self._human_input_dim: Optional[int] = None
        self._object_input_dim: Optional[int] = None
        self._human_pose_dim: Optional[int] = None
        self._object_velocity_dim: Optional[int] = None

        self._load_sequences()
        self._build_index()

    @property
    def human_input_dim(self) -> int:
        return int(self._human_input_dim or 0)

    @property
    def object_input_dim(self) -> int:
        return int(self._object_input_dim or 0)

    @property
    def human_pose_dim(self) -> int:
        return int(self._human_pose_dim or 0)

    @property
    def object_velocity_dim(self) -> int:
        return int(self._object_velocity_dim or 0)

    def _load_sequences(self) -> None:
        for dataset_name in self.dataset_names:
            dataset_dir = os.path.join(self.data_root, dataset_name, self.subset)
            if not os.path.isdir(dataset_dir):
                print(f"[DIPObjectDataset] Warning: directory {dataset_dir} does not exist, skipping.")
                continue

            pt_files = sorted(glob.glob(os.path.join(dataset_dir, "*.pt")))
            if not pt_files:
                print(f"[DIPObjectDataset] Warning: no .pt files found in {dataset_dir}.")

            for pt_path in pt_files:
                try:
                    bundle = load_dynaip_sequence(
                        pt_path,
                        fps=self.fps_override,
                        trim_frames=self.trim_frames,
                        keep_raw_keys=False,
                    )
                    processed: Dict[str, Any] = bundle["processed"]
                    meta: Dict[str, Any] = bundle["meta"]
                    fps_value = float(meta.get("fps", 30.0))
                    dt = 1.0 / max(fps_value, 1e-8)

                    imu = processed["imu"]["imu"].float().contiguous()
                    T = imu.shape[0]
                    if T < 2:
                        continue

                    obj_imu = processed["imu"].get("obj_imu", None)
                    if obj_imu is None:
                        print(f"[DIPObjectDataset] Warning: missing obj_imu in {pt_path}, skipping sequence.")
                        continue
                    obj_imu = obj_imu.float().contiguous()

                    pose = processed["joint"]["ori_glb_reduced"].float().contiguous()
                    human_pose = pose.view(T, -1)
                    human_velocity = _finite_difference(human_pose, dt)

                    obj_velocity = processed.get("object", {}).get("velocity", None)
                    obj_position = processed.get("object", {}).get("position", None)
                    if obj_velocity is None or obj_position is None:
                        print(f"[DIPObjectDataset] Warning: missing object trajectory in {pt_path}, skipping sequence.")
                        continue
                    obj_velocity = obj_velocity.float().contiguous()
                    obj_position = _apply_zero_offset(obj_position.float().contiguous())

                    human_imu = imu.view(T, -1)
                    object_imu = obj_imu.view(T, -1)

                    sequence = {
                        "human_imu": human_imu,
                        "object_imu": object_imu,
                        "human_pose": human_pose,
                        "human_velocity": human_velocity,
                        "object_velocity": obj_velocity.view(T, -1),
                        "object_position": obj_position.view(T, -1),
                        "dt": torch.tensor(dt, dtype=torch.float32),
                    }
                    self.sequences.append(sequence)
                    self.sequence_lengths.append(T)
                except Exception as exc:
                    print(f"[DIPObjectDataset] Error loading {pt_path}: {exc}")
                    import traceback

                    traceback.print_exc()

        if not self.sequences:
            raise RuntimeError("No valid sequences loaded for DIPObjectDataset.")

        sample = self.sequences[0]
        self._human_input_dim = sample["human_imu"].shape[-1]
        self._object_input_dim = sample["object_imu"].shape[-1]
        self._human_pose_dim = sample["human_pose"].shape[-1]
        self._object_velocity_dim = sample["object_velocity"].shape[-1]

    def _build_index(self) -> None:
        self.cumulative_sizes.clear()
        total = 0
        for length in self.sequence_lengths:
            if self.use_full_sequence:
                # 使用完整序列模式：每个序列只返回1次完整数据
                steps = 1
            elif self.random_sample:
                # 随机采样模式：每个序列算1个样本（但每次随机起点）
                steps = 1
            else:
                # 滑动窗口模式：生成所有可能的窗口
                steps = max(1, length - self.seq_len + 1)
            total += steps
            self.cumulative_sizes.append(total)
        self.total_samples = total

    def __len__(self) -> int:
        return self.total_samples

    def _locate_sequence(self, index: int) -> Tuple[int, int]:
        seq_idx = bisect_right(self.cumulative_sizes, index)
        prev = 0 if seq_idx == 0 else self.cumulative_sizes[seq_idx - 1]
        offset = index - prev
        return seq_idx, offset

    def _slice_window(self, tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
        window = tensor[start:end].clone()
        cur_len = window.shape[0]
        if cur_len >= self.seq_len:
            return window[: self.seq_len]
        pad_len = self.seq_len - cur_len
        pad = window[-1:].repeat(pad_len, *([1] * (tensor.dim() - 1)))
        return torch.cat([window, pad], dim=0)

    def _add_noise(
        self, human_imu: torch.Tensor, object_imu: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.imu_noise_std <= 0:
            return human_imu, object_imu
        noise_human = torch.randn_like(human_imu) * self.imu_noise_std
        noise_object = torch.randn_like(object_imu) * self.imu_noise_std
        return human_imu + noise_human, object_imu + noise_object

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        seq_idx, offset = self._locate_sequence(index)
        sequence = self.sequences[seq_idx]
        length = self.sequence_lengths[seq_idx]

        if self.use_full_sequence:
            # 使用完整序列：直接返回整个序列，不做切片
            start = 0
            end = length
            human_imu = sequence["human_imu"]
            object_imu = sequence["object_imu"]
            human_pose = sequence["human_pose"]
            human_velocity = sequence["human_velocity"]
            object_velocity = sequence["object_velocity"]
            object_position = sequence["object_position"]
        else:
            # 固定长度窗口模式
            if self.random_sample:
                # 随机采样模式：每次随机选择起点
                if length > self.seq_len:
                    start = torch.randint(0, length - self.seq_len + 1, (1,)).item()
                else:
                    start = 0
            else:
                # 滑动窗口模式：使用预先计算的 offset
                start = min(offset, max(0, length - self.seq_len))
            end = min(start + self.seq_len, length)

            human_imu = self._slice_window(sequence["human_imu"], start, end)
            object_imu = self._slice_window(sequence["object_imu"], start, end)
            human_pose = self._slice_window(sequence["human_pose"], start, end)
            human_velocity = self._slice_window(sequence["human_velocity"], start, end)
            object_velocity = self._slice_window(sequence["object_velocity"], start, end)
            object_position = self._slice_window(sequence["object_position"], start, end)

        # 添加噪声
        human_imu, object_imu = self._add_noise(human_imu, object_imu)

        sample = {
            "human_imu": human_imu.contiguous(),
            "object_imu": object_imu.contiguous(),
            "human_pose": human_pose.contiguous(),
            "human_velocity": human_velocity.contiguous(),
            "object_velocity": object_velocity.contiguous(),
            "object_position": object_position.contiguous(),
            "object_init_pos": object_position[0].contiguous(),
        }
        return sample

