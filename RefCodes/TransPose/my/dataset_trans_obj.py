# -*- coding: utf-8 -*-
"""
优化的Dataset - 预加载到内存，减少运行时计算
"""
from __future__ import annotations

import glob
import os
from bisect import bisect_right
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from pytorch3d import transforms
from config import joint_set
from utils import normalize_and_concat


_SENSOR_POS_INDICES = [7, 8, 15, 20, 21, 0]
_SENSOR_ROT_INDICES = [4, 5, 15, 18, 19, 0]
_REDUCED_INDICES = joint_set.reduced
_LEAF_INDICES = joint_set.leaf
_FULL_INDICES = joint_set.full


def _ensure_24_joints(rot: torch.Tensor) -> torch.Tensor:
    if rot.shape[1] == 24:
        return rot
    if rot.shape[1] != 22:
        raise ValueError(f"Unexpected rotation dimension: {rot.shape}")
    padded = torch.zeros(rot.shape[0], 24, 3, 3, dtype=rot.dtype, device=rot.device)
    padded[:, :22] = rot
    return padded


def _ensure_24_joints_pos(pos: torch.Tensor) -> torch.Tensor:
    if pos.shape[1] == 24:
        return pos
    if pos.shape[1] != 22:
        raise ValueError(f"Unexpected position dimension: {pos.shape}")
    padded = torch.zeros(pos.shape[0], 24, 3, dtype=pos.dtype, device=pos.device)
    padded[:, :22] = pos
    return padded


def _central_diff(values: torch.Tensor, dt: float) -> torch.Tensor:
    if values.shape[0] <= 1:
        return torch.zeros_like(values)
    vel = torch.zeros_like(values)
    vel[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    vel[0] = (values[1] - values[0]) / dt
    vel[-1] = (values[-1] - values[-2]) / dt
    return vel

def _build_human_imu(rotation_global: torch.Tensor,
                     position_global: torch.Tensor,
                     fps: float) -> torch.Tensor:
    dt = 1.0 / fps
    sensors_rot = rotation_global[:, _SENSOR_ROT_INDICES]
    sensors_pos = position_global[:, _SENSOR_POS_INDICES]
    vel = _central_diff(sensors_pos, dt)
    acc = _central_diff(vel, dt)
    imu = normalize_and_concat(acc, sensors_rot)
    return imu.contiguous()


def _compute_joint_targets(rotation_global: torch.Tensor,
                           position_global: torch.Tensor,
                           fps: float) -> Dict[str, torch.Tensor]:
    dt = 1.0 / fps
    root_rot = rotation_global[:, 0]
    root_rot_T = root_rot.transpose(1, 2)
    root_pos_world = position_global[:, 0]
    root_pos_rel = root_pos_world - root_pos_world[:1]

    pos_rel = position_global - root_pos_world.unsqueeze(1)
    pos_root = torch.einsum('tij,tkj->tki', root_rot_T, pos_rel)

    leaf_pos = pos_root[:, _LEAF_INDICES]
    full_pos = pos_root[:, _FULL_INDICES]

    reduced_rot = rotation_global[:, _REDUCED_INDICES]
    reduced_rel = torch.matmul(root_rot_T.unsqueeze(1), reduced_rot)
    reduced_pose_6d = transforms.matrix_to_rotation_6d(reduced_rel).reshape(reduced_rel.shape[0], len(_REDUCED_INDICES), 6)
    root_vel_world = _central_diff(root_pos_world, dt)
    root_vel_local = torch.matmul(root_rot_T, root_vel_world.unsqueeze(-1)).squeeze(-1)

    root_pos_integrated = root_pos_rel

    return {
        'leaf_pos': leaf_pos.contiguous(),
        'full_pos': full_pos.contiguous(),
        'reduced_pose': reduced_pose_6d.contiguous(),
        'root_velocity_world': root_vel_world.contiguous(),
        'velocity_local': root_vel_local.contiguous(),
        'root_position_world': root_pos_integrated.contiguous(),
    }


def _build_object_targets(obj_rot: torch.Tensor,
                          obj_pos: torch.Tensor,
                          fps: float) -> Dict[str, torch.Tensor]:
    dt = 1.0 / fps
    if obj_rot.shape[-1] == 6:
        obj_rot_mat = transforms.rotation_6d_to_matrix(obj_rot)
    else:
        obj_rot_mat = obj_rot
    vel = _central_diff(obj_pos, dt)
    acc = _central_diff(vel, dt)
    imu = torch.cat((obj_rot_mat.reshape(obj_rot_mat.shape[0], -1), acc), dim=-1)
    pos_rel = obj_pos - obj_pos[:1]
    return {
        'obj_imu': imu.contiguous(),
        'obj_velocity': vel.contiguous(),
        'obj_position': pos_rel.contiguous(),
        'obj_rot': obj_rot_mat.contiguous(),
    }


def _trim_tensor(tensor: torch.Tensor, trim: int) -> torch.Tensor:
    if trim <= 0:
        return tensor.clone()
    if tensor.shape[0] <= trim * 2:
        return tensor.new_empty((0,) + tensor.shape[1:])
    return tensor[trim:-trim].clone()


def _convert_raw_sequence(raw: Dict[str, Any],
                          fps: float,
                          trim_frames: int) -> Tuple[Optional[Dict[str, torch.Tensor]], Dict[str, Any]]:
    if 'rotation_global' not in raw or 'position_global_full_gt_world' not in raw:
        return None, {}

    rotation_global = _ensure_24_joints(raw['rotation_global'].float())
    position_global = _ensure_24_joints_pos(raw['position_global_full_gt_world'].float())
    fps = float(fps)

    imu = _build_human_imu(rotation_global, position_global, fps)
    joint_targets = _compute_joint_targets(rotation_global, position_global, fps)

    lfoot_contact = raw.get('lfoot_contact', torch.zeros(rotation_global.shape[0], dtype=torch.float32))
    rfoot_contact = raw.get('rfoot_contact', torch.zeros_like(lfoot_contact))
    contact = torch.stack((lfoot_contact, rfoot_contact), dim=-1).float()

    obj_targets: Dict[str, torch.Tensor] = {}
    if 'obj_trans' in raw and 'obj_rot' in raw:
        obj_targets = _build_object_targets(raw['obj_rot'].float(), raw['obj_trans'].float(), fps)

    trim = max(0, int(trim_frames))
    meta = {
        'fps': fps,
        'trim_start': trim,
        'trim_end': rotation_global.shape[0] - trim if trim > 0 else rotation_global.shape[0],
        'original_length': rotation_global.shape[0],
    }

    def _trim_dict(data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: _trim_tensor(v, trim) for k, v in data.items()}

    imu_trim = _trim_tensor(imu, trim)
    if imu_trim.shape[0] < 2:
        return None, meta
    
    pose_6d_trim = _trim_tensor(raw['rotation_local_full_gt_list'].float(), trim)
    processed: Dict[str, torch.Tensor] = {
        'imu': imu_trim,
        'contact': _trim_tensor(contact, trim),
        'fps': torch.full((imu_trim.shape[0],), fps, dtype=torch.float32),
        'pose': pose_6d_trim,
    }
    processed.update(_trim_dict(joint_targets))
    if obj_targets:
        processed.update(_trim_dict(obj_targets))

    return processed, meta


def load_transpose_sequence(path: str,
                            fps: Optional[float] = None,
                            trim_frames: int = 6) -> Dict[str, Any]:
    raw: Dict[str, Any] = torch.load(path)
    fps_value = float(raw.get('fps', fps if fps is not None else 60.0))

    if all(key in raw for key in ('imu', 'leaf_pos', 'full_pos', 'reduced_pose')):
        processed = {k: v.float().contiguous() for k, v in raw.items() if isinstance(v, torch.Tensor)}
        meta = {
            'fps': fps_value,
            'trim_start': 0,
            'trim_end': processed['imu'].shape[0],
            'original_length': processed['imu'].shape[0],
        }
    else:
        processed, meta = _convert_raw_sequence(raw, fps_value, trim_frames)
        if processed is None:
            raise ValueError(f"Sequence {path} is invalid after trimming.")

    meta['path'] = path
    meta['trim_frames'] = trim_frames
    return {'processed': processed, 'meta': meta}


class TransPoseObjectDataset(Dataset):
    """
    优化的Dataset：
    1. 预加载所有数据到内存（可选择pin到GPU）
    2. 减少运行时的clone操作
    3. 优化slice和padding逻辑
    """

    def __init__(self,
                 datasets: List[str],
                 seq_len: int,
                 data_root: str,
                 subset: str = 'train',
                 random_sample: bool = True,
                 use_full_sequence: bool = False,
                 fps_default: float = 60.0,
                 trim_frames: int = 6,
                 pin_memory: bool = False,
                 device: Optional[str] = None):
        super().__init__()
        self.datasets = list(datasets)
        self.seq_len = int(seq_len)
        self.data_root = data_root
        self.subset = subset
        self.random_sample = random_sample
        self.use_full_sequence = use_full_sequence
        self.fps_default = float(fps_default)
        self.trim_frames = int(trim_frames)
        self.pin_memory = pin_memory
        self.device = device

        self.sequences: List[Dict[str, torch.Tensor]] = []
        self.sequence_lengths: List[int] = []
        self.samples_per_sequence: List[int] = []
        self.cumulative_samples: List[int] = []
        self.total_samples = 0

        self._prepare()

    def __len__(self) -> int:
        return self.total_samples

    def _prepare(self) -> None:
        """预加载所有数据到内存"""
        print("[Dataset] 开始预加载数据...")
        for dataset_name in self.datasets:
            dataset_dir = os.path.join(self.data_root, dataset_name, self.subset)
            if not os.path.isdir(dataset_dir):
                print(f"[Dataset] Missing directory: {dataset_dir}")
                continue

            for path in glob.glob(os.path.join(dataset_dir, '*.pt')):
                try:
                    bundle = load_transpose_sequence(path, fps=self.fps_default, trim_frames=self.trim_frames)
                except Exception as exc:
                    print(f"[Dataset] Failed to load {path}: {exc}")
                    continue

                # 预处理并固定到内存
                processed = {}
                for k, v in bundle['processed'].items():
                    if isinstance(v, torch.Tensor):
                        v = v.contiguous()
                        if self.pin_memory:
                            v = v.pin_memory()
                        if self.device is not None:
                            v = v.to(self.device)
                        processed[k] = v
                    else:
                        processed[k] = v

                length = processed['imu'].shape[0]
                if length < 2:
                    continue

                # 确保所有序列都有object数据（用零填充）
                if 'obj_imu' not in processed:
                    device_target = self.device if self.device else ('cpu')
                    processed['obj_imu'] = torch.zeros(length, 12, device=device_target)
                    processed['obj_velocity'] = torch.zeros(length, 3, device=device_target)
                    processed['obj_position'] = torch.zeros(length, 3, device=device_target)
                    processed['obj_rot'] = torch.zeros(length, 3, 3, device=device_target)
                    if self.pin_memory and device_target == 'cpu':
                        processed['obj_imu'] = processed['obj_imu'].pin_memory()
                        processed['obj_velocity'] = processed['obj_velocity'].pin_memory()
                        processed['obj_position'] = processed['obj_position'].pin_memory()
                        processed['obj_rot'] = processed['obj_rot'].pin_memory()

                self.sequences.append(processed)
                self.sequence_lengths.append(length)

        self._build_index()
        print(f"[Dataset] 已加载 {len(self.sequences)} 序列 ({self.total_samples} samples)")

    def _build_index(self) -> None:
        self.samples_per_sequence.clear()
        self.cumulative_samples.clear()
        total = 0
        for length in self.sequence_lengths:
            if self.use_full_sequence:
                count = 1
            elif self.random_sample:
                count = 1
            else:
                count = max(1, length - self.seq_len + 1)
            total += count
            self.samples_per_sequence.append(count)
            self.cumulative_samples.append(total)
        self.total_samples = total

    def _locate_sequence(self, global_idx: int) -> Tuple[int, int]:
        seq_idx = bisect_right(self.cumulative_samples, global_idx)
        prev_cum = 0 if seq_idx == 0 else self.cumulative_samples[seq_idx - 1]
        offset = global_idx - prev_cum
        return seq_idx, offset

    @staticmethod
    def _slice_and_pad(tensor: torch.Tensor, start: int, end: int, target_len: int) -> torch.Tensor:
        """优化的切片和padding - 使用更高效的方式"""
        window = tensor[start:end]
        cur_len = window.shape[0]
        if cur_len >= target_len:
            return window[:target_len].clone()
        
        # 使用expand而不是repeat可以减少内存分配
        pad_shape = [target_len - cur_len] + list(window.shape[1:])
        pad = window[-1:].expand(pad_shape).clone()
        return torch.cat((window, pad), dim=0)

    def _slice_sequence(self,
                        sequence: Dict[str, torch.Tensor],
                        start: int,
                        end: int) -> Dict[str, torch.Tensor]:
        return {
            key: self._slice_and_pad(value, start, end, self.seq_len)
            for key, value in sequence.items()
            if key != 'obj_pos_init'
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.total_samples == 0:
            raise IndexError("Dataset is empty.")

        seq_idx, offset = self._locate_sequence(index)
        sequence = self.sequences[seq_idx]
        length = self.sequence_lengths[seq_idx]

        if self.use_full_sequence:
            # 避免不必要的clone
            sample = {k: v for k, v in sequence.items()}
            sample['seq_len'] = torch.tensor(length, dtype=torch.long)
        else:
            if self.random_sample:
                start = 0 if length <= self.seq_len else torch.randint(0, length - self.seq_len + 1, (1,)).item()
            else:
                start = min(offset, max(0, length - self.seq_len))
            end = min(start + self.seq_len, length)
            sliced = self._slice_sequence(sequence, start, end)
            sample = {**sliced, 'seq_len': torch.tensor(self.seq_len, dtype=torch.long)}

        # 位置归零（相对位置）
        if 'root_position_world' in sample:
            root_pos = sample['root_position_world']
            sample['root_position_world'] = root_pos - root_pos[:1]
        if 'obj_position' in sample:
            obj_pos = sample['obj_position']
            sample['obj_pos_init'] = obj_pos[0].clone()
            sample['obj_position'] = obj_pos - obj_pos[:1]
        else:
            device_target = self.device if self.device else 'cpu'
            sample['obj_pos_init'] = torch.zeros(3, device=device_target)

        return sample


def collate_transpose(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """优化的collate - 减少不必要的操作"""
    if not batch:
        return {}
    
    keys = batch[0].keys()
    collated: Dict[str, torch.Tensor] = {}
    
    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            # 使用stack而不是多次cat
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = values
    return collated

