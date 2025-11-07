# -*- coding: utf-8 -*-
import os
import glob
from bisect import bisect_right
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

def normalize_imu(acc, ori):
    r"""
    normalize imu w.r.t the root sensor
    """
    acc = acc.view(-1, 6, 3)
    ori = ori.view(-1, 6, 3, 3)
    acc = torch.cat((acc[:, :1], acc[:, 1:] - acc[:, :1]), dim=1).bmm(ori[:, 0])
    ori = torch.cat((ori[:, :1], ori[:, :1].transpose(2, 3).matmul(ori[:, 1:])), dim=1)
    data = torch.cat((ori.view(-1, 6, 9), acc), dim=-1)
    return data

# Sensor indices follow the original DynaIP/Xsens convention
_SENSOR_POS_INDICES = [0, 7, 8, 15, 20, 21]
_SENSOR_ROT_INDICES = [0, 4, 5, 15, 18, 19]


def _central_diff(a: torch.Tensor, dt: float) -> torch.Tensor:
    if a.shape[0] <= 1:
        return torch.zeros_like(a)
    vel = torch.zeros_like(a)
    vel[1:-1] = (a[2:] - a[:-2]) / (2.0 * dt)
    vel[0] = (a[1] - a[0]) / dt
    vel[-1] = (a[-1] - a[-2]) / dt
    return vel


def _build_imu_from_joints(rotation_global: torch.Tensor,
                           position_global: torch.Tensor,
                           fps: float) -> torch.Tensor:
    T = rotation_global.shape[0]
    dt = 1.0 / fps
    sel_R = rotation_global[:, _SENSOR_ROT_INDICES]  # [T,6,3,3]
    sel_pos = position_global[:, _SENSOR_POS_INDICES]  # [T,6,3]
    vel = _central_diff(sel_pos, dt)
    acc = _central_diff(vel, dt)  # [T,6,3]
    imu = normalize_imu(acc.view(T, 6, 3), sel_R.view(T, 6, 3, 3))
    return imu  # [T,6,12]


def _rotation_6d_to_matrix(r6: torch.Tensor) -> torch.Tensor:
    a1 = r6[..., :3]
    a2 = r6[..., 3:6]
    a1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
    a2 = a2 - (a1 * a2).sum(dim=-1, keepdim=True) * a1
    a2 = a2 / (torch.norm(a2, dim=-1, keepdim=True) + 1e-8)
    a3 = torch.cross(a1, a2, dim=-1)
    return torch.stack((a1, a2, a3), dim=-1)


def _build_object_imu(obj_rot: torch.Tensor, obj_pos: torch.Tensor, fps: float) -> torch.Tensor:
    if obj_rot.shape[-1] == 6:
        rot_matrix = _rotation_6d_to_matrix(obj_rot)
    else:
        rot_matrix = obj_rot
    orientation = rot_matrix.reshape(rot_matrix.shape[0], 9)
    vel = _central_diff(obj_pos, 1.0 / fps)
    acc = _central_diff(vel, 1.0 / fps)
    return torch.cat([orientation, acc], dim=-1)


def _trim_tensor(t: torch.Tensor, trim: int) -> torch.Tensor:
    if trim <= 0:
        return t.clone()
    if t.shape[0] <= 2 * trim:
        return t.new_empty((0,) + t.shape[1:])
    return t[trim:-trim].clone()


def _convert_raw_sequence(raw: Dict[str, Any], fps: float, trim_frames: int) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    if 'rotation_global' not in raw or 'position_global_full_gt_world' not in raw:
        return None, {}

    rot = raw['rotation_global'].float()
    pose = raw['rotation_local_full_gt_list'].float()
    trans = torch.as_tensor(raw['trans']).float()
    # [T, 22, 3, 3] => [T, 24, 3, 3], with zeros in pad positions
    if rot.shape[1] == 22:
        T = rot.shape[0]
        rot_padded = torch.zeros(T, 24, 3, 3, dtype=rot.dtype, device=rot.device)
        rot_padded[:, :22] = rot
        rot = rot_padded
    pos = raw['position_global_full_gt_world'].float()
    T = rot.shape[0]

    imu = _build_imu_from_joints(rot, pos, fps)

    obj_rot = raw.get('obj_rot', None)
    obj_trans = raw.get('obj_trans', None)
    obj_imu = None
    obj_velocity = None
    obj_position = None
    if obj_rot is not None and obj_trans is not None:
        obj_rot = obj_rot.float()
        obj_trans = obj_trans.float()
        obj_imu = _build_object_imu(obj_rot, obj_trans, fps)
        obj_velocity = _central_diff(obj_trans, 1.0 / fps)
        obj_position = obj_trans

    trim = max(0, int(trim_frames))
    meta = {
        'trim_start': trim,
        'trim_end': T - trim if trim > 0 else T,
        'original_length': T,
        'fps': fps,
    }

    imu_trim = _trim_tensor(imu, trim)
    if imu_trim.shape[0] < 2:
        return None, meta

    pose_trim = _trim_tensor(pose, trim)
    trans_trim = _trim_tensor(trans, trim)
    processed: Dict[str, Any] = {
        'imu': {
            'imu': imu_trim.contiguous(),
        },
        'joint': {
            'pose': pose_trim.contiguous(),
            'trans': trans_trim.contiguous(),
        },
    }

    if obj_imu is not None and obj_velocity is not None and obj_position is not None:
        processed['imu']['obj_imu'] = _trim_tensor(obj_imu, trim).contiguous()
        processed['object'] = {
            'velocity': _trim_tensor(obj_velocity, trim).contiguous(),
            'position': _trim_tensor(obj_position, trim).contiguous(),
            'rot': _trim_tensor(obj_rot, trim).contiguous(),
        }

    return processed, meta



def load_dynaip_sequence(pt_path: str,
                         fps: Optional[float] = None,
                         trim_frames: int = 6,
                         keep_raw_keys: bool = False) -> Dict[str, Any]:
    raw_data: Dict[str, Any] = torch.load(pt_path)
    fps_value = float(raw_data.get('fps', fps if fps is not None else 30.0))

    processed, meta = _convert_raw_sequence(raw_data, fps_value, trim_frames)
    if processed is None:
        raise ValueError('Sequence too short after trimming or missing required keys.')

    meta['path'] = pt_path
    meta['trim_frames'] = trim_frames

    result: Dict[str, Any] = {
        'processed': processed,
        'meta': meta,
    }
    if keep_raw_keys:
        result['raw'] = raw_data
    return result


class MotionDatasetWithObjectAndTrans(Dataset):
    """Dataset that lazily converts raw preprocessed sequences to the DynaIP format."""

    def __init__(self,
                 datasets: List[str],
                 seq_len: int,
                 data_root: str,
                 device: str = 'cuda:0',
                 subset: str = 'train',
                 random_sample: bool = True,
                 fps: float = 30.0,
                 trim_frames: int = 6,
                 use_full_sequence: bool = False,
                 imu_noise_std: float = 0.0):
        super().__init__()
        self.datasets = datasets
        self.seq_len = seq_len
        self.data_root = data_root
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.subset = subset
        self.random_sample = random_sample
        self.fps_default = fps
        self.trim_frames = trim_frames
        self.use_full_sequence = use_full_sequence  # 新增：是否使用完整序列
        self.imu_noise_std = imu_noise_std  # 新增：IMU噪声标准差

        self.sequences: List[Dict[str, torch.Tensor]] = []
        self.sequence_lengths: List[int] = []
        self.samples_per_sequence: List[int] = []
        self.cumulative_samples: List[int] = []
        self.total_samples: int = 0

        self._prepare()

    def __len__(self) -> int:
        return self.total_samples

    def _build_index(self) -> None:
        self.samples_per_sequence.clear()
        self.cumulative_samples.clear()
        total = 0
        for length in self.sequence_lengths:
            if self.use_full_sequence:
                # 使用完整序列模式：每个序列只返回1次完整数据
                num = 1
            elif self.random_sample:
                # 随机采样模式：每个序列算1个样本（但每次随机起点）
                num = 1
            else:
                # 滑动窗口模式：生成所有可能的窗口
                num = max(1, length - self.seq_len + 1)
            self.samples_per_sequence.append(num)
            total += num
            self.cumulative_samples.append(total)
        self.total_samples = total

    def _locate_sequence(self, global_idx: int) -> Tuple[int, int]:
        seq_idx = bisect_right(self.cumulative_samples, global_idx)
        prev_cum = 0 if seq_idx == 0 else self.cumulative_samples[seq_idx - 1]
        offset = global_idx - prev_cum
        return seq_idx, offset

    def _prepare(self) -> None:
        for dataset in self.datasets:
            dataset_dir = os.path.join(self.data_root, dataset, self.subset)
            if not os.path.isdir(dataset_dir):
                print(f"Warning: dataset directory missing: {dataset_dir}")
                continue

            for pt_path in glob.glob(os.path.join(dataset_dir, '*.pt')):
                try:
                    bundle = load_dynaip_sequence(pt_path, fps=self.fps_default, trim_frames=self.trim_frames, keep_raw_keys=False)
                    processed = bundle['processed']
                    imu = processed['imu']['imu'].float()
                    T_total = imu.shape[0]
                    if T_total < 2:
                        continue

                    obj_imu = processed['imu'].get('obj_imu', torch.zeros(T_total, 12))
                    obj_velocity = processed.get('object', {}).get('velocity', torch.zeros(T_total, 3))
                    obj_position = processed.get('object', {}).get('position', torch.zeros(T_total, 3))

                    sequence = {
                        'imu': imu.contiguous(),
                        'pose': processed['joint']['pose'].contiguous(),
                        'trans': processed['joint']['trans'].contiguous(),
                        'obj_imu': obj_imu.contiguous(),
                        'obj_vel': obj_velocity.contiguous(),
                        'obj_pos': obj_position.contiguous(),
                        'obj_rot': processed['object']['rot'].contiguous(),
                    }
                    sequence['seq_name'] = os.path.splitext(os.path.basename(pt_path))[0]
                    sequence['source_path'] = pt_path
                    self.sequences.append(sequence)
                    self.sequence_lengths.append(T_total)
                except Exception as exc:
                    print(f"Error loading {pt_path}: {exc}")
                    import traceback
                    traceback.print_exc()

        if not self.sequences:
            print("Warning: no valid sequences loaded.")
        self._build_index()
        print(f"Loaded {len(self.sequences)} sequences, total samples: {self.total_samples}")

    def _slice_and_pad(self, tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
        window = tensor[start:end].clone()
        cur_len = window.shape[0]
        if cur_len >= self.seq_len:
            return window[:self.seq_len]
        pad_len = self.seq_len - cur_len
        pad = window[-1:].repeat(pad_len, *([1] * (tensor.dim() - 1)))
        return torch.cat([window, pad], dim=0)

    def _add_imu_noise(self, imu: torch.Tensor, obj_imu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为IMU数据添加高斯噪声
        
        Args:
            imu: 人体IMU数据 [T, 6, 12]
            obj_imu: 物体IMU数据 [T, 12]
        
        Returns:
            添加噪声后的IMU数据
        """
        if self.imu_noise_std <= 0:
            return imu, obj_imu
        
        # 为人体IMU添加噪声
        imu_noisy = imu.clone()
        noise_human = torch.randn_like(imu) * self.imu_noise_std
        imu_noisy = imu_noisy + noise_human
        
        # 为物体IMU添加噪声
        obj_imu_noisy = obj_imu.clone()
        noise_obj = torch.randn_like(obj_imu) * self.imu_noise_std
        obj_imu_noisy = obj_imu_noisy + noise_obj
        
        return imu_noisy, obj_imu_noisy

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.total_samples == 0:
            raise IndexError('Empty dataset')
        seq_idx, offset = self._locate_sequence(index)
        sequence = self.sequences[seq_idx]
        length = self.sequence_lengths[seq_idx]

        if self.use_full_sequence:
            # 使用完整序列：直接返回整个序列，不做切片
            start = 0
            end = length
            imu_window = sequence['imu']
            pose_window = sequence['pose']
            trans_window = sequence['trans']
            obj_imu_window = sequence['obj_imu']
            obj_vel_window = sequence['obj_vel']
            obj_pos_window = sequence['obj_pos']
            obj_rot_window = sequence['obj_rot']
        else:
            # 原有逻辑：采样固定长度窗口
            if self.random_sample:
                if length <= self.seq_len:
                    start = 0
                else:
                    start = torch.randint(0, length - self.seq_len + 1, (1,)).item()
            else:
                start = min(offset, max(0, length - self.seq_len))
            end = min(start + self.seq_len, length)

            imu_window = self._slice_and_pad(sequence['imu'], start, end)
            pose_window = self._slice_and_pad(sequence['pose'], start, end)
            trans_window = self._slice_and_pad(sequence['trans'], start, end)
            obj_imu_window = self._slice_and_pad(sequence['obj_imu'], start, end)
            obj_vel_window = self._slice_and_pad(sequence['obj_vel'], start, end)
            obj_pos_window = self._slice_and_pad(sequence['obj_pos'], start, end)
            obj_rot_window = self._slice_and_pad(sequence['obj_rot'], start, end)

        # 添加IMU噪声
        if self.imu_noise_std > 0:
            imu_window, obj_imu_window = self._add_imu_noise(imu_window, obj_imu_window)
        
        sample = {
            'imu': imu_window,
            'pose': pose_window,
            'trans': trans_window,
            'obj_imu': obj_imu_window,
            'obj_vel': obj_vel_window,
            'obj_pos': obj_pos_window,
            'obj_rot': obj_rot_window,
            'obj_v_init': obj_vel_window[0],
            'obj_p_init': obj_pos_window[0],
        }
        return {k: v.contiguous() for k, v in sample.items()}


def collate_fn_with_object_and_trans(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    collated: Dict[str, torch.Tensor] = {}
    for key in keys:
        collated[key] = torch.stack([item[key] for item in batch], dim=0)
    return collated
