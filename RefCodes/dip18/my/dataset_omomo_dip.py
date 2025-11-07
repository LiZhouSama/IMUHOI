"""
Direct OMOMO to DIP format dataset loader.
This implementation loads OMOMO data directly and converts to DIP format without 
intermediate conversions that may reduce pose representation.

Key differences from dataset_dip_style.py:
- Uses full SMPL pose representation (not reduced)
- Follows DIP's original data format from genSynData.py
- Direct conversion from OMOMO .pt files
"""

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# Sensor indices following DIP/Xsens convention
_SENSOR_ROT_INDICES = [0, 4, 5, 15, 18, 19]  # For orientation
_SENSOR_POS_INDICES = [0, 7, 8, 15, 20, 21]  # For acceleration computation
_SENSOR_NAMES = ['Root', 'LeftLowerLeg', 'RightLowerLeg', 'Head', 'LeftForeArm', 'RightForeArm']

# SMPL joints to use (following DIP convention)
# These are the body joints excluding hands/feet endpoints
_SMPL_JOINT_INDICES = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]


class DataOperator:
    """Data preprocessing operator for normalization."""
    
    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.mean = mean
        self.std = std
        self.enabled = mean is not None and std is not None
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization: (x - mean) / std"""
        if not self.enabled:
            return data
        return (data - self.mean) / (self.std + 1e-8)
    
    def undo(self, data: np.ndarray) -> np.ndarray:
        """Undo normalization: x * std + mean"""
        if not self.enabled:
            return data
        return data * (self.std + 1e-8) + self.mean


def _compute_statistics(data_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute mean and std statistics across all sequences."""
    if not data_list:
        return {'mean_channel': np.zeros(1), 'std_channel': np.ones(1)}
    
    all_data = np.concatenate(data_list, axis=0)
    mean_channel = np.mean(all_data, axis=0)
    std_channel = np.std(all_data, axis=0)
    std_channel = np.where(std_channel < 1e-8, 1.0, std_channel)
    
    return {
        'mean_channel': mean_channel,
        'std_channel': std_channel
    }


def _central_diff(a: np.ndarray, dt: float) -> np.ndarray:
    """Central difference for velocity/acceleration computation."""
    if a.shape[0] <= 1:
        return np.zeros_like(a)
    vel = np.zeros_like(a)
    vel[1:-1] = (a[2:] - a[:-2]) / (2.0 * dt)
    vel[0] = (a[1] - a[0]) / dt
    vel[-1] = (a[-1] - a[-2]) / dt
    return vel


def _normalize_imu_numpy(acc: np.ndarray, ori: np.ndarray) -> np.ndarray:
    """
    Normalize IMU data w.r.t the root sensor (numpy version).
    
    Args:
        acc: [T, 6, 3] acceleration
        ori: [T, 6, 3, 3] orientation matrices
    
    Returns:
        [T, 6, 12] normalized IMU data (9D rotation + 3D acceleration)
    """
    # Normalize acceleration relative to root
    acc_normalized = np.concatenate([
        acc[:, :1],  # Root acceleration unchanged
        acc[:, 1:] - acc[:, :1]  # Other sensors relative to root
    ], axis=1)
    
    # Transform acceleration to root frame
    root_ori = ori[:, 0]  # [T, 3, 3]
    acc_normalized = np.einsum('tij,tnj->tni', root_ori, acc_normalized)
    
    # Normalize orientation relative to root
    root_ori_inv = np.transpose(root_ori, (0, 2, 1))  # [T, 3, 3]
    ori_normalized = ori.copy()
    ori_normalized[:, 0] = np.eye(3)[np.newaxis, :, :]  # Root becomes identity
    for j in range(1, 6):
        ori_normalized[:, j] = root_ori_inv @ ori[:, j]
    
    # Flatten orientation matrices [T, 6, 3, 3] -> [T, 6, 9]
    ori_flat = ori_normalized.reshape(ori_normalized.shape[0], 6, 9)
    
    # Concatenate [T, 6, 9+3]
    imu = np.concatenate([ori_flat, acc_normalized], axis=-1)
    return imu


def _build_imu_from_smpl(
    rotation_global: np.ndarray,
    position_global: np.ndarray,
    fps: float
) -> np.ndarray:
    """
    Build IMU data from SMPL joint rotations and positions.
    
    Args:
        rotation_global: [T, 24, 3, 3] global rotation matrices
        position_global: [T, 24, 3] global positions
        fps: frame rate
    
    Returns:
        [T, 6, 12] IMU data
    """
    T = rotation_global.shape[0]
    dt = 1.0 / fps
    
    # Select sensor rotations and positions
    sel_rot = rotation_global[:, _SENSOR_ROT_INDICES]  # [T, 6, 3, 3]
    sel_pos = position_global[:, _SENSOR_POS_INDICES]  # [T, 6, 3]
    
    # Compute velocity and acceleration
    vel = _central_diff(sel_pos, dt)  # [T, 6, 3]
    acc = _central_diff(vel, dt)  # [T, 6, 3]
    
    # Normalize IMU
    imu = _normalize_imu_numpy(acc, sel_rot)  # [T, 6, 12]
    
    return imu


def _extract_full_pose_matrices(rotation_global: np.ndarray) -> np.ndarray:
    """
    Extract full pose matrices for selected SMPL joints (DIP format).
    
    Args:
        rotation_global: [T, 24, 3, 3] global rotation matrices
    
    Returns:
        [T, J*9] flattened rotation matrices for J joints
    """
    T = rotation_global.shape[0]
    
    # Select joints (following DIP convention)
    selected_joints = rotation_global[:, _SMPL_JOINT_INDICES]  # [T, J, 3, 3]
    
    # Convert to local rotations (relative to root)
    root_global = rotation_global[:, 0:1]  # [T, 1, 3, 3]
    root_inv = np.transpose(root_global, (0, 1, 3, 2))  # [T, 1, 3, 3]
    
    local_rotations = np.zeros_like(selected_joints)
    local_rotations[:, 0] = selected_joints[:, 0]  # Root stays in global frame
    
    for j in range(1, len(_SMPL_JOINT_INDICES)):
        local_rotations[:, j] = root_inv[:, 0] @ selected_joints[:, j]
    
    # Flatten to [T, J*9]
    pose_flat = local_rotations.reshape(T, -1)
    
    return pose_flat


def _build_object_imu(
    obj_rot: np.ndarray,
    obj_pos: np.ndarray,
    fps: float
) -> np.ndarray:
    """
    Build object IMU from rotation and position.
    
    Args:
        obj_rot: [T, 6] (6D representation) or [T, 3, 3] (rotation matrix)
        obj_pos: [T, 3] position
        fps: frame rate
    
    Returns:
        [T, 12] object IMU (9D rotation + 3D acceleration)
    """
    T = obj_rot.shape[0]
    dt = 1.0 / fps
    
    # Convert to rotation matrix if needed
    if obj_rot.shape[-1] == 6:
        # 6D to rotation matrix
        a1 = obj_rot[:, :3]
        a2 = obj_rot[:, 3:6]
        a1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
        a2 = a2 - np.sum(a1 * a2, axis=-1, keepdims=True) * a1
        a2 = a2 / (np.linalg.norm(a2, axis=-1, keepdims=True) + 1e-8)
        a3 = np.cross(a1, a2, axis=-1)
        rot_matrix = np.stack([a1, a2, a3], axis=-1)  # [T, 3, 3]
    else:
        rot_matrix = obj_rot
    
    # Flatten rotation matrix
    ori_flat = rot_matrix.reshape(T, 9)
    
    # Compute acceleration
    vel = _central_diff(obj_pos, dt)
    acc = _central_diff(vel, dt)
    
    # Concatenate [T, 9+3]
    obj_imu = np.concatenate([ori_flat, acc], axis=-1)
    
    return obj_imu


def _compute_object_velocity_position(
    obj_pos: np.ndarray,
    fps: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute object velocity and zero-centered position.
    
    Args:
        obj_pos: [T, 3] object position
        fps: frame rate
    
    Returns:
        velocity: [T, 3]
        position: [T, 3] (zero-centered)
    """
    dt = 1.0 / fps
    
    # Zero-center position
    pos = obj_pos.copy()
    pos[:, 0] -= pos[0, 0]
    pos[:, 2] -= pos[0, 2]
    
    # Compute velocity
    vel = _central_diff(pos, dt)
    
    return vel, pos


def _trim_sequence(data: np.ndarray, trim: int) -> np.ndarray:
    """Trim frames from start and end."""
    if trim <= 0:
        return data
    if data.shape[0] <= 2 * trim:
        return np.empty((0,) + data.shape[1:], dtype=data.dtype)
    return data[trim:-trim]


class OMOMODIPDataset(Dataset):
    """
    Dataset that loads OMOMO data and converts directly to DIP format.
    
    This maintains full SMPL pose representation without reduction.
    """
    
    def __init__(
        self,
        dataset_names: List[str],
        data_root: str,
        subset: str = "train",
        seq_len: int = 120,
        random_sample: bool = True,
        use_full_sequence: bool = False,
        fps: float = 30.0,
        trim_frames: int = 6,
        imu_noise_std: float = 0.0,
        normalize: bool = True,
        data_stats: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            dataset_names: List of dataset folder names
            data_root: Root directory containing datasets
            subset: 'train', 'test', or 'val'
            seq_len: Sequence length for fixed-length windows
            random_sample: Random sampling for training
            use_full_sequence: Return full sequences (for validation)
            fps: Frame rate for data processing
            trim_frames: Frames to trim from start/end
            imu_noise_std: Gaussian noise std for IMU augmentation
            normalize: Whether to apply normalization
            data_stats: Pre-computed statistics
        """
        super().__init__()
        
        self.dataset_names = dataset_names
        self.data_root = data_root
        self.subset = subset
        self.seq_len = seq_len
        self.random_sample = random_sample
        self.use_full_sequence = use_full_sequence
        self.fps = fps
        self.trim_frames = trim_frames
        self.imu_noise_std = imu_noise_std
        self.normalize = normalize
        
        # Storage
        self.sequences: List[Dict[str, np.ndarray]] = []
        self.sequence_lengths: List[int] = []
        self.cumulative_sizes: List[int] = []
        self.total_samples: int = 0
        
        # Data dimensions
        self.human_input_dim: int = 72  # 6 sensors * 12 features
        self.object_input_dim: int = 12  # 9D rotation + 3D acceleration
        self.human_pose_dim: int = len(_SMPL_JOINT_INDICES) * 9  # J joints * 9
        self.object_velocity_dim: int = 3
        
        # Data statistics and operators
        self.data_stats: Dict[str, Dict[str, np.ndarray]] = {}
        self.human_imu_operator: Optional[DataOperator] = None
        self.object_imu_operator: Optional[DataOperator] = None
        self.human_pose_operator: Optional[DataOperator] = None
        self.object_velocity_operator: Optional[DataOperator] = None
        
        # Load data
        print(f"[OMOMODIPDataset] Loading {subset} data from {len(dataset_names)} datasets...")
        self._load_sequences()
        
        # Compute or use provided statistics
        if data_stats is None and self.normalize:
            print("[OMOMODIPDataset] Computing data statistics...")
            self._compute_statistics()
        elif data_stats is not None:
            self.data_stats = data_stats
        
        # Create data operators
        if self.normalize:
            self._create_operators()
        
        # Build sample index
        self._build_index()
        
        print(f"[OMOMODIPDataset] Loaded {len(self.sequences)} sequences, {self.total_samples} samples")
        print(f"[OMOMODIPDataset] Dimensions - Human IMU: {self.human_input_dim}, "
              f"Object IMU: {self.object_input_dim}, "
              f"Human Pose: {self.human_pose_dim}, "
              f"Object Vel: {self.object_velocity_dim}")
    
    def _load_sequences(self) -> None:
        """Load all .pt files and convert to DIP format."""
        for dataset_name in self.dataset_names:
            dataset_dir = os.path.join(self.data_root, dataset_name, self.subset)
            if not os.path.isdir(dataset_dir):
                print(f"[OMOMODIPDataset] Warning: {dataset_dir} does not exist, skipping.")
                continue
            
            pt_files = sorted(glob.glob(os.path.join(dataset_dir, "*.pt")))
            if not pt_files:
                print(f"[OMOMODIPDataset] Warning: no .pt files in {dataset_dir}")
                continue
            
            for pt_path in pt_files:
                try:
                    # Load raw data
                    raw_data = torch.load(pt_path)
                    
                    # Extract SMPL data
                    if 'rotation_global' not in raw_data or 'position_global_full_gt_world' not in raw_data:
                        continue
                    
                    rot_global = raw_data['rotation_global'].float().numpy()  # [T, J, 3, 3]
                    pos_global = raw_data['position_global_full_gt_world'].float().numpy()  # [T, J, 3]
                    
                    # Handle 22-joint to 24-joint padding
                    if rot_global.shape[1] == 22:
                        T = rot_global.shape[0]
                        rot_global_padded = np.zeros((T, 24, 3, 3), dtype=rot_global.dtype)
                        rot_global_padded[:, :22] = rot_global
                        rot_global_padded[:, 22:] = np.eye(3)[np.newaxis, np.newaxis, :, :]
                        rot_global = rot_global_padded
                        
                        pos_global_padded = np.zeros((T, 24, 3), dtype=pos_global.dtype)
                        pos_global_padded[:, :22] = pos_global
                        pos_global = pos_global_padded
                    
                    T = rot_global.shape[0]
                    
                    # Build human IMU
                    human_imu = _build_imu_from_smpl(rot_global, pos_global, self.fps)  # [T, 6, 12]
                    
                    # Extract full pose representation
                    human_pose = _extract_full_pose_matrices(rot_global)  # [T, J*9]
                    
                    # Compute human pose velocity
                    dt = 1.0 / self.fps
                    human_velocity = _central_diff(human_pose, dt)
                    
                    # Extract object data
                    if 'obj_rot' not in raw_data or 'obj_trans' not in raw_data:
                        continue
                    
                    obj_rot = raw_data['obj_rot'].float().numpy()
                    obj_pos = raw_data['obj_trans'].float().numpy()
                    
                    # Build object IMU
                    object_imu = _build_object_imu(obj_rot, obj_pos, self.fps)  # [T, 12]
                    
                    # Compute object velocity and position
                    obj_velocity, obj_position = _compute_object_velocity_position(obj_pos, self.fps)
                    
                    # Trim sequences
                    trim = max(0, int(self.trim_frames))
                    if trim > 0:
                        human_imu = _trim_sequence(human_imu, trim)
                        human_pose = _trim_sequence(human_pose, trim)
                        human_velocity = _trim_sequence(human_velocity, trim)
                        object_imu = _trim_sequence(object_imu, trim)
                        obj_velocity = _trim_sequence(obj_velocity, trim)
                        obj_position = _trim_sequence(obj_position, trim)
                    
                    T_trimmed = human_imu.shape[0]
                    if T_trimmed < 2:
                        continue
                    
                    # Flatten human IMU [T, 6, 12] -> [T, 72]
                    human_imu_flat = human_imu.reshape(T_trimmed, -1)
                    
                    sequence = {
                        "human_imu": human_imu_flat.astype(np.float32),
                        "object_imu": object_imu.astype(np.float32),
                        "human_pose": human_pose.astype(np.float32),
                        "human_velocity": human_velocity.astype(np.float32),
                        "object_velocity": obj_velocity.astype(np.float32),
                        "object_position": obj_position.astype(np.float32),
                    }
                    
                    self.sequences.append(sequence)
                    self.sequence_lengths.append(T_trimmed)
                    
                except Exception as exc:
                    print(f"[OMOMODIPDataset] Error loading {pt_path}: {exc}")
                    import traceback
                    traceback.print_exc()
        
        if not self.sequences:
            raise RuntimeError("No valid sequences loaded.")
        
        # Verify dimensions
        sample = self.sequences[0]
        actual_human_input_dim = sample["human_imu"].shape[-1]
        actual_human_pose_dim = sample["human_pose"].shape[-1]
        
        if actual_human_input_dim != self.human_input_dim:
            print(f"[OMOMODIPDataset] Warning: Expected human_input_dim={self.human_input_dim}, "
                  f"got {actual_human_input_dim}")
            self.human_input_dim = actual_human_input_dim
        
        if actual_human_pose_dim != self.human_pose_dim:
            print(f"[OMOMODIPDataset] Warning: Expected human_pose_dim={self.human_pose_dim}, "
                  f"got {actual_human_pose_dim}")
            self.human_pose_dim = actual_human_pose_dim
    
    def _compute_statistics(self) -> None:
        """Compute mean and std for normalization."""
        human_imu_list = [seq["human_imu"] for seq in self.sequences]
        object_imu_list = [seq["object_imu"] for seq in self.sequences]
        human_pose_list = [seq["human_pose"] for seq in self.sequences]
        object_vel_list = [seq["object_velocity"] for seq in self.sequences]
        
        self.data_stats = {
            "human_imu": _compute_statistics(human_imu_list),
            "object_imu": _compute_statistics(object_imu_list),
            "human_pose": _compute_statistics(human_pose_list),
            "object_velocity": _compute_statistics(object_vel_list),
        }
    
    def _create_operators(self) -> None:
        """Create data preprocessing operators."""
        if "human_imu" in self.data_stats:
            stats = self.data_stats["human_imu"]
            self.human_imu_operator = DataOperator(
                stats["mean_channel"], stats["std_channel"]
            )
        
        if "object_imu" in self.data_stats:
            stats = self.data_stats["object_imu"]
            self.object_imu_operator = DataOperator(
                stats["mean_channel"], stats["std_channel"]
            )
        
        if "human_pose" in self.data_stats:
            stats = self.data_stats["human_pose"]
            self.human_pose_operator = DataOperator(
                stats["mean_channel"], stats["std_channel"]
            )
        
        if "object_velocity" in self.data_stats:
            stats = self.data_stats["object_velocity"]
            self.object_velocity_operator = DataOperator(
                stats["mean_channel"], stats["std_channel"]
            )
    
    def _build_index(self) -> None:
        """Build cumulative index for dataset samples."""
        self.cumulative_sizes.clear()
        total = 0
        for length in self.sequence_lengths:
            if self.use_full_sequence:
                steps = 1
            elif self.random_sample:
                steps = 1
            else:
                steps = max(1, length - self.seq_len + 1)
            total += steps
            self.cumulative_sizes.append(total)
        self.total_samples = total
    
    def __len__(self) -> int:
        return self.total_samples
    
    def _locate_sequence(self, index: int) -> Tuple[int, int]:
        """Find which sequence and offset corresponds to a given index."""
        from bisect import bisect_right
        seq_idx = bisect_right(self.cumulative_sizes, index)
        prev = 0 if seq_idx == 0 else self.cumulative_sizes[seq_idx - 1]
        offset = index - prev
        return seq_idx, offset
    
    def _slice_window(self, array: np.ndarray, start: int, end: int) -> np.ndarray:
        """Extract a window from array with padding if needed."""
        window = array[start:end].copy()
        cur_len = window.shape[0]
        if cur_len >= self.seq_len:
            return window[:self.seq_len]
        # Pad with last frame
        pad_len = self.seq_len - cur_len
        pad = np.repeat(window[-1:], pad_len, axis=0)
        return np.concatenate([window, pad], axis=0)
    
    def _add_noise(self, human_imu: np.ndarray, object_imu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Add Gaussian noise to IMU data for augmentation."""
        if self.imu_noise_std <= 0:
            return human_imu, object_imu
        
        noise_human = np.random.randn(*human_imu.shape) * self.imu_noise_std
        noise_object = np.random.randn(*object_imu.shape) * self.imu_noise_std
        return human_imu + noise_human, object_imu + noise_object
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        seq_idx, offset = self._locate_sequence(index)
        sequence = self.sequences[seq_idx]
        length = self.sequence_lengths[seq_idx]
        
        if self.use_full_sequence:
            # Use full sequence
            start = 0
            end = length
            human_imu = sequence["human_imu"]
            object_imu = sequence["object_imu"]
            human_pose = sequence["human_pose"]
            human_velocity = sequence["human_velocity"]
            object_velocity = sequence["object_velocity"]
            object_position = sequence["object_position"]
            actual_len = length
        else:
            # Fixed-length window
            if self.random_sample:
                if length > self.seq_len:
                    start = np.random.randint(0, length - self.seq_len + 1)
                else:
                    start = 0
            else:
                start = min(offset, max(0, length - self.seq_len))
            end = min(start + self.seq_len, length)
            
            human_imu = self._slice_window(sequence["human_imu"], start, end)
            object_imu = self._slice_window(sequence["object_imu"], start, end)
            human_pose = self._slice_window(sequence["human_pose"], start, end)
            human_velocity = self._slice_window(sequence["human_velocity"], start, end)
            object_velocity = self._slice_window(sequence["object_velocity"], start, end)
            object_position = self._slice_window(sequence["object_position"], start, end)
            actual_len = min(end - start, self.seq_len)
        
        # Add noise (augmentation)
        human_imu, object_imu = self._add_noise(human_imu, object_imu)
        
        # Apply normalization
        if self.normalize:
            if self.human_imu_operator:
                human_imu = self.human_imu_operator.apply(human_imu)
            if self.object_imu_operator:
                object_imu = self.object_imu_operator.apply(object_imu)
            if self.human_pose_operator:
                human_pose = self.human_pose_operator.apply(human_pose)
                human_velocity = self.human_pose_operator.apply(human_velocity)
            if self.object_velocity_operator:
                object_velocity = self.object_velocity_operator.apply(object_velocity)
        
        # Convert to tensors
        sample = {
            "seq_len": actual_len,
            "human_imu": torch.from_numpy(human_imu).float(),
            "object_imu": torch.from_numpy(object_imu).float(),
            "human_pose": torch.from_numpy(human_pose).float(),
            "human_velocity": torch.from_numpy(human_velocity).float(),
            "object_velocity": torch.from_numpy(object_velocity).float(),
            "object_position": torch.from_numpy(object_position).float(),
            "object_init_pos": torch.from_numpy(object_position[0]).float(),
        }
        
        return sample
    
    def get_statistics(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Return computed statistics for saving/reuse."""
        return self.data_stats
    
    def undo_normalization_pose(self, pose_normalized: np.ndarray) -> np.ndarray:
        """Undo normalization for human pose predictions."""
        if self.human_pose_operator:
            return self.human_pose_operator.undo(pose_normalized)
        return pose_normalized
    
    def undo_normalization_velocity(self, vel_normalized: np.ndarray) -> np.ndarray:
        """Undo normalization for object velocity predictions."""
        if self.object_velocity_operator:
            return self.object_velocity_operator.undo(vel_normalized)
        return vel_normalized


def collate_fn_omomo_dip(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching OMOMO-DIP samples.
    Handles variable-length sequences with padding.
    """
    # Find max sequence length in batch
    seq_lens = [item["seq_len"] for item in batch]
    max_len = max(seq_lens)
    batch_size = len(batch)
    
    # Get dimensions from first sample
    human_imu_dim = batch[0]["human_imu"].shape[-1]
    object_imu_dim = batch[0]["object_imu"].shape[-1]
    human_pose_dim = batch[0]["human_pose"].shape[-1]
    object_vel_dim = batch[0]["object_velocity"].shape[-1]
    
    # Initialize padded tensors
    human_imu_batch = torch.zeros(batch_size, max_len, human_imu_dim)
    object_imu_batch = torch.zeros(batch_size, max_len, object_imu_dim)
    human_pose_batch = torch.zeros(batch_size, max_len, human_pose_dim)
    human_velocity_batch = torch.zeros(batch_size, max_len, human_pose_dim)
    object_velocity_batch = torch.zeros(batch_size, max_len, object_vel_dim)
    object_position_batch = torch.zeros(batch_size, max_len, object_vel_dim)
    object_init_pos_batch = torch.zeros(batch_size, object_vel_dim)
    
    # Fill with actual data
    for i, item in enumerate(batch):
        seq_len = seq_lens[i]
        human_imu_batch[i, :seq_len] = item["human_imu"][:seq_len]
        object_imu_batch[i, :seq_len] = item["object_imu"][:seq_len]
        human_pose_batch[i, :seq_len] = item["human_pose"][:seq_len]
        human_velocity_batch[i, :seq_len] = item["human_velocity"][:seq_len]
        object_velocity_batch[i, :seq_len] = item["object_velocity"][:seq_len]
        object_position_batch[i, :seq_len] = item["object_position"][:seq_len]
        object_init_pos_batch[i] = item["object_init_pos"]
    
    return {
        "seq_len": torch.tensor(seq_lens, dtype=torch.long),
        "human_imu": human_imu_batch,
        "object_imu": object_imu_batch,
        "human_pose": human_pose_batch,
        "human_velocity": human_velocity_batch,
        "object_velocity": object_velocity_batch,
        "object_position": object_position_batch,
        "object_init_pos": object_init_pos_batch,
    }
