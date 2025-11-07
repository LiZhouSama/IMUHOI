"""
DIP-style dataset for loading OMOMO data directly from .pt files.
This implementation follows the original DIP dataset structure but works with PyTorch.

Key features:
- Direct loading from .pt files (no intermediate DynaIP format)
- Data normalization with mean/std statistics
- Sample generator interface similar to original DIP
- Support for human pose + object trajectory prediction
"""

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class DataOperator:
    """
    Data preprocessing operator for normalization, similar to DIP's Operator class.
    """
    
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
    """
    Compute mean and std statistics across all sequences.
    
    Args:
        data_list: List of arrays, each with shape (T, F)
    
    Returns:
        Dictionary with 'mean_channel' and 'std_channel' keys
    """
    if not data_list:
        return {'mean_channel': np.zeros(1), 'std_channel': np.ones(1)}
    
    all_data = np.concatenate(data_list, axis=0)  # (total_frames, features)
    mean_channel = np.mean(all_data, axis=0)
    std_channel = np.std(all_data, axis=0)
    std_channel = np.where(std_channel < 1e-8, 1.0, std_channel)
    
    return {
        'mean_channel': mean_channel,
        'std_channel': std_channel
    }


def _finite_difference(sequence: np.ndarray, dt: float) -> np.ndarray:
    """Compute velocity using finite differences."""
    if sequence.shape[0] <= 1:
        return np.zeros_like(sequence)
    diff = np.zeros_like(sequence)
    diff[1:] = (sequence[1:] - sequence[:-1]) / dt
    diff[0] = diff[1]
    return diff


def _apply_zero_offset(position: np.ndarray) -> np.ndarray:
    """Zero-center the position trajectory."""
    if position.size == 0:
        return position
    offset = position[0].copy()
    return position - offset[np.newaxis, :]


class DIPStyleDataset(Dataset):
    """
    DIP-style dataset that loads OMOMO data directly from .pt files.
    
    This dataset mimics the structure of the original DIP ImuDataset but:
    - Loads from .pt files instead of .npz
    - Includes object trajectory data
    - Uses PyTorch tensors internally
    """
    
    def __init__(
        self,
        dataset_names: List[str],
        data_root: str,
        subset: str = "train",
        seq_len: int = 120,
        random_sample: bool = True,
        use_full_sequence: bool = False,
        fps_override: Optional[float] = None,
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
            fps_override: Override FPS when loading
            trim_frames: Frames to trim from start/end
            imu_noise_std: Gaussian noise std for IMU augmentation
            normalize: Whether to apply normalization
            data_stats: Pre-computed statistics (if None, will compute)
        """
        super().__init__()
        
        self.dataset_names = dataset_names
        self.data_root = data_root
        self.subset = subset
        self.seq_len = seq_len
        self.random_sample = random_sample
        self.use_full_sequence = use_full_sequence
        self.fps_override = fps_override
        self.trim_frames = trim_frames
        self.imu_noise_std = imu_noise_std
        self.normalize = normalize
        
        # Storage for loaded sequences
        self.sequences: List[Dict[str, np.ndarray]] = []
        self.sequence_lengths: List[int] = []
        self.cumulative_sizes: List[int] = []
        self.total_samples: int = 0
        
        # Data dimensions
        self.human_input_dim: int = 0
        self.object_input_dim: int = 0
        self.human_pose_dim: int = 0
        self.object_velocity_dim: int = 0
        
        # Data statistics and operators
        self.data_stats: Dict[str, Dict[str, np.ndarray]] = {}
        self.human_imu_operator: Optional[DataOperator] = None
        self.object_imu_operator: Optional[DataOperator] = None
        self.human_pose_operator: Optional[DataOperator] = None
        self.object_velocity_operator: Optional[DataOperator] = None
        
        # Load data
        print(f"[DIPStyleDataset] Loading {subset} data from {len(dataset_names)} datasets...")
        self._load_sequences()
        
        # Compute or use provided statistics
        if data_stats is None and self.normalize:
            print("[DIPStyleDataset] Computing data statistics...")
            self._compute_statistics()
        elif data_stats is not None:
            self.data_stats = data_stats
        
        # Create data operators
        if self.normalize:
            self._create_operators()
        
        # Build sample index
        self._build_index()
        
        print(f"[DIPStyleDataset] Loaded {len(self.sequences)} sequences, {self.total_samples} samples")
        print(f"[DIPStyleDataset] Dimensions - Human IMU: {self.human_input_dim}, "
              f"Object IMU: {self.object_input_dim}, "
              f"Human Pose: {self.human_pose_dim}, "
              f"Object Vel: {self.object_velocity_dim}")
    
    def _load_sequences(self) -> None:
        """Load all .pt files from specified datasets."""
        from my.dataset_dynaip_trans_obj import load_dynaip_sequence
        
        for dataset_name in self.dataset_names:
            dataset_dir = os.path.join(self.data_root, dataset_name, self.subset)
            if not os.path.isdir(dataset_dir):
                print(f"[DIPStyleDataset] Warning: {dataset_dir} does not exist, skipping.")
                continue
            
            pt_files = sorted(glob.glob(os.path.join(dataset_dir, "*.pt")))
            if not pt_files:
                print(f"[DIPStyleDataset] Warning: no .pt files in {dataset_dir}")
                continue
            
            for pt_path in pt_files:
                try:
                    bundle = load_dynaip_sequence(
                        pt_path,
                        fps=self.fps_override,
                        trim_frames=self.trim_frames,
                        keep_raw_keys=False,
                    )
                    processed = bundle["processed"]
                    meta = bundle["meta"]
                    fps_value = float(meta.get("fps", 30.0))
                    dt = 1.0 / max(fps_value, 1e-8)
                    
                    # Extract human IMU
                    imu = processed["imu"]["imu"].float().numpy()  # [T, 6, 12]
                    T = imu.shape[0]
                    if T < 2:
                        continue
                    
                    # Extract object IMU
                    obj_imu = processed["imu"].get("obj_imu", None)
                    if obj_imu is None:
                        continue
                    obj_imu = obj_imu.float().numpy()  # [T, 12]
                    
                    # Extract human pose
                    pose = processed["joint"]["ori_glb_reduced"].float().numpy()  # [T, 60]
                    human_pose = pose.reshape(T, -1)
                    human_velocity = _finite_difference(human_pose, dt)
                    
                    # Extract object trajectory
                    obj_velocity = processed.get("object", {}).get("velocity", None)
                    obj_position = processed.get("object", {}).get("position", None)
                    if obj_velocity is None or obj_position is None:
                        continue
                    obj_velocity = obj_velocity.float().numpy()
                    obj_position = _apply_zero_offset(obj_position.float().numpy())
                    
                    # Flatten IMU to match DIP format
                    human_imu = imu.reshape(T, -1)  # [T, 72]
                    object_imu = obj_imu.reshape(T, -1)  # [T, 12]
                    
                    sequence = {
                        "human_imu": human_imu,
                        "object_imu": object_imu,
                        "human_pose": human_pose,
                        "human_velocity": human_velocity,
                        "object_velocity": obj_velocity.reshape(T, -1),
                        "object_position": obj_position.reshape(T, -1),
                        "dt": dt,
                    }
                    
                    self.sequences.append(sequence)
                    self.sequence_lengths.append(T)
                    
                except Exception as exc:
                    print(f"[DIPStyleDataset] Error loading {pt_path}: {exc}")
        
        if not self.sequences:
            raise RuntimeError("No valid sequences loaded.")
        
        # Set dimensions from first sequence
        sample = self.sequences[0]
        self.human_input_dim = sample["human_imu"].shape[-1]
        self.object_input_dim = sample["object_imu"].shape[-1]
        self.human_pose_dim = sample["human_pose"].shape[-1]
        self.object_velocity_dim = sample["object_velocity"].shape[-1]
    
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
        """
        Get a single sample.
        
        Returns:
            Dictionary with keys:
                - 'seq_len': actual sequence length (int)
                - 'human_imu': [T, F_h] human IMU input
                - 'object_imu': [T, F_o] object IMU input
                - 'human_pose': [T, P_h] human pose target
                - 'human_velocity': [T, P_h] human pose velocity target
                - 'object_velocity': [T, P_o] object velocity target
                - 'object_position': [T, P_o] object position target
                - 'object_init_pos': [P_o] initial object position
        """
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
                # Note: position is not normalized directly, it's derived from velocity
        
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
    
    def sample_generator(self):
        """
        Generator that yields samples one at a time (DIP-style interface).
        This is useful for custom data feeding pipelines.
        """
        for idx in range(len(self)):
            yield self[idx]
    
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


def collate_fn_dip_style(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching DIP-style samples.
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

