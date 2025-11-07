import os
import glob
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset


class MotionDatasetWithObject(Dataset):
    """
    Dataset for human IMU -> human velocity/pose prediction augmented with object IMU -> object velocity/position.

    Expects directory structure:
      <data_root>/train/<dataset_name>/*.pt

    Each .pt file should contain a dict with keys:
      - 'imu': {
            'imu': FloatTensor [T, 6, 12]   (6 sensors: orientation(9)+acc(3))
            'obj_imu': FloatTensor [T, 12]  (optional) object IMU orientation(9)+acc(3)
        }
      - 'pose': FloatTensor [T, 11, 6]      (local 6D pose for 11 target joints)
      - 'joint': {
            'velocity': FloatTensor [T, J, 3]  with J==24 if SMPL-like (preferred), else J==Xsense size
        }
      - 'object': {
            'velocity': FloatTensor [T, 3]     object COM velocity (world or consistent frame)
            'position': FloatTensor [T, 3]     object COM position (same frame), optional
        } (all optional if not available)

    Returned items are variable-length sequences; the collate_fn should keep them as lists.
    """

    def __init__(self,
                 datasets: List[str],
                 seq_len: int,
                 data_root: str,
                 device: str = 'cuda:0'):
        super().__init__()
        self.datasets = datasets
        self.seq_len = seq_len
        self.data_root = data_root
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.samples: Dict[str, List[torch.Tensor]] = {
            'imu': [],           # [T, 6, 12]
            'pose': [],          # [T, 11, 6]
            'velocity': [],      # [T, 6, 3] (selected joints order applied here)
            'v_init': [],        # [1, 6, 3]
            'p_init': [],        # [1, 11, 6]
            'obj_imu': [],       # [T, 12]
            'obj_vel': [],       # [T, 3]
            'obj_pos': [],       # [T, 3]
            'obj_v_init': [],    # [1, 3]
            'obj_p_init': [],    # [1, 3]
        }

        self._prepare()

    def __len__(self) -> int:
        return len(self.samples['imu'])

    def _prepare(self) -> None:
        for dataset in self.datasets:
            dataset_dir = os.path.join(self.data_root, 'train', dataset)
            if not os.path.isdir(dataset_dir):
                continue
            for pt_path in glob.glob(os.path.join(dataset_dir, '*.pt')):
                data: Dict[str, Any] = torch.load(pt_path)

                # IMU
                imu = data['imu']['imu'].float()  # [T, 6, 12]
                # Subsample into chunks of seq_len
                imu_chunks = torch.split(imu, self.seq_len, dim=0)

                # Velocity: select the 6 target joints used in training order
                # For SMPL-like 22/24 joints: indices [0, 15, 20, 21, 7, 8]
                vel_all = data['joint']['velocity'].float()  # [T, J, 3]
                if vel_all.shape[1] >= 22:
                    # SMPL/SMPLH 22-joint subset uses indices:
                    # Root(0), Head(15), LeftHand(20), RightHand(21), LeftAnkle(7), RightAnkle(8)
                    vel_mask = torch.tensor([0, 15, 20, 21, 7, 8])
                else:
                    # Xsens fallback
                    # vel_mask = torch.tensor([0, 6, 14, 10, 21, 17])
                    print("Error: vel_all.shape", vel_all.shape)
                    return
                vel_sel = vel_all[:, vel_mask]
                vel_chunks = torch.split(vel_sel, self.seq_len, dim=0)

                # Pose (T, 11, 6): derive from 16x6 orientation stored in 'joint.orientation' (flattened)
                # DynaIP joint_mask(16) order in process.py: [1,2,3,4,5,3,6,9,12,13,14,15,16,17,18,19]
                # Select 11 indices from that 16 to match p_names order used by model
                # Indices in the 16-list: [0,1,2,5,6,7,8,9,10,12,13]
                orient16_flat = data['joint']['orientation'].float()  # [T, 16*6]
                T = orient16_flat.shape[0]
                orient16 = orient16_flat.view(T, 16, 6)
                sel_11_from_16 = torch.tensor([0, 1, 2, 5, 6, 7, 8, 9, 10, 12, 13])
                pose = orient16[:, sel_11_from_16]
                pose_chunks = torch.split(pose, self.seq_len, dim=0)

                # Object IMU/labels if available
                obj_imu = data['imu']['obj_imu'].float()
                obj_vel = data['object']['velocity'].float()
                obj_pos = data['object']['position'].float()
                obj_imu_chunks = torch.split(obj_imu.float(), self.seq_len, dim=0)
                obj_vel_chunks = torch.split(obj_vel.float(), self.seq_len, dim=0)
                obj_pos_chunks = torch.split(obj_pos.float(), self.seq_len, dim=0)

                # Align chunk counts
                n = min(len(imu_chunks), len(vel_chunks), len(pose_chunks), len(obj_imu_chunks), len(obj_vel_chunks), len(obj_pos_chunks))

                for i in range(n):
                    self.samples['imu'].append(imu_chunks[i])
                    self.samples['velocity'].append(vel_chunks[i])
                    self.samples['pose'].append(pose_chunks[i])
                    self.samples['v_init'].append(vel_chunks[i][:1])
                    self.samples['p_init'].append(pose_chunks[i][:1])

                    self.samples['obj_imu'].append(obj_imu_chunks[i])
                    self.samples['obj_vel'].append(obj_vel_chunks[i])
                    self.samples['obj_pos'].append(obj_pos_chunks[i])
                    self.samples['obj_v_init'].append(obj_vel_chunks[i][:1])
                    self.samples['obj_p_init'].append(obj_pos_chunks[i][:1])

    def __getitem__(self, index: int):
        # Move tensors at collate-time to avoid repeated HtoD transfers for lists
        return (
            self.samples['imu'][index],
            self.samples['velocity'][index],
            self.samples['pose'][index],
            self.samples['v_init'][index],
            self.samples['p_init'][index],
            self.samples['obj_imu'][index],
            self.samples['obj_vel'][index],
            self.samples['obj_pos'][index],
            self.samples['obj_v_init'][index],
            self.samples['obj_p_init'][index],
        )


def collate_fn_with_object(batch):
    # Keep variable-length sequences as lists; stack only inits
    imu = [item[0] for item in batch]
    vel = [item[1] for item in batch]
    pose = [item[2] for item in batch]
    v_init = torch.cat([item[3] for item in batch], dim=0)
    p_init = torch.cat([item[4] for item in batch], dim=0)
    obj_imu = [item[5] for item in batch]
    obj_vel = [item[6] for item in batch]
    obj_pos = [item[7] for item in batch]
    obj_v_init = torch.cat([item[8] for item in batch], dim=0)
    obj_p_init = torch.cat([item[9] for item in batch], dim=0)

    return imu, vel, pose, v_init, p_init, obj_imu, obj_vel, obj_pos, obj_v_init, obj_p_init


