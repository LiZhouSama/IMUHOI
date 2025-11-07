import os
import glob
from typing import List, Tuple

import torch
import torch.nn.functional as F

from utils.data import normalize_imu


def central_diff(a: torch.Tensor, dt: float) -> torch.Tensor:
    # a: [T, ...]
    if a.shape[0] <= 1:
        return torch.zeros_like(a)
    # simple forward/backward for endpoints, central inside
    vel = torch.zeros_like(a)
    vel[1:-1] = (a[2:] - a[:-2]) / (2.0 * dt)
    vel[0] = (a[1] - a[0]) / dt
    vel[-1] = (a[-1] - a[-2]) / dt
    return vel


def build_imu_from_joints(rotation_global: torch.Tensor,
                          position_global: torch.Tensor,
                          sensor_indices_pos: List[int],
                          sensor_indices_rot: List[int],
                          fps: float = 60.0) -> torch.Tensor:
    """
    Build 6-IMU data with possibly different joint selections for acceleration (pos) and orientation (rot),
    matching the pattern used in dataloader.IMUDataset (IMU_JOINTS_POS vs IMU_JOINTS_ROT).
    Returns imu: [T, 6, 12] (ori(9) + acc(3) per sensor) in DynaIP normalize_imu format.
    """
    T = rotation_global.shape[0]
    dt = 1.0 / fps
    sel_R = rotation_global[:, sensor_indices_rot]  # [T, 6, 3, 3]
    sel_pos = position_global[:, sensor_indices_pos]  # [T, 6, 3]

    # accelerations from positions (world frame finite differences)
    vel = central_diff(sel_pos, dt)
    acc = central_diff(vel, dt)  # [T, 6, 3]

    # normalize to get the same layout/order as DynaIP (root first sensor is pelvis/belly proxy)
    data = normalize_imu(acc.view(T, 6, 3), sel_R.view(T, 6, 3, 3))
    return data  # [T, 6, 12]


def compute_joint_velocity_and_position_like_process_xsens(rotation_global: torch.Tensor,
                                                           position_global: torch.Tensor,
                                                           fps: float = 60.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Match Ref Codes/dynaip/datasets/process.py::process_xsens
      - remove horizontal root x/z motion
      - velocity = diff * fps; prepend first frame copy
      - relative vel: [root, others - root]
      - rotate by root orientation to root frame
      - position rotated to root frame (after horizontal removal)
    Returns: (velocity_root_frame [T,J,3], position_root_frame [T,J,3])
    """
    T = position_global.shape[0]
    pos = position_global.clone()
    # remove horizon movement
    pos[:, :, 0] = pos[:, :, 0] - pos[:, :1, 0]
    pos[:, :, 2] = pos[:, :, 2] - pos[:, :1, 2]

    # velocity in world
    vel_w = (pos[1:] - pos[:-1]) * fps
    vel_w = torch.cat((vel_w[:1], vel_w), dim=0)

    # relative to root
    root_vel = vel_w[:, :1]  # [T,1,3]
    rel_vel = torch.cat((root_vel, vel_w[:, 1:] - root_vel), dim=1)  # [T,J,3]

    # rotate by root orientation to root frame
    root_R = rotation_global[:, 0]  # [T,3,3]
    vel_root = rel_vel.bmm(root_R)
    pos_root = pos.bmm(root_R)
    return vel_root, pos_root


def extract_orientation_16x6_from_rotation_global_dip(rotation_global: torch.Tensor) -> torch.Tensor:
    """Return 16x6 orientation in the exact DIP joint_mask order used in process.py."""
    T = rotation_global.shape[0]
    DIP_MASK_16 = [1, 2, 3, 4, 5, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    r16 = rotation_global[:, DIP_MASK_16]  # [T,16,3,3]
    r16_root = r16[:,:1].transpose(2, 3).matmul(r16)
    r16_6d = r16_root[:, :, :, :2].transpose(2, 3).reshape(T, 16, 6)
    return r16_6d


def build_object_imu(obj_rot: torch.Tensor, obj_com_pos: torch.Tensor, fps: float = 60.0) -> torch.Tensor:
    """
    Build object IMU data from object rotation and COM position.
    Returns obj_imu: [T, 12] (orientation(9) + acceleration(3))
    """
    T = obj_rot.shape[0]
    dt = 1.0 / fps
    
    # Convert 6D rotation to 9D (first 2 columns of rotation matrix)
    if obj_rot.shape[-1] == 6:
        # 6D representation to rotation matrix
        a1 = obj_rot[..., :3] / (torch.norm(obj_rot[..., :3], dim=-1, keepdim=True) + 1e-8)
        a2 = obj_rot[..., 3:6]
        a2 = a2 - torch.sum(a1 * a2, dim=-1, keepdim=True) * a1
        a2 = a2 / (torch.norm(a2, dim=-1, keepdim=True) + 1e-8)
        a3 = torch.cross(a1, a2, dim=-1)
        obj_rot_matrix = torch.stack([a1, a2, a3], dim=-1)  # [T, 3, 3]
    else:
        obj_rot_matrix = obj_rot  # assume already [T, 3, 3]
    
    # Flatten orientation to 9D
    orientation = obj_rot_matrix.reshape(T, 9)
    
    # Compute acceleration from position
    vel = central_diff(obj_com_pos, dt)
    acc = central_diff(vel, dt)  # [T, 3]
    
    # Combine orientation and acceleration
    obj_imu = torch.cat([orientation, acc], dim=-1)  # [T, 12]
    return obj_imu


def compute_object_velocity_and_position(obj_com_pos: torch.Tensor, fps: float = 60.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute object velocity and position (normalized).
    Returns: (velocity [T, 3], position [T, 3])
    """
    T = obj_com_pos.shape[0]
    
    # Remove horizontal root movement (align with human processing)
    pos = obj_com_pos.clone()
    pos[:, 0] = pos[:, 0] - pos[0, 0]  # remove x offset
    pos[:, 2] = pos[:, 2] - pos[0, 2]  # remove z offset
    
    # Velocity computation
    vel = central_diff(pos, 1.0 / fps)
    
    return vel, pos


def convert_split(source_dir: str, target_root: str, dataset_name: str, fps: float = 60.0):
    if not os.path.isdir(source_dir):
        return
    # source_dir should be .../<phase> (train or test)
    phase = os.path.basename(source_dir)
    out_dir = os.path.join(target_root, phase, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    pt_files = glob.glob(os.path.join(source_dir, '*.pt'))
    for pt in pt_files:
        d = torch.load(pt)
        rot = d['rotation_global'].float()  # [T, J, 3, 3]
        pos = d['position_global_full_gt_world'].float()  # [T, J, 3]

        # DIP-style 6-IMU mapping on SMPL indices, in order:
        #   [root proxy, left knee, right knee, head, left elbow, right elbow]
        sensors_pos = [0, 7, 8, 15, 20, 21]  # for accelerations (positions)
        sensors_rot = [0, 4, 5, 15, 18, 19]  # for orientations (rotations)
        imu = build_imu_from_joints(rot, pos, sensors_pos, sensors_rot, fps=fps)  # [T, 6, 12]

        vel_root, pos_root = compute_joint_velocity_and_position_like_process_xsens(rot, pos, fps=fps)  # [T, J, 3]

        # Build 16x6 orientation in DIP/XSENS joint_mask order (as in process.py),
        # and store it in 'joint.orientation' flattened to match DynaIP style.
        orient16 = extract_orientation_16x6_from_rotation_global_dip(rot)  # [T,16,6]
        
        # Build object data if available
        obj_imu_data = None
        obj_velocity = None
        obj_position = None
        
        if 'obj_rot' in d and 'obj_trans' in d:
            obj_rot = d['obj_rot'].float()
            obj_trans = d['obj_trans'].float()
            
            # Compute object IMU data
            obj_imu_data = build_object_imu(obj_rot, obj_trans, fps=fps)
            
            # Compute object velocity and position
            obj_velocity, obj_position = compute_object_velocity_and_position(obj_trans, fps=fps)
        
        out = {
            'joint': {
                'orientation': orient16.reshape(rot.shape[0], -1)[6:-6],  # [T, 16*6]
                'velocity': vel_root[6:-6],
                'position': pos_root[6:-6],
            },
            'imu': {
                'imu': imu[6:-6],
            }
        }
        
        # Add object IMU data to imu section
        if obj_imu_data is not None:
            out['imu']['obj_imu'] = obj_imu_data[6:-6]
        
        # Add object motion data
        if obj_velocity is not None and obj_position is not None:
            out['object'] = {
                'velocity': obj_velocity[6:-6],
                'position': obj_position[6:-6],
            }

        # # carry through object annotations if present from preprocess.py
        # for k in ('obj_trans', 'obj_rot', 'obj_scale', 'obj_com_pos'):
        #     if k in d:
        #         out[k] = d[k]
        # for k in ('lhand_contact', 'rhand_contact', 'obj_contact'):
        #     if k in d:
        #         out[k] = d[k]

        out_path = os.path.join(out_dir, os.path.basename(pt))
        torch.save(out, out_path)


def run(source_train_dir: str, source_test_dir: str, target_root: str, dataset_name: str, fps: float = 60.0):
    convert_split(source_train_dir, target_root, dataset_name, fps=fps)
    convert_split(source_test_dir, target_root, dataset_name, fps=fps)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_train_dir', type=str, default='/mnt/d/a_WORK/Projects/PhD/tasks/EgoIMU/processed_data_0701/train', help='Source train directory')
    parser.add_argument('--source_test_dir', type=str, default='/mnt/d/a_WORK/Projects/PhD/tasks/EgoIMU/processed_data_0701/test', help='Source test directory')
    parser.add_argument('--target_root', type=str, default='/mnt/d/a_WORK/Projects/PhD/tasks/EgoIMU/Ref Codes/dynaip/datasets/work', help='Target root directory')
    parser.add_argument('--dataset_name', type=str, default='OMOMO', help='Dataset name')
    parser.add_argument('--fps', type=float, default=30.0)
    args = parser.parse_args()
    run(args.source_train_dir, args.source_test_dir, args.target_root, args.dataset_name, args.fps)