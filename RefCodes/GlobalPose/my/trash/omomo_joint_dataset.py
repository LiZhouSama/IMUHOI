import os
import torch
from torch.utils.data import Dataset


def _finite_diff_velocity(x: torch.Tensor, dt: float) -> torch.Tensor:
    """
    x: [T, 3] position -> velocity [T, 3], v[0]=0
    """
    T = x.shape[0]
    if T <= 1:
        return torch.zeros_like(x)
    v = torch.zeros_like(x)
    v[1:] = (x[1:] - x[:-1]) / dt
    return v


class OMOMOJointSeqDataset(Dataset):
    """
    Joint dataset for human + object from processed single-sequence .pt files.
    Required keys per file: 'aM' [T,6,3], 'wM' [T,6,3], 'RMB' [T,6,3,3], 'pose' [T,24,3], 'tran' [T,3],
                           'obj_imu' [T,1,9], 'obj_trans' [T,3]
    """

    def __init__(self, seq_dir: str, fps: int = 30):
        super().__init__()
        self.seq_dir = seq_dir
        self.dt = 1.0 / float(fps)
        files = [f for f in os.listdir(seq_dir) if f.endswith('.pt')]
        files.sort()
        self.files = [os.path.join(seq_dir, f) for f in files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        d = torch.load(self.files[idx], map_location='cpu')

        aM = d['aM'].float()            # [T,6,3]
        wM = d['wM'].float()            # [T,6,3]
        RMB = d['RMB'].float()          # [T,6,3,3]
        tran = d['tran'].float()        # [T,3]
        obj_imu = d['obj_imu'].float()   # [T,9]
        if obj_imu.ndim == 3 and obj_imu.size(1) == 1:
            obj_imu = obj_imu[:, 0, :]   # -> [T,9]
        elif obj_imu.ndim == 2 and obj_imu.size(-1) == 9:
            pass
        else:
            raise ValueError(f"obj_imu unexpected shape: {tuple(obj_imu.shape)}")
        obj_trans = d['obj_trans'].float()     # [T,3]
        pose_aa24 = d.get('pose', None)        # [T,24,3] axis-angle
        pose_aa24 = pose_aa24.float() if pose_aa24 is not None else None

        human_vel = _finite_diff_velocity(tran, self.dt)  # [T,3]
        obj_vel = _finite_diff_velocity(obj_trans, self.dt)

        return {
            'aM': aM,                # [T,6,3]
            'wM': wM,                # [T,6,3]
            'RMB': RMB,              # [T,6,3,3]
            'tran': tran,            # [T,3]
            'human_vel': human_vel,  # [T,3]
            'obj_imu': obj_imu,      # [T,9]
            'obj_trans': obj_trans,  # [T,3]
            'obj_vel': obj_vel,      # [T,3]
            'pose': pose_aa24,       # [T,24,3]
            'dt': torch.tensor(self.dt),
            'name': os.path.basename(self.files[idx]).replace('.pt', '')
        }


def collate_pad_joint(batch):
    B = len(batch)
    lengths = [item['aM'].shape[0] for item in batch]
    T_max = max(lengths)

    def pad(tensors, pad_shape):
        out = torch.zeros((B, T_max) + pad_shape, dtype=tensors[0].dtype)
        for i, t in enumerate(tensors):
            out[i, : t.shape[0]] = t
        return out

    aM = pad([b['aM'] for b in batch], (6, 3))
    wM = pad([b['wM'] for b in batch], (6, 3))
    RMB = pad([b['RMB'] for b in batch], (6, 3, 3))
    tran = pad([b['tran'] for b in batch], (3,))
    human_vel = pad([b['human_vel'] for b in batch], (3,))
    obj_imu = pad([b['obj_imu'] for b in batch], (9,))
    obj_trans = pad([b['obj_trans'] for b in batch], (3,))
    obj_vel = pad([b['obj_vel'] for b in batch], (3,))
    # 可选的姿态 [T,24,3]
    has_pose = all(b.get('pose', None) is not None for b in batch)
    pose = pad([b['pose'] for b in batch], (24, 3)) if has_pose else None

    mask = torch.zeros(B, T_max, dtype=torch.bool)
    for i, L in enumerate(lengths):
        mask[i, : L] = True

    result = {
        'aM': aM,
        'wM': wM,
        'RMB': RMB,
        'tran': tran,
        'human_vel': human_vel,
        'obj_imu': obj_imu,
        'obj_trans': obj_trans,
        'obj_vel': obj_vel,
        'mask': mask,
        'lengths': torch.tensor(lengths),
        'dt': batch[0]['dt'],
        'names': [b['name'] for b in batch]
    }
    if has_pose:
        result['pose'] = pose
    return result


class AggregatedDataset:
    """适配聚合格式数据的Dataset"""
    
    def __init__(self, data_file, fps=30):
        self.fps = fps
        self.dt = 1.0 / fps
        
        print(f"加载聚合数据: {data_file}")
        self.data = torch.load(data_file, map_location='cpu')
        
        self.num_sequences = len(self.data['name'])
        self.has_object_data = 'obj_imu' in self.data and 'obj_trans' in self.data
        
        print(f"加载了 {self.num_sequences} 个序列")
        print(f"包含物体数据: {self.has_object_data}")
        
        if not self.has_object_data:
            print("警告: 数据中缺少物体信息，物体网络训练将被跳过")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        """返回格式与OMOMOJointSeqDataset兼容的数据"""
        name = self.data['name'][idx]
        RIS = self.data['RIS'][idx].float()    # [T, 6, 3, 3]
        aS = self.data['aS'][idx].float()      # [T, 6, 3]
        wS = self.data['wS'][idx].float()      # [T, 6, 3]
        tran = self.data['tran'][idx].float()  # [T, 3]
        pose = self.data['pose'][idx].float()  # [T, 24, 3]
        
        T = RIS.shape[0]
        
        # 转换为世界坐标系下的IMU数据（与OMOMOJointSeqDataset格式一致）
        G_VEC = torch.tensor([0.0, -9.8, 0.0])
        aM = torch.einsum('tijk,tik->tij', RIS, aS) + G_VEC.expand(T, 6, 3)
        wM = torch.einsum('tijk,tik->tij', RIS, wS)
        RMB = torch.eye(3).expand(T, 6, 3, 3)  # 单位矩阵
        
        human_vel = _finite_diff_velocity(tran, self.dt)
        
        # 物体数据 - 确保正确的shape
        if self.has_object_data:
            obj_imu = self.data['obj_imu'][idx].float()  # [T, 9] 直接使用
            obj_trans = self.data['obj_trans'][idx].float()  # [T, 3]
            obj_vel = _finite_diff_velocity(obj_trans, self.dt)
        else:
            # 如果没有物体数据，创建零值（避免训练错误）
            obj_imu = torch.zeros(T, 9)  # [T, 9] 与collate_pad_joint兼容
            obj_trans = torch.zeros(T, 3)
            obj_vel = torch.zeros(T, 3)
        
        return {
            'aM': aM,                # [T, 6, 3]
            'wM': wM,                # [T, 6, 3]
            'RMB': RMB,              # [T, 6, 3, 3]
            'tran': tran,            # [T, 3]
            'human_vel': human_vel,  # [T, 3]
            'obj_imu': obj_imu,      # [T, 9] - 与collate兼容
            'obj_trans': obj_trans,  # [T, 3]
            'obj_vel': obj_vel,      # [T, 3]
            'pose': pose,            # [T, 24, 3]
            'dt': torch.tensor(self.dt),
            'name': name
        }


