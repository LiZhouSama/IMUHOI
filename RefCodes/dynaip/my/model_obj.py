import torch
import torch.nn.functional as F
from torch import nn
import articulate as art
from human_body_prior.body_model.body_model import BodyModel
from pytorch3d import transforms
from my.dataset_trans_obj import _SENSOR_POS_INDICES, _SENSOR_ROT_INDICES, _VEL_SELECTION_INDICES, \
                             _REDUCED_INDICES, _IGNORED_INDICES, _SENSOR_NAMES, _SENSOR_VEL_NAMES, _REDUCED_POSE_NAMES

class RNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=False, dropout=0.2):
        super(RNN, self).__init__()
        self.n_hidden = n_hidden
        self.n_rnn_layer = n_rnn_layer
        self.num_directions = 2 if bidirectional else 1
        self.rnn = nn.LSTM(
            n_hidden,
            n_hidden,
            n_rnn_layer,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if n_rnn_layer > 1 else 0.0,
        )
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.linear2 = nn.Linear(n_hidden * self.num_directions, n_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h=None):
        x = self.dropout(F.relu(self.linear1(x)))
        output, _ = self.rnn(x, h)
        output = self.linear2(output)
        return output


class RNNWithInit(RNN):
    def __init__(self, n_input: int, n_output: int, n_hidden: int, n_init: int, n_rnn_layer: int,
                 bidirectional=False, dropout=0.2):
        super().__init__(n_input, n_output, n_hidden, n_rnn_layer, bidirectional, dropout)
        self.n_rnn_layer = n_rnn_layer
        self.n_hidden = n_hidden
        self.bidirectional = bidirectional
        self.init_net = torch.nn.Sequential(
            torch.nn.Linear(n_init, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden * n_rnn_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden * n_rnn_layer, 2 * (2 if bidirectional else 1) * n_rnn_layer * n_hidden)
        )

    def forward(self, inputs, _=None):
        x, x_init = inputs
        batch_size = x.shape[0]
        num_directions = 2 if self.bidirectional else 1
        nd = self.n_rnn_layer * num_directions
        nh = self.n_hidden
        init = self.init_net(x_init).view(batch_size, 2, nd, nh)
        h, c = init[:, 0], init[:, 1]
        h = h.permute(1, 0, 2).contiguous()
        c = c.permute(1, 0, 2).contiguous()
        return super(RNNWithInit, self).forward(x, (h, c))


class SubPoser(nn.Module):
    def __init__(self, n_input, v_output, p_output, n_hidden, num_layer, dropout, extra_dim=0):
        super(SubPoser, self).__init__()
        self.extra_dim = extra_dim
        self.v_output = v_output
        self.p_output = p_output
        self.rnn1 = RNNWithInit(
            n_input=n_input - extra_dim,
            n_output=v_output,
            n_hidden=n_hidden,
            n_init=v_output,
            n_rnn_layer=num_layer,
            bidirectional=False,
            dropout=dropout,
        )
        self.rnn2 = RNNWithInit(
            n_input=n_input + v_output,
            n_output=p_output,
            n_hidden=n_hidden,
            n_init=p_output,
            n_rnn_layer=num_layer,
            bidirectional=False,
            dropout=dropout,
        )

    def forward(self, x, v_init, p_init):
        if self.extra_dim != 0:
            x_v = x[..., :-self.extra_dim]
        else:
            x_v = x
        v = self.rnn1((x_v, v_init))
        p_input = torch.cat((x, v), dim=-1)
        p = self.rnn2((p_input, p_init))
        return v, p


class ObjectPoser(nn.Module):
    """Predict object velocity from object IMU (+ optional global human context)."""

    def __init__(self, n_glb: int, dropout: float = 0.2):
        super().__init__()
        n_input = 12 + n_glb  # obj imu (12) + global human context
        n_hidden = 128
        num_layer = 2
        self.obj_rnn = RNNWithInit(
            n_input=n_input,
            n_output=3,
            n_hidden=n_hidden,
            n_init=3,
            n_rnn_layer=num_layer,
            bidirectional=False,
            dropout=dropout,
        )

    def forward(self, obj_imu: torch.Tensor, s_glb: torch.Tensor, obj_v_init: torch.Tensor) -> torch.Tensor:
        x = torch.cat((obj_imu, s_glb), dim=-1)
        return self.obj_rnn((x, obj_v_init))


class PoserWithObject(nn.Module):
    """基于DynaIP的姿态估计网络，增加了物体估计分支"""
    
    def __init__(self, body_model_path=None, fps=60.0):
        super(PoserWithObject, self).__init__()
        n_hidden = 200
        num_layer = 2
        dropout = 0.2
        n_glb = 6
        self.fps = fps

        self.posers = nn.ModuleList([
            SubPoser(n_input=48 + n_glb, v_output=12, p_output=12,
                     n_hidden=n_hidden, num_layer=num_layer, dropout=dropout, extra_dim=n_glb), # 腿
            SubPoser(n_input=24 + n_glb, v_output=6, p_output=24,
                     n_hidden=n_hidden, num_layer=num_layer, dropout=dropout, extra_dim=n_glb), # 躯干
            SubPoser(n_input=36 + n_glb, v_output=6, p_output=24,
                     n_hidden=n_hidden, num_layer=num_layer, dropout=dropout, extra_dim=n_glb), # 手臂
        ])

        self.glb = RNN(n_input=72, n_output=n_glb, n_hidden=36, n_rnn_layer=1, dropout=dropout)
        self.obj_branch = ObjectPoser(n_glb=n_glb, dropout=dropout)

        self.sensor_names = _SENSOR_NAMES
        self.v_names = _SENSOR_VEL_NAMES
        self.p_names = _REDUCED_POSE_NAMES
        self.body_model = None
        self.body_model_device = None
        if body_model_path is not None and BodyModel is not None:
            try:
                self.body_model = BodyModel(bm_fname=body_model_path, num_betas=16)
                self.body_model.eval()
                for param in self.body_model.parameters():
                    param.requires_grad_(False)
                self.body_model_device = torch.device('cpu')
                print(f"成功加载Body Model: {body_model_path}")
            except Exception as e:
                print(f"加载Body Model失败: {e}")
        self.smpl_parents = torch.tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21], dtype=torch.long)

        self._generate_indices_list()


    def _find_indices(self, elements, lst):
        return [lst.index(e) for e in elements if e in lst]

    def _generate_indices_list(self):   # 排列方式取决于_REDUCED_POSE_NAMES
        posers_config = [
            {'sensor': ['Root', 'LeftLowerLeg', 'RightLowerLeg', 'Head'], 'velocity': ['Root', 'LeftFoot', 'RightFoot', 'Head'],  # 腿
             'pose': ['LeftHip', 'RightHip']},
            {'sensor': ['Root', 'Head'], 'velocity': ['Root', 'Head'],  # 躯干
             'pose': ['Spine1', 'Spine2', 'Spine3', 'Neck']},
            {'sensor': ['Root', 'LeftForeArm', 'RightForeArm'], 'velocity': ['LeftHand', 'RightHand'],  # 手臂
             'pose': ['LeftCollar', 'RightCollar', 'LeftShoulder', 'RightShoulder']},
        ]
        self.indices = []
        for i in range(len(self.posers)):
            self.indices.append({
                'sensor_indices': self._find_indices(posers_config[i]['sensor'], self.sensor_names),
                'v_indices': self._find_indices(posers_config[i]['velocity'], self.v_names),
                'p_indices': self._find_indices(posers_config[i]['pose'], self.p_names)
            })

    def _global2local(self, global_rotmats, parents):
        """
        将全局旋转矩阵转换为局部旋转矩阵。

        Args:
            global_rotmats: 全局旋转矩阵 [batch_size, num_joints, 3, 3]
            parents: 父关节索引数组 (NumPy array)，-1 表示根关节

        Returns:
            local_rotmats: 局部旋转矩阵 [batch_size, num_joints, 3, 3]
        """
        batch_size, num_joints, _, _ = global_rotmats.shape
        device = global_rotmats.device
        
        local_rotmats = torch.zeros_like(global_rotmats)
        
        # 根关节的局部旋转等于其全局旋转
        local_rotmats[:, 0] = global_rotmats[:, 0]
        
        # 遍历非根关节
        for i in range(1, num_joints):
            parent_idx = parents[i]
            
            # 获取父关节和当前关节的全局旋转
            R_global_parent = global_rotmats[:, parent_idx]
            R_global_current = global_rotmats[:, i]
            
            # 计算父关节全局旋转的逆（转置）
            R_global_parent_inv = R_global_parent.transpose(-1, -2)
            
            # 计算局部旋转: R_local = R_parent_inv * R_global
            R_local_current = torch.matmul(R_global_parent_inv, R_global_current)
            
            local_rotmats[:, i] = R_local_current
        
        return local_rotmats 

    def _reduced_glb_6d_to_full_glb_mat(self, glb_reduced_pose, orientation):
        ignored_parents = self.smpl_parents[_IGNORED_INDICES]
        root_rotation = orientation[:, 0].view(-1, 3, 3)
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, len(_REDUCED_INDICES), 3, 3)
        glb_reduced_pose = root_rotation.unsqueeze(1).matmul(glb_reduced_pose)
        orientation[:, 1:] = root_rotation.unsqueeze(1).matmul(orientation[:, 1:])
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, _REDUCED_INDICES] = glb_reduced_pose
        global_full_pose[:, _SENSOR_ROT_INDICES] = orientation
        global_full_pose[:, _IGNORED_INDICES] = global_full_pose[:, ignored_parents]
        return global_full_pose

    def forward(self, x, v_init, p_init, obj_imu, obj_v_init):
        """x: [B, T, 6, 12] IMU tensor (batch-first)."""
        B, T, _, _ = x.shape
        device = x.device

        # -------- 1) 全局特征 --------
        x_flat = x.view(B, T, -1)                      # [B, T, 72]
        s_glb = self.glb(x_flat)                       # [B, T, Dg]

        # -------- 2) 分支 poser（已有批处理，不动）--------
        v_components, p_components = [], []

        for i, poser in enumerate(self.posers):
            sensor_indices = self.indices[i]['sensor_indices']
            sensor_feat = x[:, :, sensor_indices].contiguous().view(B, T, -1)  # [B, T, C_i]
            poser_input = torch.cat((sensor_feat, s_glb), dim=-1)              # [B, T, C_i + Dg]

            vi = v_init[:, self.indices[i]['v_indices']].contiguous().view(B, -1)
            pi = p_init[:, self.indices[i]['p_indices']].contiguous().view(B, -1)

            v_i, p_i = poser(poser_input, vi, pi)      # 期望返回 [B, T, *]
            v_components.append(v_i)
            p_components.append(p_i)

        v_pred = torch.cat(v_components, dim=-1)       # [B, T, Dv]
        p_pred = torch.cat(p_components, dim=-1)       # [B, T, Dp]

        # -------- 3) 物体分支（已有批处理，不动）--------
        obj_v_pred = self.obj_branch(obj_imu, s_glb, obj_v_init)

        return v_pred, p_pred, obj_v_pred



    @staticmethod
    def integrate_object_position(obj_v_list, obj_p0: torch.Tensor, fps: float = 60.0):
        """
        Integrate per-sequence object velocity into position estimates.
        obj_v_list: list of (T_i, 3)
        obj_p0: (B, 3) initial positions per sequence
        Returns: list of (T_i, 3) positions
        """
        dt = 1.0 / fps
        pos_list = []
        for i, v in enumerate(obj_v_list):
            p = torch.zeros_like(v)
            p0 = obj_p0[i]
            if v.shape[0] > 0:
                p[0] = p0
                if v.shape[0] > 1:
                    p[1:] = p0 + torch.cumsum(v[1:] * dt, dim=0)
            pos_list.append(p)
        return pos_list
    
    @torch.no_grad()
    def predict(self, x, v_init, p_init, obj_imu=None, obj_v_init=None):
        self.eval()

        def to_batched_tensor(tensor, target_dim, default_shape):
            if tensor is None:
                return torch.zeros(default_shape, device=device)
            if isinstance(tensor, torch.Tensor):
                if tensor.dim() == target_dim - 1:
                    return tensor.unsqueeze(0)
                return tensor
            stacked = torch.stack(tensor, dim=0)
            return stacked

        if isinstance(x, torch.Tensor):
            if x.dim() == 3:
                xs = x.unsqueeze(0)
            elif x.dim() == 4:
                xs = x
            else:
                raise ValueError('Unsupported IMU tensor shape')
        else:
            xs = torch.stack(x, dim=0)
        xs = xs.contiguous()
        device = xs.device
        B, T, _, _ = xs.shape

        obj_imu_shape = (B, T, 12)
        obj_imu_tensor = to_batched_tensor(obj_imu, 3, obj_imu_shape)
        obj_imu_tensor = obj_imu_tensor.to(device=device, dtype=xs.dtype)

        v_init_tensor = to_batched_tensor(v_init, 3, (B, len(_SENSOR_VEL_NAMES), 3)).to(device=device, dtype=xs.dtype)
        p_init_tensor = to_batched_tensor(p_init, 3, (B, len(_REDUCED_POSE_NAMES), 6)).to(device=device, dtype=xs.dtype)

        if obj_v_init is None:
            obj_v_init_tensor = torch.zeros(B, 3, device=device, dtype=xs.dtype)
        else:
            obj_v_init_tensor = to_batched_tensor(obj_v_init, 2, (B, 3)).to(device=device, dtype=xs.dtype)

        v_pred, p_pred, obj_v_pred = self.forward(
            xs, v_init_tensor, p_init_tensor, obj_imu_tensor, obj_v_init_tensor
        )

        glb_full_pose_list = []
        obj_v_seq_list = []

        for i in range(B):
            pose = p_pred[i].view(T, len(_REDUCED_POSE_NAMES), 6)
            orientation = xs[i, :, :, :9].view(T, 6, 3, 3)
            glb_full_pose = self._reduced_glb_6d_to_full_glb_mat(pose.cpu(), orientation.cpu())
            obj_v_seq = obj_v_pred[i].detach().cpu()
            glb_full_pose_list.append(glb_full_pose)
            obj_v_seq_list.append(obj_v_seq)

        if B == 1:
            return (
                glb_full_pose_list[0],
                obj_v_seq_list[0]
            )

        return (
            glb_full_pose_list,
            obj_v_seq_list,
        )