import torch
import torch.nn.functional as F
from torch import nn
from typing import List
import articulate as art
from human_body_prior.body_model.body_model import BodyModel
from pytorch3d import transforms


class RNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=False, dropout=0.2):
        super().__init__()
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
        self.init_net = nn.Sequential(
            nn.Linear(n_init, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden * n_rnn_layer),
            nn.ReLU(),
            nn.Linear(n_hidden * n_rnn_layer, 2 * (2 if bidirectional else 1) * n_rnn_layer * n_hidden)
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
        return super().forward(x, (h, c))


class SubPoser(nn.Module):
    def __init__(self, n_input, v_output, p_output, n_hidden, num_layer, dropout, extra_dim=0):
        super().__init__()
        self.extra_dim = extra_dim
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
    def __init__(self, n_glb: int, dropout: float = 0.2):
        super().__init__()
        n_input = 12 + n_glb
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


class PoserWithObjectAndTransV2(nn.Module):
    """变体：trans_b1使用IMU全量信息+IMU位置，trans_b2使用IMU全量信息+全身关节位置"""

    def __init__(self, body_model_path=None, fps=30.0):
        super().__init__()
        n_hidden = 200
        num_layer = 2
        dropout = 0.2
        n_glb = 6
        self.fps = fps

        self.posers = nn.ModuleList([
            SubPoser(n_input=36 + n_glb, v_output=6, p_output=24,
                     n_hidden=n_hidden, num_layer=num_layer, dropout=dropout, extra_dim=n_glb),
            SubPoser(n_input=48 + n_glb, v_output=12, p_output=12,
                     n_hidden=n_hidden, num_layer=num_layer, dropout=dropout, extra_dim=n_glb),
            SubPoser(n_input=24 + n_glb, v_output=6, p_output=30,
                     n_hidden=n_hidden, num_layer=num_layer, dropout=dropout, extra_dim=n_glb)
        ])

        self.glb = RNN(n_input=72, n_output=n_glb, n_hidden=36, n_rnn_layer=1, dropout=dropout)
        self.obj_branch = ObjectPoser(n_glb=n_glb, dropout=dropout)

        self.tran_b1 = RNN(n_input=72 + 18, n_output=2,
                           n_hidden=64, n_rnn_layer=2, bidirectional=True, dropout=dropout)

        self.tran_b2 = RNN(n_input=72 + 24 * 3, n_output=3,
                           n_hidden=128, n_rnn_layer=2, bidirectional=False, dropout=dropout)

        self.sensor_names = ['Root', 'LeftLowerLeg', 'RightLowerLeg', 'Head', 'LeftForeArm', 'RightForeArm']
        self.v_names = ['Root', 'Head', 'LeftHand', 'RightHand', 'LeftFoot', 'RightFoot']
        self.p_names = ['LeftUpperLeg', 'RightUpperLeg', 'L5', 'L3',
                        'T12', 'T8', 'Neck', 'LeftShoulder', 'RightShoulder', 'LeftUpperArm',
                        'RightUpperArm']

        self.position_sensor_indices = torch.tensor([0, 20, 16, 6, 13, 9], dtype=torch.long)

        self.prob_threshold = (0.5, 0.9)
        self.gravity_velocity = torch.tensor([0, 0, -0.018])
        self.floor_height = 0.0
        self.prevent_floor_penetration = True
        self._foot_joint_indices = (7, 8)

        self.smpl_parents = torch.tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21], dtype=torch.long)

        self.body_model = None
        self.body_model_device = None
        if body_model_path is not None and BodyModel is not None:
            try:
                self.body_model = BodyModel(bm_fname=body_model_path, num_betas=16)
                self.body_model.eval()
                for p in self.body_model.parameters():
                    p.requires_grad_(False)
                self.body_model_device = torch.device('cpu')
                print(f"成功加载Body Model: {body_model_path}")
            except Exception as e:
                print(f"加载Body Model失败: {e}")
                self.body_model = None

        self._generate_indices_list()

    def _find_indices(self, elements: List[str], lst: List[str]):
        return [lst.index(e) for e in elements if e in lst]

    def _generate_indices_list(self):
        posers_config = [
            {'sensor': ['Root', 'LeftForeArm', 'RightForeArm'], 'velocity': ['LeftHand', 'RightHand'],
             'pose': ['LeftShoulder', 'LeftUpperArm', 'RightShoulder', 'RightUpperArm']},
            {'sensor': ['Root', 'LeftLowerLeg', 'RightLowerLeg', 'Head'], 'velocity': ['Root', 'LeftFoot', 'RightFoot', 'Head'],
             'pose': ['LeftUpperLeg', 'RightUpperLeg']},
            {'sensor': ['Root', 'Head'], 'velocity': ['Root', 'Head'],
             'pose': ['L5', 'L3', 'T12', 'T8', 'Neck']},
        ]
        self.indices = []
        for cfg in posers_config:
            self.indices.append({
                'sensor_indices': self._find_indices(cfg['sensor'], self.sensor_names),
                'v_indices': self._find_indices(cfg['velocity'], self.v_names),
                'p_indices': self._find_indices(cfg['pose'], self.p_names)
            })

    def _prob_to_weight(self, p: torch.Tensor) -> torch.Tensor:
        return (p.clamp(self.prob_threshold[0], self.prob_threshold[1]) - self.prob_threshold[0]) / (
            self.prob_threshold[1] - self.prob_threshold[0])


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

    def _reduced_glb_6d_to_full_glb_mat_xsens(self, glb_reduced_pose, orientation):
        joint_set = [19, 15, 1, 2, 3, 4, 5, 11, 7, 12, 8]
        sensor_set = [0, 20, 16, 6, 13, 9]
        ignored = [10, 14, 17, 18, 21, 22]
        parent = [9, 13, 16, 16, 20, 20]
        root_rotation = orientation[:, 0].view(-1, 3, 3)
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, len(joint_set), 3, 3)
        glb_reduced_pose = root_rotation.unsqueeze(1).matmul(glb_reduced_pose)
        orientation[:, 1:] = root_rotation.unsqueeze(1).matmul(orientation[:, 1:])
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 23, 1, 1)
        global_full_pose[:, joint_set] = glb_reduced_pose
        global_full_pose[:, sensor_set] = orientation
        global_full_pose[:, ignored] = global_full_pose[:, parent]
        return global_full_pose

    def _glb_mat_xsens_to_glb_mat_smpl(self, glb_full_pose_xsens):
        glb_full_pose_smpl = torch.eye(3, device=glb_full_pose_xsens.device).repeat(glb_full_pose_xsens.shape[0], 24, 1, 1)
        indices = [0, 19, 15, 1, 20, 16, 3, 21, 17, 4, 22, 18, 5, 11, 7, 6, 12, 8, 13, 9, 13, 9, 13, 9]
        for idx, i in enumerate(indices):
            glb_full_pose_smpl[:, idx] = glb_full_pose_xsens[:, i]
        return glb_full_pose_smpl

    def _compute_fk_joints_batched(self, glb_p_out_tensor: torch.Tensor, orientation: torch.Tensor):
        if self.body_model is None:
            return None

        B, T, _ = glb_p_out_tensor.shape
        device = glb_p_out_tensor.device
        BT = B * T

        glb_pose_6d = glb_p_out_tensor.reshape(BT, 11, 6)
        idx_order = [4, 5, 6, 7, 8, 9, 10, 0, 2, 1, 3]
        glb_pose_6d = glb_pose_6d[:, idx_order]

        orient_bt = orientation.reshape(BT, 6, 3, 3)
        glb_pose_xsens = self._reduced_glb_6d_to_full_glb_mat_xsens(glb_pose_6d, orient_bt)
        glb_pose_smpl = self._glb_mat_xsens_to_glb_mat_smpl(glb_pose_xsens)
        local_pose_smpl = self._global2local(glb_pose_smpl, self.smpl_parents.tolist())

        pose_aa = transforms.matrix_to_axis_angle(local_pose_smpl.detach().cpu()).to(device)

        try:
            with torch.no_grad():
                body_out = self.body_model(
                    pose_body=pose_aa[:, 1:22].reshape(BT, 63),
                    root_orient=pose_aa[:, 0].reshape(BT, 3),
                    trans=torch.zeros(BT, 3, device=device)
                )
            joints_bt = body_out.Jtr[:, :24, :]
            return joints_bt.reshape(B, T, 24, 3)
        except Exception as e:
            print(f"FK计算失败(batch): {e}")
            return None

    def forward(self, x, v_init, p_init, obj_imu, obj_v_init):
        B, T, _, _ = x.shape
        device = x.device

        if self.body_model is not None and (self.body_model_device is None or self.body_model_device != device):
            self.body_model = self.body_model.to(device)
            self.body_model_device = device

        x_flat = x.view(B, T, -1)
        s_glb = self.glb(x_flat)

        v_components: List[torch.Tensor] = []
        p_components: List[torch.Tensor] = []
        v_lower = None

        for i, poser in enumerate(self.posers):
            sensor_idx = self.indices[i]['sensor_indices']
            sensor_feat = x[:, :, sensor_idx].contiguous().view(B, T, -1)
            poser_input = torch.cat((sensor_feat, s_glb), dim=-1)

            vi = v_init[:, self.indices[i]['v_indices']].contiguous().view(B, -1)
            pi = p_init[:, self.indices[i]['p_indices']].contiguous().view(B, -1)

            v_i, p_i = poser(poser_input, vi, pi)
            v_components.append(v_i)
            p_components.append(p_i)

        v_pred = torch.cat(v_components, dim=-1)
        glb_p_pred = torch.cat(p_components, dim=-1)
        obj_v_pred = self.obj_branch(obj_imu, s_glb, obj_v_init)

        orientation = x[:, :, :, :9].contiguous().view(B, T, 6, 3, 3)
        root_R = orientation[:, :, 0]

        joints_pos = self._compute_fk_joints_batched(glb_p_pred, orientation)
        if joints_pos is not None:
            root_pos = joints_pos[:, :, 0]
            rel_pos = joints_pos - root_pos.unsqueeze(2)
            root_rot_T = root_R.transpose(-1, -2)
            joints_root = torch.matmul(root_rot_T.unsqueeze(2), rel_pos.unsqueeze(-1)).squeeze(-1)
        else:
            joints_root = torch.zeros(B, T, 24, 3, device=device)
        # joints_root = joints_pos

        sensor_positions = torch.index_select(
            joints_root,
            dim=2,
            index=self.position_sensor_indices.to(device)
        ).reshape(B, T, -1)
        full_joint_positions = joints_root.reshape(B, T, -1)

        tran_b1_input = torch.cat((x_flat, sensor_positions), dim=-1)
        contact_pred = self.tran_b1(tran_b1_input)

        tran_b2_input = torch.cat((x_flat, full_joint_positions), dim=-1)
        root_vel_local_pred = self.tran_b2(tran_b2_input)

        if joints_pos is not None:
            root_vel_pred = self._fuse_velocities_batched(
                root_vel_local_pred, contact_pred, joints_pos, root_R
            )
        else:
            root_vel_pred = torch.matmul(root_R, root_vel_local_pred.unsqueeze(-1)).squeeze(-1) / self.fps

        root_vel_pred = self._apply_floor_penetration(root_vel_pred, joints_pos)
        root_trans_pred = self._integrate_root_velocity(root_vel_pred)

        return v_pred, glb_p_pred, obj_v_pred, contact_pred, root_vel_local_pred, root_vel_pred, root_trans_pred


    def _apply_floor_penetration(self, root_velocities: torch.Tensor, joints_pos: torch.Tensor) -> torch.Tensor:
        """
        root_velocities: [B, T, 3] 或 [B, T, D]，z-up 假设 axis=2
        joints_pos:      [B, T, J, 3] 关节世界坐标或相对 root（与你原先一致即可）
        返回:            调整后的 root_velocities（同形状）
        """
        if (not getattr(self, "prevent_floor_penetration", False)) or joints_pos is None:
            return root_velocities

        assert root_velocities.dim() == 3, "expect [B, T, D]"
        B, T, D = root_velocities.shape
        axis = 2  # z-up
        foot_start, foot_end = self._foot_joint_indices

        # 没有脚关节可用时，直接返回（避免 amin 空轴报错）
        if foot_end < foot_start or (joints_pos.size(2) == 0):
            return root_velocities

        device = root_velocities.device
        dtype = root_velocities.dtype
        F_floor = torch.as_tensor(self.floor_height, device=device, dtype=dtype)

        # 1) 取出该帧最低的脚 z 偏移/坐标，形状 [B, T]
        #    （与你原先语义一致：这里使用 foot_start..foot_end 的最小 z）
        foot_z = joints_pos[:, :, foot_start:foot_end + 1, axis]              # [B, T, F_foots]
        foot_min_z = foot_z.amin(dim=2)                                       # [B, T]

        # 2) 原始 z 方向速度序列 s，形状 [B, T]
        s = root_velocities[..., axis]                                        # [B, T]

        # 3) 原始累计和 S_naive_{t+1}（从 0 开始累计）
        S_naive_next = torch.cumsum(s, dim=1)                                 # [B, T]

        # 4) 每一步的累计下界 A_t = F - f_t（h_0 = 0）
        A = F_floor - foot_min_z                                              # [B, T]

        # 5) 需要的“累计修正”：U = relu(A - S_naive_{t+1}); C = cummax(U)
        U_pos = torch.clamp_min(A - S_naive_next, 0)                          # [B, T]
        C = torch.cummax(U_pos, dim=1).values                                 # [B, T]
        C_prev = F.pad(C[:, :-1], (1, 0))                                     # [B, T], 左侧补 0

        # 6) 差分回每一步的修正 Δs_t，并得到 s_adj
        delta = C - C_prev                                                    # [B, T]
        s_adj = s + delta                                                     # [B, T]

        adjusted = root_velocities.clone()
        adjusted[..., axis] = s_adj
        return adjusted


    def _integrate_root_velocity(self, root_velocities: torch.Tensor) -> torch.Tensor:
        """
        矢量化积分：按时间维做 cumsum，再除以 fps。
        支持 [T, D] 或 [B, T, D]。
        """
        fps = float(self.fps)
        if root_velocities.dim() == 2:            # [T, D]
            return torch.cumsum(root_velocities, dim=0) / fps
        elif root_velocities.dim() == 3:          # [B, T, D]
            return torch.cumsum(root_velocities, dim=1) / fps
        else:
            raise ValueError("root_velocities must be [T,D] or [B,T,D]")

    def _fuse_velocities_batched(self, root_vel_pred, contact_logits, joints_pos, root_rotation, vel_scale: float = 3.0):
        device = root_vel_pred.device
        B, T, _ = root_vel_pred.shape

        tran_b2_vel = torch.matmul(root_rotation, root_vel_pred.unsqueeze(-1)).squeeze(-1) * vel_scale / self.fps

        lfoot_pos = joints_pos[:, :, 7]
        rfoot_pos = joints_pos[:, :, 8]
        lfoot_vel = torch.zeros_like(lfoot_pos)
        rfoot_vel = torch.zeros_like(rfoot_pos)
        if T > 1:
            lfoot_vel[:, 1:] = lfoot_pos[:, :-1] - lfoot_pos[:, 1:]
            rfoot_vel[:, 1:] = rfoot_pos[:, :-1] - rfoot_pos[:, 1:]

        contact_sigmoid = torch.sigmoid(contact_logits)
        contact_idx = contact_logits.argmax(dim=-1).unsqueeze(-1)
        contact_mask = (contact_idx == 0)
        tran_b1_vel = torch.where(contact_mask, lfoot_vel, rfoot_vel)

        gravity = self.gravity_velocity.to(device)
        tran_b1_vel = tran_b1_vel + gravity

        contact_max = contact_sigmoid.max(dim=-1).values.unsqueeze(-1)
        weight = self._prob_to_weight(contact_max)
        fused_vel = art.math.lerp(tran_b2_vel, tran_b1_vel, weight) * self.fps
        return fused_vel


    @staticmethod
    def velocity_to_root_position(velocity, fps=30):
        velocity = velocity / fps
        if velocity.numel() == 0:
            return velocity.clone()
        return torch.cumsum(velocity, dim=0)



