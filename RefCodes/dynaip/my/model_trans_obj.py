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


class PoserWithObjectAndTrans(nn.Module):
    """基于DynaIP的姿态估计网络，增加了人体translation和物体估计分支"""
    
    def __init__(self, body_model_path=None, fps=60.0):
        super(PoserWithObjectAndTrans, self).__init__()
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

        # Translation分支（参考TransPose）
        # tran_b1: 脚部接触概率估计
        # 输入: 下肢双脚速度(6维，对应LeftFoot+RightFoot) + 下肢双脚IMU(24维，2个传感器) + glb(6维)
        n_feet_vel = 6  # 双脚速度 (2脚 * 3维)
        n_feet_imu = 24  # 双脚IMU (2传感器 * 12维)
        self.tran_b1 = RNN(n_input=n_feet_vel + n_feet_imu + n_glb, n_output=2, 
                          n_hidden=64, n_rnn_layer=2, bidirectional=True, dropout=dropout)
        
        # tran_b2: 根关节速度估计
        # 输入: 躯干6关节速度(18维) + 躯干IMU(24维) + glb(6维)
        n_torso_vel = 18  # 躯干6个关节速度 (6关节 * 3维)
        n_torso_imu = 24  # 躯干IMU (2传感器 * 12维)
        self.tran_b2 = RNN(n_input=n_torso_vel + n_torso_imu + n_glb, n_output=3,
                          n_hidden=128, n_rnn_layer=2, bidirectional=False, dropout=dropout)

        self.sensor_names = _SENSOR_NAMES
        self.v_names = _SENSOR_VEL_NAMES
        self.p_names = _REDUCED_POSE_NAMES
        
        # 躯干关节索引（SMPL 24关节）：Pelvis(0), Spine1(3), Spine2(6), Spine3(9), Neck(12), Head(15)
        self.torso_joints = [0, 3, 6, 9, 12, 15]
        
        # TransPose融合参数
        self.prob_threshold = (0.5, 0.9)
        self.gravity_velocity = torch.tensor([0, -0.018, 0])  # 重力速度
        self.floor_height = 0.0
        self.prevent_floor_penetration = True
        self._foot_joint_indices = (7, 8)
        self.smpl_parents = torch.tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21], dtype=torch.long)

        
        # 加载SMPL body model用于FK
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
    
    def _prob_to_weight(self, p):
        """将接触概率转换为融合权重（参考TransPose）"""
        return (p.clamp(self.prob_threshold[0], self.prob_threshold[1]) - self.prob_threshold[0]) / \
               (self.prob_threshold[1] - self.prob_threshold[0])
    
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


    def _compute_fk_joints_batched(self, glb_p_out_tensor: torch.Tensor, orientation: torch.Tensor):
        if self.body_model is None:
            return None

        B, T, _ = glb_p_out_tensor.shape
        device = glb_p_out_tensor.device
        BT = B * T

        glb_pose_6d = glb_p_out_tensor.reshape(BT, len(_REDUCED_POSE_NAMES), 6)
        imu_ori = orientation.reshape(BT, len(_SENSOR_ROT_INDICES), 3, 3)
        full_glb_pose = self._reduced_glb_6d_to_full_glb_mat(glb_pose_6d, imu_ori)
        local_pose = self._global2local(full_glb_pose, self.smpl_parents.tolist())

        pose_aa = transforms.matrix_to_axis_angle(local_pose.detach().cpu()).to(device)

        try:
            with torch.no_grad():
                body_out = self.body_model(
                    pose_body=pose_aa[:, 1:22].reshape(BT, 63),
                    root_orient=pose_aa[:, 0].reshape(BT, 3)
                )
            joints_bt = body_out.Jtr[:, :24, :]
            return joints_bt.reshape(B, T, 24, 3)
        except Exception as e:
            print(f"FK计算失败(batch): {e}")
            return None


    def _compute_torso_velocity_batched(self, joints_pos):
        """
        批量躯干速度：从关节位置差分计算躯干速度
        参数:
            joints_pos: [B, T, 24, 3]
        返回:
            torso_vel : [B, T, 18]  (6个躯干关节 * 3轴)
        """
        if joints_pos is None:
            return None

        assert joints_pos.dim() == 4 and joints_pos.shape[2] >= 16, \
            f"输入维度不对: joints_pos {joints_pos.shape}"

        B, T, J, _ = joints_pos.shape
        device = joints_pos.device

        # 关节索引与单样本版一致；确保是 LongTensor
        if isinstance(self.torso_joints, torch.Tensor):
            torso_idx = self.torso_joints.to(device=device, dtype=torch.long)
        else:
            torso_idx = torch.tensor(self.torso_joints, device=device, dtype=torch.long)
        # [B, T, 6, 3]
        torso_pos = torch.index_select(joints_pos, dim=2, index=torso_idx).contiguous()

        # 速度（沿 T 差分 * fps）
        torso_vel = torch.zeros_like(torso_pos)
        if T > 1:
            torso_vel[:, 1:] = (torso_pos[:, 1:] - torso_pos[:, :-1]) * self.fps
            torso_vel[:, 0]  = torso_vel[:, 1]      # 第0帧复制第1帧，保持长度一致
        # T == 1 时保持全零

        return torso_vel.view(B, T, -1)             # [B, T, 18]

    def forward(self, x, v_init, p_init, obj_imu, obj_v_init):
        """x: [B, T, 6, 12] IMU tensor (batch-first)."""
        B, T, _, _ = x.shape
        device = x.device

        # 确保 body_model 在同一设备
        if self.body_model is not None and (self.body_model_device is None or self.body_model_device != device):
            self.body_model = self.body_model.to(device)
            self.body_model_device = device

        # -------- 1) 全局特征 --------
        x_flat = x.view(B, T, -1)                      # [B, T, 72]
        s_glb = self.glb(x_flat)                       # [B, T, Dg]

        # -------- 2) 分支 poser（已有批处理，不动）--------
        v_components, p_components = [], []
        v_lower = None

        for i, poser in enumerate(self.posers):
            sensor_indices = self.indices[i]['sensor_indices']
            sensor_feat = x[:, :, sensor_indices].contiguous().view(B, T, -1)  # [B, T, C_i]
            poser_input = torch.cat((sensor_feat, s_glb), dim=-1)              # [B, T, C_i + Dg]

            vi = v_init[:, self.indices[i]['v_indices']].contiguous().view(B, -1)
            pi = p_init[:, self.indices[i]['p_indices']].contiguous().view(B, -1)

            v_i, p_i = poser(poser_input, vi, pi)      # 期望返回 [B, T, *]
            v_components.append(v_i)
            p_components.append(p_i)
            if i == 0:
                v_lower = v_i                          # [B, T, D_vlower]

        v_pred = torch.cat(v_components, dim=-1)       # [B, T, Dv]
        p_pred = torch.cat(p_components, dim=-1)       # [B, T, Dp]

        # -------- 3) 物体分支（已有批处理，不动）--------
        obj_v_pred = self.obj_branch(obj_imu, s_glb, obj_v_init)

        # -------- 4) 批量取出所需的 IMU / 姿态 / 速度特征 --------
        # v_lower: [B, T, ...]，脚部线速度片段
        feet_vel = v_lower[:, :, 3:9]                  # [B, T, 6]

        # 取 IMU 的通道，与原代码保持一致（k1,k2 用 [1,2]；躯干 [0,3]）
        feet_imu  = x[:, :, [1, 2]].reshape(B, T, -1)  # [B, T, 24]  (2*12)
        torso_imu = x[:, :, [0, 3]].reshape(B, T, -1)  # [B, T, 24]

        # 姿态矩阵：取前 9 个数并 reshape 成 [3,3]
        orientation = x[:, :, :, :9].contiguous().view(B, T, 6, 3, 3)  # [B, T, 6, 3, 3]
        root_R = orientation[:, :, 0]                                  # [B, T, 3, 3]

        # -------- 5) FK（批量）→ 躯干部速度（批量）--------
        joints_pos = None
        if self.body_model is not None:
            joints_pos = self._compute_fk_joints_batched(p_pred, orientation)   # 期望 [B, T, J, 3]

        if joints_pos is not None:
            torso_vel = self._compute_torso_velocity_batched(joints_pos)             # [B, T, 18] 
        else:
            torso_vel = torch.zeros(B, T, 18, device=device)

        # -------- 6) 两个 trans 分支（批量）--------
        tran_b1_input = torch.cat((feet_vel, feet_imu, s_glb), dim=-1)   # [B, T, C1]
        contact_pred  = self.tran_b1(tran_b1_input)                       # [B, T, Cc]  (等价 contact_logits)

        tran_b2_input = torch.cat((torso_vel, torso_imu, s_glb), dim=-1)  # [B, T, C2]
        root_vel_local_pred = self.tran_b2(tran_b2_input)                 # [B, T, 3]

        # -------- 7) 速度融合 / 坐标系变换（批量）--------
        if joints_pos is not None:
            root_vel_pred = self._fuse_velocities_batched(
                root_vel_local_pred, contact_pred, joints_pos, root_R
            )
        else:
            # fallback: convert local velocity to world frame
            # [B,T,3,3] @ [B,T,3,1] -> [B,T,3,1]
            root_vel_pred = torch.matmul(root_R, root_vel_local_pred.unsqueeze(-1)).squeeze(-1) / self.fps  # [B, T, 3]

        root_vel_pred = self._apply_floor_penetration(root_vel_pred, joints_pos)
        root_trans_pred = self._integrate_root_velocity(root_vel_pred)

        return v_pred, p_pred, obj_v_pred, contact_pred, root_vel_local_pred, root_vel_pred, root_trans_pred


    
    def _apply_floor_penetration(self, root_velocities: torch.Tensor, joints_pos: torch.Tensor) -> torch.Tensor:
        """
        root_velocities: [B, T, 3] 或 [B, T, D]，y-up 假设 axis=1
        joints_pos:      [B, T, J, 3] 关节世界坐标或相对 root（与你原先一致即可）
        返回:            调整后的 root_velocities（同形状）
        """
        if (not getattr(self, "prevent_floor_penetration", False)) or joints_pos is None:
            return root_velocities

        assert root_velocities.dim() == 3, "expect [B, T, D]"
        B, T, D = root_velocities.shape
        axis = 1  # y-up
        foot_start, foot_end = self._foot_joint_indices

        # 没有脚关节可用时，直接返回（避免 amin 空轴报错）
        if foot_end < foot_start or (joints_pos.size(2) == 0):
            return root_velocities

        device = root_velocities.device
        dtype = root_velocities.dtype
        F_floor = torch.as_tensor(self.floor_height, device=device, dtype=dtype)

        # 1) 取出该帧最低的脚 y 偏移/坐标，形状 [B, T]
        #    （与你原先语义一致：这里使用 foot_start..foot_end 的最小 y）
        foot_y = joints_pos[:, :, foot_start:foot_end + 1, axis]              # [B, T, F_foots]
        foot_min_y = foot_y.amin(dim=2)                                       # [B, T]

        # 2) 原始 y 方向速度序列 s，形状 [B, T]
        s = root_velocities[..., axis]                                        # [B, T]

        # 3) 原始累计和 S_naive_{t+1}（从 0 开始累计）
        S_naive_next = torch.cumsum(s, dim=1)                                 # [B, T]

        # 4) 每一步的累计下界 A_t = F - f_t（h_0 = 0）
        A = F_floor - foot_min_y                                              # [B, T]

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

    def _fuse_velocities_batched(
        self,
        root_vel_pred,     # [B, T, 3]  tran_b2 预测的根速度（local frame）
        contact_logits,    # [B, T, 2]  脚部接触概率（logits）
        joints_pos,        # [B, T, 24, 3]  FK 得到的关节位置
        root_rotation,     # [B, T, 3, 3]  根关节旋转矩阵（将 local->world）
        vel_scale: float = 3.0,
    ):
        """
        融合 tran_b1 和 tran_b2 的速度估计（批量版）
        返回:
            fused_vel: [B, T, 3]  融合后的速度（world frame）
        """
        import torch

        device = root_vel_pred.device
        B, T, _ = root_vel_pred.shape

        # ---- 1) tran_b2: 局部 -> 世界系 ----
        # [B,T,3,3] @ [B,T,3,1] -> [B,T,3]
        tran_b2_vel = torch.matmul(root_rotation, root_vel_pred.unsqueeze(-1)).squeeze(-1)
        tran_b2_vel = tran_b2_vel * vel_scale / self.fps  # 与单样本版一致

        # ---- 2) tran_b1: 基于脚部接触的速度估计 ----
        # 取左右脚踝位置（与单样本版索引一致：7=左脚踝, 8=右脚踝）
        lfoot_pos = joints_pos[:, :, 7, :]            # [B,T,3]
        rfoot_pos = joints_pos[:, :, 8, :]            # [B,T,3]

        # 位置差分（t-1 - t），首帧为零；与单样本版一致
        lfoot_vel = torch.zeros_like(lfoot_pos)       # [B,T,3]
        rfoot_vel = torch.zeros_like(rfoot_pos)
        if T > 1:
            lfoot_vel[:, 1:] = lfoot_pos[:, :-1] - lfoot_pos[:, 1:]
            rfoot_vel[:, 1:] = rfoot_pos[:, :-1] - rfoot_pos[:, 1:]

        # 根据接触最大类别选择左右脚速度：argmax==0 -> 左脚，否则右脚
        contact_idx  = contact_logits.argmax(dim=-1)          # [B,T]
        left_mask    = (contact_idx == 0).unsqueeze(-1)       # [B,T,1] -> broadcast 到 3
        tran_b1_vel  = torch.where(left_mask, lfoot_vel, rfoot_vel)  # [B,T,3]

        # 加上重力速度项（与单样本版一致）
        gravity = self.gravity_velocity.to(device).view(1, 1, 3)     # [1,1,3] 广播到 [B,T,3]
        tran_b1_vel = tran_b1_vel + gravity

        # ---- 3) 计算融合权重 ----
        contact_sigmoid = torch.sigmoid(contact_logits)        # [B,T,2]
        contact_max = contact_sigmoid.max(dim=-1).values       # [B,T]

        # 复用你现有的 _prob_to_weight（可能要求 1D），做个兼容
        try:
            weight = self._prob_to_weight(contact_max)         # 期望 [B,T]
        except Exception:
            weight = self._prob_to_weight(contact_max.reshape(-1)).view(B, T)
        weight = weight.unsqueeze(-1)                          # [B,T,1]

        # ---- 4) 融合：lerp(b2, b1, weight) ----
        # 避免外部依赖，直接 (1-w)*a + w*b
        fused_vel = (1.0 - weight) * tran_b2_vel + weight * tran_b1_vel
        fused_vel = fused_vel * self.fps                       # 与单样本版保持一致（单位 m/s）

        return fused_vel



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

        v_pred, p_pred, obj_v_pred, contact_pred, root_vel_local_pred, root_vel_pred, root_trans_pred = self.forward(
            xs, v_init_tensor, p_init_tensor, obj_imu_tensor, obj_v_init_tensor
        )

        glb_full_pose_list = []
        obj_v_seq_list = []
        contact_prob_list = []
        fused_trans_list = []

        for i in range(B):
            pose = p_pred[i].view(T, len(_REDUCED_POSE_NAMES), 6)
            orientation = xs[i, :, :, :9].view(T, 6, 3, 3)
            glb_full_pose = self._reduced_glb_6d_to_full_glb_mat(pose.cpu(), orientation.cpu())

            obj_v_seq = obj_v_pred[i].detach().cpu()
            contact_seq = torch.sigmoid(contact_pred[i]).detach().cpu()
            root_trans_seq = root_trans_pred[i].detach().cpu()

            glb_full_pose_list.append(glb_full_pose)
            obj_v_seq_list.append(obj_v_seq)
            contact_prob_list.append(contact_seq)
            fused_trans_list.append(root_trans_seq)

        if B == 1:
            return (
                glb_full_pose_list[0],
                obj_v_seq_list[0],
                contact_prob_list[0],
                fused_trans_list[0],
            )

        return (
            glb_full_pose_list,
            obj_v_seq_list,
            contact_prob_list,
            fused_trans_list,
        )
    @staticmethod
    def velocity_to_root_position(velocity, fps = 30):
        """
        将速度转换为根位置（参考TransPose）
        
        参数:
            velocity: [T, 3] 速度
        
        返回:
            position: [T, 3] 累积位置
        """
        velocity = velocity / fps
        if velocity.numel() == 0:
            return velocity.clone()
        return torch.cumsum(velocity, dim=0)


