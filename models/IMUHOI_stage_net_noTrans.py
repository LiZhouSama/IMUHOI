from scipy.spatial import transform
import torch
import torch.nn as nn
import torch.nn.functional as F

from human_body_prior.body_model.body_model import BodyModel
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d

from configs.global_config_IMUHOI import (
    FRAME_RATE,
    _SENSOR_NAMES,
    _SENSOR_VEL_NAMES,
    _REDUCED_POSE_NAMES,
    _REDUCED_INDICES,
    _IGNORED_INDICES,
    _SENSOR_ROT_INDICES,
    _SENSOR_POS_INDICES,
    _VEL_SELECTION_INDICES,
)


class RNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=False, dropout=0.2):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_rnn_layer = n_rnn_layer
        self.num_directions = 2 if bidirectional else 1
        self.rnn = nn.LSTM(
            input_size=n_hidden,
            hidden_size=n_hidden,
            num_layers=n_rnn_layer,
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
    def __init__(self, n_input, n_output, n_hidden, n_init, n_rnn_layer, bidirectional=False, dropout=0.2):
        super().__init__(n_input, n_output, n_hidden, n_rnn_layer, bidirectional, dropout)
        num_directions = 2 if bidirectional else 1
        self.init_net = nn.Sequential(
            nn.Linear(n_init, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden * n_rnn_layer),
            nn.ReLU(),
            nn.Linear(n_hidden * n_rnn_layer, 2 * num_directions * n_rnn_layer * n_hidden),
        )

    def forward(self, inputs, _=None):
        x, x_init = inputs
        batch_size = x.shape[0]
        num_directions = self.num_directions
        nd = self.n_rnn_layer * num_directions
        nh = self.n_hidden

        init = self.init_net(x_init).view(batch_size, 2, nd, nh)
        h0 = init[:, 0].permute(1, 0, 2).contiguous()
        c0 = init[:, 1].permute(1, 0, 2).contiguous()
        return super().forward(x, (h0, c0))


class SubPoser(nn.Module):
    def __init__(self, n_input, v_output, p_output, n_hidden, num_layer, dropout, extra_dim=0):
        super().__init__()
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
        if self.extra_dim:
            x_v = x[..., :-self.extra_dim]
        else:
            x_v = x
        v = self.rnn1((x_v, v_init))
        p_input = torch.cat((x, v), dim=-1)
        p = self.rnn2((p_input, p_init))
        return v, p


class VelocityContactModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_human_imus = getattr(cfg, "num_human_imus", len(_SENSOR_NAMES))
        self.imu_dim = getattr(cfg, "imu_dim", 9)
        self.obj_imu_dim = getattr(cfg, "obj_imu_dim", 9)
        hidden_dim = getattr(cfg, "velocity_hidden_dim", 128)
        num_layers = getattr(cfg, "velocity_num_layers", 2)
        dropout = getattr(cfg, "velocity_dropout", 0.2)
        self.obj_imu_acc_threshold = 0.01
        self.obj_imu_ori_threshold = 0.005
        self.velocity_suppression_strength = 0.9
        self.contact_suppression_strength = 0.95

        self.hand_vel_net = RNNWithInit(
            n_input=self.num_human_imus * self.imu_dim,
            n_output=6,
            n_hidden=hidden_dim,
            n_init=6,
            n_rnn_layer=num_layers,
            bidirectional=False,
            dropout=dropout,
        )

        self.obj_vel_net = RNNWithInit(
            n_input=self.obj_imu_dim,
            n_output=3,
            n_hidden=hidden_dim,
            n_init=3,
            n_rnn_layer=num_layers,
            bidirectional=False,
            dropout=dropout,
        )

        contact_input_dim = 2 * self.imu_dim + self.obj_imu_dim + 6 + 3
        contact_hidden = max(hidden_dim // 2, 32)
        self.contact_net = RNNWithInit(
            n_input=contact_input_dim,
            n_output=3,
            n_hidden=contact_hidden,
            n_init=6,
            n_rnn_layer=1,
            bidirectional=False,
            dropout=dropout,
        )

        self.left_hand_sensor = _SENSOR_NAMES.index("LeftForeArm")
        self.right_hand_sensor = _SENSOR_NAMES.index("RightForeArm")

    def detect_object_motion_from_imu(self, obj_imu):
        """
        基于物体IMU数据检测物体运动状态，参考preprocess.py中的逻辑
        
        Args:
            obj_imu: 物体IMU数据 [batch_size, seq_len, 9] 
                        格式通常为 [acc_x, acc_y, acc_z, ori_6d] (前3维加速度，后6维方向)
            
        Returns:
            motion_mask: 物体运动掩码 [batch_size, seq_len] (True表示有运动)
        """
        batch_size, seq_len, imu_dim = obj_imu.shape
        device = obj_imu.device
        
        # 初始化运动掩码，第一帧默认为False
        motion_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        if seq_len <= 1:
            return motion_mask
        
        # 提取加速度信息 (前3维)
        obj_acc = obj_imu[:, :, :3]  # [batch_size, seq_len, 3]
        
        # 提取方向信息 (后6维，6D旋转表示)
        obj_ori_6d = obj_imu[:, :, 3:]  # [batch_size, seq_len, 6]
        
        # 将6D旋转表示转换为旋转矩阵，以与preprocess.py保持一致
        obj_rot_mat = rotation_6d_to_matrix(obj_ori_6d.reshape(-1, 6)).reshape(batch_size, seq_len, 3, 3)
        
        # 计算相邻帧的加速度变化
        acc_diff = torch.norm(obj_acc[:, 1:] - obj_acc[:, :-1], dim=2)  # [batch_size, seq_len-1]
        
        # 计算相邻帧的旋转变化 (使用Frobenius范数，与preprocess.py保持一致)
        rot_diff = torch.norm(obj_rot_mat[:, 1:] - obj_rot_mat[:, :-1], dim=(2, 3))  # [batch_size, seq_len-1]
        
        # 使用配置的IMU检测阈值
        acc_threshold = self.obj_imu_acc_threshold  # IMU加速度变化阈值
        ori_threshold = self.obj_imu_ori_threshold  # IMU方向变化阈值 (对应旋转矩阵的Frobenius范数)
        
        # 判断运动：加速度或旋转变化超过阈值
        motion_detected = (acc_diff > acc_threshold) | (rot_diff > ori_threshold)
        
        # 填充运动掩码 (第一帧保持False)
        motion_mask[:, 1:] = motion_detected
        
        return motion_mask

    def forward(
        self,
        human_imu: torch.Tensor,
        obj_imu: torch.Tensor,
        hand_vel_init: torch.Tensor,
        obj_vel_init: torch.Tensor,
        contact_init: torch.Tensor = None,
    ):
        if human_imu.dim() != 4:
            raise ValueError(f"human_imu must be [B, T, num_imu, imu_dim], got {human_imu.shape}")
        batch_size, seq_len, _, _ = human_imu.shape
        device = human_imu.device
        dtype = human_imu.dtype

        if obj_imu is None:
            obj_imu = torch.zeros(batch_size, seq_len, self.obj_imu_dim, device=device, dtype=dtype)
        else:
            if obj_imu.dim() == 4:
                obj_imu = obj_imu.reshape(batch_size, seq_len, -1)
            elif obj_imu.dim() != 3:
                raise ValueError(f"obj_imu must be [B, T, obj_dim], got {obj_imu.shape}")
            if obj_imu.shape[-1] != self.obj_imu_dim:
                raise ValueError(f"obj_imu feature dim {obj_imu.shape[-1]} != expected {self.obj_imu_dim}")

        if hand_vel_init.dim() != 3 or hand_vel_init.shape[1:] != (2, 3):
            raise ValueError(f"hand_vel_init must be [B,2,3], got {hand_vel_init.shape}")
        hand_vel_init_vec = hand_vel_init.reshape(batch_size, -1)

        if obj_vel_init.dim() == 1:
            obj_vel_init_vec = obj_vel_init.unsqueeze(0).expand(batch_size, -1)
        elif obj_vel_init.dim() == 2:
            obj_vel_init_vec = obj_vel_init
        else:
            raise ValueError(f"obj_vel_init must be [B,3] or [3], got {obj_vel_init.shape}")

        # human_flat = human_imu.reshape(batch_size, seq_len, -1)
        # imu_root_ori_mat = rotation_6d_to_matrix(human_imu[:, :, 0, -6:])  # [bs, T, 3, 3]
        human_imu_acc = human_imu[:, :, :, :3]
        human_imu_ori = human_imu[:, :, :, 3:9]
        human_imu_ori_6d = human_imu_ori.reshape(-1, 6)
        human_imu_ori_mat = rotation_6d_to_matrix(human_imu_ori_6d).reshape(batch_size, seq_len, 6, 3, 3)
        R0T = human_imu_ori_mat[:, :, 0].transpose(-1, -2)      # [bs, T, 3, 3]
        acc_world = torch.matmul(human_imu_acc, R0T)            # [bs, T, 6, 3]
        acc0_world   = acc_world[:, :, :1, :]                   # [bs, T, 1, 3]
        acc_rest_mix = acc_world[:, :, 1:, :] + acc0_world      # [bs, T, 5, 3]
        human_imu_acc_denorm = torch.cat([acc0_world, acc_rest_mix], dim=2)  # [bs, T, 6, 3]
        human_imu_ori_denorm = torch.cat([human_imu_ori_mat[:, :, :1], human_imu_ori_mat[:, :, :1].matmul(human_imu_ori_mat[:,:,1:])], dim=2)
        human_imu_ori_denorm_6d = matrix_to_rotation_6d(human_imu_ori_denorm)
        human_imu_denorm = torch.cat([human_imu_acc_denorm, human_imu_ori_denorm_6d], dim=-1)
        human_imu_denorm_flat = human_imu_denorm.reshape(batch_size, seq_len, -1)

        hand_vel_flat = self.hand_vel_net((human_imu_denorm_flat, hand_vel_init_vec))  # [bs, T, 6]
        hand_vel = hand_vel_flat.view(batch_size, seq_len, 2, 3)  # [bs, T, 2, 3]
        # hand_vel = torch.matmul(
        #     imu_root_ori_mat.unsqueeze(2),  # [bs, T, 1, 3, 3]
        #     hand_vel.unsqueeze(-1),  # [bs, T, 2, 3, 1]
        # ).squeeze(-1)  # [bs, T, 2, 3]

        obj_vel = self.obj_vel_net((obj_imu, obj_vel_init_vec))

        hand_imu_feat = human_imu[:, :, [self.left_hand_sensor, self.right_hand_sensor], :].reshape(
            batch_size, seq_len, -1
        )

        contact_input = torch.cat(
            (
                hand_imu_feat,
                obj_imu,
                hand_vel.view(batch_size, seq_len, -1),
                obj_vel,
            ),
            dim=-1,
        )

        if contact_init is None:
            contact_init_vec = torch.cat((torch.zeros(batch_size, 3, device=device, dtype=dtype), obj_vel_init_vec), dim=-1)
        else:
            if contact_init.dim() == 1:
                contact_init_vec = contact_init.unsqueeze(0).expand(batch_size, -1)
            else:
                contact_init_vec = contact_init

        contact_logits = self.contact_net((contact_input, contact_init_vec))
        contact_prob = torch.sigmoid(contact_logits)

        # if obj_imu.numel() > 0:
        #     # 基于物体IMU数据检测物体运动状态
        #     motion_mask = self.detect_object_motion_from_imu(obj_imu)  # [batch_size, seq_len]
            
        #     # 1. 物体速度抑制：当物体没有运动时，大幅降低预测速度
        #     velocity_suppression_factor = torch.where(
        #         motion_mask.unsqueeze(-1),  # [batch_size, seq_len, 1]
        #         torch.ones_like(obj_vel),  # 有运动时不调整
        #         torch.full_like(obj_vel, 1.0 - self.velocity_suppression_strength)  # 无运动时降低速度
        #     )
            
        #     # 应用速度抑制因子
        #     obj_vel = obj_vel * velocity_suppression_factor
            
        #     # 2. 手部接触概率抑制：当物体没有运动时，大幅降低接触概率
        #     # motion_mask为False的位置表示物体没有运动
        #     contact_suppression_factor = torch.where(
        #         motion_mask.unsqueeze(-1),  # [batch_size, seq_len, 1]
        #         torch.ones_like(contact_prob),  # 有运动时不调整
        #         torch.full_like(contact_prob, 1.0 - self.contact_suppression_strength)  # 无运动时降低概率
        #     )
            
        #     # 应用接触概率抑制因子
        #     contact_prob = contact_prob * contact_suppression_factor

        return {
            "pred_hand_glb_vel": hand_vel,
            "pred_obj_vel": obj_vel,
            "pred_hand_contact_logits": contact_logits,
            "pred_hand_contact_prob": contact_prob,
        }


class HumanPoseModule(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        self.num_human_imus = getattr(cfg, "num_human_imus", len(_SENSOR_NAMES))
        self.imu_dim = getattr(cfg, "imu_dim", 9)
        n_hidden = getattr(cfg, "human_pose_hidden", 200)
        num_layer = getattr(cfg, "human_pose_layers", 2)
        dropout = getattr(cfg, "human_pose_dropout", 0.2)
        self.fps = float(getattr(cfg, "frame_rate", FRAME_RATE))
        self.sensor_names = list(_SENSOR_NAMES)
        self.v_names = list(_SENSOR_VEL_NAMES)
        self.p_names = list(_REDUCED_POSE_NAMES)
        n_glb = 6

        self.posers_config = [
            {
                "sensor": ["Root", "LeftLowerLeg", "RightLowerLeg", "Head"],
                "velocity": ["Root", "LeftFoot", "RightFoot", "Head"],
                "pose": ["LeftHip", "RightHip"],
            },
            {
                "sensor": ["Root", "Head"],
                "velocity": ["Root", "Head"],
                "pose": ["Spine1", "Spine2", "Spine3", "Neck"],
            },
            {
                "sensor": ["Root", "LeftForeArm", "RightForeArm"],
                "velocity": ["LeftHand", "RightHand"],
                "pose": ["LeftCollar", "RightCollar", "LeftShoulder", "RightShoulder"],
            },
        ]

        self.posers = nn.ModuleList()
        for config in self.posers_config:
            n_sensor = len(config["sensor"])
            n_input = n_sensor * self.imu_dim + n_glb
            v_output = len(config["velocity"]) * 3
            p_output = len(config["pose"]) * 6
            self.posers.append(
                SubPoser(
                    n_input=n_input,
                    v_output=v_output,
                    p_output=p_output,
                    n_hidden=n_hidden,
                    num_layer=num_layer,
                    dropout=dropout,
                    extra_dim=n_glb,
                )
            )

        human_feature_dim = self.num_human_imus * self.imu_dim
        self.glb = RNN(
            n_input=human_feature_dim,
            n_output=n_glb,
            n_hidden=36,
            n_rnn_layer=1,
            dropout=dropout,
        )
        self.smpl_parents = torch.tensor(
            [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
            dtype=torch.long,
        )
        self.body_model = None
        self.body_model_device = None
        body_model_path = getattr(cfg, "body_model_path", None)
        if body_model_path is None:
            raise ValueError("body_model_path is not set")
        try:
            self.body_model = BodyModel(bm_fname=body_model_path, num_betas=16)
            self.body_model.eval()
            for param in self.body_model.parameters():
                param.requires_grad_(False)
            self.body_model_device = torch.device("cpu")
        except Exception as exc:
            print(f"加载Body Model失败: {exc}")
            self.body_model = None

        if isinstance(_VEL_SELECTION_INDICES, torch.Tensor):
            self.vel_indices = _VEL_SELECTION_INDICES.tolist()
        else:
            self.vel_indices = list(_VEL_SELECTION_INDICES)

        self._generate_indices_list()
        self.hand_joint_indices = (20, 21)

    def _find_indices(self, names, pool):
        return [pool.index(name) for name in names]

    def _generate_indices_list(self):
        self.indices = []
        for config in self.posers_config:
            self.indices.append(
                {
                    "sensor_indices": self._find_indices(config["sensor"], self.sensor_names),
                    "v_indices": self._find_indices(config["velocity"], self.v_names),
                    "p_indices": self._find_indices(config["pose"], self.p_names),
                }
            )

    def _prob_to_weight(self, p):
        p_clamped = p.clamp(self.prob_threshold[0], self.prob_threshold[1])
        return (p_clamped - self.prob_threshold[0]) / (self.prob_threshold[1] - self.prob_threshold[0] + 1e-8)

    def _global2local(self, global_rotmats, parents):
        batch_size, num_joints, _, _ = global_rotmats.shape
        local_rotmats = torch.zeros_like(global_rotmats)
        local_rotmats[:, 0] = global_rotmats[:, 0]
        for i in range(1, num_joints):
            parent_idx = parents[i]
            R_parent = global_rotmats[:, parent_idx]
            R_parent_inv = R_parent.transpose(-1, -2)
            local_rotmats[:, i] = torch.matmul(R_parent_inv, global_rotmats[:, i])
        local_rotmats[:, _IGNORED_INDICES] = torch.eye(3, device=global_rotmats.device, dtype=global_rotmats.dtype).view(1, 1, 3, 3).repeat(
            batch_size, len(_IGNORED_INDICES), 1, 1
        )
        return local_rotmats

    def _reduced_glb_6d_to_full_glb_mat(self, glb_reduced_pose, orientation):
        root_rotation = orientation[:, 0]
        reduced_rot = rotation_6d_to_matrix(glb_reduced_pose.reshape(-1, 6)).reshape(
            glb_reduced_pose.shape[0], len(_REDUCED_POSE_NAMES), 3, 3
        )
        reduced_rot_global = torch.matmul(root_rotation.unsqueeze(1), reduced_rot)
        orientation_global = orientation.clone()
        orientation_global[:, 1:] = torch.matmul(root_rotation.unsqueeze(1), orientation[:, 1:])
        dtype = glb_reduced_pose.dtype
        device = glb_reduced_pose.device
        full_pose = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).repeat(
            glb_reduced_pose.shape[0], 24, 1, 1
        )
        full_pose[:, _REDUCED_INDICES] = reduced_rot_global
        full_pose[:, _SENSOR_ROT_INDICES] = orientation_global
        ignored_parents = self.smpl_parents[_IGNORED_INDICES]
        full_pose[:, _IGNORED_INDICES] = full_pose[:, ignored_parents]
        return full_pose

    def _compute_fk_joints_batched(self, glb_p_out_tensor: torch.Tensor, orientation: torch.Tensor):
        if self.body_model is None:
            return None

        batch_size, seq_len, _ = glb_p_out_tensor.shape
        device = glb_p_out_tensor.device
        BT = batch_size * seq_len

        glb_pose = glb_p_out_tensor.reshape(BT, len(_REDUCED_POSE_NAMES), 6)
        orientation = orientation[:, :, : len(_SENSOR_ROT_INDICES)].reshape(BT, len(_SENSOR_ROT_INDICES), 3, 3)
        full_glb = self._reduced_glb_6d_to_full_glb_mat(glb_pose, orientation)
        local_pose = self._global2local(full_glb, self.smpl_parents.tolist())
        pose_aa = matrix_to_axis_angle(local_pose.reshape(-1, 3, 3)).reshape(BT, 24, 3)

        try:
            with torch.no_grad():
                body_out = self.body_model(
                    pose_body=pose_aa[:, 1:22].reshape(BT, 63),
                    root_orient=pose_aa[:, 0].reshape(BT, 3),
                )
            joints = body_out.Jtr[:, :24, :]
            return joints.reshape(batch_size, seq_len, 24, 3)
        except Exception as exc:
            print(f"FK计算失败: {exc}")
            return None

    def forward(self, human_imu: torch.Tensor, v_init: torch.Tensor, p_init: torch.Tensor, trans_gt: torch.Tensor = None):
        if human_imu.dim() != 4:
            raise ValueError(f"human_imu must be [B, T, num_imu, imu_dim], got {human_imu.shape}")
        batch_size, seq_len, _, _ = human_imu.shape
        device = human_imu.device

        if self.body_model is not None and (self.body_model_device != device):
            self.body_model = self.body_model.to(device)
            self.body_model_device = device

        human_flat = human_imu.reshape(batch_size, seq_len, -1)
        s_glb = self.glb(human_flat)

        v_components = []
        p_components = []
        v_lower = None

        for poser_idx, poser in enumerate(self.posers):
            indices = self.indices[poser_idx]
            sensor_feat = human_imu[:, :, indices["sensor_indices"], :].reshape(batch_size, seq_len, -1)
            poser_input = torch.cat((sensor_feat, s_glb), dim=-1)
            v_init_sub = v_init[:, indices["v_indices"], :].reshape(batch_size, -1)
            p_init_sub = p_init[:, indices["p_indices"], :].reshape(batch_size, -1)

            v_i, p_i = poser(poser_input, v_init_sub, p_init_sub)
            v_components.append(v_i)
            p_components.append(p_i)

            if poser_idx == 0:
                v_lower = v_i

        v_pred = torch.cat(v_components, dim=-1)
        p_pred = torch.cat(p_components, dim=-1)

        orientation_6d = human_imu[..., -6:]
        orientation_mat = rotation_6d_to_matrix(orientation_6d.reshape(-1, 6)).reshape(
            batch_size, seq_len, self.num_human_imus, 3, 3
        )
        root_R = orientation_mat[:, :, 0]

        joints_pos = None
        if self.body_model is not None:
            joints_pos = self._compute_fk_joints_batched(p_pred, orientation_mat.clone())

        if trans_gt is None:
            trans_gt = torch.zeros(batch_size, seq_len, 3, device=device, dtype=human_imu.dtype)
        else:
            trans_gt = trans_gt.to(device=device, dtype=human_imu.dtype)
            if trans_gt.dim() == 2:
                trans_gt = trans_gt.unsqueeze(1).expand(batch_size, seq_len, 3)
            elif trans_gt.dim() == 3 and (trans_gt.shape[0] != batch_size or trans_gt.shape[1] != seq_len):
                trans_gt = trans_gt.reshape(batch_size, seq_len, 3)
            elif trans_gt.dim() != 3:
                trans_gt = trans_gt.view(batch_size, seq_len, 3)

        if joints_pos is not None:
            lhand = joints_pos[:, :, self.hand_joint_indices[0], :] + trans_gt
            rhand = joints_pos[:, :, self.hand_joint_indices[1], :] + trans_gt
            pred_hand_glb_pos = torch.stack((lhand, rhand), dim=2)
        else:
            pred_hand_glb_pos = torch.zeros(batch_size, seq_len, 2, 3, device=device, dtype=human_imu.dtype)

        return {
            "v_pred": v_pred,
            "p_pred": p_pred,
            "pred_hand_glb_pos": pred_hand_glb_pos,
        }


class ObjectTransModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.imu_dim = getattr(cfg, "imu_dim", 9)
        self.num_human_imus = getattr(cfg, "num_human_imus", len(_SENSOR_NAMES))
        hidden_dim_multiplier = getattr(cfg, "hidden_dim_multiplier", 1)

        self.gating_prior_beta = getattr(cfg, "gating_prior_beta", 5.0)
        self.gating_temperature = getattr(cfg, "gating_temperature", 5.0)
        self.gating_smoothing_enabled = getattr(cfg, "gating_smoothing_enabled", False)
        self.gating_smoothing_alpha = getattr(cfg, "gating_smoothing_alpha", 0.6)
        self.gating_max_change = getattr(cfg, "gating_max_change", 0.25)

        n_fk_branch_input = 34
        n_gating_input = 9

        self.lhand_fk_head = RNNWithInit(
            n_input=n_fk_branch_input,
            n_output=4,
            n_hidden=128 * hidden_dim_multiplier,
            n_init=4,
            n_rnn_layer=2,
            bidirectional=False,
            dropout=0.2,
        )
        self.rhand_fk_head = RNNWithInit(
            n_input=n_fk_branch_input,
            n_output=4,
            n_hidden=128 * hidden_dim_multiplier,
            n_init=4,
            n_rnn_layer=2,
            bidirectional=False,
            dropout=0.2,
        )
        self.gating_head = RNNWithInit(
            n_input=n_gating_input,
            n_output=3,
            n_hidden=64 * hidden_dim_multiplier,
            n_init=6,
            n_rnn_layer=1,
            bidirectional=False,
            dropout=0.2,
        )

    @staticmethod
    def _unit_vector(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
        return x / norm

    @staticmethod
    def _softplus_positive(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) + 1e-4

    def _smooth_gating_weights(self, weights: torch.Tensor) -> torch.Tensor:
        if self.training or (not self.gating_smoothing_enabled) or weights.size(1) < 2:
            return weights

        LHAND_FK = 0
        RHAND_FK = 1
        IMU_BRANCH = 2

        smoothed_weights = weights.clone()
        prev_smoothed = weights[:, 0, :]

        for t in range(1, weights.size(1)):
            current_weights = weights[:, t, :]
            prev_dominant = prev_smoothed.argmax(dim=-1)
            curr_dominant = current_weights.argmax(dim=-1)
            frame_weights = current_weights.clone()

            transition_mask = prev_dominant != curr_dominant
            if transition_mask.any():
                for b in range(weights.size(0)):
                    if not transition_mask[b]:
                        continue
                    prev_dom = prev_dominant[b].item()
                    curr_dom = curr_dominant[b].item()
                    need_smoothing = False
                    if (prev_dom == IMU_BRANCH and curr_dom in [LHAND_FK, RHAND_FK]) or (
                        prev_dom in [LHAND_FK, RHAND_FK] and curr_dom in [LHAND_FK, RHAND_FK]
                    ):
                        need_smoothing = True
                    if need_smoothing:
                        frame_weights[b, :] = (
                            self.gating_smoothing_alpha * prev_smoothed[b, :]
                            + (1.0 - self.gating_smoothing_alpha) * current_weights[b, :]
                        )
                        if self.gating_max_change > 0:
                            weight_change = frame_weights[b, :] - prev_smoothed[b, :]
                            change_norm = torch.norm(weight_change)
                            if change_norm > self.gating_max_change:
                                weight_change = weight_change * (
                                    self.gating_max_change / (change_norm + 1e-8)
                                )
                                frame_weights[b, :] = prev_smoothed[b, :] + weight_change
                        frame_weights[b, :] = F.softmax(
                            torch.log(frame_weights[b, :] + 1e-8) * self.gating_temperature, dim=-1
                        )
            smoothed_weights[:, t, :] = frame_weights
            prev_smoothed = frame_weights

        return smoothed_weights

    def _build_fk_inputs(
        self,
        obj_rot6d,
        hand_pos,
        hand_contact_scalar,
        obj_imu9,
        hand_imu9,
        obj_vel3,
        obj_rot_delta3,
    ):
        return torch.cat(
            [obj_rot6d, hand_pos, hand_contact_scalar, obj_imu9, hand_imu9, obj_vel3, obj_rot_delta3],
            dim=2,
        )

    def _build_gating_inputs(self, contact_prob3, obj_vel3, obj_imu_acc3):
        return torch.cat([contact_prob3, obj_vel3, obj_imu_acc3], dim=2)

    def _rot6d_delta(self, rot6d: torch.Tensor) -> torch.Tensor:
        B, T, _ = rot6d.shape
        R = rotation_6d_to_matrix(rot6d.reshape(-1, 6)).reshape(B, T, 3, 3)
        rel = torch.matmul(R[:, 1:].transpose(-1, -2), R[:, :-1])  # [B,T-1,3,3]
        aa = matrix_to_axis_angle(rel.reshape(-1, 3, 3)).reshape(B, T-1, 3)
        aa = F.pad(aa, (0, 0, 1, 0))  # 前面补一帧零 → [B,T,3]
        return aa

    def _compute_init_dir_len(self, hand_pos_0, obj_rotm_0, obj_pos_0):
        vec_world = obj_pos_0 - hand_pos_0
        lb0 = vec_world.norm(dim=-1, keepdim=True)
        unit_world = self._unit_vector(vec_world)
        obj_Rt = obj_rotm_0.transpose(-1, -2)
        oe0 = torch.bmm(obj_Rt, unit_world.unsqueeze(-1)).squeeze(-1)
        return oe0, lb0

    def pred_obj_pos_fk(
        self,
        pred_hand_contact_prob: torch.Tensor,
        pred_hand_positions: torch.Tensor,
        obj_rotm: torch.Tensor,
        obj_trans_init: torch.Tensor,
    ):
        """
        使用接触预测与手部位置做FK，估计物体位置。
        Args:
            pred_hand_contact_prob: [B, T, 3]
            pred_hand_positions: [B, T, 2, 3]
            obj_rotm: [B, T, 3, 3] 由物体IMU旋转获得
            obj_trans_init: [B, 3] 初始帧的物体位置
        Returns:
            computed_obj_trans: [B, T, 3]
            fk_info: dict 包含方向与长度
        """
        batch_size, seq_len, _ = pred_hand_contact_prob.shape
        device = pred_hand_contact_prob.device
        dtype = pred_hand_positions.dtype

        computed_obj_trans = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        computed_obj_trans[:, 0] = obj_trans_init

        fk_lhand_bone_length = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
        fk_rhand_bone_length = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
        fk_lhand_direction = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        fk_rhand_direction = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)

        contact_threshold = 0.5

        for b in range(batch_size):
            lhand_contact_prob = pred_hand_contact_prob[b, :, 0]
            rhand_contact_prob = pred_hand_contact_prob[b, :, 1]
            lhand_pos = pred_hand_positions[b, :, 0, :]
            rhand_pos = pred_hand_positions[b, :, 1, :]
            obj_rot_mat = obj_rotm[b]
            init_obj_pos = obj_trans_init[b]

            lhand_contact = (lhand_contact_prob > contact_threshold).float()
            rhand_contact = (rhand_contact_prob > contact_threshold).float()
            if seq_len > 0:
                lhand_contact[0] = 0
                rhand_contact[0] = 0

            lhand_start_contact = torch.zeros_like(lhand_contact)
            rhand_start_contact = torch.zeros_like(rhand_contact)
            for t in range(1, seq_len):
                if lhand_contact[t] > 0 and lhand_contact[t - 1] == 0:
                    lhand_start_contact[t] = 1
                if rhand_contact[t] > 0 and rhand_contact[t - 1] == 0:
                    rhand_start_contact[t] = 1

            contact_segments = []
            current_contact = None
            segment_start = None

            for t in range(seq_len):
                new_contact = None
                if lhand_start_contact[t] > 0:
                    new_contact = "left"
                elif rhand_start_contact[t] > 0:
                    new_contact = "right"

                has_contact = False
                if current_contact == "left":
                    has_contact = lhand_contact[t] > 0
                elif current_contact == "right":
                    has_contact = rhand_contact[t] > 0

                has_contact_another = False
                if current_contact == "left":
                    has_contact_another = rhand_contact[t] > 0
                elif current_contact == "right":
                    has_contact_another = lhand_contact[t] > 0

                if new_contact is not None:
                    if current_contact is not None and segment_start is not None:
                        contact_segments.append({"hand": current_contact, "start": segment_start, "end": t - 1})
                    current_contact = new_contact
                    segment_start = t
                elif not has_contact and has_contact_another:
                    if current_contact is not None and segment_start is not None:
                        contact_segments.append({"hand": current_contact, "start": segment_start, "end": t - 1})
                    current_contact = "left" if current_contact == "right" else "right"
                    segment_start = t
                elif not has_contact and current_contact is not None:
                    contact_segments.append({"hand": current_contact, "start": segment_start, "end": t - 1})
                    current_contact = None
                    segment_start = None

            if current_contact is not None and segment_start is not None:
                contact_segments.append({"hand": current_contact, "start": segment_start, "end": seq_len - 1})

            contact_segments.sort(key=lambda x: x["start"])

            current_obj_position = init_obj_pos.clone()

            if contact_segments:
                first_seg_start = contact_segments[0]["start"]
                computed_obj_trans[b, : first_seg_start + 1] = current_obj_position
            else:
                computed_obj_trans[b] = current_obj_position

            for segment_idx, segment in enumerate(contact_segments):
                hand_type = segment["hand"]
                start_frame = segment["start"]
                end_frame = segment["end"]

                hand_pos_segment = (
                    lhand_pos[start_frame : end_frame + 1] if hand_type == "left" else rhand_pos[start_frame : end_frame + 1]
                )

                initial_hand_pos = hand_pos_segment[0]
                initial_obj_rot_mat = obj_rot_mat[start_frame]

                if segment_idx == 0:
                    initial_obj_pos = init_obj_pos
                else:
                    initial_obj_pos = current_obj_position

                hand_to_obj_world = initial_obj_pos - initial_hand_pos
                initial_distance = torch.norm(hand_to_obj_world)
                if initial_distance > 1e-6:
                    hand_to_obj_unit = hand_to_obj_world / initial_distance
                else:
                    hand_to_obj_unit = torch.tensor(
                        [0.0, 0.0, 1.0], device=device, dtype=dtype
                    )
                    initial_distance = torch.tensor(0.1, device=device, dtype=dtype)

                obj_direction_initial = initial_obj_rot_mat.transpose(0, 1) @ hand_to_obj_unit

                if hand_type == "left":
                    fk_lhand_bone_length[b, start_frame : end_frame + 1] = initial_distance
                    fk_lhand_direction[b, start_frame : end_frame + 1] = obj_direction_initial.unsqueeze(0).repeat(
                        end_frame - start_frame + 1, 1
                    )
                else:
                    fk_rhand_bone_length[b, start_frame : end_frame + 1] = initial_distance
                    fk_rhand_direction[b, start_frame : end_frame + 1] = obj_direction_initial.unsqueeze(0).repeat(
                        end_frame - start_frame + 1, 1
                    )

                for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
                    current_hand_pos = hand_pos_segment[i]
                    current_obj_rot_mat = obj_rot_mat[frame_idx]
                    direction_world = current_obj_rot_mat @ obj_direction_initial
                    predicted_obj_pos = current_hand_pos + direction_world * initial_distance
                    computed_obj_trans[b, frame_idx] = predicted_obj_pos

                current_obj_position = computed_obj_trans[b, end_frame].clone()

                if segment_idx > 0:
                    prev_segment = contact_segments[segment_idx - 1]
                    prev_end_frame = prev_segment["end"]
                    prev_last_position = computed_obj_trans[b, prev_end_frame].clone()
                    for gap_frame in range(prev_end_frame + 1, start_frame):
                        computed_obj_trans[b, gap_frame] = prev_last_position

            if contact_segments:
                last_segment = contact_segments[-1]
                last_end_frame = last_segment["end"]
                if current_obj_position is not None:
                    for frame_idx in range(last_end_frame + 1, seq_len):
                        computed_obj_trans[b, frame_idx] = current_obj_position
            else:
                computed_obj_trans[b] = current_obj_position

        fk_info = {
            "fk_lhand_bone_length": fk_lhand_bone_length,
            "fk_rhand_bone_length": fk_rhand_bone_length,
            "fk_lhand_direction": fk_lhand_direction,
            "fk_rhand_direction": fk_rhand_direction,
        }

        return computed_obj_trans, fk_info

    def forward(
        self,
        hand_positions: torch.Tensor,
        pred_hand_contact_prob: torch.Tensor,
        obj_trans_init: torch.Tensor,
        obj_imu: torch.Tensor = None,
        human_imu: torch.Tensor = None,
        obj_vel_input: torch.Tensor = None,
        contact_init: torch.Tensor = None,
        has_object_mask: torch.Tensor = None,
        compute_fk: bool = False,
    ):
        if hand_positions is None:
            raise ValueError("hand_positions cannot be None")
        if hand_positions.dim() == 3:
            bs, seq_len, _ = hand_positions.shape
            hand_positions = hand_positions.view(bs, seq_len, 2, 3)
        elif hand_positions.dim() == 4:
            bs, seq_len = hand_positions.shape[:2]
        else:
            raise ValueError(f"Unexpected hand_positions shape {hand_positions.shape}")

        device = hand_positions.device
        dtype = hand_positions.dtype
        lhand_position = hand_positions[:, :, 0, :]
        rhand_position = hand_positions[:, :, 1, :]

        if obj_imu is None:
            obj_imu = torch.zeros(bs, seq_len, self.imu_dim, device=device, dtype=dtype)
        else:
            if obj_imu.dim() == 4:
                obj_imu = obj_imu.reshape(bs, seq_len, -1)
            if obj_imu.shape[-1] != self.imu_dim:
                raise ValueError(f"obj_imu feature dim {obj_imu.shape[-1]} != expected {self.imu_dim}")

        if human_imu is None:
            human_imu = torch.zeros(bs, seq_len, self.num_human_imus * self.imu_dim, device=device, dtype=dtype)
        if human_imu.dim() == 3 and human_imu.shape[-1] == self.num_human_imus * self.imu_dim:
            human_imu = human_imu.view(bs, seq_len, self.num_human_imus, self.imu_dim)
        elif human_imu.dim() == 4:
            pass
        else:
            raise ValueError(f"Unexpected human_imu shape {human_imu.shape}")

        obj_rot = obj_imu[:, :, 3:9]
        obj_rot_delta = self._rot6d_delta(obj_rot)
        obj_rotm = rotation_6d_to_matrix(obj_rot.reshape(-1, 6)).reshape(bs, seq_len, 3, 3)
        obj_imu_acc = obj_imu[:, :, :3]

        l_idx = _SENSOR_NAMES.index("LeftForeArm")
        r_idx = _SENSOR_NAMES.index("RightForeArm")
        lhand_imu9 = human_imu[:, :, l_idx, :]
        rhand_imu9 = human_imu[:, :, r_idx, :]

        if obj_vel_input is None:
            obj_vel_input = torch.zeros(bs, seq_len, 3, device=device, dtype=dtype)

        pL = pred_hand_contact_prob[:, :, 0:1]
        pR = pred_hand_contact_prob[:, :, 1:2]

        fk_l_input = self._build_fk_inputs(obj_rot, lhand_position, pL, obj_imu, lhand_imu9, obj_vel_input, obj_rot_delta)
        fk_r_input = self._build_fk_inputs(obj_rot, rhand_position, pR, obj_imu, rhand_imu9, obj_vel_input, obj_rot_delta)

        obj_pos_0 = obj_trans_init
        obj_R_0 = obj_rotm[:, 0, :, :]
        l_hand_0 = lhand_position[:, 0, :]
        r_hand_0 = rhand_position[:, 0, :]
        l_oe0, l_lb0 = self._compute_init_dir_len(l_hand_0, obj_R_0, obj_pos_0)
        r_oe0, r_lb0 = self._compute_init_dir_len(r_hand_0, obj_R_0, obj_pos_0)

        l_init_vec = torch.cat((l_oe0, l_lb0), dim=-1)
        r_init_vec = torch.cat((r_oe0, r_lb0), dim=-1)

        if contact_init is None:
            contact_init_vec = torch.cat((torch.zeros(bs, 3, device=device, dtype=dtype), obj_vel_input[:, 0, :]), dim=-1)
        else:
            if contact_init.dim() == 1:
                contact_init_vec = contact_init.unsqueeze(0).expand(bs, -1)
            else:
                contact_init_vec = contact_init

        l_fk_out = self.lhand_fk_head((fk_l_input, l_init_vec))
        r_fk_out = self.rhand_fk_head((fk_r_input, r_init_vec))
        l_dir = self._unit_vector(l_fk_out[:, :, :3])
        r_dir = self._unit_vector(r_fk_out[:, :, :3])
        l_len = self._softplus_positive(l_fk_out[:, :, 3])
        r_len = self._softplus_positive(r_fk_out[:, :, 3])

        obj_rotm_flat = obj_rotm.reshape(bs * seq_len, 3, 3)
        l_dir_world = torch.bmm(obj_rotm_flat, l_dir.reshape(bs * seq_len, 3, 1)).reshape(bs, seq_len, 3)
        r_dir_world = torch.bmm(obj_rotm_flat, r_dir.reshape(bs * seq_len, 3, 1)).reshape(bs, seq_len, 3)
        l_pos_fk = lhand_position + l_dir_world * l_len.unsqueeze(-1)
        r_pos_fk = rhand_position + r_dir_world * r_len.unsqueeze(-1)

        gating_input = self._build_gating_inputs(pred_hand_contact_prob, obj_vel_input, obj_imu_acc)
        gate_logits = self.gating_head((gating_input, contact_init_vec))
        prior_im = 1.0 - torch.maximum(pL.squeeze(-1), pR.squeeze(-1))
        prior = torch.stack([pL.squeeze(-1), pR.squeeze(-1), prior_im], dim=-1)
        gate_logits = gate_logits + self.gating_prior_beta * torch.log(prior + 1e-6)
        weights_raw = F.softmax(gate_logits / self.gating_temperature, dim=-1)
        weights = self._smooth_gating_weights(weights_raw)

        fused_pos = torch.zeros(bs, seq_len, 3, device=device, dtype=dtype)
        dt = 1.0 / FRAME_RATE
        for t in range(seq_len):
            prev_pos = fused_pos[:, t - 1, :] if t > 0 else obj_trans_init
            pos_imu_integrated = prev_pos + obj_vel_input[:, t, :] * dt
            fused_pos[:, t, :] = (
                weights[:, t, 0:1] * l_pos_fk[:, t, :]
                + weights[:, t, 1:2] * r_pos_fk[:, t, :]
                + weights[:, t, 2:3] * pos_imu_integrated
            )

        vel_from_pos = torch.zeros_like(fused_pos)
        acc_from_pos = torch.zeros_like(fused_pos)
        if seq_len > 1:
            vel_from_pos[:, 1:] = (fused_pos[:, 1:] - fused_pos[:, :-1]) * FRAME_RATE
        if seq_len > 2:
            acc_from_pos[:, 2:] = (
                (fused_pos[:, 2:] - 2 * fused_pos[:, 1:-1] + fused_pos[:, :-2]) * (FRAME_RATE**2)
            )

        fk_obj_pos = None
        fk_info = None
        if compute_fk:
            fk_obj_pos, fk_info = self.pred_obj_pos_fk(
                pred_hand_contact_prob,
                hand_positions,
                obj_rotm,
                obj_trans_init,
            )

        if has_object_mask is not None:
            if has_object_mask.dim() > 1:
                has_object_mask = has_object_mask.view(bs)
            mask = has_object_mask.to(dtype=dtype, device=device).view(bs, 1, 1)
            fused_pos = fused_pos * mask
            vel_from_pos = vel_from_pos * mask
            acc_from_pos = acc_from_pos * mask
            weights = weights * mask
            weights_raw = weights_raw * mask
            l_pos_fk = l_pos_fk * mask
            r_pos_fk = r_pos_fk * mask
            l_dir = l_dir * mask
            r_dir = r_dir * mask
            l_len = l_len * mask.squeeze(-1)
            r_len = r_len * mask.squeeze(-1)
            l_oe0 = l_oe0 * mask.squeeze(-1)
            r_oe0 = r_oe0 * mask.squeeze(-1)
            l_lb0 = (l_lb0 * mask.squeeze(-1).unsqueeze(-1)).squeeze(-1)
            r_lb0 = (r_lb0 * mask.squeeze(-1).unsqueeze(-1)).squeeze(-1)
            if fk_obj_pos is not None:
                fk_obj_pos = fk_obj_pos * mask
                mask_len = has_object_mask.to(dtype=dtype, device=device).view(bs, 1)
                fk_info["fk_lhand_bone_length"] = fk_info["fk_lhand_bone_length"] * mask_len
                fk_info["fk_rhand_bone_length"] = fk_info["fk_rhand_bone_length"] * mask_len
                fk_info["fk_lhand_direction"] = fk_info["fk_lhand_direction"] * mask
                fk_info["fk_rhand_direction"] = fk_info["fk_rhand_direction"] * mask
        else:
            l_lb0 = l_lb0.squeeze(-1)
            r_lb0 = r_lb0.squeeze(-1)

        result = {
            "pred_obj_trans": fused_pos,
            "gating_weights": weights,
            "gating_weights_raw": weights_raw,
            "pred_obj_vel_from_posdiff": vel_from_pos,
            "pred_obj_acc_from_posdiff": acc_from_pos,
            "obj_vel_input": obj_vel_input,
            "pred_lhand_obj_direction": l_dir,
            "pred_rhand_obj_direction": r_dir,
            "pred_lhand_lb": l_len,
            "pred_rhand_lb": r_len,
            "pred_lhand_obj_trans": l_pos_fk,
            "pred_rhand_obj_trans": r_pos_fk,
            "init_lhand_oe_ho": l_oe0,
            "init_rhand_oe_ho": r_oe0,
            "init_lhand_lb": l_lb0,
            "init_rhand_lb": r_lb0,
            "gating_smoothing_applied": (not self.training) and self.gating_smoothing_enabled,
        }

        if fk_obj_pos is not None and fk_info is not None:
            result.update(
                {
                    "pred_obj_trans_fk": fk_obj_pos,
                    "fk_lhand_bone_length": fk_info["fk_lhand_bone_length"],
                    "fk_rhand_bone_length": fk_info["fk_rhand_bone_length"],
                    "fk_lhand_direction": fk_info["fk_lhand_direction"],
                    "fk_rhand_direction": fk_info["fk_rhand_direction"],
                }
            )
        return result

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device):
        zeros_pos = torch.zeros(batch_size, seq_len, 3, device=device)
        zeros_dir = torch.zeros(batch_size, seq_len, 3, device=device)
        zeros_scalar = torch.zeros(batch_size, seq_len, device=device)
        zeros_weights = torch.zeros(batch_size, seq_len, 3, device=device)
        return {
            "pred_obj_trans": zeros_pos,
            "gating_weights": zeros_weights,
            "gating_weights_raw": zeros_weights,
            "pred_obj_vel_from_posdiff": zeros_pos,
            "pred_obj_acc_from_posdiff": zeros_pos,
            "obj_vel_input": zeros_pos,
            "pred_lhand_obj_direction": zeros_dir,
            "pred_rhand_obj_direction": zeros_dir,
            "pred_lhand_lb": zeros_scalar,
            "pred_rhand_lb": zeros_scalar,
            "pred_lhand_obj_trans": zeros_pos,
            "pred_rhand_obj_trans": zeros_pos,
            "init_lhand_oe_ho": torch.zeros(batch_size, 3, device=device),
            "init_rhand_oe_ho": torch.zeros(batch_size, 3, device=device),
            "init_lhand_lb": torch.zeros(batch_size, device=device),
            "init_rhand_lb": torch.zeros(batch_size, device=device),
            "gating_smoothing_applied": False,
        }


class TransPoseNet(nn.Module):
    def __init__(self, cfg, pretrained_modules=None, skip_modules=None):
        super().__init__()
        self.cfg = cfg
        self.num_human_imus = getattr(cfg, "num_human_imus", len(_SENSOR_NAMES))
        self.imu_dim = getattr(cfg, "imu_dim", 9)

        if hasattr(cfg, "device"):
            self.device = torch.device(cfg.device)
        elif hasattr(cfg, "gpus") and cfg.gpus:
            self.device = torch.device(f"cuda:{cfg.gpus[0]}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if pretrained_modules is None:
            pretrained_modules = {}
        if skip_modules is None:
            skip_modules = []

        self._initialize_modules(cfg, pretrained_modules, skip_modules)

    def _initialize_modules(self, cfg, pretrained_modules, skip_modules):
        if "velocity_contact" in skip_modules:
            self.velocity_contact_module = None
        elif "velocity_contact" in pretrained_modules:
            self.velocity_contact_module = self._load_single_module(
                pretrained_modules["velocity_contact"], "velocity_contact", cfg
            )
        else:
            self.velocity_contact_module = VelocityContactModule(cfg)

        if "human_pose" in skip_modules:
            self.human_pose_module = None
        elif "human_pose" in pretrained_modules:
            self.human_pose_module = self._load_single_module(
                pretrained_modules["human_pose"], "human_pose", cfg
            )
        else:
            self.human_pose_module = HumanPoseModule(cfg, self.device)

        if "object_trans" in skip_modules:
            self.object_trans_module = None
        elif "object_trans" in pretrained_modules:
            self.object_trans_module = self._load_single_module(
                pretrained_modules["object_trans"], "object_trans", cfg
            )
        else:
            self.object_trans_module = ObjectTransModule(cfg)

    def _load_single_module(self, checkpoint_path, module_name, cfg):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if module_name == "velocity_contact":
                module = VelocityContactModule(cfg)
            elif module_name == "human_pose":
                module = HumanPoseModule(cfg, self.device)
            elif module_name == "object_trans":
                module = ObjectTransModule(cfg)
            else:
                raise ValueError(f"未知模块名 {module_name}")

            state_dict = checkpoint.get("module_state_dict", checkpoint.get("state_dict", checkpoint))
            module.load_state_dict(state_dict)
            return module
        except Exception as exc:
            print(f"加载{module_name}模块失败: {exc}")
            if module_name == "velocity_contact":
                return VelocityContactModule(cfg)
            if module_name == "human_pose":
                return HumanPoseModule(cfg, self.device)
            if module_name == "object_trans":
                return ObjectTransModule(cfg)
            raise

    def get_module_state_dict(self, module_name):
        module = getattr(self, f"{module_name}_module", None)
        return module.state_dict() if module is not None else None

    def save_module(self, module_name, save_path, epoch, additional_info=None):
        module_state_dict = self.get_module_state_dict(module_name)
        if module_state_dict is None:
            print(f"模块 {module_name} 不存在，无法保存")
            return False

        checkpoint_data = {
            "module_name": module_name,
            "module_state_dict": module_state_dict,
            "epoch": epoch,
        }
        if additional_info:
            checkpoint_data.update(additional_info)

        try:
            torch.save(checkpoint_data, save_path)
            return True
        except Exception as exc:
            print(f"保存{module_name}模块失败: {exc}")
            return False

    def freeze_module(self, module_name):
        module = getattr(self, f"{module_name}_module", None)
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module_name):
        module = getattr(self, f"{module_name}_module", None)
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = True

    def configure_training_modules(self, active_modules, frozen_modules=None):
        all_modules = ["velocity_contact", "human_pose", "object_trans"]
        for name in all_modules:
            self.freeze_module(name)
        for name in active_modules:
            self.unfreeze_module(name)
        if frozen_modules:
            for name in frozen_modules:
                self.freeze_module(name)

    def format_input(self, data_dict):
        """
        简化版本的输入格式化，假设输入已经是正确的形状
        期望输入格式（来自 build_model_input_dict）：
        - human_imu: [bs, seq, num_imus, imu_dim]
        - obj_imu: [bs, seq, obj_imu_dim]
        - v_init: [bs, len(_SENSOR_VEL_NAMES), 3]
        - p_init: [bs, len(_REDUCED_POSE_NAMES), 6]
        - trans_init: [bs, 3]
        - trans_gt: [bs, seq, 3]
        - obj_trans_init: [bs, 3]
        - obj_vel_init: [bs, 3]
        - hand_vel_glb_init: [bs, 2, 3]
        - contact_init: [bs, 9]
        - has_object: [bs] bool tensor
        """
        # 直接获取已经格式化好的输入，只做必要的设备和类型转换
        human_imu = data_dict['human_imu']  # [bs, seq, num_imus, imu_dim]
        batch_size, seq_len = human_imu.shape[:2]
        device = human_imu.device
        dtype = human_imu.dtype

        # 直接获取其他输入，假设已经是正确的形状
        obj_imu = data_dict.get('obj_imu')  # [bs, seq, obj_imu_dim]
        if obj_imu is None:
            obj_imu = torch.zeros(batch_size, seq_len, self.imu_dim, device=device, dtype=dtype)
        
        v_init = data_dict.get('v_init')  # [bs, len(_SENSOR_VEL_NAMES), 3]
        if v_init is None:
            v_init = torch.zeros(batch_size, len(_SENSOR_VEL_NAMES), 3, device=device, dtype=dtype)
        
        p_init = data_dict.get('p_init')  # [bs, len(_REDUCED_POSE_NAMES), 6]
        if p_init is None:
            p_init = torch.zeros(batch_size, len(_REDUCED_POSE_NAMES), 6, device=device, dtype=dtype)
        
        trans_init = data_dict.get('trans_init')  # [bs, 3]
        if trans_init is None:
            trans_init = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        else:
            trans_init = trans_init.to(device=device, dtype=dtype)

        trans_gt = data_dict.get('trans_gt')  # [bs, seq, 3]
        if trans_gt is None:
            trans_gt = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        else:
            trans_gt = trans_gt.to(device=device, dtype=dtype)
            if trans_gt.dim() == 2:
                trans_gt = trans_gt.unsqueeze(1).expand(batch_size, seq_len, 3)
            elif trans_gt.dim() == 3 and (trans_gt.shape[0] != batch_size or trans_gt.shape[1] != seq_len):
                trans_gt = trans_gt.reshape(batch_size, seq_len, 3)
            elif trans_gt.dim() != 3:
                trans_gt = trans_gt.view(batch_size, seq_len, 3)
        
        obj_trans_init = data_dict.get('obj_trans_init')  # [bs, 3]
        if obj_trans_init is None:
            obj_trans_init = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        
        obj_vel_init = data_dict.get('obj_vel_init')  # [bs, 3]
        if obj_vel_init is None:
            obj_vel_init = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        
        hand_vel_glb_init = data_dict.get('hand_vel_glb_init')  # [bs, 2, 3]
        if hand_vel_glb_init is None:
            hand_vel_glb_init = torch.zeros(batch_size, 2, 3, device=device, dtype=dtype)
        
        contact_init = data_dict.get('contact_init')  # [bs, 9]
        if contact_init is None:
            contact_init = torch.zeros(batch_size, 9, device=device, dtype=dtype)
        
        has_object = data_dict.get('has_object')  # [bs] bool
        if has_object is None:
            has_object = torch.ones(batch_size, dtype=torch.bool, device=device)

        return {
            'human_imu': human_imu,
            'obj_imu': obj_imu,
            'v_init': v_init,
            'p_init': p_init,
            'trans_init': trans_init,
            'trans_gt': trans_gt,
            'obj_trans_init': obj_trans_init,
            'obj_vel_init': obj_vel_init,
            'hand_vel_glb_init': hand_vel_glb_init,
            'contact_init': contact_init,
            'has_object': has_object,
        }
    def forward(self, data_dict, use_object_data=None, compute_fk=False):
        formatted = self.format_input(data_dict)
        human_imu = formatted["human_imu"]
        obj_imu = formatted["obj_imu"]
        has_object = formatted["has_object"]
        batch_size, seq_len = human_imu.shape[:2]

        results = {}

        if self.velocity_contact_module is not None:
            vc_out = self.velocity_contact_module(
                human_imu,
                obj_imu,
                formatted["hand_vel_glb_init"],
                formatted["obj_vel_init"],
                contact_init=formatted["contact_init"]
            )
        else:
            vc_out = {
                "pred_hand_glb_vel": torch.zeros(batch_size, seq_len, 2, 3, device=human_imu.device),
                "pred_obj_vel": torch.zeros(batch_size, seq_len, 3, device=human_imu.device),
                "pred_hand_contact_logits": torch.zeros(batch_size, seq_len, 3, device=human_imu.device),
                "pred_hand_contact_prob": torch.zeros(batch_size, seq_len, 3, device=human_imu.device),
            }
        results.update(vc_out)

        if self.human_pose_module is not None:
            hp_out = self.human_pose_module(
                human_imu,
                formatted["v_init"],
                formatted["p_init"],
                formatted["trans_gt"],
            )
        else:
            hp_out = {
                "v_pred": torch.zeros(batch_size, seq_len, len(_SENSOR_VEL_NAMES) * 3, device=human_imu.device),
                "p_pred": torch.zeros(batch_size, seq_len, len(_REDUCED_POSE_NAMES) * 6, device=human_imu.device),
                "contact_pred": torch.zeros(batch_size, seq_len, 2, device=human_imu.device),
                "root_vel_local_pred": torch.zeros(batch_size, seq_len, 3, device=human_imu.device),
                "root_vel_pred": torch.zeros(batch_size, seq_len, 3, device=human_imu.device),
                "pred_hand_glb_pos": torch.zeros(batch_size, seq_len, 2, 3, device=human_imu.device),
            }
        results.update(hp_out)

        if use_object_data is None:
            use_object_data = data_dict.get("use_object_data", True)

        if self.object_trans_module is not None and use_object_data and has_object.any():
            obj_out = self.object_trans_module(
                hp_out["pred_hand_glb_pos"],
                vc_out["pred_hand_contact_prob"],
                formatted["obj_trans_init"],
                obj_imu=obj_imu,
                human_imu=human_imu,
                obj_vel_input=vc_out["pred_obj_vel"],
                contact_init=formatted["contact_init"],
                has_object_mask=has_object,
                compute_fk=compute_fk,
            )
        else:
            obj_out = ObjectTransModule.empty_output(batch_size, seq_len, human_imu.device)
        results.update(obj_out)

        results["has_object"] = has_object
        return results
