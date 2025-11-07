# -*- coding: utf-8 -*-
"""
Wrapper network that augments the original TransPoseNet with an object branch.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn.functional import relu
from pytorch3d import transforms
from config import joint_set, vel_scale, paths
from human_body_prior.body_model.body_model import BodyModel
import numpy as np

smpl_parents = torch.tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],dtype=torch.long,)
def global2local(global_rotmats, parents):
    local_rotmats = torch.zeros_like(global_rotmats)
    local_rotmats[:, 0] = global_rotmats[:, 0]
    for i in range(1, global_rotmats.shape[1]):
        parent_idx = parents[i]
        R_parent = global_rotmats[:, parent_idx]
        R_parent_inv = R_parent.transpose(-1, -2)
        local_rotmats[:, i] = torch.matmul(R_parent_inv, global_rotmats[:, i])
    return local_rotmats

class RNN(torch.nn.Module):
    r"""
    An RNN Module including a linear input layer, an RNN, and a linear output layer.
    支持 [B, T, F] 或 [T, F] 输入（优化版）
    """
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=True, dropout=0.2):
        super(RNN, self).__init__()
        # ✅ 改为 batch_first=True，方便处理 [B, T, F] 输入
        self.rnn = torch.nn.LSTM(n_hidden, n_hidden, n_rnn_layer, bidirectional=bidirectional, batch_first=True, dropout=dropout if n_rnn_layer > 1 else 0.0)
        self.linear1 = torch.nn.Linear(n_input, n_hidden)
        self.linear2 = torch.nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, h=None):
        """
        支持 [B, T, F] 或 [T, F] 输入
        """
        # 检测输入维度
        squeeze_output = False
        if x.dim() == 2:
            # [T, F] -> [1, T, F]
            x = x.unsqueeze(0)
            squeeze_output = True
        
        # 现在 x 是 [B, T, F]
        x = self.dropout(x)
        x = relu(self.linear1(x))  # [B, T, hidden]
        x, h = self.rnn(x, h)      # [B, T, hidden*directions]
        x = self.linear2(x)        # [B, T, n_output]
        
        if squeeze_output:
            x = x.squeeze(0)  # [1, T, n_output] -> [T, n_output]
        
        return x, h


class TransPoseNet(torch.nn.Module):
    r"""
    Whole pipeline for pose and translation estimation.
    """
    def __init__(self, num_past_frame=20, num_future_frame=5, hip_length=None, upper_leg_length=None,
                 lower_leg_length=None, prob_threshold=(0.5, 0.9), gravity_velocity=-0.018):
        r"""
        :param num_past_frame: Number of past frames for a biRNN window.
        :param num_future_frame: Number of future frames for a biRNN window.
        :param hip_length: Hip length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param upper_leg_length: Upper leg length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param lower_leg_length: Lower leg length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param prob_threshold: The probability threshold used to control the fusion of the two translation branches.
        :param gravity_velocity: The gravity velocity added to the Trans-B1 when the body is not on the ground.
        """
        super().__init__()
        n_imu = 6 * 3 + 6 * 9   # acceleration (vector3) and rotation matrix (matrix3x3) of 6 IMUs
        self.pose_s1 = RNN(n_imu,                         joint_set.n_leaf * 3,       256)
        self.pose_s2 = RNN(joint_set.n_leaf * 3 + n_imu,  joint_set.n_full * 3,       64)
        self.pose_s3 = RNN(joint_set.n_full * 3 + n_imu,  joint_set.n_reduced * 6,    128)
        self.tran_b1 = RNN(joint_set.n_leaf * 3 + n_imu,  2,                          64)
        self.tran_b2 = RNN(joint_set.n_full * 3 + n_imu,  3,                          256,    bidirectional=False)

        # lower body joint
        self.num_past_frame = num_past_frame
        self.num_future_frame = num_future_frame
        self.num_total_frame = num_past_frame + num_future_frame + 1
        self.prob_threshold = prob_threshold
        self.gravity_velocity = torch.tensor([0, gravity_velocity, 0])

        # variable
        self.rnn_state = None
        self.imu = None
        self.current_root_y = 0
        self.last_root_pos = torch.zeros(3)
        self.reset()
        self.body_model = BodyModel(bm_fname='../../smpl_models/smplh/male/model.npz', num_betas=16)

        self.eval()

    def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
        glb_reduced_pose = transforms.rotation_6d_to_matrix(glb_reduced_pose.reshape(-1, 6)).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = global2local(global_full_pose, smpl_parents.tolist()).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        pose[:, 0] = root_rotation.view(-1, 3, 3)
        return pose

    def _prob_to_weight(self, p):
        return (p.clamp(self.prob_threshold[0], self.prob_threshold[1]) - self.prob_threshold[0]) / \
               (self.prob_threshold[1] - self.prob_threshold[0])

    def reset(self):
        r"""
        Reset online forward states.
        """
        self.rnn_state = None
        self.imu = None
        self.current_root_y = 0

    def forward(self, imu, rnn_state=None):
        """
        Args:
            imu: [B, T, IMU_DIM] 或 [T, IMU_DIM]
        """
        leaf_joint_position = self.pose_s1.forward(imu)[0]
        full_joint_position = self.pose_s2.forward(torch.cat((leaf_joint_position, imu), dim=-1))[0]
        global_reduced_pose = self.pose_s3.forward(torch.cat((full_joint_position, imu), dim=-1))[0]
        contact_probability = self.tran_b1.forward(torch.cat((leaf_joint_position, imu), dim=-1))[0]
        velocity, rnn_state = self.tran_b2.forward(torch.cat((full_joint_position, imu), dim=-1), rnn_state)
        return leaf_joint_position, full_joint_position, global_reduced_pose, contact_probability, velocity, rnn_state

    @torch.no_grad()
    def forward_offline(self, imu):
        r"""
        Offline forward.

        :param imu: Tensor in shape [num_frame, input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [num_frame, 24, 3, 3] and translation tensor in shape [num_frame, 3].
        """
        _, _, global_reduced_pose, contact_probability, velocity, _ = self.forward(imu)

        # calculate pose (local joint rotation matrices)
        root_rotation = imu[:, -9:].view(-1, 3, 3)
        pose = self._reduced_glb_6d_to_full_local_mat(root_rotation.cpu(), global_reduced_pose.cpu())

        # calculate velocity (translation between two adjacent frames in 60fps in world space)
        j = self.body_model(
            pose_body=pose[:, 1:22].reshape(-1, 63),
            root_orient=pose[:, 0].reshape(-1, 3)
        ).Jtr[:, :24, :].to(imu.device).view(-1, 24, 3)
        tran_b1_vel = self.gravity_velocity + torch.lerp(
            torch.cat((torch.zeros(1, 3), j[:-1, 7] - j[1:, 7])),
            torch.cat((torch.zeros(1, 3), j[:-1, 8] - j[1:, 8])),
            contact_probability.max(dim=1).indices.view(-1, 1)
        )
        tran_b2_vel = root_rotation.bmm(velocity.unsqueeze(-1)).squeeze(-1) * vel_scale / self.fps   # to world space
        weight = self._prob_to_weight(contact_probability.cpu().max(dim=1).values.sigmoid()).view(-1, 1)
        velocity = torch.lerp(tran_b2_vel, tran_b1_vel, weight)

        # remove penetration
        current_root_y = 0
        for i in range(velocity.shape[0]):
            current_foot_y = current_root_y + j[i, 7:9, 1].min().item()
            if current_foot_y + velocity[i, 1].item() <= self.floor_y:
                velocity[i, 1] = self.floor_y - current_foot_y
            current_root_y += velocity[i, 1].item()

        return pose, self.velocity_to_root_position(velocity)

    @staticmethod
    def velocity_to_root_position(velocity):
        r"""
        Change velocity to root position. (not optimized)

        :param velocity: Velocity tensor in shape [num_frame, 3].
        :return: Translation tensor in shape [num_frame, 3] for root positions.
        """
        return torch.stack([velocity[:i+1].sum(dim=0) for i in range(velocity.shape[0])])


_HUMAN_IMU_DIM = 6 * 3 + 6 * 9
_OBJECT_IMU_DIM = 12
_OBJ_NET_HIDDEN = 128
_OBJ_NET_LAYERS = 2


class ObjectNet(nn.Module):
    """Predict object velocity from concatenated human/object IMUs."""

    def __init__(self,
                 input_dim: int = _HUMAN_IMU_DIM + _OBJECT_IMU_DIM,
                 hidden_dim: int = _OBJ_NET_HIDDEN,
                 num_layers: int = _OBJ_NET_LAYERS):
        super().__init__()
        self.net = RNN(
            n_input=input_dim,
            n_output=3,
            n_hidden=hidden_dim,
            n_rnn_layer=num_layers,
            bidirectional=False,
            dropout=0.2,
        )

    def forward(self, human_imu: torch.Tensor, object_imu: torch.Tensor) -> torch.Tensor:
        """
        Args:
            human_imu: Tensor of shape [B, T, _HUMAN_IMU_DIM] 或 [T, _HUMAN_IMU_DIM]
            object_imu: Tensor of shape [B, T, _OBJECT_IMU_DIM] 或 [T, _OBJECT_IMU_DIM]
        Returns:
            object velocity predictions [B, T, 3] 或 [T, 3] (in m/s).
        """
        x = torch.cat((human_imu, object_imu), dim=-1)
        vel, _ = self.net(x)
        return vel


class TransPoseWithObject(nn.Module):
    """
    High-level network that reuses the original TransPoseNet for human pose
    estimation and adds a lightweight RNN branch for object motion.
    """

    def __init__(self,
                 base_kwargs: Optional[Dict[str, float]] = None,
                 obj_input_dim: int = _HUMAN_IMU_DIM + _OBJECT_IMU_DIM,
                 fps: float = 30.0,
                 freeze_base: bool = False):
        super().__init__()
        base_kwargs = base_kwargs or {}
        self.base = TransPoseNet(**base_kwargs)
        self.object_net = ObjectNet(input_dim=obj_input_dim)
        self.fps = float(fps)
        self.vel_scale = float(vel_scale)

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.train(mode)
        self.object_net.train(mode)
        return self

    @staticmethod
    def _reshape_human_outputs(leaf: torch.Tensor,
                               full: torch.Tensor,
                               reduced: torch.Tensor) -> Dict[str, torch.Tensor]:
        leaf_pos = leaf.view(-1, joint_set.n_leaf, 3)
        full_pos = full.view(-1, joint_set.n_full, 3)
        reduced_pose = reduced.view(-1, joint_set.n_reduced, 6)
        return {
            'leaf_pos': leaf_pos,
            'full_pos': full_pos,
            'reduced_pose': reduced_pose,
            'reduced_pose_flat': reduced,
        }

    def _fuse_root_velocity_batch(self,
                                   imu_seq: torch.Tensor,
                                   reduced_pose_flat: torch.Tensor,
                                   contact_logits: torch.Tensor,
                                   local_velocity: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Batch版本的root velocity融合，支持 [B, T, ...] 输入
        Args:
            imu_seq: [B, T, IMU_DIM]
            reduced_pose_flat: [B, T, n_reduced*6]
            contact_logits: [B, T, 2]
            local_velocity: [B, T, 3]
        """
        device = imu_seq.device
        dtype = imu_seq.dtype
        B, T = imu_seq.shape[:2]
        
        # [B, T, 9] -> [B*T, 3, 3]
        root_rotation = imu_seq[:, :, -9:].reshape(B * T, 3, 3)
        pose = self.base._reduced_glb_6d_to_full_local_mat(root_rotation, reduced_pose_flat.reshape(B * T, -1))
        pose_aa = transforms.matrix_to_axis_angle(pose)
        # FK计算 - 批量处理
        fk_out = self.base.body_model(
            pose_body=pose_aa[:, 1:22].reshape(-1, 63),
            root_orient=pose_aa[:, 0].reshape(-1, 3)
        ).Jtr[:, :24, :].to(device).view(B, T, 24, 3)
        # 计算位移 - 向量化操作
        zeros = torch.zeros(B, 1, 3, device=device, dtype=dtype)
        left_disp = torch.cat((zeros, fk_out[:, :-1, 7] - fk_out[:, 1:, 7]), dim=1)  # [B, T, 3]
        right_disp = torch.cat((zeros, fk_out[:, :-1, 8] - fk_out[:, 1:, 8]), dim=1)  # [B, T, 3]
        
        # 接触选择
        contact_choice = torch.argmax(contact_logits, dim=-1, keepdim=True).to(dtype)  # [B, T, 1]
        gravity_velocity = self.base.gravity_velocity.to(device=device, dtype=dtype)
        tran_b1 = gravity_velocity + torch.lerp(left_disp, right_disp, contact_choice)  # [B, T, 3]
        
        # Trans-B2分支 - 批量矩阵乘法
        root_rotation_reshaped = root_rotation.view(B, T, 3, 3)
        tran_b2 = torch.matmul(root_rotation_reshaped, local_velocity.unsqueeze(-1)).squeeze(-1)
        tran_b2 = tran_b2 * (self.vel_scale / self.fps)
        
        # 融合权重
        weight = self.base._prob_to_weight(contact_logits.sigmoid().max(dim=-1).values)  # [B, T]
        weight = weight.unsqueeze(-1)  # [B, T, 1]
        fused_velocity = torch.lerp(tran_b2, tran_b1, weight)
        
        # 批量积分位置
        root_position = self._velocity_to_root_position_batch(fused_velocity)
        
        return {
            'velocity_world': fused_velocity,
            'root_position_world': root_position,
        }

    def _forward_batch(self,
                       human_imu: torch.Tensor,
                       object_imu: torch.Tensor,
                       obj_pos_init: torch.Tensor,
                       fps: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        批量前向传播，支持 [B, T, ...] 输入（优化：不展平）
        Args:
            human_imu: [B, T, _HUMAN_IMU_DIM]
            object_imu: [B, T, _OBJECT_IMU_DIM]
            obj_pos_init: [B, 3]
            fps: [B] or scalar
        """
        B, T = human_imu.shape[:2]
        device = human_imu.device
        
        # ✅ 直接使用 [B, T, F] 输入，不展平
        base_leaf, base_full, base_reduced, contact_logits, local_velocity, _ = self.base.forward(human_imu)
        
        # 重塑人体输出
        leaf_pos = base_leaf.view(B, T, joint_set.n_leaf, 3)
        full_pos = base_full.view(B, T, joint_set.n_full, 3)
        reduced_pose = base_reduced.view(B, T, joint_set.n_reduced, 6)
        
        # 批量融合root velocity
        fusion = self._fuse_root_velocity_batch(human_imu, base_reduced, contact_logits, local_velocity)
        
        # ✅ 批量处理物体速度，不展平
        obj_velocity = self.object_net(human_imu, object_imu)  # [B, T, 3]
        
        # 批量积分物体位置
        obj_position = self._integrate_object_velocity_batch(obj_velocity, obj_pos_init, fps)
        
        return {
            'leaf_pos': leaf_pos,
            'full_pos': full_pos,
            'reduced_pose': reduced_pose,
            'contact_logits': contact_logits,
            'velocity_local': local_velocity,
            'root_velocity_world': fusion['velocity_world'],
            'root_position_world': fusion['root_position_world'],
            'obj_velocity': obj_velocity,
            'obj_position': obj_position,
        }

    def forward(self,
                human_imu: torch.Tensor,
                object_imu: torch.Tensor,
                obj_pos_init: torch.Tensor,
                fps: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        批量前向传播（向量化版本，无for循环）
        Args:
            human_imu: [B, T, _HUMAN_IMU_DIM]
            object_imu: [B, T, _OBJECT_IMU_DIM]
            obj_pos_init: [B, 3]
            fps: Optional tensor [B] or scalar representing sequence FPS.
        Returns:
            Dictionary of batched predictions with keys matching the loss terms.
        """
        if human_imu.dim() != 3:
            raise ValueError("human_imu must be [B, T, F].")
        if object_imu.shape[:2] != human_imu.shape[:2]:
            raise ValueError("object_imu must share batch/sequence dimensions with human_imu.")

        B = human_imu.shape[0]
        
        # 处理fps参数
        if fps is None:
            fps_tensor = torch.full((B,), self.fps, device=human_imu.device, dtype=human_imu.dtype)
        elif fps.dim() == 0:  # scalar
            fps_tensor = fps.expand(B)
        elif fps.dim() == 2:  # [B, T]
            fps_tensor = fps[:, 0]  # 取第一帧的fps
        else:  # [B]
            fps_tensor = fps
        
        # 直接调用批量前向传播
        return self._forward_batch(human_imu, object_imu, obj_pos_init, fps_tensor)

    @staticmethod
    def _integrate_object_velocity_batch(velocity: torch.Tensor,
                                          init_pos: torch.Tensor,
                                          fps: torch.Tensor) -> torch.Tensor:
        """
        批量积分速度到位置 [B, T, 3]
        Args:
            velocity: [B, T, 3] in m/s
            init_pos: [B, 3]
            fps: [B]
        Returns:
            position: [B, T, 3]
        """
        if velocity.numel() == 0:
            return velocity.clone()
        
        B, T, _ = velocity.shape
        dt = 1.0 / fps.clamp(min=1e-6)  # [B]
        dt = dt.view(B, 1, 1)  # [B, 1, 1]
        
        disp = velocity * dt  # [B, T, 3]
        cumulative = torch.cumsum(disp, dim=1)  # [B, T, 3]
        pos = init_pos.unsqueeze(1) + cumulative  # [B, 1, 3] + [B, T, 3]
        pos[:, 0] = init_pos  # 第一帧设为初始位置
        return pos
    
    @staticmethod
    def _velocity_to_root_position_batch(velocity: torch.Tensor) -> torch.Tensor:
        """
        批量积分root velocity到position [B, T, 3]
        Args:
            velocity: [B, T, 3]
        Returns:
            position: [B, T, 3]
        """
        return torch.cumsum(velocity, dim=1)

    @torch.no_grad()
    def forward_offline(self,
                        human_imu: torch.Tensor,
                        object_imu: Optional[torch.Tensor] = None,
                        obj_pos_init: Optional[torch.Tensor] = None,
                        fps: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        单序列推理的便捷接口（用于评估模式）
        Args:
            human_imu: [T, _HUMAN_IMU_DIM]
            object_imu: [T, _OBJECT_IMU_DIM] or None
            obj_pos_init: [3] or None
            fps: float or None
        """
        if human_imu.dim() != 2:
            raise ValueError("human_imu must be [T, F] for offline inference.")
        
        # 转换为batch形式 [1, T, ...]
        human_imu_batch = human_imu.unsqueeze(0)
        obj_imu_seq = object_imu if object_imu is not None else torch.zeros(
            human_imu.size(0), _OBJECT_IMU_DIM, device=human_imu.device, dtype=human_imu.dtype
        )
        obj_imu_batch = obj_imu_seq.unsqueeze(0)
        
        init_pos = obj_pos_init if obj_pos_init is not None else torch.zeros(
            3, device=human_imu.device, dtype=human_imu.dtype
        )
        init_pos_batch = init_pos.unsqueeze(0)
        
        fps_value = float(fps if fps is not None else self.fps)
        fps_tensor = torch.tensor([fps_value], device=human_imu.device, dtype=human_imu.dtype)
        
        # 调用批量前向传播
        out = self._forward_batch(human_imu_batch, obj_imu_batch, init_pos_batch, fps_tensor)
        
        # 移除batch维度
        return {k: v.squeeze(0) for k, v in out.items()}
