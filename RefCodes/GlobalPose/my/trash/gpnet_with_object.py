import torch
import numpy as np
import sys
import os

# 添加根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from net import GPNet
from articulate.utils.torch import RNNWithInit


class ObjectVRNet(torch.nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.net = RNNWithInit(
            input_linear=False,
            # 输入：人体 VRNet 全量 243 维 + 物体 IMU 9 维
            input_size=243 + 9,
            output_size=3,    # v_obj in root/world
            hidden_size=hidden_size,
            num_rnn_layer=num_layers,
            dropout=dropout
        )
        self.reset_state()

    @torch.no_grad()
    def reset_state(self, init_vel=None, batch_size: int = 1):
        # init_vel: [B,3] or None
        if init_vel is None:
            init_vec = torch.zeros(batch_size, 3, device=next(self.parameters()).device)
        else:
            init_vel = init_vel.view(-1, 3)
            init_vec = init_vel
            batch_size = init_vec.shape[0]
        hc = self.net.init_net(init_vec).view(batch_size, 2, self.net.num_layers, self.net.hidden_size).permute(1, 2, 0, 3)
        self.hc = [hc[0].contiguous(), hc[1].contiguous()]

    def forward_seq(self, x_full_seq):
        """
        批量序列前向（训练用）：
        x_full_seq: [B, T, 252]，其中前 243 为人体 VRNet 输入，后 9 为物体 IMU
        returns:    [B, T, 3]
        """
        B, T, _ = x_full_seq.shape
        device = x_full_seq.device
        init_vec = torch.zeros(B, 3, device=device)
        hc = self.net.init_net(init_vec).view(B, 2, self.net.num_layers, self.net.hidden_size).permute(1, 2, 0, 3)
        h0, c0 = hc[0].contiguous(), hc[1].contiguous()
        x = x_full_seq.transpose(0, 1)          # [T,B,252]
        y, _ = self.net.rnn(x, (h0, c0))        # [T,B,H]
        v = self.net.linear2(y)                  # [T,B,3]
        return v.transpose(0, 1)                 # [B,T,3]

    def forward_frame(self, x_full):
        """
        在线/推理逐帧接口（保持兼容）：
        x_full: [B, 252] or [252]，其中前 243 为人体 VRNet 输入，后 9 为物体 IMU
        returns: [B, 3] or [3] (matching batch)
        """
        if x_full.dim() == 1:
            x, self.hc = self.net.rnn(x_full.view(1, 1, -1), self.hc)
            v = self.net.linear2(x.squeeze())
            return v
        else:
            B = x_full.shape[0]
            x, self.hc = self.net.rnn(x_full.view(1, B, -1), self.hc)
            v = self.net.linear2(x.squeeze(0))  # [B, 3]
            return v


class GPNetWithObject(torch.nn.Module):
    def __init__(self, dt=1/30):
        super().__init__()
        self.human = GPNet()
        self.human.dt = dt
        self.object_vr = ObjectVRNet()
        self.obj_tran = None

    @torch.no_grad()
    def rnn_initialize(self, init_pose=None, init_vel=None, init_obj_vel=None, init_obj_tran=None):
        self.human.rnn_initialize(init_pose, init_vel)
        self.object_vr.reset_state(init_obj_vel)
        device = next(self.parameters()).device
        self.obj_tran = (torch.zeros(3, device=device) if init_obj_tran is None
                         else init_obj_tran.to(device).view(3))

    @torch.no_grad()
    def forward_frame_with_object(self, aM, wM, RMB, obj_imu):
        pose, tran = self.human.forward_frame(aM, wM, RMB)
        # 组装与训练一致的人体 VRNet 输入 243 维（仅填充 a/w，其余置 0），再拼接物体 9 维
        # aM,wM: [6,3]，RMB: [6,3,3]
        device = next(self.parameters()).device
        x_vr_full = torch.zeros(243, device=device)
        # 204:222 放 a(18)，222:240 放 w(18)
        x_vr_full[204:222] = aM.view(-1).to(device)
        x_vr_full[222:240] = wM.view(-1).to(device)
        x_obj = torch.cat([x_vr_full, obj_imu.to(device).view(-1)], dim=0)  # [252]
        v_obj = self.object_vr.forward_frame(x_obj)
        self.obj_tran = self.obj_tran + v_obj * self.human.dt
        return pose, tran, self.obj_tran.clone()
