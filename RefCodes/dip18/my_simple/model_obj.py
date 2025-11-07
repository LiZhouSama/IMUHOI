import torch
import torch.nn.functional as F
from torch import nn
from human_body_prior.body_model.body_model import BodyModel
from pytorch3d import transforms
from my_simple.dataset_obj import _SENSOR_POS_INDICES, _SENSOR_ROT_INDICES

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


class PoserWithObject(nn.Module):
    """基于DynaIP的姿态估计网络，增加了物体估计分支"""
    
    def __init__(self, body_model_path=None, fps=30.0):
        super(PoserWithObject, self).__init__()
        dropout = 0.2
        n_glb = 6
        self.fps = fps

        self.glb = RNN(n_input=72, n_output=n_glb, n_hidden=36, n_rnn_layer=1, dropout=dropout)
        self.human_branch = RNN(n_input=72 + n_glb, n_output=22*9, n_hidden=512, n_rnn_layer=2, bidirectional = True, dropout=dropout)
        self.obj_branch = RNN(n_input=12 + n_glb, n_output=3, n_hidden=128, n_rnn_layer=2, bidirectional=False, dropout=dropout)
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
                
    def forward(self, x, obj_imu):
        """x: [B, T, 6, 12] IMU tensor (batch-first)."""
        B, T, _, _ = x.shape
        device = x.device

        # -------- 1) 全局特征 --------
        x_flat = x.view(B, T, -1)                      # [B, T, 72]
        obj_flat = obj_imu.view(B, T, -1)              # [B, T, 12]
        s_glb = self.glb(x_flat)                       # [B, T, Dg]

        # -------- 2) 分支 poser（已有批处理，不动）--------
        p_pred = self.human_branch(torch.cat((x_flat, s_glb), dim=-1))

        # -------- 3) 物体分支（已有批处理，不动）--------
        obj_v_pred = self.obj_branch(torch.cat((obj_flat, s_glb), dim=-1))

        return p_pred, obj_v_pred


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
    def predict(self, x, obj_imu=None, obj_p0=None):
        """
        Args:
            x: [B, T, 6, 12] IMU tensor or [T, 6, 12] (auto-batched)
            obj_imu: [B, T, 12] or [T, 12] or None
            obj_p0: (B, 3) or (3,) initial object positions. If None, uses zeros.
        
        Returns:
            pose_aa: (B, T, 22, 3) Body pose axis-angle (含根), B=1时直接(T,22,3)
            obj_pos: (B, T, 3) Object position, B=1时直接(T,3)
        """
        self.eval()

        # Convert to batched tensor
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
        dtype = xs.dtype
        B, T, _, _ = xs.shape

        # Handle obj_imu
        if obj_imu is None:
            obj_imu_tensor = torch.zeros(B, T, 12, device=device, dtype=dtype)
        else:
            if isinstance(obj_imu, torch.Tensor):
                if obj_imu.dim() == 2:
                    obj_imu_tensor = obj_imu.unsqueeze(0)
                else:
                    obj_imu_tensor = obj_imu
            else:
                obj_imu_tensor = torch.stack(obj_imu, dim=0)
            obj_imu_tensor = obj_imu_tensor.to(device=device, dtype=dtype)
            if obj_imu_tensor.shape[0] != B:
                raise ValueError("Batch size mismatch between x and obj_imu")
            if obj_imu_tensor.shape[1] != T:
                raise ValueError("Time dim mismatch between x and obj_imu")

        # forward: (B,T,22*9), (B,T,3)
        pose_mat, obj_v = self.forward(xs, obj_imu_tensor)

        # 将人体9D转为轴角 (B,T,22,9) -> (B,T,22,3), 22个关节
        pose_mat = pose_mat.view(B, T, 22, 3, 3)   # (B,T,22,3,3)
        pose_aa = transforms.matrix_to_axis_angle(pose_mat)   # (B,T,22,3)

        # 对物体速度进行积分
        dt = 1.0 / self.fps

        # 处理obj_p0
        if obj_p0 is None:
            obj_p0_tensor = torch.zeros(B, 3, device=device, dtype=dtype)
        else:
            obj_p0_tensor = obj_p0
            if isinstance(obj_p0_tensor, torch.Tensor):
                if obj_p0_tensor.dim() == 1:
                    obj_p0_tensor = obj_p0_tensor.unsqueeze(0)
                obj_p0_tensor = obj_p0_tensor.to(device=device, dtype=dtype)
            else:
                obj_p0_tensor = torch.tensor(obj_p0_tensor, device=device, dtype=dtype)
            if obj_p0_tensor.shape[0] != B:
                raise ValueError(f"obj_p0 batch size mismatch ({obj_p0_tensor.shape[0]}) vs x ({B})")

        obj_pos = []
        for i in range(B):
            v = obj_v[i]        # (T,3)
            p = torch.zeros_like(v)
            if v.shape[0] > 0:
                p[0] = obj_p0_tensor[i]
                if v.shape[0] > 1:
                    p[1:] = obj_p0_tensor[i] + torch.cumsum(v[1:] * dt, dim=0)
            obj_pos.append(p)
        obj_pos = torch.stack(obj_pos, dim=0)  # (B, T, 3)

        if B == 1:
            return pose_aa[0], obj_pos[0]
        return pose_aa, obj_pos