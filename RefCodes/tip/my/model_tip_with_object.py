from dataclasses import asdict, dataclass
from typing import Dict

import torch
from torch import nn

from simple_transformer_with_state import TF_RNN_Past_State


@dataclass
class TIPWithObjectConfig:
    """Configuration helper mirroring the DynaIP style module configs."""

    num_imus_total: int
    state_dim: int
    rnn_hid_size: int = 512
    tf_hid_size: int = 1024
    tf_in_dim: int = 256
    n_heads: int = 16
    tf_layers: int = 4
    dropout: float = 0.0
    in_dropout: float = 0.0
    past_state_dropout: float = 0.8
    with_rnn: bool = True
    with_acc_sum: bool = False
    add_object_head: bool = True

    def build(self) -> "TIPWithObject":
        """Instantiate a model from this configuration."""
        return TIPWithObject(**asdict(self))


class TIPWithObject(nn.Module):
    """
    Wrapper around ``TF_RNN_Past_State`` that optionally refines the object branch.

    The original TIP training expects an input composed of two streams:
        - x_imu: IMU features (accelerations + 6D orientations per sensor)
        - x_s:   history state (human rot6d + root supervision + object velocity)
    The forward output keeps ``state_dim`` unchanged so that legacy checkpoints remain valid.
    """

    def __init__(
        self,
        num_imus_total: int,
        state_dim: int,
        rnn_hid_size: int,
        tf_hid_size: int,
        tf_in_dim: int,
        n_heads: int,
        tf_layers: int,
        dropout: float,
        in_dropout: float,
        past_state_dropout: float,
        with_rnn: bool = True,
        with_acc_sum: bool = False,
        add_object_head: bool = True,
    ) -> None:
        super().__init__()

        self.num_imus_total = num_imus_total
        self.imu_feat_per_sensor = 9
        self.state_dim = state_dim
        self.add_object_head = add_object_head

        input_channels = (
            num_imus_total * (self.imu_feat_per_sensor + 3)
            if with_acc_sum
            else num_imus_total * self.imu_feat_per_sensor
        )

        self.core = TF_RNN_Past_State(
            input_size_imu=input_channels,
            size_s=state_dim,
            rnn_hid_size=rnn_hid_size,
            tf_hid_size=tf_hid_size,
            tf_in_dim=tf_in_dim,
            n_heads=n_heads,
            tf_layers=tf_layers,
            dropout=dropout,
            in_dropout=in_dropout,
            past_state_dropout=past_state_dropout,
            with_rnn=with_rnn,
            with_acc_sum=with_acc_sum,
        )

        if add_object_head:
            self.obj_refine = nn.Sequential(
                nn.Linear(state_dim, state_dim),
                nn.ReLU(inplace=True),
                nn.Linear(state_dim, 3),
            )
        else:
            self.obj_refine = None

    @property
    def output_dim(self) -> int:
        return self.state_dim

    def forward(self, x_imu: torch.Tensor, x_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_imu:  [B, T, C_imu]
            x_state: [B, T, state_dim]
        Returns:
            Predicted next-state tensor of shape [B, T, state_dim].
        """
        s_pred = self.core(x_imu, x_state)
        if self.obj_refine is None:
            return s_pred

        obj_delta = self.obj_refine(s_pred)
        s_out = s_pred.clone()
        s_out[..., -3:] = s_pred[..., -3:] + obj_delta
        return s_out

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Convenience wrapper to align with the DynaIP training loop style."""
        return self.forward(batch["imu"], batch["state_hist"])

    @torch.no_grad()
    def predict(
        self,
        imu: torch.Tensor,
        state_hist: torch.Tensor,
        keep_batch: bool = False,
    ) -> torch.Tensor:
        """
        Run inference on a single sequence or batched sequences.

        Args:
            imu: [B, T, C] or [T, C]
            state_hist: [B, T, D] or [T, D]
            keep_batch: if False and B==1, squeeze the batch dimension.
        """
        was_training = self.training
        self.eval()

        if imu.dim() == 2:
            imu = imu.unsqueeze(0)
        if state_hist.dim() == 2:
            state_hist = state_hist.unsqueeze(0)

        pred = self.forward(imu, state_hist)
        if not keep_batch and pred.shape[0] == 1:
            pred = pred.squeeze(0)

        if was_training:
            self.train()

        return pred

    @staticmethod
    def integrate_object_position(
        obj_vel_seq: torch.Tensor,
        obj_pos_init: torch.Tensor,
        fps: float,
    ) -> torch.Tensor:
        """
        Integrate predicted object velocity to obtain positions in the same frame.

        Args:
            obj_vel_seq: [T, 3] velocity sequence in root coordinates.
            obj_pos_init: [3] initial object position in root coordinates.
            fps: sampling rate used to scale the integration.
        """
        if obj_vel_seq.ndim != 2 or obj_vel_seq.shape[-1] != 3:
            raise ValueError("obj_vel_seq must have shape [T, 3]")
        dt = 1.0 / float(fps)
        pos = torch.zeros_like(obj_vel_seq)
        if obj_vel_seq.shape[0] == 0:
            return pos
        pos[0] = obj_pos_init
        if obj_vel_seq.shape[0] > 1:
            pos[1:] = obj_pos_init + torch.cumsum(obj_vel_seq[1:] * dt, dim=0)
        return pos


__all__ = ["TIPWithObject", "TIPWithObjectConfig"]
