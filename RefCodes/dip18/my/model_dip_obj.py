"""
DIP-style model with object trajectory prediction.
This implementation follows the structure used in DynaIP and TIP:
- Simple RNN base module (similar to original DIP)
- Wrapper that adds object branch on top of the human pose branch
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    """
    Simple RNN module following DIP/DynaIP structure.
    Architecture: Linear -> ReLU -> Dropout -> LSTM -> Linear
    
    This is the PyTorch equivalent of the BiRNN model from the original DIP.
    """
    
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int,
        n_rnn_layer: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_rnn_layer = n_rnn_layer
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection
        self.linear1 = nn.Linear(n_input, n_hidden)
        
        # RNN core (LSTM)
        self.rnn = nn.LSTM(
            input_size=n_hidden,
            hidden_size=n_hidden,
            num_layers=n_rnn_layer,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if n_rnn_layer > 1 else 0.0,
        )
        
        # Output projection
        self.linear2 = nn.Linear(n_hidden * self.num_directions, n_output)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_input)
            h: Optional hidden state tuple (h_0, c_0)
            
        Returns:
            output: (B, T, n_output)
        """
        # Input layer with activation and dropout
        x = self.dropout(F.relu(self.linear1(x)))
        
        # RNN layer
        x, _ = self.rnn(x, h)
        
        # Output layer
        output = self.linear2(x)
        
        return output


class DIPModelWithObject(nn.Module):
    """
    DIP-style model with object trajectory prediction.
    
    Following the structure of model_tip_with_object.py and DynaIP's model_obj.py:
    - human_branch: RNN for human pose estimation
    - object_branch: RNN for object velocity prediction (takes human IMU + object IMU)
    - Optional velocity integration to get object positions
    
    This is a simpler, cleaner implementation that closely follows the original DIP architecture.
    """
    
    def __init__(
        self,
        human_input_size: int,
        human_output_size: int,
        object_input_size: int,
        object_velocity_size: int = 3,
        n_hidden: int = 512,
        n_rnn_layer: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
        dt: float = 1.0 / 30.0,
        integrate_position: bool = True,
    ):
        """
        Args:
            human_input_size: Input dimension for human IMU (e.g., orientation + acceleration)
            human_output_size: Output dimension for human pose
            object_input_size: Input dimension for object IMU
            object_velocity_size: Output dimension for object velocity (typically 3 for 3D)
            n_hidden: Hidden size for RNN
            n_rnn_layer: Number of RNN layers
            bidirectional: Whether to use bidirectional RNN
            dropout: Dropout rate
            dt: Time step for velocity integration
            integrate_position: Whether to integrate velocity to get position
        """
        super().__init__()
        self.dt = dt
        self.integrate_position = integrate_position
        
        # Human pose branch - just takes human IMU
        self.human_branch = RNN(
            n_input=human_input_size,
            n_output=human_output_size,
            n_hidden=n_hidden,
            n_rnn_layer=n_rnn_layer,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        
        # Object branch - takes both human IMU and object IMU
        # This allows the object prediction to be conditioned on human motion
        object_input_total = human_input_size + object_input_size
        self.object_branch = RNN(
            n_input=object_input_total,
            n_output=object_velocity_size,
            n_hidden=n_hidden,
            n_rnn_layer=n_rnn_layer,
            bidirectional=bidirectional,
            dropout=dropout,
        )
    
    def forward(
        self,
        human_imu: torch.Tensor,
        object_imu: torch.Tensor,
        object_init_position: Optional[torch.Tensor] = None,
        human_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        object_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """
        Forward pass.
        
        Args:
            human_imu: (B, T, human_input_size) - Human IMU data
            object_imu: (B, T, object_input_size) - Object IMU data
            object_init_position: (B, 3) or (B, 1, 3) - Initial object position
            human_state: Not used (for compatibility)
            object_state: Not used (for compatibility)
            
        Returns:
            human_pose: (B, T, human_output_size)
            object_velocity: (B, T, object_velocity_size)
            object_position: (B, T, object_velocity_size) - Integrated position
            None: Placeholder for human_state (for compatibility)
            None: Placeholder for object_state (for compatibility)
        """
        # Human pose prediction
        human_pose = self.human_branch(human_imu)
        
        # Object velocity prediction - conditioned on both human and object motion
        object_input = torch.cat([human_imu, object_imu], dim=-1)
        object_velocity = self.object_branch(object_input)
        
        # Integrate velocity to get position
        if self.integrate_position:
            object_position = self.integrate_velocity(object_velocity, object_init_position)
        else:
            object_position = torch.zeros_like(object_velocity)
        
        return human_pose, object_velocity, object_position, None, None
    
    def integrate_velocity(
        self,
        velocity: torch.Tensor,
        init_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Integrate velocity to position using cumulative sum.
        
        Args:
            velocity: (B, T, 3) - Velocity sequence
            init_position: (B, 3) or (B, 1, 3) - Initial position
            
        Returns:
            position: (B, T, 3) - Position sequence
        """
        # Compute displacement at each timestep
        displacement = velocity * self.dt
        
        # Cumulative sum along time axis
        cumsum = torch.cumsum(displacement, dim=1)
        
        # Add initial position if provided
        if init_position is None:
            return cumsum
        
        # Ensure init_position has shape (B, 1, 3)
        if init_position.dim() == 2:
            init_position = init_position.unsqueeze(1)
        
        return cumsum + init_position
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_default_model(
    human_input_size: int,
    human_output_size: int,
    object_input_size: int,
    object_velocity_size: int = 3,
    dt: float = 1.0 / 60.0,
) -> DIPModelWithObject:
    """
    Build a DIP model with default hyperparameters matching the original DIP paper.
    
    Args:
        human_input_size: Input size for human IMU
        human_output_size: Output size for human pose
        object_input_size: Input size for object IMU
        object_velocity_size: Output size for object velocity
        dt: Time step for integration
        
    Returns:
        DIPModelWithObject instance
    """
    return DIPModelWithObject(
        human_input_size=human_input_size,
        human_output_size=human_output_size,
        object_input_size=object_input_size,
        object_velocity_size=object_velocity_size,
        n_hidden=512,
        n_rnn_layer=2,
        bidirectional=True,  # DIP uses BiRNN
        dropout=0.2,
        dt=dt,
        integrate_position=True,
    )
