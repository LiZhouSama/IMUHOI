"""
Configuration management similar to DIP's original Configuration class.
Provides a structured way to manage training hyperparameters.
"""

import json
import os
from typing import Any, Dict, Optional


class DIPStyleConfig:
    """
    Configuration manager for DIP-style training.
    Similar to the original DIP Configuration class but simplified for PyTorch.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize configuration with keyword arguments.
        """
        self.config: Dict[str, Any] = {}
        
        # Set defaults
        self._set_defaults()
        
        # Override with provided kwargs
        for key, value in kwargs.items():
            self.config[key] = value
    
    def _set_defaults(self):
        """Set default configuration values."""
        defaults = {
            # Data settings
            'datasets_train': [],
            'datasets_val': [],
            'train_subset': 'train',
            'val_subset': 'test',
            'sequence_length': 120,
            'fps_override': 30.0,
            'trim_frames': 6,
            'imu_noise_train': 0.1,
            'imu_noise_val': 0.05,
            'normalize_data': True,
            
            # Model architecture
            'input_fc_layers': 1,
            'input_fc_size': 512,
            'rnn_hidden_size': 512,
            'rnn_layers': 2,
            'rnn_bidirectional': False,
            'output_fc_layers': 1,
            'output_fc_size': 256,
            'dropout': 0.0,
            'activation': 'relu',
            
            # Training settings
            'num_epochs': 60,
            'batch_size': 256,
            'learning_rate': 2e-4,
            'learning_rate_min': 2e-5,
            'learning_rate_type': 'cosine',  # 'fixed', 'exponential', 'cosine'
            'grad_clip': 1.0,
            'num_workers': 12,
            'device': 'cuda:0',
            'seed': 42,
            
            # Loss weights
            'w_human_pose': 1.0,
            'w_human_velocity': 0.0,  # Not used in current loss
            'w_object_velocity': 1.0,
            'w_object_position': 1.0,
            
            # Integration
            'integration_fps': 30.0,
            'disable_position_integration': False,
            
            # Validation and checkpointing
            'evaluate_every_step': 5,  # epochs
            'print_every_step': 1,  # epochs
            'checkpoint_every_step': 10,  # epochs
            'early_stopping_tolerance': 15,  # epochs without improvement
            'validate_model': True,
            
            # Directories
            'save_dir': 'checkpoints/dip_obj_style',
            'model_dir': None,  # Will be set automatically
            'experiment_name': '',
            
            # Misc
            'tensorboard_verbose': 1,
            'create_timeline': False,
        }
        
        self.config.update(defaults)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any, override: bool = False):
        """Set a configuration value."""
        if key not in self.config or override:
            self.config[key] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values."""
        self.config.update(updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()
    
    def save_json(self, path: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        print(f"[Config] Saved to {path}")
    
    @classmethod
    def load_json(cls, path: str) -> 'DIPStyleConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        config = cls()
        config.config.update(config_dict)
        print(f"[Config] Loaded from {path}")
        return config
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        lines = ["DIPStyleConfig:"]
        for key, value in sorted(self.config.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


def create_default_config() -> DIPStyleConfig:
    """Create a default configuration."""
    return DIPStyleConfig()


def create_config_from_args(args) -> DIPStyleConfig:
    """
    Create configuration from argparse Namespace.
    
    Args:
        args: argparse.Namespace object
    
    Returns:
        DIPStyleConfig object
    """
    config = DIPStyleConfig()
    
    # Update with all args that exist in config
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None:  # Only override if explicitly set
            config.set(key, value, override=True)
    
    return config

