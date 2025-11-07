"""
DIP-style training script for human pose estimation with object trajectory.
This implementation follows the structure of the original DIP training code but uses PyTorch.

Key features:
- Configuration management similar to DIP
- Early stopping with tolerance
- Regular validation and checkpointing
- Detailed logging and monitoring
- Support for fine-tuning
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from my.config_dip_style import DIPStyleConfig, create_config_from_args
from my.dataset_omomo_dip import OMOMODIPDataset, collate_fn_omomo_dip
from my.loss_dip_obj import loss_p_obj
from my.model_dip_obj import DIPModelWithObject


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TrainingEngine:
    """
    Main training engine, similar to DIP's TrainingEngine.
    Manages the entire training loop, validation, and checkpointing.
    """
    
    def __init__(
        self,
        config: DIPStyleConfig,
        device: torch.device,
        is_fine_tuning: bool = False,
    ):
        """
        Args:
            config: Configuration object
            device: torch.device for training
            is_fine_tuning: Whether this is fine-tuning from a checkpoint
        """
        self.config = config
        self.device = device
        self.is_fine_tuning = is_fine_tuning
        
        # Training settings
        self.num_epochs = config.get('num_epochs')
        self.batch_size = config.get('batch_size')
        self.evaluate_every_step = config.get('evaluate_every_step')
        self.print_every_step = config.get('print_every_step')
        self.checkpoint_every_step = config.get('checkpoint_every_step')
        self.early_stopping_tolerance = config.get('early_stopping_tolerance')
        self.validate_model = config.get('validate_model')
        
        # Create model directory
        self.model_dir = self._create_model_directory()
        config.set('model_dir', self.model_dir, override=True)
        
        # Save configuration
        config.save_json(os.path.join(self.model_dir, 'config.json'))
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(self.model_dir, 'tensorboard'))
        
        print(f"\n{'='*60}")
        print(f"Training Configuration")
        print(f"{'='*60}")
        print(f"Model directory: {self.model_dir}")
        print(f"Device: {device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of epochs: {self.num_epochs}")
        print(f"Learning rate: {config.get('learning_rate')}")
        print(f"Early stopping tolerance: {self.early_stopping_tolerance}")
        print(f"{'='*60}\n")
        
        # Will be initialized in prepare()
        self.train_dataset: Optional[OMOMODIPDataset] = None
        self.val_dataset: Optional[OMOMODIPDataset] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.model: Optional[DIPModelWithObject] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        
        self.start_epoch = 1
        self.global_step = 0
        self.best_validation_loss = float('inf')
    
    def _create_model_directory(self) -> str:
        """Create a unique directory for this training run."""
        save_dir = self.config.get('save_dir')
        experiment_name = self.config.get('experiment_name', '')
        
        # Create timestamp-based directory name
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if experiment_name:
            dir_name = f"run-{timestamp}-{experiment_name}"
        else:
            dir_name = f"run-{timestamp}"
        
        model_dir = os.path.join(save_dir, dir_name)
        os.makedirs(model_dir, exist_ok=True)
        
        return model_dir
    
    def load_datasets(self) -> Tuple[OMOMODIPDataset, Optional[OMOMODIPDataset]]:
        """
        Load training and validation datasets.
        
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        print("\n[TrainingEngine] Loading datasets...")
        
        # Load training dataset
        train_dataset = OMOMODIPDataset(
            dataset_names=self.config.get('datasets_train'),
            data_root=self.config.get('data_root'),
            subset=self.config.get('train_subset'),
            seq_len=self.config.get('sequence_length'),
            random_sample=True,
            use_full_sequence=False,
            fps=self.config.get('fps_override'),
            trim_frames=self.config.get('trim_frames'),
            imu_noise_std=self.config.get('imu_noise_train'),
            normalize=self.config.get('normalize_data'),
            data_stats=None,  # Will compute statistics
        )
        
        # Save statistics for validation dataset
        train_stats = train_dataset.get_statistics()
        
        # Load validation dataset
        val_dataset = None
        if self.validate_model:
            val_dataset = OMOMODIPDataset(
                dataset_names=self.config.get('datasets_val'),
                data_root=self.config.get('data_root'),
                subset=self.config.get('val_subset'),
                seq_len=self.config.get('sequence_length'),
                random_sample=False,
                use_full_sequence=True,  # Use full sequences for validation
                fps=self.config.get('fps_override'),
                trim_frames=self.config.get('trim_frames'),
                imu_noise_std=self.config.get('imu_noise_val'),
                normalize=self.config.get('normalize_data'),
                data_stats=train_stats,  # Use training statistics
            )
        
        print(f"[TrainingEngine] Training samples: {len(train_dataset)}")
        if val_dataset:
            print(f"[TrainingEngine] Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def create_data_loaders(self):
        """Create PyTorch DataLoaders."""
        print("\n[TrainingEngine] Creating data loaders...")
        
        self.train_dataset, self.val_dataset = self.load_datasets()
        
        pin_memory = self.device.type == 'cuda'
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.get('num_workers'),
            pin_memory=pin_memory,
            collate_fn=collate_fn_omomo_dip,
            drop_last=True,
        )
        
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=1,  # Process full sequences one at a time
                shuffle=False,
                num_workers=max(2, self.config.get('num_workers') // 2),
                pin_memory=pin_memory,
                collate_fn=collate_fn_omomo_dip,
                drop_last=False,
            )
        
        print(f"[TrainingEngine] Training batches per epoch: {len(self.train_loader)}")
        if self.val_loader:
            print(f"[TrainingEngine] Validation batches: {len(self.val_loader)}")
    
    def create_model(self):
        """Create the DIP model with simplified architecture."""
        print("\n[TrainingEngine] Creating model...")
        
        dt = 1.0 / max(self.config.get('integration_fps'), 1e-8)
        
        # Create model with simplified RNN-based architecture
        # This closely follows the original DIP and DynaIP implementations
        self.model = DIPModelWithObject(
            human_input_size=self.train_dataset.human_input_dim,
            human_output_size=self.train_dataset.human_pose_dim,
            object_input_size=self.train_dataset.object_input_dim,
            object_velocity_size=self.train_dataset.object_velocity_dim,
            n_hidden=self.config.get('rnn_hidden_size'),
            n_rnn_layer=self.config.get('rnn_layers'),
            bidirectional=self.config.get('rnn_bidirectional'),
            dropout=self.config.get('dropout'),
            dt=dt,
            integrate_position=not self.config.get('disable_position_integration'),
        ).to(self.device)
        
        num_params = self.model.count_parameters()
        print(f"[TrainingEngine] Model created with {num_params:,} parameters")
        print(f"[TrainingEngine] Architecture: Simplified RNN (DIP-style)")
        print(f"[TrainingEngine] Hidden size: {self.config.get('rnn_hidden_size')}")
        print(f"[TrainingEngine] RNN layers: {self.config.get('rnn_layers')}")
        print(f"[TrainingEngine] Bidirectional: {self.config.get('rnn_bidirectional')}")
        print(f"[TrainingEngine] Human input dim: {self.train_dataset.human_input_dim}")
        print(f"[TrainingEngine] Human output dim: {self.train_dataset.human_pose_dim}")
        print(f"[TrainingEngine] Object input dim: {self.train_dataset.object_input_dim}")
        print(f"[TrainingEngine] Object velocity dim: {self.train_dataset.object_velocity_dim}")
    
    def create_optimizer_and_scheduler(self):
        """Create optimizer and learning rate scheduler."""
        print("\n[TrainingEngine] Creating optimizer and scheduler...")
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate'),
            betas=(0.9, 0.999),
        )
        
        lr_type = self.config.get('learning_rate_type')
        if lr_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=self.config.get('learning_rate_min'),
            )
        elif lr_type == 'exponential':
            decay_rate = self.config.get('learning_rate_decay_rate', 0.96)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=decay_rate,
            )
        else:  # 'fixed'
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: 1.0,
            )
        
        print(f"[TrainingEngine] Optimizer: Adam")
        print(f"[TrainingEngine] LR scheduler: {lr_type}")
        print(f"[TrainingEngine] Initial LR: {self.config.get('learning_rate')}")
    
    def prepare(self):
        """Prepare all components for training."""
        self.create_data_loaders()
        self.create_model()
        self.create_optimizer_and_scheduler()
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """
        Execute one training step.
        
        Args:
            batch: Dictionary of batched tensors
        
        Returns:
            Tuple of (loss_value, metrics_dict)
        """
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Forward pass
        self.optimizer.zero_grad(set_to_none=True)
        
        human_pose_pred, obj_vel_pred, obj_pos_pred, _, _ = self.model(
            batch['human_imu'],
            batch['object_imu'],
            batch['object_init_pos'],
        )
        
        # Compute loss
        loss, metrics = loss_p_obj(
            p_pred=human_pose_pred,
            p_gt=batch['human_pose'],
            obj_v_pred=obj_vel_pred,
            obj_v_gt=batch['object_velocity'],
            obj_p_pred=obj_pos_pred,
            obj_p_gt=batch['object_position'],
            w_human_pose=self.config.get('w_human_pose'),
            w_obj_vel=self.config.get('w_object_velocity'),
            w_obj_pos=self.config.get('w_object_position'),
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        grad_clip = self.config.get('grad_clip')
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        
        self.optimizer.step()
        
        return loss.item(), metrics
    
    def validation_step(self) -> Tuple[float, Dict[str, float]]:
        """
        Run validation on the entire validation set.
        
        Returns:
            Tuple of (avg_loss, avg_metrics)
        """
        if not self.val_loader:
            return 0.0, {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_metrics: Dict[str, float] = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                
                # Forward pass
                human_pose_pred, obj_vel_pred, obj_pos_pred, _, _ = self.model(
                    batch['human_imu'],
                    batch['object_imu'],
                    batch['object_init_pos'],
                )
                
                # Compute loss
                loss, metrics = loss_p_obj(
                    p_pred=human_pose_pred,
                    p_gt=batch['human_pose'],
                    obj_v_pred=obj_vel_pred,
                    obj_v_gt=batch['object_velocity'],
                    obj_p_pred=obj_pos_pred,
                    obj_p_gt=batch['object_position'],
                    w_human_pose=self.config.get('w_human_pose'),
                    w_obj_vel=self.config.get('w_object_velocity'),
                    w_obj_pos=self.config.get('w_object_position'),
                )
                
                total_loss += loss.item()
                for key, value in metrics.items():
                    total_metrics[key] = total_metrics.get(key, 0.0) + value
                num_batches += 1
        
        # Average
        avg_loss = total_loss / max(1, num_batches)
        avg_metrics = {k: v / max(1, num_batches) for k, v in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save a checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_validation_loss': self.best_validation_loss,
            'config': self.config.to_dict(),
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.model_dir, f'checkpoint_epoch_{epoch:03d}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"[TrainingEngine] Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.model_dir, 'model_best.pt')
            torch.save(checkpoint, best_path)
            print(f"[TrainingEngine] Saved best model: {best_path}")
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}\n")
        
        num_steps_without_improvement = 0
        
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            epoch_start_time = time.time()
            
            # Training
            self.model.train()
            running_loss = 0.0
            running_metrics: Dict[str, float] = {}
            
            progress = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}/{self.num_epochs}",
                leave=False,
            )
            
            for batch_idx, batch in enumerate(progress):
                loss, metrics = self.training_step(batch)
                
                running_loss += loss
                for key, value in metrics.items():
                    running_metrics[key] = running_metrics.get(key, 0.0) + value
                
                self.global_step += 1
                
                # Update progress bar
                progress.set_postfix({
                    'loss': f"{loss:.4f}",
                    'pose': f"{metrics.get('loss_pose', 0):.4f}",
                })
            
            # Compute epoch averages
            num_batches = len(self.train_loader)
            epoch_loss = running_loss / max(1, num_batches)
            epoch_metrics = {k: v / max(1, num_batches) for k, v in running_metrics.items()}
            
            # Step scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log to tensorboard
            self.writer.add_scalar('train/loss', epoch_loss, epoch)
            self.writer.add_scalar('train/lr', current_lr, epoch)
            for key, value in epoch_metrics.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)
            
            # Print training summary
            if epoch % self.print_every_step == 0:
                epoch_time = time.time() - epoch_start_time
                print(f"\nEpoch {epoch:03d}/{self.num_epochs} | "
                      f"Loss: {epoch_loss:.4f} | "
                      f"Pose: {epoch_metrics.get('loss_pose', 0):.4f} | "
                      f"ObjVel: {epoch_metrics.get('loss_obj_vel', 0):.4f} | "
                      f"ObjPos: {epoch_metrics.get('loss_obj_pos', 0):.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {epoch_time:.1f}s")
            
            # Validation
            if self.validate_model and epoch % self.evaluate_every_step == 0:
                val_loss, val_metrics = self.validation_step()
                
                # Log to tensorboard
                self.writer.add_scalar('val/loss', val_loss, epoch)
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{key}', value, epoch)
                
                print(f"  Validation | "
                      f"Loss: {val_loss:.4f} | "
                      f"Pose: {val_metrics.get('loss_pose', 0):.4f} | "
                      f"ObjVel: {val_metrics.get('loss_obj_vel', 0):.4f} | "
                      f"ObjPos: {val_metrics.get('loss_obj_pos', 0):.4f}")
                
                # Early stopping check
                if val_loss < self.best_validation_loss:
                    improvement = self.best_validation_loss - val_loss
                    self.best_validation_loss = val_loss
                    num_steps_without_improvement = 0
                    print(f"  New best validation loss! (improved by {improvement:.4f})")
                    
                    # Save best model
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    num_steps_without_improvement += self.evaluate_every_step
                    print(f"  No improvement for {num_steps_without_improvement} epochs")
                
                # Check early stopping
                if num_steps_without_improvement >= self.early_stopping_tolerance:
                    print(f"\n[TrainingEngine] Early stopping triggered after {epoch} epochs")
                    break
            
            # Regular checkpointing
            if epoch % self.checkpoint_every_step == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        # Save final model
        final_path = os.path.join(self.model_dir, 'model_final.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
        }, final_path)
        print(f"\n[TrainingEngine] Training complete! Final model saved to {final_path}")
        
        self.writer.close()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DIP model for human pose + object trajectory (DIP-style implementation)"
    )
    
    # Data arguments
    parser.add_argument('--datasets-train', nargs='+', help='Training dataset names')
    parser.add_argument('--datasets-val', nargs='+', help='Validation dataset names')
    parser.add_argument('--data-root', type=str, help='Root directory for datasets')
    parser.add_argument('--train-subset', type=str, help='Training subset name')
    parser.add_argument('--val-subset', type=str, help='Validation subset name')
    
    # Model arguments
    parser.add_argument('--sequence-length', type=int, help='Sequence length')
    parser.add_argument('--rnn-hidden-size', type=int, help='RNN hidden size')
    parser.add_argument('--rnn-layers', type=int, help='Number of RNN layers')
    
    # Training arguments
    parser.add_argument('--num-epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--device', type=str, help='Device (e.g., cuda:0, cpu)')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    # Other
    parser.add_argument('--config-file', type=str, help='Load config from JSON file')
    parser.add_argument('--experiment-name', type=str, help='Experiment name')
    parser.add_argument('--save-dir', type=str, help='Directory to save checkpoints')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Load or create configuration
    if args.config_file:
        # Try to find config file in multiple locations
        config_path = None
        possible_paths = [
            args.config_file,  # Exact path as provided
            os.path.join('my', args.config_file),  # In my/ subdirectory
            os.path.join(os.path.dirname(__file__), args.config_file),  # Same dir as this script
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if config_path:
            print(f"[Main] Loading configuration from: {config_path}")
            config = DIPStyleConfig.load_json(config_path)
            # Override with command line args
            args_dict = {k: v for k, v in vars(args).items() if v is not None and k != 'config_file'}
            if args_dict:
                print(f"[Main] Overriding config with command line args: {list(args_dict.keys())}")
            config.update(args_dict)
        else:
            print(f"[Main] ERROR: Config file not found: {args.config_file}")
            print(f"[Main] Searched in: {possible_paths}")
            print(f"[Main] Using default configuration instead.")
            config = create_config_from_args(args)
    else:
        print("[Main] No config file specified, using default configuration")
        config = create_config_from_args(args)
    
    # Set random seed
    set_random_seed(config.get('seed'))
    
    # Setup device
    device_str = config.get('device')
    if device_str and torch.cuda.is_available():
        device = torch.device(device_str)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Create training engine
    engine = TrainingEngine(config, device)
    
    # Prepare and train
    engine.prepare()
    engine.train()
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()

