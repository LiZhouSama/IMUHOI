"""
Training script for GPNet with object tracking.
Follows GP's standard training pipeline structure.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader, Subset, random_split

import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from my.omomo_gp_dataset import OMOMOGlobalPoseDataset
from my.losses_object_trans import GPNetWithObjectLoss
from my.models.gpnet_object_trans import GPNetWithObject


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train GPNet with object tracking on OMOMO dataset.'
    )
    parser.add_argument(
        '--train_file', type=str, default='../../process/processed_seg_data_BEHAVE/train',
        help='OMOMO data path: directory with .pt files (one per sequence) or single aggregated .pt file.'
    )
    parser.add_argument(
        '--val_file', type=str, default='../../process/processed_seg_data_BEHAVE/test',
        help='Optional OMOMO data path for validation (directory or file).'
    )
    parser.add_argument(
        '--save_dir', type=str, default='my/checkpoints/gp_object_trans_BEHAVE'
    )
    parser.add_argument(
        '--sequence_len', type=int, default=120,
        help='Truncated sequence length for TBPTT.'
    )
    parser.add_argument(
        '--min_seq_len', type=int, default=120,
        help='Drop sequences shorter than this length.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=25,
        help='Number of sequences per batch.'
    )
    parser.add_argument(
        '--num_workers', type=int, default=12,
        help='Number of dataloader workers.'
    )
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument(
        '--pretrained', type=str, default=None,
        help='Optional pretrained GP checkpoint.'
    )
    parser.add_argument(
        '--eval_interval', type=int, default=1,
        help='Validate every N epochs.'
    )
    parser.add_argument(
        '--log_interval', type=int, default=50,
        help='Print training loss every N iterations.'
    )
    parser.add_argument(
        '--val_split', type=float, default=0.1,
        help='Validation split ratio when val_file is absent.'
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fps', type=int, default=30, help='Dataset FPS')
    
    # Loss weights
    parser.add_argument('--w_pl_pos', type=float, default=1.0)
    parser.add_argument('--w_pl_ori', type=float, default=0.5)
    parser.add_argument('--w_ik_pos', type=float, default=1.0)
    parser.add_argument('--w_ik_ori', type=float, default=0.5)
    parser.add_argument('--w_rr_rot', type=float, default=1.0)
    parser.add_argument('--w_vr_vel', type=float, default=0.5)
    parser.add_argument('--w_vr_stat', type=float, default=1.0)
    parser.add_argument('--w_obj_vel', type=float, default=1.0)
    
    # Training stages
    parser.add_argument(
        '--train_pl', action='store_true',default=True,
        help='Train PL network (freeze by default)'
    )
    parser.add_argument(
        '--train_ik', action='store_true',default=True,
        help='Train IK network (freeze by default)'
    )
    parser.add_argument(
        '--train_rr', action='store_true',default=True,
        help='Train RR network (freeze by default)'
    )
    parser.add_argument(
        '--train_vr', action='store_true', default=True,
        help='Train VR network (enabled by default)'
    )
    parser.add_argument(
        '--train_object', action='store_true', default=True,
        help='Train object network (enabled by default)'
    )
    
    return parser.parse_args()


def collate_to_device(
    batch: Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor], Optional[List[int]]],
    device: torch.device
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor], Optional[List[int]]]:
    """Move batch data to device，同时保留样本索引用于物体分支。"""
    if len(batch) == 3:
        data, labels, indices = batch
    else:
        data, labels = batch
        indices = None
    data = [(seq.to(device), init.to(device)) for seq, init in data]
    labels = [target.to(device) for target in labels]
    if indices is not None:
        indices = [int(i) for i in indices]
    return data, labels, indices


def build_object_inputs(
    dataset: OMOMOGlobalPoseDataset,
    data_indices: List[int],
    human_preds: List[torch.Tensor],
    device: torch.device
) -> Tuple[Optional[List[Tuple[torch.Tensor, torch.Tensor]]], Optional[List[torch.Tensor]]]:
    """
    Build object network inputs from human predictions and object data.
    
    Args:
        dataset: OMOMO dataset
        data_indices: Indices of sequences in batch
        human_preds: Human network predictions [T, 189] per sequence
        device: Target device
        
    Returns:
        (obj_inputs, obj_targets) or (None, None) if no object data
    """
    if not dataset.has_object_data:
        return None, None
    
    obj_inputs = []
    obj_targets = []
    
    for idx, human_pred in zip(data_indices, human_preds):
        obj_data = dataset.get_object_data(idx)
        if obj_data is None:
            continue
        human_pred = human_pred.detach()
        
        T = human_pred.shape[0]
        
        obj_imu = obj_data['obj_imu'].to(device)
        obj_vel_root = obj_data['obj_vel_root'].to(device)
        seq_len = min(T, obj_imu.shape[0], obj_vel_root.shape[0])
        if seq_len == 0:
            continue
        if seq_len != T:
            human_pred = human_pred[:seq_len]
        obj_imu = obj_imu[:seq_len]
        obj_vel_root = obj_vel_root[:seq_len]
        
        # Build full VR input (243 dims) from predictions
        # RRJ (135) + pRJ (69) + aRB (18) + wRB (18) + gR (3) = 243
        # For simplicity, we use predicted values where available
        x_vr = torch.zeros(seq_len, 243, device=device)
        
        # Fill with predicted values from RR and IK stages
        # RR output: rrj [90] -> need to expand to full 135
        # IK output: pRJ [69] + gR [3]
        rr_pred = human_pred[:, 90:180]  # [seq_len, 90] - reduced joint rotations
        ik_pred = human_pred[:, 18:90]  # [seq_len, 72] - pRJ (69) + gR (3)
        
        # Approximate: place predictions in corresponding positions
        # This is a simplified version; full version would need proper mapping
        x_vr[:, :90] = rr_pred  # Reduced rotations
        x_vr[:, 135:204] = ik_pred[:, :69]  # Joint positions
        x_vr[:, 240:243] = ik_pred[:, 69:72]  # Gravity direction
        
        # Concatenate with object IMU
        x_obj = torch.cat([x_vr, obj_imu], dim=-1)  # [seq_len, 252]
        
        # Object target: velocity in root frame
        obj_vel_target = obj_vel_root  # [seq_len, 3]
        
        # Create init vector（RNNWithInit 需要首帧目标作为初始状态）
        init_vec = obj_vel_target[0].detach()
        
        obj_inputs.append((x_obj, init_vec))
        obj_targets.append(obj_vel_target)
    
    if not obj_inputs:
        return None, None
    
    return obj_inputs, obj_targets


def run_epoch(
    model: GPNetWithObject,
    loader: DataLoader,
    criterion: GPNetWithObjectLoss,
    optimizer=None,
    device: torch.device = torch.device('cpu'),
    log_interval: int = 50,
    grad_clip: float = 0.0,
    train_object: bool = True
) -> Tuple[float, Dict[str, float]]:
    """Run one epoch of training or validation."""
    training = optimizer is not None
    model.train(mode=training)
    
    total_loss = 0.0
    metrics_sum: Dict[str, float] = {}
    num_batches = 0
    
    for step, batch in enumerate(loader, start=1):
        data, target, batch_indices = collate_to_device(batch, device)
        
        # Human forward pass
        human_preds, _ = model(data, x_object=None, fast=True)
        
        # Object forward pass (if available)
        obj_preds, obj_targets = None, None
        if train_object and batch_indices is not None:
            dataset_ref = loader.dataset
            base_dataset = dataset_ref.dataset if isinstance(dataset_ref, Subset) else dataset_ref
            if isinstance(base_dataset, OMOMOGlobalPoseDataset):
                obj_inputs, obj_targets = build_object_inputs(base_dataset, batch_indices, human_preds, device)
                if obj_inputs is not None and obj_targets is not None:
                    obj_preds = model.object_vr(obj_inputs)
        
        # Compute loss
        loss, metrics = criterion(human_preds, target, obj_preds, obj_targets)
        
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        total_loss += float(loss.item())
        for key, value in metrics.items():
            metrics_sum[key] = metrics_sum.get(key, 0.0) + float(value)
        num_batches += 1
        
        if training and step % log_interval == 0:
            print(f"  Iter {step:05d} | Loss {loss.item():.4f}")
    
    mean_loss = total_loss / max(1, num_batches)
    mean_metrics = {k: v / max(1, num_batches) for k, v in metrics_sum.items()}
    return mean_loss, mean_metrics


def _calc_subset_stats(dataset: Subset) -> Dict[str, float]:
    """Calculate statistics for a Subset."""
    base = dataset.dataset
    if not isinstance(base, OMOMOGlobalPoseDataset):
        return {'num_sequences': float(len(dataset))}
    lengths = [base.data[i].shape[0] for i in dataset.indices]
    if not lengths:
        return {
            'num_sequences': 0.0,
            'mean_length': 0.0,
            'min_length': 0.0,
            'max_length': 0.0
        }
    return {
        'num_sequences': float(len(lengths)),
        'mean_length': float(sum(lengths) / len(lengths)),
        'min_length': float(min(lengths)),
        'max_length': float(max(lengths)),
    }


def _print_dataset_info(name: str, dataset) -> None:
    """Print dataset information."""
    if dataset is None:
        return
    if isinstance(dataset, OMOMOGlobalPoseDataset):
        stats = dataset.stats()
    elif isinstance(dataset, Subset):
        stats = _calc_subset_stats(dataset)
    else:
        stats = {'num_sequences': float(len(dataset))}
    print(f"{name}: {len(dataset)} sequences | stats: {stats}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print('=' * 80)
    print('Training GPNet with Object Tracking')
    print('=' * 80)
    
    # Load datasets
    print('\nLoading training data...')
    base_train_dataset = OMOMOGlobalPoseDataset(
        args.train_file,
        sequence_len=args.sequence_len,
        drop_last=True,
        min_seq_len=args.min_seq_len,
        fps=args.fps,
        device=torch.device('cpu')  # Load on CPU, move to GPU in batches
    )
    
    train_dataset = base_train_dataset
    val_dataset = None
    
    if args.val_file:
        print(f'Loading validation data from {args.val_file}...')
        val_dataset = OMOMOGlobalPoseDataset(
            args.val_file,
            sequence_len=args.sequence_len,
            drop_last=False,
            min_seq_len=args.min_seq_len,
            fps=args.fps,
            device=torch.device('cpu')
        )
    else:
        val_ratio = max(0.0, min(1.0, args.val_split))
        total_len = len(base_train_dataset)
        if val_ratio > 0.0 and total_len > 1:
            val_len = max(1, int(total_len * val_ratio))
            if val_len >= total_len:
                val_len = total_len - 1
            if val_len > 0:
                generator = torch.Generator().manual_seed(args.seed)
                train_dataset, val_dataset = random_split(
                    base_train_dataset,
                    [total_len - val_len, val_len],
                    generator=generator
                )
                print(f'Auto validation split: {val_len}/{total_len} ≈ {val_len/total_len:.2%}')
        else:
            print('No validation set; skipping validation.')
    
    _print_dataset_info('Train set', train_dataset)
    _print_dataset_info('Validation set', val_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=OMOMOGlobalPoseDataset.collate_fn
    )
    
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=OMOMOGlobalPoseDataset.collate_fn
        )
    
    # Create model
    print('\nInitializing model...')
    model = GPNetWithObject(
        pretrained_path=args.pretrained,
        dt=1.0/args.fps
    ).to(device)
    
    # Configure trainable parameters
    trainable_params = []
    
    if args.train_pl:
        print('  - Training PL network')
        for p in model.human.plnet.parameters():
            p.requires_grad = True
            trainable_params.append(p)
    else:
        for p in model.human.plnet.parameters():
            p.requires_grad = False
    
    if args.train_ik:
        print('  - Training IK network')
        for p in model.human.iknet.parameters():
            p.requires_grad = True
            trainable_params.append(p)
    else:
        for p in model.human.iknet.parameters():
            p.requires_grad = False
    
    if args.train_vr:
        print('  - Training VR network')
        for p in model.human.vrnet.parameters():
            p.requires_grad = True
            trainable_params.append(p)
    else:
        for p in model.human.vrnet.parameters():
            p.requires_grad = False
    
    if args.train_object and base_train_dataset.has_object_data:
        print('  - Training Object network')
        for p in model.object_vr.parameters():
            p.requires_grad = True
            trainable_params.append(p)
    else:
        for p in model.object_vr.parameters():
            p.requires_grad = False
    
    # Loss function
    loss_weights = {
        'pl_position': args.w_pl_pos,
        'pl_orientation': args.w_pl_ori,
        'ik_position': args.w_ik_pos,
        'ik_orientation': args.w_ik_ori,
        'rr_rotation': args.w_rr_rot,
        'vr_velocity': args.w_vr_vel,
        'vr_stationary': args.w_vr_stat,
        'obj_velocity': args.w_obj_vel,
    }
    criterion = GPNetWithObjectLoss(weights=loss_weights)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    print(f'\nTotal trainable parameters: {sum(p.numel() for p in trainable_params):,}')
    
    # Training loop
    best_val = float('inf')
    history: List[Dict[str, float]] = []
    
    print('\n' + '=' * 80)
    print('Starting training...')
    print('=' * 80)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print('-' * 80)
        
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, optimizer,
            device, args.log_interval, args.grad_clip,
            train_object=args.train_object
        )
        
        log: Dict[str, float] = {'epoch': epoch, 'train_loss': train_loss}
        log.update({f'train_{k}': v for k, v in train_metrics.items()})
        
        # Validation
        if val_loader and (epoch % args.eval_interval == 0):
            print('  Running validation...')
            with torch.no_grad():
                val_loss, val_metrics = run_epoch(
                    model, val_loader, criterion,
                    optimizer=None, device=device,
                    train_object=args.train_object
                )
            log['val_loss'] = val_loss
            log.update({f'val_{k}': v for k, v in val_metrics.items()})
            
            if val_loss < best_val:
                best_val = val_loss
                save_dict = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': vars(args),
                    'loss_weights': loss_weights
                }
                torch.save(save_dict, Path(args.save_dir) / 'best.pt')
                print(f"  ★ New best validation loss: {val_loss:.4f}")
        
        # Save checkpoint
        save_dict = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
            'loss_weights': loss_weights
        }
        torch.save(save_dict, Path(args.save_dir) / 'last.pt')
        
        history.append(log)
        print(json.dumps(log, indent=2))
    
    # Save training history
    with open(Path(args.save_dir) / 'train_log.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=True)
    
    print('\n' + '=' * 80)
    print('Training completed!')
    print(f'Best validation loss: {best_val:.4f}')
    print(f'Checkpoints saved to: {args.save_dir}')
    print('=' * 80)


if __name__ == '__main__':
    main()
