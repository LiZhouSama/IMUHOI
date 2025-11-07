"""
优化的训练脚本（简化版本，不含translation branch）
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

if __package__ is None or __package__ == '':
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _PARENT_DIR = os.path.dirname(_THIS_DIR)
    if _PARENT_DIR not in sys.path:
        sys.path.insert(0, _PARENT_DIR)

from config import vel_scale
from my.dataset_trans_obj import (
    TransPoseObjectDataset,
    collate_transpose,
)
from my.loss_obj import transpose_with_object_loss
from my.model_obj import TransPoseWithObject


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """优化的数据传输 - 支持non_blocking"""
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
        for k, v in batch.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TransPose with object branch (Optimized).")
    parser.add_argument('--data_root', type=str, default='../../process/',
                        help='Root directory containing prepared datasets.')
    parser.add_argument('--datasets', nargs='+', 
                        default=['processed_seg_data_BEHAVE'],
                        help='Names of dataset subsets inside data_root/<name>/train.')
    parser.add_argument('--save_dir', type=str, default=os.path.join('checkpoints', 'transpose_object_behave'),
                        help='Directory to store checkpoints.')
    parser.add_argument('--seq_len', type=int, default=120,
                        help='Sequence length for training windows.')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='Learning rate.')
    parser.add_argument('--trim_frames', type=int, default=6,
                        help='Frames trimmed from the start/end of each sequence.')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Epoch interval for saving checkpoints.')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Frame rate assumed during training.')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (defaults to cuda if available).')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision training.')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of dataloader workers.')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Prefetch factor for dataloader.')
    parser.add_argument('--val_interval', type=int, default=5,
                        help='Validation interval in epochs.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps.')
    return parser.parse_args()


def train() -> None:
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device) if args.device is not None else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"[Train] 使用设备: {device}")
    print(f"[Train] 混合精度训练: {args.use_amp}")

    print(f"[Train] 加载训练数据...")
    dataset = TransPoseObjectDataset(
        datasets=args.datasets,
        seq_len=args.seq_len,
        data_root=args.data_root,
        subset='train',
        fps_default=args.fps,
        trim_frames=args.trim_frames,
        random_sample=True,
        use_full_sequence=False,
        pin_memory=(device.type == 'cuda'),
        device=None,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_transpose,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
    )

    print(f"[Train] 加载验证数据...")
    val_dataset = TransPoseObjectDataset(
        datasets=args.datasets,
        seq_len=args.seq_len,
        data_root=args.data_root,
        subset='test',
        fps_default=args.fps,
        trim_frames=args.trim_frames,
        random_sample=False,
        use_full_sequence=True,
        pin_memory=(device.type == 'cuda'),
        device=None,
    )
    
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            collate_fn=collate_transpose,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            persistent_workers=(args.num_workers > 0),
        )
        print(f"[Train] 验证集样本数: {len(val_dataset)}")

    model = TransPoseWithObject(fps=args.fps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=args.use_amp)

    print(f"[Train] vel_scale={vel_scale} | fps={args.fps}")
    print(f"[Train] Gradient accumulation steps: {args.gradient_accumulation_steps}")

    def evaluate_validation():
        """优化的验证函数"""
        if val_loader is None or len(val_loader) == 0:
            return None
        
        model.eval()
        total_loss = 0.0
        metrics_accumulator: Dict[str, torch.Tensor] = {}  # ✅ 改为tensor类型
        
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch_to_device(batch, device)
                
                with autocast(enabled=args.use_amp):
                    outputs = model(
                        human_imu=batch['imu'],
                        object_imu=batch['obj_imu'],
                        obj_pos_init=batch['obj_pos_init'],
                        fps=batch['fps'],
                    )
                    targets = {
                        'leaf_pos': batch['leaf_pos'],
                        'full_pos': batch['full_pos'],
                        'reduced_pose': batch['reduced_pose'],
                        'obj_velocity': batch['obj_velocity'],
                        'obj_position': batch['obj_position'],
                    }
                    loss, metrics = transpose_with_object_loss(outputs, targets)
                
                total_loss += loss.item()
                # ✅ 累积tensor metrics
                for key, value in metrics.items():
                    if key not in metrics_accumulator:
                        metrics_accumulator[key] = value.clone()
                    else:
                        metrics_accumulator[key] += value
        
        num_batches = max(1, len(val_loader))
        total_loss /= num_batches
        # ✅ 转换为float字典
        metrics_float = {k: float(v.cpu()) / num_batches for k, v in metrics_accumulator.items()}
        
        model.train()
        return total_loss, metrics_float

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        metrics_accumulator: Dict[str, torch.Tensor] = {}  # ✅ 改为tensor类型
        
        data_time_acc = 0.0
        compute_time_acc = 0.0
        total_batch_time_acc = 0.0
        
        iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        last_end_time = time.time()

        for batch_idx, batch in enumerate(iterator):
            data_start_time = time.time()
            batch = _move_batch_to_device(batch, device)
            data_time = time.time() - data_start_time
            data_time_acc += data_time

            compute_start_time = time.time()
            
            with autocast(enabled=args.use_amp):
                outputs = model(
                    human_imu=batch['imu'],
                    object_imu=batch['obj_imu'],
                    obj_pos_init=batch['obj_pos_init'],
                    fps=batch['fps'],
                )

                targets = {
                    'leaf_pos': batch['leaf_pos'],
                    'full_pos': batch['full_pos'],
                    'reduced_pose': batch['reduced_pose'],
                    'obj_velocity': batch['obj_velocity'],
                    'obj_position': batch['obj_position'],
                }

                loss, metrics = transpose_with_object_loss(outputs, targets)
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            compute_time = time.time() - compute_start_time
            compute_time_acc += compute_time

            current_time = time.time()
            batch_time = current_time - last_end_time
            total_batch_time_acc += batch_time
            last_end_time = current_time

            avg_data_time = data_time_acc / (batch_idx + 1)
            avg_compute_time = compute_time_acc / (batch_idx + 1)
            avg_batch_time = total_batch_time_acc / (batch_idx + 1)
            
            iterator.set_postfix({
                'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
                'data': f'{avg_data_time:.3f}s',
                'compute': f'{avg_compute_time:.3f}s', 
                'data%': f'{avg_data_time/avg_batch_time*100:.1f}%',
            })

            epoch_loss += loss.item() * args.gradient_accumulation_steps
            # ✅ 累积tensor metrics，避免GPU同步
            for key, value in metrics.items():
                if key not in metrics_accumulator:
                    metrics_accumulator[key] = value.clone()
                else:
                    metrics_accumulator[key] += value

        num_batches = max(1, len(dataloader))
        epoch_loss /= num_batches
        
        final_avg_data_time = data_time_acc / num_batches
        final_avg_compute_time = compute_time_acc / num_batches
        final_avg_batch_time = total_batch_time_acc / num_batches
        
        # ✅ epoch末尾一次性转换为float
        metrics_str = " | ".join(
            f"{k}: {float(v.cpu()) / num_batches:.4f}" for k, v in sorted(metrics_accumulator.items())
        )
        
        print(f"Epoch {epoch + 1:03d} | Loss: {epoch_loss:.4f} | {metrics_str}")
        print(f"    [Timing] Data: {final_avg_data_time:.3f}s ({final_avg_data_time/final_avg_batch_time*100:.1f}%) | "
              f"Compute: {final_avg_compute_time:.3f}s ({final_avg_compute_time/final_avg_batch_time*100:.1f}%) | "
              f"Total: {final_avg_batch_time:.3f}s | "
              f"Throughput: {args.batch_size/final_avg_batch_time:.1f} samples/s")

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Train] Saved checkpoint: {ckpt_path}")

        if val_loader is not None and (epoch + 1) % args.val_interval == 0:
            val_result = evaluate_validation()
            if val_result is not None:
                val_loss, val_metrics = val_result
                val_metrics_str = " | ".join(
                    f"{k}: {v:.4f}" for k, v in sorted(val_metrics.items())
                )
                print(f"    [Validation] Loss: {val_loss:.4f} | {val_metrics_str}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(args.save_dir, 'best_val.pth')
                    torch.save(model.state_dict(), best_path)
                    print(f'    [Validation] 最优模型已保存 (loss: {best_val_loss:.4f})')

    final_ckpt = os.path.join(args.save_dir, "final.pth")
    torch.save(model.state_dict(), final_ckpt)
    print(f"[Train] Training finished. Model saved to {final_ckpt}")
    if best_val_loss < float('inf'):
        print(f"[Train] 最佳验证损失: {best_val_loss:.4f}")


if __name__ == '__main__':
    train()

