"""Training script for GPNet based on the GlobalPose paper."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split

from gp_training import GlobalPoseDataset, GPNetLoss
from net import GPNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train GPNet on GlobalPose style datasets.')
    parser.add_argument('--train_files', nargs='+', required=True,
                        help='One or more .pt files used for training (and optional validation split).')
    parser.add_argument('--val_files', nargs='+', default=None,
                        help='Optional extra .pt files used purely for validation.')
    parser.add_argument('--sequence_len', type=int, default=240,
                        help='Truncated sequence length for TBPTT.')
    parser.add_argument('--min_seq_len', type=int, default=60,
                        help='Drop sequences shorter than this length.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of sequences per batch.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Optional pretrained checkpoint for GPNet initialisation.')
    parser.add_argument('--save_dir', type=str, default='checkpoints/gpnet')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='Validate every N epochs.')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Print training loss every N iterations.')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Ratio used to carve validation from training files when val_files is absent.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for dataset splitting.')
    return parser.parse_args()


def collate_to_device(batch: Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor]],
                       device: torch.device) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor]]:
    data, labels = batch
    data = [(seq.to(device), init.to(device)) for seq, init in data]
    labels = [target.to(device) for target in labels]
    return data, labels


def run_epoch(model: GPNet, loader: DataLoader, criterion: GPNetLoss, optimizer=None,
              device: torch.device = torch.device('cpu'), log_interval: int = 50,
              grad_clip: float = 0.0) -> Tuple[float, Dict[str, float]]:
    training = optimizer is not None
    model.train(mode=training)
    total_loss = 0.0
    metrics_sum: Dict[str, float] = {}
    num_batches = 0

    for step, batch in enumerate(loader, start=1):
        data, target = collate_to_device(batch, device)
        preds = model(data)
        loss, metrics = criterion(preds, target)
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
            print(f"Iter {step:05d} | Loss {loss.item():.4f}")

    mean_loss = total_loss / max(1, num_batches)
    mean_metrics = {k: v / max(1, num_batches) for k, v in metrics_sum.items()}
    return mean_loss, mean_metrics


def _calc_subset_stats(dataset: Subset) -> Dict[str, float]:
    base = dataset.dataset
    if not isinstance(base, GlobalPoseDataset):
        return {'num_sequences': float(len(dataset))}
    lengths = [base.data[i].shape[0] for i in dataset.indices]
    if not lengths:
        return {'num_sequences': 0.0, 'mean_length': 0.0, 'min_length': 0.0, 'max_length': 0.0}
    return {
        'num_sequences': float(len(lengths)),
        'mean_length': float(sum(lengths) / len(lengths)),
        'min_length': float(min(lengths)),
        'max_length': float(max(lengths)),
    }


def _print_dataset_info(name: str, dataset) -> None:
    if dataset is None:
        return
    if isinstance(dataset, GlobalPoseDataset):
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

    print('Loading training files...')
    base_train_dataset = GlobalPoseDataset(args.train_files, sequence_len=args.sequence_len,
                                           drop_last=True, min_seq_len=args.min_seq_len)

    train_dataset = base_train_dataset
    val_dataset = None

    if args.val_files:
        val_dataset = GlobalPoseDataset(args.val_files, sequence_len=args.sequence_len,
                                        drop_last=False, min_seq_len=args.min_seq_len)
    else:
        val_ratio = max(0.0, min(1.0, args.val_split))
        total_len = len(base_train_dataset)
        if val_ratio > 0.0 and total_len > 1:
            val_len = max(1, int(total_len * val_ratio))
            if val_len >= total_len:
                val_len = total_len - 1
            if val_len > 0:
                generator = torch.Generator().manual_seed(args.seed)
                train_dataset, val_dataset = random_split(base_train_dataset, [total_len - val_len, val_len],
                                                          generator=generator)
                print(f"Auto validation split: {val_len} / {total_len} ≈ {val_len / total_len:.2%}")
        else:
            print('No validation set provided and val_split <= 0 (or not enough data); skipping validation.')

    _print_dataset_info('Train set', train_dataset)
    _print_dataset_info('Validation set', val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=GlobalPoseDataset.collate_fn)

    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=GlobalPoseDataset.collate_fn)
    else:
        val_loader = None

    model = GPNet(pretrained_path=args.pretrained).to(device)
    criterion = GPNetLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float('inf')
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer,
                                              device, args.log_interval, args.grad_clip)
        log: Dict[str, float] = {'epoch': epoch, 'train_loss': train_loss}
        log.update({f'train_{k}': v for k, v in train_metrics.items()})

        if val_loader and (epoch % args.eval_interval == 0):
            with torch.no_grad():
                val_loss, val_metrics = run_epoch(model, val_loader, criterion,
                                                  optimizer=None, device=device)
            log['val_loss'] = val_loss
            log.update({f'val_{k}': v for k, v in val_metrics.items()})
            if val_loss < best_val:
                best_val = val_loss
                torch.save({'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'args': vars(args)}, Path(args.save_dir) / 'best.pt')
                print(f"  * Validation best loss improved to {val_loss:.4f}")

        torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': vars(args)}, Path(args.save_dir) / 'last.pt')
        history.append(log)
        print(json.dumps(log, ensure_ascii=True, indent=2))

    with open(Path(args.save_dir) / 'train_log.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=True, indent=2)


if __name__ == '__main__':
    main()
