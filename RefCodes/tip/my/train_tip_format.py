"""
Training script that closely follows TIP's original train_model.py structure.
Uses the TIP-format dataset and original TIP components.
"""
import argparse
import os
import sys
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_transformer_with_state import TF_RNN_Past_State
from my.dataset_omomo_tip_v2 import OMOMODatasetTIPFormat
from learning_utils import set_seed, loss_q_only_2axis, loss_jerk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train TIP model on OMOMO data (TIP-format)')
    
    # Data arguments
    parser.add_argument('--train_dirs', type=str, nargs='+', 
                       default=['../../process/processed_data_OMOMO/train'],
                       help='Training data directories')
    parser.add_argument('--val_dirs', type=str, nargs='+',
                       default=['../../process/processed_data_OMOMO/test'],
                       help='Validation data directories')
    
    # Training arguments (matching TIP's defaults)
    parser.add_argument('--batch_size', type=int, default=128,
                       help='batch size (default: 128)')
    parser.add_argument('--cuda', action='store_true',
                       help='use CUDA (default: False)')
    parser.add_argument('--rnn_dropout', type=float, default=0.0,
                       help='dropout applied to layers (default: 0.0)')
    parser.add_argument('--in_dropout', type=float, default=0.0,
                       help='dropout applied to IMU input (default: 0.0)')
    parser.add_argument('--clip', type=float, default=5.0,
                       help='gradient clip, -1 means no clip (default: 5.0)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='upper epoch limit (default: 200)')
    parser.add_argument('--seq_len', type=int, default=60,
                       help='sequence window length for input (default: 60)')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='report interval (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='initial learning rate (default: 2e-4)')
    parser.add_argument('--optim', type=str, default='AdamW',
                       help='optimizer to use (default: AdamW)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='for AdamW')
    
    # Model arguments (matching TIP's defaults)
    parser.add_argument('--rnn_nhid', type=int, default=512,
                       help='hidden size of rnn (default: 512)')
    parser.add_argument('--tf_nhid', type=int, default=1024,
                       help='hidden size of transformer')
    parser.add_argument('--tf_in_dim', type=int, default=256,
                       help='input dimension of transformer')
    parser.add_argument('--n_heads', type=int, default=16,
                       help='num of heads for transformer')
    parser.add_argument('--tf_layers', type=int, default=4,
                       help='num of layers for transformer')
    parser.add_argument('--past_dropout', type=float, default=0.8,
                       help='input dropout for past state in transformer')
    parser.add_argument('--with_acc_sum', action='store_true',
                       help='use accelerometer sum features')
    
    # Additional arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='random seed (default: 42)')
    parser.add_argument('--save_path', type=str, default='checkpoints/tip_omomo_format',
                       help='model save path')
    parser.add_argument('--cosine_lr', action='store_true',
                       help='use cosine learning rate (default: False)')
    parser.add_argument('--warm_start', type=str, default=None,
                       help='path to pretrained model')
    parser.add_argument('--noise_input_hist', type=float, default=0.1,
                       help='noise magnitude for history state augmentation')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='number of data loading workers')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='frame rate (default: 30.0)')
    parser.add_argument('--use_object_imu', action='store_true',
                       help='include object IMU as additional sensor')
    parser.add_argument('--lambda_obj', type=float, default=1.0,
                       help='weight for object velocity loss')
    parser.add_argument('--patience', type=int, default=20,
                       help='early stopping patience')
    
    return parser.parse_args()


def build_dataloader(
    dirs: list,
    args: argparse.Namespace,
    shuffle: bool
) -> DataLoader:
    """Build DataLoader matching TIP's data format."""
    dataset = OMOMODatasetTIPFormat(
        data_dirs=dirs,
        seq_len=args.seq_len,
        frame_rate=args.fps,
        use_object_imu=args.use_object_imu,
        human_joint_num_for_output=18,
        with_acc_sum=args.with_acc_sum,
        random_sample=shuffle,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return loader, dataset


def train_epoch(model, loader, optimizer, lr_scheduler, args, device):
    """Training loop for one epoch (matching TIP's train function)."""
    model.train()
    
    batch_idx = 1
    total_loss = 0
    total_loss_q = 0
    total_loss_obj = 0
    total_loss_j = 0
    i = 0
    num_samples = len(loader.dataset)
    
    for (x_imu, x_s, y) in loader:
        i += x_imu.size()[0]
        start = time.time()
        
        if args.cuda:
            x_imu = x_imu.cuda()
            x_s = x_s.cuda()
            y = y.cuda()
        
        # Add noise to history state (matching TIP's augmentation)
        noise_s = (torch.rand(x_s.size()) - 0.5) * (args.noise_input_hist * 2)
        if args.cuda:
            noise_s = noise_s.cuda()
        
        y_pred = model(x_imu, x_s + noise_s)
        
        # Compute jerk loss on rotation part only (18*6 = 108 dims)
        rot_dim = 18 * 6
        loss_j = loss_jerk(y_pred[:, :, :rot_dim])  # Only rotation part
        
        # Flatten for loss computation
        y_pred_flat = y_pred.reshape(-1, y_pred.size()[-1])
        y_flat = y.reshape(-1, y.size()[-1])
        
        # Split human and object parts
        # State: [18*6 rot, 3 root_vel, 3 obj_vel]
        human_dim = 18 * 6 + 3  # rot + root_vel
        
        # Human loss (rotation + root velocity)
        loss_q = loss_q_only_2axis(y_flat[:, :human_dim], y_pred_flat[:, :human_dim])
        
        # Object velocity loss (simple MSE)
        obj_vel_gt = y_flat[:, -3:]
        obj_vel_pred = y_pred_flat[:, -3:]
        loss_obj = ((obj_vel_pred - obj_vel_gt) ** 2).mean() * args.lambda_obj * 100.0
        
        # Total loss
        loss = loss_q + loss_obj
        if loss_j is not None:
            loss += loss_j
        
        total_loss += loss.item()
        total_loss_q += loss_q.item()
        total_loss_obj += loss_obj.item()
        if loss_j is not None:
            total_loss_j += loss_j.item()
        
        optimizer.zero_grad()
        loss.backward()
        
        total_norm = None
        if args.clip > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        
        optimizer.step()
        
        if args.cosine_lr and lr_scheduler is not None:
            lr_scheduler.step()
            cur_lr = lr_scheduler.get_last_lr()[0]
        else:
            cur_lr = args.lr
        
        batch_idx += 1
        
        end = time.time()
        if batch_idx % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            cur_loss_q = total_loss_q / args.log_interval
            cur_loss_obj = total_loss_obj / args.log_interval
            cur_loss_j = total_loss_j / args.log_interval if loss_j is not None else 0.0
            processed = min(i, num_samples)
            
            print(f'Train Batch {batch_idx:4d} [{processed:6d}/{num_samples:6d} '
                  f'({100. * processed / num_samples:.0f}%)] | '
                  f'LR: {cur_lr:.7f} | Loss: {cur_loss:.6f} '
                  f'(Q:{cur_loss_q:.4f}, Obj:{cur_loss_obj:.4f}, J:{cur_loss_j:.4f}) | '
                  f'Time: {end - start:.4f}s', flush=True)
            
            if total_norm is not None:
                print(f'  Grad norm: {total_norm:.4f}')
            
            total_loss = 0
            total_loss_q = 0
            total_loss_obj = 0
            total_loss_j = 0


def evaluate(model, loader, args, device):
    """Evaluation loop."""
    model.eval()
    
    total_loss = 0
    total_loss_q = 0
    total_loss_obj = 0
    num_batches = 0
    
    with torch.no_grad():
        for (x_imu, x_s, y) in loader:
            if args.cuda:
                x_imu = x_imu.cuda()
                x_s = x_s.cuda()
                y = y.cuda()
            
            y_pred = model(x_imu, x_s)
            
            y_pred_flat = y_pred.reshape(-1, y_pred.size()[-1])
            y_flat = y.reshape(-1, y.size()[-1])
            
            human_dim = 18 * 6 + 3
            loss_q = loss_q_only_2axis(y_flat[:, :human_dim], y_pred_flat[:, :human_dim])
            
            obj_vel_gt = y_flat[:, -3:]
            obj_vel_pred = y_pred_flat[:, -3:]
            loss_obj = ((obj_vel_pred - obj_vel_gt) ** 2).mean() * args.lambda_obj * 100.0
            
            loss = loss_q + loss_obj
            
            total_loss += loss.item()
            total_loss_q += loss_q.item()
            total_loss_obj += loss_obj.item()
            num_batches += 1
    
    if num_batches == 0:
        return 0.0, 0.0, 0.0
    
    avg_loss = total_loss / num_batches
    avg_loss_q = total_loss_q / num_batches
    avg_loss_obj = total_loss_obj / num_batches
    
    return avg_loss, avg_loss_q, avg_loss_obj


def save_model(model, save_path, epoch):
    """Save model checkpoint."""
    if epoch == 1 or epoch % 10 == 0:
        save_filename = os.path.join(save_path, f"it{epoch}.pt")
        torch.save(model.state_dict(), save_filename)
        print(f'Saved checkpoint as {save_filename}')
    
    # Always save latest
    torch.save(model.state_dict(), os.path.join(save_path, "latest.pt"))


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory
    try:
        os.makedirs(args.save_path)
    except FileExistsError:
        print("Warning: save path already exists")
    except OSError:
        print("Error: cannot create save path")
        return
    
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(args)
    
    # Build dataloaders
    print("Building dataloaders...")
    train_loader, train_dataset = build_dataloader(args.train_dirs, args, shuffle=True)
    val_loader = None
    if args.val_dirs:
        val_loader, _ = build_dataloader(args.val_dirs, args, shuffle=False)
    
    print(f"Train dataset size: {len(train_dataset)}")
    if val_loader:
        print(f"Val dataset size: {len(val_loader.dataset)}")
    
    # Build model (matching TIP's architecture)
    input_channels = train_dataset.input_imu_dim
    output_channels = train_dataset.state_dim  # 18*6 + 3 + 3 = 129
    
    print(f"Model input channels: {input_channels}")
    print(f"Model output channels: {output_channels}")
    
    model = TF_RNN_Past_State(
        input_channels, output_channels,
        rnn_hid_size=args.rnn_nhid,
        tf_hid_size=args.tf_nhid,
        tf_in_dim=args.tf_in_dim,
        n_heads=args.n_heads,
        tf_layers=args.tf_layers,
        dropout=args.rnn_dropout,
        in_dropout=args.in_dropout,
        past_state_dropout=args.past_dropout,
        with_rnn=True,
        with_acc_sum=args.with_acc_sum
    )
    
    if args.warm_start is not None:
        print(f"Loading pretrained model from {args.warm_start}")
        model.load_state_dict(torch.load(args.warm_start))
    
    model.to(device)
    
    # Build optimizer
    if args.optim == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)
    
    lr_scheduler = None
    if args.cosine_lr:
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    
    # Training loop
    best_val_loss = float('inf')
    bad_epochs = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*80}")
        
        # Train
        train_epoch(model, train_loader, optimizer, lr_scheduler, args, device)
        
        # Validate
        if val_loader:
            val_loss, val_loss_q, val_loss_obj = evaluate(model, val_loader, args, device)
            print(f"\nValidation: Loss={val_loss:.6f} (Q:{val_loss_q:.4f}, Obj:{val_loss_obj:.4f})")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.save_path, "best.pt"))
                print(f"  New best model saved! (val_loss={val_loss:.6f})")
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                    break
        
        # Save checkpoint
        save_model(model, args.save_path, epoch)
    
    print("\nTraining finished!")


if __name__ == "__main__":
    main()

