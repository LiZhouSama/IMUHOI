import argparse
import os
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from learning_utils import set_seed
from my.dataset_omomo_tip import OMOMODatasetWithObject, collate_tip_with_object
from my.loss_tip_obj import tip_human_object_loss
from my.model_tip_with_object import TIPWithObject, TIPWithObjectConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TIP on OMOMO with object IMU (DynaIP-style pipeline).")
    parser.add_argument("--train_dirs", type=str, nargs="+", default=['../../process/processed_data_IMHD_split/train', '../../process/processed_data_BEHAVE_split/train','../../process/processed_data_OMOMO/train',], help="Training sequence directories.")
    parser.add_argument("--val_dirs", type=str, nargs="+", default=['../../process/processed_data_IMHD_split/test', '../../process/processed_data_BEHAVE_split/test','../../process/processed_data_OMOMO/test',], help="Optional validation directories.")
    parser.add_argument("--save_path", type=str, default="output/tip_omomo_obj", help="Directory for checkpoints.")

    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--cosine_lr", action="store_true", help="Use cosine annealing schedule.")
    parser.add_argument("--clip", type=float, default=5.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--pin_memory", action="store_true")

    parser.add_argument("--root_supervision", type=str, default="vel", choices=["vel", "pos"])
    parser.add_argument("--imu_noise_std", type=float, default=0.1, help="Gaussian noise std applied to IMU features.")
    parser.add_argument("--noise_input_hist", type=float, default=0.1, help="Uniform noise range for history state.")
    parser.add_argument("--lambda_obj", type=float, default=0.5, help="Weight for the object velocity loss term.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs).")

    parser.add_argument("--rnn_nhid", type=int, default=512)
    parser.add_argument("--tf_nhid", type=int, default=1024)
    parser.add_argument("--tf_in_dim", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--tf_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--in_dropout", type=float, default=0.0)
    parser.add_argument("--past_dropout", type=float, default=0.8)
    parser.add_argument("--with_acc_sum", action="store_true", help="Enable accelerometer sum trick from TIP.")
    return parser.parse_args()


def build_dataloader(
    dirs: Optional[Sequence[str]],
    args: argparse.Namespace,
    shuffle: bool,
) -> Optional[Tuple[DataLoader, OMOMODatasetWithObject]]:
    if dirs is None or len(dirs) == 0:
        return None

    dataset = OMOMODatasetWithObject(
        data_dirs=dirs,
        seq_len=args.seq_len,
        frame_rate=args.fps,
        use_object_imu=True,
        root_supervision=args.root_supervision,
        random_sample=shuffle,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_tip_with_object,
        drop_last=False,
    )

    return loader, dataset


def build_model(args: argparse.Namespace, dataset: OMOMODatasetWithObject) -> TIPWithObject:
    cfg = TIPWithObjectConfig(
        num_imus_total=dataset.num_imus_total,
        state_dim=dataset.state_dim,
        rnn_hid_size=args.rnn_nhid,
        tf_hid_size=args.tf_nhid,
        tf_in_dim=args.tf_in_dim,
        n_heads=args.n_heads,
        tf_layers=args.tf_layers,
        dropout=args.dropout,
        in_dropout=args.in_dropout,
        past_state_dropout=args.past_dropout,
        with_acc_sum=args.with_acc_sum,
        add_object_head=True,
    )
    return cfg.build()


def _apply_history_noise(tensor: torch.Tensor, magnitude: float) -> torch.Tensor:
    if magnitude <= 0:
        return tensor
    noise = (torch.rand_like(tensor) - 0.5) * (magnitude * 2.0)
    return tensor + noise


def _with_imu_noise(tensor: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return tensor
    return tensor + torch.randn_like(tensor) * std


def train() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    train_result = build_dataloader(args.train_dirs, args, shuffle=True)
    if train_result is None:
        raise RuntimeError("No training data provided.")
    train_loader, train_dataset = train_result

    val_loader = None
    if args.val_dirs:
        val_result = build_dataloader(args.val_dirs, args, shuffle=False)
        if val_result is not None:
            val_loader, _ = val_result

    model = build_model(args, train_dataset).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), verbose=False)
        if args.cosine_lr
        else None
    )

    human_root_dim = train_dataset.human_out_rot_dim + train_dataset.root_pos_dim

    def run_epoch(loader: DataLoader, train_mode: bool) -> Tuple[float, Dict[str, float]]:
        if loader is None:
            return 0.0, {"loss_human": 0.0, "loss_obj": 0.0}

        model.train(train_mode)
        running_loss = 0.0
        stats_acc = {"loss_human": 0.0, "loss_obj": 0.0}
        sample_count = 0

        iterator = tqdm(loader, desc="Train" if train_mode else "Val", leave=False)
        for batch in iterator:
            imu = batch["imu"].to(device, non_blocking=True)
            state_hist = batch["state_hist"].to(device, non_blocking=True)
            target_state = batch["state_target"].to(device, non_blocking=True)

            imu_input = _with_imu_noise(imu, args.imu_noise_std)
            if train_mode:
                state_input = _apply_history_noise(state_hist, args.noise_input_hist)
            else:
                state_input = state_hist

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            pred = model(imu_input, state_input)
            loss, stats = tip_human_object_loss(
                pred, target_state, human_root_dim=human_root_dim, lambda_obj=args.lambda_obj
            )

            if train_mode:
                loss.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

            running_loss += loss.item() * imu.size(0)
            sample_count += imu.size(0)
            for key in stats_acc:
                stats_acc[key] += stats.get(key, 0.0) * imu.size(0)

        if sample_count == 0:
            return 0.0, {k: 0.0 for k in stats_acc}
        avg_loss = running_loss / sample_count
        avg_stats = {k: v / sample_count for k, v in stats_acc.items()}
        return avg_loss, avg_stats

    best_val_loss = float("inf")
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_stats = run_epoch(train_loader, train_mode=True)
        val_loss, val_stats = run_epoch(val_loader, train_mode=False) if val_loader else (0.0, {})

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch:03d} | train={train_loss:.6f} (human={train_stats['loss_human']:.4f}, "
            f"obj={train_stats['loss_obj']:.4f})",
            end="",
        )
        if val_loader:
            print(
                f" | val={val_loss:.6f} (human={val_stats.get('loss_human', 0.0):.4f}, "
                f"obj={val_stats.get('loss_obj', 0.0):.4f})"
            )
        else:
            print()

        torch.save(model.state_dict(), os.path.join(args.save_path, f"epoch_{epoch}.pt"))

        if val_loader:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.save_path, "best.pt"))
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"Early stopping at epoch {epoch} (no val improvement for {args.patience} epochs).")
                    break

    print("Training finished.")


if __name__ == "__main__":
    train()
