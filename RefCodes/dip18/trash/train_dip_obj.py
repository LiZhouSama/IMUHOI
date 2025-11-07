import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from my.dataset_dip_obj import DIPObjectDataset
from my.loss_dip_obj import loss_p_obj
from my.model_dip_obj import DIPCoreConfig, DIPModelWithObject


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[DataLoader, Optional[DataLoader], DIPObjectDataset, Optional[DIPObjectDataset]]:
    pin_memory = device.type == "cuda"
    train_dataset = DIPObjectDataset(
        dataset_names=args.datasets_train,
        seq_len=args.sequence_length,
        data_root=args.data_root,
        subset=args.train_subset,
        random_sample=True,
        use_full_sequence=False,
        fps_override=args.fps_override,
        trim_frames=args.trim_frames,
        imu_noise_std=args.imu_noise_train,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader: Optional[DataLoader] = None
    val_dataset: Optional[DIPObjectDataset] = None
    if args.datasets_val:
        val_dataset = DIPObjectDataset(
            dataset_names=args.datasets_val,
            seq_len=args.sequence_length,
            data_root=args.data_root,
            subset=args.val_subset,
            random_sample=False,
            use_full_sequence=True,
            fps_override=args.fps_override,
            trim_frames=args.trim_frames,
            imu_noise_std=args.imu_noise_val,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=max(0, min(2, args.num_workers)),
            pin_memory=pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader, train_dataset, val_dataset


def train_dip_obj(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)

    train_loader, val_loader, train_dataset, _ = build_dataloaders(args, device)

    dt = 1.0 / max(args.integration_fps, 1e-8)

    human_cfg = DIPCoreConfig(
        input_size=train_dataset.human_input_dim,
        output_size=train_dataset.human_pose_dim,
        input_fc_layers=args.input_fc_layers,
        input_fc_size=args.input_fc_size,
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_layers=args.rnn_layers,
        rnn_dropout=args.dropout,
        output_fc_layers=args.output_fc_layers,
        output_fc_size=args.output_fc_size,
        activation=args.activation,
    )
    object_cfg = human_cfg.clone_with(
        input_size=train_dataset.human_input_dim + train_dataset.object_input_dim,
        output_size=train_dataset.object_velocity_dim,
    )
    model = DIPModelWithObject(
        human_input_size=train_dataset.human_input_dim,
        human_output_size=train_dataset.human_pose_dim,
        object_input_size=train_dataset.object_input_dim,
        object_velocity_size=train_dataset.object_velocity_dim,
        dt=dt,
        human_config=human_cfg,
        object_config=object_cfg,
        integrate_position=not args.disable_position_integration,
    ).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.num_epochs), eta_min=args.learning_rate * 0.1
    )

    best_val = float("inf")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.num_epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        metrics_accumulator: Dict[str, float] = {}

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}", leave=False)
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            human_pose_pred, obj_vel_pred, obj_pos_pred, _, _ = model(
                batch["human_imu"],
                batch["object_imu"],
                batch["object_init_pos"],
            )

            loss, metrics = loss_p_obj(
                p_pred=human_pose_pred,
                p_gt=batch["human_pose"],
                obj_v_pred=obj_vel_pred,
                obj_v_gt=batch["object_velocity"],
                obj_p_pred=obj_pos_pred,
                obj_p_gt=batch["object_position"],
                w_human_pose=args.w_human_pose,
                w_obj_vel=args.w_object_velocity,
                w_obj_pos=args.w_object_position,
            )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += loss.item()
            for key, value in metrics.items():
                metrics_accumulator[key] = metrics_accumulator.get(key, 0.0) + value

        scheduler.step()
        epoch_time = time.time() - epoch_start

        num_batches = max(1, len(train_loader))
        epoch_loss = running_loss / num_batches
        epoch_metrics = {key: value / num_batches for key, value in metrics_accumulator.items()}

        print(
            f"Epoch {epoch:03d} | Train Loss: {epoch_loss:.4f} | "
            f"Pose: {epoch_metrics.get('loss_pose', 0.0):.4f} | "
            f"Vel: {epoch_metrics.get('loss_vel', 0.0):.4f} | "
            f"ObjVel: {epoch_metrics.get('loss_obj_vel', 0.0):.4f} | "
            f"ObjPos: {epoch_metrics.get('loss_obj_pos', 0.0):.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

        if val_loader is not None and epoch % args.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, args)
            print(f"    Validation Loss: {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": vars(args),
                        "dt": dt,
                    },
                    save_dir / "best_model.pt",
                )
                print("    Saved new best checkpoint: best_model.pt")

        if epoch % args.checkpoint_interval == 0:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": vars(args),
                    "dt": dt,
                },
                save_dir / f"epoch_{epoch:03d}.pt",
            )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": vars(args),
            "dt": dt,
        },
        save_dir / "last_model.pt",
    )
    print("Training finished.")


@torch.no_grad()
def evaluate(
    model: DIPModelWithObject,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    model.eval()
    losses: List[float] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        human_pose_pred, obj_vel_pred, obj_pos_pred, _, _ = model(
            batch["human_imu"],
            batch["object_imu"],
            batch["object_init_pos"],
        )
        loss, _ = loss_p_obj(
            p_pred=human_pose_pred,
            p_gt=batch["human_pose"],
            obj_v_pred=obj_vel_pred,
            obj_v_gt=batch["object_velocity"],
            obj_p_pred=obj_pos_pred,
            obj_p_gt=batch["object_position"],
            w_human_pose=args.w_human_pose,
            w_obj_vel=args.w_object_velocity,
            w_obj_pos=args.w_object_position,
        )
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DIP human pose estimator with object trajectory prediction.")
    parser.add_argument("--datasets-train", nargs="+", default=['processed_data_BEHAVE_split', 'processed_data_IMHD_split', 'processed_data_OMOMO'], help="Training dataset folder names under data_root.")
    parser.add_argument("--datasets-val", nargs="+", default=['processed_data_BEHAVE_split', 'processed_data_IMHD_split', 'processed_data_OMOMO'], help="Validation dataset folder names under data_root.")
    parser.add_argument("--data-root", type=str, default="../../process", help="Root directory that contains processed .pt sequences.")
    parser.add_argument("--train-subset", type=str, default="train", help="Sub-folder name for training split.")
    parser.add_argument("--val-subset", type=str, default="test", help="Sub-folder name for validation split.")
    parser.add_argument("--sequence-length", type=int, default=120, help="Temporal window size in frames.")
    parser.add_argument("--fps-override", type=float, default=30.0, help="Override sequence FPS when loading data.")
    parser.add_argument("--trim-frames", type=int, default=6, help="Frames trimmed from start/end when processing raw sequences.")
    parser.add_argument("--imu-noise-train", type=float, default=0.1, help="Gaussian noise std applied to IMUs during training.")
    parser.add_argument("--imu-noise-val", type=float, default=0.05, help="Gaussian noise std applied to IMUs during validation.")
    parser.add_argument("--save-dir", type=str, default="checkpoints/dip_obj", help="Directory to store checkpoints.")
    parser.add_argument("--num-epochs", type=int, default=60, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Initial learning rate.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm (<=0 to disable).")
    parser.add_argument("--num-workers", type=int, default=12, help="Number of DataLoader worker processes.")
    parser.add_argument("--device", type=str, default=None, help="Device string, e.g., 'cuda:0' or 'cpu'.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--eval-interval", type=int, default=5, help="Validation frequency in epochs.")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Checkpoint frequency in epochs.")
    parser.add_argument("--integration-fps", type=float, default=30.0, help="FPS assumed for integrating object velocity.")
    parser.add_argument("--disable-position-integration", action="store_true", help="Skip velocity integration for object head.")

    parser.add_argument("--input-fc-layers", type=int, default=1, help="Number of FC layers before the RNN.")
    parser.add_argument("--input-fc-size", type=int, default=512, help="Hidden size of input FC layers.")
    parser.add_argument("--rnn-hidden-size", type=int, default=512, help="Hidden size of RNN layers.")
    parser.add_argument("--rnn-layers", type=int, default=2, help="Number of stacked RNN layers.")
    parser.add_argument("--output-fc-layers", type=int, default=1, help="Number of FC layers after the RNN.")
    parser.add_argument("--output-fc-size", type=int, default=256, help="Hidden size of output FC layers.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout applied within FC/RNN stacks.")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function for FC layers.")

    parser.add_argument("--w-human-pose", type=float, default=1.0, help="Loss weight for human pose.")
    parser.add_argument("--w-human-velocity", type=float, default=1.0, help="Loss weight for human pose velocity.")
    parser.add_argument("--w-object-velocity", type=float, default=1.0, help="Loss weight for object velocity.")
    parser.add_argument("--w-object-position", type=float, default=1.0, help="Loss weight for object position.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train_dip_obj(args)


if __name__ == "__main__":
    main()
