import os
import argparse
import time
from multiprocessing import cpu_count
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import random
import numpy as np

import utils.config as cfg
from my.dataset_trans_obj import MotionDatasetWithObjectAndTrans, collate_fn_with_object_and_trans
from my.model_trans_obj import PoserWithObjectAndTrans
from my.model_trans_obj2 import PoserWithObjectAndTransV2
from my.loss_trans_obj import loss_vp_obj_trans


def train_trans_obj():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
    print('Seed:', seed)

    default_workers = max(1, (cpu_count() or 1) - 1)

    parser = argparse.ArgumentParser(description="Training configs (with object and translation)")
    parser.add_argument('--lr', type=float, default=1.2e-3, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--save_dir', type=str, default=os.path.join(cfg.save_dir, 'trans_obj'),
                        help='Folder for saving checkpoints and logs')
    parser.add_argument('--save_interval', type=int, default=50, help='Epoch interval for saving model checkpoints')
    parser.add_argument('--train_seg_len', type=int, default=120, help='Window length (frames)')
    parser.add_argument('--batch_size', type=int, default=80, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=default_workers, help='Number of DataLoader worker processes')
    parser.add_argument('--datasets_train', nargs='+', default=['processed_data_BEHAVE_split', 'processed_data_IMHD_split_sample', 'processed_data_OMOMO'], help='Names of dataset subsets inside data_root/<name>/train.')
    parser.add_argument('--datasets_test', nargs='+', default=['processed_data_BEHAVE', 'processed_data_IMHD', 'processed_data_OMOMO'], help='Names of dataset subsets inside data_root/<name>/test.')
    parser.add_argument('--data_root', type=str, default='../../', help='Prepared work directory root')
    parser.add_argument('--body_model_path', type=str,
                        default='../../smpl_models/smplh/male/model.npz',
                        help='Path to SMPL body model for FK (e.g., body_models/smplh/male/model.npz)')
    # loss weights
    parser.add_argument('--w_human', type=float, default=10.0, help='Human pose/velocity loss weight')
    parser.add_argument('--w_obj', type=float, default=1.0, help='Object velocity loss weight')
    parser.add_argument('--w_contact', type=float, default=0.1, help='Foot contact loss weight')
    parser.add_argument('--w_root_vel_local', type=float, default=10.0, help='Root velocity (local) loss weight')
    parser.add_argument('--w_root_vel', type=float, default=0.3, help='Root velocity (world) loss weight')
    parser.add_argument('--w_root_trans', type=float, default=0.2, help='Root translation loss weight')
    # IMU noise
    parser.add_argument('--imu_noise_train', type=float, default=0.1, help='IMU Gaussian noise std for training')
    parser.add_argument('--imu_noise_val', type=float, default=0.05, help='IMU Gaussian noise std for validation')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print('加载训练数据...')
    train_dataset = MotionDatasetWithObjectAndTrans(
        args.datasets_train,
        args.train_seg_len,
        data_root=args.data_root,
        device=device,
        subset='train',
        random_sample=True,
        imu_noise_std=args.imu_noise_train
    )
    pin_memory = device.type == 'cuda'
    num_workers = max(0, args.num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn_with_object_and_trans
    )

    print('加载验证数据...')
    val_dataset = MotionDatasetWithObjectAndTrans(
        args.datasets_test,
        args.train_seg_len,
        data_root=args.data_root,
        device=device,
        subset='test',
        random_sample=False,
        use_full_sequence=True,  # 验证集使用完整序列
        imu_noise_std=args.imu_noise_val
    )
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # 完整序列模式下必须用batch_size=1
            shuffle=False,
            num_workers=0,  # 完整序列模式下建议关闭多进程
            pin_memory=pin_memory,
            persistent_workers=False,
            collate_fn=collate_fn_with_object_and_trans
        )

    def compute_root_translation_batch(root_vel_batch: torch.Tensor, fps: float) -> torch.Tensor:
        with torch.no_grad():
            translations = []
            for b in range(root_vel_batch.shape[0]):
                translations.append(model.velocity_to_root_position(root_vel_batch[b], fps))
            return torch.stack(translations, dim=0)

    def evaluate_validation():
        if val_loader is None or len(val_loader) == 0:
            return None
        model.eval()
        total_loss = 0.0
        metrics = {
            'loss_v': 0.0,
            'loss_p': 0.0,
            'loss_obj': 0.0,
            'loss_contact': 0.0,
            'loss_root_vel_local': 0.0,
            'loss_root_vel': 0.0,
            'loss_root_trans': 0.0,
        }
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                root_vel_local_gt = batch['velocity'][:, :, 0, :]
                root_trans_gt = compute_root_translation_batch(batch['root_velocity'], model.fps)
                v_pred, glb_p_pred, obj_v_pred, contact_pred, root_vel_local_pred, root_vel_pred, root_trans_pred = model(
                    batch['imu'], batch['v_init'], batch['p_init'], batch['obj_imu'], batch['obj_v_init']
                )
                loss, details = loss_vp_obj_trans(
                    v_pred, glb_p_pred, batch['velocity'], batch['ori_glb_reduced'],
                    obj_v_pred, batch['obj_vel'],
                    contact_pred, batch['foot_contact'],
                    root_vel_local_pred, root_vel_local_gt,
                    root_vel_pred, batch['root_velocity'],
                    root_trans_pred, root_trans_gt,
                    w_human=args.w_human,
                    w_obj=args.w_obj,
                    w_contact=args.w_contact,
                    w_root_vel_local=args.w_root_vel_local,
                    w_root_vel=args.w_root_vel,
                    w_root_trans=args.w_root_trans
                )
                total_loss += loss.item()
                for key in metrics:
                    metrics[key] += details.get(key, 0.0)
        denom = max(1, len(val_loader))
        total_loss /= denom
        for key in metrics:
            metrics[key] /= denom
        model.train()
        return total_loss, metrics

    # Prepare body model path
    body_model_path = args.body_model_path

    model = PoserWithObjectAndTrans(body_model_path=body_model_path, fps=30.0).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.num_epochs // 2), verbose=False)

    best_val_loss = float('inf')
    print(f"开始训练 {args.num_epochs} 轮 | batch_size={args.batch_size} | datasets={args.datasets_train}")
    print(f"损失权重: w_human={args.w_human}, w_obj={args.w_obj}, w_contact={args.w_contact}, "
          f"w_root_vel_local={args.w_root_vel_local}, w_root_vel={args.w_root_vel}, w_root_trans={args.w_root_trans}")
    print(f"IMU噪声: train={args.imu_noise_train}, val={args.imu_noise_val}")

    for epoch in range(args.num_epochs):
        train_loss = 0.0
        loss_details = {
            'loss_v': 0.0,
            'loss_p': 0.0,
            'loss_obj': 0.0,
            'loss_contact': 0.0,
            'loss_root_vel_local': 0.0,
            'loss_root_vel': 0.0,
            'loss_root_trans': 0.0,
        }
        start_time = time.time()

        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", leave=False)
        data_time_acc = 0.0
        compute_time_acc = 0.0
        last_end = time.time()

        for batch_idx, batch in enumerate(epoch_iterator):
            data_time_acc += time.time() - last_end
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            root_vel_local_gt = batch['velocity'][:, :, 0, :]
            root_trans_gt = compute_root_translation_batch(batch['root_velocity'], model.fps)

            batch_compute_start = time.time()
            optimizer.zero_grad(set_to_none=True)
            v_pred, glb_p_pred, obj_v_pred, contact_pred, root_vel_local_pred, root_vel_pred, root_trans_pred = model(
                batch['imu'], batch['v_init'], batch['p_init'], batch['obj_imu'], batch['obj_v_init']
            )
            loss, details = loss_vp_obj_trans(
                v_pred, glb_p_pred, batch['velocity'], batch['ori_glb_reduced'],
                obj_v_pred, batch['obj_vel'],
                contact_pred, batch['foot_contact'],
                root_vel_local_pred, root_vel_local_gt,
                root_vel_pred, batch['root_velocity'],
                root_trans_pred, root_trans_gt,
                w_human=args.w_human,
                w_obj=args.w_obj,
                w_contact=args.w_contact,
                w_root_vel_local=args.w_root_vel_local,
                w_root_vel=args.w_root_vel,
                w_root_trans=args.w_root_trans
            )
            loss.backward()
            optimizer.step()

            current_time = time.time()
            compute_time_acc += current_time - batch_compute_start
            avg_data = data_time_acc / (batch_idx + 1)
            avg_compute = compute_time_acc / (batch_idx + 1)
            epoch_iterator.set_postfix({'data': f'{avg_data:.3f}s', 'compute': f'{avg_compute:.3f}s'})
            last_end = current_time

            train_loss += loss.item()
            for key in loss_details:
                loss_details[key] += details.get(key, 0.0)

        train_loss /= max(1, len(train_loader))
        for key in loss_details:
            loss_details[key] /= max(1, len(train_loader))

        scheduler.step()
        end_time = time.time()

        print(f"Epoch {epoch + 1:03d} | Train Loss: {train_loss:.4f} | "
              f"V: {loss_details['loss_v']:.4f} | P: {loss_details['loss_p']:.4f} | Obj: {loss_details['loss_obj']:.4f} | "
              f"Contact: {loss_details['loss_contact']:.4f} | RootLocal: {loss_details['loss_root_vel_local']:.4f} | "
              f"Root: {loss_details['loss_root_vel']:.4f} | RootTrans: {loss_details['loss_root_trans']:.4f} | Time: {end_time - start_time:.2f}s")

        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f'epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved: epoch {epoch + 1}')

        if val_loader is not None and (epoch + 1) % 10 == 0:
            val_result = evaluate_validation()
            if val_result is not None:
                val_loss, val_metrics = val_result
                print(f"    Validation | Loss: {val_loss:.4f} | V: {val_metrics['loss_v']:.4f} | P: {val_metrics['loss_p']:.4f} | "
                      f"Obj: {val_metrics['loss_obj']:.4f} | Contact: {val_metrics['loss_contact']:.4f} | "
                      f"RootLocal: {val_metrics['loss_root_vel_local']:.4f} | Root: {val_metrics['loss_root_vel']:.4f} | RootTrans: {val_metrics['loss_root_trans']:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_val.pth'))
                    print('    验证集最优模型已保存为 best_val.pth')

    if best_val_loss < float('inf'):
        print(f'最佳验证损失: {best_val_loss:.4f}')
    print('训练结束 (含 translation 分支)')


if __name__ == '__main__':
    train_trans_obj()
