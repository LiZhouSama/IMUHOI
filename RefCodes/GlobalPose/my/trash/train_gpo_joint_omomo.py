import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import articulate as art
from torch.cuda.amp import autocast, GradScaler

from net import GPNet
from my.models.gpnet_with_object import ObjectVRNet
from my.dataloaders.omomo_joint_dataset import OMOMOJointSeqDataset, AggregatedDataset, collate_pad_joint


# ---------------------------
# 基础设置
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 性能优先（需要完全复现可改回 deterministic=True / benchmark=False）
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def compute_masked_mse(pred, gt, mask):
    diff = (pred - gt).pow(2).sum(-1)        # [..., 3] -> [..., ]
    diff = diff[mask]
    return diff.mean() if diff.numel() > 0 else pred.new_zeros(())


# ---------------------------
# 一些监督的辅助构造
# ---------------------------
@torch.no_grad()
def gt_root_R_and_g_from_pose(pose_aa: torch.Tensor, mask: torch.Tensor, g_world: torch.Tensor):
    """
    pose_aa: [B,T,24,3] 轴角（局部）
    mask:   [B,T] 有效帧
    g_world:[3] 世界重力方向（单位向量，例如 [0,-1,0]）
    return: R_root [B,T,3,3], g_root [B,T,3]
    """
    B, T = pose_aa.shape[:2]
    root_aa = pose_aa[..., 0, :]  # [B,T,3]
    R_root = art.math.axis_angle_to_rotation_matrix(root_aa.reshape(-1, 3)).reshape(B, T, 3, 3)
    g_root = torch.einsum('btij,j->bti', R_root.transpose(-1, -2), g_world.to(pose_aa.device))  # R^T * g
    if mask is not None:
        inv = ~mask
        if inv.any():
            R_root[inv] = torch.eye(3, device=pose_aa.device)
            g_root[inv] = 0.0
    return R_root, g_root


@torch.no_grad()
def fk_joint_positions(pose_aa: torch.Tensor, tran: torch.Tensor, pm: art.ParametricModel, mask: torch.Tensor):
    """
    基于 articulate.ParametricModel 的 FK（在 pm.device 上运行，推荐 CPU）：
    pose_aa: [B,T,24,3] 轴角（局部），可在任意设备
    tran:    [B,T,3] 根平移（世界），可在任意设备
    pm:      ParametricModel（推荐 device='cpu'，节省显存）
    return:  jpos_world [B,T,24,3]（返回到 pm.device）
    """
    # 统一在 CPU 执行 FK，避免大张量占显存
    device = torch.device('cpu')
    B, T = tran.shape[:2]
    pose_aa_cpu = pose_aa.detach().to(device)
    tran_cpu = tran.detach().to(device)
    R_local = art.math.axis_angle_to_rotation_matrix(pose_aa_cpu.reshape(-1, 3)).reshape(B, T, 24, 3, 3)
    R_local_flat = R_local.reshape(B * T, 24, 3, 3)
    tran_flat = tran_cpu.reshape(B * T, 3)
    # pm.forward_kinematics 返回 (pose_global, joint_world)
    _, jpos = pm.forward_kinematics(R_local_flat, tran=tran_flat)
    jpos = jpos.reshape(B, T, 24, 3)
    if mask is not None:
        inv = ~mask
        if inv.any():
            jpos[inv] = 0.0
    return jpos


@torch.no_grad()
def make_stationary_labels(jpos: torch.Tensor, mask: torch.Tensor, dt: float,
                           joints=(0, 10, 11, 22, 23), thresh: float = 0.08):
    """
    依据关节线速度阈值生成静止标签：
    jpos:  [B,T,24,3] 世界系关节位置（通常在 CPU）
    mask:  [B,T]
    dt:    float (秒)
    joints: 参与静止判定的 5 个关节索引（root, L/R 脚, L/R 手）
    thresh: 速度阈值 m/s
    return: y_static [B,T,5] in {0,1}
    """
    B, T = jpos.shape[:2]
    Jsel = torch.stack([jpos[..., j, :] for j in joints], dim=-2)  # [B,T,5,3]
    vel = torch.zeros_like(Jsel)
    if T > 1:
        vel[:, 1:, :, :] = (Jsel[:, 1:, :, :] - Jsel[:, :-1, :, :]) / max(dt, 1e-8)
    speed = vel.norm(dim=-1)  # [B,T,5]
    y = (speed < thresh).float()
    if mask is not None:
        y[~mask] = 0.0
    return y


# ---------------------------
# 训练主循环
# ---------------------------
def train_loop(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    set_seed(args.seed)

    # dataset：自动检测 .pt（聚合）或目录（单文件）
    if args.train_seq_dir.endswith('.pt') and os.path.isfile(args.train_seq_dir):
        print("检测到聚合数据格式")
        train_set = AggregatedDataset(args.train_seq_dir, fps=args.fps)
        val_set = AggregatedDataset(args.val_seq_dir, fps=args.fps) if args.val_seq_dir and args.val_seq_dir.endswith('.pt') else None
    else:
        print("检测到单文件数据格式")
        train_set = OMOMOJointSeqDataset(args.train_seq_dir, fps=args.fps)
        val_set = OMOMOJointSeqDataset(args.val_seq_dir, fps=args.fps) if args.val_seq_dir else None

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_pad_joint, pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0)
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_pad_joint, pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0)
    ) if val_set is not None else None

    # models
    gp = GPNet().to(device)
    obj = ObjectVRNet(hidden_size=args.obj_hidden, num_layers=args.obj_layers, dropout=args.obj_dropout).to(device)

    # FK 模型（仅用于静止标签），放 CPU 以节省显存
    pm = art.ParametricModel('models/SMPL_male.pkl', device='cpu')

    # freeze or unfreeze PL/IK
    for p in gp.plnet.parameters():
        p.requires_grad = args.train_pl
    for p in gp.iknet.parameters():
        p.requires_grad = args.train_ik
    for p in gp.vrnet.parameters():
        p.requires_grad = True  # VR 始终训练

    train_object = hasattr(train_set, 'has_object_data') and train_set.has_object_data
    print(f"训练物体网络: {train_object}")

    params = [p for p in gp.parameters() if p.requires_grad]
    if train_object:
        params += list(obj.parameters())

    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    bce = nn.BCEWithLogitsLoss(reduction='none')

    best_val = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        gp.train(); obj.train()
        total = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [train]')
        for batch in pbar:
            # ==== H2D 非阻塞传输 ====
            aM = batch['aM'].to(device, non_blocking=True)              # [B,T,6,3]
            wM = batch['wM'].to(device, non_blocking=True)              # [B,T,6,3]
            RMB = batch['RMB'].to(device, non_blocking=True)            # [B,T,6,3,3]
            human_vel = batch['human_vel'].to(device, non_blocking=True)  # [B,T,3]
            obj_imu = batch['obj_imu'].to(device, non_blocking=True)      # [B,T,9]
            obj_vel = batch['obj_vel'].to(device, non_blocking=True)      # [B,T,3]
            mask = batch['mask'].to(device, non_blocking=True)            # [B,T]
            dt = float(batch.get('dt', torch.tensor(1.0 / args.fps)))
            pose_aa = batch.get('pose', None)
            if isinstance(pose_aa, torch.Tensor):
                pose_aa_gpu = pose_aa.to(device, non_blocking=True)
            else:
                pose_aa_gpu = None

            B, T = obj_imu.shape[:2]

            # ==== 先构造 g_root_gt，再一次性前向 gp.vrnet ====
            if isinstance(pose_aa_gpu, torch.Tensor):
                _, g_root_gt = gt_root_R_and_g_from_pose(pose_aa_gpu, mask, torch.tensor([0.0, -1.0, 0.0], device=device))
            else:
                g_root_gt = torch.tensor([0.0, -1.0, 0.0], device=device).view(1, 1, 3).expand(B, T, 3)

            # 组装与 VRNet 一致的输入 243 维：RRJ(0) + pRJ(0) + a(18) + w(18) + gR(3)
            x_vr_full = torch.zeros(B, T, 243, device=device)
            x_vr_full[:, :, 204:222] = aM.view(B, T, -1)   # 18
            x_vr_full[:, :, 222:240] = wM.view(B, T, -1)   # 18
            x_vr_full[:, :, 240:243] = g_root_gt           # 3
            x_vr = [(x_vr_full[b], torch.zeros(9, device=device)) for b in range(B)]

            with autocast(enabled=(device.type == 'cuda')):
                vr_out_seq = gp.vrnet(x_vr)                     # list len=B, each [T,9]
                vr_out = torch.stack(vr_out_seq, dim=1).permute(1, 0, 2)  # [B,T,9]

            # 解析 VRNet 输出
            v_par_mag_pred = vr_out[..., 0]                    # [B,T]
            v_perp_vec_pred = vr_out[..., 1:4]                 # [B,T,3]
            s_logits = vr_out[..., 4:9]                        # [B,T,5] 未过sigmoid的logits
            p_static_pred = torch.sigmoid(s_logits)            # 仅用于日志/可视化，不参与loss

            # === GT 分解（根系） ===
            v_root_gt = human_vel  # world
            if isinstance(pose_aa_gpu, torch.Tensor):
                R_root_gt, _ = gt_root_R_and_g_from_pose(pose_aa_gpu, mask, torch.tensor([0.0, -1.0, 0.0], device=device))
                v_root_gt = torch.einsum('btij,btj->bti', R_root_gt.transpose(-1, -2), human_vel)  # world->root

            g_hat = F.normalize(g_root_gt, dim=-1, eps=1e-6)
            v_par_gt = (v_root_gt * g_hat).sum(-1)                 # [B,T]
            v_perp_gt = v_root_gt - v_par_gt.unsqueeze(-1) * g_hat  # [B,T,3]

            with autocast(enabled=(device.type == 'cuda')):
                L_par = ((v_par_mag_pred - v_par_gt) ** 2)[mask].mean() if mask.any().item() else vr_out.new_zeros(())
                L_perp = ((v_perp_vec_pred - v_perp_gt) ** 2).sum(-1)[mask].mean() if mask.any().item() else vr_out.new_zeros(())
                L_ortho = (((v_perp_vec_pred * g_hat).sum(-1)) ** 2)[mask].mean() if mask.any().item() else vr_out.new_zeros(())
                loss_v = L_par + L_perp + args.lambda_v_ortho * L_ortho

            # === 静止标签：在 CPU 计算，只把标签搬到 GPU ===
            if isinstance(pose_aa_gpu, torch.Tensor):
                tran_cpu = batch['tran'].detach().to('cpu')
                jpos_cpu = fk_joint_positions(pose_aa_gpu.detach().to('cpu'), tran_cpu, pm, mask.detach().to('cpu'))
                y_static_cpu = make_stationary_labels(jpos_cpu, mask.detach().to('cpu'), dt,
                                                      joints=(0, 10, 11, 22, 23), thresh=args.static_thresh)
                y_static = y_static_cpu.to(device, non_blocking=True)
            else:
                y_static = None

            with autocast(enabled=(device.type == 'cuda')):
                if y_static is not None:
                    # 注意：这里用 logits 直接进损失，不要再 sigmoid
                    L_static = bce(s_logits, y_static.float()).mean(-1)  # [B,T]
                    L_static = L_static[mask].mean() if mask.any().item() else vr_out.new_zeros(())
                else:
                    L_static = vr_out.new_zeros(())

            # === 物体 VR：整段一次性前向 ===
            if train_object:
                with autocast(enabled=(device.type == 'cuda')):
                    x_obj_seq = torch.cat([x_vr_full, obj_imu], dim=-1)  # [B,T,252]
                    pred_o = obj.forward_seq(x_obj_seq)                  # [B,T,3]
                    loss_o = compute_masked_mse(pred_o, obj_vel, mask)
                    loss = loss_v + args.lambda_static * L_static + args.lambda_obj * loss_o
            else:
                loss_o = vr_out.new_zeros(())
                loss = loss_v + args.lambda_static * L_static

            # === 反传 ===
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(optim)
            scaler.update()

            total += loss.item()
            loss_o_val = loss_o.item() if train_object else 0.0
            pbar.set_postfix({'loss': f'{loss.item():.6f}',
                              'V': f'{loss_v.item():.6f}',
                              'S': f'{L_static.item():.6f}',
                              'O': f'{loss_o_val:.6f}'})

        avg_train = total / max(1, len(train_loader))

        # ==== 验证 ====
        avg_val = avg_train
        if val_loader is not None:
            gp.eval(); obj.eval()
            vtot = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f'Epoch {epoch}/{args.epochs} [val]'):
                    aM = batch['aM'].to(device, non_blocking=True)
                    wM = batch['wM'].to(device, non_blocking=True)
                    tran = batch['tran'].to(device, non_blocking=True)
                    human_vel = batch['human_vel'].to(device, non_blocking=True)
                    obj_imu = batch['obj_imu'].to(device, non_blocking=True)
                    obj_vel = batch['obj_vel'].to(device, non_blocking=True)
                    mask = batch['mask'].to(device, non_blocking=True)
                    dt_val = float(batch.get('dt', torch.tensor(1.0 / args.fps)))
                    pose_aa = batch.get('pose', None)
                    if isinstance(pose_aa, torch.Tensor):
                        pose_aa_gpu = pose_aa.to(device, non_blocking=True)
                    else:
                        pose_aa_gpu = None

                    B, T = obj_imu.shape[:2]

                    x_vr_full = torch.zeros(B, T, 243, device=device)
                    x_vr_full[:, :, 204:222] = aM.view(B, T, -1)
                    x_vr_full[:, :, 222:240] = wM.view(B, T, -1)
                    if isinstance(pose_aa_gpu, torch.Tensor):
                        _, g_root = gt_root_R_and_g_from_pose(pose_aa_gpu, mask, torch.tensor([0.0, -1.0, 0.0], device=device))
                    else:
                        g_root = torch.tensor([0.0, -1.0, 0.0], device=device).view(1, 1, 3).expand(B, T, 3)
                    x_vr_full[:, :, 240:243] = g_root

                    x_vr = [(x_vr_full[b], torch.zeros(9, device=device)) for b in range(B)]
                    vr_out = torch.stack(gp.vrnet(x_vr), dim=1).permute(1, 0, 2)
                    v_par_mag_pred = vr_out[..., 0]
                    v_perp_vec_pred = vr_out[..., 1:4]
                    s_logits = vr_out[..., 4:9]
                    p_static_pred = torch.sigmoid(s_logits)
                    
                    

                    # GT 分解
                    if isinstance(pose_aa_gpu, torch.Tensor):
                        R_root_gt, g_root_gt = gt_root_R_and_g_from_pose(pose_aa_gpu, mask, torch.tensor([0.0, -1.0, 0.0], device=device))
                        v_root_gt = torch.einsum('btij,btj->bti', R_root_gt.transpose(-1, -2), human_vel)
                    else:
                        g_root_gt = torch.tensor([0.0, -1.0, 0.0], device=device).view(1, 1, 3).expand(B, T, 3)
                        v_root_gt = human_vel

                    g_hat = F.normalize(g_root_gt, dim=-1, eps=1e-6)
                    v_par_gt = (v_root_gt * g_hat).sum(-1)
                    v_perp_gt = v_root_gt - v_par_gt.unsqueeze(-1) * g_hat

                    L_par = ((v_par_mag_pred - v_par_gt) ** 2)[mask].mean() if mask.any().item() else torch.tensor(0.0, device=device)
                    L_perp = ((v_perp_vec_pred - v_perp_gt) ** 2).sum(-1)[mask].mean() if mask.any().item() else torch.tensor(0.0, device=device)
                    L_ortho = (((v_perp_vec_pred * g_hat).sum(-1)) ** 2)[mask].mean() if mask.any().item() else torch.tensor(0.0, device=device)
                    loss_v = L_par + L_perp + args.lambda_v_ortho * L_ortho

                    # 静止 BCE
                    if isinstance(pose_aa_gpu, torch.Tensor):
                        jpos_gt = fk_joint_positions(pose_aa_gpu.detach().to('cpu'), tran.detach().to('cpu'), pm, mask.detach().to('cpu'))
                        y_static = make_stationary_labels(jpos_gt, mask.detach().to('cpu'), dt_val,
                                                          joints=(0, 10, 11, 22, 23), thresh=args.static_thresh)
                        y_static = y_static.to(device, non_blocking=True)
                        # 验证同样用 logits 版本
                        L_static = nn.functional.binary_cross_entropy_with_logits(s_logits, y_static.float(), reduction='none').mean(-1)
                        L_static = L_static[mask].mean() if mask.any().item() else torch.tensor(0.0, device=device)
                    
                    else:
                        L_static = torch.tensor(0.0, device=device)

                    if train_object:
                        x_obj = torch.cat([x_vr_full, obj_imu], dim=-1)  # [B,T,252]
                        pred_o = obj.forward_seq(x_obj)                  # [B,T,3]
                        loss_o = compute_masked_mse(pred_o, obj_vel, mask)
                        vtot += (loss_v + args.lambda_static * L_static + args.lambda_obj * loss_o).item()
                    else:
                        vtot += (loss_v + args.lambda_static * L_static).item()

            avg_val = vtot / max(1, len(val_loader))

        print(f'Epoch {epoch:03d} | train {avg_train:.6f} | val {avg_val:.6f}')

        # ==== 保存 best ====
        if avg_val < best_val:
            best_val = avg_val
            save = {
                'gp_vrnet': gp.vrnet.state_dict(),
                'gp_plnet': gp.plnet.state_dict(),
                'gp_iknet_net1': gp.iknet.net1.state_dict(),
                'gp_iknet_net2': gp.iknet.net2.state_dict(),
            }
            if train_object:
                save['object_vr'] = obj.state_dict()
            torch.save(save, os.path.join(args.save_dir, 'best_weights.pt'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_seq_dir', type=str, default='my/datasets/omomo_train_globalpose_with_objects.pt',
                   help='训练数据路径：可以是包含.pt文件的目录或聚合格式的.pt文件')
    p.add_argument('--val_seq_dir', type=str, default='my/datasets/omomo_test_globalpose_with_objects.pt',
                   help='验证数据路径：可以是包含.pt文件的目录或聚合格式的.pt文件')
    p.add_argument('--save_dir', type=str, default='my/results/checkpoints/gpo_joint_omomo')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch_size', type=int, default=60)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--lambda_obj', type=float, default=1.0)
    p.add_argument('--lambda_static', type=float, default=0.5)
    p.add_argument('--lambda_v_ortho', type=float, default=0.01)
    p.add_argument('--static_thresh', type=float, default=0.08)
    p.add_argument('--train_pl', action='store_true')
    p.add_argument('--train_ik', action='store_true')
    p.add_argument('--obj_hidden', type=int, default=256)
    p.add_argument('--obj_layers', type=int, default=2)
    p.add_argument('--obj_dropout', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    train_loop(args)


if __name__ == '__main__':
    main()
