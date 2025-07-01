import os
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from tqdm import tqdm
from datetime import datetime
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d

from utils.utils import tensor2numpy
from models.TransPose_net import TransPoseNet, joint_set
from models.ContactAwareLoss import ContactAwareLoss
from torch.cuda.amp import autocast, GradScaler


def do_train_imu_TransPose(cfg, train_loader, test_loader=None, trial=None, model=None, optimizer=None):
    """
    训练IMU到全身姿态及物体变换的TransPose模型
    
    Args:
        cfg: 配置信息
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        trial: Optuna试验（如果使用超参数搜索）
        model: 预训练模型（如果有）
        optimizer: 预训练模型的优化器（如果有）
    """
    # 初始化配置
    device = torch.device(f'cuda:{cfg.gpus[0]}' if torch.cuda.is_available() else 'cpu')
    model_name = cfg.model_name
    use_tensorboard = cfg.use_tensorboard and not cfg.debug
    pose_rep = 'rot6d'  # 使用6D表示
    max_epoch = cfg.epoch
    save_dir = cfg.save_dir
    scaler = GradScaler()

    # 打印训练配置
    print(f'Training: {model_name} (using TransPose), pose_rep: {pose_rep}')
    print(f'use_tensorboard: {use_tensorboard}, device: {device}')
    print(f'max_epoch: {max_epoch}')
    if not cfg.debug:
        os.makedirs(save_dir, exist_ok=True)

    # 初始化模型（如果没有提供预训练模型）
    if model is None:
        model = TransPoseNet(cfg)
        print(f'Initialized TransPose model with {sum(p.numel() for p in model.parameters())} parameters')
        model = model.to(device)

        # 设置优化器（如果没有提供预训练优化器）
        if optimizer is None:
            optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        print(f'Using pre-trained TransPose model with {sum(p.numel() for p in model.parameters())} parameters')
        model = model.to(device)
        
        # 如果没有提供优化器，创建新的优化器
        if optimizer is None:
            optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 定义学习率调度器
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma)

    # 初始化ContactAwareLoss（如果启用）
    use_contact_aware_loss = getattr(cfg, 'use_contact_aware_loss', False)
    contact_loss_fn = None
    if use_contact_aware_loss:
        contact_loss_fn = ContactAwareLoss(
            contact_distance=getattr(cfg, 'contact_distance', 0.1),
            ramp_up_steps=getattr(cfg, 'contact_ramp_up_steps', 1000),
            loss_weights=getattr(cfg, 'contact_loss_weights', {
                'contact_distance': 1.0,
                'contact_velocity': 0.5,
                'approach_smoothness': 0.3,
                'contact_consistency': 0.2
            })
        )
        print(f'Initialized ContactAwareLoss with contact_distance={contact_loss_fn.contact_distance}')
    else:
        print('ContactAwareLoss is disabled')

    # 如果使用tensorboard，初始化
    writer = None
    if use_tensorboard:
        log_dir = os.path.join(save_dir, 'tensorboard_logs', datetime.now().strftime("%m%d%H%M"))
        writer = SummaryWriter(log_dir=log_dir)
        print(f'TensorBoard logs will be saved to: {log_dir}')

    # 训练循环
    best_loss = float('inf')
    train_losses = defaultdict(list)
    test_losses = defaultdict(list)
    n_iter = 0
    training_step_count = 0  # 用于ContactAwareLoss的渐进式权重

    for epoch in range(max_epoch):
        # 训练阶段
        model.train()
        train_loss = 0
        train_loss_rot = 0
        train_loss_root_pos = 0
        train_loss_leaf_pos = 0
        train_loss_full_pos = 0
        train_loss_root_vel = 0
        train_loss_hand_contact = 0
        train_loss_obj_trans = 0
        train_loss_contact_aware = 0
        train_loss_obj_vel = 0
        train_loss_leaf_vel = 0
        
        train_iter = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch in train_iter:
            # 准备数据 (确保所有需要的键都存在于batch中)
            root_pos = batch["root_pos"].to(device)  # [bs, seq, 3]
            motion = batch["motion"].to(device)  # [bs, seq, num_joints * joint_dim]
            human_imu = batch["human_imu"].to(device)  # [bs, seq, num_imus, 9]
            root_vel = batch["root_vel"].to(device) # [bs, seq, 3]
            
            # 处理可选的物体数据
            bs, seq = human_imu.shape[:2]
            obj_imu = batch.get("obj_imu", None)
            obj_rot = batch.get("obj_rot", None)
            obj_trans = batch.get("obj_trans", None) # 新增：提取物体平移
            
            # 获取速度真值
            obj_vel = batch.get("obj_vel", torch.zeros((bs, seq, 3), device=device, dtype=root_pos.dtype)).to(device) # [bs, seq, 3]
            leaf_vel = batch.get("leaf_vel", torch.zeros((bs, seq, joint_set.n_leaf, 3), device=device, dtype=root_pos.dtype)).to(device) # [bs, seq, n_leaf, 3]
            
            if obj_imu is not None:
                obj_imu = obj_imu.to(device)
            else:
                obj_imu = torch.zeros((bs, seq, 1, cfg.imu_dim if hasattr(cfg, 'imu_dim') else 9), device=device, dtype=human_imu.dtype)
                
            if obj_rot is not None:
                obj_rot = obj_rot.to(device)
            else:
                obj_rot = torch.zeros((bs, seq, 6), device=device, dtype=motion.dtype)

            if obj_trans is not None:
                obj_trans = obj_trans.to(device)
            else:
                obj_trans = torch.zeros((bs, seq, 3), device=device, dtype=root_pos.dtype)

            # 处理可选的BPS特征
            bps_features = batch.get("bps_features", None)
            if bps_features is not None:
                bps_features = bps_features.to(device)
            
            # --- 获取手部接触真值 ---
            lhand_contact_gt = batch.get("lhand_contact", torch.zeros((bs, seq), dtype=torch.bool, device=device)).bool().to(device) # [bs, seq]
            rhand_contact_gt = batch.get("rhand_contact", torch.zeros((bs, seq), dtype=torch.bool, device=device)).bool().to(device) # [bs, seq]
            obj_contact_gt = batch.get("obj_contact", torch.zeros((bs, seq), dtype=torch.bool, device=device)).bool().to(device) # [bs, seq]
            
            hand_contact_gt = torch.stack([lhand_contact_gt, rhand_contact_gt, obj_contact_gt], dim=2).float() # [bs, seq, 3]
            # --- 结束获取手部接触真值 ---

            # 构建传递给模型的data_dict (包含所有需要的键)
            data_dict = {
                "human_imu": human_imu,
                "obj_imu": obj_imu,
                "motion": motion,             # 新增
                "root_pos": root_pos,           # 新增
                "obj_rot": obj_rot,             # 新增
                "obj_trans": obj_trans          # 新增
            }
            
            if bps_features is not None: # 如果有BPS特征，也加入
                data_dict["bps_features"] = bps_features
            
            # 前向传播
            optimizer.zero_grad()
            
            # TransPose直接输出预测结果
            pred_dict = model(data_dict)
            
            # 计算真实的关节位置
            with torch.no_grad():
                gt_pose_6d_flat = motion.reshape(-1, model.num_joints * model.joint_dim) # [bs*seq, num_joints*6]
                gt_pose_mat_flat = rotation_6d_to_matrix(gt_pose_6d_flat.reshape(-1, model.num_joints, 6)) # [bs*seq, num_joints, 3, 3]
                gt_pose_axis_angle_flat = matrix_to_axis_angle(gt_pose_mat_flat).reshape(bs * seq, -1) # [bs*seq, num_joints*3]
                gt_trans_flat = root_pos.reshape(bs*seq, 3) # [bs*seq, 3]
                
                body_model_output_gt = model.body_model(
                    root_orient=gt_pose_axis_angle_flat[:, :3],      # [bs*seq, 3]
                    pose_body=gt_pose_axis_angle_flat[:, 3:], # [bs*seq, (num_joints-1)*3]
                    trans=gt_trans_flat # 使用真实的trans
                )
                gt_j_global_flat = body_model_output_gt.Jtr[:, :model.num_joints, :] # [bs*seq, num_joints, 3]
                gt_j_local_flat = gt_j_global_flat - gt_trans_flat.unsqueeze(1)
                gt_j_seq = gt_j_local_flat.reshape(bs, seq, model.num_joints, 3) # [bs, seq, num_joints, 3]
                
                gt_leaf_pos = gt_j_seq[:, :, joint_set.leaf, :] # [bs, seq, n_leaf, 3]
                gt_full_pos = gt_j_seq[:, :, joint_set.full, :] # [bs, seq, n_full, 3] (joint_set.full is 1..21)
                # gt_reduced_motion = rotation_6d_to_matrix(motion.clone().reshape(bs, seq, model.num_joints, model.joint_dim))
                # gt_reduced_motion[:, :, joint_set.ignored, :] = torch.eye(3, device=gt_reduced_motion.device).repeat(bs, seq, joint_set.n_ignored, 1, 1)
                # gt_reduced_motion = matrix_to_rotation_6d(gt_reduced_motion).reshape(bs, seq, model.num_joints * model.joint_dim)

            
            # 计算损失
            # 1. 叶子节点位置损失 (L1)
            loss_leaf_pos = torch.nn.functional.l1_loss(pred_dict["pred_leaf_pos"], gt_leaf_pos)
            
            # 2. 全身关节位置损失 (L1, 加权)
            l1_diff_full = torch.abs(pred_dict["pred_full_pos"] - gt_full_pos) # [bs, seq, n_full, 3]
            weights_full = torch.ones_like(l1_diff_full)
            weights_full[:, :, [19, 20], :] = 4.0
            loss_full_pos = (l1_diff_full * weights_full).mean()
            
            # 3. 根关节速度损失 (L1)， 真值：全局根关节速度
            loss_root_vel = torch.nn.functional.l1_loss(pred_dict["root_vel"], root_vel)
            
            # 4. 姿态损失
            loss_rot = torch.nn.functional.mse_loss(pred_dict["motion"], motion)
            
            # 5. 根节点位置损失
            loss_root_pos = torch.nn.functional.mse_loss(pred_dict["root_pos"], root_pos)
            
            # 6. 手部接触损失 (BCE)
            loss_hand_contact = torch.nn.functional.binary_cross_entropy(pred_dict["pred_hand_contact_prob"], hand_contact_gt)
            
            # 7. 物体平移损失 (MSE or L1) - 这里使用MSE为例
            # 确保 obj_trans (真值) 已经移动到 device 并且存在
            obj_trans_gt = data_dict["obj_trans"] # 从已经 .to(device) 的 data_dict 中获取
            loss_obj_trans = torch.nn.functional.mse_loss(pred_dict["pred_obj_trans"], obj_trans_gt)
            
            # 8. 接触感知损失 (ContactAwareLoss) - 可选
            if use_contact_aware_loss and contact_loss_fn is not None:
                contact_loss, contact_loss_dict = contact_loss_fn(
                    pred_hand_pos=pred_dict["pred_hand_pos"],      # [bs, seq, 2, 3]
                    pred_obj_pos=pred_dict["pred_obj_trans"],      # [bs, seq, 3]
                    contact_probs=pred_dict["pred_hand_contact_prob"], # [bs, seq, 3]
                    training_step=training_step_count
                )
            else:
                contact_loss = torch.tensor(0.0, device=device)
                contact_loss_dict = {}
            
            # 9. 速度损失
            loss_obj_vel = torch.nn.functional.mse_loss(pred_dict["pred_obj_vel"], obj_vel) # 物体速度损失
            loss_leaf_vel = torch.nn.functional.mse_loss(pred_dict["pred_leaf_vel"], leaf_vel) # 叶子节点速度损失
            
            # 计算总损失（加权）
            if hasattr(cfg, 'loss_weights'):
                w_rot = cfg.loss_weights.rot if hasattr(cfg.loss_weights, 'rot') else 1.0
                w_root_pos = cfg.loss_weights.root_pos if hasattr(cfg.loss_weights, 'root_pos') else 0.1
                w_leaf_pos = cfg.loss_weights.leaf_pos if hasattr(cfg.loss_weights, 'leaf_pos') else 0.1
                w_full_pos = cfg.loss_weights.full_pos if hasattr(cfg.loss_weights, 'full_pos') else 0.1
                w_root_vel = cfg.loss_weights.root_vel if hasattr(cfg.loss_weights, 'root_vel') else 0.1
                w_hand_contact = cfg.loss_weights.hand_contact if hasattr(cfg.loss_weights, 'hand_contact') else 0.1
                w_obj_trans = cfg.loss_weights.obj_trans if hasattr(cfg.loss_weights, 'obj_trans') else 0.1
                w_contact_aware = cfg.loss_weights.contact_aware if hasattr(cfg.loss_weights, 'contact_aware') else 0.1
                w_obj_vel = cfg.loss_weights.obj_vel if hasattr(cfg.loss_weights, 'obj_vel') else 0.1
                w_leaf_vel = cfg.loss_weights.leaf_vel if hasattr(cfg.loss_weights, 'leaf_vel') else 0.1
            else:
                # 默认权重
                w_rot = 1.0
                w_root_pos = 0.1
                w_leaf_pos = 0.1
                w_full_pos = 0.1
                w_root_vel = 0.1
                w_hand_contact = 0.1
                w_obj_trans = 0.1
                w_contact_aware = 0.1
                w_obj_vel = 0.1
                w_leaf_vel = 0.1
            
            # 如果ContactAwareLoss被禁用，将其权重设为0
            if not use_contact_aware_loss:
                w_contact_aware = 0.0
            
            loss = (w_rot * loss_rot + 
                    w_root_pos * loss_root_pos + 
                    w_leaf_pos * loss_leaf_pos + 
                    w_full_pos * loss_full_pos + 
                    w_root_vel * loss_root_vel +
                    w_hand_contact * loss_hand_contact +
                    w_obj_trans * loss_obj_trans +
                    w_contact_aware * contact_loss +
                    w_obj_vel * loss_obj_vel +
                    w_leaf_vel * loss_leaf_vel)
            
            
            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update() 
            
            # 记录损失
            train_loss += loss.item()
            train_loss_rot += loss_rot.item()
            train_loss_root_pos += loss_root_pos.item()
            train_loss_leaf_pos += loss_leaf_pos.item()
            train_loss_full_pos += loss_full_pos.item()
            train_loss_root_vel += loss_root_vel.item()
            train_loss_hand_contact += loss_hand_contact.item()
            train_loss_obj_trans += loss_obj_trans.item()
            train_loss_contact_aware += contact_loss.item()
            train_loss_obj_vel += loss_obj_vel.item()
            train_loss_leaf_vel += loss_leaf_vel.item()
            
            # 更新tqdm描述
            postfix_dict = {
                'loss': loss.item(),
                'rot': loss_rot.item(), 
                'root_pos': loss_root_pos.item(), 
                'hand_contact': loss_hand_contact.item(),
                'obj_trans': loss_obj_trans.item(),
                'obj_vel': loss_obj_vel.item(),
                'leaf_vel': loss_leaf_vel.item(),
            }
            
            # 只有启用ContactAwareLoss时才显示相关指标
            if use_contact_aware_loss:
                postfix_dict['contact_aware'] = contact_loss.item()
                
            train_iter.set_postfix(postfix_dict)
            
            # 记录tensorboard
            if writer is not None:
                writer.add_scalar('train/loss', loss.item(), n_iter)
                writer.add_scalar('train/loss_rot', loss_rot.item(), n_iter)
                writer.add_scalar('train/loss_root_pos', loss_root_pos.item(), n_iter)
                writer.add_scalar('train/loss_leaf_pos', loss_leaf_pos.item(), n_iter)
                writer.add_scalar('train/loss_full_pos', loss_full_pos.item(), n_iter)
                writer.add_scalar('train/loss_root_vel', loss_root_vel.item(), n_iter)
                writer.add_scalar('train/loss_hand_contact', loss_hand_contact.item(), n_iter)
                writer.add_scalar('train/loss_obj_trans', loss_obj_trans.item(), n_iter)
                writer.add_scalar('train/loss_obj_vel', loss_obj_vel.item(), n_iter)
                writer.add_scalar('train/loss_leaf_vel', loss_leaf_vel.item(), n_iter)
                
                # 只有启用ContactAwareLoss时才记录相关指标
                if use_contact_aware_loss:
                    writer.add_scalar('train/loss_contact_aware', contact_loss.item(), n_iter)
                    # 记录ContactAwareLoss的详细信息
                    for loss_name, loss_value in contact_loss_dict.items():
                        if isinstance(loss_value, torch.Tensor):
                            writer.add_scalar(f'contact/{loss_name}', loss_value.item(), n_iter)
                        else:
                            writer.add_scalar(f'contact/{loss_name}', loss_value, n_iter)
                
            n_iter += 1
            training_step_count += 1

        # 计算平均训练损失
        train_loss /= len(train_loader)
        train_loss_rot /= len(train_loader)
        train_loss_root_pos /= len(train_loader)
        train_loss_leaf_pos /= len(train_loader)
        train_loss_full_pos /= len(train_loader)
        train_loss_root_vel /= len(train_loader)
        train_loss_hand_contact /= len(train_loader)
        train_loss_obj_trans /= len(train_loader)
        train_loss_contact_aware /= len(train_loader)
        train_loss_obj_vel /= len(train_loader)
        train_loss_leaf_vel /= len(train_loader)
        
        train_losses['loss'].append(train_loss)
        train_losses['loss_rot'].append(train_loss_rot)
        train_losses['loss_root_pos'].append(train_loss_root_pos)
        train_losses['loss_leaf_pos'].append(train_loss_leaf_pos)
        train_losses['loss_full_pos'].append(train_loss_full_pos)
        train_losses['loss_root_vel'].append(train_loss_root_vel)
        train_losses['loss_hand_contact'].append(train_loss_hand_contact)
        train_losses['loss_obj_trans'].append(train_loss_obj_trans)
        train_losses['loss_contact_aware'].append(train_loss_contact_aware)
        train_losses['loss_obj_vel'].append(train_loss_obj_vel)
        train_losses['loss_leaf_vel'].append(train_loss_leaf_vel)
        
        # 打印训练损失
        loss_msg = (f'Epoch {epoch}, Train Loss: {train_loss:.6f}, '
                   f'Rot Loss: {train_loss_rot:.6f}, '
                   f'Root Pos Loss: {train_loss_root_pos:.6f}, '
                   f'Leaf Pos Loss: {train_loss_leaf_pos:.6f}, '
                   f'Full Pos Loss: {train_loss_full_pos:.6f}, '
                   f'Root Vel Loss: {train_loss_root_vel:.6f}, '
                   f'Hand Contact Loss: {train_loss_hand_contact:.6f}, '
                   f'Obj Trans Loss: {train_loss_obj_trans:.6f}, '
                   f'Obj Vel Loss: {train_loss_obj_vel:.6f}, '
                   f'Leaf Vel Loss: {train_loss_leaf_vel:.6f}')
        
        # 只有启用ContactAwareLoss时才添加相关信息
        if use_contact_aware_loss:
            loss_msg += f', Contact Aware Loss: {train_loss_contact_aware:.6f}'
            
        print(loss_msg)

        # 每5个epoch进行一次测试和保存
        if epoch % 10 == 0 and test_loader is not None:
            # 测试阶段
            model.eval()
            test_loss = 0
            test_loss_rot = 0
            test_loss_root_pos = 0
            test_loss_leaf_pos = 0
            test_loss_full_pos = 0
            test_loss_root_vel = 0
            test_loss_hand_contact = 0
            test_loss_obj_trans = 0
            test_loss_contact_aware = 0
            test_loss_obj_vel = 0
            test_loss_leaf_vel = 0
            
            with torch.no_grad():
                test_iter = tqdm(test_loader, desc=f'Test Epoch {epoch}')
                for batch in test_iter:
                    # 准备数据
                    root_pos = batch["root_pos"].to(device)
                    motion = batch["motion"].to(device)
                    human_imu = batch["human_imu"].to(device)
                    root_vel = batch["root_vel"].to(device)
                    
                    bs, seq = human_imu.shape[:2]
                    obj_imu = batch.get("obj_imu", None)
                    obj_rot = batch.get("obj_rot", None)
                    obj_trans = batch.get("obj_trans", None) # 新增

                    if obj_imu is not None: obj_imu = obj_imu.to(device)
                    else: obj_imu = torch.zeros((bs, seq, 1, cfg.imu_dim if hasattr(cfg, 'imu_dim') else 9), device=device, dtype=human_imu.dtype)
                    
                    if obj_rot is not None: obj_rot = obj_rot.to(device)
                    else: obj_rot = torch.zeros((bs, seq, 6), device=device, dtype=motion.dtype)
                    
                    if obj_trans is not None: obj_trans = obj_trans.to(device)
                    else: obj_trans = torch.zeros((bs, seq, 3), device=device, dtype=root_pos.dtype)

                    # 获取速度真值（测试）
                    obj_vel_eval = batch.get("obj_vel", torch.zeros((bs, seq, 3), device=device, dtype=root_pos.dtype)).to(device) # [bs, seq, 3]
                    leaf_vel_eval = batch.get("leaf_vel", torch.zeros((bs, seq, joint_set.n_leaf, 3), device=device, dtype=root_pos.dtype)).to(device) # [bs, seq, n_leaf, 3]

                    bps_features = batch.get("bps_features", None)
                    if bps_features is not None: bps_features = bps_features.to(device)
                    
                    # --- 获取手部接触真值 (评估) ---
                    lhand_contact_gt_eval = batch.get("lhand_contact", torch.zeros((bs, seq), dtype=torch.bool, device=device)).bool().to(device)
                    rhand_contact_gt_eval = batch.get("rhand_contact", torch.zeros((bs, seq), dtype=torch.bool, device=device)).bool().to(device)
                    obj_contact_gt_eval = batch.get("obj_contact", torch.zeros((bs, seq), dtype=torch.bool, device=device)).bool().to(device)
                    hand_contact_gt_eval = torch.stack([lhand_contact_gt_eval, rhand_contact_gt_eval, obj_contact_gt_eval], dim=2).float()
                    # --- 结束获取手部接触真值 (评估) ---
                    
                    # 构建 data_dict_eval
                    data_dict_eval = {
                        "human_imu": human_imu,
                        "obj_imu": obj_imu,
                        "motion": motion,             # 新增
                        "root_pos": root_pos,           # 新增
                        "obj_rot": obj_rot,             # 新增
                        "obj_trans": obj_trans          # 新增
                    }
                    
                    if bps_features is not None:
                        data_dict_eval["bps_features"] = bps_features
                    
                    # TransPose直接输出预测结果
                    pred_dict = model(data_dict_eval)
                    
                    # 计算真实的关节位置 (用于评估)
                    gt_pose_6d_flat_eval = motion.reshape(-1, model.num_joints * model.joint_dim) # [bs*seq, num_joints*6]
                    gt_pose_mat_flat_eval = rotation_6d_to_matrix(gt_pose_6d_flat_eval.reshape(-1, model.num_joints, 6)) # [bs*seq, num_joints, 3, 3]
                    gt_pose_axis_angle_flat_eval = matrix_to_axis_angle(gt_pose_mat_flat_eval).reshape(bs * seq, -1) # [bs*seq, num_joints*3]
                    gt_trans_flat_eval = root_pos.reshape(bs*seq, 3) # [bs*seq, 3]
                    body_model_output_gt_eval = model.body_model(
                        root_orient=gt_pose_axis_angle_flat_eval[:, :3], # [bs*seq, 3]
                        pose_body=gt_pose_axis_angle_flat_eval[:, 3:], # [bs*seq, (num_joints-1)*3]
                        trans=gt_trans_flat_eval # [bs*seq, 3]
                    )

                    gt_j_global_flat_eval = body_model_output_gt_eval.Jtr[:, :model.num_joints, :] # [bs*seq, num_joints, 3]
                    gt_j_local_flat_eval = gt_j_global_flat_eval - gt_trans_flat_eval.unsqueeze(1)
                    gt_j_seq_eval = gt_j_local_flat_eval.reshape(bs, seq, model.num_joints, 3) # [bs, seq, num_joints, 3]

                    gt_leaf_pos_eval = gt_j_seq_eval[:, :, joint_set.leaf, :] # [bs, seq, n_leaf, 3]
                    gt_full_pos_eval = gt_j_seq_eval[:, :, joint_set.full, :] # [bs, seq, n_full, 3]
                    # gt_reduced_motion_eval = rotation_6d_to_matrix(motion.clone().reshape(bs, seq, model.num_joints, model.joint_dim))
                    # gt_reduced_motion_eval[:, :, joint_set.ignored, :] = torch.eye(3, device=gt_reduced_motion_eval.device).repeat(bs, seq, joint_set.n_ignored, 1, 1)
                    # gt_reduced_motion_eval = matrix_to_rotation_6d(gt_reduced_motion_eval).reshape(bs, seq, model.num_joints * model.joint_dim)
                    
                    # 计算评估指标
                    # 1. 姿态损失
                    loss_rot = torch.nn.functional.mse_loss(pred_dict["motion"], motion)
                    
                    # 2. 根节点位置损失
                    loss_root_pos = torch.nn.functional.mse_loss(pred_dict["root_pos"], root_pos)
                    
                    # 5. 叶子节点位置损失 (L1)
                    loss_leaf_pos = torch.nn.functional.l1_loss(pred_dict["pred_leaf_pos"], gt_leaf_pos_eval)
                    
                    # 6. 全身关节位置损失 (L1, 加权)
                    l1_diff_full_eval = torch.abs(pred_dict["pred_full_pos"] - gt_full_pos_eval) # [bs, seq, n_full, 3]
                    weights_full_eval = torch.ones_like(l1_diff_full_eval)
                    weights_full_eval[:, :, [19, 20], :] = 4.0
                    loss_full_pos = (l1_diff_full_eval * weights_full_eval).mean()
                    
                    # 7. 根关节速度损失 (L1)
                    loss_root_vel = torch.nn.functional.l1_loss(pred_dict["root_vel"], root_vel)
                    
                    # 8. 手部接触损失 (BCE)
                    loss_hand_contact = torch.nn.functional.binary_cross_entropy(pred_dict["pred_hand_contact_prob"], hand_contact_gt_eval)
                    
                    # 9. 物体平移损失 (MSE or L1)
                    obj_trans_gt_eval = data_dict_eval["obj_trans"] # 从已经 .to(device) 的 data_dict_eval 中获取
                    loss_obj_trans = torch.nn.functional.mse_loss(pred_dict["pred_obj_trans"], obj_trans_gt_eval)
                    
                    # 10. 接触感知损失 (ContactAwareLoss) - 可选
                    if use_contact_aware_loss and contact_loss_fn is not None:
                        contact_loss_eval, contact_loss_dict_eval = contact_loss_fn(
                            pred_hand_pos=pred_dict["pred_hand_pos"],      # [bs, seq, 2, 3]
                            pred_obj_pos=pred_dict["pred_obj_trans"],      # [bs, seq, 3]
                            contact_probs=pred_dict["pred_hand_contact_prob"], # [bs, seq, 3]
                            training_step=training_step_count  # 使用当前训练步数
                        )
                    else:
                        contact_loss_eval = torch.tensor(0.0, device=device)
                        contact_loss_dict_eval = {}
                    
                    # 11. 速度损失（评估）
                    loss_obj_vel_eval = torch.nn.functional.mse_loss(pred_dict["pred_obj_vel"], obj_vel_eval) # 物体速度损失
                    loss_leaf_vel_eval = torch.nn.functional.mse_loss(pred_dict["pred_leaf_vel"], leaf_vel_eval) # 叶子节点速度损失
                    
                    # 计算总损失（加权）- 使用与训练相同的权重进行评估
                    if hasattr(cfg, 'loss_weights'):
                        w_rot = cfg.loss_weights.rot if hasattr(cfg.loss_weights, 'rot') else 1.0
                        w_root_pos = cfg.loss_weights.root_pos if hasattr(cfg.loss_weights, 'root_pos') else 0.1
                        w_leaf_pos = cfg.loss_weights.leaf_pos if hasattr(cfg.loss_weights, 'leaf_pos') else 0.1
                        w_full_pos = cfg.loss_weights.full_pos if hasattr(cfg.loss_weights, 'full_pos') else 0.1
                        w_root_vel = cfg.loss_weights.root_vel if hasattr(cfg.loss_weights, 'root_vel') else 0.1
                        w_hand_contact = cfg.loss_weights.hand_contact if hasattr(cfg.loss_weights, 'hand_contact') else 0.1
                        w_obj_trans = cfg.loss_weights.obj_trans if hasattr(cfg.loss_weights, 'obj_trans') else 0.1
                        w_contact_aware = cfg.loss_weights.contact_aware if hasattr(cfg.loss_weights, 'contact_aware') else 0.1
                        w_obj_vel = cfg.loss_weights.obj_vel if hasattr(cfg.loss_weights, 'obj_vel') else 0.1
                        w_leaf_vel = cfg.loss_weights.leaf_vel if hasattr(cfg.loss_weights, 'leaf_vel') else 0.1
                    else:
                        w_rot = 1.0
                        w_root_pos = 0.1
                        w_leaf_pos = 0.1
                        w_full_pos = 0.1
                        w_root_vel = 0.1
                        w_hand_contact = 0.1
                        w_obj_trans = 0.1
                        w_contact_aware = 0.1
                        w_obj_vel = 0.1
                        w_leaf_vel = 0.1
                    
                    # 如果ContactAwareLoss被禁用，将其权重设为0
                    if not use_contact_aware_loss:
                        w_contact_aware = 0.0
                    
                    test_metric = (w_rot * loss_rot + 
                                   w_root_pos * loss_root_pos + 
                                   w_leaf_pos * loss_leaf_pos + 
                                   w_full_pos * loss_full_pos + 
                                   w_root_vel * loss_root_vel +
                                   w_hand_contact * loss_hand_contact +
                                   w_obj_trans * loss_obj_trans +
                                   w_contact_aware * contact_loss_eval +
                                   w_obj_vel * loss_obj_vel_eval +
                                   w_leaf_vel * loss_leaf_vel_eval)
                    
                    # # --- 计算手部细化评估指标 --- #
                    # gt_wrist_l_6d_eval = motion.reshape(bs * seq, model.num_joints, 6)[:, model.wrist_l_idx, :] # [bs*seq, 6]
                    # gt_wrist_r_6d_eval = motion.reshape(bs * seq, model.num_joints, 6)[:, model.wrist_r_idx, :] # [bs*seq, 6]
                    # gt_wrist_6d_eval = torch.stack([gt_wrist_l_6d_eval, gt_wrist_r_6d_eval], dim=1) # [bs*seq, 2, 6]
                    # pred_wrist_refined_6d_flat_eval = pred_dict["pred_wrist_refined_6d"].reshape(bs * seq, 2, 6) # [bs*seq, 2, 6]
                    # pred_wrist_delta_6d_flat_eval = pred_dict["pred_wrist_delta_6d"].reshape(bs * seq, 2, 6) # [bs*seq, 2, 6]
                    
                    # R_refined_pred_eval = rotation_6d_to_matrix(pred_wrist_refined_6d_flat_eval.reshape(-1, 6)) # [bs*seq*2, 3, 3]
                    # R_gt_eval = rotation_6d_to_matrix(gt_wrist_6d_eval.reshape(-1, 6)) # [bs*seq*2, 3, 3]
                    # loss_refine_eval = model._so3_geodesic_distance(R_refined_pred_eval, R_gt_eval).mean()
                    
                    # dR_pred_eval = rotation_6d_to_matrix(pred_wrist_delta_6d_flat_eval.reshape(-1, 6)) # [bs*seq*2, 3, 3]
                    # identity_eval = torch.eye(3, device=dR_pred_eval.device).unsqueeze(0).expand_as(dR_pred_eval)
                    # loss_reg_eval = ((dR_pred_eval - identity_eval)**2).mean()
                    
                    # # 添加到总评估指标
                    # test_metric += w_hand_refine * loss_refine_eval + w_hand_reg * loss_reg_eval
                    
                    # 记录损失
                    test_loss += test_metric.item()
                    test_loss_rot += loss_rot.item()
                    test_loss_root_pos += loss_root_pos.item()
                    test_loss_leaf_pos += loss_leaf_pos.item()
                    test_loss_full_pos += loss_full_pos.item()
                    test_loss_root_vel += loss_root_vel.item()
                    test_loss_hand_contact += loss_hand_contact.item()
                    test_loss_obj_trans += loss_obj_trans.item()
                    test_loss_contact_aware += contact_loss_eval.item()
                    test_loss_obj_vel += loss_obj_vel_eval.item()
                    test_loss_leaf_vel += loss_leaf_vel_eval.item()
                    
                    # 更新tqdm描述
                    test_postfix_dict = {
                        'test_metric': test_metric.item(),
                        'loss_rot': loss_rot.item(),
                        'loss_root_pos': loss_root_pos.item(),
                        'loss_leaf_pos': loss_leaf_pos.item(),
                        'loss_full_pos': loss_full_pos.item(),
                        'loss_root_vel': loss_root_vel.item(),
                        'hand_contact': loss_hand_contact.item(),
                        'obj_trans': loss_obj_trans.item(),
                        'obj_vel': loss_obj_vel_eval.item(),
                        'leaf_vel': loss_leaf_vel_eval.item()
                    }
                    
                    # 只有启用ContactAwareLoss时才显示相关指标
                    if use_contact_aware_loss:
                        test_postfix_dict['contact_aware'] = contact_loss_eval.item()
                        
                    test_iter.set_postfix(test_postfix_dict)
            
            # 计算平均测试损失
            test_loss /= len(test_loader)
            test_loss_rot /= len(test_loader)
            test_loss_root_pos /= len(test_loader)
            test_loss_leaf_pos /= len(test_loader)
            test_loss_full_pos /= len(test_loader)
            test_loss_root_vel /= len(test_loader)
            test_loss_hand_contact /= len(test_loader)
            test_loss_obj_trans /= len(test_loader)
            test_loss_contact_aware /= len(test_loader)
            test_loss_obj_vel /= len(test_loader)
            test_loss_leaf_vel /= len(test_loader)
            
            test_losses['loss'].append(test_loss)
            test_losses['loss_rot'].append(test_loss_rot)
            test_losses['loss_root_pos'].append(test_loss_root_pos)
            test_losses['loss_leaf_pos'].append(test_loss_leaf_pos)
            test_losses['loss_full_pos'].append(test_loss_full_pos)
            test_losses['loss_root_vel'].append(test_loss_root_vel)
            test_losses['loss_hand_contact'].append(test_loss_hand_contact)
            test_losses['loss_obj_trans'].append(test_loss_obj_trans)
            test_losses['loss_contact_aware'].append(test_loss_contact_aware)
            test_losses['loss_obj_vel'].append(test_loss_obj_vel)
            test_losses['loss_leaf_vel'].append(test_loss_leaf_vel)
            
            # 打印测试损失
            test_loss_msg = (f'Epoch {epoch}, Test Metric: {test_loss:.6f}, '
                            f'Rot Loss: {test_loss_rot:.6f}, '
                            f'Root Pos Loss: {test_loss_root_pos:.6f}, '
                            f'Leaf Pos Loss: {test_loss_leaf_pos:.6f}, '
                            f'Full Pos Loss: {test_loss_full_pos:.6f}, '
                            f'Root Vel Loss: {test_loss_root_vel:.6f}, '
                            f'Hand Contact Loss: {test_loss_hand_contact:.6f}, '
                            f'Obj Trans Loss: {test_loss_obj_trans:.6f}, '
                            f'Obj Vel Loss: {test_loss_obj_vel:.6f}, '
                            f'Leaf Vel Loss: {test_loss_leaf_vel:.6f}')
            
            # 只有启用ContactAwareLoss时才添加相关信息
            if use_contact_aware_loss:
                test_loss_msg += f', Contact Aware Loss: {test_loss_contact_aware:.6f}'
                
            print(test_loss_msg)
            
            if writer is not None:
                writer.add_scalar('test/metric', test_loss, n_iter)
                writer.add_scalar('test/loss_rot', test_loss_rot, n_iter)
                writer.add_scalar('test/loss_root_pos', test_loss_root_pos, n_iter)
                writer.add_scalar('test/loss_leaf_pos', test_loss_leaf_pos, n_iter)
                writer.add_scalar('test/loss_full_pos', test_loss_full_pos, n_iter)
                writer.add_scalar('test/loss_root_vel', test_loss_root_vel, n_iter)
                writer.add_scalar('test/loss_hand_contact', test_loss_hand_contact, n_iter)
                writer.add_scalar('test/loss_obj_trans', test_loss_obj_trans, n_iter)
                writer.add_scalar('test/loss_obj_vel', test_loss_obj_vel, n_iter)
                writer.add_scalar('test/loss_leaf_vel', test_loss_leaf_vel, n_iter)
                
                # 只有启用ContactAwareLoss时才记录相关指标
                if use_contact_aware_loss:
                    writer.add_scalar('test/loss_contact_aware', test_loss_contact_aware, n_iter)
            
            # 保存最佳模型
            if test_loss < best_loss and not cfg.debug:
                best_loss = test_loss
                save_path = os.path.join(save_dir, f'epoch_{epoch}_best.pt')
                print(f'Saving best model to {save_path}')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, save_path)
                
        
        # 定期保存模型
        if epoch % 50 == 0 and not cfg.debug:
            save_path = os.path.join(save_dir, f'epoch_{epoch}.pt')
            print(f'Saving model to {save_path}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, save_path)

        # 更新学习率
        scheduler.step()
    
    # 保存最终模型
    if not cfg.debug:
        final_path = os.path.join(save_dir, 'final.pt')
        print(f'Saving final model to {final_path}')
        torch.save({
            'epoch': max_epoch - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, final_path)
        
        # 保存最终的损失曲线
        loss_curves = {
            'train_losses': train_losses,
            'test_losses': test_losses,
        }
        with open(os.path.join(save_dir, 'loss_curves.pkl'), 'wb') as f:
            pickle.dump(loss_curves, f)
    
    # 如果使用tensorboard，保存最终指标并关闭writer
    if writer is not None:
        writer.add_scalar('final/train_loss', train_loss, max_epoch)
        if test_loader is not None:
            writer.add_scalar('final/test_loss', test_loss, max_epoch)
            writer.add_scalar('final/best_test_loss', best_loss, max_epoch)
        log_dir = writer.log_dir
        writer.close()
        print(f'TensorBoard logs saved. You can view them with: tensorboard --logdir {os.path.dirname(log_dir)}')
    
    # 如果是超参数搜索，返回最佳测试损失
    if trial is not None:
        return best_loss
        
    return model, optimizer


def load_transpose_model(cfg, checkpoint_path):
    """
    加载TransPose模型
    
    Args:
        cfg: 配置信息
        checkpoint_path: 模型检查点路径
        
    Returns:
        model: 加载的模型
    """
    device = torch.device(f'cuda:{cfg.gpus[0]}' if torch.cuda.is_available() else 'cpu')
    model = TransPoseNet(cfg).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f'Loaded TransPose model from {checkpoint_path}, epoch {checkpoint["epoch"]}')
    
    return model