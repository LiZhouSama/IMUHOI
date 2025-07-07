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


def get_training_stage(epoch, staged_training_config=None, is_debug=False):
    """
    根据epoch判断当前训练阶段
    
    Args:
        epoch: 当前epoch
        staged_training_config: 分阶段训练配置
        is_debug: 是否为debug模式
    
    Returns:
        dict: 包含当前阶段信息的字典
    """
    if not staged_training_config or not staged_training_config.get('enabled', False):
        # 如果没有启用分阶段训练，默认训练所有模块
        return {
            'name': 'all_modules',
            'active_modules': ['velocity_contact', 'human_pose', 'object_trans'],
            'frozen_modules': [],
            'datasets': ['mixed'],
            'use_object_data': True
        }
    
    # 根据是否为debug模式选择相应的stages配置
    if is_debug and 'debug_stages' in staged_training_config:
        stages = staged_training_config.get('debug_stages', [])
        print(f"Debug模式: 使用debug_stages配置 (共{len(stages)}个阶段)")
    else:
        stages = staged_training_config.get('stages', [])
    
    for stage in stages:
        start_epoch, end_epoch = stage['epochs']
        if start_epoch <= epoch <= end_epoch:
            return {
                'name': stage['name'],
                'active_modules': stage['modules'],
                'frozen_modules': [],
                'datasets': stage['datasets'],
                'use_object_data': 'omomo' in stage['datasets'] or 'mixed' in stage['datasets']
            }
    
    # 如果没有匹配的阶段，默认返回最后一个阶段
    if stages:
        last_stage = stages[-1]
        return {
            'name': last_stage['name'],
            'active_modules': last_stage['modules'],
            'frozen_modules': [],
            'datasets': last_stage['datasets'],
            'use_object_data': 'omomo' in last_stage['datasets'] or 'mixed' in last_stage['datasets']
        }
    
    # 兜底情况
    return {
        'name': 'all_modules',
        'active_modules': ['velocity_contact', 'human_pose', 'object_trans'],
        'frozen_modules': [],
        'datasets': ['mixed'],
        'use_object_data': True
    }


def configure_training_modules(model, stage_info):
    """
    配置训练模块：激活指定模块，冻结其他模块
    
    Args:
        model: TransPose模型
        stage_info: 当前阶段信息
    """
    actual_model = get_actual_model(model)
    actual_model.configure_training_modules(
        active_modules=stage_info['active_modules'],
        frozen_modules=stage_info.get('frozen_modules', [])
    )
    
    print(f"当前训练阶段: {stage_info['name']}")
    print(f"激活模块: {stage_info['active_modules']}")
    if stage_info.get('frozen_modules'):
        print(f"冻结模块: {stage_info['frozen_modules']}")
    print(f"使用数据集: {stage_info['datasets']}")
    print(f"使用物体数据: {stage_info['use_object_data']}")


def compute_stage_specific_loss(pred_dict, batch, stage_info, cfg, training_step_count, contact_loss_fn, device, model=None):
    """
    根据训练阶段计算相应的损失
    
    Args:
        pred_dict: 模型预测结果
        batch: 批次数据
        stage_info: 当前阶段信息
        cfg: 配置
        training_step_count: 训练步数
        contact_loss_fn: 接触感知损失函数
        device: 设备
        model: 模型实例（用于获取body_model等）
    
    Returns:
        tuple: (总损失, 损失字典, 加权损失字典)
    """
    stage_name = stage_info['name']
    active_modules = stage_info['active_modules']
    use_object_data = stage_info['use_object_data']
    
    # 从batch中获取真值数据
    bs, seq = batch["human_imu"].shape[:2]
    root_pos = batch["root_pos"].to(device)
    motion = batch["motion"].to(device)
    root_vel = batch["root_vel"].to(device)
    
    # 获取速度真值
    obj_vel = batch.get("obj_vel", torch.zeros((bs, seq, 3), device=device, dtype=root_pos.dtype)).to(device)
    leaf_vel = batch.get("leaf_vel", torch.zeros((bs, seq, joint_set.n_leaf, 3), device=device, dtype=root_pos.dtype)).to(device)
    
    # 获取接触真值
    lhand_contact_gt = batch.get("lhand_contact", torch.zeros((bs, seq), dtype=torch.bool, device=device)).bool().to(device)
    rhand_contact_gt = batch.get("rhand_contact", torch.zeros((bs, seq), dtype=torch.bool, device=device)).bool().to(device)
    obj_contact_gt = batch.get("obj_contact", torch.zeros((bs, seq), dtype=torch.bool, device=device)).bool().to(device)
    hand_contact_gt = torch.stack([lhand_contact_gt, rhand_contact_gt, obj_contact_gt], dim=2).float()
    
    lfoot_contact_gt = batch.get("lfoot_contact").float().to(device)
    rfoot_contact_gt = batch.get("rfoot_contact").float().to(device)
    foot_contact_gt = torch.stack([lfoot_contact_gt, rfoot_contact_gt], dim=2)
    
    # 处理物体数据
    obj_trans = batch.get("obj_trans", torch.zeros((bs, seq, 3), device=device, dtype=root_pos.dtype)).to(device)
    
    # 初始化损失字典
    loss_dict = {}
    
    # 根据激活的模块计算相应的损失
    if 'velocity_contact' in active_modules:
        # 速度估计损失
        if use_object_data:
            loss_dict['obj_vel'] = torch.nn.functional.mse_loss(pred_dict["pred_obj_vel"], obj_vel)
        else:
            loss_dict['obj_vel'] = torch.tensor(0.0, device=device)
        
        loss_dict['leaf_vel'] = torch.nn.functional.mse_loss(pred_dict["pred_leaf_vel"], leaf_vel)
        
        # 手部接触损失
        if use_object_data:
            loss_dict['hand_contact'] = torch.nn.functional.binary_cross_entropy(pred_dict["pred_hand_contact_prob"], hand_contact_gt)
        else:
            loss_dict['hand_contact'] = torch.tensor(0.0, device=device)
    
    if 'human_pose' in active_modules:
        # 需要计算真实的关节位置用于姿态损失
        actual_model = get_actual_model(model) if model is not None else None
        # 计算真实关节位置（这部分逻辑来自原始训练代码）
        with torch.no_grad():
            gt_pose_6d_flat = motion.reshape(-1, actual_model.num_joints * actual_model.joint_dim)
            gt_pose_mat_flat = rotation_6d_to_matrix(gt_pose_6d_flat.reshape(-1, actual_model.num_joints, 6))
            gt_pose_axis_angle_flat = matrix_to_axis_angle(gt_pose_mat_flat).reshape(bs * seq, -1)
            gt_trans_flat = root_pos.reshape(bs*seq, 3)
            
            # 访问human_pose_module中的body_model
            body_model_output_gt = actual_model.human_pose_module.body_model(
                root_orient=gt_pose_axis_angle_flat[:, :3],
                pose_body=gt_pose_axis_angle_flat[:, 3:],
                trans=gt_trans_flat
            )
            gt_j_global_flat = body_model_output_gt.Jtr[:, :actual_model.num_joints, :]
            gt_j_local_flat = gt_j_global_flat - gt_trans_flat.unsqueeze(1)
            gt_j_seq = gt_j_local_flat.reshape(bs, seq, actual_model.num_joints, 3)
            
            gt_leaf_pos = gt_j_seq[:, :, joint_set.leaf, :]
            gt_full_pos = gt_j_seq[:, :, joint_set.full, :]
        
        # 姿态损失
        loss_dict['rot'] = torch.nn.functional.mse_loss(pred_dict["motion"], motion)
        
        # 根节点位置和速度损失
        loss_dict['root_pos'] = torch.nn.functional.mse_loss(pred_dict["root_pos"], root_pos)
        loss_dict['root_vel'] = torch.nn.functional.l1_loss(pred_dict["root_vel"], root_vel)
        
        # 关节位置损失
        loss_dict['leaf_pos'] = torch.nn.functional.l1_loss(pred_dict["pred_leaf_pos"], gt_leaf_pos)

        # 全身节点位置损失
        l1_diff_full = torch.abs(pred_dict["pred_full_pos"] - gt_full_pos)
        weights_full = torch.ones_like(l1_diff_full)
        weights_full[:, :, [19, 20], :] = 4.0   # 手部loss增强
        loss_dict['full_pos'] = (l1_diff_full * weights_full).mean()
        
        # 足部接触损失
        loss_dict['foot_contact'] = torch.nn.functional.binary_cross_entropy(pred_dict["contact_probability"], foot_contact_gt)
    
    if 'object_trans' in active_modules and use_object_data:
        # 物体平移损失
        loss_dict['obj_trans'] = torch.nn.functional.mse_loss(pred_dict["pred_obj_trans"], obj_trans)
        
        # 接触感知损失
        use_contact_aware_loss = getattr(cfg, 'use_contact_aware_loss', False)
        if use_contact_aware_loss and contact_loss_fn is not None and "pred_hand_pos" in pred_dict:
            contact_loss, contact_loss_dict = contact_loss_fn(
                pred_hand_pos=pred_dict["pred_hand_pos"],
                pred_obj_pos=pred_dict["pred_obj_trans"],
                contact_probs=pred_dict["pred_hand_contact_prob"],
                training_step=training_step_count
            )
            loss_dict['contact_aware'] = contact_loss
        else:
            loss_dict['contact_aware'] = torch.tensor(0.0, device=device)
    else:
        loss_dict['obj_trans'] = torch.tensor(0.0, device=device)
        loss_dict['contact_aware'] = torch.tensor(0.0, device=device)
    
    # 设置默认权重
    default_weights = {
        'rot': 10, 'root_pos': 10, 'leaf_pos': 1.0, 'full_pos': 1.0,
        'root_vel': 10, 'hand_contact': 1, 'obj_trans': 10, 'contact_aware': 1.0,
        'obj_vel': 1, 'leaf_vel': 10, 'foot_contact': 1
    }
    
    # 获取配置的权重
    weights = {}
    if hasattr(cfg, 'loss_weights'):
        for key in default_weights:
            weights[key] = getattr(cfg.loss_weights, key, default_weights[key])
    else:
        weights = default_weights
    
    # 根据数据集类型动态调整足部接触损失权重
    dataset_types = stage_info.get('datasets', ['mixed'])
    if 'amass' in dataset_types and 'omomo' not in dataset_types:
        weights['foot_contact'] *= 0.7
    elif 'omomo' in dataset_types and 'amass' not in dataset_types:
        weights['foot_contact'] *= 1
    elif 'mixed' in dataset_types:
        # 混合数据：使用中等权重
        weights['foot_contact'] *= 0.7
    
    # 根据训练阶段调整权重
    if stage_name == 'velocity_contact':
        # 第一阶段：只关注速度和接触
        for key in weights:
            if key not in ['obj_vel', 'leaf_vel', 'hand_contact']:
                weights[key] = 0.0
    elif stage_name == 'human_pose':
        # 第二阶段：只关注人体相关
        for key in weights:
            if key not in ['rot', 'root_pos', 'leaf_pos', 'full_pos', 'root_vel', 'foot_contact']:
                weights[key] = 0.0
    elif stage_name == 'object_trans':
        # 第三阶段：只关注物体平移
        for key in weights:
            if key not in ['obj_trans', 'contact_aware']:
                weights[key] = 0.0
    # 联合训练阶段：使用所有权重
    
    # 计算加权总损失
    total_loss = torch.tensor(0.0, device=device)
    weighted_losses = {}
    
    for key, loss_value in loss_dict.items():
        weight = weights.get(key, 0.0)
        weighted_loss = weight * loss_value
        weighted_losses[key] = weighted_loss
        total_loss += weighted_loss
    
    return total_loss, loss_dict, weighted_losses


def get_actual_model(model):
    """
    获取实际的模型，处理DataParallel包装
    
    Args:
        model: 可能被DataParallel包装的模型
    
    Returns:
        实际的模型实例
    """
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def save_stage_checkpoint(model, optimizer, epoch, stage_info, save_dir, loss, prefix="stage"):
    """
    保存阶段检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        stage_info: 阶段信息
        save_dir: 保存目录
        loss: 损失值
        prefix: 文件前缀
    """
    if save_dir is None:
        return
    
    checkpoint_name = f"{prefix}_{stage_info['name']}_epoch_{epoch}.pt"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    
    model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    
    torch.save({
        'epoch': epoch,
        'stage_info': stage_info,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    print(f"保存阶段检查点: {checkpoint_path}")


def do_train_imu_TransPose(cfg, train_loader, test_loader=None, trial=None, model=None, optimizer=None):
    """
    训练IMU到全身姿态及物体变换的TransPose模型，支持分阶段训练
    
    Args:
        cfg: 配置信息
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        trial: Optuna试验（如果使用超参数搜索）
        model: 预训练模型（如果有）
        optimizer: 预训练模型的优化器（如果有）
    """
    # 初始化配置
    device = torch.device(cfg.device if hasattr(cfg, 'device') else f'cuda:{cfg.gpus[0]}' if torch.cuda.is_available() else 'cpu')
    model_name = cfg.model_name
    use_tensorboard = cfg.use_tensorboard and not cfg.debug
    use_multi_gpu = getattr(cfg, 'use_multi_gpu', False) and len(cfg.gpus) > 1
    pose_rep = 'rot6d'
    max_epoch = cfg.epoch
    save_dir = cfg.save_dir
    scaler = GradScaler()

    # 获取分阶段训练配置
    staged_training_config = getattr(cfg, 'staged_training', None)

    # 打印训练配置
    print(f'Training: {model_name} (using TransPose), pose_rep: {pose_rep}')
    print(f'use_tensorboard: {use_tensorboard}, device: {device}')
    print(f'use_multi_gpu: {use_multi_gpu}, gpus: {cfg.gpus if use_multi_gpu else [cfg.gpus[0]]}')
    print(f'max_epoch: {max_epoch}')
    
    if staged_training_config and staged_training_config.get('enabled', False):
        print("启用分阶段训练:")
        # 根据是否为debug模式显示相应的stages
        if cfg.debug and 'debug_stages' in staged_training_config:
            print("  [Debug模式] 使用debug_stages配置:")
            for stage in staged_training_config.get('debug_stages', []):
                print(f"    {stage['name']}: epochs {stage['epochs']}, modules: {stage['modules']}, datasets: {stage['datasets']}")
        else:
            print("  [正常模式] 使用stages配置:")
            for stage in staged_training_config.get('stages', []):
                print(f"    {stage['name']}: epochs {stage['epochs']}, modules: {stage['modules']}, datasets: {stage['datasets']}")
    
    if not cfg.debug:
        os.makedirs(save_dir, exist_ok=True)

    # 初始化模型（如果没有提供预训练模型）
    if model is None:
        model = TransPoseNet(cfg)
        print(f'Initialized TransPose model with {sum(p.numel() for p in model.parameters())} parameters')
        model = model.to(device)
        
        # 多GPU包装
        if use_multi_gpu:
            print(f'Wrapping model with DataParallel for GPUs: {cfg.gpus}')
            model = torch.nn.DataParallel(model, device_ids=cfg.gpus)

        # 设置优化器（如果没有提供预训练优化器）
        if optimizer is None:
            optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        print(f'Using pre-trained TransPose model with {sum(p.numel() for p in model.parameters())} parameters')
        model = model.to(device)
        
        # 多GPU包装
        if use_multi_gpu:
            print(f'Wrapping pre-trained model with DataParallel for GPUs: {cfg.gpus}')
            model = torch.nn.DataParallel(model, device_ids=cfg.gpus)
        
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
    training_step_count = 0
    current_stage_info = None

    # 如果启用分阶段训练且没有提供初始数据加载器，创建第一个阶段的数据加载器
    if train_loader is None and staged_training_config and staged_training_config.get('enabled', False):
        initial_stage_info = get_training_stage(0, staged_training_config, is_debug=cfg.debug)
        from train_transpose import create_staged_dataloaders
        train_loader, test_loader = create_staged_dataloaders(cfg, initial_stage_info)
        
        if train_loader is None:
            print("错误: 无法创建初始阶段的数据加载器")
            return model, optimizer
        
        print(f"已创建初始阶段 '{initial_stage_info['name']}' 的数据加载器")
        
        # 设置当前阶段信息并配置训练模块，避免在第一个epoch时重复创建数据加载器
        current_stage_info = initial_stage_info
        configure_training_modules(model, current_stage_info)
        
        # 重新创建优化器（只优化激活的参数）
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma)
        
        print(f"初始化优化器，可训练参数数量: {sum(p.numel() for p in trainable_params)}")

    for epoch in range(max_epoch):
        # 确定当前训练阶段
        new_stage_info = get_training_stage(epoch, staged_training_config, is_debug=cfg.debug)
        
        # 检查是否切换了训练阶段
        if current_stage_info is None or new_stage_info['name'] != current_stage_info['name']:
            print(f"\n=== Epoch {epoch}: 切换到训练阶段 '{new_stage_info['name']}' ===")
            
            # 保存上一个阶段的检查点（如果有）
            if current_stage_info is not None and not cfg.debug:
                save_stage_checkpoint(model, optimizer, epoch-1, current_stage_info, save_dir, best_loss, "stage_end")
            
            # 配置新阶段的训练模块
            configure_training_modules(model, new_stage_info)
            current_stage_info = new_stage_info
            
            # 重新创建优化器（只优化激活的参数）
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma)
            
            print(f"重新创建优化器，可训练参数数量: {sum(p.numel() for p in trainable_params)}")
            
            # 创建或更新数据加载器（如果启用分阶段训练）
            if staged_training_config and staged_training_config.get('enabled', False):
                from train_transpose import create_staged_dataloaders
                new_train_loader, new_test_loader = create_staged_dataloaders(cfg, new_stage_info)
                
                # 如果成功创建了新的数据加载器，则更新
                if new_train_loader is not None:
                    train_loader = new_train_loader
                    test_loader = new_test_loader
                    print(f"已为阶段 '{new_stage_info['name']}' 创建新的数据加载器")
                elif train_loader is None:
                    print(f"错误: 无法为阶段 '{new_stage_info['name']}' 创建数据加载器")
                    return model, optimizer
        
        # 如果 current_stage_info 仍然是 None（传统训练模式），设置一个默认的阶段信息
        if current_stage_info is None:
            current_stage_info = get_training_stage(epoch, staged_training_config, is_debug=cfg.debug)
            configure_training_modules(model, current_stage_info)

        # 训练阶段
        model.train()
        train_loss = 0
        stage_losses = defaultdict(float)
        
        train_iter = tqdm(train_loader, desc=f'Epoch {epoch} - {current_stage_info["name"]}')
        
        for batch in train_iter:
            # 准备数据
            root_pos = batch["root_pos"].to(device)
            motion = batch["motion"].to(device)
            human_imu = batch["human_imu"].to(device)
            root_vel = batch["root_vel"].to(device)
            
            # 处理可选的物体数据
            bs, seq = human_imu.shape[:2]
            obj_imu = batch.get("obj_imu", None)
            obj_rot = batch.get("obj_rot", None)
            obj_trans = batch.get("obj_trans", None)
            
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
            
            # 构建传递给模型的data_dict
            data_dict = {
                "human_imu": human_imu,
                "obj_imu": obj_imu,
                "motion": motion,
                "root_pos": root_pos,
                "obj_rot": obj_rot,
                "obj_trans": obj_trans
            }
            
            # 前向传播
            optimizer.zero_grad()
            
            # 根据当前阶段决定是否使用物体数据
            use_object_data = current_stage_info['use_object_data']
            pred_dict = model(data_dict, use_object_data=use_object_data)
            
            # 计算阶段特定的损失
            total_loss, loss_dict, weighted_losses = compute_stage_specific_loss(
                pred_dict, batch, current_stage_info, cfg, training_step_count, contact_loss_fn, device, model
            )
            
            # 反向传播和优化
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update() 
            
            # 记录损失
            train_loss += total_loss.item()
            for key, loss_value in weighted_losses.items():
                if isinstance(loss_value, torch.Tensor):
                    stage_losses[key] += loss_value.item()
                else:
                    stage_losses[key] += loss_value
            
            # 更新tqdm描述
            postfix_dict = {'loss': total_loss.item()}
            for key, loss_value in loss_dict.items():
                if isinstance(loss_value, torch.Tensor) and loss_value.item() != 0.0:
                    postfix_dict[key] = loss_value.item()
            train_iter.set_postfix(postfix_dict)
            
            # 记录tensorboard
            if writer is not None:
                writer.add_scalar('train/total_loss', total_loss.item(), n_iter)
                writer.add_scalar('train/stage', epoch, n_iter)  # 记录当前阶段
                for key, loss_value in loss_dict.items():
                    if isinstance(loss_value, torch.Tensor):
                        writer.add_scalar(f'train_raw/loss_{key}_raw', loss_value.item(), n_iter)
                for key, weighted_loss in weighted_losses.items():
                    if isinstance(weighted_loss, torch.Tensor):
                        writer.add_scalar(f'train/loss_{key}', weighted_loss.item(), n_iter)
            
            n_iter += 1
            training_step_count += 1

        # 计算平均训练损失
        train_loss /= len(train_loader)
        for key in stage_losses:
            stage_losses[key] /= len(train_loader)
            train_losses[key].append(stage_losses[key])
        
        train_losses['total_loss'].append(train_loss)
        
        # 打印训练损失
        loss_msg = f'Epoch {epoch}, Stage: {current_stage_info["name"]}, Train Loss: {train_loss:.6f}'
        for key, loss_value in stage_losses.items():
            if loss_value != 0.0:
                loss_msg += f', {key}: {loss_value:.6f}'
        print(loss_msg)

        # 每10个epoch进行一次测试和保存
        if epoch % 10 == 0 and test_loader is not None:
            # 测试阶段
            model.eval()
            test_loss = 0
            test_stage_losses = defaultdict(float)
            
            with torch.no_grad():
                test_iter = tqdm(test_loader, desc=f'Test Epoch {epoch} - {current_stage_info["name"]}')
                for batch in test_iter:
                    # 准备测试数据（与训练类似）
                    root_pos = batch["root_pos"].to(device)
                    motion = batch["motion"].to(device)
                    human_imu = batch["human_imu"].to(device)
                    root_vel = batch["root_vel"].to(device)
                    
                    bs, seq = human_imu.shape[:2]
                    obj_imu = batch.get("obj_imu", None)
                    obj_rot = batch.get("obj_rot", None)
                    obj_trans = batch.get("obj_trans", None)

                    if obj_imu is not None: obj_imu = obj_imu.to(device)
                    else: obj_imu = torch.zeros((bs, seq, 1, cfg.imu_dim if hasattr(cfg, 'imu_dim') else 9), device=device, dtype=human_imu.dtype)
                    if obj_rot is not None: obj_rot = obj_rot.to(device)
                    else: obj_rot = torch.zeros((bs, seq, 6), device=device, dtype=motion.dtype)
                    if obj_trans is not None: obj_trans = obj_trans.to(device)
                    else: obj_trans = torch.zeros((bs, seq, 3), device=device, dtype=root_pos.dtype)
                    
                    # 构建 data_dict_eval
                    data_dict_eval = {
                        "human_imu": human_imu,
                        "obj_imu": obj_imu,
                        "motion": motion,
                        "root_pos": root_pos,
                        "obj_rot": obj_rot,
                        "obj_trans": obj_trans
                    }
                    
                    # 前向传播
                    use_object_data = current_stage_info['use_object_data']
                    pred_dict = model(data_dict_eval, use_object_data=use_object_data)
                    
                    # 计算测试损失
                    total_loss_eval, loss_dict_eval, weighted_losses_eval = compute_stage_specific_loss(
                        pred_dict, batch, current_stage_info, cfg, training_step_count, contact_loss_fn, device, model
                    )
                    
                    test_loss += total_loss_eval.item()
                    for key, loss_value in weighted_losses_eval.items():
                        if isinstance(loss_value, torch.Tensor):
                            test_stage_losses[key] += loss_value.item()
                        else:
                            test_stage_losses[key] += loss_value
                    
                    # 更新tqdm描述
                    test_postfix_dict = {'test_loss': total_loss_eval.item()}
                    for key, loss_value in loss_dict_eval.items():
                        if isinstance(loss_value, torch.Tensor) and loss_value.item() != 0.0:
                            test_postfix_dict[key] = loss_value.item()
                    test_iter.set_postfix(test_postfix_dict)
            
            # 计算平均测试损失
            test_loss /= len(test_loader)
            for key in test_stage_losses:
                test_stage_losses[key] /= len(test_loader)
                test_losses[key].append(test_stage_losses[key])
            
            test_losses['total_loss'].append(test_loss)
            
            # 打印测试损失
            test_loss_msg = f'Epoch {epoch}, Stage: {current_stage_info["name"]}, Test Loss: {test_loss:.6f}'
            for key, loss_value in test_stage_losses.items():
                if loss_value != 0.0:
                    test_loss_msg += f', {key}: {loss_value:.6f}'
            print(test_loss_msg)
            
            if writer is not None:
                writer.add_scalar('test/total_loss', test_loss, n_iter)
                for key, loss_value in test_stage_losses.items():
                    if loss_value != 0.0:
                        writer.add_scalar(f'test/loss_{key}', loss_value, n_iter)
            
            # 保存最佳模型
            if test_loss < best_loss and not cfg.debug:
                best_loss = test_loss
                save_path = os.path.join(save_dir, f'epoch_{epoch}_best.pt')
                print(f'Saving best model to {save_path}')
                model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'stage_info': current_stage_info,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, save_path)
        
        # 定期保存模型
        if epoch % 50 == 0 and not cfg.debug:
            save_path = os.path.join(save_dir, f'epoch_{epoch}.pt')
            print(f'Saving model to {save_path}')
            model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save({
                'epoch': epoch,
                'stage_info': current_stage_info,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, save_path)

        # 更新学习率
        scheduler.step()
    
    # 保存最终阶段的检查点
    if current_stage_info is not None and not cfg.debug:
        save_stage_checkpoint(model, optimizer, max_epoch-1, current_stage_info, save_dir, best_loss, "final_stage")
    
    # 保存最终模型
    if not cfg.debug:
        final_path = os.path.join(save_dir, 'final.pt')
        print(f'Saving final model to {final_path}')
        model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        torch.save({
            'epoch': max_epoch - 1,
            'stage_info': current_stage_info,
            'model_state_dict': model_state_dict,
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
    device = torch.device(cfg.device if hasattr(cfg, 'device') else f'cuda:{cfg.gpus[0]}' if torch.cuda.is_available() else 'cpu')
    model = TransPoseNet(cfg).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 如果检查点包含阶段信息，打印出来
    if 'stage_info' in checkpoint:
        stage_info = checkpoint['stage_info']
        print(f'Loaded model trained with stage: {stage_info["name"]}')
        print(f'Active modules were: {stage_info["active_modules"]}')
    
    print(f'Loaded TransPose model from {checkpoint_path}, epoch {checkpoint["epoch"]}')
    
    # 多GPU包装（如果需要）
    use_multi_gpu = getattr(cfg, 'use_multi_gpu', False) and len(cfg.gpus) > 1
    if use_multi_gpu:
        print(f'Wrapping loaded model with DataParallel for GPUs: {cfg.gpus}')
        model = torch.nn.DataParallel(model, device_ids=cfg.gpus)
    
    return model