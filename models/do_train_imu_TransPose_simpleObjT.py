import os
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch import optim
from tqdm import tqdm
from datetime import datetime
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d

from utils.utils import tensor2numpy
from models.TransPose_net_simpleObjT import TransPoseNet
from configs.global_config import joint_set
from models.ContactAwareLoss import ContactAwareLoss
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler


def flatten_lstm_parameters(module):
    """递归调用所有 LSTM 模块的 flatten_parameters()"""
    for child in module.children():
        if isinstance(child, torch.nn.LSTM):
            child.flatten_parameters()
        else:
            flatten_lstm_parameters(child)

def build_modular_config_for_stage(new_stage_info, save_dir, initial_pretrained_modules=None):
    """
    为SimpleObjT版本构建模块化配置（只包含object_trans模块）
    
    Args:
        new_stage_info: 新阶段信息
        save_dir: 模型保存目录
        initial_pretrained_modules: 初始的预训练模块配置
    
    Returns:
        tuple: (pretrained_modules, skip_modules)
    """
    pretrained_modules = {}
    skip_modules = ['velocity_contact', 'human_pose']  # SimpleObjT版本跳过人体相关模块
    
    # 优先使用初始提供的预训练模块配置
    if initial_pretrained_modules and 'object_trans' in initial_pretrained_modules:
        pretrained_modules['object_trans'] = initial_pretrained_modules['object_trans']
        print(f"  - 使用配置的预训练模块: object_trans <- {pretrained_modules['object_trans']}")
    elif save_dir:
        # 尝试从模块目录加载
        module_path = os.path.join(save_dir, "modules", "object_trans_best.pt")
        if os.path.exists(module_path):
            pretrained_modules['object_trans'] = module_path
            print(f"  - 自动检测到预训练模块: object_trans <- {module_path}")
        else:
            print(f"  - 初始化模块: object_trans (未找到预训练权重)")
    else:
        print(f"  - 初始化模块: object_trans (新训练)")
    
    return pretrained_modules, skip_modules


def configure_training_modules(model, stage_info):
    """
    配置训练模块：对于SimpleObjT版本，只激活object_trans模块
    
    Args:
        model: 模型实例
        stage_info: 阶段信息
    """
    actual_model = get_actual_model(model)
    
    # SimpleObjT版本只有object_trans模块
    if hasattr(actual_model, 'object_trans_module') and actual_model.object_trans_module is not None:
        for param in actual_model.object_trans_module.parameters():
            param.requires_grad = True
        print(f"激活模块: object_trans")
    
    print(f"SimpleObjT模型配置完成，专注于物体位移估计")


def get_training_stage(epoch, staged_training_config=None, is_debug=False):
    """
    获取当前epoch对应的训练阶段信息
    SimpleObjT版本：专注于物体位移估计
    """
    return {
            'name': 'object_trans',
            'active_modules': ['object_trans'],
            'frozen_modules': [],
            'datasets': ['omomo'],
            'use_object_data': True
        }


def extract_module_from_checkpoint(checkpoint_path, module_name, save_dir):
    """
    从完整的模型检查点中提取单个模块并保存
    
    Args:
        checkpoint_path: 完整模型检查点路径
        module_name: 要提取的模块名称
        save_dir: 保存目录
    
    Returns:
        str: 提取的模块文件路径，失败时返回None
    """
    try:
        # 加载完整检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' not in checkpoint:
            print(f"警告: 检查点 {checkpoint_path} 中未找到 model_state_dict")
            return None
        
        full_state_dict = checkpoint['model_state_dict']
        
        # 提取指定模块的状态字典
        module_prefix = f"{module_name}_module."
        module_state_dict = {}
        
        for key, value in full_state_dict.items():
            if key.startswith(module_prefix):
                # 去掉模块前缀
                new_key = key[len(module_prefix):]
                module_state_dict[new_key] = value
        
        if not module_state_dict:
            print(f"警告: 在检查点中未找到模块 {module_name} 的权重")
            return None
        
        # 保存提取的模块
        modules_dir = os.path.join(save_dir, "modules")
        os.makedirs(modules_dir, exist_ok=True)
        
        extracted_path = os.path.join(modules_dir, f"{module_name}_extracted.pt")
        
        extracted_checkpoint = {
            'module_name': module_name,
            'module_state_dict': module_state_dict,
            'epoch': checkpoint.get('epoch', 0),
            'extracted_from': checkpoint_path,
        }
        
        torch.save(extracted_checkpoint, extracted_path)
        return extracted_path
        
    except Exception as e:
        print(f"从检查点提取模块失败: {e}")
        return None


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
    
    # 获取速度真值
    obj_vel = batch.get("obj_vel", torch.zeros((bs, seq, 3), device=device, dtype=root_pos.dtype)).to(device)
    
    # 处理物体数据
    obj_trans = batch.get("obj_trans", torch.zeros((bs, seq, 3), device=device, dtype=root_pos.dtype)).to(device)
    obj_imu_acc_watch = batch["obj_imu"][...,:3].detach().cpu().numpy()
    obj_vel_watch = obj_vel.detach().cpu().numpy()
    obj_trans_watch = obj_trans.detach().cpu().numpy()
    pred_obj_vel_watch = pred_dict["pred_obj_vel"].detach().cpu().numpy()
    pred_obj_trans_watch = pred_dict["pred_obj_trans"].detach().cpu().numpy()
    
    # 初始化损失字典
    loss_dict = {}
    
    if 'object_trans' in active_modules and use_object_data:
        loss_dict['pred_obj_vel'] = torch.nn.functional.mse_loss(pred_dict["pred_obj_vel"], obj_vel)
        loss_dict['obj_trans'] = torch.nn.functional.mse_loss(pred_dict["pred_obj_trans"], obj_trans)

    else:
        print("寄了寄了寄了寄了\n\n\n\n\n寄了寄了寄了寄了1")
    
    # 设置默认权重
    weights = {
        'obj_trans': 1, 
        'pred_obj_vel': 10,
    }
    
    
    # 计算加权总损失
    total_loss = torch.tensor(0.0, device=device)
    weighted_losses = {}
    
    for key, loss_value in loss_dict.items():
        weight = weights.get(key, 0.0)
        weighted_loss = weight * loss_value
        weighted_losses[key] = weighted_loss
        total_loss += weighted_loss
    
    return total_loss, loss_dict, weighted_losses


def compute_stage_specific_test_loss(pred_dict, batch, stage_info, cfg, device):
    """
    根据训练阶段计算相应的测试损失（用于模型选择）
    
    Args:
        pred_dict: 模型预测结果
        batch: 批次数据
        stage_info: 当前阶段信息
        device: 设备
    
    Returns:
        tuple: (测试损失, 损失组件字典)
    """
    stage_name = stage_info['name']
    use_object_data = stage_info.get('use_object_data', False)
    bs, seq = batch["obj_imu"].shape[:2]
    root_pos = batch["root_pos"].to(device)
    # 获取速度真值
    obj_vel = batch.get("obj_vel", torch.zeros((bs, seq, 3), device=device, dtype=root_pos.dtype)).to(device)
    # 处理物体数据
    obj_trans = batch.get("obj_trans", torch.zeros((bs, seq, 3), device=device, dtype=root_pos.dtype)).to(device)
    
    weights = {
        'obj_trans': 1, 
        'pred_obj_vel': 20,
    }
    
    loss_components = {}
    
    # 根据阶段计算相应的测试损失  
    if stage_name == 'object_trans':
        loss_components['obj_trans'] = torch.nn.functional.mse_loss(pred_dict["pred_obj_trans"], obj_trans) * weights['obj_trans']
        loss_components['pred_obj_vel'] = torch.nn.functional.mse_loss(pred_dict["pred_obj_vel"], obj_vel) * weights['pred_obj_vel']
        test_loss = loss_components['obj_trans'] + loss_components['pred_obj_vel']
        return test_loss, loss_components
    else:
        print("寄了寄了寄了寄了\n\n\n\n\n寄了寄了寄了寄了2")
        return None, {}

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

def save_stage_checkpoint(model, optimizer, epoch, stage_info, save_dir, loss, comprehensive_loss=None, prefix="stage"):
    """
    保存阶段检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        stage_info: 阶段信息
        save_dir: 保存目录
        loss: 损失值
        comprehensive_loss: 综合损失值（用于模型选择）
        prefix: 文件前缀
    """
    if save_dir is None:
        return
    
    checkpoint_name = f"{prefix}_{stage_info['name']}_epoch_{epoch}.pt"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    
    model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    
    checkpoint_data = {
        'epoch': epoch,
        'stage_info': stage_info,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if comprehensive_loss is not None:
        checkpoint_data['comprehensive_loss'] = comprehensive_loss
    
    torch.save(checkpoint_data, checkpoint_path)
    
    print(f"保存阶段检查点: {checkpoint_path}")
    if comprehensive_loss is not None:
        print(f"综合损失: {comprehensive_loss:.6f}")


def load_previous_stage_best_model(save_dir, previous_stage_name, device):
    """
    加载上一阶段的最佳模型
    
    Args:
        save_dir: 模型保存目录
        previous_stage_name: 上一阶段名称
        device: 设备
    
    Returns:
        model_state_dict: 模型状态字典，如果没找到则返回None
        epoch: 加载的epoch
    """
    if save_dir is None:
        return None, None
    
    # 查找上一阶段的最佳模型文件
    best_model_path = os.path.join(save_dir, f"stage_best_{previous_stage_name}.pt")
    
    if not os.path.exists(best_model_path):
        print(f"未找到上一阶段最佳模型: {best_model_path}")
        return None, None
    
    try:
        checkpoint = torch.load(best_model_path, map_location=device)
        print(f"加载上一阶段最佳模型: {best_model_path}")
        print(f"模型来自epoch {checkpoint['epoch']}, 综合损失: {checkpoint.get('comprehensive_loss', 'N/A')}")
        return checkpoint['model_state_dict'], checkpoint['epoch']
    except Exception as e:
        print(f"加载上一阶段模型失败: {e}")
        return None, None

def do_train_imu_TransPose(cfg, train_loader, test_loader=None, trial=None, model=None, optimizer=None):
    """
    训练IMU到全身姿态及物体变换的TransPose模型，支持分阶段训练和模块化训练
    
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
    
    # 读取模块化训练配置
    modular_training_config = None
    pretrained_modules = None
    skip_modules = None
    start_stage_name = None
    start_epoch = 0
    
    if staged_training_config and staged_training_config.get('enabled', False):
        modular_training_config = staged_training_config.get('modular_training', {})
        if modular_training_config and modular_training_config.get('enabled', False):
            start_stage_name = modular_training_config.get('start_from_stage', 'velocity_contact')
            pretrained_modules = modular_training_config.get('pretrained_modules', {})
            
            print(f"模块化训练已启用，从阶段 '{start_stage_name}' 开始")
            print(f"预训练模块配置: {pretrained_modules}")
    
    # 确定训练的epoch范围
    if staged_training_config and staged_training_config.get('enabled', False):
        # 分阶段训练：根据阶段配置确定epoch范围
        stages = staged_training_config.get('debug_stages' if cfg.debug else 'stages', [])
        if stages:
            # 如果指定了起始阶段，找到对应的起始epoch
            if start_stage_name:
                for stage in stages:
                    if stage['name'] == start_stage_name:
                        start_epoch = stage['epochs'][0]
                        break
                else:
                    print(f"警告: 未找到起始阶段 '{start_stage_name}'，将从epoch 0开始")
                    start_epoch = 0
            else:
                start_epoch = stages[0]['epochs'][0]
            
            # 设置max_epoch为最后一个阶段的结束epoch + 1（因为range是左闭右开的）
            max_epoch = stages[-1]['epochs'][1] + 1
            print(f"分阶段训练：epoch范围 {start_epoch} 到 {max_epoch-1}")
        else:
            print("警告: 启用了分阶段训练但未找到阶段配置，使用默认epoch设置")
    else:
        # 传统训练：使用配置的max_epoch
        print(f"传统训练模式：使用配置的max_epoch {max_epoch}")

    # 打印训练配置
    print(f'Training: {model_name} (using TransPose), pose_rep: {pose_rep}')
    print(f'use_tensorboard: {use_tensorboard}, device: {device}')
    print(f'use_multi_gpu: {use_multi_gpu}, gpus: {cfg.gpus if use_multi_gpu else [cfg.gpus[0]]}')
    print(f'epoch范围: {start_epoch} 到 {max_epoch-1} (共 {max_epoch-start_epoch} 个epoch)')
    
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
        # 如果启用模块化训练，需要根据起始阶段配置模块
        if modular_training_config and modular_training_config.get('enabled', False):
            # 获取起始阶段信息
            initial_stage_info = get_training_stage(start_epoch, staged_training_config, is_debug=cfg.debug)
            
            # 动态构建起始阶段的模块配置
            initial_pretrained_modules, initial_skip_modules = build_modular_config_for_stage(
                initial_stage_info, save_dir, pretrained_modules
            )
            
            model = TransPoseNet(cfg, pretrained_modules=initial_pretrained_modules, skip_modules=initial_skip_modules)
            print(f'Initialized modular TransPose model for stage "{initial_stage_info["name"]}" with {sum(p.numel() for p in model.parameters())} parameters')
            print(f'Initial active modules: {initial_stage_info["active_modules"]}')
        else:
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

    # 展平 LSTM 参数
    flatten_lstm_parameters(model)
    
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
    current_stage_best_loss = float('inf')  # 当前阶段的最佳损失
    train_losses = defaultdict(list)
    test_losses = defaultdict(list)
    n_iter = 0
    training_step_count = 0
    current_stage_info = None
    previous_stage_name = None  # 记录上一阶段名称

    # 如果启用分阶段训练且没有提供初始数据加载器，创建第一个阶段的数据加载器
    if train_loader is None and staged_training_config and staged_training_config.get('enabled', False):
        initial_stage_info = get_training_stage(start_epoch, staged_training_config, is_debug=cfg.debug)
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
        
        # 初始化第一阶段的损失值
        current_stage_best_loss = float('inf')
        print(f"初始阶段 '{initial_stage_info['name']}'：将使用阶段特定测试损失进行模型选择")

    for epoch in range(start_epoch, max_epoch):
        # 确定当前训练阶段
        new_stage_info = get_training_stage(epoch, staged_training_config, is_debug=cfg.debug)
        
        # 检查是否切换了训练阶段
        if current_stage_info is None or new_stage_info['name'] != current_stage_info['name']:
            print(f"\n=== Epoch {epoch}: 切换到训练阶段 '{new_stage_info['name']}' ===")
            # 重置当前阶段的损失值，为新阶段做准备
            current_stage_best_loss = float('inf')
            print(f"阶段 '{new_stage_info['name']}'：将使用阶段特定测试损失进行模型选择")
            print(f"重置当前阶段最佳测试损失为: {current_stage_best_loss}")
            
            # 保存上一个阶段的检查点（如果有）
            if current_stage_info is not None and not cfg.debug:
                # 使用当前阶段的最佳损失值保存检查点
                stage_loss = current_stage_best_loss
                save_stage_checkpoint(model, optimizer, epoch-1, current_stage_info, save_dir, stage_loss, None, "stage_end")
                previous_stage_name = current_stage_info['name']  # 记录上一阶段名称
            
            # 模块化训练：重新构建模型以正确加载/跳过模块
            if modular_training_config and modular_training_config.get('enabled', False):
                print(f"模块化训练模式：为阶段 '{new_stage_info['name']}' 重新构建模型")
                
                # 动态构建新阶段的模块配置
                new_pretrained_modules, new_skip_modules = build_modular_config_for_stage(
                    new_stage_info, save_dir, pretrained_modules
                )
                
                # 重新构建模型
                old_model = model
                model = TransPoseNet(cfg, pretrained_modules=new_pretrained_modules, skip_modules=new_skip_modules)
                model = model.to(device)
                
                # 多GPU包装
                if use_multi_gpu:
                    print(f'为重建模型包装DataParallel: {cfg.gpus}')
                    model = torch.nn.DataParallel(model, device_ids=cfg.gpus)
                
                # 展平 LSTM 参数
                flatten_lstm_parameters(model)
                
                print(f"重新构建完成，参数数量: {sum(p.numel() for p in model.parameters())}")
                
                # 删除旧模型释放内存
                del old_model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            else:
                # 非模块化训练：使用传统方式加载上一阶段最佳模型
                if previous_stage_name is not None and not cfg.debug:
                    prev_model_state, prev_epoch = load_previous_stage_best_model(save_dir, previous_stage_name, device)
                    if prev_model_state is not None:
                        # 加载上一阶段的最佳模型参数
                        actual_model = get_actual_model(model)
                        actual_model.load_state_dict(prev_model_state)
                        print(f"已加载上一阶段 '{previous_stage_name}' 的最佳模型参数")
                
                # 配置新阶段的训练模块
                configure_training_modules(model, new_stage_info)
            
            # 更新当前阶段信息
            current_stage_info = new_stage_info
            
            # 重新创建优化器（只优化激活的参数）
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma)
            
            print(f"重新创建优化器，可训练参数数量: {sum(p.numel() for p in trainable_params)}")
            
            # 创建或更新数据加载器（如果启用分阶段训练）
            if staged_training_config and staged_training_config.get('enabled', False):
                # 显式清理旧的DataLoader（如果存在）
                if 'train_loader' in locals() and train_loader is not None:
                    print("清理旧的DataLoader...")
                    # 使用dataset的cleanup方法清理共享内存
                    if hasattr(train_loader.dataset, 'cleanup'):
                        train_loader.dataset.cleanup()
                    if hasattr(test_loader, 'dataset') and hasattr(test_loader.dataset, 'cleanup'):
                        test_loader.dataset.cleanup()
                    
                    # 删除引用
                    del train_loader
                    del test_loader
                    
                    # 强制垃圾回收
                    import gc
                    gc.collect()
                    
                    # 清理GPU缓存（如果使用GPU）
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    print("旧DataLoader清理完成")
                
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
            
            # 如果是模块化训练模式，也需要正确配置模块
            if modular_training_config and modular_training_config.get('enabled', False):
                print(f"模块化训练模式：为初始阶段 '{current_stage_info['name']}' 配置模块")
                # 注意：模型已经在初始化时正确配置了，这里只需要配置训练参数
                configure_training_modules(model, current_stage_info)
            else:
                configure_training_modules(model, current_stage_info)
            
            # 初始化传统训练模式的损失值
            current_stage_best_loss = float('inf')
            print(f"传统训练模式阶段 '{current_stage_info['name']}'：将使用阶段特定测试损失进行模型选择")

        # 训练阶段
        model.train()
        train_loss = 0
        stage_losses = defaultdict(float)
        
        train_iter = tqdm(train_loader, desc=f'Epoch {epoch} - {current_stage_info["name"]}', leave=False)
        
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
                obj_imu_acc = obj_imu[...,:3]
                obj_imu_acc_watch = obj_imu_acc.detach().cpu().numpy()
                obj_imu_noise = torch.randn_like(obj_imu) * 0.001
                noisy_obj_imu = obj_imu + obj_imu_noise
                noisy_obj_imu_acc_watch = noisy_obj_imu[...,:3].detach().cpu().numpy()
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
            # 根据当前阶段决定是否使用物体数据
            use_object_data = current_stage_info['use_object_data']
            
            data_dict = {
                "obj_imu": noisy_obj_imu,
                "obj_trans": obj_trans,
                "use_object_data": use_object_data
            }
            
            # 前向传播
            optimizer.zero_grad()
            
            pred_dict = model(data_dict)
            
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
        
        # 打印训练损失（覆盖前一个Epoch的输出）
        loss_msg = f'Epoch {epoch}, Stage: {current_stage_info["name"]}, Train Loss: {train_loss:.2f}'
        for key, loss_value in stage_losses.items():
            if loss_value != 0.0:
                loss_msg += f', {key}: {loss_value:.2f}'
        
        # 限制输出长度，避免行太长导致\r失效
        max_length = 120  # 最大字符数
        if len(loss_msg) > max_length:
            loss_msg = loss_msg[:max_length-3] + '...'
        
        print(f'\r{loss_msg}', end='', flush=True)

        # 每10个epoch进行一次测试和保存
        if epoch % 10 == 0 and test_loader is not None:
            # 测试阶段
            model.eval()
            
            # 计算阶段特定的测试损失
            stage_test_loss = 0
            stage_test_components = defaultdict(float)
            
            with torch.no_grad():
                test_iter = tqdm(test_loader, desc=f'Test Epoch {epoch} - {current_stage_info["name"]}', leave=False)
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
                    
                    # 前向传播
                    use_object_data = current_stage_info['use_object_data']
                    
                    # 构建 data_dict_eval
                    data_dict_eval = {
                        "obj_imu": obj_imu,
                        "obj_trans": obj_trans,
                        "use_object_data": use_object_data
                    }
                    
                    pred_dict = model(data_dict_eval)
                    
                    # 计算阶段特定的测试损失（用于模型选择）
                    batch_stage_test_loss, batch_stage_components = compute_stage_specific_test_loss(
                        pred_dict, batch, current_stage_info, cfg, device
                    )
                    
                    if batch_stage_test_loss is not None:
                        stage_test_loss += batch_stage_test_loss.item()
                        for key, loss_value in batch_stage_components.items():
                            if isinstance(loss_value, torch.Tensor):
                                stage_test_components[key] += loss_value.item()
                            else:
                                stage_test_components[key] += loss_value
                        current_batch_loss = batch_stage_test_loss.item()
                    else:
                        # 如果返回None，需要计算一个默认的测试损失
                        # 这里可以计算一个简单的总损失作为备选
                        total_loss_eval, _, _ = compute_stage_specific_loss(
                            pred_dict, batch, current_stage_info, cfg, training_step_count, contact_loss_fn, device, model
                        )
                        stage_test_loss += total_loss_eval.item()
                        current_batch_loss = total_loss_eval.item()
                    
                    # 更新tqdm描述
                    test_postfix_dict = {'stage_test_loss': current_batch_loss}
                    
                    for key, loss_value in batch_stage_components.items():
                        if isinstance(loss_value, torch.Tensor) and loss_value.item() != 0.0:
                            test_postfix_dict[key] = loss_value.item()
                    test_iter.set_postfix(test_postfix_dict)
            
            # 计算平均测试损失
            stage_test_loss /= len(test_loader)
            
            for key in stage_test_components:
                stage_test_components[key] /= len(test_loader)
            
            test_losses['stage_test_loss'].append(stage_test_loss)
            
            # 打印测试损失（覆盖前一个Epoch的输出）
            test_loss_msg = f'Epoch {epoch}, Stage: {current_stage_info["name"]}, Stage Test Loss: {stage_test_loss:.2f}'
            
            for key, loss_value in stage_test_components.items():
                if loss_value != 0.0:
                    test_loss_msg += f', {key}: {loss_value:.2f}'
            
            # 限制输出长度，避免行太长导致\r失效
            max_length = 120  # 最大字符数
            if len(test_loss_msg) > max_length:
                test_loss_msg = test_loss_msg[:max_length-3] + '...'
            
            print(f'\r{test_loss_msg}', end='', flush=True)
            
            # 打印阶段测试损失组件（换行显示）
            if stage_test_components:
                stage_comp_msg = f"Stage Test Loss Components: "
                for key, loss_value in stage_test_components.items():
                    if loss_value != 0.0:
                        stage_comp_msg += f'{key}: {loss_value:.2f}, '
                print(f'\n{stage_comp_msg}')
            
            if writer is not None:
                writer.add_scalar('test/stage_test_loss', stage_test_loss, n_iter)
                for key, loss_value in stage_test_components.items():
                    if loss_value != 0.0:
                        writer.add_scalar(f'test/stage_{key}', loss_value, n_iter)
            
            # 根据阶段选择保存策略
            should_save_model = False
            
            # 使用阶段特定的测试损失进行模型选择
            if stage_test_loss < current_stage_best_loss and not cfg.debug:
                current_stage_best_loss = stage_test_loss
                # 同时更新全局最佳损失（用于跨阶段比较）
                if stage_test_loss < best_loss:
                    best_loss = stage_test_loss
                should_save_model = True
                save_metric = f"Stage Test Loss: {current_stage_best_loss:.2f}"
                print(f"\n阶段 '{current_stage_info['name']}' 新的最佳阶段测试损失: {current_stage_best_loss:.2f}")
            
            if should_save_model:
                # 保存当前阶段的最佳模型
                stage_best_path = os.path.join(save_dir, f'stage_best_{current_stage_info["name"]}.pt')
                print(f'Saving stage best model to {stage_best_path} ({save_metric})')
                model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                
                checkpoint_data = {
                    'epoch': epoch,
                    'stage_info': current_stage_info,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'stage_test_loss': stage_test_loss,
                    'stage_test_components': stage_test_components,
                }
                
                torch.save(checkpoint_data, stage_best_path)
                
                # 保存当前激活模块的单独文件
                actual_model = get_actual_model(model)
                modules_dir = os.path.join(save_dir, "modules")
                if not os.path.exists(modules_dir):
                    os.makedirs(modules_dir, exist_ok=True)
                
                for module_name in current_stage_info['active_modules']:
                    module_save_path = os.path.join(modules_dir, f'{module_name}_best.pt')
                    additional_info = {
                        'stage_info': current_stage_info,
                        'stage_test_loss': stage_test_loss,
                    }
                    
                    success = actual_model.save_module(module_name, module_save_path, epoch, additional_info)
                    if success:
                        print(f'Saved {module_name} module to {module_save_path}')
                
                # 如果是最后一个阶段，同时保存为全局最佳模型
                if current_stage_info['name'] == 'joint_training' or 'joint' in current_stage_info['name']:
                    final_best_path = os.path.join(save_dir, f'epoch_{epoch}_best.pt')
                    print(f'Saving final best model to {final_best_path}')
                    torch.save(checkpoint_data, final_best_path)
        

        # 更新学习率
        scheduler.step()
    
    # 保存最终阶段的检查点
    if current_stage_info is not None and not cfg.debug:
        # 使用当前阶段的最佳损失值保存最终检查点
        final_stage_loss = current_stage_best_loss
        save_stage_checkpoint(model, optimizer, max_epoch-1, current_stage_info, save_dir, final_stage_loss, None, "final_stage")
    
    # 保存最终模型（基于最佳综合损失）
    if not cfg.debug:
        # 尝试加载最后阶段的最佳模型作为最终模型
        if current_stage_info is not None:
            final_best_stage_path = os.path.join(save_dir, f'stage_best_{current_stage_info["name"]}.pt')
            if os.path.exists(final_best_stage_path):
                # 复制最佳模型为最终模型
                import shutil
                final_path = os.path.join(save_dir, 'final.pt')
                shutil.copy2(final_best_stage_path, final_path)
                print(f'Copying best stage model to final model: {final_path}')
                
                # 读取并打印最佳模型信息
                checkpoint = torch.load(final_best_stage_path, map_location=device)
                print(f'Final model - Epoch: {checkpoint["epoch"]}, Stage Test Loss: {checkpoint.get("stage_test_loss", "N/A")}')
            else:
                # 兜底：保存当前模型
                final_path = os.path.join(save_dir, 'final.pt')
                print(f'Saving current model as final to {final_path}')
                model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save({
                    'epoch': max_epoch - 1,
                    'stage_info': current_stage_info,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    'stage_test_loss': current_stage_best_loss,
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
            writer.add_scalar('final/test_loss', stage_test_loss, max_epoch)
            writer.add_scalar('final/best_test_loss', current_stage_best_loss, max_epoch)
        log_dir = writer.log_dir
        writer.close()
        print(f'TensorBoard logs saved. You can view them with: tensorboard --logdir {os.path.dirname(log_dir)}')
    
    # 训练完成后清理DataLoader
    print("训练完成，清理DataLoader...")
    if train_loader is not None and hasattr(train_loader.dataset, 'cleanup'):
        train_loader.dataset.cleanup()
    if test_loader is not None and hasattr(test_loader.dataset, 'cleanup'):
        test_loader.dataset.cleanup()
    
    # 强制垃圾回收
    import gc
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("训练结束清理完成")
    
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
    
    # 读取模块化训练配置
    staged_training_config = getattr(cfg, 'staged_training', None)
    pretrained_modules = None
    skip_modules = None
    
    if staged_training_config and staged_training_config.get('enabled', False):
        modular_training_config = staged_training_config.get('modular_training', {})
        if modular_training_config and modular_training_config.get('enabled', False):
            pretrained_modules = modular_training_config.get('pretrained_modules', {})
            print(f'Using modular training configuration: {pretrained_modules}')
    
    # 如果启用了模块化训练，使用模块化方式创建模型
    if pretrained_modules is not None:
        model = TransPoseNet(cfg, pretrained_modules=pretrained_modules, skip_modules=skip_modules).to(device)
        print(f'Created modular TransPose model')
    else:
        model = TransPoseNet(cfg).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f'Loaded TransPose model from {checkpoint_path}, epoch {checkpoint["epoch"]}')
        
        # 如果检查点包含阶段信息，打印出来
        if 'stage_info' in checkpoint:
            stage_info = checkpoint['stage_info']
            print(f'Loaded model trained with stage: {stage_info["name"]}')
            print(f'Active modules were: {stage_info["active_modules"]}')
    
    # 解决 LSTM 内存警告：调用 flatten_parameters()
    def flatten_lstm_parameters(module):
        """递归调用所有 LSTM 模块的 flatten_parameters()"""
        for child in module.children():
            if isinstance(child, torch.nn.LSTM):
                child.flatten_parameters()
            else:
                flatten_lstm_parameters(child)
    
    flatten_lstm_parameters(model)
    
    # 多GPU包装（如果需要）
    use_multi_gpu = getattr(cfg, 'use_multi_gpu', False) and len(cfg.gpus) > 1
    if use_multi_gpu:
        print(f'Wrapping loaded model with DataParallel for GPUs: {cfg.gpus}')
        model = torch.nn.DataParallel(model, device_ids=cfg.gpus)
    
    return model