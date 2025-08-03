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
from models.TransPose_net import TransPoseNet, joint_set
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

def get_training_stage(epoch, staged_training_config=None, is_debug=False):
    """
    根据epoch判断当前训练阶段
    
    Args:
        epoch: 当前epoch
        staged_training_config: 分阶段训练配置
        is_debug: 是否为debug模式
    
    Returns:
        dict: 包含当前阶段信息的字典，包括阶段特定的超参数
    """
    if not staged_training_config or not staged_training_config.get('enabled', False):
        # 如果没有启用分阶段训练，默认训练所有模块
        return {
            'name': 'all_modules',
            'active_modules': ['velocity_contact', 'human_pose', 'object_trans'],
            'frozen_modules': [],
            'datasets': ['mixed'],
            'use_object_data': True,
            'stage_epoch': epoch,  # 阶段内的epoch
            'stage_start_epoch': 0,  # 阶段起始epoch
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
            # 计算阶段内的epoch（从0开始）
            stage_epoch = epoch - start_epoch
            
            stage_info = {
                'name': stage['name'],
                'active_modules': stage['modules'],
                'frozen_modules': [],
                'datasets': stage['datasets'],
                'use_object_data': 'omomo' in stage['datasets'] or 'mixed' in stage['datasets'],
                'stage_epoch': stage_epoch,  # 阶段内的epoch
                'stage_start_epoch': start_epoch,  # 阶段起始epoch
            }
            
            # 添加阶段特定的超参数
            for param in ['batch_size', 'lr', 'weight_decay', 'milestones', 'gamma', 'num_workers']:
                if param in stage:
                    stage_info[param] = stage[param]
            
            return stage_info
    
    # 如果没有匹配的阶段，默认返回最后一个阶段
    if stages:
        last_stage = stages[-1]
        # 如果超出了最后阶段的范围，使用最后阶段的配置
        stage_start_epoch = last_stage['epochs'][0]
        stage_epoch = epoch - stage_start_epoch
        
        stage_info = {
            'name': last_stage['name'],
            'active_modules': last_stage['modules'],
            'frozen_modules': [],
            'datasets': last_stage['datasets'],
            'use_object_data': 'omomo' in last_stage['datasets'] or 'mixed' in last_stage['datasets'],
            'stage_epoch': stage_epoch,
            'stage_start_epoch': stage_start_epoch,
        }
        
        # 添加阶段特定的超参数
        for param in ['batch_size', 'lr', 'weight_decay', 'milestones', 'gamma', 'num_workers']:
            if param in last_stage:
                stage_info[param] = last_stage[param]
        
        return stage_info
    
    # 兜底情况
    return {
        'name': 'all_modules',
        'active_modules': ['velocity_contact', 'human_pose', 'object_trans'],
        'frozen_modules': [],
        'datasets': ['mixed'],
        'use_object_data': True,
        'stage_epoch': epoch,
        'stage_start_epoch': 0,
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


def build_modular_config_for_stage(new_stage_info, save_dir, initial_pretrained_modules=None):
    """
    根据新的训练阶段动态构建pretrained_modules和skip_modules配置
    
    Args:
        new_stage_info: 新阶段信息
        save_dir: 模型保存目录
        initial_pretrained_modules: 初始的预训练模块配置
    
    Returns:
        tuple: (pretrained_modules, skip_modules)
    """
    all_modules = ['velocity_contact', 'human_pose', 'object_trans']
    stage_order = ['velocity_contact', 'human_pose', 'object_trans', 'joint_training']
    
    # 确定当前阶段在stage_order中的位置
    if new_stage_info['name'] in stage_order:
        current_stage_idx = stage_order.index(new_stage_info['name'])
    else:
        # 如果是自定义阶段名，根据激活的模块判断
        if 'velocity_contact' in new_stage_info['active_modules']:
            current_stage_idx = 0
        elif 'human_pose' in new_stage_info['active_modules'] and 'object_trans' not in new_stage_info['active_modules']:
            current_stage_idx = 1
        elif 'object_trans' in new_stage_info['active_modules'] and 'human_pose' not in new_stage_info['active_modules']:
            current_stage_idx = 2
        else:
            current_stage_idx = 3  # joint_training
    
    pretrained_modules = {}
    skip_modules = []
    
    # 优先使用初始提供的预训练模块配置
    if initial_pretrained_modules:
        pretrained_modules.update(initial_pretrained_modules)
    
    for i, module_name in enumerate(all_modules):
        if i < current_stage_idx:
            # 前面的模块：优先使用初始配置，其次尝试从save_dir加载
            if module_name not in pretrained_modules and save_dir:
                # 尝试从模块目录加载
                module_path = os.path.join(save_dir, "modules", f"{module_name}_best.pt")
                if os.path.exists(module_path):
                    pretrained_modules[module_name] = module_path
                    print(f"  - 自动检测到预训练模块: {module_name} <- {module_path}")
                else:
                    # 如果模块目录没有，尝试从stage检查点提取模块权重
                    stage_names = ['velocity_contact', 'human_pose', 'object_trans']
                    if i < len(stage_names):
                        stage_path = os.path.join(save_dir, f"stage_best_{stage_names[i]}.pt")
                        if os.path.exists(stage_path):
                            # 从stage检查点中提取并保存单个模块
                            extracted_module_path = extract_module_from_checkpoint(stage_path, module_name, save_dir)
                            if extracted_module_path:
                                pretrained_modules[module_name] = extracted_module_path
                                print(f"  - 从阶段检查点提取模块: {module_name} <- {extracted_module_path}")
                            else:
                                # skip_modules.append(module_name)
                                print(f"  - 初始化模块: {module_name} (提取失败)")
                        else:
                            # skip_modules.append(module_name)
                            print(f"  - 初始化模块: {module_name} (未找到预训练权重)")
            elif module_name in pretrained_modules:
                print(f"  - 使用配置的预训练模块: {module_name} <- {pretrained_modules[module_name]}")
            else:
                # skip_modules.append(module_name)
                print(f"  - 初始化模块: {module_name} (未提供预训练路径)")
                
        elif module_name in new_stage_info['active_modules']:
            # 当前阶段需要训练的模块
            print(f"  - 将训练模块: {module_name}")
            
        else:
            # 后面的模块暂时跳过
            skip_modules.append(module_name)
            print(f"  - 跳过模块: {module_name} (尚未到训练阶段)")
    
    return pretrained_modules, skip_modules


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
        # # 物体全局平移损失
        # loss_dict['obj_trans'] = torch.nn.functional.mse_loss(pred_dict["pred_obj_trans"], obj_trans)
        
        # 物体相对于根关节的平移损失（在根关节坐标系中）
        # 计算真值：物体相对于根关节的全局相对位置
        obj_relative_trans_global_gt = obj_trans - root_pos  # [bs, seq, 3]
        
        # 获取根关节旋转矩阵（从motion中提取）
        gt_root_orient_6d = motion[:, :, :6]  # [bs, seq, 6]
        gt_root_orient_mat = rotation_6d_to_matrix(gt_root_orient_6d.reshape(-1, 6)).reshape(bs, seq, 3, 3)  # [bs, seq, 3, 3]
        
        # 将物体相对位置转换到根关节坐标系
        gt_root_orient_mat_flat = gt_root_orient_mat.reshape(bs * seq, 3, 3)
        gt_root_orient_mat_inv = gt_root_orient_mat_flat.transpose(-1, -2)  # 旋转矩阵的逆等于转置
        obj_relative_trans_global_gt_flat = obj_relative_trans_global_gt.reshape(bs * seq, 3)
        
        obj_relative_trans_local_gt_flat = torch.bmm(gt_root_orient_mat_inv, obj_relative_trans_global_gt_flat.unsqueeze(-1)).squeeze(-1)
        obj_relative_trans_local_gt = obj_relative_trans_local_gt_flat.reshape(bs, seq, 3)
        
        # 计算损失 - obj_relative_trans已被移除
        # loss_dict['obj_relative_trans'] = torch.nn.functional.mse_loss(pred_dict["pred_obj_relative_trans_local"], obj_relative_trans_local_gt)

        # --- 新的虚拟关节监督损失：^Ov_{HO}方向向量 ---
        # 左手物体方向向量损失（^Ov_{HO}）
        if "pred_lhand_obj_direction" in pred_dict and "lhand_obj_direction" in batch:
            lhand_obj_direction_gt = batch["lhand_obj_direction"].to(device)  # [bs, seq, 3]
            loss_dict['lhand_obj_direction'] = torch.nn.functional.mse_loss(pred_dict["pred_lhand_obj_direction"], lhand_obj_direction_gt)
        else:
            loss_dict['lhand_obj_direction'] = torch.tensor(0.0, device=device)
        
        # 右手物体方向向量损失（^Ov_{HO}）
        if "pred_rhand_obj_direction" in pred_dict and "rhand_obj_direction" in batch:
            rhand_obj_direction_gt = batch["rhand_obj_direction"].to(device)  # [bs, seq, 3]
            loss_dict['rhand_obj_direction'] = torch.nn.functional.mse_loss(pred_dict["pred_rhand_obj_direction"], rhand_obj_direction_gt)
        else:
            loss_dict['rhand_obj_direction'] = torch.tensor(0.0, device=device)
        
        # --- FK重建的物体位置损失 ---
        # 使用预测的^Ov_{HO}和骨长，通过FK公式重建物体位置并与真值比较
        if "pred_obj_trans_from_fk" in pred_dict:
            loss_dict['obj_trans_from_fk'] = torch.nn.functional.mse_loss(pred_dict["pred_obj_trans_from_fk"], obj_trans)
        else:
            loss_dict['obj_trans_from_fk'] = torch.tensor(0.0, device=device)
            
    else:
        # 没有物体数据时，设置新的虚拟关节损失为0
        loss_dict['obj_relative_trans'] = torch.tensor(0.0, device=device)
        loss_dict['lhand_obj_direction'] = torch.tensor(0.0, device=device)
        loss_dict['rhand_obj_direction'] = torch.tensor(0.0, device=device)
        loss_dict['obj_trans_from_fk'] = torch.tensor(0.0, device=device)
    
    # 设置默认权重
    default_weights = {
        'rot': 10, 'root_pos': 10, 'leaf_pos': 1.0, 'full_pos': 1.0,
        'root_vel': 10, 'hand_contact': 1, 'obj_relative_trans': 10, 
        'obj_vel': 1, 'leaf_vel': 10, 'foot_contact': 1,
        'lhand_obj_direction': 6, 'rhand_obj_direction': 6,  # 新的方向向量损失权重
        'obj_trans_from_fk': 8                               # FK重建位置损失权重
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
        # 第三阶段：只关注物体方向向量和FK重建位置损失
        for key in weights:
            if key not in ['lhand_obj_direction', 'rhand_obj_direction', 'obj_trans_from_fk']:
                weights[key] = 0.0
    elif stage_name == 'joint_training':
        # 联合训练阶段：使用后两个阶段权重
        for key in weights:
            if key in ['obj_vel', 'leaf_vel', 'hand_contact']:
                weights[key] = 0.0
    
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
    bs, seq = batch["human_imu"].shape[:2]
    root_pos = batch["root_pos"].to(device)
    motion = batch["motion"].to(device)
    # 获取速度真值
    obj_vel = batch.get("obj_vel", torch.zeros((bs, seq, 3), device=device, dtype=root_pos.dtype)).to(device)
    leaf_vel = batch.get("leaf_vel", torch.zeros((bs, seq, joint_set.n_leaf, 3), device=device, dtype=root_pos.dtype)).to(device)
    
    # 获取接触真值
    lhand_contact_gt = batch.get("lhand_contact", torch.zeros((bs, seq), dtype=torch.bool, device=device)).bool().to(device)
    rhand_contact_gt = batch.get("rhand_contact", torch.zeros((bs, seq), dtype=torch.bool, device=device)).bool().to(device)
    obj_contact_gt = batch.get("obj_contact", torch.zeros((bs, seq), dtype=torch.bool, device=device)).bool().to(device)
    hand_contact_gt = torch.stack([lhand_contact_gt, rhand_contact_gt, obj_contact_gt], dim=2).float()

    default_weights = {
        'rot': 10, 'root_pos': 10, 'leaf_pos': 1.0, 'full_pos': 1.0,
        'root_vel': 10, 'hand_contact': 1, 'obj_relative_trans': 10, 
        'obj_vel': 1, 'leaf_vel': 10, 'foot_contact': 1,
        'lhand_obj_direction': 6, 'rhand_obj_direction': 6, 'obj_trans_from_fk': 8
    }
    
    # 获取配置的权重
    weights = {}
    if hasattr(cfg, 'loss_weights'):
        for key in default_weights:
            weights[key] = getattr(cfg.loss_weights, key, default_weights[key])
    else:
        weights = default_weights
    
    # 处理物体数据
    obj_trans = batch.get("obj_trans", torch.zeros((bs, seq, 3), device=device, dtype=root_pos.dtype)).to(device)
    
    loss_components = {}
    
    # 根据阶段计算相应的测试损失
    if stage_name == 'velocity_contact':
        # 速度估计损失
        if use_object_data:
            loss_components['obj_vel'] = torch.nn.functional.mse_loss(pred_dict["pred_obj_vel"], obj_vel)
        else:
            loss_components['obj_vel'] = torch.tensor(0.0, device=device)
        
        loss_components['leaf_vel'] = torch.nn.functional.mse_loss(pred_dict["pred_leaf_vel"], leaf_vel)
        
        # 手部接触损失
        if use_object_data:
            loss_components['hand_contact'] = torch.nn.functional.binary_cross_entropy(pred_dict["pred_hand_contact_prob"], hand_contact_gt)
        else:
            loss_components['hand_contact'] = torch.tensor(0.0, device=device)

        # 计算加权损失
        test_loss = torch.tensor(0.0, device=device)
        active_components = 0
        
        for key, loss_value in loss_components.items():
            if loss_value.item() != 0.0:
                test_loss += weights[key] * loss_value
                active_components += 1
        
        if active_components > 0:
            test_loss = test_loss / active_components
        
        return test_loss, loss_components
        
    elif stage_name == 'human_pose':
        # human_pose阶段：只看rot和root_pos
        if "motion" in pred_dict:
            loss_components['rot'] = torch.nn.functional.mse_loss(pred_dict["motion"], motion)
        else:
            loss_components['rot'] = torch.tensor(0.0, device=device)
            
        if "root_pos" in pred_dict:
            loss_components['root_pos'] = torch.nn.functional.mse_loss(pred_dict["root_pos"], root_pos)
        else:
            loss_components['root_pos'] = torch.tensor(0.0, device=device)
            
        # 计算加权损失
        test_loss = torch.tensor(0.0, device=device)
        active_components = 0
        
        for key, loss_value in loss_components.items():
            if loss_value.item() != 0.0:
                test_loss += weights[key] * loss_value
                active_components += 1
        
        if active_components > 0:
            test_loss = test_loss / active_components
            
        return test_loss, loss_components
        
    elif stage_name == 'object_trans':
        # object_trans阶段：看FK重建的物体位置损失
        if "pred_obj_trans_from_fk" in pred_dict:
            loss_components['obj_trans_from_fk'] = torch.nn.functional.mse_loss(pred_dict["pred_obj_trans_from_fk"], obj_trans)
        else:
            loss_components['obj_trans_from_fk'] = torch.tensor(0.0, device=device)
            
        # 直接返回FK重建损失
        test_loss = loss_components['obj_trans_from_fk']
        return test_loss, loss_components
        
    elif stage_name == 'joint_training':
        # joint_training阶段：看rot、root_pos和obj_trans_from_pose
        if "motion" in pred_dict:
            loss_components['rot'] = torch.nn.functional.mse_loss(pred_dict["motion"], motion)
        else:
            loss_components['rot'] = torch.tensor(0.0, device=device)
            
        if "root_pos" in pred_dict:
            loss_components['root_pos'] = torch.nn.functional.mse_loss(pred_dict["root_pos"], root_pos)
        else:
            loss_components['root_pos'] = torch.tensor(0.0, device=device)
            
        if "pred_obj_trans_from_fk" in pred_dict:
            loss_components['obj_trans_from_fk'] = torch.nn.functional.mse_loss(pred_dict["pred_obj_trans_from_fk"], obj_trans)
        else:
            loss_components['obj_trans_from_fk'] = torch.tensor(0.0, device=device)
            
        # 计算加权损失
        test_loss = torch.tensor(0.0, device=device)
        active_components = 0
        
        for key, loss_value in loss_components.items():
            if loss_value.item() != 0.0:
                test_loss += weights[key] * loss_value
                active_components += 1
        
        if active_components > 0:
            test_loss = test_loss / active_components
            
        return test_loss, loss_components
        
    else:
        # 其他阶段：使用总损失
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
    
    # 初始化学习率调度器（将在阶段切换时重新创建）
    scheduler = None

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
        
        # 重新创建优化器（只优化激活的参数），使用阶段特定的超参数
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        stage_lr = initial_stage_info.get('lr', cfg.lr)
        stage_weight_decay = initial_stage_info.get('weight_decay', cfg.weight_decay)
        stage_milestones = initial_stage_info.get('milestones', cfg.milestones)
        stage_gamma = initial_stage_info.get('gamma', cfg.gamma)
        
        # 多GPU时调整学习率
        if use_multi_gpu:
            stage_lr = stage_lr * len(cfg.gpus)
            print(f"多GPU训练，学习率调整为: {stage_lr}")
        
        optimizer = optim.AdamW(trainable_params, lr=stage_lr, weight_decay=stage_weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=stage_milestones, gamma=stage_gamma)
        
        print(f"初始化优化器，可训练参数数量: {sum(p.numel() for p in trainable_params)}")
        print(f"初始阶段 '{initial_stage_info['name']}' 超参数:")
        print(f"  lr: {stage_lr}")
        print(f"  weight_decay: {stage_weight_decay}")
        print(f"  milestones: {stage_milestones}")
        print(f"  gamma: {stage_gamma}")
        
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
            
            # 重新创建优化器（只优化激活的参数），使用阶段特定的超参数
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            stage_lr = new_stage_info.get('lr', cfg.lr)
            stage_weight_decay = new_stage_info.get('weight_decay', cfg.weight_decay)
            stage_milestones = new_stage_info.get('milestones', cfg.milestones)
            stage_gamma = new_stage_info.get('gamma', cfg.gamma)
            
            # 多GPU时调整学习率
            if use_multi_gpu:
                stage_lr = stage_lr * len(cfg.gpus)
                print(f"多GPU训练，学习率调整为: {stage_lr}")
            
            optimizer = optim.AdamW(trainable_params, lr=stage_lr, weight_decay=stage_weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=stage_milestones, gamma=stage_gamma)
            
            print(f"重新创建优化器，可训练参数数量: {sum(p.numel() for p in trainable_params)}")
            print(f"阶段 '{new_stage_info['name']}' 超参数:")
            print(f"  lr: {stage_lr}")
            print(f"  weight_decay: {stage_weight_decay}")
            print(f"  milestones: {stage_milestones}")
            print(f"  gamma: {stage_gamma}")
            
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
            
            # 为传统训练模式创建基于阶段配置的优化器和调度器
            if optimizer is None:
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                stage_lr = current_stage_info.get('lr', cfg.lr)
                stage_weight_decay = current_stage_info.get('weight_decay', cfg.weight_decay)
                stage_milestones = current_stage_info.get('milestones', cfg.milestones)
                stage_gamma = current_stage_info.get('gamma', cfg.gamma)
                
                # 多GPU时调整学习率
                if use_multi_gpu:
                    stage_lr = stage_lr * len(cfg.gpus)
                    print(f"多GPU训练，学习率调整为: {stage_lr}")
                
                optimizer = optim.AdamW(trainable_params, lr=stage_lr, weight_decay=stage_weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=stage_milestones, gamma=stage_gamma)
                
                print(f"传统训练模式初始化优化器，可训练参数数量: {sum(p.numel() for p in trainable_params)}")
                print(f"传统训练阶段 '{current_stage_info['name']}' 超参数:")
                print(f"  lr: {stage_lr}")
                print(f"  weight_decay: {stage_weight_decay}")
                print(f"  milestones: {stage_milestones}")
                print(f"  gamma: {stage_gamma}")
            
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
                "human_imu": human_imu,
                "obj_imu": obj_imu,
                "motion": motion,
                "root_pos": root_pos,
                "obj_rot": obj_rot,
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
                        "human_imu": human_imu,
                        "obj_imu": obj_imu,
                        "motion": motion,
                        "root_pos": root_pos,
                        "obj_rot": obj_rot,
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
        
        # 定期保存模型
        # if epoch % 50 == 0 and not cfg.debug:
        #     save_path = os.path.join(save_dir, f'epoch_{epoch}.pt')
        #     print(f'Saving model to {save_path}')
        #     model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        #     torch.save({
        #         'epoch': epoch,
        #         'stage_info': current_stage_info,
        #         'model_state_dict': model_state_dict,
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': train_loss,
        #     }, save_path)
            
        #     # 同时保存当前激活模块的单独文件
        #     if current_stage_info is not None:
        #         actual_model = get_actual_model(model)
        #         modules_dir = os.path.join(save_dir, "modules")
        #         if not os.path.exists(modules_dir):
        #             os.makedirs(modules_dir, exist_ok=True)
                
        #         for module_name in current_stage_info['active_modules']:
        #             module_save_path = os.path.join(modules_dir, f'{module_name}_epoch_{epoch}.pt')
        #             additional_info = {
        #                 'stage_info': current_stage_info,
        #                 'loss': train_loss,
        #             }
                    
        #             success = actual_model.save_module(module_name, module_save_path, epoch, additional_info)
        #             if success:
        #                 print(f'Saved {module_name} module to {module_save_path}')

        # 更新学习率（使用阶段内的epoch）
        if scheduler is not None:
            scheduler.step()
            # 打印当前学习率
            current_lr = scheduler.get_last_lr()[0]
            if epoch % 10 == 0:  # 每10个epoch打印一次学习率
                print(f'\nEpoch {epoch}, Stage: {current_stage_info["name"]}, Stage Epoch: {current_stage_info["stage_epoch"]}, LR: {current_lr:.6f}')
    
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