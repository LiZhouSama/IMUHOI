import os
import pickle
import random
import time
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import transform
import torch
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from torch import optim
from tqdm import tqdm
from datetime import datetime
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d

from models.IMUHOI_stage_net import TransPoseNet
from configs.global_config_IMUHOI import (
    _SENSOR_POS_INDICES,
    _SENSOR_VEL_NAMES,
    _REDUCED_POSE_NAMES,
)
from torch.cuda.amp.grad_scaler import GradScaler


_DEBUG_STAGES_PRINTED = False

def compute_stage_schedule(staged_training_config, is_debug: bool = False):
    """Convert staged training configuration into a flat schedule list."""
    if not staged_training_config or not staged_training_config.get('enabled', False):
        return [], 0

    stages_raw = staged_training_config.get('debug_stages' if is_debug else 'stages', [])
    schedule = []
    running_start = 0

    for stage in stages_raw:
        stage_copy = dict(stage)
        epochs_val = stage_copy.get('epochs', None)
        if not isinstance(epochs_val, int) or epochs_val <= 0:
            raise ValueError(
                f"阶段 {stage_copy.get('name', '?')} 的 epochs 配置必须为正整数时长，收到: {epochs_val}"
            )
        duration = int(epochs_val)
        start_epoch = running_start
        end_epoch = start_epoch + duration - 1
        running_start = end_epoch + 1

        stage_copy['stage_start_epoch'] = start_epoch
        stage_copy['stage_end_epoch'] = end_epoch
        stage_copy['stage_duration'] = duration
        schedule.append(stage_copy)

    total_epochs = schedule[-1]['stage_end_epoch'] + 1 if schedule else 0
    return schedule, total_epochs


def flatten_lstm_parameters(module):
    """递归调用所有 LSTM 模块的 flatten_parameters()"""
    for child in module.children():
        if isinstance(child, torch.nn.LSTM):
            child.flatten_parameters()
        else:
            flatten_lstm_parameters(child)


def get_training_stage(epoch, staged_training_config=None, is_debug=False):
    """Determine the active training stage for a given epoch."""
    if not staged_training_config or not staged_training_config.get('enabled', False):
        # Staged training disabled: run all modules in a single phase
        return {
            'name': 'all_modules',
            'active_modules': ['velocity_contact', 'human_pose', 'object_trans'],
            'frozen_modules': [],
            'datasets': [],
            'use_object_data': True,
            'stage_epoch': epoch,
            'stage_start_epoch': 0,
        }

    schedule, _ = compute_stage_schedule(staged_training_config, is_debug=is_debug)
    if is_debug:
        global _DEBUG_STAGES_PRINTED
        if not _DEBUG_STAGES_PRINTED:
            print(f"Debug模式: 使用debug_stages配置 (共{len(schedule)}个阶段)")
            _DEBUG_STAGES_PRINTED = True

    for stage in schedule:
        start_epoch, end_epoch = stage['stage_start_epoch'], stage['stage_end_epoch']
        if start_epoch <= epoch <= end_epoch:
            stage_epoch = epoch - start_epoch
            stage_info = {
                'name': stage['name'],
                'active_modules': stage['modules'],
                'frozen_modules': [],
                'datasets': stage['datasets'],
                'use_object_data': True,
                'stage_epoch': stage_epoch,
                'stage_start_epoch': start_epoch,
            }
            for param in ['batch_size', 'lr', 'weight_decay', 'milestones', 'gamma', 'num_workers']:
                if param in stage:
                    stage_info[param] = stage[param]
            return stage_info

    if schedule:
        last_stage = schedule[-1]
        stage_start_epoch = last_stage['stage_start_epoch']
        stage_epoch = epoch - stage_start_epoch
        stage_info = {
            'name': last_stage['name'],
            'active_modules': last_stage['modules'],
            'frozen_modules': [],
            'datasets': last_stage['datasets'],
            'use_object_data': True,
            'stage_epoch': stage_epoch,
            'stage_start_epoch': stage_start_epoch,
        }
        for param in ['batch_size', 'lr', 'weight_decay', 'milestones', 'gamma', 'num_workers']:
            if param in last_stage:
                stage_info[param] = last_stage[param]
        return stage_info

    return {
        'name': 'all_modules',
        'active_modules': ['velocity_contact', 'human_pose', 'object_trans'],
        'frozen_modules': [],
        'datasets': [],
        'use_object_data': True,
        'stage_epoch': epoch,
        'stage_start_epoch': 0,
    }
def configure_training_modules(model, stage_info):
    """
    配置模型的训练模块
    
    Args:
        model: TransPoseNet模型实例（可能被DataParallel包装）
        stage_info: 阶段信息字典，包含active_modules和frozen_modules
    """
    actual_model = get_actual_model(model)
    active_modules = stage_info.get('active_modules', [])
    frozen_modules = stage_info.get('frozen_modules', [])
    actual_model.configure_training_modules(active_modules, frozen_modules)
    print(f"已配置训练模块 - 激活: {active_modules}, 冻结: {frozen_modules}")




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
            current_stage_idx = 2  # object_trans
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
                                print(f"  - 初始化模块: {module_name} (提取失败)")
                        else:
                            print(f"  - 初始化模块: {module_name} (未找到预训练权重)")
            elif module_name in pretrained_modules:
                print(f"  - 使用配置的预训练模块: {module_name} <- {pretrained_modules[module_name]}")
            else:
                print(f"  - 初始化模块: {module_name} (未提供预训练路径)")
                
        elif module_name in new_stage_info['active_modules']:
            # 当前阶段需要训练的模块
            # 可选：从上一阶段继承权重初始化
            if save_dir and module_name not in pretrained_modules:
                # 尝试从模块目录加载之前保存的权重（用于继续训练或微调）
                module_path = os.path.join(save_dir, "modules", f"{module_name}_best.pt")
                if os.path.exists(module_path):
                    # 这里可以选择是否加载，根据需求决定
                    # 如果希望从头训练则跳过，如果希望继续训练则加载
                    # pretrained_modules[module_name] = module_path
                    # print(f"  - 从之前阶段初始化模块: {module_name} <- {module_path}")
                    pass
            print(f"  - 将训练模块: {module_name}")
            
        else:
            # 后面的模块暂时跳过
            skip_modules.append(module_name)
            print(f"  - 跳过模块: {module_name} (尚未到训练阶段)")
    
    return pretrained_modules, skip_modules


LOSS_KEYS_BY_MODULE = {
    'velocity_contact': {'obj_vel', 'hand_vel', 'hand_contact'},
    'human_pose': {'vel_root', 'pose_reduced', 'root_vel_local', 'root_vel', 'root_trans', 'hand_pos'},
    'object_trans': {
        'obj_trans',
        'lhand_obj_direction',
        'rhand_obj_direction',
        'lhand_lb',
        'rhand_lb',
        'hoi_error_l',
        'hoi_error_r',
        'obj_vel_cons',
        'obj_acc_cons',
    },
}
ALL_LOSS_KEYS = sorted(set().union(*LOSS_KEYS_BY_MODULE.values()))
ALL_LOSS_KEYS_SET = set(ALL_LOSS_KEYS)

# 测试阶段用于模型选择的loss键集合，可与训练阶段不同
# 默认与训练一致，仅对需要定制的模块进行覆盖，例如：human_pose仅使用pose_reduced
LOSS_KEYS_BY_MODULE_TEST = {
    'velocity_contact': {'obj_vel', 'hand_vel', 'hand_contact'},
    'human_pose': {'pose_reduced','root_trans'},
    'object_trans': {
        'lhand_obj_direction',
        'rhand_obj_direction',
        'lhand_lb',
        'rhand_lb',
        'hoi_error_l',
        'hoi_error_r',
    },
}

def _resolve_loss_weights(cfg):
    loss_weights_cfg = getattr(cfg, 'loss_weights', {})
    weights = {}
    if isinstance(loss_weights_cfg, dict):
        for key, value in loss_weights_cfg.items():
            try:
                weights[key] = float(value)
            except (TypeError, ValueError):
                weights[key] = 0.0
    else:
        for key in loss_weights_cfg:
            try:
                weights[key] = float(getattr(loss_weights_cfg, key))
            except (TypeError, ValueError):
                weights[key] = 0.0
    return weights


def _loss_keys_for_modules(active_modules):
    keys = set()
    for module in active_modules:
        keys.update(LOSS_KEYS_BY_MODULE.get(module, set()))
    return keys


def _masked_mse(pred, target, mask, zero):
    if mask is None:
        return F.mse_loss(pred, target)
    mask_bool = mask.bool()
    if mask_bool.sum() == 0:
        return zero.clone()
    return F.mse_loss(pred[mask_bool], target[mask_bool])


def _masked_mean(values, mask, zero):
    if mask is None:
        return values.mean()
    mask_bool = mask.bool()
    if mask_bool.sum() == 0:
        return zero.clone()
    return values[mask_bool].mean()


def _apply_loss_weights(loss_dict, cfg, active_modules, device):
    weights = _resolve_loss_weights(cfg)
    allowed_keys = _loss_keys_for_modules(active_modules)
    if not allowed_keys:
        allowed_keys = ALL_LOSS_KEYS_SET
    dtype = next(iter(loss_dict.values())).dtype if loss_dict else torch.float32
    total_loss = torch.zeros(1, device=device, dtype=dtype)
    weighted_losses = {}
    for key, loss in loss_dict.items():
        weight_default = 1.0 if key in allowed_keys else 0.0
        weight = weights.get(key, weight_default)
        if key not in allowed_keys:
            weight = 0.0
        weighted_losses[key] = loss * weight
        total_loss = total_loss + weighted_losses[key]
    return total_loss.squeeze(), weighted_losses


def _compute_loss_terms(pred_dict, batch, stage_info, device, model=None):
    active_modules = set(stage_info['active_modules'])
    use_object_data = bool(stage_info.get('use_object_data', True))

    human_imu = batch['human_imu'].to(device)
    dtype = human_imu.dtype
    bs, seq = human_imu.shape[:2]
    zero = human_imu.new_tensor(0.0)
    root_ori_gt = rotation_6d_to_matrix(human_imu[:, :, 0, -6:])

    losses = {key: zero.clone() for key in ALL_LOSS_KEYS}

    trans_gt = batch.get('trans')
    if isinstance(trans_gt, torch.Tensor):
        trans_gt = trans_gt.to(device)
    else:
        trans_gt = torch.zeros(bs, seq, 3, device=device, dtype=dtype)

    root_vel_gt = batch.get('root_vel')
    if isinstance(root_vel_gt, torch.Tensor):
        root_vel_gt = root_vel_gt.to(device)
    else:
        root_vel_gt = torch.zeros(bs, seq, 3, device=device, dtype=dtype)

    root_vel_local_gt = root_ori_gt.transpose(-1, -2).matmul(root_vel_gt.unsqueeze(-1)).squeeze(-1)

    sensor_vel_root_gt = batch.get('sensor_vel_root')
    if isinstance(sensor_vel_root_gt, torch.Tensor):
        sensor_vel_root_gt = sensor_vel_root_gt.to(device)
        if sensor_vel_root_gt.dim() == 3:
            sensor_vel_root_gt = sensor_vel_root_gt.unsqueeze(0).expand(bs, -1, -1, -1)
    else:
        sensor_vel_root_gt = torch.zeros(bs, seq, len(_SENSOR_VEL_NAMES), 3, device=device, dtype=dtype)

    sensor_vel_glb_gt = batch.get('sensor_vel_glb')
    if isinstance(sensor_vel_glb_gt, torch.Tensor):
        sensor_vel_glb_gt = sensor_vel_glb_gt.to(device)
        if sensor_vel_glb_gt.dim() == 3:
            sensor_vel_glb_gt = sensor_vel_glb_gt.unsqueeze(0).expand(bs, -1, -1, -1)
    else:
        sensor_vel_glb_gt = torch.zeros(bs, seq, len(_SENSOR_POS_INDICES), 3, device=device, dtype=dtype)

    obj_vel_gt = batch.get('obj_vel')
    if isinstance(obj_vel_gt, torch.Tensor):
        obj_vel_gt = obj_vel_gt.to(device)
    else:
        obj_vel_gt = torch.zeros(bs, seq, 3, device=device, dtype=dtype)

    obj_imu_gt = batch.get('obj_imu')
    if isinstance(obj_imu_gt, torch.Tensor):
        obj_imu_gt = obj_imu_gt.to(device)
    else:
        obj_imu_gt = None

    obj_trans_gt = batch.get('obj_trans')
    if isinstance(obj_trans_gt, torch.Tensor):
        obj_trans_gt = obj_trans_gt.to(device)
    else:
        obj_trans_gt = torch.zeros(bs, seq, 3, device=device, dtype=dtype)

    ori_root_reduced_gt = batch.get('ori_root_reduced')
    if isinstance(ori_root_reduced_gt, torch.Tensor):
        ori_root_reduced_gt = ori_root_reduced_gt.to(device)
    else:
        ori_root_reduced_gt = None

    position_global_gt = batch.get('position_global')
    if isinstance(position_global_gt, torch.Tensor):
        position_global_gt = position_global_gt.to(device)
    else:
        position_global_gt = None

    lhand_dir_gt = batch.get('lhand_obj_direction')
    if isinstance(lhand_dir_gt, torch.Tensor):
        lhand_dir_gt = lhand_dir_gt.to(device)
    else:
        lhand_dir_gt = None

    rhand_dir_gt = batch.get('rhand_obj_direction')
    if isinstance(rhand_dir_gt, torch.Tensor):
        rhand_dir_gt = rhand_dir_gt.to(device)
    else:
        rhand_dir_gt = None

    lhand_contact_gt = batch.get('lhand_contact')
    if isinstance(lhand_contact_gt, torch.Tensor):
        lhand_contact_gt = lhand_contact_gt.to(device).bool()
    else:
        lhand_contact_gt = torch.zeros(bs, seq, device=device, dtype=torch.bool)

    rhand_contact_gt = batch.get('rhand_contact')
    if isinstance(rhand_contact_gt, torch.Tensor):
        rhand_contact_gt = rhand_contact_gt.to(device).bool()
    else:
        rhand_contact_gt = torch.zeros(bs, seq, device=device, dtype=torch.bool)

    obj_contact_gt = batch.get('obj_contact')
    if isinstance(obj_contact_gt, torch.Tensor):
        obj_contact_gt = obj_contact_gt.to(device).bool()
    else:
        obj_contact_gt = torch.zeros(bs, seq, device=device, dtype=torch.bool)

    has_object = batch.get('has_object')
    if isinstance(has_object, torch.Tensor):
        has_object_mask = has_object.to(device=device, dtype=torch.bool)
        if has_object_mask.dim() == 0:
            has_object_mask = has_object_mask.unsqueeze(0)
        if has_object_mask.dim() == 1:
            has_object_mask = has_object_mask.unsqueeze(1).expand(has_object_mask.shape[0], seq)
        else:
            has_object_mask = has_object_mask.bool()
    elif isinstance(has_object, (list, tuple)):
        has_object_mask = torch.tensor(has_object, device=device, dtype=torch.bool).unsqueeze(1).expand(-1, seq)
    elif isinstance(has_object, bool):
        has_object_mask = torch.full((bs, seq), has_object, device=device, dtype=torch.bool)
    else:
        has_object_mask = torch.ones(bs, seq, device=device, dtype=torch.bool)
    if has_object_mask.shape[0] != bs:
        has_object_mask = has_object_mask[0].unsqueeze(0).expand(bs, seq)

    hand_contact_gt = torch.stack(
        [
            lhand_contact_gt.float(),
            rhand_contact_gt.float(),
            obj_contact_gt.float(),
        ],
        dim=-1,
    )

    if 'velocity_contact' in active_modules:
        if use_object_data and 'pred_obj_vel' in pred_dict:
            losses['obj_vel'] = F.mse_loss(pred_dict['pred_obj_vel'], obj_vel_gt)
        if 'pred_hand_glb_vel' in pred_dict:
            hand_indices = [-2, -1]
            gt_hand_vel = sensor_vel_glb_gt[:, :, hand_indices, :]
            losses['hand_vel'] = F.mse_loss(pred_dict['pred_hand_glb_vel'], gt_hand_vel)
        if use_object_data and 'pred_hand_contact_prob' in pred_dict:
            losses['hand_contact'] = F.binary_cross_entropy(pred_dict['pred_hand_contact_prob'], hand_contact_gt)

    if 'human_pose' in active_modules:
        if 'v_pred' in pred_dict:
            v_pred = pred_dict['v_pred'].view(bs, seq, -1, 3)
            vel_indices = [0, 1, 2, 3, 0, 3, 4, 5]  # 根据posers_config排序，腿、躯干、手
            target_vel = sensor_vel_root_gt[:, :, vel_indices,:]
            losses['vel_root'] = F.mse_loss(v_pred, target_vel)
        if 'p_pred' in pred_dict and ori_root_reduced_gt is not None:
            pose_gt_6d = matrix_to_rotation_6d(
                ori_root_reduced_gt.reshape(-1, 3, 3)
            ).reshape(bs, seq, len(_REDUCED_POSE_NAMES), 6)
            p_pred = pred_dict['p_pred'].view(bs, seq, len(_REDUCED_POSE_NAMES), 6)
            losses['pose_reduced'] = F.mse_loss(p_pred, pose_gt_6d)
        if 'root_vel_local_pred' in pred_dict:
            losses['root_vel_local'] = F.mse_loss(pred_dict['root_vel_local_pred'], root_vel_local_gt)
        if 'root_vel_pred' in pred_dict:
            losses['root_vel'] = F.mse_loss(pred_dict['root_vel_pred'], root_vel_gt)
        if 'root_trans_pred' in pred_dict:
            losses['root_trans'] = F.mse_loss(pred_dict['root_trans_pred'], trans_gt)
        if 'pred_hand_glb_pos' in pred_dict and position_global_gt is not None:
            hand_pos_gt = torch.stack(
                [
                    position_global_gt[:, :, 20, :],
                    position_global_gt[:, :, 21, :],
                ],
                dim=2,
            )
            losses['hand_pos'] = F.mse_loss(pred_dict['pred_hand_glb_pos'], hand_pos_gt)

    if 'object_trans' in active_modules and use_object_data and has_object_mask.any():
        obj_mask = has_object_mask
        if 'pred_obj_trans' in pred_dict:
            losses['obj_trans'] = _masked_mse(pred_dict['pred_obj_trans'], obj_trans_gt, obj_mask, zero)
        if 'pred_obj_vel_from_posdiff' in pred_dict:
            losses['obj_vel_cons'] = _masked_mse(pred_dict['pred_obj_vel_from_posdiff'], obj_vel_gt, obj_mask, zero)
        if 'pred_obj_acc_from_posdiff' in pred_dict and obj_imu_gt is not None:
            losses['obj_acc_cons'] = _masked_mse(pred_dict['pred_obj_acc_from_posdiff'], obj_imu_gt[:, :, :3], obj_mask, zero)
        if position_global_gt is not None:
            lhand_pos_gt = position_global_gt[:, :, 20, :]
            rhand_pos_gt = position_global_gt[:, :, 21, :]
            lb_l_gt = torch.norm(obj_trans_gt - lhand_pos_gt, dim=-1)
            lb_r_gt = torch.norm(obj_trans_gt - rhand_pos_gt, dim=-1)
            mask_l = (lhand_contact_gt & obj_mask)
            mask_r = (rhand_contact_gt & obj_mask)
            if 'pred_lhand_lb' in pred_dict:
                losses['lhand_lb'] = _masked_mse(pred_dict['pred_lhand_lb'], lb_l_gt, mask_l, zero)
            if 'pred_rhand_lb' in pred_dict:
                losses['rhand_lb'] = _masked_mse(pred_dict['pred_rhand_lb'], lb_r_gt, mask_r, zero)
            if 'pred_lhand_obj_direction' in pred_dict and lhand_dir_gt is not None:
                losses['lhand_obj_direction'] = _masked_mse(pred_dict['pred_lhand_obj_direction'], lhand_dir_gt, mask_l, zero)
            if 'pred_rhand_obj_direction' in pred_dict and rhand_dir_gt is not None:
                losses['rhand_obj_direction'] = _masked_mse(pred_dict['pred_rhand_obj_direction'], rhand_dir_gt, mask_r, zero)
            if (
                'pred_lhand_obj_direction' in pred_dict
                and 'pred_lhand_lb' in pred_dict
                and lhand_dir_gt is not None
            ):
                hoi_mask_l = mask_l
                if hoi_mask_l.any():
                    vec_gt_l = lhand_dir_gt * lb_l_gt.unsqueeze(-1)
                    vec_pred_l = pred_dict['pred_lhand_obj_direction'] * pred_dict['pred_lhand_lb'].unsqueeze(-1)
                    diff_l = torch.norm(vec_pred_l - vec_gt_l, dim=-1)
                    losses['hoi_error_l'] = _masked_mean(diff_l, hoi_mask_l, zero)
            if (
                'pred_rhand_obj_direction' in pred_dict
                and 'pred_rhand_lb' in pred_dict
                and rhand_dir_gt is not None
            ):
                hoi_mask_r = mask_r
                if hoi_mask_r.any():
                    vec_gt_r = rhand_dir_gt * lb_r_gt.unsqueeze(-1)
                    vec_pred_r = pred_dict['pred_rhand_obj_direction'] * pred_dict['pred_rhand_lb'].unsqueeze(-1)
                    diff_r = torch.norm(vec_pred_r - vec_gt_r, dim=-1)
                    losses['hoi_error_r'] = _masked_mean(diff_r, hoi_mask_r, zero)

    return losses


def compute_stage_specific_loss(pred_dict, batch, stage_info, cfg, training_step_count, contact_loss_fn, device, model=None):
    active_modules = set(stage_info['active_modules'])
    loss_dict = _compute_loss_terms(pred_dict, batch, stage_info, device, model)
    total_loss, weighted_losses = _apply_loss_weights(loss_dict, cfg, active_modules, device)
    return total_loss, loss_dict, weighted_losses


def compute_stage_specific_test_loss(pred_dict, batch, stage_info, cfg, device, model=None):
    active_modules = set(stage_info['active_modules'])
    loss_dict = _compute_loss_terms(pred_dict, batch, stage_info, device, model)

    # 仅选择用于测试阶段模型选择的loss键
    test_keys = set()
    for module in active_modules:
        test_keys.update(LOSS_KEYS_BY_MODULE_TEST.get(module, LOSS_KEYS_BY_MODULE.get(module, set())))

    filtered_loss_dict = {k: v for k, v in loss_dict.items() if k in test_keys}

    # 使用与训练相同的权重策略，但仅对筛选后的loss进行加权求和
    total_loss, _ = _apply_loss_weights(filtered_loss_dict, cfg, active_modules, device)
    return total_loss, filtered_loss_dict
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


def build_optimizer_and_scheduler(model: torch.nn.Module, cfg, stage_info, use_multi_gpu: bool):
    """基于阶段超参数创建优化器和调度器，仅包含可训练参数"""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    stage_lr = stage_info.get('lr', cfg.lr)
    stage_weight_decay = stage_info.get('weight_decay', cfg.weight_decay)
    stage_milestones = stage_info.get('milestones', cfg.milestones)
    stage_gamma = stage_info.get('gamma', cfg.gamma)

    if use_multi_gpu:
        stage_lr = stage_lr * len(cfg.gpus)
        print(f"多GPU训练，学习率调整为: {stage_lr}")

    optimizer = optim.AdamW(trainable_params, lr=stage_lr, weight_decay=stage_weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=stage_milestones, gamma=stage_gamma)

    print(f"可训练参数数量: {sum(p.numel() for p in trainable_params)}")
    print(f"阶段超参数: lr={stage_lr}, weight_decay={stage_weight_decay}, milestones={stage_milestones}, gamma={stage_gamma}")
    return optimizer, scheduler


def rebuild_dataloaders_if_needed(cfg, new_stage_info, train_loader, test_loader):
    """释放旧 DataLoader（含 dataset.cleanup 支持）并创建新 Loader"""
    # 显式清理旧的DataLoader（如果存在）
    if train_loader is not None:
        print("清理旧的DataLoader...")
        if hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'cleanup'):
            train_loader.dataset.cleanup()
        if test_loader is not None and hasattr(test_loader, 'dataset') and hasattr(test_loader.dataset, 'cleanup'):
            test_loader.dataset.cleanup()
        del train_loader
        del test_loader
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("旧DataLoader清理完成")

    create_fn = None
    main_module = sys.modules.get("__main__")
    if main_module is not None and hasattr(main_module, "create_staged_dataloaders"):
        create_fn = getattr(main_module, "create_staged_dataloaders")
    else:
        from train_IMUHOI import create_staged_dataloaders as create_fn
    new_train_loader, new_test_loader = create_fn(cfg, new_stage_info)
    if new_train_loader is None:
        print(f"错误: 无法为阶段 '{new_stage_info['name']}' 创建数据加载器")
        return None, None
    print(f"已为阶段 '{new_stage_info['name']}' 创建新的数据加载器")
    return new_train_loader, new_test_loader


def build_model_input_dict(batch, current_stage_info, cfg, device, add_noise: bool = True):
    human_imu = batch['human_imu'].to(device)
    bs, seq = human_imu.shape[:2]
    dtype = human_imu.dtype

    imu_noise_std = getattr(cfg, 'imu_noise_std', 0.1)
    if add_noise and imu_noise_std > 0:
        human_imu = human_imu + torch.randn_like(human_imu) * imu_noise_std

    obj_imu = batch.get('obj_imu')
    if isinstance(obj_imu, torch.Tensor):
        obj_imu = obj_imu.to(device)
        obj_noise_std = getattr(cfg, 'obj_imu_noise_std', 0.1)
        if add_noise and obj_noise_std > 0:
            obj_imu = obj_imu + torch.randn_like(obj_imu) * obj_noise_std
    else:
        obj_feat_dim = getattr(cfg, 'obj_imu_dim', 9)
        obj_imu = torch.zeros(bs, seq, obj_feat_dim, device=device, dtype=dtype)

    def _get_tensor(key, tensor_dtype=dtype):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            value = value.to(device=device)
            if tensor_dtype is not None and value.dtype != tensor_dtype:
                value = value.to(dtype=tensor_dtype)
            return value
        return None

    def _ensure_bt(tensor, default_shape, fill_dtype):
        if tensor is None:
            return torch.zeros(*default_shape, device=device, dtype=fill_dtype)
        tensor = tensor.to(device=device)
        if tensor.dim() == len(default_shape) - 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] == 1 and bs > 1:
            tensor = tensor.expand(bs, *tensor.shape[1:])
        if tensor.shape[:len(default_shape)] != tuple(default_shape):
            tensor = tensor.reshape(*default_shape)
        if tensor.dtype != fill_dtype:
            tensor = tensor.to(dtype=fill_dtype)
        return tensor

    sensor_vel_root = _ensure_bt(_get_tensor('sensor_vel_root'), (bs, seq, len(_SENSOR_VEL_NAMES), 3), dtype)
    sensor_vel_glb = _ensure_bt(_get_tensor('sensor_vel_glb'), (bs, seq, len(_SENSOR_POS_INDICES), 3), dtype)
    obj_vel = _ensure_bt(_get_tensor('obj_vel'), (bs, seq, 3), dtype)
    trans = _ensure_bt(_get_tensor('trans'), (bs, seq, 3), dtype)
    obj_trans = _ensure_bt(_get_tensor('obj_trans'), (bs, seq, 3), dtype)

    v_init = sensor_vel_root[:, 0]
    hand_indices = [len(_SENSOR_POS_INDICES) - 2, len(_SENSOR_POS_INDICES) - 1]
    hand_vel_glb_init = sensor_vel_glb[:, 0, hand_indices, :]
    obj_vel_init = obj_vel[:, 0, :]
    trans_init = trans[:, 0, :]
    obj_trans_init = obj_trans[:, 0, :]

    ori_root_reduced_val = _get_tensor('ori_root_reduced')
    if isinstance(ori_root_reduced_val, torch.Tensor):
        ori_root_reduced = ori_root_reduced_val
        if ori_root_reduced.dim() == 4:
            ori_root_reduced = ori_root_reduced.unsqueeze(0)
        if ori_root_reduced.shape[0] == 1 and bs > 1:
            ori_root_reduced = ori_root_reduced.expand(bs, *ori_root_reduced.shape[1:])
        if ori_root_reduced.shape[0] != bs:
            ori_root_reduced = ori_root_reduced.reshape(bs, seq, len(_REDUCED_POSE_NAMES), 3, 3)
        p_init = matrix_to_rotation_6d(
            ori_root_reduced[:, 0].reshape(bs * len(_REDUCED_POSE_NAMES), 3, 3)
        ).reshape(bs, len(_REDUCED_POSE_NAMES), 6)
    else:
        p_init = torch.zeros(bs, len(_REDUCED_POSE_NAMES), 6, device=device, dtype=dtype)

    lhand_contact = _ensure_bt(_get_tensor('lhand_contact', tensor_dtype=torch.bool), (bs, seq), torch.bool)
    rhand_contact = _ensure_bt(_get_tensor('rhand_contact', tensor_dtype=torch.bool), (bs, seq), torch.bool)
    obj_contact = _ensure_bt(_get_tensor('obj_contact', tensor_dtype=torch.bool), (bs, seq), torch.bool)

    if lhand_contact is not None and rhand_contact is not None and obj_contact is not None:
        contact_first = torch.stack(
            [
                lhand_contact[:, 0].float().to(device=device, dtype=dtype),
                rhand_contact[:, 0].float().to(device=device, dtype=dtype),
                obj_contact[:, 0].float().to(device=device, dtype=dtype),
            ],
            dim=-1,
        )
    else:
        contact_first = torch.zeros(bs, 3, device=device, dtype=dtype)
    contact_init = torch.cat((contact_first, obj_vel_init), dim=-1)

    def _prepare_has_object(value):
        if value is None:
            return torch.ones(bs, dtype=torch.bool, device=device)
        if isinstance(value, torch.Tensor):
            value = value.to(device=device, dtype=torch.bool)
            if value.dim() == 0:
                value = value.view(1)
            if value.shape[0] == 1 and bs > 1:
                value = value.expand(bs)
            return value
        if isinstance(value, (bool, int)):
            return torch.tensor([bool(value)], dtype=torch.bool, device=device).expand(bs)
        value = torch.as_tensor(value, dtype=torch.bool, device=device)
        if value.dim() == 0:
            value = value.view(1)
        if value.shape[0] == 1 and bs > 1:
            value = value.expand(bs)
        return value

    has_object = _prepare_has_object(batch.get('has_object'))

    data_dict = {
        'human_imu': human_imu,
        'obj_imu': obj_imu,
        'v_init': v_init,
        'p_init': p_init,
        'trans_init': trans_init,
        'obj_trans_init': obj_trans_init,
        'obj_vel_init': obj_vel_init,
        'hand_vel_glb_init': hand_vel_glb_init,
        'contact_init': contact_init,
        'has_object': has_object,
        'use_object_data': current_stage_info.get('use_object_data', True),
    }

    return data_dict
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
    
    # 仅支持分阶段训练：根据阶段配置确定epoch范围
    assert staged_training_config and staged_training_config.get('enabled', False), \
        "当前精简版本仅支持分阶段训练，请在配置中启用 staged_training.enabled=True"
    schedule, total_epochs = compute_stage_schedule(staged_training_config, is_debug=cfg.debug)
    if schedule:
        if start_stage_name:
            matched = False
            for s in schedule:
                if s['name'] == start_stage_name:
                    start_epoch = s['stage_start_epoch']
                    matched = True
                    break
            if not matched:
                print(f"警告: 未找到起始阶段 '{start_stage_name}'，将从epoch 0开始")
                start_epoch = 0
        else:
            start_epoch = schedule[0]['stage_start_epoch']
        max_epoch = total_epochs
        print(f"分阶段训练：epoch范围 {start_epoch} 到 {max_epoch-1}")
    else:
        raise ValueError("启用了分阶段训练但未找到阶段配置，请检查配置文件中的 staged_training.stages 或 debug_stages")

    # 打印训练配置
    print(f'Training: {model_name} (using TransPose), pose_rep: {pose_rep}')
    print(f'use_tensorboard: {use_tensorboard}, device: {device}')
    print(f'use_multi_gpu: {use_multi_gpu}, gpus: {cfg.gpus if use_multi_gpu else [cfg.gpus[0]]}')
    print(f'epoch范围: {start_epoch} 到 {max_epoch-1} (共 {max_epoch-start_epoch} 个epoch)')
    
    if staged_training_config and staged_training_config.get('enabled', False):
        print("启用分阶段训练:")
        # 显示统一后的调度信息
        print("  阶段调度:")
        for s in schedule:
            print(
                f"    {s['name']}: duration {s['stage_duration']}, "
                f"range [{s['stage_start_epoch']}, {s['stage_end_epoch']}], "
                f"modules: {s['modules']}, datasets: {s['datasets']}"
            )
    
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
    contact_loss_fn = None  # 接触损失函数（当前版本未使用）
    # 非模块化训练路径已移除

    # 如果启用分阶段训练且没有提供初始数据加载器，创建第一个阶段的数据加载器
    if train_loader is None and staged_training_config and staged_training_config.get('enabled', False):
        initial_stage_info = get_training_stage(start_epoch, staged_training_config, is_debug=cfg.debug)
        create_fn = None
        main_module = sys.modules.get("__main__")
        if main_module is not None and hasattr(main_module, "create_staged_dataloaders"):
            create_fn = getattr(main_module, "create_staged_dataloaders")
        else:
            from train_IMUHOI import create_staged_dataloaders as create_fn
        train_loader, test_loader = create_fn(cfg, initial_stage_info)
        
        if train_loader is None:
            print("错误: 无法创建初始阶段的数据加载器")
            return model, optimizer
        
        print(f"已创建初始阶段 '{initial_stage_info['name']}' 的数据加载器")
        
        # 设置当前阶段信息并配置训练模块，避免在第一个epoch时重复创建数据加载器
        current_stage_info = initial_stage_info
        configure_training_modules(model, current_stage_info)
        
        # 重新创建优化器/调度器（只优化激活的参数）
        optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, initial_stage_info, use_multi_gpu)
        
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
            
            # 可选：保存上一个阶段结束检查点（当前简化版本默认不保存，避免冗余）
            if False and current_stage_info is not None and not cfg.debug:
                stage_loss = current_stage_best_loss
                save_stage_checkpoint(model, optimizer, epoch-1, current_stage_info, save_dir, stage_loss, None, "stage_end")
            
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
                
                # 根据新阶段配置需要训练/冻结的模块
                configure_training_modules(model, new_stage_info)
                
                print(f"重新构建完成，参数数量: {sum(p.numel() for p in model.parameters())}")
                
                # 删除旧模型释放内存
                del old_model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            else:
                configure_training_modules(model, new_stage_info)
            
            # 更新当前阶段信息
            current_stage_info = new_stage_info
            
            # 重新创建优化器/调度器（只优化激活的参数）
            optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, new_stage_info, use_multi_gpu)
            
            # 创建或更新数据加载器（如果启用分阶段训练）
            if staged_training_config and staged_training_config.get('enabled', False):
                train_loader, test_loader = rebuild_dataloaders_if_needed(cfg, new_stage_info, train_loader, test_loader)
                if train_loader is None:
                    return model, optimizer
        
        # current_stage_info 在分阶段路径中总会被设置

        # 训练阶段
        model.train()
        train_loss = 0
        stage_losses = defaultdict(float)
        
        train_iter = tqdm(train_loader, desc=f'Epoch {epoch} - {current_stage_info["name"]}', leave=False)
        
        for batch in train_iter:
            # 构建前向输入
            data_dict = build_model_input_dict(batch, current_stage_info, cfg, device, add_noise=True)
            
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
                    # 构建评估输入
                    data_dict_eval = build_model_input_dict(batch, current_stage_info, cfg, device, add_noise=True)
                    
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
            if stage_test_loss < current_stage_best_loss:
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
        
        # 更新学习率（使用阶段内的epoch）
        if scheduler is not None:
            scheduler.step()
            # 打印当前学习率
            current_lr = scheduler.get_last_lr()[0]
            if epoch % 10 == 0:  # 每10个epoch打印一次学习率
                print(f'\nEpoch {epoch}, Stage: {current_stage_info["name"]}, Stage Epoch: {current_stage_info["stage_epoch"]}, LR: {current_lr:.6f}')
    
    # 保存最终阶段的检查点
    if current_stage_info is not None:
        # 使用当前阶段的最佳损失值保存最终检查点
        final_stage_loss = current_stage_best_loss
        save_stage_checkpoint(model, optimizer, max_epoch-1, current_stage_info, save_dir, final_stage_loss, None, "final_stage")
    
    # 保存最终模型（基于阶段最佳损失）
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
