import os
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from datetime import datetime

from dataloader.dataloader_IMUHOI import IMUDataset
from models.do_train_IMUHOI import do_train_imu_TransPose, compute_stage_schedule
# from models.do_train_imu_TransPose_simpleObjT import do_train_imu_TransPose
from utils.parser_util import get_args, merge_file
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def create_staged_dataloaders(cfg, stage_info):
    """
    根据训练阶段创建相应的数据加载器
    
    Args:
        cfg: 配置对象
        stage_info: 当前阶段信息
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    datasets = stage_info['datasets']
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 从阶段配置中获取特定的超参数，如果没有则使用全局默认值
    stage_batch_size = stage_info.get('batch_size', getattr(cfg, 'batch_size', 50))
    stage_num_workers = stage_info.get('num_workers', getattr(cfg, 'num_workers', 12))
    
    print(f"阶段 '{stage_info['name']}' 超参数:")
    print(f"  batch_size: {stage_batch_size}")
    print(f"  num_workers: {stage_num_workers}")
    
    # 根据阶段确定数据路径
    train_paths = []
    test_paths = []
    
    # 检查是否为debug模式
    is_debug_mode = getattr(cfg, 'debug', False)
    
    for dataset_type in datasets:
        # 检查数据集配置是否存在
        if not hasattr(cfg, 'datasets') or not hasattr(cfg.datasets, dataset_type):
            raise ValueError(f"{dataset_type.upper()}数据集配置不存在")
        
        dataset_config = getattr(cfg.datasets, dataset_type)
        
        if is_debug_mode and hasattr(dataset_config, 'debug_path'):
            # Debug模式：使用debug_path作为训练和测试路径
            debug_path = os.path.join(base_dir, dataset_config.debug_path)
            train_paths.append(debug_path)
            test_paths.append(debug_path)
        else:
            # 正常模式：使用train_path和test_path
            train_paths.append(os.path.join(base_dir, dataset_config.train_path))
            if hasattr(dataset_config, 'test_path'):
                test_paths.append(os.path.join(base_dir, dataset_config.test_path))
            
    print(f"当前阶段 '{stage_info['name']}' 使用的数据集:")
    if is_debug_mode:
        print(f"  *** DEBUG模式 *** 使用调试数据集")
    print(f"  训练数据路径: {train_paths}")
    if test_paths:
        print(f"  测试数据路径: {test_paths}")
    
    # 检查debug模式下的数据集是否存在
    if is_debug_mode:
        missing_paths = []
        for path in train_paths:
            if not os.path.exists(path):
                missing_paths.append(path)
        
        if missing_paths:
            print(f"警告: 以下debug数据集路径不存在:")
            for path in missing_paths:
                print(f"  - {path}")
            print("请确保已经创建了相应的debug数据集。")
            print("提示: 可以通过以下方式创建debug数据集:")
            print("  1. 从训练集中复制少量文件到debug目录")
            print("  2. 或者修改配置文件中的debug_path指向现有的小数据集")
    
    # 创建训练数据集
    train_dataset = IMUDataset(
        data_dir=train_paths,
        window_size=cfg.train.window,
        debug=cfg.debug
    )
    
    if len(train_dataset) == 0:
        print(f"错误: 阶段 '{stage_info['name']}' 的训练数据集为空，无法继续训练。")
        print(f"请检查数据路径: {train_paths}")
        return None, None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=stage_batch_size,  # 使用阶段特定的batch_size
        shuffle=True,
        num_workers=stage_num_workers,  # 使用阶段特定的num_workers
        pin_memory=True,
        drop_last=True
    )
    
    print(f"阶段 '{stage_info['name']}' 训练数据集大小: {len(train_dataset)}")
    print(f"训练批次数量: {len(train_loader)}")
    
    # 创建测试数据集（如果有测试路径）
    test_loader = None
    if test_paths:
        test_dataset = IMUDataset(
            data_dir=test_paths,
            window_size=cfg.test.window,
            debug=cfg.debug
        )
        
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=stage_batch_size,  # 测试也使用阶段特定的batch_size
                shuffle=False,
                num_workers=stage_num_workers,  # 使用阶段特定的num_workers
                pin_memory=True,
                drop_last=False
            )
            
            print(f"阶段 '{stage_info['name']}' 测试数据集大小: {len(test_dataset)}")
            print(f"测试批次数量: {len(test_loader)}")
        else:
            print(f"警告: 阶段 '{stage_info['name']}' 的测试数据集为空")
    
    return train_loader, test_loader


def main():
    """
    TransPose模型主训练函数：解析参数、准备数据集、初始化训练
    支持分阶段训练和数据集切换，以及模块化训练
    """
    # 解析命令行参数
    cfg_args = get_args()
    args = merge_file(cfg_args)
    
    # 设置随机种子以确保可重复性
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # 设置CUDA和确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    
    # 从args中获取配置
    cfg = args  # 使用合并后的配置
    
    # 设置GPU配置
    if torch.cuda.is_available():
        # 检查可用的GPU数量
        available_gpus = torch.cuda.device_count()
        print(f"可用GPU数量: {available_gpus}")
        
        # 验证配置的GPU是否可用
        valid_gpus = [gpu for gpu in cfg.gpus if gpu < available_gpus]
        if len(valid_gpus) != len(cfg.gpus):
            print(f"警告: 配置的GPU {cfg.gpus} 中部分不可用，使用可用GPU: {valid_gpus}")
            cfg.gpus = valid_gpus
        
        # 设置多GPU配置
        cfg.use_multi_gpu = getattr(cfg, 'use_multi_gpu', True) and len(cfg.gpus) > 1
        cfg.device = f"cuda:{cfg.gpus[0]}"  # 主GPU
        
        # 设置CUDA设备
        torch.cuda.set_device(cfg.gpus[0])
        
        if cfg.use_multi_gpu:
            print(f"启用多GPU训练: {cfg.gpus}")
            # 多GPU训练时调整学习率
            cfg.lr = cfg.lr * len(cfg.gpus)  # 线性缩放学习率
            print(f"多GPU训练，学习率调整为: {cfg.lr}")
        else:
            print(f"使用单GPU训练: {cfg.gpus[0]}")
    else:
        print("CUDA不可用，使用CPU训练")
        cfg.device = "cpu"
        cfg.use_multi_gpu = False
    
    # 设置保存目录
    time_stamp = datetime.now().strftime("%m%d%H%M")
    run_name = f"transpose_{time_stamp}"
    if getattr(cfg, 'debug', False):
        run_name = f"{run_name}_debug"
    save_dir = os.path.join(cfg.save_dir, run_name)
    # Debug 下也需要保存（避免模块化训练缺失权重），此处总是创建保存目录
    cfg.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    # 创建modules子目录用于保存单个模块
    modules_dir = os.path.join(save_dir, "modules")
    os.makedirs(modules_dir, exist_ok=True)
    
    # 检查是否启用模块化训练
    modular_training_config = None
    if hasattr(cfg, 'staged_training') and cfg.staged_training.get('enabled', False):
        modular_training_config = cfg.staged_training.get('modular_training', {})
    
    # 打印训练配置
    print("=" * 50)
    print(f"模型类型: TransPose")
    print(f"GPU配置: {cfg.gpus} (多GPU: {cfg.use_multi_gpu})")
    print(f"批次大小: {cfg.batch_size}")
    print(f"学习率: {cfg.lr}")
    print(f"训练帧窗口大小: {cfg.train.window}")
    print(f"测试帧窗口大小: {cfg.test.window}")
    print(f"保存目录: {cfg.save_dir}")
    
    # 仅支持模块化分阶段训练
    staged_training_config = getattr(cfg, 'staged_training', None)
    assert staged_training_config and staged_training_config.get('enabled', False), \
        "当前精简版本仅支持分阶段训练，请在配置中启用 staged_training.enabled=True"
    assert staged_training_config.get('modular_training', {}).get('enabled', True), \
        "当前精简版本仅支持模块化分阶段训练，请在配置中启用 staged_training.modular_training.enabled=True"

    if modular_training_config and modular_training_config.get('enabled', False):
        start_stage_name = modular_training_config.get('start_from_stage', 'velocity_contact')
        print(f"\n启用模块化分阶段训练，从阶段 '{start_stage_name}' 开始:")
    else:
        print("\n分阶段训练已启用:")

    # 统一打印调度（仅新写法整数时长）
    schedule, total_epochs = compute_stage_schedule(staged_training_config, is_debug=cfg.debug)
    mode_str = "Debug" if cfg.debug else "正常"
    print(f"  [{mode_str}模式] 阶段调度 (total_epochs={total_epochs}):")
    for s in schedule:
        print(
            f"    阶段 '{s['name']}': duration {s['stage_duration']}, range [{s['stage_start_epoch']}, {s['stage_end_epoch']}], "
            f"模块: {s['modules']}, 数据集: {s['datasets']}"
        )

    print("\n开始分阶段训练...")
    model, optimizer = do_train_imu_TransPose(cfg, None, None, model=None, optimizer=None)
    
    print(f"训练完成！TransPose模型已保存到 {cfg.save_dir}")
    
    # 保存完整配置
    if not cfg.debug:
        import yaml
        with open(os.path.join(cfg.save_dir, 'training_config.yaml'), 'w') as f:
            yaml.dump(cfg.__dict__, f)
    
    return

if __name__ == "__main__":
    main() 
