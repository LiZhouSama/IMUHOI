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

from dataloader.dataloader import IMUDataset
from models.do_train_imu_TransPose import do_train_imu_TransPose, load_transpose_model
# from models.do_train_imu_TransPose_humanOnly import do_train_imu_TransPose_humanOnly, load_transpose_model_humanOnly
from utils.parser_util import get_args, merge_file


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
    
    # 根据阶段确定数据路径
    train_paths = []
    test_paths = []
    
    # 检查是否为debug模式
    is_debug_mode = getattr(cfg, 'debug', False)
    
    for dataset_type in datasets:
        if dataset_type == 'amass':
            # AMASS数据集路径
            if hasattr(cfg, 'datasets') and hasattr(cfg.datasets, 'amass'):
                if is_debug_mode and hasattr(cfg.datasets.amass, 'debug_path'):
                    # Debug模式：使用debug_path作为训练和测试路径
                    debug_path = os.path.join(base_dir, cfg.datasets.amass.debug_path)
                    train_paths.append(debug_path)
                    test_paths.append(debug_path)
                else:
                    # 正常模式：使用train_path和test_path
                    train_paths.append(os.path.join(base_dir, cfg.datasets.amass.train_path))
                    if hasattr(cfg.datasets.amass, 'test_path'):
                        test_paths.append(os.path.join(base_dir, cfg.datasets.amass.test_path))
            else:
                # 默认AMASS路径
                if is_debug_mode:
                    debug_path = os.path.join(base_dir, "processed_amass_data_0703/debug")
                    train_paths.append(debug_path)
                    test_paths.append(debug_path)
                else:
                    train_paths.append(os.path.join(base_dir, "processed_amass_data_0703/train"))
                    test_paths.append(os.path.join(base_dir, "processed_amass_data_0703/test"))
        elif dataset_type == 'omomo':
            # OMOMO数据集路径
            if hasattr(cfg, 'datasets') and hasattr(cfg.datasets, 'omomo'):
                if is_debug_mode and hasattr(cfg.datasets.omomo, 'debug_path'):
                    # Debug模式：使用debug_path作为训练和测试路径
                    debug_path = os.path.join(base_dir, cfg.datasets.omomo.debug_path)
                    train_paths.append(debug_path)
                    test_paths.append(debug_path)
                else:
                    # 正常模式：使用train_path和test_path
                    train_paths.append(os.path.join(base_dir, cfg.datasets.omomo.train_path))
                    if hasattr(cfg.datasets.omomo, 'test_path'):
                        test_paths.append(os.path.join(base_dir, cfg.datasets.omomo.test_path))
            else:
                # 默认OMOMO路径
                if is_debug_mode:
                    debug_path = os.path.join(base_dir, "processed_data_0701/debug")
                    train_paths.append(debug_path)
                    test_paths.append(debug_path)
                else:
                    train_paths.append(os.path.join(base_dir, "processed_data_0701/train"))
                    test_paths.append(os.path.join(base_dir, "processed_data_0701/test"))
        elif dataset_type == 'mixed':
            # 混合数据集：同时使用AMASS和OMOMO
            if hasattr(cfg, 'datasets'):
                # 处理AMASS部分
                if hasattr(cfg.datasets, 'amass'):
                    if is_debug_mode and hasattr(cfg.datasets.amass, 'debug_path'):
                        debug_path = os.path.join(base_dir, cfg.datasets.amass.debug_path)
                        train_paths.append(debug_path)
                        test_paths.append(debug_path)
                    else:
                        train_paths.append(os.path.join(base_dir, cfg.datasets.amass.train_path))
                        if hasattr(cfg.datasets.amass, 'test_path'):
                            test_paths.append(os.path.join(base_dir, cfg.datasets.amass.test_path))
                # 处理OMOMO部分
                if hasattr(cfg.datasets, 'omomo'):
                    if is_debug_mode and hasattr(cfg.datasets.omomo, 'debug_path'):
                        debug_path = os.path.join(base_dir, cfg.datasets.omomo.debug_path)
                        train_paths.append(debug_path)
                        test_paths.append(debug_path)
                    else:
                        train_paths.append(os.path.join(base_dir, cfg.datasets.omomo.train_path))
                        if hasattr(cfg.datasets.omomo, 'test_path'):
                            test_paths.append(os.path.join(base_dir, cfg.datasets.omomo.test_path))
            else:
                # 默认混合路径
                if is_debug_mode:
                    train_paths.extend([
                        os.path.join(base_dir, "processed_amass_data_0703/debug"),
                        os.path.join(base_dir, "processed_data_0701/debug")
                    ])
                    test_paths.extend([
                        os.path.join(base_dir, "processed_amass_data_0703/debug"),
                        os.path.join(base_dir, "processed_data_0701/debug")
                    ])
                else:
                    train_paths.extend([
                        os.path.join(base_dir, "processed_amass_data_0703/train"),
                        os.path.join(base_dir, "processed_data_0701/train")
                    ])
                    test_paths.extend([
                        os.path.join(base_dir, "processed_amass_data_0703/test"),
                        os.path.join(base_dir, "processed_data_0701/test")
                    ])
    
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
        normalize=cfg.train.normalize,
        debug=cfg.debug
    )
    
    if len(train_dataset) == 0:
        print(f"错误: 阶段 '{stage_info['name']}' 的训练数据集为空，无法继续训练。")
        print(f"请检查数据路径: {train_paths}")
        return None, None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
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
            normalize=cfg.test.normalize,
            debug=cfg.debug
        )
        
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
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
    支持分阶段训练和数据集切换
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
    save_dir = os.path.join(cfg.save_dir, f"transpose_{time_stamp}")
    cfg.save_dir = save_dir if not cfg.debug else None
    if not cfg.debug:
        os.makedirs(save_dir, exist_ok=True)
    
    # 打印训练配置
    print("=" * 50)
    print(f"模型类型: TransPose")
    print(f"GPU配置: {cfg.gpus} (多GPU: {cfg.use_multi_gpu})")
    print(f"批次大小: {cfg.batch_size}")
    print(f"学习率: {cfg.lr}")
    print(f"训练帧窗口大小: {cfg.train.window}")
    print(f"测试帧窗口大小: {cfg.test.window}")
    print(f"保存目录: {cfg.save_dir}")
    
    # 检查是否启用分阶段训练
    staged_training_config = getattr(cfg, 'staged_training', None)
    if staged_training_config and staged_training_config.get('enabled', False):
        print("\n分阶段训练已启用:")
        
        # 根据是否为debug模式显示相应的stages
        if cfg.debug and 'debug_stages' in staged_training_config:
            print("  [Debug模式] 使用debug_stages配置:")
            for stage in staged_training_config.get('debug_stages', []):
                print(f"    阶段 '{stage['name']}': epochs {stage['epochs']}, 模块: {stage['modules']}, 数据集: {stage['datasets']}")
        else:
            print("  [正常模式] 使用stages配置:")
            for stage in staged_training_config.get('stages', []):
                print(f"    阶段 '{stage['name']}': epochs {stage['epochs']}, 模块: {stage['modules']}, 数据集: {stage['datasets']}")
        
        # 分阶段训练模式
        print("\n开始分阶段训练...")
        
        # 检查是否从预训练模型继续训练
        model = None
        optimizer = None
        if hasattr(cfg, 'pretrained_checkpoint') and cfg.pretrained_checkpoint:
            pretrained_path = cfg.pretrained_checkpoint
            print(f"从预训练模型继续训练: {pretrained_path}")
            model = load_transpose_model(cfg, pretrained_path)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        
        # 创建一个包装函数来处理分阶段数据加载
        def staged_train_wrapper():
            # 这个函数将在训练过程中被调用，根据不同阶段提供不同的数据加载器
            # 实际的数据加载器创建将在do_train_imu_TransPose中处理
            return None, None
        
        # 开始分阶段训练（数据加载器将在训练过程中动态创建）
        model, optimizer = do_train_imu_TransPose(cfg, None, None, model=model, optimizer=optimizer)
        
    else:
        # 传统训练模式：使用固定的数据集
        print("\n使用传统训练模式（非分阶段）...")
        
        # 设置数据集路径 - 使用绝对路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 检查是否使用小型数据集进行快速测试
        if cfg.debug:
            print("Debug模式：使用小型测试数据集进行快速开发...")
            # 优先使用配置的debug路径，如果没有则使用传统的debug_data_path
            if hasattr(cfg, 'datasets'):
                debug_paths = []
                if hasattr(cfg.datasets, 'omomo') and hasattr(cfg.datasets.omomo, 'debug_path'):
                    debug_paths.append(os.path.join(base_dir, cfg.datasets.omomo.debug_path))
                elif hasattr(cfg.train, 'debug_data_path'):
                    debug_paths.append(os.path.join(base_dir, cfg.train.debug_data_path))
                
                if debug_paths:
                    train_path = debug_paths
                    test_path = debug_paths
                else:
                    # 如果都没有配置，使用默认的debug路径
                    train_path = os.path.join(base_dir, "processed_data_0701/debug")
                    test_path = train_path
            else:
                # 如果没有datasets配置，使用传统方式
                if hasattr(cfg.train, 'debug_data_path'):
                    train_path = os.path.join(base_dir, cfg.train.debug_data_path)
                else:
                    train_path = os.path.join(base_dir, "processed_data_0701/debug")
                test_path = train_path
        else:
            # 处理多个数据路径（支持列表格式）
            if isinstance(cfg.train.data_path, list):
                # 如果是列表，将每个路径与base_dir拼接
                train_path = [os.path.join(base_dir, path) for path in cfg.train.data_path]
            else:
                # 如果是字符串，保持原有逻辑
                train_path = os.path.join(base_dir, cfg.train.data_path)
            
            # 处理测试数据路径
            if hasattr(cfg.test, 'data_path') and cfg.test.data_path:
                if isinstance(cfg.test.data_path, list):
                    # 如果是列表，将每个路径与base_dir拼接
                    test_path = [os.path.join(base_dir, path) for path in cfg.test.data_path]
                else:
                    # 如果是字符串，保持原有逻辑
                    test_path = os.path.join(base_dir, cfg.test.data_path)
            else:
                test_path = None
        
        print(f"训练数据路径: {train_path}")
        if test_path:
            print(f"测试数据路径: {test_path}")
        
        # 加载训练数据集
        print("准备训练数据集...")
        train_dataset = IMUDataset(
            data_dir=train_path,
            window_size=cfg.train.window,
            normalize=cfg.train.normalize,
            debug=cfg.debug
        )
        
        # 如果数据集为空，则无法继续
        if len(train_dataset) == 0:
            print("错误: 训练数据集为空，无法继续训练。请检查数据路径和数据格式。")
            return
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        print(f"训练数据集大小: {len(train_dataset)}")
        print(f"每批次数据大小: {cfg.batch_size}")
        print(f"训练批次数量: {len(train_loader)}")
        
        # 加载测试数据集（如果有）
        test_loader = None
        if test_path:
            print("准备测试数据集...")
            test_dataset = IMUDataset(
                data_dir=test_path,
                window_size=cfg.test.window,
                normalize=cfg.test.normalize,
                debug=cfg.debug
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=False
            )
            
            print(f"测试数据集大小: {len(test_dataset)}")
            print(f"测试批次数量: {len(test_loader)}")
        
        # 检查是否从预训练模型继续训练
        if hasattr(cfg, 'pretrained_checkpoint') and cfg.pretrained_checkpoint:
            pretrained_path = cfg.pretrained_checkpoint
            print(f"从预训练模型继续训练: {pretrained_path}")
            # 加载预训练模型
            model = load_transpose_model(cfg, pretrained_path)
            # 设置优化器
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            # 开始训练
            model, optimizer = do_train_imu_TransPose(cfg, train_loader, test_loader, model=model, optimizer=optimizer)
        else:
            # 开始训练
            print("开始TransPose模型训练过程...")
            model, optimizer = do_train_imu_TransPose(cfg, train_loader, test_loader)
    
    print(f"训练完成！TransPose模型已保存到 {cfg.save_dir}")
    
    # 保存完整配置
    if not cfg.debug:
        import yaml
        with open(os.path.join(cfg.save_dir, 'training_config.yaml'), 'w') as f:
            yaml.dump(cfg.__dict__, f)
    
    return

if __name__ == "__main__":
    main() 