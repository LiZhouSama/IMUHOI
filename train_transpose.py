import os
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from datetime import datetime

from dataloader.dataloader import IMUDataset
from models.do_train_imu_TransPose import do_train_imu_TransPose, load_transpose_model
# from models.do_train_imu_TransPose_humanOnly import do_train_imu_TransPose_humanOnly, load_transpose_model_humanOnly
from utils.parser_util import get_args, merge_file


def main():
    """
    TransPose模型主训练函数：解析参数、准备数据集、初始化训练
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
    
    # 设置保存目录
    time_stamp = datetime.now().strftime("%m%d%H%M")
    save_dir = os.path.join(cfg.save_dir, f"transpose_{time_stamp}")
    cfg.save_dir = save_dir if not cfg.debug else None
    if not cfg.debug:
        os.makedirs(save_dir, exist_ok=True)
    
    # 打印训练配置
    print("=" * 50)
    print(f"模型类型: TransPose")
    print(f"批次大小: {cfg.batch_size}")
    print(f"训练帧窗口大小: {cfg.train.window}")
    print(f"测试帧窗口大小: {cfg.test.window}")
    print(f"保存目录: {cfg.save_dir}")
    
    # 打印损失权重信息（如果有）
    if hasattr(cfg, 'loss_weights'):
        print("\n损失权重配置:")
        print(f"姿态损失权重: {cfg.loss_weights.rot}")
        print(f"根节点位置损失权重: {cfg.loss_weights.root_pos}")
        print(f"物体旋转损失权重: {cfg.loss_weights.obj_rot}")
        print(f"叶子节点位置损失权重: {cfg.loss_weights.leaf_pos}")
        print(f"全身关节位置损失权重: {cfg.loss_weights.full_pos}")
        print(f"根关节速度损失权重: {cfg.loss_weights.root_vel}")
    print("=" * 50)

    # 设置数据集路径 - 使用绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 检查是否使用小型数据集进行快速测试
    use_small_dataset = os.path.exists(os.path.join(base_dir, cfg.train.debug_data_path)) and cfg.debug
    
    if use_small_dataset:
        print("使用小型测试数据集进行快速开发...")
        train_path = os.path.join(base_dir, cfg.train.debug_data_path)
        test_path = os.path.join(base_dir, cfg.train.debug_data_path)  # 使用相同数据进行测试
    else:
        train_path = os.path.join(base_dir, cfg.train.data_path)
        test_path = os.path.join(base_dir, cfg.test.data_path) if hasattr(cfg.test, 'data_path') else None
    
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
    import yaml
    with open(os.path.join(cfg.save_dir, 'training_config.yaml'), 'w') as f:
        yaml.dump(cfg.__dict__, f)
    
    return


if __name__ == "__main__":
    main() 