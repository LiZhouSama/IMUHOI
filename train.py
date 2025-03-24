import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import yaml
from pathlib import Path
from datasets.imu_dataset import IMUDataset
from modules.model import IMUPoseGenerationModel
from utils.trainer import IMUPoseTrainer

# 设置种子以确保可重复性
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 工作进程初始化函数
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def parse_args():
    parser = argparse.ArgumentParser(description="训练IMU姿态生成模型")
    
    # 数据参数
    parser.add_argument('--train_dir', type=str, default="processed_data_0322/train", help='训练数据目录路径')
    parser.add_argument('--val_dir', type=str, default=None, help='验证数据目录路径(可选)')
    parser.add_argument('--window_size', type=int, default=120, help='序列窗口大小')
    parser.add_argument('--stride', type=int, default=30, help='窗口滑动步长')
    parser.add_argument('--normalize_imu', action='store_true', help='是否归一化IMU数据')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=6, help='Transformer层数')
    parser.add_argument('--nhead', type=int, default=8, help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--diffusion_steps', type=int, default=1000, help='扩散模型步数')
    parser.add_argument('--use_object', action='store_true', help='是否使用物体信息')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='权重衰减')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--save_interval', type=int, default=10, help='保存间隔(轮次)')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--config', type=str, default=None, help='YAML配置文件路径')
    
    args = parser.parse_args()
    
    # 如果提供了配置文件，从文件加载参数
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # 更新参数
        args_dict = vars(args)
        for key, value in config.items():
            if value is not None:  # 只更新配置中存在的参数
                args_dict[key] = value
    
    return args

def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置多进程起始方法为spawn（解决CUDA在子进程中的问题）
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn', force=True)
        # 减少内存碎片
        torch.cuda.empty_cache()
        # 预热GPU
        torch.zeros(1).cuda()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建训练数据集
    print("创建训练数据集...")
    train_dataset = IMUDataset(
        data_dir=args.train_dir,
        window_size=args.window_size,
        stride=args.stride,
        normalize_imu=args.normalize_imu,
        mode="train"
    )
    
    print("创建数据加载器...")
    # 配置数据加载器，正确处理多进程
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if args.num_workers > 0 else 0,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0
    )
    
    # 创建验证集(如果指定了验证数据目录)
    val_loader = None
    if args.val_dir:
        print("创建验证数据集...")
        val_dataset = IMUDataset(
            data_dir=args.val_dir,
            window_size=args.window_size,
            stride=args.stride,
            normalize_imu=args.normalize_imu,
            mode="val"
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers if args.num_workers > 0 else 0,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0
        )
    
    # 获取第一个批次以确定输入维度
    sample_batch = next(iter(train_loader))
    imu_dim = sample_batch['imu_data'].shape[-1]
    print(f"IMU数据维度: {imu_dim}")
    
    # 创建模型
    print("创建模型...")
    model = IMUPoseGenerationModel(
        input_dim=66,           # 姿态参数维度(root_orient(3) + pose_body(63))
        imu_dim=imu_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        diffusion_steps=args.diffusion_steps,
        use_object=args.use_object,
        max_seq_len=args.window_size
    )
    
    # 使用PyTorch 2.0编译(如果可用)
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            model = torch.compile(model)
            print("使用PyTorch 2.0编译优化模型")
        except:
            print("PyTorch编译不可用，使用标准模型")
    
    # 计算模型参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {num_params:,}")
    
    # 创建训练器
    print("创建训练器...")
    trainer = IMUPoseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        output_dir=args.output_dir,
        save_interval=args.save_interval
    )
    
    # 恢复训练(如果指定了检查点路径)
    if args.resume:
        print(f"从检查点恢复训练: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    print(f"开始训练，共{args.num_epochs}轮...")
    trainer.train(args.num_epochs)
    
    print("\n训练完成！")

if __name__ == "__main__":
    main() 