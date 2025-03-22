import os
import torch
import argparse
import yaml
import numpy as np
from pathlib import Path
from datasets.imu_dataset import IMUDataset
from modules.model import IMUPoseGenerationModel
from utils.trainer import IMUPoseTrainer
from utils.data_utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description='IMU姿态生成训练脚本')
    
    # 数据参数
    parser.add_argument('--train_dir', type=str, required=True, help='训练数据目录')
    parser.add_argument('--val_dir', type=str, default=None, help='验证数据目录')
    parser.add_argument('--window_size', type=int, default=120, help='序列窗口大小')
    parser.add_argument('--window_stride', type=int, default=30, help='窗口滑动步长')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪值')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=6, help='Transformer层数')
    parser.add_argument('--nhead', type=int, default=8, help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--diffusion_steps', type=int, default=1000, help='扩散步数')
    parser.add_argument('--bps_n_points', type=int, default=1024, help='BPS点数')
    parser.add_argument('--use_object', action='store_true', help='是否使用物体信息')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--log_dir', type=str, default=None, help='日志目录')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='检查点目录')
    parser.add_argument('--save_interval', type=int, default=10, help='保存间隔(轮次)')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    
    args = parser.parse_args()
    
    # 如果提供了配置文件，从YAML加载配置
    if args.config is not None and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # 更新参数
        for key, value in config.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)
    
    # 设置输出目录
    if args.log_dir is None:
        args.log_dir = os.path.join(args.output_dir, 'logs')
    
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    
    return args

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 保存配置
    config_path = os.path.join(args.output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(vars(args), f)
    
    # 加载数据集
    print(f"加载训练数据集: {args.train_dir}")
    train_dataset = IMUDataset(
        data_dir=args.train_dir, 
        window_size=args.window_size,
        window_stride=args.window_stride,
        use_object=args.use_object,
        bps_n_points=args.bps_n_points
    )
    
    if args.val_dir:
        print(f"加载验证数据集: {args.val_dir}")
        val_dataset = IMUDataset(
            data_dir=args.val_dir, 
            window_size=args.window_size,
            window_stride=args.window_size,  # 验证集使用不重叠的窗口
            use_object=args.use_object,
            bps_n_points=args.bps_n_points
        )
    else:
        val_dataset = None
    
    # 创建模型
    model_config = {
        'input_dim': 144,  # SMPL姿态参数维度
        'imu_dim': 42,     # 7关节 x (3轴加速度 + 3轴旋转)
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'nhead': args.nhead,
        'dropout': args.dropout,
        'diffusion_steps': args.diffusion_steps,
        'use_object': args.use_object,
        'max_seq_len': args.window_size
    }
    
    print("创建模型...")
    model = IMUPoseGenerationModel.from_config(model_config)
    model = model.to(device)
    
    # 打印模型信息
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 创建训练器
    trainer = IMUPoseTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_value=args.grad_clip,
        device=device,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.save_interval
    )
    
    # 恢复训练
    start_epoch = 1
    if args.resume:
        print(f"从检查点恢复训练: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume) + 1
    
    # 开始训练
    print(f"开始训练，共 {args.num_epochs} 轮...")
    trainer.train(args.num_epochs - start_epoch + 1)
    
    print("训练完成!")

if __name__ == "__main__":
    main() 