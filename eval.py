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
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='IMU姿态生成评估脚本')
    
    # 数据参数
    parser.add_argument('--test_dir', type=str, required=True, help='测试数据目录')
    parser.add_argument('--window_size', type=int, default=120, help='序列窗口大小')
    
    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--config', type=str, required=True, help='训练配置文件路径')
    parser.add_argument('--use_object', action='store_true', help='是否使用物体信息')
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_samples', type=int, default=10, help='生成样本数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='评估结果输出目录')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    
    args = parser.parse_args()
    
    # 从配置文件中加载模型配置
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # 只加载模型相关参数
        model_params = ['hidden_dim', 'num_layers', 'nhead', 'dropout', 
                        'diffusion_steps', 'bps_n_points']
        
        for param in model_params:
            if param in config and not hasattr(args, param):
                setattr(args, param, config[param])
    
    return args

def compute_metrics(pred, gt, padding_mask=None):
    """计算预测与真实值之间的评估指标"""
    metrics = {}
    
    # 确保为张量
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)
    if not isinstance(gt, torch.Tensor):
        gt = torch.tensor(gt)
    
    # 应用padding_mask
    if padding_mask is not None:
        pred = pred[padding_mask]
        gt = gt[padding_mask]
    
    # 平均绝对误差
    mae = torch.mean(torch.abs(pred - gt)).item()
    metrics['mae'] = mae
    
    # 均方误差
    mse = torch.mean((pred - gt) ** 2).item()
    metrics['mse'] = mse
    
    # 均方根误差
    rmse = torch.sqrt(mse).item()
    metrics['rmse'] = rmse
    
    return metrics

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
    
    # 加载测试数据集
    print(f"加载测试数据集: {args.test_dir}")
    test_dataset = IMUDataset(
        data_dir=args.test_dir, 
        window_size=args.window_size,
        window_stride=args.window_size,  # 测试集不重叠窗口
        use_object=args.use_object,
        bps_n_points=getattr(args, 'bps_n_points', 1024)
    )
    
    # 创建数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 重建模型
    model_config = {
        'input_dim': 144,  # SMPL姿态参数维度
        'imu_dim': 42,     # 7关节 x (3轴加速度 + 3轴旋转)
        'hidden_dim': getattr(args, 'hidden_dim', 512),
        'num_layers': getattr(args, 'num_layers', 6),
        'nhead': getattr(args, 'nhead', 8),
        'dropout': getattr(args, 'dropout', 0.1),
        'diffusion_steps': getattr(args, 'diffusion_steps', 1000),
        'use_object': args.use_object,
        'max_seq_len': args.window_size
    }
    
    print("创建模型...")
    model = IMUPoseGenerationModel.from_config(model_config)
    model = model.to(device)
    
    # 加载检查点
    print(f"加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置为评估模式
    model.eval()
    
    # 评估指标
    all_metrics = []
    
    # 生成和评估样本
    print(f"生成 {args.num_samples} 个样本...")
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(test_loader)):
            if i >= args.num_samples:
                break
            
            # 准备条件数据
            cond_data = {
                'imu_data': batch_data['imu_data'].to(device)
            }
            
            # 如果有物体数据
            if 'obj_trans' in batch_data and args.use_object:
                cond_data['obj_trans'] = batch_data['obj_trans'].to(device)
                cond_data['obj_bps'] = batch_data['obj_bps'].to(device)
            
            # 如果有填充掩码
            padding_mask = None
            if 'padding_mask' in batch_data:
                padding_mask = batch_data['padding_mask'].to(device)
            
            # 生成样本
            generated = model.sample(cond_data, padding_mask)
            
            # 获取真实值
            ground_truth = batch_data['pose_params'].to(device)
            
            # 计算评估指标
            batch_metrics = compute_metrics(generated, ground_truth, padding_mask)
            all_metrics.append(batch_metrics)
            
            # 保存生成的样本
            sample_output = {
                'generated': generated.cpu().numpy(),
                'ground_truth': ground_truth.cpu().numpy(),
                'metrics': batch_metrics
            }
            
            # 保存条件数据
            for key, value in cond_data.items():
                sample_output[f'cond_{key}'] = value.cpu().numpy()
            
            if padding_mask is not None:
                sample_output['padding_mask'] = padding_mask.cpu().numpy()
            
            # 保存样本
            np.save(os.path.join(args.output_dir, f'sample_{i}.npy'), sample_output)
    
    # 计算平均指标
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        avg_metrics[metric] = np.mean([m[metric] for m in all_metrics])
    
    # 保存评估结果
    results = {
        'average_metrics': avg_metrics,
        'all_metrics': all_metrics
    }
    
    np.save(os.path.join(args.output_dir, 'evaluation_results.npy'), results)
    
    # 打印结果
    print("\n评估结果:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 