#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets.imu_dataset import IMUDataset

def test_preprocess(args):
    """测试预处理输出和数据集加载"""
    
    # 检查处理后的数据目录
    print(f"检查数据目录: {args.data_dir}")
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录 {args.data_dir} 不存在")
        return
    
    # 检查BPS特征目录
    bps_dir = os.path.join(os.path.dirname(args.data_dir), "bps_features")
    print(f"检查BPS特征目录: {bps_dir}")
    if not os.path.exists(bps_dir):
        print(f"警告: BPS特征目录 {bps_dir} 不存在")
    else:
        # 统计BPS文件数量
        bps_files = [f for f in os.listdir(bps_dir) if f.endswith('.npy')]
        print(f"找到 {len(bps_files)} 个BPS特征文件")
    
    # 创建数据集
    print(f"创建数据集，窗口大小={args.window_size}")
    dataset = IMUDataset(
        data_dir=args.data_dir,
        window_size=args.window_size,
        stride=args.stride,
        normalize_imu=True
    )
    
    print(f"数据集包含 {len(dataset)} 个窗口")
    
    # 随机检查几个样本
    num_samples = min(args.num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        print(f"\n检查样本 {i+1}/{num_samples} (索引={idx})")
        sample = dataset[idx]
        
        # 检查样本信息
        print(f"序列名称: {sample['seq_name']}")
        print(f"性别: {sample['gender']}")
        print(f"窗口信息: {sample['window_info']}")
        
        # 检查IMU数据
        imu_data = sample["imu_data"]
        print(f"IMU数据形状: {imu_data.shape}")
        
        # 检查位置数据
        pos_data = sample["position_global"]
        print(f"位置数据形状: {pos_data.shape}")
        
        # 检查物体数据 (如果存在)
        if "obj_trans" in sample:
            print("物体数据:")
            print(f"  物体位置形状: {sample['obj_trans'].shape}")
            print(f"  物体旋转形状: {sample['obj_rot_mat'].shape}")
            print(f"  物体缩放形状: {sample['obj_scale'].shape}")
            
            # 检查BPS特征
            if "obj_bps" in sample:
                obj_bps = sample["obj_bps"]
                print(f"  物体BPS特征形状: {obj_bps.shape}")
                # 检查BPS特征是否包含NaN或Inf
                if torch.isnan(obj_bps).any() or torch.isinf(obj_bps).any():
                    print("  警告: BPS特征包含NaN或Inf值")
        
        # 可视化第一个样本的一些数据
        if i == 0 and args.visualize:
            visualize_sample(sample)
    
    print("\n测试完成")

def visualize_sample(sample):
    """可视化样本数据"""
    plt.figure(figsize=(15, 10))
    
    # 绘制IMU加速度数据
    imu_data = sample["imu_data"]
    num_imus = imu_data.shape[1] // 6
    
    plt.subplot(2, 2, 1)
    for i in range(num_imus):
        # 绘制前3个IMU传感器的加速度x分量
        plt.plot(imu_data[:, i*6], label=f"IMU {i} acc-x")
    plt.title("IMU 加速度 (X轴)")
    plt.xlabel("帧")
    plt.ylabel("加速度")
    plt.legend()
    
    # 绘制位置数据
    plt.subplot(2, 2, 2)
    pos_data = sample["position_global"]
    for i in range(min(3, pos_data.shape[1])):
        plt.plot(pos_data[:, i, 0], label=f"Joint {i} pos-x")
    plt.title("关节位置 (X轴)")
    plt.xlabel("帧")
    plt.ylabel("位置")
    plt.legend()
    
    # 如果有物体数据，绘制物体位置
    if "obj_trans" in sample:
        plt.subplot(2, 2, 3)
        obj_trans = sample["obj_trans"]
        plt.plot(obj_trans[:, 0], label="X")
        plt.plot(obj_trans[:, 1], label="Y")
        plt.plot(obj_trans[:, 2], label="Z")
        plt.title("物体位置")
        plt.xlabel("帧")
        plt.ylabel("位置")
        plt.legend()
        
        # 绘制物体BPS特征的一部分
        if "obj_bps" in sample:
            plt.subplot(2, 2, 4)
            obj_bps = sample["obj_bps"]
            # 只绘制前3个维度
            for i in range(min(3, obj_bps.shape[1])):
                plt.plot(obj_bps[:, i], label=f"BPS dim {i}")
            plt.title("物体BPS特征")
            plt.xlabel("帧")
            plt.ylabel("特征值")
            plt.legend()
    
    plt.tight_layout()
    plt.savefig("sample_visualization.png")
    print(f"可视化结果已保存为 sample_visualization.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试预处理输出和数据集加载")
    parser.add_argument("--data_dir", type=str, default="processed_data/train",
                        help="预处理数据目录")
    parser.add_argument("--window_size", type=int, default=120,
                        help="窗口大小")
    parser.add_argument("--stride", type=int, default=60,
                        help="窗口滑动步长")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="要检查的样本数量")
    parser.add_argument("--visualize", action="store_true",
                        help="是否可视化数据")
    
    args = parser.parse_args()
    test_preprocess(args) 