#!/usr/bin/env python3
"""
分析AMASS和OMOMO数据集中足部接触标签的分布差异
帮助诊断足部接触loss激增的问题
"""

import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import matplotlib as mpl
mpl.rcParams['font.family'] = 'WenQuanYi Zen Hei'      # 使用文泉驿正黑:contentReference[oaicite:5]{index=5}
mpl.rcParams['axes.unicode_minus'] = False       # 解决负号 '-' 显示为方块的问题

def analyze_foot_contact_distribution(data_dir, dataset_type, sample_limit=100):
    """
    分析指定数据集中的足部接触标签分布
    
    Args:
        data_dir: 数据目录路径
        dataset_type: 数据集类型 ("amass" 或 "omomo")
        sample_limit: 分析的样本数量限制
    
    Returns:
        分析结果字典
    """
    print(f"\n=== 分析 {dataset_type.upper()} 数据集足部接触分布 ===")
    print(f"数据目录: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在: {data_dir}")
        return None
    
    # 查找所有.pt文件
    pt_files = glob.glob(os.path.join(data_dir, "*.pt"))
    if len(pt_files) == 0:
        print(f"警告: 在 {data_dir} 中未找到.pt文件")
        return None
    
    print(f"找到 {len(pt_files)} 个序列文件")
    
    # 限制分析的文件数量
    if len(pt_files) > sample_limit:
        pt_files = pt_files[:sample_limit]
        print(f"限制分析前 {sample_limit} 个文件")
    
    # 统计数据
    stats = {
        'total_sequences': 0,
        'total_frames': 0,
        'lfoot_contact_frames': 0,
        'rfoot_contact_frames': 0,
        'both_feet_contact_frames': 0,
        'no_contact_frames': 0,
        'sequence_lengths': [],
        'lfoot_contact_ratios': [],
        'rfoot_contact_ratios': [],
        'foot_speed_stats': {'lfoot': [], 'rfoot': []},
        'contact_transition_counts': 0
    }
    
    print("开始分析序列...")
    for i, file_path in enumerate(pt_files):
        try:
            if i % 20 == 0:
                print(f"处理进度: {i+1}/{len(pt_files)}")
            
            # 加载数据
            data = torch.load(file_path, map_location='cpu')
            
            # 检查必要的键
            if 'lfoot_contact' not in data or 'rfoot_contact' not in data:
                continue
            
            lfoot_contact = data['lfoot_contact']
            rfoot_contact = data['rfoot_contact']
            
            # 确保数据格式正确
            if isinstance(lfoot_contact, torch.Tensor):
                lfoot_contact = lfoot_contact.float()
            else:
                lfoot_contact = torch.tensor(lfoot_contact, dtype=torch.float32)
                
            if isinstance(rfoot_contact, torch.Tensor):
                rfoot_contact = rfoot_contact.float()
            else:
                rfoot_contact = torch.tensor(rfoot_contact, dtype=torch.float32)
            
            seq_len = len(lfoot_contact)
            stats['total_sequences'] += 1
            stats['total_frames'] += seq_len
            stats['sequence_lengths'].append(seq_len)
            
            # 统计接触帧数
            lfoot_contact_count = (lfoot_contact > 0.5).sum().item()
            rfoot_contact_count = (rfoot_contact > 0.5).sum().item()
            both_contact_count = ((lfoot_contact > 0.5) & (rfoot_contact > 0.5)).sum().item()
            no_contact_count = ((lfoot_contact <= 0.5) & (rfoot_contact <= 0.5)).sum().item()
            
            stats['lfoot_contact_frames'] += lfoot_contact_count
            stats['rfoot_contact_frames'] += rfoot_contact_count
            stats['both_feet_contact_frames'] += both_contact_count
            stats['no_contact_frames'] += no_contact_count
            
            # 计算接触比例
            lfoot_ratio = lfoot_contact_count / seq_len
            rfoot_ratio = rfoot_contact_count / seq_len
            stats['lfoot_contact_ratios'].append(lfoot_ratio)
            stats['rfoot_contact_ratios'].append(rfoot_ratio)
            
            # 计算足部速度统计（如果有位置数据）
            if 'position_global_full_gt_world' in data:
                positions = data['position_global_full_gt_world']
                if seq_len > 1:
                    lfoot_pos = positions[:, 7, :]  # 左脚踝
                    rfoot_pos = positions[:, 8, :]  # 右脚踝
                    
                    lfoot_vel = torch.norm(lfoot_pos[1:] - lfoot_pos[:-1], dim=1)
                    rfoot_vel = torch.norm(rfoot_pos[1:] - rfoot_pos[:-1], dim=1)
                    
                    stats['foot_speed_stats']['lfoot'].extend(lfoot_vel.tolist())
                    stats['foot_speed_stats']['rfoot'].extend(rfoot_vel.tolist())
            
            # 统计接触状态转换次数
            lfoot_transitions = ((lfoot_contact[1:] > 0.5) != (lfoot_contact[:-1] > 0.5)).sum().item()
            rfoot_transitions = ((rfoot_contact[1:] > 0.5) != (rfoot_contact[:-1] > 0.5)).sum().item()
            stats['contact_transition_counts'] += lfoot_transitions + rfoot_transitions
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue
    
    if stats['total_sequences'] == 0:
        print("错误: 没有成功处理任何序列")
        return None
    
    # 计算汇总统计
    stats['avg_sequence_length'] = np.mean(stats['sequence_lengths'])
    stats['lfoot_contact_ratio'] = stats['lfoot_contact_frames'] / stats['total_frames']
    stats['rfoot_contact_ratio'] = stats['rfoot_contact_frames'] / stats['total_frames']
    stats['both_feet_contact_ratio'] = stats['both_feet_contact_frames'] / stats['total_frames']
    stats['no_contact_ratio'] = stats['no_contact_frames'] / stats['total_frames']
    stats['avg_transitions_per_frame'] = stats['contact_transition_counts'] / stats['total_frames']
    
    if stats['foot_speed_stats']['lfoot']:
        stats['lfoot_speed_mean'] = np.mean(stats['foot_speed_stats']['lfoot'])
        stats['lfoot_speed_std'] = np.std(stats['foot_speed_stats']['lfoot'])
        stats['rfoot_speed_mean'] = np.mean(stats['foot_speed_stats']['rfoot'])
        stats['rfoot_speed_std'] = np.std(stats['foot_speed_stats']['rfoot'])
    
    return stats


def print_analysis_results(amass_stats, omomo_stats):
    """打印分析结果对比"""
    print("\n" + "="*60)
    print("足部接触分布分析结果对比")
    print("="*60)
    
    if amass_stats is None and omomo_stats is None:
        print("错误: 两个数据集都无法分析")
        return
    
    datasets = []
    if amass_stats: datasets.append(("AMASS", amass_stats))
    if omomo_stats: datasets.append(("OMOMO", omomo_stats))
    
    for name, stats in datasets:
        print(f"\n{name} 数据集统计:")
        print(f"  总序列数: {stats['total_sequences']}")
        print(f"  总帧数: {stats['total_frames']}")
        print(f"  平均序列长度: {stats['avg_sequence_length']:.1f}")
        print(f"  左脚接触比例: {stats['lfoot_contact_ratio']:.3f}")
        print(f"  右脚接触比例: {stats['rfoot_contact_ratio']:.3f}")
        print(f"  双脚接触比例: {stats['both_feet_contact_ratio']:.3f}")
        print(f"  无接触比例: {stats['no_contact_ratio']:.3f}")
        print(f"  每帧平均状态转换次数: {stats['avg_transitions_per_frame']:.4f}")
        
        if 'lfoot_speed_mean' in stats:
            print(f"  左脚平均速度: {stats['lfoot_speed_mean']:.4f} ± {stats['lfoot_speed_std']:.4f}")
            print(f"  右脚平均速度: {stats['rfoot_speed_mean']:.4f} ± {stats['rfoot_speed_std']:.4f}")
    
    # 如果两个数据集都有数据，进行对比
    if len(datasets) == 2:
        print(f"\n数据集对比:")
        amass_name, amass_stats = datasets[0]
        omomo_name, omomo_stats = datasets[1]
        
        print(f"  左脚接触比例差异: {abs(amass_stats['lfoot_contact_ratio'] - omomo_stats['lfoot_contact_ratio']):.3f}")
        print(f"  右脚接触比例差异: {abs(amass_stats['rfoot_contact_ratio'] - omomo_stats['rfoot_contact_ratio']):.3f}")
        print(f"  双脚接触比例差异: {abs(amass_stats['both_feet_contact_ratio'] - omomo_stats['both_feet_contact_ratio']):.3f}")
        
        if 'lfoot_speed_mean' in amass_stats and 'lfoot_speed_mean' in omomo_stats:
            print(f"  左脚速度差异: {abs(amass_stats['lfoot_speed_mean'] - omomo_stats['lfoot_speed_mean']):.4f}")
            print(f"  右脚速度差异: {abs(amass_stats['rfoot_speed_mean'] - omomo_stats['rfoot_speed_mean']):.4f}")


def plot_distribution_comparison(amass_stats, omomo_stats, save_path="foot_contact_analysis.png"):
    """绘制分布对比图"""
    if amass_stats is None and omomo_stats is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('足部接触分布对比分析', fontsize=16)
    
    datasets = []
    if amass_stats: datasets.append(("AMASS", amass_stats))
    if omomo_stats: datasets.append(("OMOMO", omomo_stats))
    
    # 1. 接触比例对比
    ax1 = axes[0, 0]
    categories = ['左脚接触', '右脚接触', '双脚接触', '无接触']
    
    for name, stats in datasets:
        values = [
            stats['lfoot_contact_ratio'],
            stats['rfoot_contact_ratio'], 
            stats['both_feet_contact_ratio'],
            stats['no_contact_ratio']
        ]
        x = np.arange(len(categories))
        width = 0.35
        offset = -width/2 if name == "AMASS" else width/2
        ax1.bar(x + offset, values, width, label=name)
    
    ax1.set_xlabel('接触类型')
    ax1.set_ylabel('比例')
    ax1.set_title('足部接触比例对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 序列长度分布
    ax2 = axes[0, 1]
    for name, stats in datasets:
        ax2.hist(stats['sequence_lengths'], bins=30, alpha=0.7, label=name)
    ax2.set_xlabel('序列长度')
    ax2.set_ylabel('频次')
    ax2.set_title('序列长度分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 接触比例分布（每个序列的接触比例）
    ax3 = axes[1, 0]
    for name, stats in datasets:
        ax3.hist(stats['lfoot_contact_ratios'], bins=20, alpha=0.7, label=f'{name}-左脚')
        ax3.hist(stats['rfoot_contact_ratios'], bins=20, alpha=0.7, label=f'{name}-右脚')
    ax3.set_xlabel('每序列接触比例')
    ax3.set_ylabel('频次')
    ax3.set_title('各序列接触比例分布')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 足部速度分布（如果有数据）
    ax4 = axes[1, 1]
    has_speed_data = False
    for name, stats in datasets:
        if stats['foot_speed_stats']['lfoot']:
            speeds = stats['foot_speed_stats']['lfoot'] + stats['foot_speed_stats']['rfoot']
            ax4.hist(speeds, bins=50, alpha=0.7, label=name, range=(0, 0.05))
            has_speed_data = True
    
    if has_speed_data:
        ax4.set_xlabel('足部速度 (m/frame)')
        ax4.set_ylabel('频次') 
        ax4.set_title('足部速度分布')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axvline(x=0.008, color='red', linestyle='--', label='原始阈值')
    else:
        ax4.text(0.5, 0.5, '无足部速度数据', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('足部速度分布（无数据）')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n分析图表已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='分析足部接触标签分布')
    parser.add_argument('--amass_dir', type=str, 
                       default='processed_amass_data_0703/train',
                       help='AMASS数据目录')
    parser.add_argument('--omomo_dir', type=str,
                       default='processed_data_0701/train', 
                       help='OMOMO数据目录')
    parser.add_argument('--sample_limit', type=int, default=100,
                       help='每个数据集分析的最大样本数')
    parser.add_argument('--output', type=str, default='foot_contact_analysis.png',
                       help='输出图表路径')
    
    args = parser.parse_args()
    
    print("开始足部接触标签分布分析...")
    
    # 分析AMASS数据
    amass_stats = None
    if os.path.exists(args.amass_dir):
        amass_stats = analyze_foot_contact_distribution(args.amass_dir, "amass", args.sample_limit)
    else:
        print(f"警告: AMASS数据目录不存在: {args.amass_dir}")
    
    # 分析OMOMO数据  
    omomo_stats = None
    if os.path.exists(args.omomo_dir):
        omomo_stats = analyze_foot_contact_distribution(args.omomo_dir, "omomo", args.sample_limit)
    else:
        print(f"警告: OMOMO数据目录不存在: {args.omomo_dir}")
    
    # 打印结果
    print_analysis_results(amass_stats, omomo_stats)
    
    # 绘制对比图
    plot_distribution_comparison(amass_stats, omomo_stats, args.output)
    
    # 给出建议
    print("\n" + "="*60)
    print("问题诊断和建议:")
    print("="*60)
    
    if amass_stats and omomo_stats:
        lfoot_diff = abs(amass_stats['lfoot_contact_ratio'] - omomo_stats['lfoot_contact_ratio'])
        rfoot_diff = abs(amass_stats['rfoot_contact_ratio'] - omomo_stats['rfoot_contact_ratio'])
        
        if lfoot_diff > 0.2 or rfoot_diff > 0.2:
            print("⚠️  发现严重问题: AMASS和OMOMO的足部接触分布差异很大!")
            print("   建议:")
            print("   1. 使用修改后的自适应阈值重新预处理数据")
            print("   2. 采用分阶段训练策略，避免混合数据集训练")
            print("   3. 调整足部接触损失权重")
        elif lfoot_diff > 0.1 or rfoot_diff > 0.1:
            print("⚠️  发现中等问题: 足部接触分布存在一定差异")
            print("   建议适当调整训练策略和损失权重")
        else:
            print("✅ 足部接触分布差异在可接受范围内")
    
    print("\n分析完成!")


if __name__ == "__main__":
    main() 