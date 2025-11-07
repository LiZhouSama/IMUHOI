import argparse
import os
import glob
import random
from typing import List, Dict, Any

import torch
import numpy as np
from tqdm import tqdm


def downsample_sequence(data: Dict[str, Any], factor: int = 2) -> Dict[str, Any]:
    """将序列降采样（例如从60fps降到30fps）"""
    downsampled = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 1:
            # 对于时间维度的张量进行降采样
            if key in ['rotation_local_full_gt_list', 'position_global_full_gt_world', 
                      'rotation_global', 'trans', 'lfoot_contact', 'rfoot_contact',
                      'lhand_contact', 'rhand_contact', 'obj_contact', 'obj_trans', 
                      'obj_rot', 'obj_scale']:
                downsampled[key] = value[::factor]
            else:
                # betas等非时间序列数据保持不变
                downsampled[key] = value
        elif isinstance(value, np.ndarray) and value.ndim >= 1:
            if key in ['trans']:
                downsampled[key] = value[::factor]
            else:
                downsampled[key] = value
        else:
            # 标量或字符串等元数据保持不变
            downsampled[key] = value
    return downsampled


def split_sequence(data: Dict[str, Any], min_len: int, max_len: int) -> List[Dict[str, Any]]:
    """将序列分割为长度在[min_len, max_len]之间的多个子序列"""
    # 获取序列总长度
    T = None
    for key in ['position_global_full_gt_world', 'trans', 'rotation_local_full_gt_list']:
        if key in data:
            value = data[key]
            if isinstance(value, torch.Tensor):
                T = value.shape[0]
                break
            elif isinstance(value, np.ndarray):
                T = value.shape[0]
                break
    
    if T is None:
        raise ValueError("无法确定序列长度")
    
    segments = []
    start_idx = 0
    
    while start_idx < T:
        # 随机选择当前段的长度
        seg_len = random.randint(min_len, max_len)
        end_idx = start_idx + seg_len
        
        # 如果剩余长度不足min_len，丢弃
        if end_idx > T:
            remaining = T - start_idx
            if remaining < min_len:
                break
            end_idx = T
        
        # 创建子序列
        segment = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                if key in ['rotation_local_full_gt_list', 'position_global_full_gt_world', 
                          'rotation_global', 'trans', 'lfoot_contact', 'rfoot_contact',
                          'lhand_contact', 'rhand_contact', 'obj_contact', 'obj_trans', 
                          'obj_rot', 'obj_scale']:
                    segment[key] = value[start_idx:end_idx]
                else:
                    segment[key] = value
            elif isinstance(value, np.ndarray) and value.ndim >= 1:
                if key in ['trans']:
                    segment[key] = value[start_idx:end_idx]
                else:
                    segment[key] = value
            else:
                segment[key] = value
        
        segments.append(segment)
        start_idx = end_idx
    
    return segments


def process_pt_file(input_path: str, output_dir: str, min_len: int, max_len: int, 
                    downsample: bool = False, downsample_factor: int = 2) -> int:
    """处理单个pt文件，返回生成的分割数量"""
    # 加载数据
    data = torch.load(input_path, map_location='cpu')
    seq_name = data.get('seq_name', os.path.splitext(os.path.basename(input_path))[0])
    
    # 降采样（如果需要）
    if downsample:
        data = downsample_sequence(data, factor=downsample_factor)
    
    # 分割序列
    segments = split_sequence(data, min_len, max_len)
    
    # 保存分割后的序列
    os.makedirs(output_dir, exist_ok=True)
    for idx, segment in enumerate(segments):
        # 更新seq_name以区分不同分割
        segment['seq_name'] = f"{seq_name}_seg{idx:03d}"
        output_path = os.path.join(output_dir, f"{seq_name}_seg{idx:03d}.pt")
        torch.save(segment, output_path)
    
    return len(segments)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分割和降采样处理后的数据集序列")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="process/processed_split_data_IMHD/test",
        help="输入数据目录，包含待处理的pt文件"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="process/processed_seg_data_IMHD/test",
        help="输出目录，保存分割后的pt文件"
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=150,
        help="分割后序列的最小长度"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=300,
        help="分割后序列的最大长度"
    )
    parser.add_argument(
        "--downsample",
        action="store_true",
        help="是否进行降采样（例如从60fps降到30fps）"
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=2,
        help="降采样因子，2表示从60fps降到30fps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于可重复性"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 查找所有pt文件
    pt_files = sorted(glob.glob(os.path.join(args.input_dir, "*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"在 {args.input_dir} 中未找到pt文件")
    
    print(f"找到 {len(pt_files)} 个pt文件")
    print(f"降采样: {'是' if args.downsample else '否'} (因子={args.downsample_factor})")
    print(f"分割长度范围: [{args.min_len}, {args.max_len}]")
    
    total_segments = 0
    for pt_file in tqdm(pt_files, desc="处理序列"):
        try:
            num_segments = process_pt_file(
                pt_file,
                args.output_dir,
                args.min_len,
                args.max_len,
                args.downsample,
                args.downsample_factor
            )
            total_segments += num_segments
        except Exception as e:
            print(f"\n处理 {os.path.basename(pt_file)} 时出错: {e}")
    
    print(f"\n完成! 从 {len(pt_files)} 个文件生成了 {total_segments} 个分割片段")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()

