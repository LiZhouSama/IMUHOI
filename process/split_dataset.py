import argparse
import os
import glob
import random
import json
import shutil
from typing import List, Dict, Any
from pathlib import Path

import torch


def split_dataset(
    input_dir: str,
    train_dir: str,
    test_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    move_files: bool = False
) -> Dict[str, List[str]]:
    """
    将数据集文件夹按比例分割为训练集和测试集
    
    参数:
        input_dir: 输入数据目录，包含待分割的pt文件
        train_dir: 训练集保存目录
        test_dir: 测试集保存目录
        train_ratio: 训练集比例（默认0.8，即8:2）
        seed: 随机种子，用于可重复性
        move_files: 如果为True则移动文件，否则复制文件
    
    返回:
        包含分割信息的字典 {'train': [...], 'test': [...]}
    """
    # 设置随机种子
    random.seed(seed)
    
    # 查找所有pt文件
    pt_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"在 {input_dir} 中未找到pt文件")
    
    print(f"找到 {len(pt_files)} 个pt文件")
    
    # 随机打乱文件列表
    random.shuffle(pt_files)
    
    # 计算分割点
    num_train = int(len(pt_files) * train_ratio)
    train_files = pt_files[:num_train]
    test_files = pt_files[num_train:]
    
    print(f"训练集: {len(train_files)} 个文件 ({(len(train_files)/len(pt_files)*100):.1f}%)")
    print(f"测试集: {len(test_files)} 个文件 ({(len(test_files)/len(pt_files)*100):.1f}%)")
    
    # 创建输出目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 复制或移动文件到对应目录
    operation = shutil.move if move_files else shutil.copy2
    
    train_file_names = []
    test_file_names = []
    
    print("处理训练集文件...")
    for file_path in train_files:
        file_name = os.path.basename(file_path)
        dst_path = os.path.join(train_dir, file_name)
        operation(file_path, dst_path)
        train_file_names.append(file_name)
    
    print("处理测试集文件...")
    for file_path in test_files:
        file_name = os.path.basename(file_path)
        dst_path = os.path.join(test_dir, file_name)
        operation(file_path, dst_path)
        test_file_names.append(file_name)
    
    # 生成分割信息字典
    split_info = {
        'train_ratio': train_ratio,
        'test_ratio': 1 - train_ratio,
        'total_files': len(pt_files),
        'train_files': sorted(train_file_names),
        'test_files': sorted(test_file_names),
        'train_count': len(train_file_names),
        'test_count': len(test_file_names),
        'seed': seed
    }
    
    return split_info


def save_split_info(split_info: Dict[str, Any], output_json_path: str) -> None:
    """保存分割信息到JSON文件"""
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    print(f"分割信息已保存到: {output_json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将数据集文件夹按比例分割为训练集和测试集"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="process/processed_split_data_OMOMO",
        help="输入数据目录，包含待分割的pt文件"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="process/processed_split_data_OMOMO",
        help="输出目录，将在此目录下创建train和test子文件夹"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集比例（默认0.8，即8:2）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于可重复性"
    )
    parser.add_argument(
        "--move_files",
        action="store_true",
        help="如果设置，将移动文件而不是复制文件"
    )
    parser.add_argument(
        "--json_name",
        type=str,
        default="split_info.json",
        help="保存分割信息的JSON文件名（默认: split_info.json）"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # 验证输入目录存在
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"输入目录不存在: {args.input_dir}")
    
    # 验证训练集比例
    if not 0 < args.train_ratio < 1:
        raise ValueError(f"训练集比例必须在0和1之间，当前值: {args.train_ratio}")
    
    # 构建输出路径
    train_dir = os.path.join(args.output_dir, "train")
    test_dir = os.path.join(args.output_dir, "test")
    json_path = os.path.join(args.output_dir, args.json_name)
    
    print("=" * 60)
    print("数据集分割工具")
    print("=" * 60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练集比例: {args.train_ratio:.1%}")
    print(f"测试集比例: {1 - args.train_ratio:.1%}")
    print(f"随机种子: {args.seed}")
    print(f"操作模式: {'移动' if args.move_files else '复制'}")
    print("=" * 60)
    
    # 执行分割
    try:
        split_info = split_dataset(
            input_dir=args.input_dir,
            train_dir=train_dir,
            test_dir=test_dir,
            train_ratio=args.train_ratio,
            seed=args.seed,
            move_files=args.move_files
        )
        
        # 保存分割信息到JSON
        save_split_info(split_info, json_path)
        
        print("\n" + "=" * 60)
        print("分割完成!")
        print("=" * 60)
        print(f"训练集: {split_info['train_count']} 个文件 → {train_dir}")
        print(f"测试集: {split_info['test_count']} 个文件 → {test_dir}")
        print(f"分割信息: {json_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

