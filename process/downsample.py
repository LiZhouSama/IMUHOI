import argparse
import os
import glob
from typing import Dict, Any

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

def _moving_avg_timewise(x, k=3):
    """
    沿时间维做滑动平均：
      - 支持 torch.Tensor / numpy.ndarray
      - 输入形状可为 [T], [T, C], [T, ..., C]（例如 [T, 22, 3]）
      - 仅沿时间维平滑；其它维（关节/通道）互不混合
      - 保持原始形状与（若为浮点）dtype
    """

    is_np = isinstance(x, np.ndarray)
    xt = torch.as_tensor(x)  # 零拷贝包装
    orig_dtype = xt.dtype
    orig_shape = tuple(xt.shape)

    # 只对浮点做平滑；非浮点（如 bool 标签）不要传进来
    if not torch.is_floating_point(xt):
        xt = xt.float()

    # 统一到 [T, C_total]，C_total = 其余维度的乘积（如 22*3）
    if xt.ndim == 1:
        xt = xt[:, None]  # [T, 1]
        rest_shape = (1,)
    else:
        T = xt.shape[0]
        rest_shape = tuple(xt.shape[1:])
        C_total = int(torch.tensor(rest_shape).prod().item())
        xt = xt.reshape(T, C_total)  # [T, C_total]

    T = xt.shape[0]
    if k <= 1 or T <= 2:
        out = xt
    else:
        pad = (k - 1) // 2
        # NCL: [1, C, T]
        x_ncl = xt.transpose(0, 1).unsqueeze(0)
        x_pad = F.pad(x_ncl, (pad, pad), mode="replicate")
        y = F.avg_pool1d(x_pad, kernel_size=k, stride=1)  # 逐通道滑动平均
        out = y.squeeze(0).transpose(0, 1)  # [T, C_total]

    # 还原到原形状
    if len(rest_shape) == 1 and rest_shape[0] == 1 and len(orig_shape) == 1:
        out = out.squeeze(1)  # 还原 [T]
    else:
        out = out.reshape(orig_shape)

    if is_np:
        # numpy：保持浮点 dtype，不是浮点则返回 float32（通常输入已是浮点）
        return out.cpu().numpy().astype(x.dtype if np.issubdtype(x.dtype, np.floating) else np.float32)
    else:
        # torch：尽量还原原始浮点 dtype
        if orig_dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            out = out.to(dtype=orig_dtype)
        return out


def downsample_sequence(data: Dict[str, Any], factor: int = 2) -> Dict[str, Any]:
    """将序列降采样（例如从60fps降到30fps）"""
    downsampled = {}
    
    # 在 downsample 前对 selected keys 平滑一下（k=3 或 5）
    for k in ["trans", "position_global_full_gt_world", "obj_trans"]:
        if k in data and hasattr(data[k], "ndim") and data[k].ndim >= 1:
            data[k] = _moving_avg_timewise(data[k], k=3)

    for key, value in data.items():

        if isinstance(value, torch.Tensor) and value.ndim >= 1:
            # 对于时间维度的张量进行降采样
            if key in ['rotation_local_full_gt_list', 'position_global_full_gt_world', 
                      'rotation_global', 'trans', 'lfoot_contact', 'rfoot_contact',
                      'lhand_contact', 'rhand_contact', 'obj_contact', 'obj_trans', 
                      'obj_rot', 'obj_scale']:
                downsampled[key] = value[::factor]
            else:
                downsampled[key] = value
        elif isinstance(value, np.ndarray) and value.ndim >= 1:
            if key in ['trans']:
                downsampled[key] = value[::factor]
            else:
                downsampled[key] = value
        else:
            downsampled[key] = value
    return downsampled


def process_pt_file(input_path: str, output_path: str, downsample_factor: int = 2) -> bool:
    """处理单个pt文件进行降采样"""
    try:
        data = torch.load(input_path, map_location='cpu')
        downsampled_data = downsample_sequence(data, factor=downsample_factor)
        torch.save(downsampled_data, output_path)
        return True
    except Exception as e:
        print(f"\n处理 {os.path.basename(input_path)} 时出错: {e}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将IMHD数据集从60fps降采样到30fps")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="process/processed_data_IMHD",
        help="输入数据目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="process/processed_data_IMHD_downsample_test",
        help="输出目录"
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=2,
        help="降采样因子，2表示从60fps降到30fps"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    splits = ['train', 'test']
    total_files = 0
    total_success = 0
    
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"降采样因子: {args.downsample_factor} (60fps → 30fps)")
    print()
    
    for split in splits:
        input_split_dir = os.path.join(args.input_dir, split)
        output_split_dir = os.path.join(args.output_dir, split)
        
        if not os.path.exists(input_split_dir):
            print(f"警告: {input_split_dir} 不存在，跳过")
            continue
        
        pt_files = sorted(glob.glob(os.path.join(input_split_dir, "*.pt")))
        if not pt_files:
            print(f"警告: {input_split_dir} 中未找到pt文件")
            continue
        
        print(f"处理 {split} 集合: {len(pt_files)} 个文件")
        os.makedirs(output_split_dir, exist_ok=True)
        
        for pt_file in tqdm(pt_files, desc=f"{split:5s}", leave=False):
            output_path = os.path.join(output_split_dir, os.path.basename(pt_file))
            if process_pt_file(pt_file, output_path, args.downsample_factor):
                total_success += 1
            total_files += 1
    
    print(f"\n完成! 成功处理 {total_success}/{total_files} 个文件")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
