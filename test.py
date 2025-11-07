import os
import pickle
import numpy as np
import torch


def main():
    file_path = "processed_data_IMHD_split/train/20230825__20230825_songzn_bat__bat_holdhandle_hit_seg000.pt"

    if not os.path.isfile(file_path):
        print(f"文件不存在: {os.path.abspath(file_path)}")
        return

    # 使用torch.load读取.pt文件，设置weights_only=False以支持numpy数组
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    
    # 如果是tensor，转换为numpy以便查看
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    
    keys = list(data.keys())

    print("=== head(键列表) ===")
    print(keys)

    print("\n=== 各键的形状与类型，并打印一条完整数据 ===")
    for key in keys:
        arr = data[key]
        print(f"\n[key] {key}")
        print(f"shape: {getattr(arr, 'shape', None)}  dtype: {getattr(arr, 'dtype', type(arr))}")
        # 打印一条数据：优先取第0维的一条；若为标量/零维，则直接打印
        try:
            sample = arr[0] if hasattr(arr, "ndim") and arr.ndim > 0 else arr
        except Exception:
            sample = arr
        print("sample:")
        print(sample)


if __name__ == "__main__":
    main()


