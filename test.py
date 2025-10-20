import os
import pickle
import numpy as np


def main():
    file_path = "datasets/IMHD/IMHD Dataset/ground_truth/20230825/20230825_songzn_bat/bat_holdhandle_hit/gt_0_330_-1.pkl"

    if not os.path.isfile(file_path):
        print(f"文件不存在: {os.path.abspath(file_path)}")
        return

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
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


