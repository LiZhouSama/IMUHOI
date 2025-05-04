import torch
import os
import numpy as np
import pytorch3d.transforms as transforms
IMU_JOINTS = [20, 21, 7, 8, 0, 15]  # 左手、右手、左脚、右脚、髋部、头部
IMU_JOINT_NAMES = ['left_hand', 'right_hand', 'left_foot', 'right_foot', 'hip', 'head']

def print_dict_content(d, indent=0):
    """
    递归打印字典内容
    
    Args:
        d: 要打印的字典
        indent: 缩进级别
    """
    indent_str = "  " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"\n{indent_str}{key}:")
            print_dict_content(value, indent + 1)
        elif isinstance(value, torch.Tensor):
            print(f"\n{indent_str}{key}:")
            print(f"{indent_str}  类型: {type(value)}")
            print(f"{indent_str}  形状: {value.shape}")
            print(f"{indent_str}  数据类型: {value.dtype}")
        else:
            print(f"\n{indent_str}{key}: {type(value)}")
            if isinstance(value, (list, tuple)):
                print(f"{indent_str}  长度: {len(value)}")
                if len(value) > 0:
                    print(f"{indent_str}  第一个元素类型: {type(value[0])}")

def test_data_file():
    # 数据文件路径
    data_path = "processed_data_0422/test/0.pt"
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"错误：文件 {data_path} 不存在")
        return
    
    try:
        # 加载数据
        data = torch.load(data_path)
        
        print("\n=== 数据文件内容 ===")
        print(f"文件路径: {data_path}")
        print(f"数据类型: {type(data)}")
        
        if isinstance(data, dict):
            print("\n数据键值对:")
            print_dict_content(data)
        else:
            print(f"\n数据内容: {data}")

        lhand_contact = data['lhand_contact'].cpu().numpy()
        rhand_contact = data['rhand_contact'].cpu().numpy()    
        obj_contact = data['obj_contact'].cpu().numpy()

        print(lhand_contact.astype(int))
        print(rhand_contact.astype(int))
        print(obj_contact.astype(int))
    except Exception as e:
        print(f"错误：加载数据时发生异常 - {str(e)}")

if __name__ == "__main__":
    test_data_file() 