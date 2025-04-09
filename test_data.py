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
    data_path = "processed_data_0408/train/0.pt"
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"错误：文件 {data_path} 不存在")
        return
    
    try:
        # 加载数据
        data = torch.load(data_path)
        
        # print("\n=== 数据文件内容 ===")
        # print(f"文件路径: {data_path}")
        # print(f"数据类型: {type(data)}")
        
        # if isinstance(data, dict):
        #     print("\n数据键值对:")
        #     print_dict_content(data)
        # else:
        #     print(f"\n数据内容: {data}")
        for key, value in data.items():
            if key == "imu_global_full_gt":
                imu_global_ori = value['orientations'][0]
                denorm_imu_global_ori_exp_head = torch.matmul(imu_global_ori[-1], imu_global_ori[:-1])
                denorm_imu_global_ori = torch.cat([denorm_imu_global_ori_exp_head, imu_global_ori[-1:]], dim=0)
                print(f"{key}: {denorm_imu_global_ori.detach().cpu().numpy()}")
            if key == "rotation_local_full_gt_list":
                rotation_local_full_gt_list_6d = value[0].reshape(-1, 22, 6)
                rotation_local_full_gt_list = transforms.rotation_6d_to_matrix(rotation_local_full_gt_list_6d)
                rotation_local_full_gt_list_imu = rotation_local_full_gt_list[:, IMU_JOINTS, :, :]
                print(f"{key}: {rotation_local_full_gt_list_imu.detach().cpu().numpy()}")
    except Exception as e:
        print(f"错误：加载数据时发生异常 - {str(e)}")

if __name__ == "__main__":
    test_data_file() 