import torch
import os

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
    """
    测试函数：打印processed_data_0330/debug/0.pt中的数据键值和shape
    """
    # 数据文件路径
    data_path = "processed_data_0330/debug/0.pt"
    
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
            
    except Exception as e:
        print(f"错误：加载数据时发生异常 - {str(e)}")

if __name__ == "__main__":
    test_data_file() 