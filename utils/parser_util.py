import argparse
import yaml
from easydict import EasyDict as edict


def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser(description='通过Diffusion从IMU生成全身姿态')
    parser.add_argument('--cfg', type=str, default='configs/diffusion_0403.yaml', help='配置文件路径')
    parser.add_argument('--seed', type=int, default=10, help='随机种子')
    parser.add_argument('--batch_size', type=int, default=None, help='批量大小')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--debug', action='store_true', help='调试模式，使用小型数据集和简化流程')
    return parser.parse_args()


def merge_file(args):
    """合并配置文件和命令行参数"""
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    # 转换为EasyDict
    cfg = edict(cfg)
    
    # 命令行参数覆盖配置文件
    if args.seed is not None:
        cfg.seed = args.seed
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    
    # 添加调试模式参数
    cfg.debug = args.debug
    if args.debug:
        cfg.epochs = 1
        cfg.batch_size = 1
    # 保存配置文件路径
    cfg.cfg_file = args.cfg
    
    return cfg