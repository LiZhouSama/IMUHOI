import os
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from models.DiT_model import MotionDiffusion
from dataloader.dataloader import IMUDataset
from easydict import EasyDict as edict
from human_body_prior.body_model.body_model import BodyModel
import pytorch3d.transforms as transforms
import argparse

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = edict(config)
    return config

def load_model(model_path, config, device):
    """加载训练好的模型"""
    input_length = config.test.get('window', config.train.get('window', 60))

    # 检查模型类型并加载
    if config.get('use_transpose_model', False):
        print("Loading TransPose model...")
        from models.do_train_imu_TransPose import load_transpose_model
        model = load_transpose_model(config, model_path)
    elif config.get('use_transpose_humanOnly_model', False):
        print("Loading TransPose HumanOnly model...")
        from models.do_train_imu_TransPose_humanOnly import load_transpose_model_humanOnly
        model = load_transpose_model_humanOnly(config, model_path)
    elif config.model.get('use_dit_model', True):
        print("Loading DiT_model...")
        model = MotionDiffusion(config, input_length=input_length).to(device)
    else:
        print("Loading wrap_model...")
        from models.wrap_model import MotionDiffusion as WrapMotionDiffusion
        model = WrapMotionDiffusion(config, input_length=input_length).to(device)

    # 如果模型不是通过特定加载函数（如 load_transpose_model）加载的，则加载 state_dict
    if not (config.get('use_transpose_model', False) or config.get('use_transpose_humanOnly_model', False)):
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            # Handle potential keys like 'model_state_dict' or 'model'
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            # Remove 'module.' prefix if saved with DataParallel
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        else:
             print(f"Warning: Model checkpoint not found at {model_path}. Model weights not loaded.")

    model.eval()
    model.to(device)
    return model

def print_leg_joint_rotations(model, data_loader, device):
    """打印第一个序列的腿部关节局部旋转矩阵值"""
    
    # SMPL/SMPLH关节索引定义 (基于标准SMPL定义)
    # 这些是身体关节的索引 (不包括根关节)
    SMPL_JOINT_NAMES = [
        'left_hip',      # 0 -> 在motion中的索引: 0
        'right_hip',     # 1 -> 在motion中的索引: 1  
        'spine1',        # 2 -> 在motion中的索引: 2
        'left_knee',     # 3 -> 在motion中的索引: 3
        'right_knee',    # 4 -> 在motion中的索引: 4
        'spine2',        # 5 -> 在motion中的索引: 5
        'left_ankle',    # 6 -> 在motion中的索引: 6
        'right_ankle',   # 7 -> 在motion中的索引: 7
        'spine3',        # 8 -> 在motion中的索引: 8
        'left_foot',     # 9 -> 在motion中的索引: 9
        'right_foot',    # 10 -> 在motion中的索引: 10
        'neck',          # 11 -> 在motion中的索引: 11
        'left_collar',   # 12 -> 在motion中的索引: 12
        'right_collar',  # 13 -> 在motion中的索引: 13
        'head',          # 14 -> 在motion中的索引: 14
        'left_shoulder', # 15 -> 在motion中的索引: 15
        'right_shoulder',# 16 -> 在motion中的索引: 16
        'left_elbow',    # 17 -> 在motion中的索引: 17
        'right_elbow',   # 18 -> 在motion中的索引: 18
        'left_wrist',    # 19 -> 在motion中的索引: 19
        'right_wrist',   # 20 -> 在motion中的索引: 20
    ]
    
    # 腿部关节索引 (在21个身体关节中的索引)
    leg_joint_indices = {
        'left_hip': 0,      # 左髋关节
        'right_hip': 1,     # 右髋关节  
        'left_knee': 3,     # 左膝关节
        'right_knee': 4,    # 右膝关节
        'left_ankle': 6,    # 左踝关节
        'right_ankle': 7,   # 右踝关节
        'left_foot': 9,     # 左脚
        'right_foot': 10,   # 右脚
    }
    
    print("腿部关节索引映射:")
    for name, idx in leg_joint_indices.items():
        print(f"  {name}: 索引 {idx}")
    print()

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            print(f"处理第一个批次 (batch_idx: {batch_idx})...")
            
            # --- Ground Truth Data ---
            gt_root_pos = batch["root_pos"].to(device)
            gt_motion = batch["motion"].to(device)
            gt_human_imu = batch["human_imu"].to(device)
            
            gt_obj_imu = batch.get("obj_imu", None)
            gt_obj_trans = batch.get("obj_trans", None)
            gt_obj_rot_6d = batch.get("obj_rot", None)
            
            if gt_obj_imu is not None: gt_obj_imu = gt_obj_imu.to(device)
            if gt_obj_trans is not None: gt_obj_trans = gt_obj_trans.to(device)
            if gt_obj_rot_6d is not None: gt_obj_rot_6d = gt_obj_rot_6d.to(device)

            bs, seq_len, motion_dim = gt_motion.shape
            print(f"批次信息: bs={bs}, seq_len={seq_len}, motion_dim={motion_dim}")
            
            if motion_dim != 132:
                print(f"Warning: 期望motion维度为132 (22*6D)，但得到{motion_dim}. 跳过批次.")
                continue

            # 准备模型输入
            model_input = {
                "human_imu": gt_human_imu,
                "motion": gt_motion,
                "root_pos": gt_root_pos,
            }
            
            if gt_obj_imu is not None:
                model_input["obj_imu"] = gt_obj_imu
                model_input["obj_rot"] = gt_obj_rot_6d
                model_input["obj_trans"] = gt_obj_trans

            try:
                # 模型推理
                if hasattr(model, 'diffusion_reverse'):
                    pred_dict = model.diffusion_reverse(model_input)
                else:
                    pred_dict = model(model_input)
            except Exception as e:
                print(f"模型推理出错: {e}")
                continue

            # --- 提取预测结果 ---
            pred_motion_norm = pred_dict.get("motion", None)
            
            if pred_motion_norm is None:
                print("Warning: 模型没有输出'motion'，跳过批次")
                continue

            # 提取第一个序列的数据 (取第一个batch中的第一个序列)
            first_seq_pred_motion = pred_motion_norm[0]  # [seq_len, 132]
            first_seq_gt_motion = gt_motion[0]           # [seq_len, 132]
            
            print(f"第一个序列长度: {first_seq_pred_motion.shape[0]}")
            print()
            
            # 提取身体姿态 (跳过前6维的根关节朝向)
            pred_body_pose_6d = first_seq_pred_motion[:, 6:]  # [seq_len, 21*6]
            gt_body_pose_6d = first_seq_gt_motion[:, 6:]      # [seq_len, 21*6]
            
            # 重塑为 [seq_len, 21, 6]
            pred_body_pose_6d = pred_body_pose_6d.reshape(seq_len, 21, 6)
            gt_body_pose_6d = gt_body_pose_6d.reshape(seq_len, 21, 6)
            
            # 转换为旋转矩阵
            pred_body_pose_6d_flat = pred_body_pose_6d.reshape(-1, 6)  # [seq_len*21, 6]
            pred_body_pose_mat_flat = transforms.rotation_6d_to_matrix(pred_body_pose_6d_flat)  # [seq_len*21, 3, 3]
            pred_body_pose_mat = pred_body_pose_mat_flat.reshape(seq_len, 21, 3, 3)  # [seq_len, 21, 3, 3]
            
            gt_body_pose_6d_flat = gt_body_pose_6d.reshape(-1, 6)  # [seq_len*21, 6]
            gt_body_pose_mat_flat = transforms.rotation_6d_to_matrix(gt_body_pose_6d_flat)  # [seq_len*21, 3, 3]
            gt_body_pose_mat = gt_body_pose_mat_flat.reshape(seq_len, 21, 3, 3)  # [seq_len, 21, 3, 3]
            
            print("=" * 80)
            print("腿部关节局部旋转矩阵分析")
            print("=" * 80)
            
            # 为每个腿部关节打印旋转矩阵信息
            for joint_name, joint_idx in leg_joint_indices.items():
                print(f"\n关节: {joint_name} (索引: {joint_idx})")
                print("-" * 50)
                
                # 预测的旋转矩阵 (第一帧和最后一帧)
                pred_first_frame = pred_body_pose_mat[0, joint_idx]  # [3, 3]
                pred_last_frame = pred_body_pose_mat[-1, joint_idx]  # [3, 3]
                
                # GT的旋转矩阵 (第一帧和最后一帧)
                gt_first_frame = gt_body_pose_mat[0, joint_idx]  # [3, 3]
                gt_last_frame = gt_body_pose_mat[-1, joint_idx]  # [3, 3]
                
                print("预测 - 第一帧旋转矩阵:")
                print(pred_first_frame.detach().cpu().numpy())
                print("\n预测 - 最后一帧旋转矩阵:")
                print(pred_last_frame.detach().cpu().numpy())
                
                print("\nGT - 第一帧旋转矩阵:")
                print(gt_first_frame.detach().cpu().numpy())
                print("\nGT - 最后一帧旋转矩阵:")
                print(gt_last_frame.detach().cpu().numpy())
                
                # 计算旋转差异 (Frobenius范数)
                diff_first = torch.norm(pred_first_frame - gt_first_frame, p='fro')
                diff_last = torch.norm(pred_last_frame - gt_last_frame, p='fro')
                
                print(f"\n旋转矩阵差异 (Frobenius范数):")
                print(f"  第一帧: {diff_first.item():.6f}")
                print(f"  最后一帧: {diff_last.item():.6f}")
                
                # 检查是否几乎为单位矩阵 (表明没有旋转)
                identity = torch.eye(3, device=pred_first_frame.device)
                pred_identity_diff_first = torch.norm(pred_first_frame - identity, p='fro')
                pred_identity_diff_last = torch.norm(pred_last_frame - identity, p='fro')
                
                print(f"\n与单位矩阵的差异 (检查是否有旋转):")
                print(f"  预测第一帧: {pred_identity_diff_first.item():.6f}")
                print(f"  预测最后一帧: {pred_identity_diff_last.item():.6f}")
                
                if pred_identity_diff_first < 0.01 and pred_identity_diff_last < 0.01:
                    print(f"  ⚠️  警告: {joint_name} 几乎没有旋转变化！")
            
            # 计算整个序列中腿部关节的运动统计
            print("\n" + "=" * 80)
            print("腿部关节运动统计")
            print("=" * 80)
            
            for joint_name, joint_idx in leg_joint_indices.items():
                # 计算整个序列中该关节的旋转变化
                joint_rotations = pred_body_pose_mat[:, joint_idx]  # [seq_len, 3, 3]
                
                # 计算相邻帧之间的旋转差异
                rotation_diffs = []
                for t in range(1, seq_len):
                    diff = torch.norm(joint_rotations[t] - joint_rotations[t-1], p='fro')
                    rotation_diffs.append(diff.item())
                
                avg_frame_diff = np.mean(rotation_diffs) if rotation_diffs else 0
                max_frame_diff = np.max(rotation_diffs) if rotation_diffs else 0
                
                print(f"{joint_name}:")
                print(f"  平均帧间变化: {avg_frame_diff:.6f}")
                print(f"  最大帧间变化: {max_frame_diff:.6f}")
                
                if avg_frame_diff < 0.001:
                    print(f"  ⚠️  警告: {joint_name} 运动变化非常小！")
            
            # 只处理第一个批次
            break

def main():
    parser = argparse.ArgumentParser(description='测试腿部关节旋转')
    parser.add_argument('--config', type=str, default='configs/TransPose_train.yaml', 
                       help='配置文件路径')
    parser.add_argument('--model_path', type=str, default=None, 
                       help='模型检查点路径')
    parser.add_argument('--test_data_dir', type=str, default=None, 
                       help='测试数据目录路径')
    parser.add_argument('--batch_size', type=int, default=1, 
                       help='批次大小 (建议设为1便于分析)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载配置
    print(f"加载配置文件: {args.config}")
    config = load_config(args.config)

    # 覆盖配置值
    if args.model_path: config.model_path = args.model_path
    if args.test_data_dir: config.test.data_path = args.test_data_dir
    if args.batch_size: config.test.batch_size = args.batch_size

    # 加载模型
    model_path = config.get('model_path', None)
    if model_path is None or not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        return

    print(f"加载模型: {model_path}")
    model = load_model(model_path, config, device)

    # 加载测试数据
    test_data_dir = config.test.get('data_path', None)
    if test_data_dir is None or not os.path.exists(test_data_dir):
        print(f"错误: 测试数据路径不存在: {test_data_dir}")
        return

    print(f"加载测试数据: {test_data_dir}")
    
    test_window_size = config.test.get('window', config.train.get('window', 60))
    test_dataset = IMUDataset(
        data_dir=test_data_dir,
        window_size=test_window_size,
        normalize=config.test.get('normalize', True),
        debug=config.get('debug', False)
    )

    if len(test_dataset) == 0:
        print("错误: 测试数据集为空")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test.get('batch_size', 1),
        shuffle=False,
        num_workers=0,  # 设为0避免多进程问题
        pin_memory=True,
        drop_last=False
    )

    print(f"\n数据集大小: {len(test_dataset)}")
    print(f"批次大小: {config.test.get('batch_size', 1)}")
    print("\n开始分析腿部关节旋转...")
    
    # 打印腿部关节旋转分析
    print_leg_joint_rotations(model, test_loader, device)

if __name__ == "__main__":
    main() 