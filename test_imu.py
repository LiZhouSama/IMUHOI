import argparse
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.dataloader import IMUDataset
from diffusion_stage.wrap_model import MotionDiffusion
from utils.parser_util import get_args
from utils.utils import tensor2numpy, rot6d2matrix, calculate_mpjpe


def test_imu(cfg, model_path=None, visualize=False, output_dir=None):
    """
    测试IMU到姿态和物体变换的Diffusion模型
    
    Args:
        cfg: 配置对象
        model_path: 模型路径，如果为None则使用默认的最佳模型
        visualize: 是否可视化结果
        output_dir: 输出目录，如果为None则使用默认目录
    
    Returns:
        pose_mpjpe: 姿态平均关节位置误差
        obj_trans_error: 物体平移平均误差
        obj_rot_error: 物体旋转平均误差
    """
    # 设置设备
    device = torch.device(f'cuda:{cfg.gpus[0]}' if torch.cuda.is_available() else 'cpu')
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(cfg.save_dir, 'test_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建测试数据集
    test_dataset = IMUDataset(
        data_dir=cfg.test.data_path,
        window_size=cfg.test.window,
        window_stride=cfg.test.window_stride,
        normalize=cfg.test.normalize,
        normalize_style=cfg.test.normalize_style,
        debug=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )
    
    # 创建模型
    model = MotionDiffusion(cfg, cfg.test.window, cfg.model.num_layers, imu_input=True)
    
    # 加载模型权重
    if model_path is None:
        # 查找最佳模型
        model_dir = cfg.save_dir
        best_model_path = None
        for file in os.listdir(model_dir):
            if file.endswith('_best.pt'):
                best_model_path = os.path.join(model_dir, file)
                break
        
        if best_model_path is None:
            # 如果没有找到最佳模型，使用最终模型
            best_model_path = os.path.join(model_dir, 'final.pt')
        
        model_path = best_model_path
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 保存结果
    all_pose_errors = []
    all_obj_trans_errors = []
    all_obj_rot_errors = []
    
    # 测试循环
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            # 准备数据
            human_imu = batch["imu"].to(device)  # [bs, seq, 6, 6]
            motion_gt = batch["motion"].to(device)  # [bs, seq, 132]
            obj_imu = batch["obj_imu"].to(device) if "obj_imu" in batch else None  # [bs, seq, 6]
            obj_trans_gt = batch["obj_trans"].to(device) if "obj_trans" in batch else None  # [bs, seq, 3]
            obj_rot_gt = batch["obj_rot"].to(device) if "obj_rot" in batch else None  # [bs, seq, 3, 3]
            bps_features = batch["bps_features"].to(device) if "bps_features" in batch else None
            
            # 如果没有物体数据，使用零张量代替
            if obj_imu is None:
                bs, seq = human_imu.shape[:2]
                obj_imu = torch.zeros((bs, seq, 6), device=device)
                obj_trans_gt = torch.zeros((bs, seq, 3), device=device)
                obj_rot_gt = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(bs, seq, -1, -1)
            
            # 准备输入数据字典
            data_dict = {
                "human_imu": human_imu,
                "obj_imu": obj_imu
            }
            
            if bps_features is not None:
                data_dict["bps_features"] = bps_features
            
            # 生成预测
            pred_dict = model.diffusion_reverse(data_dict)
            
            motion_pred = pred_dict["motion"]
            obj_trans_pred = pred_dict["obj_trans"]
            obj_rot_pred = pred_dict["obj_rot"]
            
            # 计算人体姿态误差
            bs, seq, _ = motion_pred.shape
            motion_pred_rot = rot6d2matrix(motion_pred.reshape(bs * seq, 22, 6))
            motion_gt_rot = rot6d2matrix(motion_gt.reshape(bs * seq, 22, 6))
            
            joints_pred = calculate_mpjpe(motion_pred_rot, return_joints=True)
            joints_gt = calculate_mpjpe(motion_gt_rot, return_joints=True)
            
            # 计算每个批次样本的MPJPE
            joints_pred = joints_pred.reshape(bs, seq, -1, 3)
            joints_gt = joints_gt.reshape(bs, seq, -1, 3)
            
            pose_error = torch.sqrt(((joints_pred - joints_gt) ** 2).sum(dim=-1)).mean(dim=(1, 2))  # [bs]
            
            # 计算物体变换误差
            obj_trans_error = torch.sqrt(((obj_trans_pred - obj_trans_gt) ** 2).sum(dim=-1)).mean(dim=1)  # [bs]
            
            # 对旋转矩阵计算Frobenius范数作为误差
            obj_rot_error = torch.sqrt(((obj_rot_pred - obj_rot_gt) ** 2).sum(dim=(2, 3))).mean(dim=1)  # [bs]
            
            # 保存结果
            all_pose_errors.append(pose_error.cpu().numpy())
            all_obj_trans_errors.append(obj_trans_error.cpu().numpy())
            all_obj_rot_errors.append(obj_rot_error.cpu().numpy())
            
            # 可视化一些样本
            if visualize and i == 0:
                for b in range(min(4, bs)):
                    # 可视化人体姿态
                    fig = plt.figure(figsize=(24, 8))
                    
                    # 选择帧索引
                    frame_indices = np.linspace(0, seq-1, 8, dtype=int)
                    
                    for j, frame_idx in enumerate(frame_indices):
                        # 左边：预测的姿态
                        ax1 = fig.add_subplot(2, 8, j+1, projection='3d')
                        plot_pose(ax1, joints_pred[b, frame_idx].cpu().numpy(), title=f"Pred Frame {frame_idx}")
                        
                        # 右边：真实姿态
                        ax2 = fig.add_subplot(2, 8, j+9, projection='3d')
                        plot_pose(ax2, joints_gt[b, frame_idx].cpu().numpy(), title=f"GT Frame {frame_idx}")
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"sample_{i}_{b}_pose.png"))
                    plt.close()
                    
                    # 可视化物体变换
                    plt.figure(figsize=(12, 6))
                    
                    # 物体平移（已应用缩放）
                    plt.subplot(1, 2, 1)
                    plt.plot(tensor2numpy(obj_trans_pred[b, :, 0]), label='Pred X')
                    plt.plot(tensor2numpy(obj_trans_gt[b, :, 0]), '--', label='GT X')
                    plt.plot(tensor2numpy(obj_trans_pred[b, :, 1]), label='Pred Y')
                    plt.plot(tensor2numpy(obj_trans_gt[b, :, 1]), '--', label='GT Y')
                    plt.plot(tensor2numpy(obj_trans_pred[b, :, 2]), label='Pred Z')
                    plt.plot(tensor2numpy(obj_trans_gt[b, :, 2]), '--', label='GT Z')
                    plt.title('Object Translation (Scale Applied)')
                    plt.legend()
                    
                    # 旋转误差
                    plt.subplot(1, 2, 2)
                    rot_error = torch.sqrt(((obj_rot_pred[b] - obj_rot_gt[b]) ** 2).sum(dim=(1, 2)))
                    plt.plot(tensor2numpy(rot_error))
                    plt.title('Rotation Error (Frobenius Norm)')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"sample_{i}_{b}_object.png"))
                    plt.close()
    
    # 计算总体指标
    all_pose_errors = np.concatenate(all_pose_errors)
    all_obj_trans_errors = np.concatenate(all_obj_trans_errors)
    all_obj_rot_errors = np.concatenate(all_obj_rot_errors)
    
    pose_mpjpe = all_pose_errors.mean() * 1000  # 毫米单位
    obj_trans_error = all_obj_trans_errors.mean() * 1000  # 毫米单位
    obj_rot_error = all_obj_rot_errors.mean()
    
    print(f"测试结果:")
    print(f"姿态MPJPE: {pose_mpjpe:.2f} mm")
    print(f"物体平移误差: {obj_trans_error:.2f} mm")
    print(f"物体旋转误差: {obj_rot_error:.4f}")
    
    # 保存结果到文件
    with open(os.path.join(output_dir, "test_results.txt"), "w") as f:
        f.write(f"姿态MPJPE: {pose_mpjpe:.2f} mm\n")
        f.write(f"物体平移误差: {obj_trans_error:.2f} mm\n")
        f.write(f"物体旋转误差: {obj_rot_error:.4f}\n")
    
    return pose_mpjpe, obj_trans_error, obj_rot_error


def plot_pose(ax, joints, title=None):
    """
    将3D关节点绘制成人体骨架
    
    Args:
        ax: matplotlib 3D轴
        joints: 关节点坐标 [num_joints, 3]
        title: 图表标题
    """
    # 定义骨架连接
    connections = [
        (0, 1), (1, 2), (2, 3),  # 右腿
        (0, 4), (4, 5), (5, 6),  # 左腿
        (0, 7), (7, 8), (8, 9), (9, 10),  # 躯干和头部
        (8, 11), (11, 12), (12, 13),  # 右臂
        (8, 14), (14, 15), (15, 16),  # 左臂
        (11, 17), (17, 18),  # 右手
        (14, 19), (19, 20)   # 左手
    ]
    
    # 绘制关节点
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='b', marker='o', s=10)
    
    # 绘制骨架连接
    for connection in connections:
        ax.plot([joints[connection[0], 0], joints[connection[1], 0]],
                [joints[connection[0], 1], joints[connection[1], 1]],
                [joints[connection[0], 2], joints[connection[1], 2]], 'r')
    
    # 设置轴限制
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    # 设置标题
    if title:
        ax.set_title(title)
    
    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def main():
    """主函数"""
    parser = get_args()
    parser.add_argument('--visualize', action='store_true', default=True, help='是否可视化结果')
    parser.add_argument('--model_path', type=str, default=None, help='模型路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    cfg = args.cfg
    
    test_imu(cfg, args.model_path, args.visualize, args.output_dir)


if __name__ == "__main__":
    main() 