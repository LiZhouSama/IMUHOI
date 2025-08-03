import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
import pytorch3d.transforms as transforms

from dataloader.dataloader import IMUDataset
from configs.global_config import FRAME_RATE, acc_scale

# 导入模型类
from models.TransPose_net_simpleObjT import TransPoseNet as SimpleObjTTransPoseNet

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = edict(config)
    return config

def load_simple_objt_model(config, model_path, device):
    """
    加载SimpleObjT模型用于物体位移预测
    """
    if not os.path.exists(model_path):
        print(f"警告: 模型文件不存在: {model_path}")
        return None
    
    print(f"Loading SimpleObjT model from: {model_path}")
    
    # 创建模型
    pretrained_modules = {'object_trans': model_path}
    model = SimpleObjTTransPoseNet(
        cfg=config,
        pretrained_modules=pretrained_modules,
        skip_modules=['velocity_contact', 'human_pose']  # 跳过人体相关模块
    ).to(device)
    
    model.eval()
    print("SimpleObjT model loaded successfully")
    return model

def compute_velocity_from_acceleration(acceleration, dt, initial_velocity=None):
    """
    通过加速度积分计算速度
    Args:
        acceleration: [T, 3] 加速度数据
        dt: 时间步长
        initial_velocity: [3] 初始速度，默认为零
    Returns:
        velocity: [T, 3] 速度数据
    """
    if initial_velocity is None:
        initial_velocity = torch.zeros(3, device=acceleration.device, dtype=acceleration.dtype)
    
    velocity = torch.zeros_like(acceleration)
    velocity[0] = initial_velocity
    
    for t in range(1, len(acceleration)):
        velocity[t] = velocity[t-1] + acceleration[t] * dt
    
    return velocity

def compute_velocity_from_position(position, dt):
    """
    通过位置差分计算速度
    Args:
        position: [T, 3] 位置数据
        dt: 时间步长
    Returns:
        velocity: [T-1, 3] 速度数据
    """
    velocity = (position[1:] - position[:-1]) / dt
    return velocity

def denormalize_acceleration(normalized_acc, acc_scale_factor):
    """
    反归一化加速度数据
    Args:
        normalized_acc: 归一化的加速度 [T, 3]
        acc_scale_factor: 加速度缩放因子
    Returns:
        denormalized_acc: 反归一化的加速度 [T, 3]
    """
    return normalized_acc * acc_scale_factor

def compute_velocity_errors(true_velocity, pred_velocity, integrated_velocity):
    """
    计算各种速度之间的误差
    Args:
        true_velocity: [T, 3] 真实速度
        pred_velocity: [T, 3] 预测速度
        integrated_velocity: [T, 3] 积分速度
    Returns:
        errors: dict 包含各种误差指标
    """
    # 确保所有速度数据长度一致
    min_len = min(len(true_velocity), len(pred_velocity), len(integrated_velocity))
    true_vel = true_velocity[:min_len]
    pred_vel = pred_velocity[:min_len]
    int_vel = integrated_velocity[:min_len]
    
    errors = {}
    
    # 预测速度 vs 真实速度
    pred_true_error = torch.norm(pred_vel - true_vel, dim=-1)
    errors['pred_true_mse'] = torch.mean(pred_true_error ** 2).item()
    errors['pred_true_mae'] = torch.mean(pred_true_error).item()
    errors['pred_true_rmse'] = torch.sqrt(torch.mean(pred_true_error ** 2)).item()
    
    # 积分速度 vs 真实速度
    int_true_error = torch.norm(int_vel - true_vel, dim=-1)
    errors['int_true_mse'] = torch.mean(int_true_error ** 2).item()
    errors['int_true_mae'] = torch.mean(int_true_error).item()
    errors['int_true_rmse'] = torch.sqrt(torch.mean(int_true_error ** 2)).item()
    
    # 预测速度 vs 积分速度
    pred_int_error = torch.norm(pred_vel - int_vel, dim=-1)
    errors['pred_int_mse'] = torch.mean(pred_int_error ** 2).item()
    errors['pred_int_mae'] = torch.mean(pred_int_error).item()
    errors['pred_int_rmse'] = torch.sqrt(torch.mean(pred_int_error ** 2)).item()
    
    # 各轴的独立误差
    for axis, axis_name in enumerate(['x', 'y', 'z']):
        errors[f'pred_true_{axis_name}_mae'] = torch.mean(torch.abs(pred_vel[:, axis] - true_vel[:, axis])).item()
        errors[f'int_true_{axis_name}_mae'] = torch.mean(torch.abs(int_vel[:, axis] - true_vel[:, axis])).item()
    
    return errors

def test_data_consistency(config_path, model_path=None, test_data_dir=None, num_sequences=10):
    """
    测试数据处理中的坐标系一致性问题以及速度误差评估
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载配置
    config = load_config(config_path)
    if test_data_dir:
        config.test = edict({'data_path': test_data_dir})
    if not hasattr(config, 'test'):
        config.test = config.train.copy()
    
    print(f"=== 数据一致性和速度误差测试 ===")
    print(f"使用设备: {device}")
    print(f"FRAME_RATE: {FRAME_RATE}")
    print(f"acc_scale: {acc_scale}")
    
    # 加载模型
    simple_objt_model = None
    if model_path:
        simple_objt_model = load_simple_objt_model(config, model_path, device)
        if simple_objt_model is None:
            print("警告: 无法加载模型，将使用模拟预测数据")
    else:
        print("警告: 未提供模型路径，将使用模拟预测数据")
    
    # 创建数据加载器
    test_dataset = IMUDataset(
        data_dir=config.test.data_path,
        window_size=config.test.get('window', 60),
        normalize=config.test.get('normalize', True),
        debug=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    print(f"测试数据集大小: {len(test_dataset)}")
    
    results = {
        'velocity_errors': [],
        'sequence_info': [],
        'obj_imu_acc_norm': [],
        'obj_imu_acc_raw': [],
        'true_velocities': [],
        'pred_velocities': [],
        'integrated_velocities': []
    }
    
    dt = 1.0 / FRAME_RATE  # 时间步长
    
    print(f"开始测试{min(num_sequences, len(test_loader))}个序列...")
    print(f"时间步长 dt = {dt:.4f} seconds")
    
    for i, batch in enumerate(test_loader):
        if i >= num_sequences:
            break
            
        has_object = batch["has_object"].item() if hasattr(batch["has_object"], 'item') else batch["has_object"][0]
        if not has_object:
            print(f"序列 {i}: 无物体数据，跳过")
            continue
            
        # 获取原始数据（需要重新加载原始序列数据）
        sequence_files = test_dataset.sequence_files
        file_path = test_dataset.sequence_info[i]['file_path']
        seq_data = torch.load(file_path)
        
        obj_name = batch["obj_name"][0] if isinstance(batch["obj_name"], list) else batch["obj_name"]
        start_idx = 1  # 模拟dataloader的随机采样，这里固定为1
        end_idx = start_idx + config.test.get('window', 60)
        
        # 获取原始数据
        motion = seq_data["rotation_local_full_gt_list"][start_idx:end_idx]
        root_pos = seq_data["position_global_full_gt_world"][start_idx:end_idx, 0, :]
        obj_trans = seq_data["obj_trans"][start_idx:end_idx].squeeze(-1)
        obj_rot = seq_data["obj_rot"][start_idx:end_idx]
        
        # 获取IMU数据
        obj_imu_acc_normalized = batch["obj_imu"][0][:, :, :3]  # 归一化的加速度 [T, 1, 3]
        obj_imu_acc_normalized = obj_imu_acc_normalized.squeeze(1)  # [T, 3]
        
        # 获取raw数据（从dataloader）
        obj_imu_acc_raw = batch["obj_imu_acc_raw"][0].to(device)  # [T, 1, 3]
        obj_imu_acc_raw = obj_imu_acc_raw.squeeze(1)  # [T, 3]
        obj_vel_raw = batch["obj_vel_raw"][0].to(device)  # [T, 3]
        obj_trans_raw = batch["obj_trans_raw"][0].to(device)  # [T, 3]
        
        # 反归一化加速度数据
        obj_imu_acc_denorm = denormalize_acceleration(obj_imu_acc_normalized, acc_scale)
        
        # 计算真实速度（从位置差分）
        # true_velocity = compute_velocity_from_position(obj_trans, dt).to(device)  # [T-1, 3]
        true_velocity = batch['obj_vel'][0].to(device)
        
        # 使用实际模型预测速度
        pred_velocity = None
        if simple_objt_model is not None:
            try:
                with torch.no_grad():
                    # 准备模型输入，参考vis2.py
                    model_input = {
                        "obj_imu": batch["obj_imu"].to(device),  # [1, T, 1, 9/12]
                        "obj_trans": batch["obj_trans"].to(device),  # [1, T, 3]
                    }
                    
                    # 模型预测
                    pred_dict = simple_objt_model(model_input)
                    
                    if "pred_obj_trans" in pred_dict:
                        pred_velocity = pred_dict["pred_obj_vel"][0].to(device)  # [T, 3]
                        
                        print(f"序列 {i}: 成功获取模型预测")
                    else:
                        print(f"序列 {i}: 模型输出中没有pred_obj_trans")
                        
            except Exception as e:
                print(f"序列 {i}: 模型预测失败: {e}")
                
        # 如果模型预测失败，使用模拟预测速度
        if pred_velocity is None:
            print(f"序列 {i}: 使用模拟预测速度")
            noise_scale = 0.1
            pred_velocity = true_velocity + torch.randn_like(true_velocity) * noise_scale
        
        # 通过加速度积分计算速度
        # 使用反归一化的加速度
        integrated_velocity_from_norm = compute_velocity_from_acceleration(
            obj_imu_acc_denorm, dt, initial_velocity=true_velocity[0]
        ).to(device)  # [T, 3]
        
        # 使用raw加速度积分
        integrated_velocity_from_raw_acc = compute_velocity_from_acceleration(
            obj_imu_acc_raw, dt, initial_velocity=obj_vel_raw[0]
        ).to(device)  # [T, 3]
        
        # 通过速度积分计算位置
        integrated_trans_from_raw_vel = compute_velocity_from_acceleration(
            obj_vel_raw, dt, initial_velocity=obj_trans_raw[0]
        ).to(device)  # [T, 3]
        
        true_velocity_debug = true_velocity.detach().cpu().numpy()
        pred_velocity_debug = pred_velocity.detach().cpu().numpy()
        integrated_velocity_from_norm_debug = integrated_velocity_from_norm.detach().cpu().numpy()

        # 计算速度误差
        velocity_errors_norm = compute_velocity_errors(
            true_velocity, pred_velocity, integrated_velocity_from_norm
        )
        
        # 计算raw加速度积分误差
        velocity_errors_raw_acc = compute_velocity_errors(
            obj_vel_raw, pred_velocity, integrated_velocity_from_raw_acc
        )
        
        # 计算raw速度积分误差
        trans_errors_raw_vel = compute_velocity_errors(
            obj_trans_raw, obj_trans_raw, integrated_trans_from_raw_vel  # 这里用obj_trans_raw作为"预测"，实际是真实值
        )
        
        # 存储结果
        sequence_info = {
            'sequence_idx': i,
            'obj_name': obj_name,
            'window_size': len(obj_trans),
            'velocity_errors_norm': velocity_errors_norm,
            'velocity_errors_raw_acc': velocity_errors_raw_acc,
            'trans_errors_raw_vel': trans_errors_raw_vel,
            'has_model_prediction': simple_objt_model is not None
        }
        
        results['velocity_errors'].append({
            'normalized_acc': velocity_errors_norm,
            'raw_acc': velocity_errors_raw_acc,
            'raw_vel': trans_errors_raw_vel
        })
        results['sequence_info'].append(sequence_info)
        results['obj_imu_acc_norm'].append(obj_imu_acc_normalized.detach().cpu().numpy())
        results['obj_imu_acc_raw'].append(obj_imu_acc_raw.detach().cpu().numpy())
        results['true_velocities'].append(true_velocity.detach().cpu().numpy())
        results['pred_velocities'].append(pred_velocity.detach().cpu().numpy())
        results['integrated_velocities'].append({
            'from_norm': integrated_velocity_from_norm.detach().cpu().numpy(),
            'from_raw_acc': integrated_velocity_from_raw_acc.detach().cpu().numpy(),
            'from_raw_vel': integrated_trans_from_raw_vel.detach().cpu().numpy()
        })
        
        # 详细输出前几个序列的信息
        if i < 3:
            print(f"\n--- 序列 {i} ({obj_name}) ---")
            print(f"真实速度范围: [{torch.min(true_velocity):.4f}, {torch.max(true_velocity):.4f}]")
            print(f"预测速度 vs 真实速度 RMSE: {velocity_errors_norm['pred_true_rmse']:.6f}")
            print(f"积分速度(归一化加速度) vs 真实速度 RMSE: {velocity_errors_norm['int_true_rmse']:.6f}")
            print(f"积分速度(raw加速度) vs raw速度 RMSE: {velocity_errors_raw_acc['int_true_rmse']:.6f}")
            print(f"积分位置(raw速度) vs raw位置 RMSE: {trans_errors_raw_vel['int_true_rmse']:.6f}")
            print(f"acc_scale应用效果: norm={obj_imu_acc_normalized[0]} -> denorm={obj_imu_acc_denorm[0]}")
            
            if simple_objt_model is not None:
                print(f"使用实际模型预测")
            else:
                print(f"使用模拟预测（真实速度+噪声）")
    
    # 分析结果
    analyze_velocity_results(results)
    
    return results

def analyze_velocity_results(results):
    """分析速度误差测试结果"""
    if not results['velocity_errors']:
        print("没有有效的测试结果")
        return
    
    print(f"\n=== 速度误差分析结果 ===")
    
    # 统计误差
    norm_errors = [err['normalized_acc'] for err in results['velocity_errors']]
    raw_acc_errors = [err['raw_acc'] for err in results['velocity_errors']]
    raw_vel_errors = [err['raw_vel'] for err in results['velocity_errors']]
    
    # 计算各项误差的平均值
    error_metrics = ['pred_true_rmse', 'int_true_rmse', 'pred_int_rmse', 
                    'pred_true_mae', 'int_true_mae', 'pred_int_mae']
    
    print("使用归一化加速度积分的结果:")
    for metric in error_metrics:
        values = [err[metric] for err in norm_errors]
        print(f"  {metric}: 平均={np.mean(values):.6f}, 标准差={np.std(values):.6f}")
    
    print("\n使用raw加速度积分的结果:")
    for metric in error_metrics:
        values = [err[metric] for err in raw_acc_errors]
        print(f"  {metric}: 平均={np.mean(values):.6f}, 标准差={np.std(values):.6f}")
    
    print("\n使用raw速度积分的结果:")
    for metric in error_metrics:
        values = [err[metric] for err in raw_vel_errors]
        print(f"  {metric}: 平均={np.mean(values):.6f}, 标准差={np.std(values):.6f}")
    
    # 比较积分方法
    norm_int_errors = [err['int_true_rmse'] for err in norm_errors]
    raw_acc_int_errors = [err['int_true_rmse'] for err in raw_acc_errors]
    raw_vel_int_errors = [err['int_true_rmse'] for err in raw_vel_errors]
    
    print(f"\n=== 积分方法比较 ===")
    print(f"归一化加速度积分 RMSE: {np.mean(norm_int_errors):.6f} ± {np.std(norm_int_errors):.6f}")
    print(f"raw加速度积分 RMSE: {np.mean(raw_acc_int_errors):.6f} ± {np.std(raw_acc_int_errors):.6f}")
    print(f"raw速度积分 RMSE: {np.mean(raw_vel_int_errors):.6f} ± {np.std(raw_vel_int_errors):.6f}")
    
    # 比较哪种方法更好
    methods = [
        ("归一化加速度积分", np.mean(norm_int_errors)),
        ("raw加速度积分", np.mean(raw_acc_int_errors)),
        ("raw速度积分", np.mean(raw_vel_int_errors))
    ]
    best_method = min(methods, key=lambda x: x[1])
    print(f"✅ 最佳积分方法: {best_method[0]} (RMSE: {best_method[1]:.6f})")
    
    # 各轴误差分析
    print(f"\n=== 各轴误差分析 ===")
    for axis in ['x', 'y', 'z']:
        norm_axis_errors = [err[f'int_true_{axis}_mae'] for err in norm_errors]
        raw_acc_axis_errors = [err[f'int_true_{axis}_mae'] for err in raw_acc_errors]
        raw_vel_axis_errors = [err[f'int_true_{axis}_mae'] for err in raw_vel_errors]
        
        print(f"{axis.upper()}轴积分误差:")
        print(f"  归一化加速度: {np.mean(norm_axis_errors):.6f} ± {np.std(norm_axis_errors):.6f}")
        print(f"  raw加速度: {np.mean(raw_acc_axis_errors):.6f} ± {np.std(raw_acc_axis_errors):.6f}")
        print(f"  raw速度: {np.mean(raw_vel_axis_errors):.6f} ± {np.std(raw_vel_axis_errors):.6f}")
    
    # 可视化
    # create_velocity_plots(results)
    
    print(f"\n=== 问题诊断与建议 ===")
    
    avg_norm_int_error = np.mean(norm_int_errors)
    avg_raw_acc_int_error = np.mean(raw_acc_int_errors)
    avg_raw_vel_int_error = np.mean(raw_vel_int_errors)
    
    if avg_norm_int_error > 0.1:
        print("⚠️  警告: 归一化加速度积分误差较大，可能原因:")
        print("   1. acc_scale设置不正确")
        print("   2. 归一化方法存在问题")
        print("   3. 时间步长FRAME_RATE设置错误")
        print("   4. IMU加速度计算存在系统性误差")
    
    if avg_raw_acc_int_error > 0.1:
        print("⚠️  警告: raw加速度积分误差较大，可能原因:")
        print("   1. 原始加速度计算存在问题")
        print("   2. 数据预处理中的平滑参数不当")
        print("   3. 时间步长设置错误")
    
    if avg_raw_vel_int_error > 0.1:
        print("⚠️  警告: raw速度积分误差较大，可能原因:")
        print("   1. 速度计算存在问题")
        print("   2. 位置数据的噪声或抖动")
        print("   3. 积分过程中的累积误差")
    
    # 比较不同方法的差异
    norm_raw_acc_diff = abs(avg_norm_int_error - avg_raw_acc_int_error)
    if norm_raw_acc_diff > 0.01:
        print(f"⚠️  警告: 归一化与raw加速度积分误差差异较大 ({norm_raw_acc_diff:.6f})，需检查acc_scale和归一化方法")
    
    acc_vel_diff = abs(avg_raw_acc_int_error - avg_raw_vel_int_error)
    if acc_vel_diff > 0.01:
        print(f"⚠️  警告: raw加速度与raw速度积分误差差异较大 ({acc_vel_diff:.6f})，可能存在数据一致性问题")

def create_velocity_plots(results):
    """创建速度分析图表"""
    if not results['velocity_errors']:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 收集数据
    norm_errors = [err['normalized_acc'] for err in results['velocity_errors']]
    raw_acc_errors = [err['raw_acc'] for err in results['velocity_errors']]
    raw_vel_errors = [err['raw_vel'] for err in results['velocity_errors']]
    
    # 1. 预测误差分布
    pred_true_norm = [err['pred_true_rmse'] for err in norm_errors]
    pred_true_raw_acc = [err['pred_true_rmse'] for err in raw_acc_errors]
    pred_true_raw_vel = [err['pred_true_rmse'] for err in raw_vel_errors]
    
    axes[0, 0].hist(pred_true_norm, bins=10, alpha=0.7, label='归一化加速度', color='blue')
    axes[0, 0].hist(pred_true_raw_acc, bins=10, alpha=0.7, label='raw加速度', color='red')
    axes[0, 0].hist(pred_true_raw_vel, bins=10, alpha=0.7, label='raw速度', color='green')
    axes[0, 0].set_title('预测速度 vs 真实速度 RMSE')
    axes[0, 0].set_xlabel('RMSE')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].legend()
    
    # 2. 积分误差分布
    int_true_norm = [err['int_true_rmse'] for err in norm_errors]
    int_true_raw_acc = [err['int_true_rmse'] for err in raw_acc_errors]
    int_true_raw_vel = [err['int_true_rmse'] for err in raw_vel_errors]
    
    axes[0, 1].hist(int_true_norm, bins=10, alpha=0.7, label='归一化加速度', color='blue')
    axes[0, 1].hist(int_true_raw_acc, bins=10, alpha=0.7, label='raw加速度', color='red')
    axes[0, 1].hist(int_true_raw_vel, bins=10, alpha=0.7, label='raw速度', color='green')
    axes[0, 1].set_title('积分速度 vs 真实速度 RMSE')
    axes[0, 1].set_xlabel('RMSE')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].legend()
    
    # 3. 积分误差对比
    sequence_indices = range(len(int_true_norm))
    axes[0, 2].plot(sequence_indices, int_true_norm, 'bo-', alpha=0.7, label='归一化加速度')
    axes[0, 2].plot(sequence_indices, int_true_raw_acc, 'ro-', alpha=0.7, label='raw加速度')
    axes[0, 2].plot(sequence_indices, int_true_raw_vel, 'go-', alpha=0.7, label='raw速度')
    axes[0, 2].set_title('各序列积分误差对比')
    axes[0, 2].set_xlabel('序列索引')
    axes[0, 2].set_ylabel('RMSE')
    axes[0, 2].legend()
    
    # 4. 各轴误差对比
    axes_names = ['x', 'y', 'z']
    norm_axis_errors = [[err[f'int_true_{axis}_mae'] for err in norm_errors] for axis in axes_names]
    raw_acc_axis_errors = [[err[f'int_true_{axis}_mae'] for err in raw_acc_errors] for axis in axes_names]
    raw_vel_axis_errors = [[err[f'int_true_{axis}_mae'] for err in raw_vel_errors] for axis in axes_names]
    
    x_pos = np.arange(len(axes_names))
    width = 0.25
    
    norm_means = [np.mean(errors) for errors in norm_axis_errors]
    raw_acc_means = [np.mean(errors) for errors in raw_acc_axis_errors]
    raw_vel_means = [np.mean(errors) for errors in raw_vel_axis_errors]
    
    axes[1, 0].bar(x_pos - width, norm_means, width, label='归一化加速度', alpha=0.7)
    axes[1, 0].bar(x_pos, raw_acc_means, width, label='raw加速度', alpha=0.7)
    axes[1, 0].bar(x_pos + width, raw_vel_means, width, label='raw速度', alpha=0.7)
    axes[1, 0].set_title('各轴积分误差对比')
    axes[1, 0].set_xlabel('轴')
    axes[1, 0].set_ylabel('平均MAE')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([axis.upper() for axis in axes_names])
    axes[1, 0].legend()
    
    # 5. 时间序列示例（第一个序列）
    if results['true_velocities']:
        true_vel = results['true_velocities'][0]  # [T-1, 3]
        pred_vel = results['pred_velocities'][0]  # [T-1, 3]
        int_vel_norm = results['integrated_velocities'][0]['from_norm']  # [T-1, 3]
        int_vel_raw_acc = results['integrated_velocities'][0]['from_raw_acc']  # [T-1, 3]
        
        time_steps = np.arange(len(true_vel))
        
        # 只显示X轴作为示例
        axes[1, 1].plot(time_steps, true_vel[:, 0], 'k-', label='真实速度', linewidth=2)
        axes[1, 1].plot(time_steps, pred_vel[:, 0], 'g--', label='预测速度', alpha=0.7)
        axes[1, 1].plot(time_steps, int_vel_norm[:, 0], 'b:', label='积分速度(归一化)', alpha=0.7)
        axes[1, 1].plot(time_steps, int_vel_raw_acc[:, 0], 'r:', label='积分速度(raw)', alpha=0.7)
        axes[1, 1].set_title('速度时间序列示例 (X轴)')
        axes[1, 1].set_xlabel('时间步')
        axes[1, 1].set_ylabel('速度')
        axes[1, 1].legend()
    
    # 6. 误差相关性分析
    if len(norm_errors) > 1:
        pred_errors = [err['pred_true_rmse'] for err in norm_errors]
        int_errors_norm = [err['int_true_rmse'] for err in norm_errors]
        int_errors_raw_acc = [err['int_true_rmse'] for err in raw_acc_errors]
        
        axes[1, 2].scatter(pred_errors, int_errors_norm, alpha=0.6, label='归一化加速度', color='blue')
        axes[1, 2].scatter(pred_errors, int_errors_raw_acc, alpha=0.6, label='raw加速度', color='red')
        axes[1, 2].set_xlabel('预测误差 RMSE')
        axes[1, 2].set_ylabel('积分误差 RMSE')
        axes[1, 2].set_title('预测误差 vs 积分误差')
        axes[1, 2].legend()
        
        # 添加对角线
        all_errors = pred_errors + int_errors_norm + int_errors_raw_acc
        min_err = min(all_errors)
        max_err = max(all_errors)
        axes[1, 2].plot([min_err, max_err], [min_err, max_err], 'k--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('velocity_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n速度分析图表已保存为 'velocity_analysis.png'")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='测试数据处理一致性和速度误差')
    parser.add_argument('--config', type=str, default='configs/TransPose_train.yaml',
                       help='配置文件路径')
    parser.add_argument('--test_data_dir', type=str, default=None,
                       help='测试数据目录（可选）')
    parser.add_argument('--num_sequences', type=int, default=10,
                       help='测试的序列数量')
    parser.add_argument('--model_path', type=str, default=None,
                       help='SimpleObjT模型的路径，用于实际预测')
    
    args = parser.parse_args()
    
    print("=== 数据处理一致性和速度误差测试 ===")
    print(f"配置文件: {args.config}")
    print(f"数据目录: {args.test_data_dir or '使用配置文件中的路径'}")
    print(f"测试序列数: {args.num_sequences}")
    print(f"模型路径: {args.model_path or '未提供'}")
    
    results = test_data_consistency(
        config_path=args.config,
        model_path=args.model_path,
        test_data_dir=args.test_data_dir,
        num_sequences=args.num_sequences
    )
    
    return results

if __name__ == "__main__":
    results = main()
