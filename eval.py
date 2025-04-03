import os
import torch
import yaml
from torch.utils.data import DataLoader
from diffusion_stage.wrap_model import MotionDiffusion
from dataloader.dataloader import IMUDataset
from easydict import EasyDict as edict

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = edict(config)
    return config

def load_model(model_path, config, device):
    """加载模型"""
    model = MotionDiffusion(
        config, 
        input_length=config['train']['window'],
        num_layers=config['model']['num_layers'],
        imu_input=True
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def evaluate_model(model, data_loader, device):
    """评估模型性能"""
    total_loss = 0
    total_loss_rot = 0
    total_loss_root_pos = 0
    total_loss_obj_trans = 0
    total_loss_obj_rot = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            # 准备数据
            root_pos = batch["root_pos"].to(device)
            motion = batch["motion"].to(device)
            human_imu = batch["human_imu"].to(device)
            obj_imu = batch["obj_imu"].to(device) if "obj_imu" in batch else None
            obj_trans = batch["obj_trans"].to(device) if "obj_trans" in batch else None
            obj_rot = batch["obj_rot"].to(device) if "obj_rot" in batch else None
            
            # 如果没有物体数据，使用零张量代替
            if obj_imu is None:
                bs, seq = human_imu.shape[:2]
                obj_imu = torch.zeros((bs, seq, 1, 6), device=device)
                obj_trans = torch.zeros((bs, seq, 3), device=device)
                obj_rot = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(bs, seq, -1, -1)
            
            # 准备输入数据字典
            data_dict = {
                "human_imu": human_imu,
                "obj_imu": obj_imu
            }
            
            # 生成预测
            pred_dict = model.diffusion_reverse(data_dict)
            
            # 计算损失
            loss_rot = torch.nn.functional.mse_loss(pred_dict["motion"], motion)
            loss_root_pos = torch.nn.functional.mse_loss(pred_dict["root_pos"], root_pos)
            loss_obj_trans = torch.nn.functional.mse_loss(pred_dict["obj_trans"], obj_trans)
            loss_obj_rot = torch.nn.functional.mse_loss(pred_dict["obj_rot"], obj_rot)
            
            # 总损失
            loss = loss_rot + 0.1 * loss_root_pos + 0.1 * loss_obj_trans + 0.1 * loss_obj_rot
            
            # 累加损失
            total_loss += loss.item()
            total_loss_rot += loss_rot.item()
            total_loss_root_pos += loss_root_pos.item()
            total_loss_obj_trans += loss_obj_trans.item()
            total_loss_obj_rot += loss_obj_rot.item()
            
            num_batches += 1
    
    # 计算平均损失
    avg_loss = total_loss / num_batches
    avg_loss_rot = total_loss_rot / num_batches
    avg_loss_root_pos = total_loss_root_pos / num_batches
    avg_loss_obj_trans = total_loss_obj_trans / num_batches
    avg_loss_obj_rot = total_loss_obj_rot / num_batches
    
    return {
        'loss': avg_loss,
        'loss_rot': avg_loss_rot,
        'loss_root_pos': avg_loss_root_pos,
        'loss_obj_trans': avg_loss_obj_trans,
        'loss_obj_rot': avg_loss_obj_rot
    }

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载配置
    config = load_config('configs/diffusion.yaml')
    
    # 加载模型
    model_path = 'outputs/imu_04011345/epoch_0.pt'  # 替换为您的模型路径
    model = load_model(model_path, config, device)
    
    test_dataset = IMUDataset(
            data_dir='processed_data_0330/train',
            window_size=config['test']['window'],
            window_stride=config['test']['window_stride'],
            normalize=config['test']['normalize'],
            debug=False
        )
        
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    # 评估模型
    print("\n开始评估模型...")
    results = evaluate_model(model, test_loader, device)
    
    # 打印结果
    print("\n评估结果:")
    print(f"总损失: {results['loss']:.6f}")
    print(f"姿态损失: {results['loss_rot']:.6f}")
    print(f"根关节位置损失: {results['loss_root_pos']:.6f}")
    print(f"物体平移损失: {results['loss_obj_trans']:.6f}")
    print(f"物体旋转损失: {results['loss_obj_rot']:.6f}")

if __name__ == "__main__":
    main() 