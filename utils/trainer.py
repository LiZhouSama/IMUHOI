import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
import datetime
import numpy as np
import os
import json
from tqdm import tqdm

class IMUPoseTrainer:
    """IMU姿态生成模型的训练器"""
    
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        batch_size=32,
        num_workers=8,
        lr=1e-4,
        weight_decay=1e-5,
        grad_clip_value=1.0,
        device='cuda',
        log_dir='logs',
        checkpoint_dir='checkpoints',
        save_interval=10
    ):
        self.model = model
        self.device = device
        self.grad_clip_value = grad_clip_value
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_interval = save_interval
        
        # 数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            self.val_loader = None
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # 创建目录
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.global_step = 0
        
    def train_epoch(self, epoch):
        """训练一个轮次"""
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch_data in enumerate(pbar):
            # 获取输入数据
            pose_params = batch_data['pose_params'].to(self.device)
            
            # 准备条件数据
            cond_data = {
                'imu_data': batch_data['imu_data'].to(self.device)
            }
            
            # 如果有物体数据
            if 'obj_trans' in batch_data:
                cond_data['obj_trans'] = batch_data['obj_trans'].to(self.device)
                cond_data['obj_bps'] = batch_data['obj_bps'].to(self.device)
            
            # 如果有填充掩码
            padding_mask = None
            if 'padding_mask' in batch_data:
                padding_mask = batch_data['padding_mask'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            loss = self.model(pose_params, cond_data, padding_mask)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.grad_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            
            # 更新参数
            self.optimizer.step()
            
            # 记录损失
            loss_value = loss.item()
            epoch_losses.append(loss_value)
            self.train_losses.append(loss_value)
            
            # 更新进度条
            pbar.set_postfix({'loss': f"{loss_value:.4f}"})
            
            self.global_step += 1
            
        # 计算平均损失
        avg_loss = np.mean(epoch_losses)
        
        return avg_loss
    
    def validate(self):
        """在验证集上评估模型"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Validation"):
                # 获取输入数据
                pose_params = batch_data['pose_params'].to(self.device)
                
                # 准备条件数据
                cond_data = {
                    'imu_data': batch_data['imu_data'].to(self.device)
                }
                
                # 如果有物体数据
                if 'obj_trans' in batch_data:
                    cond_data['obj_trans'] = batch_data['obj_trans'].to(self.device)
                    cond_data['obj_bps'] = batch_data['obj_bps'].to(self.device)
                
                # 如果有填充掩码
                padding_mask = None
                if 'padding_mask' in batch_data:
                    padding_mask = batch_data['padding_mask'].to(self.device)
                
                # 前向传播
                loss = self.model(pose_params, cond_data, padding_mask)
                
                # 记录损失
                val_losses.append(loss.item())
        
        # 计算平均损失
        avg_loss = np.mean(val_losses)
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """保存检查点"""
        checkpoint_path = self.checkpoint_dir / f"model_epoch_{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'global_step': self.global_step
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "model_best.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        
        return checkpoint['epoch']
    
    def train(self, num_epochs):
        """训练模型指定的轮次数"""
        start_time = time.time()
        
        print(f"开始训练 {num_epochs} 轮...")
        
        for epoch in range(1, num_epochs + 1):
            # 训练一个轮次
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch}/{num_epochs}, 训练损失: {train_loss:.4f}")
            
            # 验证
            if self.val_loader is not None:
                val_loss = self.validate()
                print(f"Epoch {epoch}/{num_epochs}, 验证损失: {val_loss:.4f}")
                
                # 更新学习率
                self.scheduler.step(val_loss)
                
                # 检查是否为最佳模型
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print(f"找到新的最佳模型，验证损失: {val_loss:.4f}")
            else:
                val_loss = None
                is_best = False
            
            # 保存检查点
            if epoch % self.save_interval == 0 or epoch == num_epochs:
                self.save_checkpoint(epoch, train_loss, is_best)
            
            # 保存训练日志
            self.save_logs()
        
        # 总训练时间
        total_time = time.time() - start_time
        time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"训练完成，总时间: {time_str}")
    
    def save_logs(self):
        """保存训练日志"""
        log_data = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step
        }
        
        log_path = self.log_dir / "training_log.json"
        with open(log_path, 'w') as f:
            json.dump(log_data, f)
    
    @torch.no_grad()
    def generate_samples(self, dataloader, num_samples=5):
        """生成并返回样本"""
        self.model.eval()
        
        samples = []
        for i, batch_data in enumerate(dataloader):
            if i >= num_samples:
                break
            
            # 准备条件数据
            cond_data = {
                'imu_data': batch_data['imu_data'].to(self.device)
            }
            
            # 如果有物体数据
            if 'obj_trans' in batch_data:
                cond_data['obj_trans'] = batch_data['obj_trans'].to(self.device)
                cond_data['obj_bps'] = batch_data['obj_bps'].to(self.device)
            
            # 如果有填充掩码
            padding_mask = None
            if 'padding_mask' in batch_data:
                padding_mask = batch_data['padding_mask'].to(self.device)
            
            # 生成样本
            generated = self.model.sample(cond_data, padding_mask)
            
            # 保存样本
            sample = {
                'generated': generated.cpu(),
                'ground_truth': batch_data['pose_params'],
                'cond': {k: v.cpu() for k, v in cond_data.items()}
            }
            
            samples.append(sample)
        
        return samples 