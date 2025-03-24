import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import time
from tqdm import tqdm
# 添加混合精度训练支持
from torch.cuda.amp import autocast, GradScaler

class IMUPoseTrainer:
    """IMU姿态生成模型的训练器"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        lr=0.0001,
        weight_decay=0.0001,
        grad_clip=1.0,
        device='cuda',
        output_dir="output",
        save_interval=10
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip_value = grad_clip
        self.device = device
        self.output_dir = output_dir
        self.save_interval = save_interval
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 将模型移至设备
        self.model = self.model.to(self.device)
        
        # GPU预热 - 加速后续操作
        if device == 'cuda':
            self._warmup_gpu()
        
        # 创建优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 创建学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 初始化训练状态
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.global_step = 0
        
        # 初始化混合精度训练
        self.scaler = GradScaler()
        
        # 创建CUDA流
        if torch.cuda.is_available():
            self.data_stream = torch.cuda.Stream()
        
    def _warmup_gpu(self):
        """预热GPU以提高性能"""
        print("预热GPU...")
        dummy_input = torch.randn(8, 120, 66, device=self.device)
        dummy_cond = {'imu_data': torch.randn(8, 120, 42, device=self.device)}
        # 运行几次前向和后向传播
        for _ in range(3):
            with torch.no_grad():
                self.model(dummy_input, dummy_cond)
                
        # 清空缓存
        torch.cuda.empty_cache()
        print("GPU预热完成")
        
    def _prepare_batch(self, batch_data):
        """将批次数据准备到GPU上"""
        # 提取姿态参数
        root_orient = batch_data['body_params']['root_orient'].to(self.device, non_blocking=True)  # [B, T, 3]
        pose_body = batch_data['body_params']['pose_body'].to(self.device, non_blocking=True)      # [B, T, 63]
        pose_params = torch.cat([root_orient, pose_body], dim=-1)  # [B, T, 66]
        
        # 准备条件数据
        cond_data = {
            'imu_data': batch_data['imu_data'].to(self.device, non_blocking=True)
        }
        
        # 如果有物体数据
        if 'obj_trans' in batch_data:
            cond_data['obj_trans'] = batch_data['obj_trans'].to(self.device, non_blocking=True)
            cond_data['obj_bps'] = batch_data['obj_bps'].to(self.device, non_blocking=True)
        
        # 如果有填充掩码
        padding_mask = None
        if 'padding_mask' in batch_data:
            padding_mask = batch_data['padding_mask'].to(self.device, non_blocking=True)
            
        return pose_params, cond_data, padding_mask
        
    def train_epoch(self, epoch):
        """训练一个轮次 - 使用混合精度和CUDA流优化"""
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        # 预先获取第一个批次
        iterator = iter(self.train_loader)
        try:
            next_batch_data = next(iterator)
            # 在CUDA流中预加载数据到GPU
            if torch.cuda.is_available():
                with torch.cuda.stream(self.data_stream):
                    next_pose_params, next_cond_data, next_padding_mask = self._prepare_batch(next_batch_data)
        except StopIteration:
            return 0  # 空数据集
            
        # 同步默认流
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        for batch_idx in range(len(self.train_loader)):
            # 当前批次是已预加载的批次
            pose_params, cond_data, padding_mask = next_pose_params, next_cond_data, next_padding_mask
            
            # 在单独的流中预加载下一个批次
            if batch_idx < len(self.train_loader) - 1:
                try:
                    next_batch_data = next(iterator)
                    if torch.cuda.is_available():
                        with torch.cuda.stream(self.data_stream):
                            next_pose_params, next_cond_data, next_padding_mask = self._prepare_batch(next_batch_data)
                except StopIteration:
                    pass
            
            # 前向传播和反向传播（使用混合精度）
            self.optimizer.zero_grad()
            
            with autocast():
                loss = self.model(pose_params, cond_data, padding_mask)
            
            # 使用梯度缩放进行反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            if self.grad_clip_value > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            
            # 更新参数
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 记录损失
            loss_value = loss.item()
            epoch_losses.append(loss_value)
            self.train_losses.append(loss_value)
            
            # 更新进度条
            pbar.set_postfix({'loss': f"{loss_value:.4f}"})
            pbar.update(1)
            
            self.global_step += 1
            
            # 同步流以准备下一个批次
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # 计算平均损失
        avg_loss = np.mean(epoch_losses)
        
        return avg_loss
    
    def validate(self):
        """在验证集上评估模型 - 使用CUDA优化"""
        if self.val_loader is None:
            return None
            
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Validating"):
                # 将数据移至GPU
                pose_params, cond_data, padding_mask = self._prepare_batch(batch_data)
                
                # 使用混合精度
                with autocast():
                    loss = self.model(pose_params, cond_data, padding_mask)
                
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        self.val_losses.append(avg_val_loss)
        
        return avg_val_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
        
        print(f"检查点已保存至 {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        self.global_step = checkpoint['global_step']
        
        print(f"从 {checkpoint_path} 加载检查点")
    
    def train(self, num_epochs):
        """训练模型指定轮次"""
        start_time = time.time()
        
        for epoch in range(self.epoch, self.epoch + num_epochs):
            # 训练一个轮次
            epoch_start_time = time.time()
            train_loss = self.train_epoch(epoch + 1)
            
            # 验证
            val_loss = self.validate() if self.val_loader else None
            
            # 更新学习率调度器
            if val_loss is not None:
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step(train_loss)
            
            # 检查是否是最佳模型
            is_best = False
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True
            
            # 保存检查点
            if (epoch + 1) % self.save_interval == 0 or is_best:
                self.save_checkpoint(epoch + 1, is_best)
            
            # 更新当前轮次
            self.epoch = epoch + 1
            
            # 记录时间
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time
            
            # 打印进度
            print(f"Epoch {epoch + 1}/{self.epoch + num_epochs} completed in {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.6f}")
            if val_loss is not None:
                print(f"Val Loss: {val_loss:.6f}")
            print(f"总训练时间: {total_time:.2f}s")
        
        return self.train_losses, self.val_losses 