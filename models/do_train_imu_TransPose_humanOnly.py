import os
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import torch
import wandb
from torch import optim
from tqdm import tqdm

from utils.utils import tensor2numpy
from models.TransPose_net_humanOnly import TransPoseNet  # 导入人体姿态版本的TransPose网络
from torch.cuda.amp import autocast, GradScaler


def do_train_imu_TransPose_humanOnly(cfg, train_loader, test_loader=None, trial=None, model=None, optimizer=None):
    """
    训练IMU到人体姿态的TransPose模型（不包含物体估计）
    
    Args:
        cfg: 配置信息
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        trial: Optuna试验（如果使用超参数搜索）
        model: 预训练模型（如果有）
        optimizer: 预训练模型的优化器（如果有）
    """
    # 初始化配置
    device = torch.device(f'cuda:{cfg.gpus[0]}' if torch.cuda.is_available() else 'cpu')
    model_name = cfg.model_name
    use_wandb = cfg.use_wandb
    pose_rep = 'rot6d'  # 使用6D表示
    max_epoch = cfg.epoch
    save_dir = cfg.save_dir
    scaler = GradScaler()

    # 打印训练配置
    print(f'Training: {model_name} (using TransPose Human Only), pose_rep: {pose_rep}')
    print(f'use_wandb: {use_wandb}, device: {device}')
    print(f'max_epoch: {max_epoch}')

    os.makedirs(save_dir, exist_ok=True)

    # 初始化模型（如果没有提供预训练模型）
    if model is None:
        model = TransPoseNet(cfg)
        print(f'Initialized TransPose (Human Only) model with {sum(p.numel() for p in model.parameters())} parameters')
        model = model.to(device)

        # 设置优化器（如果没有提供预训练优化器）
        if optimizer is None:
            optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        print(f'Using pre-trained TransPose (Human Only) model with {sum(p.numel() for p in model.parameters())} parameters')
        model = model.to(device)
        
        # 如果没有提供优化器，创建新的优化器
        if optimizer is None:
            optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 定义学习率调度器
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma)

    # 如果使用wandb，初始化
    if use_wandb:
        wandb_project = "imu_transpose_humanonly"
        wandb.init(project=wandb_project, name=model_name, config=cfg)

    # 训练循环
    best_loss = float('inf')
    train_losses = defaultdict(list)
    test_losses = defaultdict(list)
    n_iter = 0

    for epoch in range(max_epoch):
        # 训练阶段
        model.train()
        train_loss = 0
        train_loss_rot = 0
        
        train_iter = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch in train_iter:
            # 准备数据
            root_pos = batch["root_pos"].to(device)  # [bs, seq, 3]
            motion = batch["motion"].to(device)  # [bs, seq, 22, 6] 或 [bs, seq, 132]
            human_imu = batch["human_imu"].to(device)  # [bs, seq, num_imus, 9]
            
            data_dict = {
                "human_imu": human_imu,
                "motion": motion
            }
            
            # 前向传播
            optimizer.zero_grad()
            
            # TransPose直接输出预测结果
            pred_dict = model(data_dict)
            
            # 计算损失
            # 姿态损失
            loss = torch.nn.functional.mse_loss(pred_dict["motion"], motion)
            
            
            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update() 
            
            # 记录损失
            train_loss += loss.item()
            
            # 更新tqdm描述
            train_iter.set_postfix({
                'loss': loss.item()
            })
            
            # 记录wandb
            if use_wandb:
                log_dict = {
                    'train_loss': loss.item()
                }
                wandb.log(log_dict, step=n_iter)
                
            n_iter += 1

        # 计算平均训练损失
        train_loss /= len(train_loader)
        
        train_losses['loss'].append(train_loss)
        
        print(f'Epoch {epoch}, Train Loss: {train_loss:.6f} ')

        # 每5个epoch进行一次测试和保存
        if epoch % 5 == 0 and test_loader is not None:
            # 测试阶段
            model.eval()
            test_loss = 0
            
            with torch.no_grad():
                test_iter = tqdm(test_loader, desc=f'Test Epoch {epoch}')
                for batch in test_iter:
                    # 准备数据
                    motion = batch["motion"].to(device)
                    human_imu = batch["human_imu"].to(device)
                    
                    data_dict_eval = {
                        "human_imu": human_imu
                    }
                    
                    # TransPose直接输出预测结果
                    pred_dict = model(data_dict_eval)
                    
                    # 计算评估指标
                    # 姿态损失
                    loss = torch.nn.functional.mse_loss(pred_dict["motion"], motion)
                    
                    # 总损失
                    test_metric = loss
                    
                    # 记录损失
                    test_loss += test_metric.item()
                    
                    # 更新tqdm描述
                    test_iter.set_postfix({
                        'test_metric': test_metric.item(),
                    })
            
            # 计算平均测试损失
            test_loss /= len(test_loader)
            
            test_losses['loss'].append(test_loss)
            
            print(f'Epoch {epoch}, Test Metric: {test_loss:.6f}')    
            
            if use_wandb:
                log_dict = {
                    'test_metric': test_loss,
                }
                wandb.log(log_dict, step=n_iter)
            
            # 保存最佳模型
            if test_loss < best_loss:
                best_loss = test_loss
                save_path = os.path.join(save_dir, f'epoch_{epoch}_best.pt')
                print(f'Saving best model to {save_path}')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, save_path)
                
        
        # 定期保存模型
        if epoch % 10 == 0:
            save_path = os.path.join(save_dir, f'epoch_{epoch}.pt')
            print(f'Saving model to {save_path}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, save_path)

        # 更新学习率
        scheduler.step()
    
    # 保存最终模型
    final_path = os.path.join(save_dir, 'final.pt')
    print(f'Saving final model to {final_path}')
    torch.save({
        'epoch': max_epoch - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
    }, final_path)
    
    # 保存最终的损失曲线
    loss_curves = {
        'train_losses': train_losses,
        'test_losses': test_losses,
    }
    with open(os.path.join(save_dir, 'loss_curves.pkl'), 'wb') as f:
        pickle.dump(loss_curves, f)
    
    # 如果使用wandb，保存训练曲线
    if use_wandb:
        wandb.log({
            'final_train_loss': train_loss,
            'final_test_loss': test_loss if test_loader is not None else None,
            'best_test_loss': best_loss if test_loader is not None else None,
        })
        wandb.finish()
    
    # 如果是超参数搜索，返回最佳测试损失
    if trial is not None:
        return best_loss
        
    return model, optimizer


def load_transpose_model_humanOnly(cfg, checkpoint_path):
    """
    加载仅人体姿态的TransPose模型
    
    Args:
        cfg: 配置信息
        checkpoint_path: 模型检查点路径
        
    Returns:
        model: 加载的模型
    """
    device = torch.device(f'cuda:{cfg.gpus[0]}' if torch.cuda.is_available() else 'cpu')
    model = TransPoseNet(cfg).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f'Loaded TransPose Human-Only model from {checkpoint_path}, epoch {checkpoint["epoch"]}')
    
    return model 