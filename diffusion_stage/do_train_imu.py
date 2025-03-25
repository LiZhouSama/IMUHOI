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
from diffusion_stage.wrap_model import MotionDiffusion


def do_train_imu(cfg, train_loader, test_loader=None, trial=None):
    """
    训练IMU到全身姿态及物体变换的Diffusion模型
    
    Args:
        cfg: 配置信息
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        trial: Optuna试验（如果使用超参数搜索）
    """
    # 初始化配置
    device = torch.device(f'cuda:{cfg.gpus[0]}' if torch.cuda.is_available() else 'cpu')
    model_name = cfg.model_name
    use_wandb = cfg.use_wandb
    pose_rep = 'rot6d'
    max_epoch = cfg.epoch
    save_dir = cfg.save_dir

    # 打印训练配置
    print(f'Training: {model_name}, pose_rep: {pose_rep}')
    print(f'use_wandb: {use_wandb}, device: {device}')
    print(f'max_epoch: {max_epoch}')

    os.makedirs(save_dir, exist_ok=True)

    # 初始化Diffusion模型
    input_length = cfg.train.window
    model = MotionDiffusion(cfg, input_length, cfg.model.num_layers, imu_input=True)
    model = model.to(device)

    # 设置优化器
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 如果使用wandb，初始化
    if use_wandb:
        wandb.init(project="imu_diff", name=model_name, config=cfg)

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
        train_loss_obj_trans = 0
        train_loss_obj_rot = 0
        
        train_iter = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch in train_iter:
            # 准备数据
            human_imu = batch["imu"].to(device)  # [bs, seq, 6, 6]
            motion = batch["motion"].to(device)  # [bs, seq, 132]
            obj_imu = batch["obj_imu"].to(device) if "obj_imu" in batch else None  # [bs, seq, 6]
            obj_trans = batch["obj_trans"].to(device) if "obj_trans" in batch else None  # [bs, seq, 3]
            obj_rot = batch["obj_rot"].to(device) if "obj_rot" in batch else None  # [bs, seq, 3, 3]
            bps_features = batch["bps_features"].to(device) if "bps_features" in batch else None
            
            # 如果没有物体数据，使用零张量代替以保持训练稳定
            if obj_imu is None:
                bs, seq = human_imu.shape[:2]
                obj_imu = torch.zeros((bs, seq, 6), device=device)
                obj_trans = torch.zeros((bs, seq, 3), device=device)
                obj_rot = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(bs, seq, -1, -1)
            
            # 准备输入数据字典
            data_dict = {
                "motion": motion,
                "human_imu": human_imu,
                "obj_imu": obj_imu,
                "obj_trans": obj_trans,
                "obj_rot": obj_rot
            }
            
            if bps_features is not None:
                data_dict["bps_features"] = bps_features
            
            # 前向传播
            optimizer.zero_grad()
            pred_dict = model(data_dict)
            
            # 计算损失
            motion_pred = pred_dict["motion"]
            obj_trans_pred = pred_dict["obj_trans"]
            obj_rot_pred = pred_dict["obj_rot"]
            
            # 人体姿态损失
            loss_rot = torch.nn.functional.mse_loss(motion_pred, motion)
            
            # 物体变换损失
            loss_obj_trans = torch.nn.functional.mse_loss(obj_trans_pred, obj_trans)
            loss_obj_rot = torch.nn.functional.mse_loss(obj_rot_pred, obj_rot)
            
            # 总损失 - 使用加权方式组合各个损失
            loss = loss_rot + 0.1 * loss_obj_trans + 0.1 * loss_obj_rot
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 记录损失
            train_loss += loss.item()
            train_loss_rot += loss_rot.item()
            train_loss_obj_trans += loss_obj_trans.item()
            train_loss_obj_rot += loss_obj_rot.item() 
            
            # 更新tqdm描述
            train_iter.set_postfix({
                'loss': loss.item(),
                'loss_rot': loss_rot.item(),
                'loss_obj': loss_obj_trans.item() + loss_obj_rot.item()
            })
            
            # 记录wandb
            if use_wandb:
                log_dict = {
                    'train_loss': loss.item(),
                    'train_loss_rot': loss_rot.item(),
                    'train_loss_obj_trans': loss_obj_trans.item(),
                    'train_loss_obj_rot': loss_obj_rot.item()
                }
                wandb.log(log_dict, step=n_iter)
                
            n_iter += 1

        # 计算平均训练损失
        train_loss /= len(train_loader)
        train_loss_rot /= len(train_loader)
        train_loss_obj_trans /= len(train_loader)
        train_loss_obj_rot /= len(train_loader)
        
        train_losses['loss'].append(train_loss)
        train_losses['loss_rot'].append(train_loss_rot)
        train_losses['loss_obj_trans'].append(train_loss_obj_trans)
        train_losses['loss_obj_rot'].append(train_loss_obj_rot)

        print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, '
              f'Rot Loss: {train_loss_rot:.6f}, '
              f'Obj Trans Loss: {train_loss_obj_trans:.6f}, '
              f'Obj Rot Loss: {train_loss_obj_rot:.6f}')

        # 每5个epoch进行一次测试和保存
        if epoch % 5 == 0 and test_loader is not None:
            # 测试阶段
            model.eval()
            test_loss = 0
            test_loss_rot = 0
            test_loss_obj_trans = 0
            test_loss_obj_rot = 0
            
            with torch.no_grad():
                test_iter = tqdm(test_loader, desc=f'Test Epoch {epoch}')
                for batch in test_iter:
                    # 准备数据
                    human_imu = batch["imu"].to(device)  # [bs, seq, 6, 6]
                    motion = batch["motion"].to(device)  # [bs, seq, 132]
                    obj_imu = batch["obj_imu"].to(device) if "obj_imu" in batch else None  # [bs, seq, 6]
                    obj_trans = batch["obj_trans"].to(device) if "obj_trans" in batch else None  # [bs, seq, 3]
                    obj_rot = batch["obj_rot"].to(device) if "obj_rot" in batch else None  # [bs, seq, 3, 3]
                    bps_features = batch["bps_features"].to(device) if "bps_features" in batch else None
                    
                    # 如果没有物体数据，使用零张量代替
                    if obj_imu is None:
                        bs, seq = human_imu.shape[:2]
                        obj_imu = torch.zeros((bs, seq, 6), device=device)
                        obj_trans = torch.zeros((bs, seq, 3), device=device)
                        obj_rot = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(bs, seq, -1, -1)
                    
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
                    
                    # 计算损失
                    loss_rot = torch.nn.functional.mse_loss(motion_pred, motion)
                    loss_obj_trans = torch.nn.functional.mse_loss(obj_trans_pred, obj_trans)
                    loss_obj_rot = torch.nn.functional.mse_loss(obj_rot_pred, obj_rot)
                    
                    loss = loss_rot + 0.1 * loss_obj_trans + 0.1 * loss_obj_rot
                    
                    # 记录损失
                    test_loss += loss.item()
                    test_loss_rot += loss_rot.item()
                    test_loss_obj_trans += loss_obj_trans.item()
                    test_loss_obj_rot += loss_obj_rot.item()
                    
                    # 更新tqdm描述
                    test_iter.set_postfix({
                        'loss': loss.item(),
                        'loss_rot': loss_rot.item(),
                        'loss_obj': loss_obj_trans.item() + loss_obj_rot.item()
                    })
            
            # 计算平均测试损失
            test_loss /= len(test_loader)
            test_loss_rot /= len(test_loader)
            test_loss_obj_trans /= len(test_loader)
            test_loss_obj_rot /= len(test_loader)
            
            test_losses['loss'].append(test_loss)
            test_losses['loss_rot'].append(test_loss_rot)
            test_losses['loss_obj_trans'].append(test_loss_obj_trans)
            test_losses['loss_obj_rot'].append(test_loss_obj_rot)
            
            print(f'Epoch {epoch}, Test Loss: {test_loss:.6f}, '
                  f'Rot Loss: {test_loss_rot:.6f}, '
                  f'Obj Trans Loss: {test_loss_obj_trans:.6f}, '
                  f'Obj Rot Loss: {test_loss_obj_rot:.6f}')
            
            if use_wandb:
                log_dict = {
                    'test_loss': test_loss,
                    'test_loss_rot': test_loss_rot,
                    'test_loss_obj_trans': test_loss_obj_trans,
                    'test_loss_obj_rot': test_loss_obj_rot
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
    
    # 保存最终模型
    final_path = os.path.join(save_dir, 'final.pt')
    print(f'Saving final model to {final_path}')
    torch.save({
        'epoch': max_epoch - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
    }, final_path)
    
    # 如果使用wandb，保存训练曲线
    if use_wandb:
        wandb.log({
            'train_losses': train_losses,
            'test_losses': test_losses,
        })
        wandb.finish()
    
    # 如果是超参数搜索，返回最佳测试损失
    if trial is not None:
        return best_loss
        
    return model, optimizer 