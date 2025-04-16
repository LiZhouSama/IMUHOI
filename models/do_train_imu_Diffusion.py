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
# from models.wrap_model import MotionDiffusion # Comment out old import
from models.DiT_model import MotionDiffusion # Import the new model


def do_train_imu(cfg, train_loader, test_loader=None, trial=None):
    """
    训练IMU到全身姿态及物体变换的Diffusion模型 (Using DiT_model)
    
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
    pose_rep = 'rot6d' # This might be implicitly defined by the target_feature_dim
    max_epoch = cfg.epoch
    save_dir = cfg.save_dir

    # 打印训练配置
    print(f'Training: {model_name} (using DiT_model), pose_rep: {pose_rep}') # Indicate new model
    print(f'use_wandb: {use_wandb}, device: {device}')
    print(f'max_epoch: {max_epoch}')

    os.makedirs(save_dir, exist_ok=True)

    # 初始化Diffusion模型 (Using DiT_model's MotionDiffusion)
    input_length = cfg.train.window
    # Pass cfg directly, the new MotionDiffusion init handles parameter extraction
    model = MotionDiffusion(cfg, input_length, imu_input=True) 
    model = model.to(device)

    # 设置优化器
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 定义学习率调度器
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma)

    # 如果使用wandb，初始化
    if use_wandb:
        wandb.init(project="imu_diff_dit", name=model_name, config=cfg) # Maybe new project name

    # 训练循环 (Logic remains the same as the last version)
    best_loss = float('inf')
    train_losses = defaultdict(list)
    test_losses = defaultdict(list)
    n_iter = 0

    for epoch in range(max_epoch):
        # 训练阶段
        model.train()
        train_loss = 0
        
        train_iter = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch in train_iter:
            # 准备数据
            root_pos = batch["root_pos"].to(device)  # [bs, seq, 3]
            motion = batch["motion"].to(device)  # [bs, seq, 132]
            human_imu = batch["human_imu"].to(device)  # [bs, seq, num_imus, 12] - 注意：现在是12D而不是6D
            obj_imu = batch["obj_imu"].to(device) if "obj_imu" in batch else None  # [bs, seq, 1, 12] - 注意：现在是12D
            obj_trans = batch["obj_trans"].to(device) if "obj_trans" in batch else None  # [bs, seq, 3]
            obj_rot = batch["obj_rot"].to(device) if "obj_rot" in batch else None  # [bs, seq, 3, 3]
            # obj_imu = None
            # obj_trans = None
            # obj_rot = None
            bps_features = batch["bps_features"].to(device) if "bps_features" in batch else None
            
            # 如果没有物体数据，使用零张量代替以保持训练稳定
            if obj_imu is None:
                bs, seq = human_imu.shape[:2]
                obj_imu = torch.zeros((bs, seq, 1, 12), device=device)  # 注意：现在是12D
                obj_trans = torch.zeros((bs, seq, 3), device=device)
                obj_rot = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(bs, seq, -1, -1)
            
            data_dict = {
                "human_imu": human_imu,
                "obj_imu": obj_imu,
                "root_pos": root_pos,
                "motion": motion,
                "obj_trans": obj_trans,
                "obj_rot": obj_rot
            }
            
            if bps_features is not None:
                data_dict["bps_features"] = bps_features
            
            # 前向传播 - 获取预测噪声和真实噪声
            optimizer.zero_grad()
            # Call the forward method of the new MotionDiffusion model
            predicted_noise, noise = model(data_dict)
            
            # 计算损失 (MSE between predicted noise and actual noise)
            # Consider slicing if object prediction is less important or noisy initially
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            # loss = torch.nn.functional.mse_loss(predicted_noise[..., :135], noise[..., :135]) # Only human
            
            # 反向传播和优化
            loss.backward()
            # 梯度裁剪 (推荐)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 记录损失
            train_loss += loss.item()
            
            # 更新tqdm描述
            train_iter.set_postfix({'loss': loss.item()})
            
            # 记录wandb
            if use_wandb:
                log_dict = {'train_loss': loss.item()}
                wandb.log(log_dict, step=n_iter)
                
            n_iter += 1

        # 计算平均训练损失
        train_loss /= len(train_loader)
        train_losses['loss'].append(train_loss)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}')

        # 每5个epoch进行一次测试和保存
        if epoch % 5 == 0 and test_loader is not None:
            # 测试阶段
            model.eval()
            test_loss = 0
            test_loss_rot = 0
            test_loss_root_pos = 0
            test_loss_obj_trans = 0
            test_loss_obj_rot = 0
            
            with torch.no_grad():
                test_iter = tqdm(test_loader, desc=f'Test Epoch {epoch}')
                for batch in test_iter:
                    # 准备数据
                    root_pos = batch["root_pos"].to(device)  # [bs, seq, 3]
                    motion = batch["motion"].to(device)  # [bs, seq, 132]
                    human_imu = batch["human_imu"].to(device)  # [bs, seq, num_imus, 12] - 注意：现在是12D而不是6D
                    obj_imu = batch["obj_imu"].to(device) if "obj_imu" in batch else None  # [bs, seq, 1, 12] - 注意：现在是12D
                    obj_trans = batch["obj_trans"].to(device) if "obj_trans" in batch else None  # [bs, seq, 3]
                    obj_rot = batch["obj_rot"].to(device) if "obj_rot" in batch else None  # [bs, seq, 3, 3]
                    # obj_imu = None
                    # obj_trans = None
                    # obj_rot = None
                    bps_features = batch["bps_features"].to(device) if "bps_features" in batch else None
                    
                    if obj_imu is None:
                        bs, seq = human_imu.shape[:2]
                        obj_imu = torch.zeros((bs, seq, 1, 12), device=device)  # 注意：现在是12D
                        obj_trans = torch.zeros((bs, seq, 3), device=device)
                        obj_rot = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(bs, seq, -1, -1)
                    
                    data_dict_eval = {
                        "human_imu": human_imu,
                        "obj_imu": obj_imu
                        # 注意：diffusion_reverse 不需要 GT 数据
                    }
                    
                    if bps_features is not None:
                        data_dict_eval["bps_features"] = bps_features
                    
                    # 生成预测
                    pred_dict = model.diffusion_reverse(data_dict_eval)
                    
                    # # 计算评估指标
                    # root_pos_pred = pred_dict["root_pos"]
                    # motion_pred = pred_dict["motion"]
                    # obj_trans_pred = pred_dict["obj_trans"]
                    # obj_rot_pred = pred_dict["obj_rot"]
                    
                    # loss_rot = torch.nn.functional.mse_loss(motion_pred, motion)
                    # loss_root_pos = torch.nn.functional.mse_loss(root_pos_pred, root_pos)
                    # loss_obj_trans = torch.nn.functional.mse_loss(obj_trans_pred, obj_trans)
                    # loss_obj_rot = torch.nn.functional.mse_loss(obj_rot_pred, obj_rot)
                    
                    # # 使用一个综合指标或分别报告
                    # test_metric = loss_rot + 0.1 * loss_root_pos + 0.1 * loss_obj_trans + 0.1 * loss_obj_rot
                    # # test_metric = loss_rot + 0.1 * loss_root_pos # 只关注人体
                    
                    # # 记录损失 (评估指标)
                    # test_loss += test_metric.item()
                    # test_loss_rot += loss_rot.item()
                    # test_loss_root_pos += loss_root_pos.item()
                    # test_loss_obj_trans += loss_obj_trans.item()
                    # test_loss_obj_rot += loss_obj_rot.item()

                    motion_pred = pred_dict["motion"]
                    obj_rot_pred = pred_dict["obj_rot"]
                    loss_rot = torch.nn.functional.mse_loss(motion_pred, motion)
                    loss_obj_rot = torch.nn.functional.mse_loss(obj_rot_pred, obj_rot)
                    test_metric = loss_rot + loss_obj_rot
                    test_loss += test_metric.item()
                    test_loss_rot += loss_rot.item()
                    test_loss_obj_rot += loss_obj_rot.item()
                    
                    # # 更新tqdm描述
                    # test_iter.set_postfix({
                    #     'test_metric': test_metric.item(),
                    #     'loss_rot': loss_rot.item(),
                    #     'loss_root_pos': loss_root_pos.item(),
                    #     'loss_obj': loss_obj_trans.item() + loss_obj_rot.item()
                    # })
                    test_iter.set_postfix({
                        'test_metric': test_metric.item(),
                        'loss_motion': loss_rot.item(),
                        'loss_obj_rot': loss_obj_rot.item()
                    })
            
            # 计算平均测试损失 (评估指标)
            test_loss /= len(test_loader)
            test_loss_rot /= len(test_loader)
            test_loss_root_pos /= len(test_loader)
            test_loss_obj_trans /= len(test_loader)
            test_loss_obj_rot /= len(test_loader)
            
            test_losses['loss'].append(test_loss)
            test_losses['loss_rot'].append(test_loss_rot)
            test_losses['loss_root_pos'].append(test_loss_root_pos)
            test_losses['loss_obj_trans'].append(test_loss_obj_trans)
            test_losses['loss_obj_rot'].append(test_loss_obj_rot)
            
            print(f'Epoch {epoch}, Test Metric: {test_loss:.6f}, ' # 打印评估指标
                  f'Rot Loss: {test_loss_rot:.6f}, '             
                  f'Root Pos Loss: {test_loss_root_pos:.6f}, '  
                  f'Obj Trans Loss: {test_loss_obj_trans:.6f}, '
                  f'Obj Rot Loss: {test_loss_obj_rot:.6f}')    
            
            # if use_wandb:
            #     log_dict = {
            #         'test_metric': test_loss,
            #         'test_loss_rot': test_loss_rot,
            #         'test_loss_root_pos': test_loss_root_pos,
            #         'test_loss_obj_trans': test_loss_obj_trans,
            #         'test_loss_obj_rot': test_loss_obj_rot
            #     }
            #     wandb.log(log_dict, step=n_iter)
            if use_wandb:
                log_dict = {
                    'test_metric': test_loss,
                    'test_loss_rot': test_loss_rot,
                    'test_loss_obj_rot': test_loss_obj_rot
                }
                wandb.log(log_dict, step=n_iter)
            
            # 保存最佳模型 (基于测试评估指标)
            if test_loss < best_loss:
                best_loss = test_loss
                save_path = os.path.join(save_dir, f'epoch_{epoch}_best.pt')
                print(f'Saving best model to {save_path}')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss, # 保存的是最佳评估指标
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