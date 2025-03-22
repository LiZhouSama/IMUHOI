import argparse
import os
import numpy as np
import torch
import joblib
from tqdm import tqdm
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.rotation_tools import aa2matrot, local2global_pose
import multiprocessing as mp
from functools import partial

# IMU关节索引，可以根据需要修改
IMU_JOINTS = [20, 21, 15, 7, 8, 0]  # 左手、右手、头部、左脚、右脚、髋部
IMU_JOINT_NAMES = ['left_hand', 'right_hand', 'head', 'left_foot', 'right_foot', 'hip']

def compute_imu_data(position_global, rotation_global, imu_joints, smooth_n=4):
    """
    计算特定关节的IMU数据
    参数:
        position_global: 全局关节位置 [T, J, 3]
        rotation_global: 全局关节旋转 [T, J, 3, 3]
        imu_joints: IMU关节索引列表
        smooth_n: 平滑窗口大小
    返回:
        IMU数据字典，包含加速度和方向信息
    """
    # 提取指定关节的位置和旋转
    imu_positions = position_global[:, imu_joints, :]  # [T, num_imus, 3]
    imu_rotations = rotation_global[:, imu_joints, :, :]  # [T, num_imus, 3, 3]
    
    # 生成加速度数据，使用二阶差分计算，参考_syn_acc函数
    def _syn_acc(v):
        """从位置生成加速度"""
        mid = smooth_n // 2
        # 基础二阶差分计算
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
        # 增加边界填充
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        
        # 平滑处理
        if mid != 0:
            acc[smooth_n:-smooth_n] = torch.stack(
                [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
                 for i in range(0, v.shape[0] - smooth_n * 2)])
        return acc
    
    # 为每个IMU关节生成加速度
    imu_accelerations = torch.zeros_like(imu_positions)
    for i in range(imu_positions.shape[1]):
        imu_accelerations[:, i, :] = _syn_acc(imu_positions[:, i, :])
    
    # 返回IMU数据字典
    imu_data = {
        'positions': imu_positions,         # [T, num_imus, 3]
        'rotations': imu_rotations,         # [T, num_imus, 3, 3]
        'accelerations': imu_accelerations  # [T, num_imus, 3]
    }
    
    return imu_data

def process_sequence(seq_data, seq_key, save_dir, bm):
    """处理单个序列并保存为pt文件"""
    
    # 提取序列数据
    seq_name = seq_data['seq_name']
    bdata_poses = np.concatenate([seq_data['root_orient'].reshape(-1, 3), 
                                 seq_data['pose_body'].reshape(-1, 63)], axis=1)
    bdata_trans = seq_data['trans']
    subject_gender = seq_data['gender']
    
    # 处理帧率
    framerate = 60  # 默认使用60FPS
    
    # 构建body参数字典
    body_parms = {
        "root_orient": torch.tensor(seq_data['root_orient']).float(),
        "pose_body": torch.tensor(seq_data['pose_body']).float(),
        "trans": torch.tensor(seq_data['trans']).float(),
        "betas": torch.tensor(seq_data['betas']).float() if 'betas' in seq_data else None
    }
    
    # 使用SMPL模型获取全局姿态
    body_pose_world = bm(
        **{k: v.cuda() for k, v in body_parms.items() 
           if k in ["pose_body", "root_orient", "trans"] and v is not None}
    )
    
    # 计算局部旋转的6D表示
    output_aa = torch.tensor(bdata_poses).float().reshape(-1, 3)
    output_6d = aa2sixd(output_aa).reshape(bdata_poses.shape[0], -1)
    rotation_local_full_gt_list = output_6d[1:]
    
    # 计算全局旋转
    rotation_local_matrot = aa2matrot(torch.tensor(bdata_poses).reshape(-1, 3))
    rotation_local_matrot = rotation_local_matrot.reshape(bdata_poses.shape[0], -1, 9)
    
    # 只使用前22个关节的层次结构
    kintree_table = bm.kintree_table[0].long()[:22]  # 只取前22个关节的父子关系
    rotation_global_matrot = local2global_pose(
        rotation_local_matrot[:, :22], kintree_table
    )
    
    # 重塑全局旋转矩阵为标准形状 [T, J, 3, 3]
    rotation_global_matrot_reshaped = rotation_global_matrot.reshape(rotation_global_matrot.shape[0], -1, 3, 3)
    
    # 提取头部全局旋转
    head_rotation_global_matrot = rotation_global_matrot[:, [15], :, :]
    
    # 获取全局关节位置
    position_global_full_gt_world = body_pose_world.Jtr[:, :22, :].cpu()
    
    # 计算头部全局变换矩阵
    position_head_world = position_global_full_gt_world[:, 15, :]
    head_global_trans = torch.eye(4).repeat(position_head_world.shape[0], 1, 1)
    head_global_trans[:, :3, :3] = head_rotation_global_matrot.squeeze()
    head_global_trans[:, :3, 3] = position_global_full_gt_world[:, 15, :]
    head_global_trans_list = head_global_trans[1:]
    
    # 计算IMU数据
    imu_global_full_gt = compute_imu_data(
        position_global_full_gt_world, 
        rotation_global_matrot_reshaped,
        IMU_JOINTS
    )
    
    # 提取物体相关信息(如果存在)
    obj_data = {}
    if 'obj_scale' in seq_data:
        obj_data = {
            'obj_scale': seq_data['obj_scale'],
            'obj_trans': seq_data['obj_trans'],
            'obj_rot': seq_data['obj_rot'],
            'obj_com_pos': seq_data['obj_com_pos']
        }
    
    # 组装输出数据，移除hmd_position_global_full_gt_list
    data = {
        "seq_name": seq_name,
        "body_parms_list": body_parms,
        "rotation_local_full_gt_list": rotation_local_full_gt_list,
        "head_global_trans_list": head_global_trans_list,
        "position_global_full_gt_world": position_global_full_gt_world[1:].float(),
        "imu_global_full_gt": imu_global_full_gt,
        "framerate": framerate,
        "gender": subject_gender
    }
    
    # 添加物体数据(如果存在)
    if obj_data:
        data.update({
            "obj_scale": torch.tensor(obj_data["obj_scale"]).float(),
            "obj_trans": torch.tensor(obj_data["obj_trans"]).float(),
            "obj_rot": torch.tensor(obj_data["obj_rot"]).float(),
            "obj_com_pos": torch.tensor(obj_data["obj_com_pos"]).float()
        })
    
    # 保存处理后的数据
    torch.save(data, os.path.join(save_dir, f"{seq_key}.pt"))
    return 1

def aa2sixd(aa):
    """轴角转6D表示"""
    return matrot2sixd(aa2matrot(aa))

def matrot2sixd(m):
    """旋转矩阵转6D表示"""
    batch_size = m.shape[0]
    sixd = torch.zeros(batch_size, 6)
    sixd[:, 0:3] = m[:, :, 0]
    sixd[:, 3:6] = m[:, :, 1]
    return sixd

def main(args):
    # 创建输出目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载数据集
    print(f"正在加载数据集：{args.data_path}")
    data_dict = joblib.load(args.data_path)
    print(f"数据集加载完成，共有{len(data_dict)}个序列")
    
    # 设置SMPL模型
    print("加载SMPL模型...")
    bm_fname_male = os.path.join(args.support_dir, f"smplh/male/model.npz")
    
    num_betas = 16  # 身体参数数量
    bm_male = BodyModel(
        bm_fname=bm_fname_male,
        num_betas=num_betas,
    ).cuda()
    
    # 处理所有序列
    print("开始处理序列...")
    
    # 如果使用多进程
    if args.num_workers > 1:
        process_func = partial(process_sequence, save_dir=args.save_dir, bm=bm_male)
        with mp.Pool(args.num_workers) as pool:
            results = list(tqdm(
                pool.starmap(process_func, [(data_dict[k], k) for k in data_dict]),
                total=len(data_dict),
                desc="处理序列"
            ))
    else:
        # 单进程处理
        for seq_key in tqdm(data_dict, desc="处理序列"):
            process_sequence(data_dict[seq_key], seq_key, args.save_dir, bm_male)
    
    print(f"所有序列处理完成，结果保存在：{args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理人体动作数据集")
    parser.add_argument("--data_path", type=str, default="dataset/train_diffusion_manip_seq_joints24.p",
                        help="输入数据集路径(.p文件)")
    parser.add_argument("--save_dir", type=str, default="processed_data/train",
                        help="输出数据保存目录")
    parser.add_argument("--support_dir", type=str, default="body_models",
                        help="SMPL模型目录")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="并行处理的工作进程数量")
    parser.add_argument("--smooth_n", type=int, default=4,
                        help="IMU加速度平滑窗口大小")
    
    args = parser.parse_args()
    main(args)