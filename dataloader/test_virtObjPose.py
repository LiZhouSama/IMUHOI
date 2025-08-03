import torch
import numpy as np
import pytorch3d.transforms as transforms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.global_config import IMU_JOINT_NAMES
from dataloader import IMUDataset
from torch.utils.data import DataLoader

def verify_obj_direction_calculation(wrist_pos, obj_trans, obj_rot_mat):
    """
    验证物体坐标系下的方向向量^Ov_{HO}的计算
    
    Args:
        wrist_pos: [seq, 3] - 手腕位置
        obj_trans: [seq, 3] - 物体位置
        obj_rot_mat: [seq, 3, 3] - 物体旋转矩阵
    
    Returns:
        obj_direction_gt: [seq, 3] - 真值的^Ov_{HO}
        bone_length_gt: [seq] - 真值的骨长
    """
    # 1. 计算世界坐标系下的向量
    v_HO_world = obj_trans - wrist_pos  # [seq, 3]
    
    # 2. 计算骨长
    bone_length_gt = torch.norm(v_HO_world, dim=1)  # [seq]
    
    # 3. 归一化得到世界坐标系下的单位向量
    v_HO_world_unit = v_HO_world / (bone_length_gt.unsqueeze(-1) + 1e-8)  # [seq, 3]
    
    # 4. 转换到物体坐标系：^Ov_{HO} = ^WR_O^T * ^Wv_{HO}
    obj_rot_inv = obj_rot_mat.transpose(-1, -2)  # [seq, 3, 3]
    obj_direction_gt = torch.bmm(obj_rot_inv, v_HO_world_unit.unsqueeze(-1)).squeeze(-1)  # [seq, 3]
    
    return obj_direction_gt, bone_length_gt

def verify_object_position_FK(wrist_pos, obj_rot_mat, obj_direction, bone_length):
    """
    用FK公式重建物体位置：p_o = p_H + ^WR_O * ^Ov_{HO} * bone_length
    
    Args:
        wrist_pos: [seq, 3] - 手腕位置
        obj_rot_mat: [seq, 3, 3] - 物体旋转矩阵
        obj_direction: [seq, 3] - 物体坐标系下的方向向量^Ov_{HO}
        bone_length: [seq] - 骨长
    
    Returns:
        predicted_pos: [seq, 3] - 预测的物体位置
    """
    # FK公式：p_o = p_H + ^WR_O * ^Ov_{HO} * bone_length
    # 将方向向量转换到世界坐标系并乘以骨长
    bone_vector_world = torch.bmm(obj_rot_mat, obj_direction.unsqueeze(-1)).squeeze(-1)  # [seq, 3]
    bone_vector_world = bone_vector_world * bone_length.unsqueeze(-1)  # [seq, 3]
    
    predicted_pos = wrist_pos + bone_vector_world  # [seq, 3]
    return predicted_pos

def test_virtual_joint_accuracy(batch_data, test_frames=10, hand_name='left'):
    """
    测试新的虚拟关节逻辑的准确性
    
    Args:
        batch_data: dataloader返回的批次数据
        test_frames: 测试帧数
        hand_name: 'left' 或 'right'
    
    Returns:
        dict: 包含误差统计的字典
    """
    print(f"\n=== 测试{hand_name}手新虚拟关节逻辑准确性 ===")
    
    # 检查是否有物体数据
    if not batch_data["has_object"]:
        print("没有物体数据，跳过测试")
        return None
    
    # 获取归一化后的数据（与虚拟关节计算一致）
    obj_trans_norm = batch_data["obj_trans"]  # [seq, 3] - 归一化后的物体位置
    obj_rot_norm_6d = batch_data["obj_rot"]  # [seq, 6] - 归一化后的6D旋转表示
    obj_rot_norm_mat = transforms.rotation_6d_to_matrix(obj_rot_norm_6d)  # [seq, 3, 3]
    
    # 获取归一化后的手腕位置
    wrist_joint_idx = 20 if hand_name == 'left' else 21
    wrist_pos_norm = batch_data["position_global_norm"][:, wrist_joint_idx, :]  # [seq, 3]
    
    # 获取方向向量数据（新格式）
    hand_prefix = hand_name[0]  # 'l' 或 'r'
    obj_direction_pred = batch_data[f"{hand_prefix}hand_obj_direction"]  # [seq, 3]
    
    # 找到有效的接触帧
    contact_mask = batch_data[f"{hand_prefix}hand_contact"]  # [seq]
    valid_indices = torch.where(contact_mask)[0]
    
    if len(valid_indices) == 0:
        print(f"没有{hand_name}手接触帧，跳过测试")
        return None
    
    # 选择测试帧
    test_indices = valid_indices[:min(test_frames, len(valid_indices))]
    print(f"测试{len(test_indices)}帧的接触数据")
    
    # === 测试1：验证^Ov_{HO}计算的正确性 ===
    print(f"\n--- 测试1：验证{hand_name}手^Ov_{{HO}}计算 ---")
    
    # 计算真值
    obj_direction_gt, bone_length_gt = verify_obj_direction_calculation(
        wrist_pos_norm, obj_trans_norm, obj_rot_norm_mat
    )
    
    direction_errors = []
    
    for i, idx in enumerate(test_indices):
        # 比较方向向量
        direction_error = torch.norm(obj_direction_pred[idx] - obj_direction_gt[idx]).item()
        direction_errors.append(direction_error)
        
        if i < 3:  # 只打印前3帧的详细信息
            print(f"帧{idx.item()}:")
            print(f"  方向向量误差: {direction_error:.6f}")
            print(f"  预测方向: {obj_direction_pred[idx].numpy()}")
            print(f"  真值方向: {obj_direction_gt[idx].numpy()}")
            print(f"  真值骨长: {bone_length_gt[idx].item():.6f}m")
    
    avg_direction_error = np.mean(direction_errors)
    print(f"平均方向向量误差: {avg_direction_error:.6f}")
    
    # === 测试2：验证FK公式的正确性 ===
    print(f"\n--- 测试2：验证{hand_name}手FK公式 ---")
    
    # 使用预测的数据进行FK计算
    predicted_obj_pos = verify_object_position_FK(
        wrist_pos_norm, obj_rot_norm_mat, obj_direction_pred, bone_length_gt # Use bone_length_gt from verify_obj_direction_calculation
    )
    
    # 使用真值数据进行FK计算（作为对照）
    gt_obj_pos = verify_object_position_FK(
        wrist_pos_norm, obj_rot_norm_mat, obj_direction_gt, bone_length_gt
    )
    
    fk_errors_pred = []
    fk_errors_gt = []
    
    for i, idx in enumerate(test_indices):
        # 预测数据的FK误差
        fk_error_pred = torch.norm(predicted_obj_pos[idx] - obj_trans_norm[idx]).item()
        fk_errors_pred.append(fk_error_pred)
        
        # 真值数据的FK误差（应该接近0）
        fk_error_gt = torch.norm(gt_obj_pos[idx] - obj_trans_norm[idx]).item()
        fk_errors_gt.append(fk_error_gt)
        
        if i < 3:  # 只打印前3帧的详细信息
            print(f"帧{idx.item()}:")
            print(f"  预测FK误差: {fk_error_pred:.6f}m")
            print(f"  真值FK误差: {fk_error_gt:.6f}m")
            print(f"  预测物体位置: {predicted_obj_pos[idx].numpy()}")
            print(f"  真实物体位置: {obj_trans_norm[idx].numpy()}")
    
    avg_fk_error_pred = np.mean(fk_errors_pred)
    avg_fk_error_gt = np.mean(fk_errors_gt)
    print(f"平均预测FK误差: {avg_fk_error_pred:.6f}m")
    print(f"平均真值FK误差: {avg_fk_error_gt:.6f}m")
    
    # === 测试3：验证方向向量的单位向量性质 ===
    print(f"\n--- 测试3：验证{hand_name}手方向向量性质 ---")
    
    direction_norms = []
    for i, idx in enumerate(test_indices):
        norm = torch.norm(obj_direction_pred[idx]).item()
        direction_norms.append(norm)
        
        if i < 3:
            print(f"帧{idx.item()}: 方向向量模长 = {norm:.6f} (应该接近1.0)")
    
    avg_norm = np.mean(direction_norms)
    norm_std = np.std(direction_norms)
    print(f"平均方向向量模长: {avg_norm:.6f} ± {norm_std:.6f}")
    
    # 返回统计结果
    results = {
        'hand_name': hand_name,
        'test_frames': len(test_indices),
        'avg_direction_error': avg_direction_error,
        'avg_fk_error_pred': avg_fk_error_pred,
        'avg_fk_error_gt': avg_fk_error_gt,
        'avg_direction_norm': avg_norm,
        'direction_norm_std': norm_std,
    }
    
    return results

def debug_virtual_joint_calculation(batch_data, hand_name='left', frame_idx=0):
    """
    调试单帧的虚拟关节计算过程
    
    Args:
        batch_data: batch数据
        hand_name: 手名称
        frame_idx: 调试的帧索引
    """
    print(f"\n=== 调试{hand_name}手第{frame_idx}帧的虚拟关节计算 ===")
    
    if not batch_data["has_object"]:
        print("没有物体数据")
        return
    
    # 检查接触状态
    hand_prefix = hand_name[0]
    contact_mask = batch_data[f"{hand_prefix}hand_contact"]
    if not contact_mask[frame_idx]:
        print(f"第{frame_idx}帧不是接触帧")
        return
    
    # 获取数据
    obj_trans = batch_data["obj_trans"][frame_idx]  # [3]
    obj_rot_6d = batch_data["obj_rot"][frame_idx]  # [6]
    obj_rot_mat = transforms.rotation_6d_to_matrix(obj_rot_6d.unsqueeze(0)).squeeze(0)  # [3, 3]
    
    wrist_joint_idx = 20 if hand_name == 'left' else 21
    wrist_pos = batch_data["position_global_norm"][frame_idx, wrist_joint_idx, :]  # [3]
    
    obj_direction_pred = batch_data[f"{hand_prefix}hand_obj_direction"][frame_idx]  # [3]
    
    print(f"物体位置: {obj_trans.numpy()}")
    print(f"手腕位置: {wrist_pos.numpy()}")
    print(f"物体旋转矩阵:\n{obj_rot_mat.numpy()}")
    
    # 手动计算真值
    v_HO_world = obj_trans - wrist_pos
    bone_length_gt = torch.norm(v_HO_world)
    v_HO_world_unit = v_HO_world / bone_length_gt
    obj_direction_gt = obj_rot_mat.T @ v_HO_world_unit
    
    print(f"\n手动计算:")
    print(f"世界坐标系向量: {v_HO_world.numpy()}")
    print(f"骨长: {bone_length_gt.item():.6f}m")
    print(f"世界坐标系单位向量: {v_HO_world_unit.numpy()}")
    print(f"物体坐标系方向向量: {obj_direction_gt.numpy()}")
    
    print(f"\nDataLoader计算:")
    print(f"物体坐标系方向向量: {obj_direction_pred.numpy()}")
    
    print(f"\n误差:")
    print(f"方向向量误差: {torch.norm(obj_direction_pred - obj_direction_gt).item():.6f}")
    
    # 验证FK：使用真值骨长进行验证
    predicted_pos = wrist_pos + obj_rot_mat @ obj_direction_pred * bone_length_gt
    fk_error = torch.norm(predicted_pos - obj_trans).item()
    print(f"FK重建误差 (使用真值骨长): {fk_error:.6f}m")

def run_virtual_joint_test():
    """
    运行虚拟关节测试
    """
    print("=== 开始新虚拟关节逻辑测试 ===")
    
    # 创建数据集
    dataset = IMUDataset(
        data_dir="processed_data_0701/test", 
        window_size=60,
        normalize=True,
        debug=True
    )
    
    if len(dataset) == 0:
        print("数据集为空")
        return
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"数据集大小: {len(dataset)}")
    
    tested_sequences = 0
    successful_tests = 0
    
    for batch_data in dataloader:
        # 移除batch维度
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.squeeze(0)
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        batch_data[key][k] = v.squeeze(0)
        
        tested_sequences += 1
        
        if not batch_data["has_object"]:
            print(f"序列{tested_sequences}: 没有物体数据，跳过")
            continue
        
        print(f"\n=== 测试序列{tested_sequences}: {batch_data['obj_name']} ===")
        
        # 调试单帧计算（可选）
        lhand_contact = batch_data["lhand_contact"]
        if lhand_contact.any():
            contact_frame = torch.where(lhand_contact)[0][0].item()
            debug_virtual_joint_calculation(batch_data, hand_name='left', frame_idx=contact_frame)
        
        # 测试左手和右手
        sequence_success = False
        for hand_name in ['left', 'right']:
            results = test_virtual_joint_accuracy(batch_data, test_frames=5, hand_name=hand_name)
            if results is not None:
                print(f"\n{hand_name}手测试结果:")
                print(f"  方向向量误差: {results['avg_direction_error']:.6f}")
                print(f"  FK误差: {results['avg_fk_error_pred']:.6f}m")
                print(f"  方向向量模长: {results['avg_direction_norm']:.6f}")
                sequence_success = True
        
        if sequence_success:
            successful_tests += 1
        
        # 测试5个序列后退出
        if tested_sequences >= 5:
            break
    
    print(f"\n=== 测试完成 ===")
    print(f"测试序列数: {tested_sequences}")
    print(f"成功测试数: {successful_tests}")

def test_single_sequence():
    """
    测试单个序列的详细分析
    """
    print("=== 单序列详细测试 ===")
    
    dataset = IMUDataset(
        data_dir="processed_data_0701/test",
        window_size=60,
        normalize=True,
        debug=True
    )
    
    if len(dataset) == 0:
        print("数据集为空")
        return
    
    # 获取第一个序列
    batch_data = dataset[0]
    
    # 分析接触情况
    lhand_contact = batch_data["lhand_contact"]
    rhand_contact = batch_data["rhand_contact"]
    
    print(f"序列长度: {len(lhand_contact)}")
    print(f"左手接触帧数: {lhand_contact.sum().item()}")
    print(f"右手接触帧数: {rhand_contact.sum().item()}")
    
    # 测试虚拟关节数据
    if batch_data["has_object"]:
        print(f"物体名称: {batch_data['obj_name']}")
        
        # 测试左手
        if lhand_contact.any():
            print("\n--- 左手详细测试 ---")
            contact_frames = torch.where(lhand_contact)[0][:3]  # 测试前3帧
            for frame_idx in contact_frames:
                debug_virtual_joint_calculation(batch_data, hand_name='left', frame_idx=frame_idx.item())
            
            results_left = test_virtual_joint_accuracy(batch_data, test_frames=10, hand_name='left')
        
        # 测试右手
        if rhand_contact.any():
            print("\n--- 右手详细测试 ---")
            contact_frames = torch.where(rhand_contact)[0][:3]  # 测试前3帧
            for frame_idx in contact_frames:
                debug_virtual_joint_calculation(batch_data, hand_name='right', frame_idx=frame_idx.item())
            
            results_right = test_virtual_joint_accuracy(batch_data, test_frames=10, hand_name='right')
    else:
        print("该序列没有物体数据")

if __name__ == "__main__":
    # 运行测试
    run_virtual_joint_test()
    
    # 可选：运行单序列详细测试
    # test_single_sequence() 