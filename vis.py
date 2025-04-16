import torch
import os
import numpy as np
import random
import argparse
import yaml
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.viewer import Viewer
import pytorch3d.transforms as transforms
import trimesh

# 导入SMPLH模型处理库
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.rotation_tools import local2global_pose
from easydict import EasyDict as edict

# 导入数据加载相关库
# from torch.utils.data import DataLoader # 不再需要
# from dataloader.dataloader import IMUDataset # 不再需要，因为我们在本地处理

# 导入所有需要的模型
from models.DiT_model import MotionDiffusion
# 添加对 TransPose 模型的支持
from models.do_train_imu_TransPose import load_transpose_model
from models.do_train_imu_TransPose_humanOnly import load_transpose_model_humanOnly

# 加载配置
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

vis_config = load_config('configs/vis.yaml')

# 加载SMPLH模型
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
body_model = BodyModel(bm_fname=vis_config.get('bm_path', 'body_models/smplh/neutral/model.npz'), 
                       num_betas=16, model_type='smplh').to(device)

# --- 定义 Z-up 到 Y-up 的旋转矩阵 ---
R_yup = torch.tensor([[1.0, 0.0, 0.0], 
                    [0.0, 0.0, 1.0], 
                    [0.0, -1.0, 0.0]], dtype=torch.float32) # -90度绕X轴
# --- 结束定义 ---

def find_pt_files(directory):
    """查找目录下所有.pt文件"""
    pt_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pt'):
                pt_files.append(os.path.join(root, file))
    return pt_files

def process_loaded_data(seq_data, normalize=True):
    """
    处理从 .pt 文件直接加载的原始数据，模拟 DataLoader 的行为。
    Args:
        seq_data: 从 torch.load() 加载的原始字典。
        normalize: 是否对 IMU 数据进行归一化。 
    Returns:
        处理后的数据字典。
    """
    processed = {}
    seq_len = seq_data.get("rotation_local_full_gt_list", torch.empty(0)).shape[0]
    if seq_len == 0:
        print("警告：序列长度为 0")
        return {} # 返回空字典或引发错误

    # 提取基本数据
    processed["root_pos"] = seq_data.get("position_global_full_gt_world", torch.zeros(seq_len, 1, 3))[:, 0, :] # [T, 3]
    processed["motion"] = seq_data.get("rotation_local_full_gt_list", torch.zeros(seq_len, 132)) # [T, 132]
    processed["head_global_trans"] = seq_data["head_global_trans"] # [T, 4, 4]

    # 提取和组合人体 IMU
    human_imu = None
    if "imu_global_full_gt" in seq_data:
        human_imu_acc = seq_data["imu_global_full_gt"].get("accelerations", None)
        # 修改：读取方向而不是角速度
        human_imu_ori = seq_data["imu_global_full_gt"].get("orientations", None)
        # 将方向矩阵展平为 9D 向量
        human_imu_ori_flat = human_imu_ori.reshape(seq_len, -1, 9) # [T, num_imus, 9]
        # 将加速度 (3D) 和展平的方向 (9D) 拼接
        human_imu = torch.cat([human_imu_acc, human_imu_ori_flat], dim=-1)  # [T, num_imus, 12]
    else:
        num_imus = 6 # 假设默认 6 个 IMU
        # 修改：调整默认形状以匹配 acc (3) + ori_flat (9) = 12
        human_imu = torch.zeros(seq_len, num_imus, 12)
    
    # 处理物体数据
    has_object = "obj_trans" in seq_data and seq_data["obj_trans"] is not None
    processed["has_object"] = has_object

    obj_imu = None

    if has_object:
        obj_scale = seq_data.get("obj_scale", None)
        obj_trans = seq_data.get("obj_trans", None)
        obj_rot = seq_data.get("obj_rot", None)

        if "obj_imu" in seq_data:
            obj_imu_acc = seq_data["obj_imu"].get("accelerations", None)
            # 获取物体IMU方向 - 应该是6D表示
            obj_imu_ori = seq_data["obj_imu"].get("orientations", None)
            obj_imu_ori_flat = obj_imu_ori.reshape(seq_len, -1, 9) # [T, num_imus, 9]
            # 将加速度 (3D) 和 6D旋转表示 拼接
            obj_imu = torch.cat([obj_imu_acc, obj_imu_ori_flat], dim=-1) # [T, num_imus, 12]
        else:
            # 修改：调整默认形状以匹配 acc (3) + ori (6) = 9
            obj_imu = torch.zeros(seq_len, 1, 9)

    # 注意：归一化逻辑
    if normalize:
        # 对人体IMU归一化 - 现在已经是9D数据(加速度3D + 旋转6D)
        if human_imu is not None:
            T, num_imus, _ = human_imu.shape
            # 分离加速度和旋转
            human_acc = human_imu[..., :3]  # [T, num_imus, 3]
            human_ori = human_imu[..., 3:12].reshape(T, -1, 3, 3)  # [T, num_imus, 3, 3]
            
            # 对加速度归一化 - 减去第一帧
            norm_human_acc = human_acc - human_acc[0:1]
            
            # 对于6D旋转表示，归一化方法需要先转换回矩阵，然后应用相对旋转
            norm_human_ori = torch.zeros(T, num_imus, 6)
            
            for i in range(num_imus):
                # 将6D转换为旋转矩阵
                human_ori_mat = human_ori[:, i]  # [T, 3, 3]
                
                # 获取第一帧的旋转矩阵并计算其逆
                first_orient = human_ori_mat[0]  # [3, 3]
                first_orient_inv = torch.inverse(first_orient)  # [3, 3]
                
                # 计算相对旋转
                rel_rotations = torch.matmul(first_orient_inv.unsqueeze(0), human_ori_mat)  # [T, 3, 3]
                
                # 转回6D表示
                rel_rotations_6d = transforms.matrix_to_rotation_6d(rel_rotations)  # [T, 6]
                norm_human_ori[:, i] = rel_rotations_6d
            
            # 重新组合IMU数据
            processed_human_imu = torch.cat([norm_human_acc, norm_human_ori], dim=-1)  # [T, num_imus, 9]
        else:
            processed_human_imu = None
            
        # 对物体IMU做相同处理    
        if obj_imu is not None and has_object:
            # 分离加速度和旋转
            obj_acc = obj_imu[..., :3]  # [T, 1, 3]
            obj_ori = obj_imu[..., 3:12]  # [T, 1, 9]
            
            # 对加速度归一化
            norm_obj_acc = obj_acc - obj_acc[0:1]
            
            # 对旋转归一化
            T = obj_acc.shape[0]
            obj_ori_mat = obj_ori.reshape(T, 3, 3)
            first_orient = obj_ori_mat[0]  # [3, 3]
            first_orient_inv = torch.inverse(first_orient)  # [3, 3]
            rel_rotations = torch.matmul(first_orient_inv.unsqueeze(0), obj_ori_mat)  # [T, 3, 3]
            rel_rotations_6d = transforms.matrix_to_rotation_6d(rel_rotations)  # [T, 6]
            
            # 重新组合
            processed_obj_imu = torch.cat([norm_obj_acc, rel_rotations_6d.reshape(T, 1, 6)], dim=-1)  # [T, 1, 9]
        else:
            processed_obj_imu = None
    else:
        processed_human_imu = human_imu.float() if human_imu is not None else None
        processed_obj_imu = obj_imu.float() if obj_imu is not None else None
    
    processed["human_imu"] = processed_human_imu
    processed["root_pos"] = processed["root_pos"].float()
    processed["motion"] = processed["motion"].float()

    if has_object:
        processed["obj_imu"] = processed_obj_imu
        processed["obj_scale"] = obj_scale
        processed["obj_trans"] = obj_trans
        processed["obj_rot"] = obj_rot
        processed["obj_name"] = seq_data.get("obj_name", "unknown")
        processed["gender"] = seq_data.get("gender", "neutral") # 或其他默认值
        # 转换其他字段为 float
        if processed["obj_trans"] is not None:
            processed["obj_trans"] = processed["obj_trans"].float()
        if processed["obj_rot"] is not None:
            processed["obj_rot"] = processed["obj_rot"].float()
        if processed["obj_scale"] is not None:
            processed["obj_scale"] = processed["obj_scale"].float() # 确保 scale 也是 float

    return processed

def apply_transformation_to_obj_geometry(obj_mesh_path, obj_rot, obj_trans, scale=None):
    """
    应用变换到物体网格 (OMOMO 方式)

    参数:
        obj_mesh_path: 物体网格路径
        obj_rot: 旋转矩阵 [T, 3, 3]
        obj_trans: 平移向量 [T, 3] (不含缩放)
        scale: 缩放因子 [T] 或 [T, 1] 或 [T, 1, 1]

    返回:
        transformed_obj_verts: 变换后的顶点 [T, Nv, 3]
        obj_mesh_faces: 物体网格的面 [Nf, 3]
    """
    try:
        mesh = trimesh.load_mesh(obj_mesh_path)
        obj_mesh_verts = np.asarray(mesh.vertices) # Nv X 3
        obj_mesh_faces = np.asarray(mesh.faces) # Nf X 3

        ori_obj_verts = torch.from_numpy(obj_mesh_verts).float()[None].repeat(obj_trans.shape[0], 1, 1) # T X Nv X 3

        # --- 首先应用缩放 (如果提供) ---
        if scale is not None:
            # 确保 scale 是 tensor 并且在正确的 device 上
            if not torch.is_tensor(scale):
                scale_tensor = torch.tensor(scale).float()
            else:
                scale_tensor = scale.float()
            scale_tensor = scale_tensor.to(ori_obj_verts.device)

            # 确保 scale_tensor 可以广播: [T, 1, 1]
            if scale_tensor.dim() == 1:
                 scale_tensor = scale_tensor.unsqueeze(-1).unsqueeze(-1) # T -> T x 1 x 1
            elif scale_tensor.dim() == 2 and scale_tensor.shape[1] == 1:
                 scale_tensor = scale_tensor.unsqueeze(-1) # T x 1 -> T x 1 x 1
            elif scale_tensor.dim() == 3 and scale_tensor.shape[1:] == (1, 1):
                 pass # 已经是 T x 1 x 1
            else:
                 print(f"警告: scale 维度无法处理 {scale_tensor.shape}, 将不应用缩放。")
                 scale_tensor = None # 重置以跳过缩放

            if scale_tensor is not None:
                 ori_obj_verts = ori_obj_verts * scale_tensor
        # --- 结束缩放应用 ---

        seq_rot_mat = torch.tensor(obj_rot).float() # T X 3 X 3
        seq_trans = torch.tensor(obj_trans).float() # T X 3 (不含缩放)

        # 确保trans的维度正确
        if seq_trans.dim() == 1:
             if seq_trans.shape[0] == obj_rot.shape[0] * 3:
                 seq_trans = seq_trans.reshape(obj_rot.shape[0], 3)
             else:
                 print(f"警告：物体平移维度无法解析 {seq_trans.shape}，将使用零向量")
                 seq_trans = torch.zeros((seq_rot_mat.shape[0], 3), dtype=torch.float32)
        elif seq_trans.shape[1] != 3:
             print(f"警告：物体平移维度错误 {seq_trans.shape}，将使用零向量")
             seq_trans = torch.zeros((seq_rot_mat.shape[0], 3), dtype=torch.float32)

        # seq_trans = seq_trans[:, :, None]  # T X 3 X 1

        # --- 应用旋转和平移到 (可能) 已缩放的顶点 ---
        transformed_obj_verts = seq_rot_mat.bmm(ori_obj_verts.transpose(1, 2)) + seq_trans
        transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X Nv X 3
    except Exception as e:
        print(f"应用变换到物体几何体失败: {e}")
        transformed_obj_verts = torch.zeros((obj_trans.shape[0] if obj_trans is not None else 1, 1, 3))
        obj_mesh_faces = np.zeros((1, 3), dtype=np.int64)

    return transformed_obj_verts, obj_mesh_faces

# --- (保留 merge_two_parts) ---
def merge_two_parts(verts_list, faces_list):
    """
    合并两个网格部分
    
    参数:
        verts_list: 顶点列表，每个元素形状为 [T, Nv, 3]
        faces_list: 面列表，每个元素形状为 [Nf, 3]
        
    返回:
        merged_verts: 合并后的顶点 [T, Nv_total, 3]
        merged_faces: 合并后的面 [Nf_total, 3]
    """
    verts_num = 0
    merged_verts_list = []
    merged_faces_list = []
    for p_idx in range(len(verts_list)):
        part_verts = verts_list[p_idx] # T X Nv X 3 
        part_faces = torch.from_numpy(faces_list[p_idx]) # Nf X 3 

        if p_idx == 0:
            merged_verts_list.append(part_verts)
            merged_faces_list.append(part_faces)
        else:
            merged_verts_list.append(part_verts)
            merged_faces_list.append(part_faces+verts_num)

        verts_num += part_verts.shape[1] 

    merged_verts = torch.cat(merged_verts_list, dim=1)
    merged_faces = torch.cat(merged_faces_list, dim=0).cpu().numpy()

    return merged_verts, merged_faces

# --- 恢复 load_object_geometry 以接收 obj_scale ---
def load_object_geometry(obj_name, obj_rot, obj_trans, obj_scale=None, obj_bottom_trans=None, obj_bottom_rot=None):
    """
    加载物体几何体并应用变换 (OMOMO 方式)

    参数:
        obj_name: 物体名称
        obj_rot: 旋转矩阵
        obj_trans: 平移向量(不含缩放)
        obj_scale: 物体缩放因子
        obj_bottom_trans: 底部平移向量（不含缩放，除非 process_loaded_data 中处理了）
        obj_bottom_rot: 底部旋转矩阵
    """
    if obj_name is None:
        print("警告: 物体名称为 None，无法加载几何体。")
        return torch.zeros((1, 1, 3)), np.zeros((1, 3), dtype=np.int64)

    obj_geo_root_folder = os.path.join('./dataset/captured_objects')
    obj_mesh_path = os.path.join(obj_geo_root_folder, f"{obj_name}_cleaned_simplified.obj")

    # 检查是否是有两部分的物体
    # 假设如果存在 bottom_trans，那么也应该存在 bottom_rot (并且 process_loaded_data 提供了它们)
    two_parts = obj_name in ["vacuum", "mop"] and obj_bottom_trans is not None and obj_bottom_rot is not None

    if two_parts:
        top_obj_mesh_path = os.path.join(obj_geo_root_folder, f"{obj_name}_cleaned_simplified_top.obj")
        bottom_obj_mesh_path = os.path.join(obj_geo_root_folder, f"{obj_name}_cleaned_simplified_bottom.obj")

        if not os.path.exists(top_obj_mesh_path) or not os.path.exists(bottom_obj_mesh_path):
             print(f"警告: 找不到物体 {obj_name} 的两部分几何文件。将尝试加载整体文件。")
             two_parts = False
             obj_mesh_path = os.path.join(obj_geo_root_folder, f"{obj_name}_cleaned_simplified.obj")
        else:
            # --- 传递 scale 给 apply_transformation_to_obj_geometry ---
            # 假设 bottom part 使用与 top part 相同的 scale
            top_obj_mesh_verts, top_obj_mesh_faces = apply_transformation_to_obj_geometry(top_obj_mesh_path,
                                                                                        obj_rot,
                                                                                        obj_trans,
                                                                                        scale=obj_scale)
            bottom_obj_mesh_verts, bottom_obj_mesh_faces = apply_transformation_to_obj_geometry(bottom_obj_mesh_path,
                                                                                             obj_bottom_rot,
                                                                                             obj_bottom_trans, # 假设已缩放 (来自 process_loaded_data)
                                                                                             scale=obj_scale) # 不再对底部应用 scale? 或使用 bottom_scale?
                                                                                             # OMOMO 原始代码似乎在这里也传递了 scale，我们先保持一致
                                                                                             # scale=obj_scale) # 假设底部也用相同 scale
            # 注意：如果 bottom_trans 确实需要单独的 scale，这里的逻辑可能需要调整以匹配 OMOMO 的精确实现

            obj_mesh_verts, obj_mesh_faces = merge_two_parts([top_obj_mesh_verts, bottom_obj_mesh_verts],
                                                            [top_obj_mesh_faces, bottom_obj_mesh_faces])

    if not two_parts:
        if not os.path.exists(obj_mesh_path):
             print(f"警告: 找不到物体几何文件: {obj_mesh_path}")
             return torch.zeros((obj_trans.shape[0] if obj_trans is not None else 1, 1, 3)), np.zeros((1, 3), dtype=np.int64)

        # --- 传递 scale 给 apply_transformation_to_obj_geometry ---
        obj_mesh_verts, obj_mesh_faces = apply_transformation_to_obj_geometry(obj_mesh_path,
                                                                          obj_rot,
                                                                          obj_trans,
                                                                          scale=obj_scale)

    return obj_mesh_verts, obj_mesh_faces

# --- 修改 visualize_human_and_objects 以使用新的 scale/trans 逻辑 ---
def visualize_human_and_objects(args, model, data, show_objects=True):
    """
    可视化人体和物体的真值和预测值
    
    参数:
        model: EgoIMU模型实例，如果为None则从config.model_path加载
        cfg: 配置字典 (来自 diffusion_config)
    """

    if data is None:
        print("错误：未能加载任何数据。")
        return

    v = Viewer(fps=1) # Viewer 初始化

    # 处理真值数据 (现在使用处理后的 'data' 字典)
    has_gt = False
    verts_gt_yup = None
    faces_gt = None
    if 'motion' in data and data['motion'][0].numel() > 0: # 检查 motion 是否有效
        has_gt = True
        motion = data['motion'][0].to(device)  # [T, 132]
        gt_root_pos = data['root_pos'][0].to(device)  # [T, 3]
        
        # --- (SMPLH 模型输入准备 - 这部分逻辑不变) ---
        gt_rot_matrices = transforms.rotation_6d_to_matrix(motion.reshape(-1, 22, 6))  # [T, 22, 3, 3]
        gt_root_orient_mat = gt_rot_matrices[:, 0, :, :].to(device)
        gt_pose_body_mat = gt_rot_matrices[:, 1:, :, :].reshape(-1, 21, 3, 3).to(device)
        gt_root_orient_axis = transforms.matrix_to_axis_angle(gt_root_orient_mat) # [T, 3]
        gt_pose_body_axis = transforms.matrix_to_axis_angle(gt_pose_body_mat.reshape(-1, 3, 3)).reshape(motion.shape[0], -1) # [T, 63]
        gt_smplh_input = {
            'root_orient': gt_root_orient_axis,
            'pose_body': gt_pose_body_axis,
            'trans': gt_root_pos
        }
        body_pose_gt = body_model(**gt_smplh_input)
        verts_gt = body_pose_gt.v.detach().cpu()
        faces_gt = body_model.f.detach().cpu().numpy() if isinstance(body_model.f, torch.Tensor) else body_model.f
        # --- 结束 SMPLH 输入准备 ---
    else:
         print("警告：未找到有效的 'motion' 数据，无法显示真值人体。")

    # 处理预测数据
    has_pred = False
    pred_obj_rot = None
    verts_pred = None

    if model is not None:
        # 如果提供了模型，使用模型进行预测
        try:
            # --- 准备模型输入 (现在使用处理后的 'data' 字典) ---
            # 应该使用 process_loaded_data 返回的规范化 IMU 数据
            human_imu_input = data['human_imu'][0].to(device)  # [T, num_imus, 9]
            # 检查并确保维度符合模型预期: [bs, seq, num_imus, 9] 和 [bs, seq, 1, 9]
            if human_imu_input.dim() == 3: # 已经是 [T, num_imus, 9]
                human_imu_input = human_imu_input.unsqueeze(0) # 添加批次维度 -> [1, T, num_imus, 9]
            elif human_imu_input.dim() == 4: # 已经是 [1, T, num_imus, 9]
                 pass # 维度正确
            else:
                 print(f"警告：human_imu 输入维度不正确: {human_imu_input.shape}")
                 # 尝试修正或跳过预测
                 raise ValueError("human_imu 维度错误")

            if data.get('has_object', [False])[0]:
                obj_imu_input = data['obj_imu'][0].to(device)      # [T, 1, 9]
                if obj_imu_input.dim() == 3: # 已经是 [T, 1, 9]
                 obj_imu_input = obj_imu_input.unsqueeze(0) # 添加批次维度 -> [1, T, 1, 9]
                elif obj_imu_input.dim() == 4: # 已经是 [1, T, 1, 9]
                    pass # 维度正确
                else:
                    print(f"警告：obj_imu 输入维度不正确: {obj_imu_input.shape}")
                    # 尝试修正或跳过预测
                    raise ValueError("obj_imu 维度错误")

            if args.use_transpose_humanOnly_model:
                model_input = {
                    "human_imu": human_imu_input
                }
            else:
                model_input = {
                    "human_imu": human_imu_input,
                    "obj_imu": obj_imu_input
                }
            
            # 如果有BPS特征，添加到输入中 (当前省略 BPS 处理)
            # if 'bps_features' in data:
            #     model_input['bps_features'] = data['bps_features'][0].unsqueeze(0).to(device)
            # --- 结束模型输入准备 ---

            # 运行模型推理
            with torch.no_grad():
                # 检查模型类型并进行相应的推理
                if hasattr(model, 'diffusion_reverse'):
                    # DiT 模型使用 diffusion_reverse 方法
                    pred_dict = model.diffusion_reverse(model_input)
                else:
                    # TransPose 模型直接前向传播
                    pred_dict = model(model_input)
            
            # --- (提取预测结果 - 这部分逻辑不变) ---
            pred_motion = pred_dict.get("motion", None)        # [1, T, 132]
            pred_root_pos = pred_dict.get("root_pos", None)    # [1, T, 3]
            pred_obj_trans = pred_dict.get("obj_trans", None)  # [T, 3]
            if pred_obj_trans is not None:
                pred_obj_trans = pred_obj_trans.squeeze(0).unsqueeze(-1).cpu().numpy()  # [T, 3]
            pred_obj_rot = pred_dict.get("obj_rot", None)      # [1, T, 6] 或 [1, T, 3, 3]

            # --- 结束提取 ---

            if pred_motion is None:
                print("模型输出缺少 'motion'")
            else:
                pred_motion = pred_motion.squeeze(0).cpu() # [T, 132]
                if pred_obj_rot is not None:
                    pred_obj_rot = pred_obj_rot.squeeze(0).cpu() # [T, 6] 或 [T, 3, 3]
                    
                    # 如果 obj_rot 是 6D 表示，转换为旋转矩阵
                    if pred_obj_rot.shape[-1] == 6:
                        pred_obj_rot = transforms.rotation_6d_to_matrix(pred_obj_rot)

                pred_rot_matrices = transforms.rotation_6d_to_matrix(pred_motion.reshape(-1, 22, 6))  # [T, 22, 3, 3]
                pred_root_orient_mat = pred_rot_matrices[:, 0, :, :].to(device)
                pred_pose_body_mat = pred_rot_matrices[:, 1:, :, :].reshape(-1, 21, 3, 3).to(device)

                head_global_trans_start = data['head_global_trans'][0, 0].to(device) # [4, 4]
                head_global_rot_start = head_global_trans_start[:3, :3]
                head_global_pos_start = head_global_trans_start[:3, 3]

                pred_root_pos = pred_root_pos + head_global_pos_start if pred_root_pos is not None else gt_root_pos
                pred_root_orient_mat = head_global_rot_start @ pred_root_orient_mat
                pred_root_orient_axis = transforms.matrix_to_axis_angle(pred_root_orient_mat) # [T, 3]
                # pred_root_orient_axis = transforms.matrix_to_axis_angle(pred_root_orient_mat).reshape(pred_motion.shape[0], -1) # [T, 3]
                pred_pose_body_axis = transforms.matrix_to_axis_angle(pred_pose_body_mat.reshape(-1, 3, 3)).reshape(pred_motion.shape[0], -1) # [T, 63]

                # --- DEBUG WATCH BEGIN ---
                debug_pred_smplh_input = {
                    'root_orient': pred_root_orient_axis.detach().cpu().numpy(),
                    'pose_body': pred_pose_body_axis.detach().cpu().numpy()
                }
                debug_gt_smplh_input = {
                    'root_orient': gt_root_orient_axis.detach().cpu().numpy(),
                    'pose_body': gt_pose_body_axis.detach().cpu().numpy()
                }
                # --- DEBUG WATCH END ---

                pred_smplh_input = {
                    'root_orient': pred_root_orient_axis,
                    'pose_body': pred_pose_body_axis,
                    'trans': pred_root_pos.squeeze(0).to(device)  # 使用真值的 root_pos
                }
                body_pose_pred = body_model(**pred_smplh_input)
                verts_pred = body_pose_pred.v.detach().cpu()
                # --- 结束预测 SMPLH 输入准备 ---
                has_pred = True # 只有在成功生成顶点后才设置为 True

        except Exception as e:
            print(f"模型预测失败: {e}")
            import traceback
            traceback.print_exc()
            # 不设置 has_pred = True
    
    # --- 可视化添加 ---
    # 如果有真值，添加到场景
    if has_gt:
        # 添加真值人体网格 (不偏移, Y轴朝上)
        verts_gt_yup = torch.matmul(verts_gt, R_yup.T.cpu())
        gt_mesh = Meshes(
                    verts_gt_yup.numpy(), # 使用旋转后的 GT verts
                    faces_gt,
                    is_selectable=False,
                    gui_affine=False,
                    name="GT-Human",
                    color=(0.1, 0.8, 0.3, 0.8)  # 绿色
                )
        v.scene.add(gt_mesh)
    
    # 如果有预测结果，添加到场景
    if has_pred and verts_pred is not None:
        # 添加预测的人体网格 (偏移, Y轴朝上)
        # verts_pred_shifted = verts_pred + torch.tensor([1.0, 0, 0])
        verts_pred_yup = torch.matmul(verts_pred, R_yup.T.cpu())
        body_mesh = Meshes(
                    verts_pred_yup.numpy(),
                    faces_gt, # 使用 GT 的 faces
                    is_selectable=False,
                    gui_affine=False,
                    name="Pred-Human (Shifted)",
                    color=(0.9, 0.2, 0.2, 0.8)  # 红色
                )
        v.scene.add(body_mesh)
    
    # 可视化物体
    if show_objects and data.get('has_object', [False])[0]:
        # --- 使用来自 'data' 的未缩放 trans 和单独的 scale ---
        gt_obj_trans = data['obj_trans'][0].cpu().numpy() # 未缩放
        gt_obj_rot = data['obj_rot'][0].cpu().numpy()
        gt_obj_scale = data['obj_scale'][0].cpu().numpy() # 单独的 scale
        obj_name = data.get('obj_name', [None])[0]
        gt_obj_bottom_trans = data.get('obj_bottom_trans', [None])[0] # 假设已缩放 (来自 process_loaded_data)
        gt_obj_bottom_rot = data.get('obj_bottom_rot', [None])[0]

        # 转换 bottom parts 为 numpy
        gt_obj_bottom_trans_np = gt_obj_bottom_trans.cpu().numpy() if torch.is_tensor(gt_obj_bottom_trans) else gt_obj_bottom_trans
        gt_obj_bottom_rot_np = gt_obj_bottom_rot.cpu().numpy() if torch.is_tensor(gt_obj_bottom_rot) else gt_obj_bottom_rot
        # --- 结束数据提取 ---

        print(f"物体名称: {obj_name}")
        
        # --- 加载真值物体几何体 (传递 scale) ---
        gt_obj_verts, obj_faces = load_object_geometry(
            obj_name,
            gt_obj_rot,
            gt_obj_trans, # 未缩放
            obj_scale=gt_obj_scale, # 传递 scale
            obj_bottom_trans=gt_obj_bottom_trans_np, 
            obj_bottom_rot=gt_obj_bottom_rot_np
        )
        
        # --- 添加真值物体网格 (应用 Y-up) ---
        if gt_obj_verts.numel() > 3:
            gt_obj_verts_yup = torch.matmul(gt_obj_verts.cpu(), R_yup.T.cpu())
            gt_obj_mesh = Meshes(
                gt_obj_verts_yup.numpy(),
                obj_faces,
                is_selectable=False,
                gui_affine=False,
                name=f"GT-{obj_name}",
                color=(0.1, 0.8, 0.3, 0.8)  # 绿色
            )
            v.scene.add(gt_obj_mesh)
        
        # --- 添加预测物体网格 ---
        if has_pred and pred_obj_rot is not None:
            # 使用 GT trans (未缩放), GT scale, Pred rot
            pred_obj_verts, _ = load_object_geometry(
                obj_name,
                gt_obj_rot, # 使用真值的旋转
                pred_obj_trans,         # 使用GT平移 (未缩放)
                obj_scale=gt_obj_scale, # 传递 GT scale
                obj_bottom_trans=gt_obj_bottom_trans_np,
                obj_bottom_rot=gt_obj_bottom_rot_np
            )
            
            # --- 添加预测物体网格 (应用偏移和 Y-up) ---
            if pred_obj_verts.numel() > 3:
                # pred_obj_verts_shifted = pred_obj_verts + torch.tensor([1.0, 0, 0])
                pred_obj_verts_yup = torch.matmul(pred_obj_verts.cpu(), R_yup.T.cpu())
                pred_obj_mesh = Meshes(
                    pred_obj_verts_yup.numpy(),
                    obj_faces,
                    is_selectable=False,
                    gui_affine=False,
                    name=f"Pred-{obj_name}",
                    color=(0.9, 0.2, 0.2, 0.8)  # 红色
                )
                v.scene.add(pred_obj_mesh)
    
    # 运行可视化
    v.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="人体与物体姿态可视化工具")
    parser.add_argument('--seq_path', type=str, default='processed_data_0408/test/0.pt',    # processed_data_0415/test/seq_8500.pt
                        help='序列文件路径，如果不指定则从测试目录随机选择')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型路径，默认使用config中设置的路径')
    parser.add_argument('--no_objects', action='store_true', 
                        help='不显示物体')
    parser.add_argument('--use_transpose_model', action='store_true', 
                        help='使用TransPose模型类型')
    parser.add_argument('--use_transpose_humanOnly_model', action='store_true', 
                        help='使用TransPose人体姿态模型类型')
    parser.add_argument('--use_diffusion_model', action='store_true', 
                        help='使用Diffusion模型类型')
    # Add argument for the diffusion config to load the model correctly
    parser.add_argument('--config', type=str, default='configs/TransPose_train.yaml',
                        help='模型配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    try:
        config_data = load_config(args.config)
        config = edict(config_data)
    except Exception as e:
        print(f"错误：无法加载配置文件 {args.config}: {e}")
        exit()

    # --- 数据加载和处理 ---
    seq_path = args.seq_path
    data = None
    if seq_path:
        print(f"正在可视化: {seq_path}")
        try:
            raw_data = torch.load(seq_path)
            # --- 处理数据 (现在 process_loaded_data 分离了 scale 和 trans) ---
            processed_data = process_loaded_data(raw_data, normalize=config.test.normalize)
            if not processed_data:
                 print("错误：处理加载的数据失败。")
                 exit()
                 
            # # --- 添加序列截断逻辑 ---
            # # 获取配置中的目标序列长度
            # target_length = config.test.window
            # print(f"将序列截断到长度: {target_length}")
            
            # # 获取当前序列长度
            # current_length = processed_data["motion"].shape[0]
            
            # # 如果当前序列长度超过目标长度，则截断所有相关数据
            # if current_length > target_length:
            #     for key in processed_data:
            #         if isinstance(processed_data[key], torch.Tensor) and processed_data[key].dim() >= 1:
            #             # 沿着第一个维度（时间维度）截断
            #             processed_data[key] = processed_data[key][:target_length]
            #     print(f"序列已从 {current_length} 帧截断到 {target_length} 帧")
            # elif current_length < target_length:
            #     print(f"警告: 序列长度 ({current_length}) 小于窗长 ({target_length})，程序退出")
            #     exit()
            # # --- 结束序列截断逻辑 ---

            # 转换为批次格式 (添加批次维度)
            batch_data = {}
            for key, value in processed_data.items():
                if isinstance(value, torch.Tensor):
                    batch_data[key] = value.unsqueeze(0)
                else:
                    batch_data[key] = [value] # 将非张量值放入列表中以模拟批次
            data = batch_data # 使用处理和批处理后的数据
            
        except Exception as e:
            print(f"加载或处理文件 {seq_path} 失败: {e}")
            import traceback
            traceback.print_exc()
            exit()
    else:
        # --- 如果没有提供 seq_path，则使用 DataLoader (旧逻辑，可能需要调整) ---
        print("未提供 seq_path，将尝试从测试目录加载第一个样本 (需要 IMUDataset)")
        # 重新启用 DataLoader (如果需要此功能)
        # test_dir = cfg.test.data_path 
        # test_dataset = IMUDataset( ... ) # 需要定义 IMUDataset 或导入
        # test_loader = DataLoader( ... )
        # data = next(iter(test_loader)) 
        print("错误：当前版本不支持从目录加载，请提供 --seq_path。")
        exit()
        # --- 结束旧逻辑 ---

    model = None
    model_path = args.model_path
    if model_path is None and 'model_path' in config: # 否则尝试从配置中获取
        model_path = config.model_path

    if model_path: # 只有在确定了 model_path 后才加载
        try:
            # 使用截断后的序列长度初始化模型
            seq_len = data['human_imu'].shape[1]
            
            # 检查模型类型并加载相应的模型
            if args.use_transpose_model:
                # 加载 TransPose 模型
                print(f"Loading TransPose model from: {model_path}")
                model = load_transpose_model(config, model_path)
                model = model.to(device)
            elif args.use_transpose_humanOnly_model:
                # 加载 TransPose 模型
                print(f"Loading TransPose model from: {model_path}")
                model = load_transpose_model_humanOnly(config, model_path)
                model = model.to(device)
            elif args.use_diffusion_model:
                # 加载 DiT 模型
                print(f"Loading DiT model from: {model_path}")
                model = MotionDiffusion(config, input_length=seq_len, imu_input=True)
                
                # 加载预训练权重
                checkpoint = torch.load(model_path, map_location=device)
                
                state_dict = checkpoint['model_state_dict']
                if list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                model = model.to(device)
            
            model.eval()
            print(f"成功加载模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            model = None # 确保加载失败时 model 为 None
    else:
        print("未提供模型路径，将仅显示真值数据。")

    # 传递配置给可视化函数
    visualize_human_and_objects(args, model, data, not args.no_objects) 