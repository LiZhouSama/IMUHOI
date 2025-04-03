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
from torch.utils.data import DataLoader
from dataloader.dataloader import IMUDataset

# 加载配置
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config('configs/vis.yaml')

# 加载SMPLH模型
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
body_model = BodyModel(bm_fname=config['smplh']['model_path'], num_betas=16).to(device)

def find_pt_files(directory):
    """查找目录下所有.pt文件"""
    pt_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pt'):
                pt_files.append(os.path.join(root, file))
    return pt_files

def apply_transformation_to_obj_geometry(obj_mesh_path, obj_scale, obj_rot, obj_trans):
    """
    应用变换到物体网格
    
    参数:
        obj_mesh_path: 物体网格路径
        obj_scale: 缩放因子 [T]
        obj_rot: 旋转矩阵 [T, 3, 3]
        obj_trans: 平移向量 [T, 3]
        
    返回:
        transformed_obj_verts: 变换后的顶点 [T, Nv, 3]
        obj_mesh_faces: 物体网格的面 [Nf, 3]
    """
    mesh = trimesh.load_mesh(obj_mesh_path)
    obj_mesh_verts = np.asarray(mesh.vertices) # Nv X 3
    obj_mesh_faces = np.asarray(mesh.faces) # Nf X 3 

    ori_obj_verts = torch.from_numpy(obj_mesh_verts).float()[None].repeat(obj_trans.shape[0], 1, 1) # T X Nv X 3 
    
    seq_scale = torch.tensor(obj_scale).float() # T 
    seq_rot_mat = torch.tensor(obj_rot).float() # T X 3 X 3 
    seq_trans = torch.tensor(obj_trans).float()[:, :, None] # T X 3 X 1
    
    transformed_obj_verts = seq_scale.unsqueeze(-1).unsqueeze(-1) * \
    seq_rot_mat.bmm(ori_obj_verts.transpose(1, 2)) + seq_trans
    transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X Nv X 3 

    return transformed_obj_verts, obj_mesh_faces

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

def load_object_geometry(obj_name, obj_scale, obj_rot, obj_trans, obj_bottom_scale=None, obj_bottom_trans=None, obj_bottom_rot=None):
    """
    加载物体几何体并应用变换
    
    参数:
        obj_name: 物体名称
        obj_scale: 缩放因子
        obj_rot: 旋转矩阵
        obj_trans: 平移向量
        obj_bottom_scale: 底部缩放因子（对于有两部分的物体）
        obj_bottom_rot: 底部旋转矩阵（对于有两部分的物体）
        obj_bottom_trans: 底部平移向量（对于有两部分的物体）
        
    返回:
        obj_mesh_verts: 变换后的物体顶点
        obj_mesh_faces: 物体网格面
    """
    obj_geo_root_folder = os.path.join('./dataset/captured_objects')
    obj_mesh_path = os.path.join(obj_geo_root_folder, f"{obj_name}_cleaned_simplified.obj")
    
    # 检查是否是有两部分的物体（例如vacuum或mop）
    two_parts = obj_name in ["vacuum", "mop"]
    
    if two_parts:
        top_obj_mesh_path = os.path.join(obj_geo_root_folder, f"{obj_name}_cleaned_simplified_top.obj")
        bottom_obj_mesh_path = os.path.join(obj_geo_root_folder, f"{obj_name}_cleaned_simplified_bottom.obj")

        top_obj_mesh_verts, top_obj_mesh_faces = apply_transformation_to_obj_geometry(top_obj_mesh_path, 
                                                                                    obj_scale, 
                                                                                    obj_rot, 
                                                                                    obj_trans)
        bottom_obj_mesh_verts, bottom_obj_mesh_faces = apply_transformation_to_obj_geometry(bottom_obj_mesh_path, 
                                                                                         obj_bottom_scale, 
                                                                                         obj_bottom_rot, 
                                                                                         obj_bottom_trans)

        obj_mesh_verts, obj_mesh_faces = merge_two_parts([top_obj_mesh_verts, bottom_obj_mesh_verts], 
                                                        [top_obj_mesh_faces, bottom_obj_mesh_faces])
    else:
        obj_mesh_verts, obj_mesh_faces = apply_transformation_to_obj_geometry(obj_mesh_path, 
                                                                          obj_scale, 
                                                                          obj_rot, 
                                                                          obj_trans)

    return obj_mesh_verts, obj_mesh_faces

def visualize_human_and_objects(model=None, data_path=None, show_objects=True):
    """
    可视化人体和物体的真值和预测值
    
    参数:
        model: EgoIMU模型实例，如果为None则从config.model_path加载
        data_path: 数据文件路径，如果为None则从测试目录随机选择
        show_objects: 是否显示物体
    """
    # 获取数据文件
    if data_path is None:
        test_dir = os.path.join(config['work_dir'], 'test')
        test_files = find_pt_files(test_dir)
        if not test_files:
            print(f"错误：在 {test_dir} 中未找到任何 .pt 文件")
            return
        data_path = random.choice(test_files)
    
    print(f"正在可视化: {data_path}")
    
    # 使用DataLoader加载数据
    data_dir = os.path.dirname(data_path)
    
    # 创建测试数据集和DataLoader
    test_dataset = IMUDataset(
        data_dir=data_dir,
        window_size=120,  # 使用与训练时相同的窗口大小
        window_stride=60,  # 使用更大的步长来减少重叠
        normalize=True,
        normalize_style="first_frame",
        debug=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 一次处理一个样本用于可视化
        shuffle=False
    )
    
    # 获取数据样本
    for batch in test_loader:
        if os.path.basename(data_path) in batch.get('seq_name', ''):
            data = batch
            break
    else:
        # 如果没有找到指定文件，使用第一个样本
        data = next(iter(test_loader))
    
    # 创建可视化实例
    v = Viewer()
    
    # 处理真值数据
    has_gt = False
    
    # 从数据中提取SMPLH参数
    if 'motion' in data:
        has_gt = True
        motion = data['motion'][0].to(device)  # [T, 132]
        
        # 分解motion为位置和旋转
        rot_matrices = transforms.rotation_6d_to_matrix(motion.reshape(-1, 22, 6))  # [T, 22, 3, 3]
        
        # 使用根关节位置作为平移
        root_pos = data['root_pos'][0].to(device)  # [T, 3]
            
        # 创建用于SMPLH的输入字典
        smplh_input = {
            'pose_body': rot_matrices[:, 1:, :, :].reshape(-1, 21, 3, 3).to(device),  # 移除根关节
            'root_orient': rot_matrices[:, 0, :, :].to(device),
            'trans': root_pos  # 使用根关节位置作为平移
        }
        
        # 使用SMPLH模型生成人体网格
        body_pose_gt = body_model(**smplh_input)
        verts_gt = body_pose_gt.v.detach().cpu()
        faces_gt = body_model.f
    
    # 处理预测数据
    has_pred = False
    
    if model is not None:
        # 如果提供了模型，使用模型进行预测
        try:
            # 提取IMU数据作为输入
            human_imu = data['imu'][0].to(device)  # [T, num_imus, 6]
            
            # 提取物体IMU数据（如果有）
            obj_imu = None
            if 'obj_imu' in data:
                obj_imu = data['obj_imu'][0].to(device)  # [T, 6]
            else:
                # 创建空的物体IMU数据
                obj_imu = torch.zeros((human_imu.shape[0], 6), device=device)
            
            # 准备模型输入
            model_input = {
                "human_imu": human_imu.unsqueeze(0),  # [1, T, num_imus, 6]
                "obj_imu": obj_imu.unsqueeze(0)  # [1, T, 6]
            }
            
            # 如果有BPS特征，添加到输入中
            if 'bps_features' in data:
                model_input['bps_features'] = data['bps_features'][0].unsqueeze(0).to(device)
            
            # 运行模型推理
            with torch.no_grad():
                pred = model.diffusion_reverse(model_input)
            
            # 从预测结果中提取姿态和物体信息
            root_pos_pred = pred["root_pos"].squeeze(0).cpu()  # [T, 3]
            motion_pred = pred["motion"].squeeze(0).cpu()  # [T, 132]
            obj_trans_pred = pred["obj_trans"].squeeze(0).cpu() if "obj_trans" in pred else None
            obj_rot_pred = pred["obj_rot"].squeeze(0).cpu() if "obj_rot" in pred else None
            
            # 分解预测的motion为位置和旋转
            rot_matrices_pred = transforms.rotation_6d_to_matrix(motion_pred.reshape(-1, 22, 6))  # [T, 22, 3, 3]
            
            # 创建用于SMPLH的输入字典
            smplh_input_pred = {
                'pose_body': rot_matrices_pred[:, 1:, :, :].reshape(-1, 21, 3, 3).to(device),
                'root_orient': rot_matrices_pred[:, 0, :, :].to(device),
                'trans': root_pos_pred.to(device)  # 使用预测的根关节位置
            }
            
            if 'betas' in smplh_input:
                smplh_input_pred['betas'] = smplh_input['betas']
            
            body_pose_pred = body_model(**smplh_input_pred)
            verts_pred = body_pose_pred.v.detach().cpu()
            has_pred = True
        except Exception as e:
            print(f"模型预测失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 如果没有模型预测但数据中包含预测结果
    elif 'predictions' in data:
        try:
            pred_data = data['predictions']
            
            if isinstance(pred_data, dict) and 'motion' in pred_data:
                root_pos_pred = pred_data.get('root_pos', None)
                motion_pred = pred_data['motion']
                
                # 分解预测的motion为位置和旋转
                rot_matrices_pred = transforms.rotation_6d_to_matrix(motion_pred.reshape(-1, 22, 6))  # [T, 22, 3, 3]
                
                # 创建用于SMPLH的输入字典
                smplh_input_pred = {
                    'pose_body': rot_matrices_pred[:, 1:, :, :].reshape(-1, 21, 3, 3).to(device),
                    'root_orient': rot_matrices_pred[:, 0, :, :].to(device),
                    'trans': root_pos_pred.to(device)  # 使用预测的根关节位置
                }
                
                if 'betas' in smplh_input:
                    smplh_input_pred['betas'] = smplh_input['betas']
                
                body_pose_pred = body_model(**smplh_input_pred)
                verts_pred = body_pose_pred.v.detach().cpu()
                has_pred = True
        except Exception as e:
            print(f"处理预测数据失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 如果有真值，添加到场景
    if has_gt:
        # 添加真值人体网格（偏移1米，以便区分）
        verts_gt_shifted = verts_gt + torch.tensor([1.0, 0, 0])
        gt_mesh = Meshes(
                    verts_gt_shifted.numpy(),
                    faces_gt,
                    is_selectable=False,
                    gui_affine=False,
                    name="真值人体姿态",
                    color=(0.9, 0.2, 0.2, 0.8)  # 红色
                )
        v.scene.add(gt_mesh)
    
    # 如果有预测结果，添加到场景
    if has_pred:
        # 添加预测的人体网格
        body_mesh = Meshes(
                    verts_pred.numpy(),
                    body_model.f,
                    is_selectable=False,
                    gui_affine=False,
                    name="预测人体姿态",
                    color=(0.1, 0.6, 0.9, 0.8)  # 蓝色
                )
        v.scene.add(body_mesh)
    
    # 可视化物体（使用真实3D模型而不是点云）
    if show_objects and 'obj_trans' in data:
        obj_trans = data['obj_trans'][0].cpu().numpy()
        obj_rot = data['obj_rot'][0].cpu().numpy()
        
        # 获取物体名称（从文件名中提取）
        obj_name = None
        if 'seq_name' in data:
            seq_name = data['seq_name'][0]
            for obj in ['vacuum', 'mop', 'largebox', 'smallbox', 'trashcan', 'tripod', 'woodchair',
                       'clothesstand', 'floorlamp', 'largetable', 'monitor', 'plasticbox', 
                       'smalltable', 'suitcase', 'whitechair']:
                if obj in seq_name:
                    obj_name = obj
                    break
        
        if obj_name is None:
            obj_name = 'smallbox'  # 默认物体
            
        # 获取物体缩放因子
        obj_scale = data.get('obj_scale', torch.ones(obj_trans.shape[0]))[0].cpu().numpy()
        
        # 处理两部分物体（如vacuum或mop）
        if obj_name in ['vacuum', 'mop'] and 'obj_bottom_trans' in data:
            obj_bottom_trans = data['obj_bottom_trans'][0].cpu().numpy()
            obj_bottom_rot = data['obj_bottom_rot'][0].cpu().numpy() 
            obj_bottom_scale = data.get('obj_bottom_scale', torch.ones(obj_bottom_trans.shape[0]))[0].cpu().numpy()
        else:
            obj_bottom_trans = obj_trans
            obj_bottom_rot = obj_rot
            obj_bottom_scale = obj_scale
        
        # 加载物体几何体
        obj_verts, obj_faces = load_object_geometry(
            obj_name, 
            obj_scale, 
            obj_rot, 
            obj_trans,
            obj_bottom_scale,
            obj_bottom_trans,
            obj_bottom_rot
        )
        
        # 添加真值物体网格（偏移1米，以便区分）
        gt_obj_verts = obj_verts + torch.tensor([1.0, 0, 0])
        gt_obj_mesh = Meshes(
            gt_obj_verts.numpy(),
            obj_faces,
            is_selectable=False,
            gui_affine=False,
            name=f"真值物体-{obj_name}",
            color=(0.9, 0.6, 0.1, 0.8)  # 橙色
        )
        v.scene.add(gt_obj_mesh)
        
        # 添加预测物体网格（如果有预测结果）
        if has_pred and 'obj_trans' in pred and 'obj_rot' in pred:
            obj_trans_pred = pred['obj_trans'].squeeze(0).cpu().numpy()
            obj_rot_pred = pred['obj_rot'].squeeze(0).cpu().numpy()
            
            # 使用相同的缩放因子
            pred_obj_verts, _ = load_object_geometry(
                obj_name, 
                obj_scale, 
                obj_rot_pred, 
                obj_trans_pred,
                obj_bottom_scale,
                obj_bottom_trans,
                obj_bottom_rot
            )
            
            pred_obj_mesh = Meshes(
                pred_obj_verts.numpy(),
                obj_faces,
                is_selectable=False,
                gui_affine=False,
                name=f"预测物体-{obj_name}",
                color=(0.1, 0.8, 0.3, 0.8)  # 绿色
            )
            v.scene.add(pred_obj_mesh)
    
    # 运行可视化
    v.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="人体与物体姿态可视化工具")
    parser.add_argument('--data_path', type=str, default=None, 
                        help='数据文件路径，如果不指定则从测试目录随机选择')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型路径，默认使用config中设置的路径')
    parser.add_argument('--no_objects', action='store_true', 
                        help='不显示物体')
    parser.add_argument('--config', type=str, default='configs/vis.yaml',
                        help='配置文件路径')
    args = parser.parse_args()
    
    # 重新加载配置（如果指定了不同的配置文件）
    if args.config != 'configs/vis.yaml':
        config = load_config(args.config)
    # 加载模型（如果指定了模型路径）
    model = None
    model_path = args.model_path if args.model_path else config['model_path']
    if model_path:
        try:
            from diffusion_stage.wrap_model import MotionDiffusion
            # 假设我们可以从配置中获取模型参数
            diffusion_cfg = load_config('configs/diffusion.yaml')
            diffusion_cfg = edict(diffusion_cfg)
            model = MotionDiffusion(diffusion_cfg, diffusion_cfg.train.window, 
                                    diffusion_cfg.model.num_layers, imu_input=True)
            
            # 加载预训练权重
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(device)
            model.eval()
            print(f"成功加载模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            model = None
    
    visualize_human_and_objects(model, args.data_path, not args.no_objects) 