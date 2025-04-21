import torch
import os
import numpy as np
import random
import argparse
import yaml
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.viewer import Viewer
from aitviewer.scene.camera import Camera
from moderngl_window.context.base import KeyModifiers
from moderngl_window.context.base import keys
import pytorch3d.transforms as transforms
import trimesh

from human_body_prior.body_model.body_model import BodyModel
from easydict import EasyDict as edict

from torch.utils.data import DataLoader
from dataloader.dataloader import IMUDataset # 从 eval.py 引入

# 导入模型相关 - 根据需要选择正确的模型加载方式
# from models.DiT_model import MotionDiffusion # 如果要用 DiT
from models.TransPose_net import TransPoseNet # 明确使用 TransPose
from models.do_train_imu_TransPose import load_transpose_model # 或者使用这个加载函数


# --- 定义 Z-up 到 Y-up 的旋转矩阵 ---
# R_yup = torch.tensor([[1.0, 0.0, 0.0],
#                       [0.0, 0.0, 1.0],
#                       [0.0, -1.0, 0.0]], dtype=torch.float32)

R_yup = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)

# === 辅助函数 (来自 eval.py 和 vis.py) ===

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = edict(config)
    return config

def load_smpl_model(smpl_model_path, device):
    """加载 SMPL 模型 using human_body_prior"""
    print(f"Loading SMPL model from: {smpl_model_path}")
    if not os.path.exists(smpl_model_path):
        print(f"Error: SMPL model path not found: {smpl_model_path}")
        raise FileNotFoundError(f"SMPL model not found at {smpl_model_path}")
    smpl_model = BodyModel(
        bm_fname=smpl_model_path,
        num_betas=16,
        model_type='smplh' # 明确使用 smplh
    ).to(device)
    return smpl_model

def apply_transformation_to_obj_geometry(obj_mesh_path, obj_rot, obj_trans, scale=None, device='cpu'):
    """
    应用变换到物体网格 (OMOMO 方式)

    参数:
        obj_mesh_path: 物体网格路径
        obj_rot: 旋转矩阵 [T, 3, 3] (torch tensor on device)
        obj_trans: 平移向量 [T, 3] (torch tensor on device, 不含缩放)
        scale: 缩放因子 [T] 或 [T, 1] 或 [T, 1, 1] (torch tensor on device)

    返回:
        transformed_obj_verts: 变换后的顶点 [T, Nv, 3] (torch tensor on device)
        obj_mesh_faces: 物体网格的面 [Nf, 3] (numpy array)
    """
    try:
        mesh = trimesh.load_mesh(obj_mesh_path)
        obj_mesh_verts = torch.from_numpy(np.asarray(mesh.vertices)).float().to(device) # Nv X 3
        obj_mesh_faces = np.asarray(mesh.faces) # Nf X 3

        T = obj_trans.shape[0]
        ori_obj_verts = obj_mesh_verts[None].repeat(T, 1, 1) # T X Nv X 3

        # --- 首先应用缩放 (如果提供) ---
        if scale is not None:
            scale_tensor = scale.float().to(device)
            # 确保 scale_tensor 可以广播: [T, 1, 1]
            if scale_tensor.dim() == 1: scale_tensor = scale_tensor[:, None, None]
            elif scale_tensor.dim() == 2: scale_tensor = scale_tensor[:, :, None]
            elif scale_tensor.dim() == 3: pass
            else:
                 print(f"警告: scale 维度无法处理 {scale_tensor.shape}, 将不应用缩放。")
                 scale_tensor = None

            if scale_tensor is not None:
                 ori_obj_verts = ori_obj_verts * scale_tensor
        # --- 结束缩放应用 ---

        seq_rot_mat = obj_rot.float().to(device) # T X 3 X 3
        seq_trans = obj_trans.float().to(device) # T X 3 (不含缩放)

        # --- 应用旋转和平移到 (可能) 已缩放的顶点 ---
        # Transpose vertices for matmul: T X 3 X Nv
        verts_rotated = torch.bmm(seq_rot_mat, ori_obj_verts.transpose(1, 2))
        # Add translation (broadcast): T X 3 X Nv + T X 3 X 1 -> T X 3 X Nv
        verts_translated = verts_rotated + seq_trans.unsqueeze(-1)
        # Transpose back: T X Nv X 3
        transformed_obj_verts = verts_translated.transpose(1, 2)

    except Exception as e:
        print(f"应用变换到物体几何体失败 for {obj_mesh_path}: {e}")
        import traceback
        traceback.print_exc()
        # Return dummy data on the correct device
        transformed_obj_verts = torch.zeros((obj_trans.shape[0] if obj_trans is not None else 1, 1, 3), device=device)
        obj_mesh_faces = np.zeros((1, 3), dtype=np.int64)

    return transformed_obj_verts, obj_mesh_faces


def merge_two_parts(verts_list, faces_list, device='cpu'):
    """ 合并两个网格部分 """
    verts_num = 0
    merged_verts_list = []
    merged_faces_list = []
    for p_idx in range(len(verts_list)):
        part_verts = verts_list[p_idx].to(device) # T X Nv X 3
        part_faces = torch.from_numpy(faces_list[p_idx]).long().to(device) # Nf X 3

        merged_verts_list.append(part_verts)
        merged_faces_list.append(part_faces + verts_num)
        verts_num += part_verts.shape[1]

    merged_verts = torch.cat(merged_verts_list, dim=1)
    merged_faces = torch.cat(merged_faces_list, dim=0).cpu().numpy()
    return merged_verts, merged_faces

def load_object_geometry(obj_name, obj_rot, obj_trans, obj_scale=None, obj_bottom_trans=None, obj_bottom_rot=None, obj_geo_root='./dataset/captured_objects', device='cpu'):
    """ 加载物体几何体并应用变换 (OMOMO 方式) """
    if obj_name is None:
        print("警告: 物体名称为 None，无法加载几何体。")
        return torch.zeros((1, 1, 3), device=device), np.zeros((1, 3), dtype=np.int64)

    # Ensure transformations are tensors on the correct device
    obj_rot = torch.as_tensor(obj_rot, dtype=torch.float32, device=device)
    obj_trans = torch.as_tensor(obj_trans, dtype=torch.float32, device=device)
    if obj_scale is not None:
        obj_scale = torch.as_tensor(obj_scale, dtype=torch.float32, device=device)
    if obj_bottom_rot is not None:
        obj_bottom_rot = torch.as_tensor(obj_bottom_rot, dtype=torch.float32, device=device)
    if obj_bottom_trans is not None:
        obj_bottom_trans = torch.as_tensor(obj_bottom_trans, dtype=torch.float32, device=device)


    obj_mesh_path = os.path.join(obj_geo_root, f"{obj_name}_cleaned_simplified.obj")
    two_parts = obj_name in ["vacuum", "mop"] and obj_bottom_trans is not None and obj_bottom_rot is not None

    if two_parts:
        top_obj_mesh_path = os.path.join(obj_geo_root, f"{obj_name}_cleaned_simplified_top.obj")
        bottom_obj_mesh_path = os.path.join(obj_geo_root, f"{obj_name}_cleaned_simplified_bottom.obj")

        if not os.path.exists(top_obj_mesh_path) or not os.path.exists(bottom_obj_mesh_path):
             print(f"警告: 找不到物体 {obj_name} 的两部分几何文件。将尝试加载整体文件。")
             two_parts = False
             obj_mesh_path = os.path.join(obj_geo_root, f"{obj_name}_cleaned_simplified.obj") # Fallback
        else:
            top_obj_mesh_verts, top_obj_mesh_faces = apply_transformation_to_obj_geometry(top_obj_mesh_path, obj_rot, obj_trans, scale=obj_scale, device=device)
            # Assume bottom uses the same scale, pass bottom transforms
            bottom_obj_mesh_verts, bottom_obj_mesh_faces = apply_transformation_to_obj_geometry(bottom_obj_mesh_path, obj_bottom_rot, obj_bottom_trans, scale=obj_scale, device=device)
            obj_mesh_verts, obj_mesh_faces = merge_two_parts([top_obj_mesh_verts, bottom_obj_mesh_verts], [top_obj_mesh_faces, bottom_obj_mesh_faces], device=device)

    if not two_parts:
        if not os.path.exists(obj_mesh_path):
             print(f"警告: 找不到物体几何文件: {obj_mesh_path}")
             return torch.zeros((obj_trans.shape[0] if obj_trans is not None else 1, 1, 3), device=device), np.zeros((1, 3), dtype=np.int64)
        obj_mesh_verts, obj_mesh_faces = apply_transformation_to_obj_geometry(obj_mesh_path, obj_rot, obj_trans, scale=obj_scale, device=device)

    return obj_mesh_verts, obj_mesh_faces


def visualize_batch_data(viewer, batch, model, smpl_model, device, obj_geo_root, show_objects=True):
    """ 在 aitviewer 场景中可视化单个批次的数据 (真值和预测) """
    # --- Revised Clearing Logic (Attempt 4) ---
    try:
        # Use collect_nodes to get all nodes in the scene
        all_nodes = viewer.scene.collect_nodes()
        renderables_to_remove = [
            node for node in all_nodes
            if hasattr(node, 'name') and node.name is not None and (node.name.startswith("GT-") or node.name.startswith("Pred-"))
        ]

        # Try removing nodes using a potential 'destroy' or 'remove' method on the node itself
        removed_count = 0
        for node_to_remove in renderables_to_remove:
            removed = False
            if hasattr(node_to_remove, 'destroy'):
                try:
                    node_to_remove.destroy()
                    removed = True
                    removed_count += 1
                except Exception as e:
                    print(f"Error calling node.destroy() for '{node_to_remove.name}': {e}")
            elif hasattr(node_to_remove, 'remove'): # Less likely but possible
                try:
                    node_to_remove.remove()
                    removed = True
                    removed_count += 1
                except Exception as e:
                    print(f"Error calling node.remove() for '{node_to_remove.name}': {e}")
            # Add more potential removal methods if needed

            # if not removed:
            #     print(f"Warning: Could not find a method to remove node '{node_to_remove.name}'")
        # print(f"Attempted to remove {removed_count} old nodes.")

    except AttributeError as e:
        print(f"Error accessing scene nodes or methods (maybe collect_nodes doesn't exist?): {e}")
    except Exception as e:
        print(f"Error during scene clearing: {e}")
    # --- End Revised Clearing Logic ---

    with torch.no_grad():
        # --- 1. 准备数据 ---
        gt_root_pos = batch["root_pos"].to(device)         # [bs, T, 3]
        gt_motion = batch["motion"].to(device)           # [bs, T, 132]
        human_imu = batch["human_imu"].to(device)        # [bs, T, num_imus, 9/12]
        obj_imu = batch.get("obj_imu", None)             # [bs, T, 1, 9/12] or None
        gt_obj_trans = batch.get("obj_trans", None)      # [bs, T, 3] or None
        gt_obj_rot_6d = batch.get("obj_rot", None)       # [bs, T, 6] or None
        obj_name = batch.get("obj_name", [None])[0]      # 物体名称 (取列表第一个)
        gt_obj_scale = batch.get("obj_scale", None)      # [bs, T] or [bs, T, 1]? Check dataloader
        gt_obj_bottom_trans = batch.get("obj_bottom_trans", None) # [bs, T, 3] or None
        gt_obj_bottom_rot = batch.get("obj_bottom_rot", None)     # [bs, T, 3, 3] or None


        if obj_imu is not None: obj_imu = obj_imu.to(device)
        if gt_obj_trans is not None: gt_obj_trans = gt_obj_trans.to(device)
        if gt_obj_rot_6d is not None: gt_obj_rot_6d = gt_obj_rot_6d.to(device)
        if gt_obj_scale is not None: gt_obj_scale = gt_obj_scale.to(device)
        if gt_obj_bottom_trans is not None: gt_obj_bottom_trans = gt_obj_bottom_trans.to(device)
        if gt_obj_bottom_rot is not None: gt_obj_bottom_rot = gt_obj_bottom_rot.to(device)


        # 仅处理批次中的第一个序列 (bs=0)
        bs = 0
        T = gt_motion.shape[1]
        gt_root_pos_seq = gt_root_pos[bs]           # [T, 3]
        gt_motion_seq = gt_motion[bs]             # [T, 132]

        # --- 2. 获取真值 SMPL ---
        gt_rot_matrices = transforms.rotation_6d_to_matrix(gt_motion_seq.reshape(T, 22, 6)) # [T, 22, 3, 3]
        gt_root_orient_mat = gt_rot_matrices[:, 0]                         # [T, 3, 3]
        gt_pose_body_mat = gt_rot_matrices[:, 1:].reshape(T * 21, 3, 3)    # [T*21, 3, 3]
        gt_root_orient_axis = transforms.matrix_to_axis_angle(gt_root_orient_mat) # [T, 3]
        gt_pose_body_axis = transforms.matrix_to_axis_angle(gt_pose_body_mat).reshape(T, -1) # [T, 63]

        gt_smplh_input = {
            'root_orient': gt_root_orient_axis,
            'pose_body': gt_pose_body_axis,
            'trans': gt_root_pos_seq
        }
        body_pose_gt = smpl_model(**gt_smplh_input)
        verts_gt_seq = body_pose_gt.v                          # [T, Nv, 3]
        faces_gt_np = smpl_model.f.cpu().numpy() if isinstance(smpl_model.f, torch.Tensor) else smpl_model.f

        # --- 3. 模型预测 ---
        pred_motion_seq = None
        pred_obj_rot_6d_seq = None
        pred_obj_trans_seq = None # TransPose 不预测
        pred_root_pos_seq = None # TransPose 不预测

        model_input = {"human_imu": human_imu} # [bs, T, num_imus, dim]
        has_object_data_for_model = obj_imu is not None
        if has_object_data_for_model:
            model_input["obj_imu"] = obj_imu # [bs, T, 1, dim]

        try:
            pred_dict = model(model_input)
            pred_motion = pred_dict.get("motion") # [bs, T, 132]
            pred_obj_rot = pred_dict.get("obj_rot") # [bs, T, 6] (TransPose 输出 6D)

            if pred_motion is not None:
                pred_motion_seq = pred_motion[bs] # [T, 132]
            else:
                print("警告: 模型未输出 'motion'")

            if pred_obj_rot is not None:
                pred_obj_rot_6d_seq = pred_obj_rot[bs] # [T, 6]
            elif has_object_data_for_model:
                print("警告: 模型未输出 'obj_rot'，即使有物体 IMU 输入")

        except Exception as e:
            print(f"模型推理失败: {e}")
            import traceback
            traceback.print_exc()

        # --- 4. 获取预测 SMPL (使用预测 motion + 真值 trans) ---
        verts_pred_seq = None
        if pred_motion_seq is not None:
            pred_rot_matrices = transforms.rotation_6d_to_matrix(pred_motion_seq.reshape(T, 22, 6)) # [T, 22, 3, 3]
            pred_root_orient_mat = pred_rot_matrices[:, 0]                         # [T, 3, 3]
            pred_pose_body_mat = pred_rot_matrices[:, 1:].reshape(T * 21, 3, 3)    # [T*21, 3, 3]
            pred_root_orient_axis = transforms.matrix_to_axis_angle(pred_root_orient_mat) # [T, 3]
            pred_pose_body_axis = transforms.matrix_to_axis_angle(pred_pose_body_mat).reshape(T, -1) # [T, 63]

            # 使用真值 root_pos
            pred_smplh_input = {
                'root_orient': pred_root_orient_axis,
                'pose_body': pred_pose_body_axis,
                'trans': gt_root_pos_seq # <-- 使用真值 trans
            }
            body_pose_pred = smpl_model(**pred_smplh_input)
            verts_pred_seq = body_pose_pred.v # [T, Nv, 3]

        # --- 5. 获取物体几何体 ---
        gt_obj_verts_seq = None
        pred_obj_verts_seq = None
        obj_faces_np = None
        has_object_gt = gt_obj_trans is not None and gt_obj_rot_6d is not None and obj_name is not None

        if show_objects and has_object_gt:
            gt_obj_trans_seq = gt_obj_trans[bs]     # [T, 3]
            gt_obj_rot_6d_seq = gt_obj_rot_6d[bs]   # [T, 6]
            gt_obj_rot_mat_seq = transforms.rotation_6d_to_matrix(gt_obj_rot_6d_seq) # [T, 3, 3]
            gt_obj_scale_seq = gt_obj_scale[bs] if gt_obj_scale is not None else None # [T] or [T, 1]?
            # Handle bottom parts if they exist
            gt_obj_bottom_trans_seq = gt_obj_bottom_trans[bs] if gt_obj_bottom_trans is not None else None
            gt_obj_bottom_rot_seq = gt_obj_bottom_rot[bs] if gt_obj_bottom_rot is not None else None


            # 获取真值物体
            gt_obj_verts_seq, obj_faces_np = load_object_geometry(
                obj_name, gt_obj_rot_mat_seq, gt_obj_trans_seq, gt_obj_scale_seq,
                obj_bottom_trans=gt_obj_bottom_trans_seq,
                obj_bottom_rot=gt_obj_bottom_rot_seq,
                obj_geo_root=obj_geo_root, device=device
            )

            # 获取预测物体 (使用预测 rot + 真值 trans + 真值 scale)
            if pred_obj_rot_6d_seq is not None:
                pred_obj_rot_mat_seq = transforms.rotation_6d_to_matrix(pred_obj_rot_6d_seq) # [T, 3, 3]
                pred_obj_verts_seq, _ = load_object_geometry(
                    obj_name, pred_obj_rot_mat_seq, gt_obj_trans_seq, gt_obj_scale_seq, # <-- 使用真值 trans, scale
                    obj_bottom_trans=gt_obj_bottom_trans_seq, # Use GT bottom parts for consistency
                    obj_bottom_rot=gt_obj_bottom_rot_seq,
                    obj_geo_root=obj_geo_root, device=device
                )
            else:
                print("无预测物体旋转，无法生成预测物体几何体")

        # --- 6. 添加到 aitviewer 场景 ---
        global R_yup # 使用全局定义的 Y-up 旋转

        # 添加真值人体 (绿色, 不偏移)
        if verts_gt_seq is not None:
            verts_gt_yup = torch.matmul(verts_gt_seq, R_yup.T.to(device))
            gt_human_mesh = Meshes(
                verts_gt_yup.cpu().numpy(), faces_gt_np,
                name="GT-Human", color=(0.1, 0.8, 0.3, 0.8), gui_affine=False, is_selectable=False
            )
            viewer.scene.add(gt_human_mesh)

        # 添加预测人体 (红色, 偏移 x=1.0)
        if verts_pred_seq is not None:
            verts_pred_shifted = verts_pred_seq + torch.tensor([1.0, 0, 0], device=device) # 偏移
            verts_pred_yup = torch.matmul(verts_pred_shifted, R_yup.T.to(device))
            pred_human_mesh = Meshes(
                verts_pred_yup.cpu().numpy(), faces_gt_np,
                name="Pred-Human", color=(0.9, 0.2, 0.2, 0.8), gui_affine=False, is_selectable=False
            )
            viewer.scene.add(pred_human_mesh)

        # 添加真值物体 (绿色, 不偏移)
        if gt_obj_verts_seq is not None and obj_faces_np is not None:
            gt_obj_verts_yup = torch.matmul(gt_obj_verts_seq, R_yup.T.to(device))
            gt_obj_mesh = Meshes(
                gt_obj_verts_yup.cpu().numpy(), obj_faces_np,
                name=f"GT-{obj_name}", color=(0.1, 0.8, 0.3, 0.8), gui_affine=False, is_selectable=False
            )
            viewer.scene.add(gt_obj_mesh)

        # 添加预测物体 (红色, 偏移 x=1.0)
        if pred_obj_verts_seq is not None and obj_faces_np is not None:
            pred_obj_verts_shifted = pred_obj_verts_seq + torch.tensor([1.0, 0, 0], device=device) # 偏移
            pred_obj_verts_yup = torch.matmul(pred_obj_verts_shifted, R_yup.T.to(device))
            pred_obj_mesh = Meshes(
                pred_obj_verts_yup.cpu().numpy(), obj_faces_np,
                name=f"Pred-{obj_name}", color=(0.9, 0.2, 0.2, 0.8), gui_affine=False, is_selectable=False
            )
            viewer.scene.add(pred_obj_mesh)


# === 自定义 Viewer 类 ===

class InteractiveViewer(Viewer):
    def __init__(self, data_list, model, smpl_model, config, device, obj_geo_root, show_objects=True, **kwargs):
        super().__init__(**kwargs)
        self.data_list = data_list # 直接使用加载到内存的列表
        self.current_index = 0
        self.model = model
        self.smpl_model = smpl_model
        self.config = config
        self.device = device
        self.show_objects = show_objects
        self.obj_geo_root = obj_geo_root

        # 设置初始相机位置 (可选)
        # self.scene.camera.position = np.array([0.0, 1.0, 3.0])
        # self.scene.camera.target = np.array([0.5, 0.8, 0.0]) # 对准偏移后的中间区域

        # 初始可视化
        self.visualize_current_sequence()

    def visualize_current_sequence(self):
        if not self.data_list:
            print("错误：数据列表为空。")
            return
        if 0 <= self.current_index < len(self.data_list):
            batch = self.data_list[self.current_index]
            print(f"Visualizing sequence index: {self.current_index}")
            try:
                visualize_batch_data(self, batch, self.model, self.smpl_model, self.device, self.obj_geo_root, self.show_objects)
                self.title = f"Sequence Index: {self.current_index}/{len(self.data_list)-1} (q/e to navigate)"
            except Exception as e:
                 print(f"Error visualizing sequence {self.current_index}: {e}")
                 import traceback
                 traceback.print_exc()
                 self.title = f"Error visualizing index: {self.current_index}"
        else:
            print("Index out of bounds.")

    def key_press_event(self, key: keys, scancode: int, mods: KeyModifiers):
        super().key_press_event(key, scancode, mods) # 调用父方法处理基本快捷键

        if key == keys.Q:
            if self.current_index > 0:
                self.current_index -= 1
                self.visualize_current_sequence()
            else:
                print("Already at the first sequence.")
        elif key == keys.E:
            if self.current_index < len(self.data_list) - 1:
                self.current_index += 1
                self.visualize_current_sequence()
            else:
                print("Already at the last sequence.")

# === 主函数 ===

def main():
    parser = argparse.ArgumentParser(description='Interactive EgoMotion Visualization Tool')
    parser.add_argument('--config', type=str, default='configs/TransPose_train.yaml', help='Path to the main configuration file (used for model, dataset params).')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the trained TransPose model checkpoint. Overrides config if provided.')
    parser.add_argument('--smpl_model_path', type=str, default=None, help='Path to the SMPLH model file. Overrides config if provided.')
    parser.add_argument('--test_data_dir', type=str, default=None, help='Path to the test dataset directory. Overrides config if provided.')
    parser.add_argument('--obj_geo_root', type=str, default='./dataset/captured_objects', help='Path to the object geometry root directory.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader (should be 1 for sequential vis).')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers.')
    parser.add_argument('--no_objects', action='store_true', help='Do not load or visualize objects.')
    parser.add_argument('--limit_sequences', type=int, default=None, help='Limit the number of sequences to load for visualization.')
    args = parser.parse_args()

    if args.batch_size != 1:
        print("Warning: Setting batch_size to 1 for interactive visualization.")
        args.batch_size = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Override config with command line args
    if args.model_path: config.model_path = args.model_path
    if args.smpl_model_path: config.bm_path = args.smpl_model_path
    if args.test_data_dir: config.test.data_path = args.test_data_dir
    if args.num_workers is not None: config.num_workers = args.num_workers
    # Ensure test config exists or copy from train
    if 'test' not in config: config.test = config.train.copy()
    config.test.batch_size = args.batch_size # Force batch size 1

    # --- Load SMPL Model ---
    smpl_model_path = config.get('bm_path', 'body_models/smplh/neutral/model.npz')
    smpl_model = load_smpl_model(smpl_model_path, device)

    # --- Load Trained Model ---
    model_path = config.get('model_path', None)
    if not model_path:
        print("Error: No model path provided in config or via --model_path.")
        return
    model = load_transpose_model(config, model_path)
    model = model.to(device)
    model.eval()

    # --- Load Test Dataset ---
    test_data_dir = config.test.get('data_path', None)
    if not test_data_dir or not os.path.exists(test_data_dir):
        print(f"Error: Test dataset path not found or invalid: {test_data_dir}")
        return
    print(f"Loading test dataset from: {test_data_dir}")

    # Use test window size from config, default if not present
    test_window_size = config.test.get('window', config.train.get('window', 60))

    test_dataset = IMUDataset(
        data_dir=test_data_dir,
        window_size=test_window_size,
        window_stride=config.test.get('window_stride', test_window_size), # Use stride from config
        normalize=config.test.get('normalize', True),
        debug=config.get('debug', False)
    )

    if len(test_dataset) == 0:
         print("Error: Test dataset is empty.")
         return

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test.batch_size, # Should be 1
        shuffle=False, # IMPORTANT: Keep order for navigation
        num_workers=config.get('num_workers', 0), # Set workers based on args/config
        pin_memory=True,
        drop_last=False
    )

    print(f"Loading data into memory (limit={args.limit_sequences})...")
    data_list = []
    for i, batch in enumerate(test_loader):
        if args.limit_sequences is not None and i >= args.limit_sequences:
            print(f"Stopped loading after {args.limit_sequences} sequences.")
            break
        data_list.append(batch)
        if i % 50 == 0 and i > 0:
            print(f"  Loaded {i+1} sequences...")
    print(f"Finished loading {len(data_list)} sequences.")

    if not data_list:
        print("Error: No data loaded into the list.")
        return

    # --- Initialize and Run Viewer ---
    print("Initializing Interactive Viewer...")
    viewer_instance = InteractiveViewer(
        data_list=data_list,
        model=model,
        smpl_model=smpl_model,
        config=config,
        device=device,
        obj_geo_root=args.obj_geo_root,
        show_objects=(not args.no_objects),
        window_size=(1920, 1080) # Example window size
        # Add other Viewer kwargs if needed (e.g., fps)
    )
    print("Viewer Initialized. Use 'q' (previous) and 'e' (next) to navigate sequences.")
    print("Other standard aitviewer controls should also work (e.g., mouse drag to rotate, scroll to zoom).")
    viewer_instance.run()


if __name__ == "__main__":
    main() 