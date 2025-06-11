import torch
import os
import numpy as np
import argparse
from aitviewer.renderables.meshes import Meshes
from aitviewer.viewer import Viewer
import pytorch3d.transforms as transforms

from human_body_prior.body_model.body_model import BodyModel

# --- 定义 Z-up 到 Y-up 的旋转矩阵 ---
# R_yup = torch.tensor([[1.0, 0.0, 0.0],
#                       [0.0, 0.0, 1.0],
#                       [0.0, -1.0, 0.0]], dtype=torch.float32)
R_yup = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)

def load_smpl_model(smpl_model_path, device):
    """加载 SMPL 模型"""
    print(f"正在加载 SMPL 模型: {smpl_model_path}")
    if not os.path.exists(smpl_model_path):
        print(f"错误: SMPL 模型路径未找到: {smpl_model_path}")
        raise FileNotFoundError(f"SMPL 模型未找到: {smpl_model_path}")
    
    smpl_model = BodyModel(
        bm_fname=smpl_model_path,
        num_betas=16,
        model_type='smplh'
    ).to(device)
    return smpl_model

def load_prediction_results(pt_file_path):
    """加载MobilePoser预测结果"""
    print(f"正在加载预测结果: {pt_file_path}")
    if not os.path.exists(pt_file_path):
        raise FileNotFoundError(f"预测结果文件未找到: {pt_file_path}")
    
    results = torch.load(pt_file_path, map_location='cpu')
    print(f"加载完成，包含 {results['frame_count']} 帧数据")
    print(f"数据键: {list(results.keys())}")
    
    return results

def generate_all_frames_mesh(pose_data, trans_data, smpl_model, device):
    """生成所有帧的网格数据"""
    total_frames = pose_data.shape[0]
    print(f"正在生成 {total_frames} 帧的网格数据...")
    
    all_verts = []
    
    with torch.no_grad():
        for frame_idx in range(total_frames):
            if frame_idx % 50 == 0:
                print(f"  处理第 {frame_idx+1}/{total_frames} 帧...")
                
            pose_frame = pose_data[frame_idx].to(device)  # [24, 3, 3] 旋转矩阵
            trans_frame = trans_data[frame_idx].to(device)  # [3]
            
            try:
                # 转换旋转矩阵为轴角表示
                pose_axis_angle = transforms.matrix_to_axis_angle(pose_frame)  # [24, 3]
                
                # 分离root orientation和body pose
                root_orient = pose_axis_angle[0]  # [3] - root orientation
                pose_body = pose_axis_angle[1:22].reshape(-1)  # [21*3 = 63] - body pose
                
                # 构建SMPL输入
                smpl_input = {
                    'root_orient': root_orient.unsqueeze(0),  # [1, 3]
                    'pose_body': pose_body.unsqueeze(0),      # [1, 63]
                    'trans': trans_frame.unsqueeze(0)         # [1, 3]
                }
                
                # 通过SMPL模型获取mesh
                body_pose = smpl_model(**smpl_input)
                verts = body_pose.v[0]  # [Nv, 3]
                
                # 应用坐标系转换
                verts_yup = torch.matmul(verts, R_yup.T.to(device))
                all_verts.append(verts_yup.cpu())
                
            except Exception as e:
                print(f"生成第 {frame_idx} 帧SMPL mesh时出错: {e}")
                # 使用前一帧的数据或零数据
                if len(all_verts) > 0:
                    all_verts.append(all_verts[-1])
                else:
                    all_verts.append(torch.zeros(6890, 3))  # SMPL模型默认顶点数
    
    # 堆叠所有帧的顶点数据
    all_verts_tensor = torch.stack(all_verts, dim=0)  # [T, Nv, 3]
    faces = smpl_model.f.cpu().numpy() if isinstance(smpl_model.f, torch.Tensor) else smpl_model.f
    
    print(f"网格数据生成完成，形状: {all_verts_tensor.shape}")
    return all_verts_tensor.numpy(), faces

def visualize_mobileposer_results(prediction_results, smpl_model, device):
    """可视化MobilePoser预测结果"""
    # 提取姿态和平移数据
    pose_data = prediction_results['pose']  # [T, ...]
    trans_data = prediction_results['translation']  # [T, 3]
    total_frames = prediction_results['frame_count']
    fps = prediction_results.get('fps', 30)
    
    print(f"总帧数: {total_frames}")
    print(f"姿态数据形状: {pose_data.shape}")
    print(f"平移数据形状: {trans_data.shape}")
    print(f"FPS: {fps}")
    
    # 生成所有帧的网格数据
    all_verts, faces = generate_all_frames_mesh(pose_data, trans_data, smpl_model, device)
    
    # 创建aitviewer实例
    viewer = Viewer(fps=fps)
    
    # 创建包含所有帧的网格对象
    human_mesh = Meshes(
        all_verts,  # [T, Nv, 3]
        faces,      # [Nf, 3]
        name="MobilePoser-Human",
        color=(0.2, 0.6, 0.8, 0.9),  # 蓝色
        gui_affine=False,
        is_selectable=False
    )
    
    # 添加到场景
    viewer.scene.add(human_mesh)
    
    print("可视化准备完成！")
    print("控制说明:")
    print("  空格: 播放/暂停")
    print("  左/右箭头: 前/后一帧")
    print("  鼠标拖拽: 旋转视角")
    print("  鼠标滚轮: 缩放")
    
    # 运行可视化
    viewer.run()

def main():
    parser = argparse.ArgumentParser(description='MobilePoser预测结果可视化工具')
    parser.add_argument('--results_path', type=str, 
                       default='Ref Codes/MobilePoser/prediction_results.pt',
                       help='预测结果pt文件路径')
    parser.add_argument('--smpl_model_path', type=str,
                       default='body_models/smplh/neutral/model.npz',
                       help='SMPL模型路径')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载预测结果
    try:
        prediction_results = load_prediction_results(args.results_path)
    except Exception as e:
        print(f"加载预测结果失败: {e}")
        return
    
    # 加载SMPL模型
    try:
        smpl_model = load_smpl_model(args.smpl_model_path, device)
    except Exception as e:
        print(f"加载SMPL模型失败: {e}")
        return
    
    # 可视化
    print("开始可视化...")
    visualize_mobileposer_results(prediction_results, smpl_model, device)

if __name__ == "__main__":
    main() 