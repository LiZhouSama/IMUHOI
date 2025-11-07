r"""
    Test the system with an example IMU measurement sequence.
"""

import sys
import os
# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加上上层目录到系统路径
root_dir = os.path.abspath(os.path.join(current_dir, "../../tasks/EgoIMU"))
# root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)
print(root_dir) 

import torch
from net import TransPoseNet
from config import paths
from utils import normalize_and_concat
import os
import articulate as art
from human_body_prior.body_model.body_model import BodyModel
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.viewer import Viewer
import pytorch3d.transforms as transforms
import trimesh


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# net = TransPoseNet().to(device)
# acc = torch.load(os.path.join(paths.example_dir, 'acc.pt'))
# ori = torch.load(os.path.join(paths.example_dir, 'ori.pt'))
# x = normalize_and_concat(acc, ori).to(device)
# pose, tran = net.forward_offline(x)     # offline
# # pose, tran = [torch.stack(_) for _ in zip(*[net.forward_online(f) for f in x])]   # online
# art.ParametricModel(paths.smpl_file).view_motion([pose], [tran])


# --- 定义 Z-up 到 Y-up 的旋转矩阵 ---
R_yup = torch.tensor([[1.0, 0.0, 0.0], 
                    [0.0, 0.0, 1.0], 
                    [0.0, -1.0, 0.0]], dtype=torch.float32) # -90度绕X轴
# --- 结束定义 ---

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = TransPoseNet(20, 5).to(device)
    data = torch.load(os.path.join(paths.omomo_dir, 'test.pt'))
    acc = data['acc'][0]
    ori = data['ori'][0]
    pose = data['pose'][0].to(device)
    tran = data['tran'][0].to(device)
    x = normalize_and_concat(acc, ori).to(device)
    pred_pose, pred_tran = net.forward_offline(x)

    body_model = BodyModel(bm_fname=paths.bm_fname_male, num_betas=16, model_type='smplh').to(device)
    gt_input = {
            'root_orient': pose[:, 0, :],
            'pose_body': pose[:, 1:22, :].reshape(pose.shape[0], -1),
            'trans': tran
        }
    body_pose_gt = body_model(**gt_input)
    verts_gt = body_pose_gt.v.detach().cpu()
    faces_gt = body_model.f.detach().cpu().numpy() if isinstance(body_model.f, torch.Tensor) else body_model.f
    verts_gt_yup = torch.matmul(verts_gt, R_yup.T.cpu())
    gt_mesh = Meshes(
                    verts_gt_yup.numpy(), # 使用旋转后的 GT verts
                    faces_gt,
                    is_selectable=False,
                    gui_affine=False,
                    name="GT-Human",
                    color=(0.1, 0.8, 0.3, 0.8)  # 绿色
                )
    pred_input = {
            'root_orient': transforms.matrix_to_axis_angle(pred_pose[:, 0, :, :]).to(device),
            'pose_body': transforms.matrix_to_axis_angle(pred_pose[:, 1:22, :, :]).reshape(pred_pose.shape[0], -1).to(device),
            'trans': tran
        }
    body_pose_pred = body_model(**pred_input)
    verts_pred = body_pose_pred.v.detach().cpu()
    faces_pred = body_model.f.detach().cpu().numpy() if isinstance(body_model.f, torch.Tensor) else body_model.f
    verts_pred_yup = torch.matmul(verts_pred, R_yup.T.cpu())
    pred_mesh = Meshes(
                    verts_pred_yup.numpy(), # 使用旋转后的 GT verts
                    faces_pred,
                    is_selectable=False,
                    gui_affine=False,
                    name="Pred-Human",
                    color=(0.1, 0.1, 0.8, 0.8)  # 蓝色
                )
    v = Viewer(fps=1) # Viewer 初始化
    v.scene.add(gt_mesh)
    v.scene.add(pred_mesh)
    v.run()
