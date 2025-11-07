import argparse
import os
import random
from typing import Optional, Sequence, Tuple

import articulate as art
import numpy as np
import torch
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.spheres import Spheres
from aitviewer.renderables.lines import Lines
from aitviewer.viewer import Viewer

import utils.config as cfg
from model.dataset_trans_obj import MotionDatasetWithObjectAndTrans
from model.model import Poser

# SMPL 24关节的父节点定义
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize OMOMO samples with Poser predictions versus ground truth "
            "using MotionDatasetWithObjectAndTrans (object/translation data is loaded but ignored)."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["omomo"],
        help="Dataset subset names expected under the data root.",
    )
    parser.add_argument(
        "--data-root",
        default=cfg.work_dir,
        help="Base directory containing subset folders (e.g., train/test).",
    )
    parser.add_argument(
        "--subset",
        default="test",
        help="Subset folder name to load from the data root.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=300,
        help="Window length sampled from each motion sequence.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=6,
        help="Sequence index to visualize; negative values pick a random sequence.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used when picking a random sequence index.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for poser inference (e.g., cuda:0 or cpu).",
    )
    parser.add_argument(
        "--weight",
        default=cfg.weight_s,
        help="Path to the poser weight file.",
    )
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Enable random temporal crops for each access; otherwise deterministic windows are used.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Fallback FPS used when loading raw sequences.",
    )
    parser.add_argument(
        "--trim-frames",
        type=int,
        default=6,
        help="Number of frames trimmed from the start/end during preprocessing.",
    )
    parser.add_argument(
        "--show-skeleton",
        action="store_true",
        help="Display skeleton (joints and bones) in addition to body meshes.",
    )
    parser.add_argument(
        "--joint-radius",
        type=float,
        default=0.02,
        help="Radius of joint spheres in skeleton visualization.",
    )
    parser.add_argument(
        "--bone-radius",
        type=float,
        default=0.01,
        help="Radius of bone lines in skeleton visualization.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"CUDA is not available; falling back to CPU instead of {requested}.")
        device = torch.device("cpu")
    return device

def load_network(weight_path: str, device: torch.device) -> Poser:
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    net = Poser().to(device)
    state_dict = torch.load(weight_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def select_index(num_samples: int, index: int, seed: Optional[int]) -> int:
    if num_samples == 0:
        raise RuntimeError("No samples available in the dataset.")
    if index >= 0:
        if index >= num_samples:
            raise IndexError(f"Requested index {index} but dataset only has {num_samples} samples.")
        return index
    if seed is not None:
        random.seed(seed)
    return random.randrange(num_samples)


def prepare_ground_truth(
    net: Poser, imu_sequence: torch.Tensor, pose_sequence: torch.Tensor
) -> torch.Tensor:
    imu_cpu = imu_sequence.detach().cpu()
    pose_cpu = pose_sequence.detach().cpu()
    pose_cpu = pose_cpu[:, [4, 5, 6, 7, 8, 9, 10, 0, 2, 1, 3]]
    orientation = imu_cpu[:, :, :9].contiguous().view(-1, 6, 3, 3)
    glb_gt_xsens = net._reduced_glb_6d_to_full_glb_mat_xsens(pose_cpu, orientation)
    glb_gt_smpl = net._glb_mat_xsens_to_glb_mat_smpl(glb_gt_xsens)
    return glb_gt_smpl


def build_skeleton(
    joints: torch.Tensor,
    name: str,
    color: Tuple[float, float, float, float],
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    joint_radius: float = 0.02,
    bone_radius: float = 0.01,
) -> Tuple[Spheres, Lines]:
    """
    构建骨架可视化对象（关节点 + 骨骼连线）
    
    Args:
        joints: [T, 24, 3] 关节位置
        name: 骨架名称
        color: RGBA颜色
        offset: 位置偏移
        joint_radius: 关节球体半径
        bone_radius: 骨骼线条半径
    
    Returns:
        (joint_spheres, bone_lines): 关节球体和骨骼线条
    """
    joints_np = joints.cpu().numpy()
    T, num_joints, _ = joints_np.shape
    
    # 添加偏移
    if offset != (0.0, 0.0, 0.0):
        joints_np = joints_np + np.array(offset)[np.newaxis, np.newaxis, :]
    
    # 创建关节点（球体）
    joint_spheres = Spheres(
        joints_np,
        radius=joint_radius,
        color=color,
        name=f"{name}_Joints",
        is_selectable=False,
    )
    
    # 创建骨骼连线
    # Lines类期望的格式是 (F, L, 3)，其中F是帧数，L是点数
    # 对于mode='lines'，点是成对的：(p0, p1), (p2, p3), (p4, p5)...
    bone_lines = []
    for t in range(T):
        frame_points = []
        for j in range(num_joints):
            parent_idx = SMPL_PARENTS[j]
            if parent_idx >= 0:  # 跳过根关节
                # 添加父关节点和子关节点（成对）
                frame_points.append(joints_np[t, parent_idx])  # 起点
                frame_points.append(joints_np[t, j])           # 终点
        bone_lines.append(frame_points)
    
    bone_lines = np.array(bone_lines)  # [T, num_bones*2, 3]
    bone_renderables = Lines(
        bone_lines,
        r_base=bone_radius,
        color=color,
        mode='lines',  # 成对连线模式
        name=f"{name}_Bones",
        is_selectable=False,
    )
    
    return joint_spheres, bone_renderables


def build_body_meshes(
    body_model: art.ParametricModel,
    glb_pred_smpl: torch.Tensor,
    glb_gt_smpl: torch.Tensor,
) -> Tuple[Meshes, Meshes]:
    local_pred = body_model.inverse_kinematics_R(glb_pred_smpl).view(glb_pred_smpl.shape[0], 24, 3, 3)
    _, _, verts_pred = body_model.forward_kinematics(local_pred, calc_mesh=True)

    local_gt = body_model.inverse_kinematics_R(glb_gt_smpl).view(glb_gt_smpl.shape[0], 24, 3, 3)
    _, _, verts_gt = body_model.forward_kinematics(local_gt, calc_mesh=True)
    verts_gt += torch.tensor([1.0, 0.0, 0.0])

    pred_mesh = Meshes(
        verts_pred.cpu().numpy(),
        body_model.face,
        is_selectable=False,
        gui_affine=False,
        name="Predicted Body Mesh",
    )
    gt_mesh = Meshes(
        verts_gt.cpu().numpy(),
        body_model.face,
        is_selectable=False,
        gui_affine=False,
        name="Ground Truth Body Mesh",
    )
    return pred_mesh, gt_mesh


def visualize(renderables: Sequence) -> None:
    """可视化所有渲染对象（网格、骨架等）"""
    viewer = Viewer()
    for obj in renderables:
        viewer.scene.add(obj)
    viewer.run()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dataset = MotionDatasetWithObjectAndTrans(
        datasets=[''],
        seq_len=args.seq_len,
        data_root=args.data_root,
        device=str(device),
        subset=args.subset,
        random_sample=args.random_sample,
        fps=args.fps,
        trim_frames=args.trim_frames,
    )
    idx = select_index(len(dataset), args.index, args.seed)
    sample = dataset[idx]
    imu = sample["imu"].float().to(device)
    pose = sample["glb_pose"].float()
    full_glb_pose = sample["full_glb_pose"].float()
    v_init = sample["v_init"].float().unsqueeze(0).to(device)
    p_init = sample["p_init"].float().unsqueeze(0).to(device)

    print(f"Visualizing sequence {idx}/{len(dataset)} with {imu.shape[0]} frames.")

    net = load_network(args.weight, device)
    _, glb_pred_smpl = net.predict(imu, v_init, p_init)

    body_model = art.ParametricModel(cfg.smpl_m, device="cpu")
    
    # 构建人体网格
    pred_mesh, gt_mesh = build_body_meshes(body_model, glb_pred_smpl, full_glb_pose)
    
    # 初始化渲染对象列表
    all_renderables = [pred_mesh, gt_mesh]
    
    # 如果启用骨架可视化，则添加骨架
    if args.show_skeleton:
        print("Building skeleton visualization...")
        
        # 获取关节位置用于骨架可视化
        local_pred = body_model.inverse_kinematics_R(glb_pred_smpl).view(glb_pred_smpl.shape[0], 24, 3, 3)
        _, joints_pred = body_model.forward_kinematics(local_pred, calc_mesh=False)
        
        local_gt = body_model.inverse_kinematics_R(full_glb_pose).view(full_glb_pose.shape[0], 24, 3, 3)
        _, joints_gt = body_model.forward_kinematics(local_gt, calc_mesh=False)
        
        # 构建骨架（预测值：红色，真值：绿色，添加偏移以便区分）
        pred_joints_sph, pred_bones_lines = build_skeleton(
            joints_pred,
            name="Predicted",
            color=(0.9, 0.2, 0.2, 1.0),  # 红色
            offset=(0.0, 0.0, 0.0),
            joint_radius=args.joint_radius,
            bone_radius=args.bone_radius,
        )
        
        gt_joints_sph, gt_bones_lines = build_skeleton(
            joints_gt,
            name="GroundTruth",
            color=(0.1, 0.8, 0.3, 1.0),  # 绿色
            offset=(1.0, 0.0, 0.0),  # 真值向右偏移1米
            joint_radius=args.joint_radius,
            bone_radius=args.bone_radius,
        )
        
        # 添加骨架到渲染列表
        all_renderables.extend([
            pred_joints_sph,
            pred_bones_lines,
            gt_joints_sph,
            gt_bones_lines,
        ])
        print(f"Skeleton added: {joints_pred.shape[1]} joints, joint_radius={args.joint_radius}, bone_radius={args.bone_radius}")
    
    visualize(all_renderables)


if __name__ == "__main__":
    main()
