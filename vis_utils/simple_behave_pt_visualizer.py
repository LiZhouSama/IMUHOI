"""从 preprocess_BEHAVE.py 生成的单个序列 .pt 文件中读取 6D 姿态并可视化 SMPL-H 人体。

脚本逻辑参考 simple_smpl_visualizer.py：
- 载入 torch.save 保存的字典，提取 rotation_local_full_gt_list (6D 局部旋转) 等信息；
- 转换为轴角，再拆分 root/body pose；
- 通过 human_body_prior.BodyModel 计算网格顶点；
- 用 matplotlib 做简单的逐帧动画，可选保存 MP4。
"""
from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional, Sequence

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from human_body_prior.body_model.body_model import BodyModel
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pytorch3d import transforms

# -----------------------------------------------------------------------------
# 帮助函数
# -----------------------------------------------------------------------------


def _find_model_path(support_dir: str, gender: Optional[str]) -> str:
    requested = (gender or "").lower()
    priority: Iterable[str] = ()
    if requested in {"male", "female", "neutral"}:
        priority = (requested,)
    priority = tuple(priority) + ("neutral", "male", "female")

    for candidate in priority:
        candidate_path = os.path.join(support_dir, "smplh", candidate, "model.npz")
        if os.path.exists(candidate_path):
            return candidate_path

    raise FileNotFoundError(
        f"在 '{support_dir}' 下未找到任何 SMPL-H 模型，"
        "请确认包含 smplh/<gender>/model.npz。"
    )


def _load_sequence(pt_path: str) -> dict:
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"序列文件不存在: {pt_path}")
    data = torch.load(pt_path, map_location="cpu")
    if "rotation_local_full_gt_list" not in data:
        raise KeyError("数据中缺少 'rotation_local_full_gt_list' 键")
    return data


def _rotation6d_to_axis_angle(rot6d: torch.Tensor) -> torch.Tensor:
    rot_mats = transforms.rotation_6d_to_matrix(rot6d.view(rot6d.shape[0], -1, 6))
    axis_angle = transforms.matrix_to_axis_angle(rot_mats)
    return axis_angle


def _prepare_betas(betas: Optional[torch.Tensor], num_frames: int, latent_dim: int) -> torch.Tensor:
    if betas is None:
        return torch.zeros((num_frames, latent_dim), dtype=torch.float32)
    if betas.ndim == 1:
        betas = betas.unsqueeze(0)
    if betas.shape[0] == 1:
        betas = betas.repeat(num_frames, 1)
    elif betas.shape[0] != num_frames:
        repeats = int(np.ceil(num_frames / betas.shape[0]))
        betas = betas.repeat(repeats, 1)[:num_frames]
    if betas.shape[1] < latent_dim:
        pad = latent_dim - betas.shape[1]
        betas = torch.cat([betas, torch.zeros((num_frames, pad), dtype=betas.dtype)], dim=1)
    else:
        betas = betas[:, :latent_dim]
    return betas


def _infer_trans(data: dict, num_frames: int) -> torch.Tensor:
    if "trans" in data:
        trans = data["trans"]
        trans_t = trans if isinstance(trans, torch.Tensor) else torch.tensor(trans)
        return trans_t.float()
    if "position_global_full_gt_world" in data:
        pos = data["position_global_full_gt_world"]
        pos_t = pos if isinstance(pos, torch.Tensor) else torch.tensor(pos)
        return pos_t.float()[:, 0, :]
    return torch.zeros((num_frames, 3), dtype=torch.float32)


def _compute_vertices(
    bm: BodyModel,
    root_orient: torch.Tensor,
    pose_body: torch.Tensor,
    betas: torch.Tensor,
    trans: torch.Tensor,
    batch_size: int = 256,
) -> torch.Tensor:
    param = next(bm.parameters(), None)
    device = param.device if param is not None else next(bm.buffers(), torch.zeros(1)).device
    verts_all = []
    with torch.no_grad():
        for start in range(0, root_orient.shape[0], batch_size):
            end = min(start + batch_size, root_orient.shape[0])
            out = bm(
                root_orient=root_orient[start:end].to(device),
                pose_body=pose_body[start:end].to(device),
                trans=trans[start:end].to(device),
                betas=betas[start:end].to(device),
            )
            verts_all.append(out.v.detach().cpu())
    return torch.cat(verts_all, dim=0)


def _set_axes_equal(ax: plt.Axes, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0
    if radius < 1e-6:
        radius = 0.5
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _infer_fps(frame_times: Optional[Sequence]) -> float:
    if frame_times is None:
        return 30.0
    try:
        np_times = np.asarray(frame_times)
        if np_times.size < 2:
            return 30.0
        if np.issubdtype(np_times.dtype, np.str_):
            parsed = []
            for item in np_times:
                item_str = str(item)
                if item_str.startswith("t"):
                    item_str = item_str[1:]
                try:
                    parsed.append(float(item_str))
                except ValueError:
                    return 30.0
            values = np.asarray(parsed, dtype=np.float64)
        else:
            values = np_times.astype(np.float64, copy=False).reshape(-1)
        diffs = np.diff(values)
        positive = diffs[diffs > 1e-6]
        if positive.size == 0:
            return 30.0
        dt = float(np.median(positive))
        return 1.0 / dt if dt > 0 else 30.0
    except Exception:
        return 30.0


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------


def visualize_pt(
    pt_path: str,
    support_dir: str,
    every: int = 1,
    output: Optional[str] = None,
    dpi: int = 120,
    device: Optional[str] = None,
    num_betas: int = 10,
) -> None:
    data = _load_sequence(pt_path)
    rot6d = data["rotation_local_full_gt_list"]
    rot6d = rot6d if isinstance(rot6d, torch.Tensor) else torch.tensor(rot6d)

    total_frames = rot6d.shape[0]
    step = max(1, every)
    indices = torch.arange(0, total_frames, step)
    rot6d = rot6d[indices]

    axis_angle = _rotation6d_to_axis_angle(rot6d)
    root_orient = axis_angle[:, 0, :]
    pose_body = axis_angle[:, 1:, :].reshape(axis_angle.shape[0], -1)

    betas = data.get("betas")
    betas_t = betas if isinstance(betas, torch.Tensor) else (torch.tensor(betas) if betas is not None else None)
    betas_full = _prepare_betas(betas_t, root_orient.shape[0], latent_dim=num_betas)

    trans_all = _infer_trans(data, total_frames)
    trans_sel = trans_all[indices]

    frame_times_raw = data.get("frame_times")
    frame_times = None
    if frame_times_raw is not None:
        np_frames = np.asarray(frame_times_raw)
        frame_times = np_frames[indices.numpy()]

    gender = str(data.get("gender", "neutral"))
    model_path = _find_model_path(support_dir, gender)
    torch_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bm = BodyModel(
        bm_fname=model_path,
        num_betas=num_betas
    ).to(torch_device)
    bm.eval()

    verts = _compute_vertices(bm, root_orient, pose_body, betas_full, trans_sel)
    faces = bm.f.detach().cpu().numpy()
    verts_np = verts.numpy()
    if verts_np.shape[0] == 0:
        raise RuntimeError("没有可视化的帧，请检查 every 参数或输入数据。")

    all_points = verts_np.reshape(-1, 3)
    fps = _infer_fps(frame_times)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    mesh_color = to_rgba("#8dc6f0", alpha=0.9)
    edge_color = to_rgba("#111111", alpha=0.05)
    poly = Poly3DCollection(verts_np[0][faces], linewidths=0.05, alpha=mesh_color[-1])
    poly.set_facecolor(mesh_color)
    poly.set_edgecolor(edge_color)
    ax.add_collection3d(poly)
    _set_axes_equal(ax, all_points)
    ax.view_init(elev=15, azim=-75)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title = os.path.basename(pt_path)
    frame_label = ax.text2D(0.02, 0.95, title, transform=ax.transAxes)

    def _update(idx: int):
        poly.set_verts(verts_np[idx][faces])
        if frame_times is not None:
            frame_label.set_text(f"Frame {indices[idx].item()} | t={frame_times[idx]}")
        else:
            frame_label.set_text(f"Frame {indices[idx].item()}")
        return poly, frame_label

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=verts_np.shape[0],
        interval=1000.0 / fps,
        blit=False,
    )

    if output:
        try:
            writer = animation.FFMpegWriter(fps=fps)
            anim.save(output, writer=writer, dpi=dpi)
        except RuntimeError as exc:
            raise RuntimeError("保存视频需要在 PATH 中安装 ffmpeg。") from exc
        finally:
            plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化 preprocess_BEHAVE.py 输出的单个序列")
    parser.add_argument("pt_path", help=".pt 文件路径")
    parser.add_argument("--support-dir", default="body_models", help="SMPL-H 模型目录 (默认 body_models)")
    parser.add_argument("--every", type=int, default=1, help="帧间隔采样，加大可加速渲染")
    parser.add_argument("--output", help="若提供则保存为 MP4 文件，否则弹窗显示")
    parser.add_argument("--dpi", type=int, default=120, help="导出视频时的 DPI")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="计算设备，默认自动")
    parser.add_argument("--num-betas", type=int, default=10, help="SMPL shape 参数维度，需与模型一致")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualize_pt(
        pt_path=args.pt_path,
        support_dir=args.support_dir,
        every=max(1, args.every),
        output=args.output,
        dpi=args.dpi,
        device=args.device,
        num_betas=max(1, args.num_betas),
    )


if __name__ == "__main__":
    main()
