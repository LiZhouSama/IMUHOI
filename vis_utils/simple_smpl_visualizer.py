"""Simple SMPL-H sequence visualizer for BEHAVE smpl_fit_all.npz files.

The script turns SMPL-H parameters into meshes with human_body_prior's
BodyModel and animates them with matplotlib.  Pass an --output path to
export an MP4, otherwise an interactive window is shown.
"""
from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from human_body_prior.body_model.body_model import BodyModel
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _find_model_path(support_dir: str, gender: Optional[str]) -> str:
    """Resolve a SMPL-H model path from support_dir for the given gender."""
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
        f"Could not find a SMPL-H model under '{support_dir}'. "
        "Expected files like smplh/male/model.npz."
    )


def _load_sequence(npz_path: str) -> dict:
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Sequence file does not exist: {npz_path}")
    with np.load(npz_path) as data:
        required = {"poses", "betas", "trans"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Sequence file is missing keys: {sorted(missing)}")
        return {key: data[key] for key in data.files}


def _split_poses(poses: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split SMPL-H axis-angle poses into root, body and hand components."""
    if poses.shape[-1] != 156:
        raise ValueError(
            "Expected 156 pose parameters per frame (SMPL-H full pose); "
            f"got {poses.shape[-1]}"
        )
    root_orient = torch.from_numpy(poses[:, :3]).float()
    pose_body = torch.from_numpy(poses[:, 3:66]).float()
    pose_hand = torch.from_numpy(poses[:, 66:]).float()
    return root_orient, pose_body, pose_hand


def _prepare_betas(betas: np.ndarray, num_frames: int) -> torch.Tensor:
    betas_t = torch.from_numpy(betas).float()
    if betas_t.ndim == 1:
        betas_t = betas_t.unsqueeze(0)
    if betas_t.shape[0] == 1:
        betas_t = betas_t.repeat(num_frames, 1)
    elif betas_t.shape[0] != num_frames:
        repeats = int(np.ceil(num_frames / betas_t.shape[0]))
        betas_t = betas_t.repeat(repeats, 1)[:num_frames]
    return betas_t


def _compute_vertices(
    body_model: BodyModel,
    root_orient: torch.Tensor,
    pose_body: torch.Tensor,
    pose_hand: torch.Tensor,
    trans: torch.Tensor,
    betas: torch.Tensor,
    batch_size: int = 256,
) -> torch.Tensor:
    """Run the body model in chunks to keep memory usage modest."""
    param = next(body_model.parameters(), None)
    if param is not None:
        device = param.device
    else:
        buffer = next(body_model.buffers(), None)
        device = buffer.device if buffer is not None else torch.device("cpu")
    verts_list = []
    with torch.no_grad():
        for start in range(0, root_orient.shape[0], batch_size):
            end = min(start + batch_size, root_orient.shape[0])
            body_out = body_model(
                root_orient=root_orient[start:end].to(device),
                pose_body=pose_body[start:end].to(device),
                pose_hand=pose_hand[start:end].to(device),
                betas=betas[start:end].to(device),
                trans=trans[start:end].to(device),
            )
            verts_list.append(body_out.v.detach().cpu())
    return torch.cat(verts_list, dim=0)


def _set_axes_equal(ax: plt.Axes, points: np.ndarray) -> None:
    """Make the 3D axes have equal scale based on the provided point cloud."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    max_range = (maxs - mins).max() / 2.0
    if max_range == 0:
        max_range = 0.5
    ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
    ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
    ax.set_zlim(centers[2] - max_range, centers[2] + max_range)


def _infer_fps(frame_times: Optional[np.ndarray]) -> float:
    if frame_times is None or frame_times.size < 2:
        return 30.0
    try:
        times = np.array([float(str(t)[1:]) for t in frame_times], dtype=np.float64)
        diffs = np.diff(times)
        positive = diffs[diffs > 1e-6]
        if positive.size == 0:
            return 30.0
        dt = float(np.median(positive))
        return 1.0 / dt if dt > 0 else 30.0
    except Exception:
        return 30.0


def visualize_sequence(
    sequence_path: str,
    support_dir: str,
    device: Optional[str] = None,
    every: int = 1,
    output_path: Optional[str] = None,
    dpi: int = 120,
) -> None:
    data = _load_sequence(sequence_path)
    gender = str(data.get("gender", "neutral"))
    frame_times = data.get("frame_times")

    indices = np.arange(data["poses"].shape[0])[::max(1, every)]
    poses = data["poses"][indices]
    betas = data["betas"][indices]
    trans = torch.from_numpy(data["trans"][indices]).float()
    frame_times = frame_times[indices] if frame_times is not None else None

    root_orient, pose_body, pose_hand = _split_poses(poses)
    betas_t = _prepare_betas(betas, root_orient.shape[0])

    model_path = _find_model_path(support_dir, gender)
    torch_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    body_model = BodyModel(
        bm_fname=model_path,
        num_betas=betas_t.shape[1]
    ).to(torch_device)
    body_model.eval()

    verts = _compute_vertices(
        body_model,
        root_orient,
        pose_body,
        pose_hand,
        trans,
        betas_t,
    )
    faces = body_model.f.detach().cpu().numpy()
    verts_np = verts.numpy()

    if verts_np.shape[0] == 0:
        raise RuntimeError("No frames were produced after subsampling.")

    all_points = verts_np.reshape(-1, 3)
    fps = _infer_fps(frame_times)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(verts_np[0][faces], linewidths=0.05, alpha=0.9)
    mesh.set_facecolor((0.55, 0.75, 0.95, 0.9))
    mesh.set_edgecolor((0.1, 0.1, 0.1, 0.05))
    ax.add_collection3d(mesh)
    _set_axes_equal(ax, all_points)
    ax.view_init(elev=15, azim=-75)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    frame_label = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def _update(frame_idx: int):
        mesh.set_verts(verts_np[frame_idx][faces])
        if frame_times is not None:
            frame_label.set_text(f"Frame {frame_idx} | t = {frame_times[frame_idx]}")
        else:
            frame_label.set_text(f"Frame {frame_idx}")
        return mesh, frame_label

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=verts_np.shape[0],
        interval=1000.0 / fps,
        blit=False,
    )

    if output_path:
        try:
            writer = animation.FFMpegWriter(fps=fps)
            anim.save(output_path, writer=writer, dpi=dpi)
        except RuntimeError as exc:
            raise RuntimeError(
                "Saving the animation requires ffmpeg to be available on PATH."
            ) from exc
        finally:
            plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a BEHAVE SMPL-H sequence.")
    parser.add_argument("sequence", help="Path to smpl_fit_all.npz file")
    parser.add_argument(
        "--support-dir",
        default="body_models",
        help="Directory that contains smplh/<gender>/model.npz (default: body_models)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Force computation device (default: auto)",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Subsample frames by this factor to speed up rendering",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save an MP4 animation instead of showing a window",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Resolution (dots per inch) when exporting a video",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualize_sequence(
        sequence_path=args.sequence,
        support_dir=args.support_dir,
        device=args.device,
        every=max(1, args.every),
        output_path=args.output,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
