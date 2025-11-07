"""Interactive viewer for GlobalPose predictions on the DIP-IMU dataset."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import articulate as art

from aitviewer.renderables.meshes import Meshes
from aitviewer.viewer import Viewer

from net import GPNet


R_YUP = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)


@dataclass
class SequenceBundle:
    name: str
    gt_verts: np.ndarray
    pred_verts: Optional[np.ndarray]
    faces: np.ndarray
    num_frames: int


class DipViewer(Viewer):
    """Minimal aitviewer wrapper with keyboard controls for browsing sequences."""

    def __init__(self, sequences: Sequence[SequenceBundle], fps: float, show_pred: bool) -> None:
        super().__init__(title="GlobalPose DIP Viewer", size=(1280, 720))
        self.sequences = list(sequences)
        self.show_pred = show_pred
        self.current_index = 0
        self.target_fps = fps
        try:
            self.scene.frame_dt = 1.0 / max(fps, 1e-5)
        except AttributeError:
            pass
        self._display_sequence(0, reset_camera=True)

    def _clear_meshes(self) -> None:
        nodes = [
            node
            for node in self.scene.collect_nodes()
            if hasattr(node, "name")
            and isinstance(node.name, str)
            and (node.name.startswith("GT-") or node.name.startswith("Pred-"))
        ]
        for node in nodes:
            try:
                self.scene.remove(node)
            except Exception as exc:  # pragma: no cover - viewer best effort cleanup
                print(f"Warning: failed to remove node {node.name}: {exc}")

    def _display_sequence(self, index: int, reset_camera: bool = False) -> None:
        if not self.sequences:
            return
        index = max(0, min(index, len(self.sequences) - 1))
        bundle = self.sequences[index]
        self._clear_meshes()

        mesh_gt = Meshes(
            bundle.gt_verts,
            bundle.faces,
            name=f"GT-{bundle.name}",
            color=(0.2, 0.8, 0.3, 0.8),
            gui_affine=False,
            is_selectable=False,
        )
        self.scene.add(mesh_gt)

        if self.show_pred and bundle.pred_verts is not None:
            mesh_pred = Meshes(
                bundle.pred_verts,
                bundle.faces,
                name=f"Pred-{bundle.name}",
                color=(0.9, 0.2, 0.2, 0.8),
                gui_affine=False,
                is_selectable=False,
            )
            self.scene.add(mesh_pred)

        self.title = (
            f"GlobalPose DIP Viewer | {bundle.name} "
            f"[{index + 1}/{len(self.sequences)} | {bundle.num_frames} frames]"
        )
        self.current_index = index
        try:
            self.scene.current_frame_id = 0
        except AttributeError:
            pass
        if reset_camera:
            try:
                self.scene.reset()
            except AttributeError:
                pass

    def key_event(self, key, action, modifiers) -> None:  # type: ignore[override]
        super().key_event(key, action, modifiers)
        if not self.sequences or action != self.wnd.keys.ACTION_PRESS:
            return
        if key == self.wnd.keys.Q:
            self._display_sequence(self.current_index - 1)
        elif key == self.wnd.keys.E:
            self._display_sequence(self.current_index + 1)
        elif key == self.wnd.keys.R:
            self._display_sequence(self.current_index, reset_camera=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise GlobalPose predictions on DIP-IMU")
    parser.add_argument(
        "--data",
        default="data/test_datasets/dipimu.pt",
        help="Path to the DIP-IMU evaluation dataset (.pt file).",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/dip/best.pt",
        help="Checkpoint produced by train.py (expects a dict with a 'model' state dict).",
    )
    parser.add_argument(
        "--smpl-model",
        default="models/SMPL_male.pkl",
        help="SMPL model file used for mesh reconstruction.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for running GPNet (default: cuda if available).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Index of the first sequence to visualise.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of sequences to preload (0 means all).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Visualise every N-th frame to keep playback light.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Playback FPS for the viewer.",
    )
    parser.add_argument(
        "--no-pred",
        action="store_true",
        help="Only show ground-truth meshes (skip network inference).",
    )
    parser.add_argument(
        "--no-align-root",
        dest="align_root",
        action="store_false",
        help="Do not align predicted translations to the first frame.",
    )
    parser.set_defaults(align_root=True)
    return parser.parse_args()


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    if not checkpoint_path:
        print("No checkpoint provided; using model initialisation.")
        return
    if not os.path.exists(checkpoint_path):
        print(f"Warning: checkpoint {checkpoint_path} not found; using model initialisation.")
        return
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint from {checkpoint_path}.")


def to_view_coords(verts: torch.Tensor) -> torch.Tensor:
    return torch.matmul(verts, R_YUP.to(verts.device).t())


def _get_sequence_name(data: dict, index: int) -> str:
    names = data.get("name")
    if isinstance(names, list) and len(names) > index:
        return str(names[index])
    return f"seq_{index:04d}"


def _maybe_get_shape(data: dict, index: int) -> Optional[torch.Tensor]:
    shapes = data.get("shape")
    if isinstance(shapes, list) and len(shapes) > index:
        return shapes[index]
    return None


@torch.no_grad()
def run_sequence(
    model: GPNet,
    data: dict,
    index: int,
    device: torch.device,
    body_model: art.ParametricModel,
    stride: int,
    align_root: bool,
    run_prediction: bool,
) -> Optional[SequenceBundle]:
    try:
        aS = data["aS"][index].to(device).float()
        wS = data["wS"][index].to(device).float()
        RIS = data["RIS"][index].to(device).float()
        RIM = data["RIM"][index].to(device).float()
        RSB = data["RSB"][index].to(device).float()
        tran_gt = data["tran"][index].to(device).float()
        pose_aa = data["pose"][index].to(device).float()
    except KeyError as exc:
        print(f"Missing key {exc} in dataset; skipping sequence {index}.")
        return None

    seq_len = pose_aa.shape[0]
    if seq_len == 0:
        print(f"Sequence {index} is empty; skipping.")
        return None

    g_vec = torch.tensor([0.0, -9.8, 0.0], device=device, dtype=torch.float32)
    rim_t = RIM.transpose(-1, -2)
    rot_im_to_model = torch.matmul(rim_t, RIS)
    RMB = torch.matmul(rot_im_to_model, RSB)
    aM = torch.matmul(rot_im_to_model, aS.unsqueeze(-1)).squeeze(-1) + g_vec
    wM = torch.matmul(rot_im_to_model, wS.unsqueeze(-1)).squeeze(-1)
    pose_gt = art.math.axis_angle_to_rotation_matrix(pose_aa).view(-1, 24, 3, 3)

    # Ground-truth mesh
    shape = _maybe_get_shape(data, index)
    pose_gt_cpu = pose_gt.cpu()
    tran_gt_cpu = tran_gt.cpu()
    shape_cpu = shape.cpu() if isinstance(shape, torch.Tensor) else None
    _, _, gt_verts = body_model.forward_kinematics(
        pose_gt_cpu,
        shape=shape_cpu,
        tran=tran_gt_cpu,
        calc_mesh=True,
    )

    pred_verts_np: Optional[np.ndarray] = None
    if run_prediction:
        model.rnn_initialize(pose_gt[0])
        pose_frames: List[torch.Tensor] = []
        tran_frames: List[torch.Tensor] = []
        for t in range(seq_len):
            pose_t, tran_t = model.forward_frame(aM[t], wM[t], RMB[t])
            pose_frames.append(pose_t.float())
            tran_frames.append(tran_t.float())
        pose_pred = torch.stack(pose_frames, dim=0)
        tran_pred = torch.stack(tran_frames, dim=0)
        if align_root and tran_pred.shape[0] > 0:
            tran_pred = tran_pred - tran_pred[0:1]
        pose_pred_cpu = pose_pred.cpu()
        tran_pred_cpu = tran_pred.cpu()
        _, _, pred_verts = body_model.forward_kinematics(
            pose_pred_cpu,
            shape=shape_cpu,
            tran=tran_pred_cpu,
            calc_mesh=True,
        )
        pred_view = to_view_coords(pred_verts.cpu())
        if stride > 1:
            pred_view = pred_view[::stride]
        pred_verts_np = pred_view.numpy()

    gt_view = to_view_coords(gt_verts.cpu())
    if stride > 1:
        gt_view = gt_view[::stride]

    faces = np.asarray(body_model.face, dtype=np.int32)
    return SequenceBundle(
        name=_get_sequence_name(data, index),
        gt_verts=gt_view.numpy(),
        pred_verts=pred_verts_np,
        faces=faces,
        num_frames=int(gt_view.shape[0]),
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    run_prediction = not args.no_pred

    gpnet = GPNet(pretrained_path="").to(device)
    gpnet.eval()
    if run_prediction:
        load_checkpoint(gpnet, args.checkpoint, device)
    else:
        print("--no-pred supplied; skipping network inference.")

    body_model = art.ParametricModel(args.smpl_model)
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset file not found: {args.data}")
    data = torch.load(args.data, map_location="cpu")

    total_sequences = len(data.get("pose", []))
    if total_sequences == 0:
        raise RuntimeError("Dataset does not contain any sequences under key 'pose'.")

    start = max(0, int(args.start))
    end_limit = total_sequences if args.limit <= 0 else min(total_sequences, start + args.limit)
    indices = list(range(start, end_limit))

    print(f"Preparing {len(indices)} sequence(s) for visualisation...")
    sequences: List[SequenceBundle] = []
    for idx in indices:
        bundle = run_sequence(
            gpnet,
            data,
            idx,
            device,
            body_model,
            stride=max(1, args.stride),
            align_root=args.align_root,
            run_prediction=run_prediction,
        )
        if bundle is not None:
            sequences.append(bundle)

    if not sequences:
        raise RuntimeError("No valid sequences were prepared for visualisation.")

    viewer = DipViewer(sequences, fps=args.fps, show_pred=run_prediction)
    viewer.run()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
