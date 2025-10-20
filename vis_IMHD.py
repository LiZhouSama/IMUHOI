import argparse
import glob
import os
import traceback
from typing import Dict, List, Optional, Tuple

import imgui
import numpy as np
import torch
import trimesh
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.spheres import Spheres
from aitviewer.viewer import Viewer
from human_body_prior.body_model.body_model import BodyModel
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix

# R_Y_UP = torch.tensor(
#     [
#         [0.0, -1.0, 0.0],
#         [1.0, 0.0, 0.0],
#         [0.0, 0.0, 1.0],
#     ],
#     dtype=torch.float32,
# )

R_Y_UP = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)

def ensure_tensor(data, device: torch.device) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=torch.float32)
    return torch.tensor(np.asarray(data), device=device, dtype=torch.float32)


class BodyModelLoader:
    def __init__(self, support_dir: str, device: torch.device, num_betas: int = 16) -> None:
        self._device = device
        self._num_betas = num_betas
        self._models: Dict[str, BodyModel] = {}
        for gender in ["male", "female", "neutral"]:
            bm_path = os.path.join(support_dir, f"smplh/{gender}/model.npz")
            if os.path.exists(bm_path):
                self._models[gender] = BodyModel(
                    bm_fname=bm_path,
                    num_betas=num_betas,
                    model_type="smplh",
                ).to(device).eval()
        if not self._models:
            raise FileNotFoundError(f"No SMPL-H model was found under {support_dir}.")
        self._default_gender = "neutral" if "neutral" in self._models else "male"

    @property
    def num_betas(self) -> int:
        return self._num_betas

    def get(self, gender: Optional[str]) -> BodyModel:
        if gender is None:
            return self._models[self._default_gender]
        key = str(gender).lower()
        return self._models.get(key, self._models[self._default_gender])


def infer_translations(data: Dict, device: torch.device) -> torch.Tensor:
    if "trans" in data:
        return ensure_tensor(data["trans"], device)
    pos_global = data.get("position_global_full_gt_world")
    if pos_global is None:
        raise KeyError("Missing 'trans' or 'position_global_full_gt_world' in sequence data.")
    pos_global_t = ensure_tensor(pos_global, device)
    return pos_global_t[:, 0, :]


def find_object_mesh(obj_name: str, objects_root: str) -> Tuple[str, np.ndarray, np.ndarray]:
    direct_file = os.path.join(objects_root, f"{obj_name}.obj")
    candidates: List[str] = []
    if os.path.exists(direct_file):
        candidates.append(direct_file)
    obj_dir = os.path.join(objects_root, obj_name)
    if os.path.isdir(obj_dir):
        objs = sorted(glob.glob(os.path.join(obj_dir, "*.obj")))
        preferred = [
            path for path in objs
            if "simplified" in os.path.basename(path) and "transformed" in os.path.basename(path)
        ]
        candidates.extend(preferred or objs)
    if not candidates:
        raise FileNotFoundError(f"No mesh file was found for object '{obj_name}'.")
    mesh_path = candidates[0]
    mesh = trimesh.load(mesh_path, process=False)
    return mesh_path, np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)


def prepare_human_vertices(data: Dict, bm_loader: BodyModelLoader, device: torch.device) -> Tuple[torch.Tensor, np.ndarray]:
    rot6d = ensure_tensor(data["rotation_local_full_gt_list"], device)
    num_frames = rot6d.shape[0]
    rot_mat = rotation_6d_to_matrix(rot6d.view(num_frames, -1, 6)).view(num_frames, -1, 3, 3)
    axis_angle = matrix_to_axis_angle(rot_mat)
    root_orient = axis_angle[:, 0, :]
    pose_body = axis_angle[:, 1:, :].reshape(num_frames, -1)
    trans = infer_translations(data, device)

    if "betas" in data:
        betas = ensure_tensor(data["betas"], device)
        if betas.dim() == 1:
            betas = betas.unsqueeze(0)
        if betas.shape[0] == 1 and num_frames > 1:
            betas = betas.repeat(num_frames, 1)
        elif betas.shape[0] < num_frames:
            repeat_count = int(np.ceil(num_frames / betas.shape[0]))
            betas = betas.repeat(repeat_count, 1)[:num_frames]
        else:
            betas = betas[:num_frames]
    else:
        betas = torch.zeros((num_frames, bm_loader.num_betas), dtype=torch.float32, device=device)

    bm = bm_loader.get(data.get("gender"))
    with torch.no_grad():
        body_out = bm(
            root_orient=root_orient,
            pose_body=pose_body,
            trans=trans,
            betas=betas,
        )
    verts = body_out.v.detach()
    faces = bm.f.detach().cpu().numpy()
    return verts, faces


def prepare_object_vertices(data: Dict, objects_root: str, device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
    obj_name = data.get("obj_name")
    if obj_name is None:
        return None, None
    obj_rot = data.get("obj_rot")
    obj_trans = data.get("obj_trans")
    if obj_rot is None or obj_trans is None:
        return None, None

    obj_rot_t = ensure_tensor(obj_rot, device)
    obj_trans_t = ensure_tensor(obj_trans, device)

    scale_tensor: Optional[torch.Tensor] = None
    if "obj_scale" in data and data["obj_scale"] is not None:
        scale_tensor = ensure_tensor(data["obj_scale"], device)
        if scale_tensor.dim() == 1:
            scale_tensor = scale_tensor.view(-1, 1, 1)
        elif scale_tensor.dim() == 2:
            scale_tensor = scale_tensor.view(scale_tensor.shape[0], 1, 1)

    try:
        _, base_vertices, faces = find_object_mesh(obj_name, objects_root)
    except FileNotFoundError as exc:
        print(exc)
        return None, None

    base_vertices_t = torch.from_numpy(base_vertices).to(device=device, dtype=torch.float32)
    centroid = base_vertices_t.mean(dim=0, keepdim=True)
    base_vertices_centered = base_vertices_t - centroid
    num_frames = obj_trans_t.shape[0]
    verts = base_vertices_centered.unsqueeze(0).repeat(num_frames, 1, 1)
    rotated = torch.bmm(obj_rot_t, verts.transpose(1, 2))
    if scale_tensor is not None:
        rotated = rotated * scale_tensor
    transformed = rotated.transpose(1, 2) + obj_trans_t.unsqueeze(1)
    return transformed, faces


def to_y_up(verts: torch.Tensor) -> torch.Tensor:
    rotation = R_Y_UP.to(device=verts.device, dtype=verts.dtype)
    return torch.matmul(verts, rotation.T)


def collect_sequences(data_path: str) -> List[str]:
    if os.path.isdir(data_path):
        return sorted(glob.glob(os.path.join(data_path, "*.pt")))
    if data_path.lower().endswith(".pt") and os.path.isfile(data_path):
        return [data_path]
    raise FileNotFoundError(f"Unrecognized data path: {data_path}")


class IMHDViewer(Viewer):
    def __init__(
        self,
        seq_files: List[str],
        bm_loader: BodyModelLoader,
        objects_root: str,
        device: torch.device,
        playback_fps: float,
    ) -> None:
        super().__init__(requires_imgui=True)
        self.seq_files = seq_files
        self.bm_loader = bm_loader
        self.objects_root = objects_root
        self.device = device
        self.playback_fps = playback_fps
        self.current_index = 0
        self.human_node = None
        self.object_node = None
        self.contact_nodes: List[Spheres] = []
        self.info_text = ""
        self.title = "vis_IMHD"
        self.scene.camera.near = 0.05
        self.scene.camera.far = 50.0
        self._load_sequence(self.current_index)

    def _clear_nodes(self) -> None:
        nodes_to_remove = [node for node in (self.human_node, self.object_node) if node is not None]
        nodes_to_remove.extend(self.contact_nodes)
        remove_fn = getattr(self.scene, "remove", None)
        remove_node_fn = getattr(self.scene, "remove_node", None)

        for node in nodes_to_remove:
            try:
                if remove_fn is not None:
                    remove_fn(node)
                elif remove_node_fn is not None:
                    remove_node_fn(node)
                else:
                    matches = [
                        existing for existing in self.scene.collect_nodes()
                        if existing is node or (
                            hasattr(existing, "name")
                            and hasattr(node, "name")
                            and existing.name == node.name
                        )
                    ]
                    for candidate in matches:
                        if remove_fn is not None:
                            remove_fn(candidate)
                        elif remove_node_fn is not None:
                            remove_node_fn(candidate)
                        elif hasattr(candidate, "remove"):
                            candidate.remove()
            except Exception as exc:
                print(f"Warning: failed to remove node {getattr(node, 'name', None)}: {exc}")
        self.human_node = None
        self.object_node = None
        self.contact_nodes = []

    def _add_contact_indicators(self, data: Dict) -> None:
        contact_radius = 0.03
        contact_nodes: List[Spheres] = []
        rotation = R_Y_UP.to(device=self.device)

        try:
            joints = data.get("position_global_full_gt_world")
            if joints is not None:
                joints_t = ensure_tensor(joints, self.device)
                joints_yup = torch.matmul(joints_t, rotation.T)
                lhand_mask = data.get("lhand_contact")
                if lhand_mask is not None:
                    mask = (ensure_tensor(lhand_mask, self.device) > 0.5).reshape(-1)
                    if mask.any():
                        lhand_points = joints_yup[mask, 20, :].detach().cpu().numpy()
                        spheres = Spheres(
                            positions=lhand_points,
                            radius=contact_radius,
                            name="GT-LHandContact",
                            color=(1.0, 0.0, 0.0, 0.8),
                            gui_affine=False,
                            is_selectable=False,
                        )
                        self.scene.add(spheres)
                        contact_nodes.append(spheres)

                rhand_mask = data.get("rhand_contact")
                if rhand_mask is not None:
                    mask = (ensure_tensor(rhand_mask, self.device) > 0.5).reshape(-1)
                    if mask.any():
                        rhand_points = joints_yup[mask, 21, :].detach().cpu().numpy()
                        spheres = Spheres(
                            positions=rhand_points,
                            radius=contact_radius,
                            name="GT-RHandContact",
                            color=(0.0, 0.0, 1.0, 0.8),
                            gui_affine=False,
                            is_selectable=False,
                        )
                        self.scene.add(spheres)
                        contact_nodes.append(spheres)

            obj_contact = data.get("obj_contact")
            obj_trans = data.get("obj_trans")
            if obj_contact is not None and obj_trans is not None:
                mask = (ensure_tensor(obj_contact, self.device) > 0.5).reshape(-1)
                if mask.any():
                    trans_t = ensure_tensor(obj_trans, self.device)
                    trans_yup = torch.matmul(trans_t, rotation.T)
                    contact_points = trans_yup[mask].detach().cpu().numpy()
                    spheres = Spheres(
                        positions=contact_points,
                        radius=0.04,
                        name="GT-ObjectContact",
                        color=(1.0, 1.0, 0.0, 0.8),
                        gui_affine=False,
                        is_selectable=False,
                    )
                    self.scene.add(spheres)
                    contact_nodes.append(spheres)
        except Exception as exc:
            print(f"Failed to add contact indicators: {exc}")

        self.contact_nodes.extend(contact_nodes)

    def _load_sequence(self, index: int) -> None:
        self._clear_nodes()
        seq_path = self.seq_files[index]
        data = torch.load(seq_path, map_location="cpu")

        human_frames = 0
        obj_frames = 0
        try:
            human_verts, human_faces = prepare_human_vertices(data, self.bm_loader, self.device)
            human_verts = to_y_up(human_verts)
            human_frames = human_verts.shape[0]
            human_mesh = Meshes(
                human_verts.cpu().numpy(),
                human_faces,
                name="IMHD-Human",
                color=(0.2, 0.7, 0.9, 1.0),
                gui_affine=False,
                is_selectable=False,
            )
            human_mesh.playback_fps = self.playback_fps
            self.scene.add(human_mesh)
            self.human_node = human_mesh
        except Exception as exc:
            print(f"Failed to load human mesh: {exc}")
            traceback.print_exc()

        obj_verts, obj_faces = prepare_object_vertices(data, self.objects_root, self.device)
        if obj_verts is not None and obj_faces is not None:
            obj_verts = to_y_up(obj_verts)
            obj_frames = obj_verts.shape[0]
            obj_mesh = Meshes(
                obj_verts.cpu().numpy(),
                obj_faces,
                name=f"IMHD-{data.get('obj_name', 'object')}",
                color=(0.9, 0.7, 0.2, 0.8),
                gui_affine=False,
                is_selectable=False,
            )
            obj_mesh.playback_fps = self.playback_fps
            self.scene.add(obj_mesh)
            self.object_node = obj_mesh

        self._add_contact_indicators(data)

        seq_name = data.get("seq_name", os.path.basename(seq_path))
        frame_count = obj_frames or human_frames
        pkls = data.get("source_pkls")
        pkls_text = f" | Segments: {len(pkls)}" if isinstance(pkls, (list, tuple)) else ""
        self.info_text = f"Sequence: {seq_name} | Frames: {frame_count} | File: {os.path.basename(seq_path)}{pkls_text}"
        self.title = f"vis_IMHD - {seq_name} ({index + 1}/{len(self.seq_files)})"
        self.scene.current_frame_id = 0
        self.scene.camera.position = np.array([2.5, 1.6, 2.5])
        self.scene.camera.target = np.array([0.0, 1.0, 0.0])

    def change_sequence(self, delta: int) -> None:
        new_index = int(np.clip(self.current_index + delta, 0, len(self.seq_files) - 1))
        if new_index != self.current_index:
            self.current_index = new_index
            self._load_sequence(self.current_index)

    def key_event(self, key, action, modifiers) -> None:  # type: ignore[override]
        super().key_event(key, action, modifiers)
        if action != self.wnd.keys.ACTION_PRESS:
            return
        io = imgui.get_io()
        if self.render_gui and (io.want_capture_keyboard or io.want_text_input):
            return
        step = 1
        if modifiers.ctrl:
            step = 10
        elif modifiers.alt:
            step = 50
        if key == self.wnd.keys.Q:
            self.change_sequence(-step)
        elif key == self.wnd.keys.E:
            self.change_sequence(step)

    def draw_ui(self) -> None:
        if not self.render_gui:
            return
        imgui.begin("IMHD Sequence Info", True)
        imgui.text(self.info_text)
        imgui.separator()
        imgui.text("Controls: Q/E change sequence, hold Ctrl or Alt to jump faster.")
        imgui.text("Playback: Space toggles play/pause (aitviewer default).")
        imgui.end()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize IMHD processed sequences.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="processed_IMHD_data_1014",
        help="Path to a sequence .pt file or a directory containing them.",
    )
    parser.add_argument(
        "--objects_root",
        type=str,
        default=r"datasets/IMHD/IMHD Dataset/object_templates",
        help="Directory that stores IMHD object meshes.",
    )
    parser.add_argument(
        "--support_dir",
        type=str,
        default="body_models",
        help="Directory with SMPL-H support files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device (cuda or cpu).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Playback frame rate.",
    )
    parser.add_argument(
        "--num_betas",
        type=int,
        default=16,
        help="Number of betas to request from the SMPL-H loader.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_files = collect_sequences(args.data_path)
    bm_loader = BodyModelLoader(args.support_dir, device, num_betas=args.num_betas)
    viewer = IMHDViewer(seq_files, bm_loader, args.objects_root, device, args.fps)
    viewer.run()


if __name__ == "__main__":
    main()
