import sys
sys.path.append("/mnt/d/a_WORK/Projects/PhD/tasks/EgoIMU")
import os
import glob
import argparse
import numpy as np
import torch
import pytorch3d.transforms as transforms
import trimesh
import imgui

from aitviewer.viewer import Viewer
from aitviewer.renderables.meshes import Meshes

from human_body_prior.body_model.body_model import BodyModel

from model.dataset_trans_obj import load_dynaip_sequence
from model.model_trans_obj2 import PoserWithObjectAndTransV2


# --- Rotate from Z-up to Y-up ---
# R_yup = torch.tensor([[1.0, 0.0, 0.0],
#                       [0.0, 0.0, 1.0],
#                       [0.0, -1.0, 0.0]], dtype=torch.float32)
R_yup = torch.eye(3, dtype=torch.float32)

VEL_SELECTION = torch.tensor([0, 15, 20, 21, 7, 8], dtype=torch.long)
POSE_SELECTION = torch.tensor([0, 1, 2, 5, 6, 7, 8, 9, 10, 12, 13], dtype=torch.long)
DEFAULT_TRIM_FRAMES = 6


def read_preprocessed_sequences(src_dir: str, fps: float, trim_frames: int = DEFAULT_TRIM_FRAMES):
    files = sorted(glob.glob(os.path.join(src_dir, "*.pt")))
    for f in files:
        try:
            bundle = load_dynaip_sequence(f, fps=fps, trim_frames=trim_frames, keep_raw_keys=True)
            yield f, bundle
        except Exception as e:
            print(f"Skip {f}: {e}")


def integrate_object_position(obj_v: torch.Tensor, p0: torch.Tensor, fps: float):
    dt = 1.0 / fps
    if obj_v.shape[0] == 0:
        return torch.zeros((0, 3), device=obj_v.device)
    disp = torch.cumsum(obj_v[1:] * dt, dim=0) if obj_v.shape[0] > 1 else torch.zeros((0, 3), device=obj_v.device)
    pos = torch.zeros_like(obj_v)
    pos[0] = p0
    if disp.shape[0] > 0:
        pos[1:] = p0 + disp
    return pos


def apply_obj_transform(obj_mesh_path, obj_rot, obj_trans, obj_scale=None, device='cpu'):
    try:
        mesh = trimesh.load_mesh(obj_mesh_path)
        verts = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
        faces = np.asarray(mesh.faces)
        T = obj_trans.shape[0]
        verts_rep = verts.unsqueeze(0).repeat(T, 1, 1)
        rot = obj_rot.float().to(device)
        trans = obj_trans.float().to(device)
        vs = torch.bmm(rot, verts_rep.transpose(1, 2))
        if obj_scale is not None:
            s = obj_scale.float().to(device).unsqueeze(-1).unsqueeze(-1)
            vs = s * vs
        vs = vs + trans.unsqueeze(-1)
        vs = vs.transpose(1, 2)
        return vs, faces
    except Exception as e:
        print(f"Object transform failed: {e}")
        return torch.zeros((1, 1, 3), device=device), np.zeros((1, 3), dtype=np.int64)


def build_body_verts_from_local6d(local6d: torch.Tensor, trans_world: torch.Tensor, bm: BodyModel):
    T = local6d.shape[0]
    mats = transforms.rotation_6d_to_matrix(local6d.view(T, -1, 6).view(T, -1, 6))
    root_axis = transforms.matrix_to_axis_angle(mats[:, 0])
    body_axis = transforms.matrix_to_axis_angle(mats[:, 1:].reshape(T * 21, 3, 3)).view(T, -1)
    out = bm(root_orient=root_axis, pose_body=body_axis, trans=trans_world)
    return out.v, out.Jtr


def global_to_local_rotations(glb: torch.Tensor, parents: torch.Tensor):
    T, J = glb.shape[0], glb.shape[1]
    local = torch.empty_like(glb)
    for j in range(J):
        p = int(parents[j].item())
        if p < 0:
            local[:, j] = glb[:, j]
        else:
            local[:, j] = glb[:, p].transpose(1, 2).matmul(glb[:, j])
    return local


def model_predict_v2(model, x, v_init, p_init, obj_imu=None, obj_v_init=None):
    model.eval()

    if isinstance(x, torch.Tensor):
        if x.dim() == 3:
            xs = x.unsqueeze(0)
        elif x.dim() == 4:
            xs = x
        else:
            raise ValueError('Unsupported IMU tensor shape')
    else:
        xs = torch.stack(x, dim=0)

    xs = xs.contiguous()
    device = xs.device
    dtype = xs.dtype
    B, T, _, _ = xs.shape

    def to_batched_tensor(tensor, target_dim, default_shape):
        if tensor is None:
            return torch.zeros(default_shape, device=device, dtype=dtype)
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.to(device=device, dtype=dtype)
            if tensor.dim() == target_dim - 1:
                tensor = tensor.unsqueeze(0)
            return tensor
        stacked = torch.stack(tensor, dim=0).to(device=device, dtype=dtype)
        if stacked.dim() == target_dim - 1:
            stacked = stacked.unsqueeze(0)
        return stacked

    obj_imu_tensor = to_batched_tensor(obj_imu, 3, (B, T, 12))
    v_init_tensor = to_batched_tensor(v_init, 3, (B, 6, 3))
    p_init_tensor = to_batched_tensor(p_init, 3, (B, 11, 6))

    if obj_v_init is None:
        obj_v_init_tensor = torch.zeros(B, 3, device=device, dtype=dtype)
    else:
        obj_v_init_tensor = to_batched_tensor(obj_v_init, 2, (B, 3))

    with torch.no_grad():
        outputs = model(xs, v_init_tensor, p_init_tensor, obj_imu_tensor, obj_v_init_tensor)

    v_pred, glb_p_pred, obj_v_pred, contact_pred, root_vel_local_pred, root_vel_pred, root_trans_pred = outputs

    glb_full_pose_xsens_list = []
    glb_full_pose_smpl_list = []
    obj_v_seq_list = []
    contact_prob_list = []
    fused_trans_list = []

    for i in range(B):
        pose = glb_p_pred[i].view(T, 11, 6)[:, [4, 5, 6, 7, 8, 9, 10, 0, 2, 1, 3]]
        orientation = xs[i, :, :, :9].view(T, 6, 3, 3)
        glb_full_pose_xsens = model._reduced_glb_6d_to_full_glb_mat_xsens(pose.detach().cpu(), orientation.detach().cpu())
        glb_full_pose_smpl = model._glb_mat_xsens_to_glb_mat_smpl(glb_full_pose_xsens)

        obj_v_seq = obj_v_pred[i].detach().cpu()
        contact_seq = torch.sigmoid(contact_pred[i]).detach().cpu()
        fused_vel_seq = root_vel_pred[i].detach().cpu()
        fps = getattr(model, 'fps', 30.0)
        fused_trans_seq = model.velocity_to_root_position(fused_vel_seq, fps)

        glb_full_pose_xsens_list.append(glb_full_pose_xsens)
        glb_full_pose_smpl_list.append(glb_full_pose_smpl)
        obj_v_seq_list.append(obj_v_seq)
        contact_prob_list.append(contact_seq)
        fused_trans_list.append(fused_trans_seq)

    if B == 1:
        return (
            glb_full_pose_xsens_list[0],
            glb_full_pose_smpl_list[0],
            obj_v_seq_list[0],
            contact_prob_list[0],
            fused_trans_list[0],
        )

    return (
        glb_full_pose_xsens_list,
        glb_full_pose_smpl_list,
        obj_v_seq_list,
        contact_prob_list,
        fused_trans_list,
    )


def main():
    parser = argparse.ArgumentParser(description="DynaIP visualization (human + object) with PoserWithObjectAndTransV2")
    parser.add_argument('--preprocessed_dir', type=str,
                        default='/mnt/d/a_WORK/Projects/PhD/tasks/EgoIMU/processed_data_1014/test',
                        help='Directory containing raw .pt files')
    parser.add_argument('--checkpoint', type=str,
                        default='weights/trans_obj2/best_val.pth',
                        help='Checkpoint for PoserWithObjectAndTransV2')
    parser.add_argument('--bm_path', type=str,
                        default='/mnt/d/a_WORK/Projects/PhD/datasets/smpl_models/smplh/male/model.npz',
                        help='SMPLH model npz path')
    parser.add_argument('--obj_geo_root', type=str,
                        default='/mnt/d/a_WORK/Projects/PhD/datasets/OMOMO/captured_objects',
                        help='Object mesh directory')
    parser.add_argument('--fps', type=float, default=30.0, help='Sequence FPS (default 30)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of sequences to visualize')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = PoserWithObjectAndTransV2(body_model_path=args.bm_path, fps=args.fps).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=True)
    model.eval()

    bm = BodyModel(bm_fname=args.bm_path, num_betas=16).to(device)

    data_list = []
    count = 0
    for fpath, bundle in read_preprocessed_sequences(args.preprocessed_dir, fps=args.fps):
        if args.limit is not None and count >= args.limit:
            break
        raw = bundle.get('raw', bundle.get('processed', {}))
        if 'position_global_full_gt_world' not in raw or 'rotation_global' not in raw:
            print(f"Skipping {os.path.basename(fpath)} due to missing keys.")
            continue
        data_list.append((os.path.basename(fpath), bundle))
        count += 1

    if len(data_list) == 0:
        print("No valid sequences found.")
        return

    class InteractiveViewer(Viewer):
        def __init__(self, data_list, model, bm, obj_geo_root, fps, **kwargs):
            super().__init__(**kwargs)
            self.data_list = data_list
            self.current_index = 0
            self.model = model
            self.bm = bm
            self.obj_geo_root = obj_geo_root
            self.fps = fps
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._render_current()

        def _clear_previous(self):
            try:
                nodes_to_remove = [n for n in self.scene.collect_nodes()
                                   if hasattr(n, 'name') and n.name is not None and
                                   (n.name.startswith('GT-') or n.name.startswith('Pred-'))]
                for n in nodes_to_remove:
                    try:
                        self.scene.remove(n)
                    except Exception:
                        pass
            except Exception as e:
                print(f"Scene clear error: {e}")

        def _render_current(self):
            self._clear_previous()
            name, bundle = self.data_list[self.current_index]
            print(f"Visualizing: {name} (index {self.current_index}/{len(self.data_list)-1})")
            device = self.device
            processed = bundle['processed']
            meta = bundle['meta']
            raw = bundle.get('raw', {})

            fps_used = float(meta.get('fps', self.fps))
            trim_start = int(meta.get('trim_start', 0))
            trim_end = int(meta.get('trim_end', trim_start + processed['imu']['imu'].shape[0]))

            imu = processed['imu']['imu'].float().to(device)
            T = imu.shape[0]

            pos_glb_raw = raw.get('position_global_full_gt_world')
            if pos_glb_raw is not None:
                pos_glb = pos_glb_raw.float()[trim_start:trim_end].to(device)
            else:
                pos_glb = torch.zeros(T, 24, 3, device=device)

            rot_glb_raw = raw.get('rotation_global')
            if rot_glb_raw is not None:
                rot_glb = rot_glb_raw.float()[trim_start:trim_end].to(device)
            else:
                rot_glb = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(T, 24, 1, 1)

            has_object = ('obj_trans' in raw) and ('obj_rot' in raw)
            if has_object:
                obj_trans = raw['obj_trans'].float()[trim_start:trim_end].to(device).view(T, 3)
                obj_rot = raw['obj_rot'].float()[trim_start:trim_end].to(device).view(T, 3, 3)
            else:
                obj_trans = torch.zeros(T, 3, device=device)
                obj_rot = torch.eye(3, device=device).unsqueeze(0).repeat(T, 1, 1)

            obj_scale = raw.get('obj_scale', None)
            if isinstance(obj_scale, torch.Tensor):
                obj_scale = obj_scale.float()[trim_start:trim_end].to(device)
            else:
                obj_scale = None

            obj_name = raw.get('obj_name', 'unknown') if has_object else None

            obj_imu = processed['imu'].get('obj_imu', torch.zeros(T, 12)).float().to(device)
            velocity_all = processed['joint']['velocity'].float().to(device)
            vel_sel = velocity_all[:, VEL_SELECTION.to(device)]
            v_init = vel_sel[:1]

            orient_full = processed['joint']['orientation'].float().view(T, 16, 6).to(device)
            glb_pose_11 = orient_full[:, POSE_SELECTION.to(device)]
            p_init = glb_pose_11[:1]

            obj_vel = processed.get('object', {}).get('velocity', torch.zeros(T, 3))
            obj_vel = obj_vel.float().to(device)
            obj_v_init = obj_vel[:1]

            glb_full_pose_smpl, obj_v_pred_seq = None, None
            with torch.no_grad():
                pred_out = model_predict_v2(self.model, imu, v_init, p_init, obj_imu, obj_v_init)
            if isinstance(pred_out, tuple) and len(pred_out) == 5:
                glb_full_pose_smpl, obj_v_pred_seq, contact_seq, root_trans_pred = pred_out[1], pred_out[2], pred_out[3], pred_out[4]
            else:
                raise RuntimeError('Unexpected output from model_predict_v2')

            if isinstance(glb_full_pose_smpl, list):
                glb_full_pose_smpl = glb_full_pose_smpl[0]
            if isinstance(obj_v_pred_seq, list):
                obj_v_pred_seq = obj_v_pred_seq[0]
            if isinstance(root_trans_pred, list):
                root_trans_pred = root_trans_pred[0]

            glb_full_pose_smpl = glb_full_pose_smpl.to(device)
            obj_v_pred_seq = obj_v_pred_seq.to(device)
            obj_pos_pred = integrate_object_position(obj_v_pred_seq, obj_trans[0], fps=fps_used).to(device)

            if 'rotation_local_full_gt_list' in raw:
                local6d = raw['rotation_local_full_gt_list'].float()[trim_start:trim_end].to(device)
                verts_gt, _ = build_body_verts_from_local6d(local6d, pos_glb[:, 0, :], self.bm)
            else:
                print("rotation_local_full_gt_list missing; skip GT human mesh for this file")
                verts_gt = None

            root_trans_pred = root_trans_pred.to(device)
            parents = torch.tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21], device=device)
            local_pred = global_to_local_rotations(glb_full_pose_smpl.to(device), parents)
            root_axis_pred = transforms.matrix_to_axis_angle(local_pred[:, 0])
            body_axis_pred = transforms.matrix_to_axis_angle(local_pred[:, 1:22].reshape(T * 21, 3, 3)).view(T, -1)
            root_offset = pos_glb[0, 0, :].unsqueeze(0)
            root_trans_full = root_trans_pred.to(device) + root_offset
            body_pose_pred = self.bm(root_orient=root_axis_pred, pose_body=body_axis_pred, trans=root_trans_full)
            verts_pred = body_pose_pred.v

            obj_verts_gt, obj_faces = None, None
            obj_verts_pred = None
            obj_mesh_path = os.path.join(self.obj_geo_root, f"{obj_name}_cleaned_simplified.obj") if obj_name else None
            if obj_name and obj_mesh_path and os.path.exists(obj_mesh_path):
                obj_verts_gt, obj_faces = apply_obj_transform(obj_mesh_path, obj_rot, obj_trans, obj_scale, device=device)
                obj_verts_pred, _ = apply_obj_transform(obj_mesh_path, obj_rot, obj_pos_pred, obj_scale, device=device)

            if verts_gt is not None:
                verts_gt = torch.matmul(verts_gt, R_yup.T.to(device))
            if verts_pred is not None:
                verts_pred = torch.matmul(verts_pred, R_yup.T.to(device))
            if obj_verts_gt is not None:
                obj_verts_gt = torch.matmul(obj_verts_gt, R_yup.T.to(device))
            if obj_verts_pred is not None:
                obj_verts_pred = torch.matmul(obj_verts_pred, R_yup.T.to(device))

            if verts_gt is not None:
                m_gt = Meshes(verts_gt.cpu().numpy(), self.bm.f.cpu().numpy() if isinstance(self.bm.f, torch.Tensor) else self.bm.f,
                              name="GT-Human", color=(0.1, 0.8, 0.3, 0.8), gui_affine=False, is_selectable=False)
                self.scene.add(m_gt)
            if verts_pred is not None:
                m_pred = Meshes(verts_pred.cpu().numpy(), self.bm.f.cpu().numpy() if isinstance(self.bm.f, torch.Tensor) else self.bm.f,
                                name="Pred-Human", color=(0.9, 0.2, 0.2, 0.8), gui_affine=False, is_selectable=False)
                self.scene.add(m_pred)
            if obj_verts_gt is not None and obj_faces is not None:
                m_obj_gt = Meshes(obj_verts_gt.cpu().numpy(), obj_faces,
                                  name=f"GT-{obj_name}", color=(0.1, 0.8, 0.3, 0.8), gui_affine=False, is_selectable=False)
                self.scene.add(m_obj_gt)
            if obj_verts_pred is not None and obj_faces is not None:
                m_obj_pred = Meshes(obj_verts_pred.cpu().numpy(), obj_faces,
                                    name=f"Pred-{obj_name}", color=(0.9, 0.2, 0.2, 0.8), gui_affine=False, is_selectable=False)
                self.scene.add(m_obj_pred)

            self.title = f"DynaIP-V2 | {name} ({self.current_index+1}/{len(self.data_list)})  [q/e: prev/next]"

        def key_event(self, key, action, modifiers):
            super().key_event(key, action, modifiers)
            io = imgui.get_io()
            if self.render_gui and (io.want_capture_keyboard or io.want_text_input):
                return
            is_press = action == self.wnd.keys.ACTION_PRESS
            if not is_press:
                return
            if key == self.wnd.keys.Q:
                if self.current_index > 0:
                    self.current_index -= 1
                    self._render_current()
                else:
                    print("Already at the first sequence.")
            elif key == self.wnd.keys.E:
                if self.current_index < len(self.data_list) - 1:
                    self.current_index += 1
                    self._render_current()
                else:
                    print("Already at the last sequence.")

    print("Launching interactive viewer... (q/e to switch sequence)")
    viewer = InteractiveViewer(data_list=data_list, model=model, bm=bm,
                               obj_geo_root=args.obj_geo_root, fps=args.fps,
                               window_size=(1600, 900))
    viewer.run()


if __name__ == "__main__":
    main()
