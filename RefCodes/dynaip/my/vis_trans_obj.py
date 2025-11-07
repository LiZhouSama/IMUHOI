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

from model.dataset_trans_obj import load_dynaip_sequence, _REDUCED_POSE_NAMES, _VEL_SELECTION_INDICES
from model.model_trans_obj import PoserWithObjectAndTrans


# --- 定义 Z-up 到 Y-up 的旋转矩阵 ---
# R_yup = torch.tensor([[1.0, 0.0, 0.0],
#                       [0.0, 0.0, 1.0],
#                       [0.0, -1.0, 0.0]], dtype=torch.float32)
R_yup = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)

DEFAULT_TRIM_FRAMES = 6

def read_preprocessed_sequences(src_dir: str, fps: float, trim_frames: int = 6):
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
    disp = torch.cumprod(torch.ones((obj_v.shape[0],), device=obj_v.device), dim=0)  # dummy to keep device
    disp = torch.cumsum(obj_v[1:] * dt, dim=0) if obj_v.shape[0] > 1 else torch.zeros((0, 3), device=obj_v.device)
    pos = torch.zeros_like(obj_v)
    pos[0] = p0
    if disp.shape[0] > 0:
        pos[1:] = p0 + disp
    return pos


def apply_obj_transform(obj_mesh_path, obj_rot, obj_trans, obj_scale=None, device='cpu'):
    try:
        mesh = trimesh.load_mesh(obj_mesh_path)
        verts = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)  # [Nv,3]
        faces = np.asarray(mesh.faces)
        T = obj_trans.shape[0]
        verts_rep = verts.unsqueeze(0).repeat(T, 1, 1)  # [T,Nv,3]
        rot = obj_rot.float().to(device)  # [T,3,3]
        trans = obj_trans.float().to(device)  # [T,3]
        vs = torch.bmm(rot, verts_rep.transpose(1, 2))  # [T,3,Nv]
        if obj_scale is not None:
            s = obj_scale.float().to(device).unsqueeze(-1).unsqueeze(-1)  # [T,1,1]
            vs = s * vs
        vs = vs + trans.unsqueeze(-1)
        vs = vs.transpose(1, 2)  # [T,Nv,3]
        return vs, faces
    except Exception as e:
        print(f"Object transform failed: {e}")
        return torch.zeros((1, 1, 3), device=device), np.zeros((1, 3), dtype=np.int64)


def build_body_verts_from_local6d(local6d: torch.Tensor, trans_world: torch.Tensor, bm: BodyModel):
    # local6d: [T, 22*6] (root + 21 body joints), trans_world: [T,3]
    T = local6d.shape[0]
    mats = transforms.rotation_6d_to_matrix(local6d.view(T, -1, 6).view(T, -1, 6))  # [T,22,3,3]
    root_axis = transforms.matrix_to_axis_angle(mats[:, 0])  # [T,3]
    body_axis = transforms.matrix_to_axis_angle(mats[:, 1:].reshape(T * 21, 3, 3)).view(T, -1)  # [T,63]
    out = bm(root_orient=root_axis, pose_body=body_axis, trans=trans_world)
    return out.v, out.Jtr


def global_to_local_rotations(glb: torch.Tensor, parents: torch.Tensor):
    """glb: [T, J, 3, 3]; parents: [J] with -1 for root"""
    T, J = glb.shape[0], glb.shape[1]
    local = torch.empty_like(glb)
    for j in range(J):
        p = int(parents[j].item())
        if p < 0:
            local[:, j] = glb[:, j]
        else:
            # glb[:, p] and glb[:, j] are [T, 3, 3]; transpose last two dims -> (1, 2)
            local[:, j] = glb[:, p].transpose(1, 2).matmul(glb[:, j])
    return local


def main():
    parser = argparse.ArgumentParser(description="DynaIP visualization (human + object) with aitviewer")
    parser.add_argument('--preprocessed_dir', type=str, default='/mnt/d/a_WORK/Projects/PhD/tasks/EgoIMU/processed_data_1014/test', help='Directory of preprocessed .pt (e.g., processed_data_0701/test)')
    parser.add_argument('--checkpoint', type=str, default='weights/trans_obj/best_val.pth', help='Path to trained PoserWithObjectAndTrans checkpoint .pth')
    parser.add_argument('--bm_path', type=str, default='/mnt/d/a_WORK/Projects/PhD/datasets/smpl_models/smplh/male/model.npz', help='SMPLH model npz path')
    parser.add_argument('--obj_geo_root', type=str, default='/mnt/d/a_WORK/Projects/PhD/datasets/OMOMO/captured_objects', help='Object mesh directory')
    parser.add_argument('--fps', type=float, default=30.0, help='Sequence FPS (OMOMO=30)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of sequences to visualize')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = PoserWithObjectAndTrans(body_model_path=args.bm_path, fps=args.fps).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as load_err:
        keys_to_drop = [k for k in state.keys() if k.startswith('tran_b1.') or k.startswith('tran_b2.')]
        if keys_to_drop:
            print('State_dict mismatch detected. Dropping translation-branch weights:', keys_to_drop)
            filtered = {k: v for k, v in state.items() if k not in keys_to_drop}
            model.load_state_dict(filtered, strict=False)
        else:
            raise load_err
    model.eval()

    # Load SMPL
    bm = BodyModel(bm_fname=args.bm_path, num_betas=16).to(device)

    # 预加载数据列表
    data_list = []
    count = 0
    for fpath, bundle in read_preprocessed_sequences(args.preprocessed_dir, fps=args.fps, trim_frames=DEFAULT_TRIM_FRAMES):
        if args.limit is not None and count >= args.limit:
            break
        data = bundle.get('raw', bundle.get('processed', {}))
        if ('position_global_full_gt_world' not in data) or ('rotation_global' not in data):
            print(f"Skipping {os.path.basename(fpath)} due to missing keys.")
            continue
        data_list.append((os.path.basename(fpath), bundle))
        count += 1

    if len(data_list) == 0:
        print("No valid preprocessed sequences found.")
        return

    # 交互式 Viewer：支持 q/e 切换
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

            # 原有单序列推理与可视化逻辑（尽量不改动），仅加入 R_yup 和场景清理
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
            
            # 新增：从 processed['joint']['velocity'] 提取初始速度
            velocity_all = processed['joint']['velocity'].float().to(device)
            vel_sel = velocity_all[:, _VEL_SELECTION_INDICES]
            v_init = vel_sel[:1]

            # 修改：使用 ori_glb_reduced 而不是 orientation
            ori_glb_reduced_flat = processed['joint']['ori_glb_reduced'].float().to(device)  # [T, 60]
            ori_glb_reduced = ori_glb_reduced_flat.view(T, len(_REDUCED_POSE_NAMES), 6)  # [T, 10, 6]
            p_init = ori_glb_reduced[:1]  # [1, 10, 6]

            obj_vel = processed.get('object', {}).get('velocity', torch.zeros(T, 3))
            obj_vel = obj_vel.float().to(device)
            obj_v_init = obj_vel[:1]  # [1, 3]


            with torch.no_grad():
                pred_out = self.model.predict(imu.unsqueeze(0), v_init, p_init, obj_imu.unsqueeze(0), obj_v_init)
            
            # 新增：处理新的返回格式（4个值）
            glb_full_pose_smpl, obj_v_pred_seq, contact_prob_seq, root_trans_pred = pred_out
            
            # 确保是正确的设备和格式
            glb_full_pose_smpl = glb_full_pose_smpl.to(device)
            root_trans_pred = root_trans_pred.to(device)
            obj_v_pred_seq = obj_v_pred_seq.to(device)
            
            parents = torch.tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21], device=device)
            local_pred = global_to_local_rotations(glb_full_pose_smpl, parents)
            root_axis_pred = transforms.matrix_to_axis_angle(local_pred[:, 0])
            body_axis_pred = transforms.matrix_to_axis_angle(local_pred[:, 1:22].reshape(T * 21, 3, 3)).view(T, -1)
            root_offset = pos_glb[0, 0, :].unsqueeze(0)
            root_trans_full = root_trans_pred + root_offset
            body_pose_pred = self.bm(root_orient=root_axis_pred, pose_body=body_axis_pred, trans=root_trans_full)
            verts_pred = body_pose_pred.v
            obj_pos_pred = integrate_object_position(obj_v_pred_seq, obj_trans[0], fps=fps_used).to(device)

            if 'rotation_local_full_gt_list' in raw:
                local6d = raw['rotation_local_full_gt_list'].float()[trim_start:trim_end].to(device)
                verts_gt, _ = build_body_verts_from_local6d(local6d, pos_glb[:, 0, :], self.bm)
            else:
                print("rotation_local_full_gt_list missing; skip GT human mesh for this file")
                verts_gt = None


            obj_verts_gt, obj_faces = None, None
            obj_verts_pred = None
            obj_mesh_path = os.path.join(self.obj_geo_root, f"{obj_name}_cleaned_simplified.obj") if obj_name else None
            if obj_name and obj_mesh_path and os.path.exists(obj_mesh_path):
                obj_verts_gt, obj_faces = apply_obj_transform(obj_mesh_path, obj_rot, obj_trans, obj_scale, device=device)
                obj_verts_pred, _ = apply_obj_transform(obj_mesh_path, obj_rot, obj_pos_pred, obj_scale, device=device)

            # 应用 Y-up 旋转
            if verts_gt is not None:
                verts_gt = torch.matmul(verts_gt, R_yup.T.to(device))
            if 'verts_pred' in locals() and verts_pred is not None:
                verts_pred = torch.matmul(verts_pred, R_yup.T.to(device))
            if obj_verts_gt is not None:
                obj_verts_gt = torch.matmul(obj_verts_gt, R_yup.T.to(device))
            if obj_verts_pred is not None:
                obj_verts_pred = torch.matmul(obj_verts_pred, R_yup.T.to(device))

            # 添加到场景
            if verts_gt is not None:
                m_gt = Meshes(verts_gt.cpu().numpy(), self.bm.f.cpu().numpy() if isinstance(self.bm.f, torch.Tensor) else self.bm.f,
                              name="GT-Human", color=(0.1, 0.8, 0.3, 0.8), gui_affine=False, is_selectable=False)
                self.scene.add(m_gt)
            if 'verts_pred' in locals() and verts_pred is not None:
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

            self.title = f"DynaIP | {name} ({self.current_index+1}/{len(self.data_list)})  [q/e: prev/next]"

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

    print("Launching interactive viewer... (q/e 切换序列)")
    viewer = InteractiveViewer(data_list=data_list, model=model, bm=bm, obj_geo_root=args.obj_geo_root, fps=args.fps,
                               window_size=(1600, 900))
    viewer.run()


if __name__ == "__main__":
    main()


