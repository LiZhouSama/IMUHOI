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

from utils.data import normalize_imu
from model.model_obj import PoserWithObject


# --- 定义 Z-up 到 Y-up 的旋转矩阵 ---
R_yup = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, -1.0, 0.0]], dtype=torch.float32)


def read_preprocessed_sequences(src_dir: str):
    files = sorted(glob.glob(os.path.join(src_dir, "*.pt")))
    for f in files:
        try:
            d = torch.load(f)
            yield f, d
        except Exception as e:
            print(f"Skip {f}: {e}")


def central_diff(a: torch.Tensor, dt: float) -> torch.Tensor:
    if a.shape[0] <= 1:
        return torch.zeros_like(a)
    vel = torch.zeros_like(a)
    vel[1:-1] = (a[2:] - a[:-2]) / (2.0 * dt)
    vel[0] = (a[1] - a[0]) / dt
    vel[-1] = (a[-1] - a[-2]) / dt
    return vel


def build_imu_from_joints(rotation_global: torch.Tensor,
                          position_global: torch.Tensor,
                          sensor_indices_pos, sensor_indices_rot, fps: float = 60.0) -> torch.Tensor:
    T = rotation_global.shape[0]
    sel_R = rotation_global[:, sensor_indices_rot]  # [T, 6, 3, 3]
    sel_pos = position_global[:, sensor_indices_pos]  # [T, 6, 3]
    dt = 1.0 / fps
    vel = central_diff(sel_pos, dt)
    acc = central_diff(vel, dt)
    # normalize_imu expects (acc [T,6,3], ori [T,6,3,3]) and returns [T,6,12]
    data = normalize_imu(acc.view(T, 6, 3), sel_R.view(T, 6, 3, 3))
    return data


def synthesize_obj_imu(obj_rot: torch.Tensor, obj_trans: torch.Tensor, fps: float = 60.0) -> torch.Tensor:
    T = obj_trans.shape[0]
    dt = 1.0 / fps
    vel = central_diff(obj_trans, dt)
    acc = central_diff(vel, dt)
    return torch.cat([obj_rot.reshape(T, 9), acc], dim=-1)  # [T, 12]


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
    parser.add_argument('--preprocessed_dir', type=str, default='../../process/processed_data_BEHAVE/test', help='Directory of preprocessed .pt (e.g., processed_data_0701/test)')
    parser.add_argument('--checkpoint', type=str, default='weights/obj/best_val.pth', help='Path to trained PoserWithObject checkpoint .pth')
    parser.add_argument('--bm_path', type=str, default='../../smpl_models/smplh/male/model.npz', help='SMPLH model npz path')
    parser.add_argument('--obj_geo_root', type=str, default='../../../datasets/OMOMO/captured_objects', help='Object mesh directory')
    parser.add_argument('--fps', type=float, default=30.0, help='Sequence FPS (OMOMO=30)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of sequences to visualize')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = PoserWithObject().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Load SMPL
    bm = BodyModel(bm_fname=args.bm_path, num_betas=16).to(device)

    # 预加载数据列表
    data_list = []
    count = 0
    for fpath, data in read_preprocessed_sequences(args.preprocessed_dir):
        if args.limit is not None and count >= args.limit:
            break
        # 简单校验所需键是否存在
        if ('position_global_full_gt_world' not in data) or ('rotation_global' not in data):
            print(f"Skipping {os.path.basename(fpath)} due to missing keys.")
            continue
        data_list.append((os.path.basename(fpath), data))
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
            name, data = self.data_list[self.current_index]
            print(f"Visualizing: {name} (index {self.current_index}/{len(self.data_list)-1})")
            device = self.device

            # 原有单序列推理与可视化逻辑（尽量不改动），仅加入 R_yup 和场景清理
            pos_glb = data['position_global_full_gt_world'].float().to(device)  # [T,J,3]
            rot_glb = data['rotation_global'].float().to(device)  # [T,J,3,3]
            T = pos_glb.shape[0]

            sensors_pos = [0, 7, 8, 15, 20, 21]
            sensors_rot = [0, 4, 5, 15, 18, 19]
            imu = build_imu_from_joints(rot_glb, pos_glb, sensors_pos, sensors_rot, fps=self.fps).to(device)

            has_object = ('obj_trans' in data) and ('obj_rot' in data)
            if has_object:
                obj_trans = data['obj_trans'].float().to(device).view(T, 3)
                obj_rot = data['obj_rot'].float().to(device).view(T, 3, 3)
                obj_imu = synthesize_obj_imu(obj_rot, obj_trans, fps=self.fps).to(device)
            else:
                obj_trans = torch.zeros(T, 3, device=device)
                obj_rot = torch.eye(3, device=device).unsqueeze(0).repeat(T, 1, 1)
                obj_imu = torch.zeros(T, 12, device=device)

            # 与训练一致的速度计算：去水平→差分→相对根→根系旋转
            vel_mask = torch.tensor([0, 15, 20, 21, 7, 8], device=device)
            pos_for_vel = pos_glb.clone()
            pos_for_vel[:, :, 0] = pos_for_vel[:, :, 0] - pos_for_vel[:, :1, 0]
            pos_for_vel[:, :, 2] = pos_for_vel[:, :, 2] - pos_for_vel[:, :1, 2]
            vel_w = (pos_for_vel[1:] - pos_for_vel[:-1]) * self.fps
            vel_w = torch.cat((vel_w[:1], vel_w), dim=0)
            root_vel = vel_w[:, :1]
            rel_vel = torch.cat((root_vel, vel_w[:, 1:] - root_vel), dim=1)
            root_R = rot_glb[:, 0]
            vel_root = rel_vel.bmm(root_R)
            vel_sel = vel_root[:, vel_mask]
            v_init = vel_sel[:1]

            # 与训练一致的姿态选择：先取 DIP 16 关节，再选 11 个
            DIP_MASK_16 = [1, 2, 3, 4, 5, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
            sel_11_from_16 = [0, 1, 2, 5, 6, 7, 8, 9, 10, 12, 13]
            # 裁掉越界（以防非 SMPL 序列）
            DIP_MASK_16 = [i for i in DIP_MASK_16 if i < rot_glb.shape[1]]
            r16 = rot_glb[:, DIP_MASK_16]  # [T,16,3,3]
            r16_root = r16[:, :1].transpose(2, 3).matmul(r16)
            r16_6d = r16_root[:, :, :, :2].transpose(2, 3).reshape(T, -1, 6)
            # 依据 16 的当前长度重新过滤 11 的索引
            valid_11 = [i for i in sel_11_from_16 if i < r16_6d.shape[1]]
            pose11_6d = r16_6d[:, valid_11]
            p_init = pose11_6d[:1]

            dt = 1.0 / self.fps
            obj_v = central_diff(obj_trans, dt)
            obj_v_init = obj_v[:1].to(device)

            with torch.no_grad():
                _, glb_full_pose_smpl, obj_v_pred_seq = self.model.predict(imu, v_init, p_init, obj_imu, obj_v_init)
                parents = torch.tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21], device=device)
                local_pred = global_to_local_rotations(glb_full_pose_smpl, parents).to(device)
                root_axis_pred = transforms.matrix_to_axis_angle(local_pred[:, 0])
                body_axis_pred = transforms.matrix_to_axis_angle(local_pred[:, 1:22].reshape(T * 21, 3, 3)).view(T, -1)
                body_pose_pred = self.bm(root_orient=root_axis_pred, pose_body=body_axis_pred, trans=pos_glb[:, 0, :])
                verts_pred = body_pose_pred.v

            obj_pos_pred = integrate_object_position(obj_v_pred_seq.to(device), obj_trans[0], fps=self.fps)

            if 'rotation_local_full_gt_list' in data:
                local6d = data['rotation_local_full_gt_list'].float().to(device)
                verts_gt, _ = build_body_verts_from_local6d(local6d, pos_glb[:, 0, :], self.bm)
            else:
                print("rotation_local_full_gt_list missing; skip GT human mesh for this file")
                verts_gt = None

            obj_name = data.get('obj_name', 'unknown') if has_object else None
            obj_scale = data.get('obj_scale', None)
            if isinstance(obj_scale, torch.Tensor):
                obj_scale = obj_scale.to(device)

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


