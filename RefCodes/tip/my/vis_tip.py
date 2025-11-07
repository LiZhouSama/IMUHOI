import os
import glob
import argparse
import numpy as np
import torch
import pytorch3d.transforms as transforms

from aitviewer.viewer import Viewer
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.spheres import Spheres

from human_body_prior.body_model.body_model import BodyModel

from my.model_tip_with_object import TIPWithObject, TIPWithObjectConfig
from my.dataset_omomo_tip import _second_diff_acc, _rotmat_to_6d


R_YUP = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)


def list_sequences(preprocessed_dir: str):
    files = sorted(glob.glob(os.path.join(preprocessed_dir, "*.pt")))
    return files


def load_seq(path: str, device):
    d = torch.load(path, map_location='cpu')
    # Mandatory keys
    pos_glb = d['position_global_full_gt_world'].float().to(device)
    rot_glb = d['rotation_global'].float().to(device)
    motion6d = d['rotation_local_full_gt_list'].float().to(device)  # [T, 22*6]
    has_object = ('obj_trans' in d) and ('obj_rot' in d)
    obj = None
    if has_object:
        obj = {
            'name': d.get('obj_name', 'unknown'),
            'trans': d['obj_trans'].float().to(device).view(-1, 3),
            'rot': d['obj_rot'].float().to(device),  # [T,3,3] or [T,6]
            'scale': d.get('obj_scale', None)
        }
        if isinstance(obj['scale'], torch.Tensor):
            obj['scale'] = obj['scale'].float().to(device)
    return d, pos_glb, rot_glb, motion6d, obj


def build_tip_inputs_from_seq(pos_glb, rot_glb, motion6d, obj=None, fps: float = 30.0, acc_scale: float = 1.0):
    """
    Build model inputs x_imu and x_s from a full sequence using the same normalization as training.
    Returns x_imu [T-1, C_imu], x_s [T-1, state_dim], plus helpers.
    """
    device = pos_glb.device
    T = pos_glb.shape[0]
    root_pos0 = pos_glb[0, 0]
    root_rot0 = rot_glb[0, 0]  # [3,3]

    # IMU joints: [root, lwrist(20), rwrist(21), lankle(7), rankle(8), head(15)]
    imu_joint_idx = [0, 20, 21, 7, 8, 15]
    imu_feats = []
    for j in imu_joint_idx:
        pj = pos_glb[:, j]
        Rj = rot_glb[:, j]
        acc = _second_diff_acc(pj, fps)
        acc_n = (acc @ root_rot0.transpose(0, 1)) / acc_scale
        Rj_n = torch.einsum('ij,tjk->tik', root_rot0.transpose(0, 1), Rj)
        ori6d = _rotmat_to_6d(Rj_n)
        imu_feats.append(torch.cat([acc_n, ori6d], dim=-1))
    # Object IMU as the 7th sensor (acc + ori6d)
    if obj is not None and obj.get('trans', None) is not None:
        obj_t = obj['trans']  # [T,3] world
        obj_acc = _second_diff_acc(obj_t, fps)
        obj_acc_n = (obj_acc @ root_rot0.transpose(0, 1)) / acc_scale
        # rotation
        obj_rot = obj.get('rot', None)
        if obj_rot is not None:
            if obj_rot.dim() == 3 and obj_rot.shape[-1] == 6:
                # assume already in root0? conservatively convert to mat and normalize
                obj_R = transforms.rotation_6d_to_matrix(obj_rot)
                obj_R_n = torch.einsum('ij,tjk->tik', root_rot0.transpose(0, 1), obj_R)
                obj_ori6d = _rotmat_to_6d(obj_R_n)
            elif obj_rot.dim() == 3 and obj_rot.shape[-2:] == (3, 3):
                obj_R_n = torch.einsum('ij,tjk->tik', root_rot0.transpose(0, 1), obj_rot)
                obj_ori6d = _rotmat_to_6d(obj_R_n)
            else:
                obj_ori6d = torch.zeros(pos_glb.shape[0], 6, device=pos_glb.device)
        else:
            obj_ori6d = torch.zeros(pos_glb.shape[0], 6, device=pos_glb.device)
        imu_feats.append(torch.cat([obj_acc_n, obj_ori6d], dim=-1))
    else:
        imu_feats.append(torch.zeros(pos_glb.shape[0], 9, device=pos_glb.device))

    imu_all = torch.cat(imu_feats, dim=-1)  # [T, 7*9]

    # State S = [18*6 human rot6d, root target (vel), obj_trans(root0)]
    # Map 22->18 joints by dropping [10,11,20,21]
    tip_18_from_22_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    chunks22 = motion6d.view(T, 22, 6)
    human18 = chunks22[:, tip_18_from_22_idx, :].reshape(T, 18 * 6)
    # root pos normalized then convert to velocity
    root_pos_n = (pos_glb[:, 0] - root_pos0) @ root_rot0.transpose(0, 1)
    root_vel = torch.zeros_like(root_pos_n)
    if T > 1:
        root_vel[1:] = (root_pos_n[1:] - root_pos_n[:-1]) * fps
    # no object here; will append later when known
    S_base = human18
    # return per-frame windows (T-1): x=0..T-2, y targets correspond to 1..T-1
    return imu_all, S_base, root_vel, root_pos0, root_rot0


def apply_obj_transform(obj_mesh_path, obj_rot, obj_trans, obj_scale=None, device='cpu'):
    import trimesh
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


class TIPViewer(Viewer):
    def __init__(self, seq_paths, model, bm, obj_geo_root, fps, device, **kwargs):
        super().__init__(**kwargs)
        self.seq_paths = seq_paths
        self.model = model
        self.bm = bm
        self.obj_geo_root = obj_geo_root
        self.fps = fps
        self.device = device
        self.idx = 0
        self._render_current()

    def _clear_prev(self):
        try:
            nodes = [n for n in self.scene.collect_nodes()
                     if hasattr(n, 'name') and n.name and (n.name.startswith('GT-') or n.name.startswith('Pred-') or n.name.startswith('Obj'))]
            for n in nodes:
                try:
                    self.scene.remove(n)
                except Exception:
                    pass
        except Exception as e:
            print(f"Scene clear error: {e}")

    def _render_current(self):
        self._clear_prev()
        path = self.seq_paths[self.idx]
        print(f"Visualizing: {os.path.basename(path)} [{self.idx+1}/{len(self.seq_paths)}]")
        device = self.device
        seq, pos_glb, rot_glb, motion6d, obj = load_seq(path, device)
        T = pos_glb.shape[0]

        # Build inputs
        imu_all, S_base, root_vel, root_pos0, root_rot0 = build_tip_inputs_from_seq(pos_glb, rot_glb, motion6d, obj=obj, fps=self.fps)

        # Object terms in S
        if obj is not None:
            # obj_trans in root0 frame
            obj_trans = obj['trans']  # [T,3] world frame
            obj_trans_n = (obj_trans - root_pos0) @ root_rot0.transpose(0, 1)
        else:
            obj_trans_n = torch.zeros(T, 3, device=device)

        S_all = torch.cat([S_base, root_vel, obj_trans_n], dim=-1)  # [T, 18*6 + 3 + 3]
        # x/y for model
        x_imu = imu_all[:-1].unsqueeze(0)  # [1, T-1, C]
        x_s = S_all[:-1].unsqueeze(0)

        with torch.no_grad():
            # add Gaussian noise to IMU during inference
            x_imu_noisy = x_imu + torch.randn_like(x_imu) * 0.1
            y_pred = self.model(x_imu_noisy, x_s)[0]  # [T-1, state_dim]

        # Parse predictions
        human18_pred = y_pred[:, :18*6].reshape(-1, 18, 6)
        root_target_pred = y_pred[:, 18*6:18*6+3]
        obj_vel_pred_n = y_pred[:, -3:]

        # Integrate root velocity to displacement, align to T frames by padding t0
        if root_target_pred.shape[0] > 0:
            dt = 1.0 / float(self.fps)
            root_pos_pred_n = torch.cumsum(root_target_pred * dt, dim=0)
            root_pos_pred_n = torch.cat([torch.zeros(1, 3, device=device), root_pos_pred_n], dim=0)  # [T]
        else:
            root_pos_pred_n = torch.zeros(T, 3, device=device)

        # Denormalize to world
        root_pos_pred_w = (root_rot0 @ root_pos_pred_n.unsqueeze(-1)).squeeze(-1) + root_pos0

        # Object position via integration of predicted velocity
        if obj_vel_pred_n.shape[0] > 0:
            obj_pos_seq = TIPWithObject.integrate_object_position(
                obj_vel_pred_n,
                torch.zeros(3, device=device, dtype=obj_vel_pred_n.dtype),
                fps=self.fps,
            )
            obj_pos_pred_n = torch.cat([torch.zeros(1, 3, device=device, dtype=obj_vel_pred_n.dtype), obj_pos_seq], dim=0)
        else:
            obj_pos_pred_n = torch.zeros(T, 3, device=device, dtype=y_pred.dtype)
        obj_pos_pred_w = (root_rot0 @ obj_pos_pred_n.unsqueeze(-1)).squeeze(-1) + root_pos0

        # Compose 22-joint rotations by inserting predicted 18 into 22 using GT for missing 4
        tip_18_from_22_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19]
        human22_gt = motion6d.reshape(T, 22, 6)
        human22_pred = human22_gt.clone()
        # Align length: predictions cover T-1 next frames (1..T-1)
        for k, j in enumerate(tip_18_from_22_idx):
            human22_pred[1:, j, :] = human18_pred[:, k, :]

        # SMPL forward: GT
        root6d_gt = human22_gt[:, 0, :]
        pose6d_gt = human22_gt[:, 1:, :].reshape(T, 21, 6)
        root_axis_gt = transforms.matrix_to_axis_angle(transforms.rotation_6d_to_matrix(root6d_gt))
        pose_axis_gt = transforms.matrix_to_axis_angle(transforms.rotation_6d_to_matrix(pose6d_gt.reshape(-1, 6)).reshape(T, 21, 3, 3)).reshape(T, 21*3)
        trans_gt = pos_glb[:, 0]
        out_gt = self.bm(root_orient=root_axis_gt, pose_body=pose_axis_gt, trans=trans_gt)
        verts_gt = out_gt.v
        faces_np = self.bm.f.cpu().numpy() if isinstance(self.bm.f, torch.Tensor) else self.bm.f

        # SMPL forward: Pred (use predicted root translation and predicted 22-rot)
        root6d_pd = human22_pred[:, 0, :]
        pose6d_pd = human22_pred[:, 1:, :].reshape(T, 21, 6)
        root_axis_pd = transforms.matrix_to_axis_angle(transforms.rotation_6d_to_matrix(root6d_pd))
        pose_axis_pd = transforms.matrix_to_axis_angle(transforms.rotation_6d_to_matrix(pose6d_pd.reshape(-1, 6)).reshape(T, 21, 3, 3)).reshape(T, 21*3)
        trans_pd = root_pos_pred_w
        out_pd = self.bm(root_orient=root_axis_pd, pose_body=pose_axis_pd, trans=trans_gt)
        verts_pd = out_pd.v

        # Add to scene (apply Y-up)
        verts_gt_y = torch.matmul(verts_gt, R_YUP.T.to(device)) + torch.tensor([1,0,0], device=device)
        verts_pd_y = torch.matmul(verts_pd, R_YUP.T.to(device))
        m_gt = Meshes(verts_gt_y.cpu().numpy(), faces_np, name='GT-Human', color=(0.1, 0.8, 0.3, 0.8), gui_affine=False, is_selectable=False)
        m_pd = Meshes(verts_pd_y.cpu().numpy(), faces_np, name='Pred-Human', color=(0.9, 0.2, 0.2, 0.8), gui_affine=False, is_selectable=False)
        self.scene.add(m_gt)
        self.scene.add(m_pd)

        # Objects
        if obj is not None and obj['name'] is not None:
            obj_rot = obj['rot']
            if obj_rot.dim() == 3 and obj_rot.shape[-1] == 6:
                obj_rot_mat = transforms.rotation_6d_to_matrix(obj_rot)
            elif obj_rot.dim() == 3 and obj_rot.shape[-2:] == (3, 3):
                obj_rot_mat = obj_rot
            else:
                obj_rot_mat = torch.eye(3, device=device).unsqueeze(0).repeat(T, 1, 1)
            # Denormalize GT object
            obj_trans_gt_w = obj['trans']
            obj_rot_w = obj_rot_mat
            obj_verts_gt, obj_faces = apply_obj_transform(os.path.join(self.obj_geo_root, f"{obj['name']}_cleaned_simplified.obj"), obj_rot_w, obj_trans_gt_w, obj_scale=obj['scale'], device=device)
            # Pred object (use GT rotation + predicted translation)
            obj_verts_pd, _ = apply_obj_transform(os.path.join(self.obj_geo_root, f"{obj['name']}_cleaned_simplified.obj"), obj_rot_w, obj_pos_pred_w, obj_scale=obj['scale'], device=device)

            obj_verts_gt_y = torch.matmul(obj_verts_gt, R_YUP.T.to(device))
            obj_verts_pd_y = torch.matmul(obj_verts_pd, R_YUP.T.to(device))
            m_obj_gt = Meshes(obj_verts_gt_y.cpu().numpy(), obj_faces, name=f"Obj-GT-{obj['name']}", color=(0.1, 0.8, 0.3, 0.8), gui_affine=False, is_selectable=False)
            m_obj_pd = Meshes(obj_verts_pd_y.cpu().numpy(), obj_faces, name=f"Obj-Pred-{obj['name']}", color=(0.9, 0.2, 0.2, 0.8), gui_affine=False, is_selectable=False)
            self.scene.add(m_obj_gt)
            self.scene.add(m_obj_pd)

        self.title = f"TIP Viewer | {os.path.basename(path)} ({self.idx+1}/{len(self.seq_paths)}) [q/e for prev/next]"

    def key_event(self, key, action, modifiers):
        super().key_event(key, action, modifiers)
        is_press = action == self.wnd.keys.ACTION_PRESS
        if not is_press:
            return
        if key == self.wnd.keys.Q:
            if self.idx > 0:
                self.idx -= 1
                self._render_current()
        elif key == self.wnd.keys.E:
            if self.idx < len(self.seq_paths) - 1:
                self.idx += 1
                self._render_current()


def main():
    parser = argparse.ArgumentParser(description='Visualize TIP predictions on OMOMO sequences')
    parser.add_argument('--preprocessed_dir', type=str, default='../../process/processed_data_OMOMO/test', help='processed_data_0701/test')
    parser.add_argument('--weights', type=str, default='checkpoints/tip_omomo_obj/best.pt', help='TIP weights path')
    parser.add_argument('--bm_path', type=str, default='../../datasets/smpl_models/smplh/male/model.npz')
    parser.add_argument('--obj_geo_root', type=str, default='../../datasets/OMOMO/captured_objects')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model consistent with training
    # For dataset-specific dims we instantiate a dummy dataset-like logic
    num_imus_total = 7
    state_dim = 18*6 + 3 + 3
    model_cfg = TIPWithObjectConfig(
        num_imus_total=num_imus_total,
        state_dim=state_dim,
        rnn_hid_size=512,
        tf_hid_size=1024,
        tf_in_dim=256,
        n_heads=16,
        tf_layers=4,
        dropout=0.0,
        in_dropout=0.0,
        past_state_dropout=0.8,
        with_acc_sum=False,
        add_object_head=True,
    )
    model = model_cfg.build()
    state = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    # SMPL-H
    bm = BodyModel(bm_fname=args.bm_path, num_betas=16).to(device)

    seq_paths = list_sequences(args.preprocessed_dir)
    if args.limit is not None:
        seq_paths = seq_paths[:args.limit]
    if not seq_paths:
        print(f"No .pt sequences found in {args.preprocessed_dir}")
        return

    viewer = TIPViewer(seq_paths, model, bm, args.obj_geo_root, args.fps, device, window_size=(1600, 900))
    viewer.run()


if __name__ == '__main__':
    main()
