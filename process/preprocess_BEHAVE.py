import argparse
import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import trimesh

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.rotation_tools import local2global_pose
import pytorch3d.transforms as transforms

from preprocess import compute_improved_contact_labels, compute_foot_contact_labels

DEFAULT_CHUNK_LEN = 1500      # 视显存/机型可调
DEFAULT_VERTEX_CHUNK = 80000  # 视物体网格大小可调

R_Y_UP = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, -1.0, 0.0],
     [0.0, 0.0, -1.0]],
    dtype=np.float32,
)

def iter_slices(T: int, chunk_len: int):
    for s in range(0, T, chunk_len):
        e = min(s + chunk_len, T)
        yield s, e

def npz_get(npz_file: np.lib.npyio.NpzFile, key: str, default=None):
    return npz_file[key] if key in npz_file.files else default


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
                ).to(device).eval()
        if not self._models:
            raise FileNotFoundError(f"No SMPL-H models found under {support_dir}")
        self._default_gender = "neutral" if "neutral" in self._models else "male"

    @property
    def num_betas(self) -> int:
        return self._num_betas

    def get(self, gender: Optional[str]) -> BodyModel:
        if gender is None:
            return self._models[self._default_gender]
        key = str(gender).lower()
        if key not in self._models:
            return self._models[self._default_gender]
        return self._models[key]


def infer_object_name(seq_name: str, object_names: List[str]) -> str:
    lowered = seq_name.lower()
    for candidate in sorted(object_names, key=len, reverse=True):
        if candidate.lower() in lowered:
            return candidate
    raise ValueError(f"Unable to infer object name from sequence {seq_name}")


_mesh_info_cache: Dict[str, Tuple[str, np.ndarray, np.ndarray]] = {}



def resolve_object_mesh(object_name: str, objects_root: str) -> Tuple[str, np.ndarray, np.ndarray]:
    if object_name in _mesh_info_cache:
        return _mesh_info_cache[object_name]
    explicit_path = None
    ply_candidates = sorted(glob.glob(os.path.join(objects_root, "*.ply")))
    if ply_candidates:
        explicit_path = ply_candidates[0]
    if explicit_path is not None and os.path.exists(explicit_path):
        mesh_path = explicit_path
    else:
        obj_dir = os.path.join(objects_root, object_name)
        if not os.path.isdir(obj_dir):
            raise FileNotFoundError(f"Object directory not found: {obj_dir}")
        ply_candidates = sorted(glob.glob(os.path.join(obj_dir, "*.ply")))
        if ply_candidates:
            mesh_path = ply_candidates[0]
        if not ply_candidates:
            raise FileNotFoundError(f"No obj mesh found under {obj_dir}")
    mesh = trimesh.load(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    centroid = vertices.mean(axis=0).astype(np.float32)
    _mesh_info_cache[object_name] = (mesh_path, centroid, vertices)
    return _mesh_info_cache[object_name]




def pad_betas(betas: np.ndarray, target_dim: int, target_frames: int) -> np.ndarray:
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim == 1:
        betas = betas[None, :]
    if betas.shape[0] == 1 and target_frames > 1:
        betas = np.repeat(betas, repeats=target_frames, axis=0)
    if betas.shape[0] < target_frames:
        betas = np.pad(betas, ((0, target_frames - betas.shape[0]), (0, 0)), mode='edge')
    else:
        betas = betas[:target_frames]
    if betas.shape[1] < target_dim:
        betas = np.concatenate([
            betas,
            np.zeros((betas.shape[0], target_dim - betas.shape[1]), dtype=betas.dtype)
        ], axis=1)
    elif betas.shape[1] > target_dim:
        betas = betas[:, :target_dim]
    return betas


def build_smpl_input(root_orient: np.ndarray, pose_body: np.ndarray, trans: np.ndarray,
                     betas: np.ndarray, device: torch.device) -> Dict[str, torch.Tensor]:
    smpl_input: Dict[str, torch.Tensor] = {
        "root_orient": torch.tensor(root_orient, dtype=torch.float32, device=device),
        "pose_body": torch.tensor(pose_body, dtype=torch.float32, device=device),
        "trans": torch.tensor(trans, dtype=torch.float32, device=device)
    }
    if betas.size > 0:
        smpl_input["betas"] = torch.tensor(betas, dtype=torch.float32, device=device)
    return smpl_input


def compute_rotation_features_chunk(
    root_orient_chunk: np.ndarray, pose_body_chunk: np.ndarray,
    bm: BodyModel, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    # [B, 3] + [B, 63] -> [B, 66]
    bdata_poses = np.concatenate([root_orient_chunk, pose_body_chunk], axis=1).astype(np.float32)
    local_rot_mat = transforms.axis_angle_to_matrix(
        torch.tensor(bdata_poses, dtype=torch.float32, device=device).reshape(-1, 3)
    ).reshape(bdata_poses.shape[0], -1, 3, 3)
    rotation_local_6d = transforms.matrix_to_rotation_6d(local_rot_mat)
    rotation_local_full_gt = rotation_local_6d.reshape(bdata_poses.shape[0], -1)

    kintree_table = bm.kintree_table[0].long()[:22]
    kintree_table[0] = -1
    rotation_global_matrot = local2global_pose(
        local_rot_mat.reshape(local_rot_mat.shape[0], -1, 9), kintree_table
    )
    rotation_global_matrot = rotation_global_matrot.reshape(rotation_global_matrot.shape[0], -1, 3, 3)
    return rotation_local_full_gt.detach().cpu(), rotation_global_matrot.detach().cpu()

def scan_global_floors(
    bm: BodyModel,
    smpl_data, obj_data,
    T: int,
    betas_raw: np.ndarray,
    ground_shift: float,
    device: torch.device,
    frame_chunk: int,
    vertex_chunk: int,
    base_vertices_centered: np.ndarray,
) -> Tuple[float, float]:
    # 人体 floor（min |y|）
    human_floor_abs: Optional[float] = None

    with torch.no_grad():
        for s, e in iter_slices(T, frame_chunk):
            root_orient = smpl_data['poses'][s:e, :3].astype(np.float32)
            pose_body   = smpl_data['poses'][s:e, 3:66].astype(np.float32)
            trans       = smpl_data['trans'][s:e].astype(np.float32)
            trans[:, 1] -= ground_shift

            betas_chunk = pad_betas(betas_raw, bm.num_betas, e - s)
            smpl_input = build_smpl_input(root_orient, pose_body, trans, betas_chunk, device)

            # 你的“去掉默认偏移”的 forward（把 trans 临时置零再加回来）
            total = smpl_input["root_orient"].shape[0]
            floor_chunk: Optional[float] = None

            # 小批量 forward，避免 GPU 爆
            bsz = 256
            for a in range(0, total, bsz):
                b = min(a + bsz, total)
                kwargs = {k: v[a:b] for k, v in smpl_input.items()}
                kwargs_zero = dict(kwargs); kwargs_zero["trans"] = torch.zeros_like(kwargs_zero["trans"])
                out = bm(**kwargs)
                out_zero = bm(**kwargs_zero)

                verts = out.v - out_zero.Jtr[:, :1, :] + kwargs["trans"].unsqueeze(1)  # [B,V,3]
                floor_ab = float(verts[:, :, 1].abs().amin().item())  # 维持你原来的 abs
                floor_chunk = floor_ab if floor_chunk is None else min(floor_chunk, floor_ab)

                del out, out_zero, verts
                if device.type == "cuda": torch.cuda.empty_cache()

            human_floor_abs = floor_chunk if human_floor_abs is None else min(human_floor_abs, floor_chunk)

    # 物体 floor（min |y|），按 vertex_chunk 来扫，避免 T×V 巨张量
    obj_rot = R.from_rotvec(obj_data['angles'][:T].astype(np.float32)).as_matrix().astype(np.float32)
    obj_trans = obj_data['trans'][:T].astype(np.float32)
    obj_trans[:, 1] -= ground_shift

    obj_rot_t   = torch.tensor(obj_rot,   dtype=torch.float32, device=device)   # [T,3,3]
    obj_trans_t = torch.tensor(obj_trans, dtype=torch.float32, device=device)   # [T,3]
    base_v_t    = torch.tensor(base_vertices_centered.astype(np.float32), dtype=torch.float32, device=device)  # [V,3]

    object_floor_abs: Optional[float] = None
    V = base_v_t.shape[0]
    with torch.no_grad():
        for vs in range(0, V, vertex_chunk):
            ve = min(vs + vertex_chunk, V)
            # [T,3,3] x [Vchunk,3] -> [T, Vchunk, 3]
            y = torch.einsum('tij,vj->tvi', obj_rot_t, base_v_t[vs:ve]) + obj_trans_t.unsqueeze(1)
            y_min_abs = float(y[:, :, 1].abs().amin().item())
            object_floor_abs = y_min_abs if object_floor_abs is None else min(object_floor_abs, y_min_abs)
            del y
            if device.type == "cuda": torch.cuda.empty_cache()

    return float(human_floor_abs or 0.0), float(object_floor_abs or 0.0)


def forward_smpl_in_batches(
    bm: BodyModel,
    smpl_input: Dict[str, torch.Tensor],
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward SMPL in manageable batches and return joints plus the global minimum |y| vertex height."""
    total_frames = smpl_input["root_orient"].shape[0]
    joint_chunks: List[torch.Tensor] = []
    min_abs_y_value: Optional[float] = None
    with torch.no_grad():
        for start in range(0, total_frames, batch_size):
            end = min(start + batch_size, total_frames)
            kwargs = {
                key: value[start:end]
                for key, value in smpl_input.items()
            }
            body_out = bm(**kwargs)
            Jtr_watch = body_out.Jtr[:, :22, :].detach().cpu().numpy()
            joints = body_out.Jtr[:, :22, :].detach().cpu()  # 去掉human_body_prior的默认偏移
            verts = body_out.v
            batch_min_abs = float(verts[:, :, 1].abs().min().item())
            if min_abs_y_value is None or batch_min_abs < min_abs_y_value:
                min_abs_y_value = batch_min_abs
            joint_chunks.append(joints)
            del body_out
    position_global = torch.cat(joint_chunks, dim=0)
    min_y = torch.tensor(min_abs_y_value if min_abs_y_value is not None else 0.0, dtype=position_global.dtype)
    return position_global, min_y


def compute_foot_floor_offset(
    bm: BodyModel,
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray,
    device: torch.device,
) -> float:
    """计算整个序列中双脚关节的最低y坐标，用于地面对齐"""
    T = root_orient.shape[0]
    smpl_input = build_smpl_input(root_orient, pose_body, trans, betas, device)
    
    foot_min_y: Optional[float] = None
    with torch.no_grad():
        kwargs = {key: value for key, value in smpl_input.items()}
        body_out = bm(**kwargs)
        joints = body_out.Jtr[:, :22, :]  # [B, 22, 3]
        # 双脚关节: 7=左脚踝, 8=右脚踝, 10=左脚, 11=右脚
        left_ankle = joints[:, 7, 1]   # [B]
        right_ankle = joints[:, 8, 1]  # [B]
        left_foot = joints[:, 10, 1]   # [B]
        right_foot = joints[:, 11, 1]  # [B]
        foot_min_y = torch.min(torch.stack([left_ankle, right_ankle, left_foot, right_foot], dim=0)).item()
    
    return float(foot_min_y or 0.0)


def process_behave_sequence(seq_dir: str, bm_loader: BodyModelLoader, objects_root: str,
                            object_names: List[str], device: torch.device,
                            output_dir: str, frame_trim: Optional[int] = None,
                            smpl_batch: int = 256) -> None:
    seq_name = os.path.basename(seq_dir.rstrip(os.sep))
    smpl_path = os.path.join(seq_dir, "smpl_fit_all.npz")
    obj_path = os.path.join(seq_dir, "object_fit_all.npz")
    if not (os.path.exists(smpl_path) and os.path.exists(obj_path)):
        raise FileNotFoundError(f"Missing expected npz files under {seq_dir}")

    smpl_data = np.load(smpl_path, mmap_mode='r')
    obj_data  = np.load(obj_path,  mmap_mode='r')

    total_frames = smpl_data['poses'].shape[0]
    obj_frames = obj_data['angles'].shape[0]
    T = min(total_frames, obj_frames)
    if frame_trim is not None:
        T = min(T, frame_trim)

    poses = smpl_data['poses'][:T].astype(np.float32)
    trans = smpl_data['trans'][:T].astype(np.float32)
    trans_first_frame = trans[0].copy()
    trans_norm = trans - trans_first_frame

    betas_raw = npz_get(smpl_data, 'betas', np.zeros((1, bm_loader.num_betas), dtype=np.float32))
    betas = pad_betas(betas_raw, bm_loader.num_betas, T)

    gender_raw = npz_get(smpl_data, 'gender', 'neutral')
    gender = str(gender_raw)

    object_angles = obj_data['angles'][:T].astype(np.float32)
    object_trans = obj_data['trans'][:T].astype(np.float32)
    object_name = infer_object_name(seq_name, object_names)

    object_trans_norm = object_trans - trans_first_frame

    mesh_path, centroid, mesh_vertices = resolve_object_mesh(object_name, objects_root)
    bm = bm_loader.get(gender)
    smpl_init_input = {
            "root_orient": torch.zeros_like(torch.from_numpy(poses[:, :3].reshape(-1, 3)).to(device).float()),
            "pose_body": torch.zeros_like(torch.from_numpy(poses[:, 3:66].reshape(-1, 63)).to(device).float()),
            "trans": torch.zeros_like(torch.from_numpy(trans_norm.reshape(-1, 3)).to(device).float())
        }
    Jtr_0 = np.asarray(bm(**smpl_init_input).Jtr[:, 0, :].cpu(), dtype=np.float32)

    r_y_up_torch = torch.from_numpy(R_Y_UP).float().to(device)
    r_y_up_repeat = r_y_up_torch.unsqueeze(0).repeat(T, 1, 1)
    root_ori_t = torch.from_numpy(poses[:, :3]).float().to(device)
    root_ori_mat = transforms.axis_angle_to_matrix(root_ori_t)
    root_ori_t = r_y_up_repeat @ root_ori_mat
    root_ori = transforms.matrix_to_axis_angle(root_ori_t).detach().cpu().numpy().astype(np.float32)
    poses[:, :3] = root_ori
    trans_norm = (trans_norm + Jtr_0) @ R_Y_UP.T - Jtr_0
    object_angles_t = torch.from_numpy(object_angles).float().to(device)
    obj_rot_mat = transforms.axis_angle_to_matrix(object_angles_t)
    obj_rot_mat = torch.matmul(r_y_up_repeat, obj_rot_mat)
    object_angles_t = transforms.matrix_to_axis_angle(obj_rot_mat)
    object_angles = object_angles_t.detach().cpu().numpy().astype(np.float32)
    object_trans_norm = object_trans_norm @ R_Y_UP.T
    
    # ---------- 计算双脚最低点偏移 ----------
    foot_floor_y = compute_foot_floor_offset(
        bm, poses[:, :3], poses[:, 3:66], trans_norm, betas, device
    )
    # 修正 trans 和 object_trans，使双脚最低点对齐到 y=0
    trans_norm[:, 1] -= foot_floor_y
    object_trans_norm[:, 1] -= foot_floor_y

    # ---------- 分块产出+拼接 ----------
    rot_local_list   : List[torch.Tensor] = []
    rot_global_list  : List[torch.Tensor] = []
    pos_global_list  : List[torch.Tensor] = []
    lfoot_list       : List[torch.Tensor] = []
    rfoot_list       : List[torch.Tensor] = []
    lhand_list       : List[torch.Tensor] = []
    rhand_list       : List[torch.Tensor] = []
    objcontact_list  : List[torch.Tensor] = []
    obj_trans_all    : List[torch.Tensor] = []
    obj_rot_all      : List[torch.Tensor] = []
    obj_scale_all    : List[torch.Tensor] = []

    with torch.no_grad():
        for s, e in iter_slices(T, DEFAULT_CHUNK_LEN):
            # 切块取数据
            root_orient_slice = poses[s:e, :3].astype(np.float32)
            pose_body_slice   = poses[s:e, 3:66].astype(np.float32)
            trans_slice       = trans_norm[s:e].astype(np.float32)

            object_angles_slice = object_angles[s:e].astype(np.float32)
            object_trans_slice  = object_trans_norm[s:e].astype(np.float32)

            betas_chunk = pad_betas(betas_raw, bm_loader.num_betas, e - s)
            smpl_input = build_smpl_input(root_orient_slice, pose_body_slice, trans_slice, betas_chunk, device)

            # 你的 forward_smpl_in_batches（本 chunk joints）
            pos_chunk, _ = forward_smpl_in_batches(bm, smpl_input, batch_size=smpl_batch)
            # 整段一致：加全局 floor（保持你的 |y| 语义）
            # pos_chunk[:, :, 1] += human_floor_abs
            pos_global_list.append(pos_chunk)

            # 旋转特征（分块）
            rot_local, rot_global = compute_rotation_features_chunk(root_orient_slice, pose_body_slice, bm, device)
            rot_local_list.append(rot_local.float())
            rot_global_list.append(rot_global.float())

            # 物体：旋转/平移张量 + 全局 floor
            obj_rot_t   = torch.tensor(R.from_rotvec(object_angles_slice).as_matrix().astype(np.float32),
                                       dtype=torch.float32, device=device)
            obj_trans_t = torch.tensor(object_trans_slice, dtype=torch.float32, device=device)
            # obj_trans_t[:, 1] += object_floor_abs  # 整段一致
            obj_scale_t = torch.ones((e - s,), dtype=torch.float32, device=device)

            # 接触（按 chunk 跑）
            pos_dev = pos_chunk.to(device)
            lfoot_c, rfoot_c = compute_foot_contact_labels(pos_dev)
            lhand_c, rhand_c, obj_c = compute_improved_contact_labels(
                obj_trans_t, obj_rot_t, obj_scale_t,
                pos_dev, os.path.dirname(mesh_path), object_name, device, e - s
            )

            lfoot_list.append(lfoot_c.cpu()); rfoot_list.append(rfoot_c.cpu())
            lhand_list.append(lhand_c.cpu()); rhand_list.append(rhand_c.cpu()); objcontact_list.append(obj_c.cpu())
            obj_trans_all.append(obj_trans_t.detach().cpu())
            obj_rot_all.append(obj_rot_t.detach().cpu())
            obj_scale_all.append(obj_scale_t.detach().cpu())

            del smpl_input, pos_chunk, pos_dev, obj_rot_t, obj_trans_t, obj_scale_t
            if device.type == "cuda": torch.cuda.empty_cache()

    # 拼接
    position_global = torch.cat(pos_global_list, dim=0).float()
    rotation_local_full_gt = torch.cat(rot_local_list,  dim=0).float()
    rotation_global        = torch.cat(rot_global_list, dim=0).float()
    lfoot_contact = torch.cat(lfoot_list, dim=0)
    rfoot_contact = torch.cat(rfoot_list, dim=0)
    lhand_contact = torch.cat(lhand_list, dim=0)
    rhand_contact = torch.cat(rhand_list, dim=0)
    obj_contact   = torch.cat(objcontact_list, dim=0)
    obj_trans_cpu = torch.cat(obj_trans_all, dim=0)
    obj_rot_cpu   = torch.cat(obj_rot_all,   dim=0)
    obj_scale_cpu = torch.cat(obj_scale_all, dim=0)

    data = {
        "seq_name": seq_name,
        "gender": gender,
        "rotation_local_full_gt_list": rotation_local_full_gt,
        "position_global_full_gt_world": position_global,
        "rotation_global": rotation_global,
        "trans": trans_norm,        
        "lfoot_contact": lfoot_contact,
        "rfoot_contact": rfoot_contact,
        "betas": torch.from_numpy(betas_raw).float(),
        "obj_name": object_name,
        "obj_scale": obj_scale_cpu,
        "obj_trans": obj_trans_cpu,
        "obj_rot": obj_rot_cpu,
        "lhand_contact": lhand_contact,
        "rhand_contact": rhand_contact,
        "obj_contact": obj_contact
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{seq_name}.pt")
    torch.save(data, out_path)

    if device.type == "cuda": 
        torch.cuda.empty_cache()



def collect_sequences(dataset_root: str) -> List[str]:
    seq_dirs: List[str] = []
    for entry in sorted(os.listdir(dataset_root)):
        full_path = os.path.join(dataset_root, entry)
        if os.path.isdir(full_path):
            smpl_path = os.path.join(full_path, "smpl_fit_all.npz")
            obj_path = os.path.join(full_path, "object_fit_all.npz")
            if os.path.exists(smpl_path) and os.path.exists(obj_path):
                seq_dirs.append(full_path)
    return seq_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess BEHAVE dataset sequences into pt files")
    parser.add_argument("--dataset_root", type=str, default='datasets/BEHAVE/sequences', help="Root directory of BEHAVE sequences")
    parser.add_argument("--objects_root", type=str, default='datasets/BEHAVE/objects', help="Directory containing BEHAVE object meshes")
    parser.add_argument("--output_dir", type=str, default='process/processed_split_data_BEHAVE', help="Directory to store processed pt files")
    parser.add_argument("--support_dir", type=str, default="body_models", help="Directory with SMPL-H support files")
    parser.add_argument("--frame_trim", type=int, default=None, help="Optional maximum number of frames per sequence")
    parser.add_argument("--device", type=str, default=None, help="Force computation device (cpu or cuda)")
    parser.add_argument("--smpl_batch", type=int, default=256, help="Batch size for SMPL forward pass")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bm_loader = BodyModelLoader(args.support_dir, device)

    seq_dirs = collect_sequences(args.dataset_root)
    if not seq_dirs:
        raise FileNotFoundError(f"No valid sequences found under {args.dataset_root}")

    object_names = [name for name in os.listdir(args.objects_root) if os.path.isdir(os.path.join(args.objects_root, name))]
    if not object_names:
        raise FileNotFoundError(f"No object subdirectories found in {args.objects_root}")

    for seq_dir in tqdm(seq_dirs, desc="Processing BEHAVE sequences"):
        process_behave_sequence(
            seq_dir,
            bm_loader,
            args.objects_root,
            object_names,
            device,
            args.output_dir,
            args.frame_trim,
            args.smpl_batch,
        )


if __name__ == "__main__":
    main()
