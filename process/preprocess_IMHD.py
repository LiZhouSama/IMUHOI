import argparse
import os
import re
import glob
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.rotation_tools import local2global_pose
import pytorch3d.transforms as transforms

from preprocess import compute_improved_contact_labels, compute_foot_contact_labels

DEFAULT_CHUNK_LEN = 4000

ROT_Z90 = np.array(
    [[0.0, -1.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


def _moving_avg_timewise(x, k=3):
    """
    沿时间维做滑动平均：
      - 支持 torch.Tensor / numpy.ndarray
      - 输入形状可为 [T], [T, C], [T, ..., C]（例如 [T, 22, 3]）
      - 仅沿时间维平滑；其它维（关节/通道）互不混合
      - 保持原始形状与（若为浮点）dtype
    """
    is_np = isinstance(x, np.ndarray)
    xt = torch.as_tensor(x)  # 零拷贝包装
    orig_dtype = xt.dtype
    orig_shape = tuple(xt.shape)

    # 只对浮点做平滑；非浮点（如 bool 标签）不要传进来
    if not torch.is_floating_point(xt):
        xt = xt.float()

    # 统一到 [T, C_total]，C_total = 其余维度的乘积（如 22*3）
    if xt.ndim == 1:
        xt = xt[:, None]  # [T, 1]
        rest_shape = (1,)
    else:
        T = xt.shape[0]
        rest_shape = tuple(xt.shape[1:])
        C_total = int(torch.tensor(rest_shape).prod().item())
        xt = xt.reshape(T, C_total)  # [T, C_total]

    T = xt.shape[0]
    if k <= 1 or T <= 2:
        out = xt
    else:
        pad = (k - 1) // 2
        # NCL: [1, C, T]
        x_ncl = xt.transpose(0, 1).unsqueeze(0)
        x_pad = F.pad(x_ncl, (pad, pad), mode="replicate")
        y = F.avg_pool1d(x_pad, kernel_size=k, stride=1)  # 逐通道滑动平均
        out = y.squeeze(0).transpose(0, 1)  # [T, C_total]

    # 还原到原形状
    if len(rest_shape) == 1 and rest_shape[0] == 1 and len(orig_shape) == 1:
        out = out.squeeze(1)  # 还原 [T]
    else:
        out = out.reshape(orig_shape)

    if is_np:
        # numpy：保持浮点 dtype，不是浮点则返回 float32（通常输入已是浮点）
        return out.cpu().numpy().astype(x.dtype if np.issubdtype(x.dtype, np.floating) else np.float32)
    else:
        # torch：尽量还原原始浮点 dtype
        if orig_dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            out = out.to(dtype=orig_dtype)
        return out

# ROT_Z90 = np.array([[1.0, 0.0, 0.0],
#                     [0.0, 1.0, 0.0],
#                     [0.0, 0.0, 1.0]], dtype=np.float32)

GT_FILE_PATTERN = re.compile(r"gt_(\d+)_(\d+)_(-?\d+)\.pkl")


def iter_slices(total: int, chunk_len: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, total, chunk_len):
        end = min(start + chunk_len, total)
        yield start, end


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
        return self._models.get(key, self._models[self._default_gender])


def infer_object_name(seq_name: str, object_names: List[str]) -> str:
    lowered = seq_name.lower()
    for candidate in sorted(object_names, key=len, reverse=True):
        if candidate.lower() in lowered:
            return candidate
    raise ValueError(f"Unable to infer object name from sequence identifier '{seq_name}'")


_mesh_cache: Dict[str, str] = {}


def resolve_object_mesh(object_name: str, objects_root: str) -> str:
    if object_name in _mesh_cache:
        return _mesh_cache[object_name]
    candidate_paths: List[str] = []
    direct = os.path.join(objects_root, f"{object_name}.obj")
    if os.path.exists(direct):
        candidate_paths.append(direct)
    object_dir = os.path.join(objects_root, object_name)
    if os.path.isdir(object_dir):
        obj_candidates = sorted(glob.glob(os.path.join(object_dir, "*.obj")))
        preferred = [
            path for path in obj_candidates
            if "simplified" in os.path.basename(path) and "transformed" in os.path.basename(path)
        ]
        candidate_paths.extend(preferred or obj_candidates)
    if not candidate_paths:
        raise FileNotFoundError(f"Unable to locate mesh for object '{object_name}' under {objects_root}")
    mesh_path = candidate_paths[0]
    _mesh_cache[object_name] = mesh_path
    return mesh_path


def pad_betas(betas: np.ndarray, target_dim: int, target_frames: int) -> np.ndarray:
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim == 1:
        betas = betas[None, :]
    if betas.shape[0] == 0:
        betas = np.zeros((target_frames, target_dim), dtype=np.float32)
    if betas.shape[0] == 1 and target_frames > 1:
        betas = np.repeat(betas, repeats=target_frames, axis=0)
    elif betas.shape[0] < target_frames:
        betas = np.pad(betas, ((0, target_frames - betas.shape[0]), (0, 0)), mode="edge")
    elif betas.shape[0] > target_frames:
        betas = betas[:target_frames]
    if betas.shape[1] < target_dim:
        betas = np.concatenate(
            [betas, np.zeros((betas.shape[0], target_dim - betas.shape[1]), dtype=betas.dtype)],
            axis=1,
        )
    elif betas.shape[1] > target_dim:
        betas = betas[:, :target_dim]
    return betas


def build_smpl_input(
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    smpl_input: Dict[str, torch.Tensor] = {
        "root_orient": torch.tensor(root_orient, dtype=torch.float32, device=device),
        "pose_body": torch.tensor(pose_body, dtype=torch.float32, device=device),
        "trans": torch.tensor(trans, dtype=torch.float32, device=device),
    }
    if betas.size > 0:
        smpl_input["betas"] = torch.tensor(betas, dtype=torch.float32, device=device)
    return smpl_input


def compute_rotation_features_chunk(
    root_orient_chunk: np.ndarray,
    pose_body_chunk: np.ndarray,
    bm: BodyModel,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    poses = np.concatenate([root_orient_chunk, pose_body_chunk], axis=1).astype(np.float32)
    local_rot_mat = transforms.axis_angle_to_matrix(
        torch.tensor(poses, dtype=torch.float32, device=device).reshape(-1, 3)
    ).reshape(poses.shape[0], -1, 3, 3)
    rotation_local_6d = transforms.matrix_to_rotation_6d(local_rot_mat)
    rotation_local_full_gt = rotation_local_6d.reshape(poses.shape[0], -1)

    kintree_table = bm.kintree_table[0].long()[:22]
    kintree_table[0] = -1
    rotation_global_matrot = local2global_pose(
        local_rot_mat.reshape(local_rot_mat.shape[0], -1, 9), kintree_table
    )
    rotation_global_matrot = rotation_global_matrot.reshape(rotation_global_matrot.shape[0], -1, 3, 3)
    return rotation_local_full_gt.detach().cpu(), rotation_global_matrot.detach().cpu()


def forward_smpl_in_batches(
    bm: BodyModel,
    smpl_input: Dict[str, torch.Tensor],
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    total_frames = smpl_input["root_orient"].shape[0]
    joint_chunks: List[torch.Tensor] = []
    min_abs_y: Optional[float] = None
    with torch.no_grad():
        for start in range(0, total_frames, batch_size):
            end = min(start + batch_size, total_frames)
            kwargs = {key: value[start:end] for key, value in smpl_input.items()}
            body_out = bm(**kwargs)
            joints = body_out.Jtr[:, :22, :].detach().cpu()
            verts = body_out.v
            batch_min_abs = float(verts[:, :, 1].abs().min().item())
            if min_abs_y is None or batch_min_abs < min_abs_y:
                min_abs_y = batch_min_abs
            joint_chunks.append(joints)
            del body_out
    position_global = torch.cat(joint_chunks, dim=0)
    min_y_tensor = torch.tensor(min_abs_y if min_abs_y is not None else 0.0, dtype=position_global.dtype)
    return position_global, min_y_tensor


def compute_foot_floor_offset(
    bm: BodyModel,
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    """计算整个序列中双脚关节的最低y坐标，用于地面对齐"""
    T = root_orient.shape[0]
    smpl_input = build_smpl_input(root_orient, pose_body, trans, betas, device)
    
    foot_min_y: Optional[float] = None
    with torch.no_grad():
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            kwargs = {key: value[start:end] for key, value in smpl_input.items()}
            body_out = bm(**kwargs)
            joints = body_out.Jtr[:, :22, :]  # [B, 22, 3]
            # 双脚关节: 7=左脚踝, 8=右脚踝, 10=左脚, 11=右脚
            left_ankle = joints[:, 7, 1]   # [B]
            right_ankle = joints[:, 8, 1]  # [B]
            left_foot = joints[:, 10, 1]   # [B]
            right_foot = joints[:, 11, 1]  # [B]
            
            batch_min = torch.min(torch.stack([left_ankle, right_ankle, left_foot, right_foot], dim=0)).item()
            foot_min_y = batch_min if foot_min_y is None else min(foot_min_y, batch_min)
            del body_out
    
    return float(foot_min_y or 0.0)


@dataclass
class SequenceEntry:
    seq_name: str
    motion_dir: str
    pkl_files: List[str]


def parse_gt_filename(path: str) -> Tuple[int, int, int]:
    name = os.path.basename(path)
    match = GT_FILE_PATTERN.match(name)
    if not match:
        raise ValueError(f"Unexpected ground truth filename: {name}")
    seg_idx, start, end = match.groups()
    return int(seg_idx), int(start), int(end)


def collect_sequences(dataset_root: str) -> List[SequenceEntry]:
    entries: List[SequenceEntry] = []
    for current_root, _, files in os.walk(dataset_root):
        gt_files = [os.path.join(current_root, f) for f in files if GT_FILE_PATTERN.match(f)]
        if not gt_files:
            continue
        rel_path = os.path.relpath(current_root, dataset_root)
        seq_name = "__".join(rel_path.split(os.sep))
        sorted_files = sorted(gt_files, key=lambda path: (parse_gt_filename(path)[1], parse_gt_filename(path)[0]))
        entries.append(SequenceEntry(seq_name=seq_name, motion_dir=current_root, pkl_files=sorted_files))
    entries.sort(key=lambda entry: entry.seq_name)
    return entries


def load_imhd_sequence(pkl_files: List[str]) -> Dict[str, np.ndarray]:
    accum: Dict[str, List[np.ndarray]] = {
        "objectRot": [],
        "objectTrans": [],
        "smplPose": [],
        "smplShape": [],
        "smplTrans": [],
    }
    for path in pkl_files:
        with open(path, "rb") as f:
            record = pickle.load(f)
        for key in accum.keys():
            if key not in record:
                raise KeyError(f"Key '{key}' missing from {path}")
            accum[key].append(np.asarray(record[key]))
    return {key: np.concatenate(chunks, axis=0) for key, chunks in accum.items()}


def process_imhd_sequence(
    entry: SequenceEntry,
    bm_loader: BodyModelLoader,
    objects_root: str,
    object_names: List[str],
    device: torch.device,
    output_dir: str,
    frame_trim: Optional[int] = None,
    smpl_batch: int = 256,
) -> None:

    data_np = load_imhd_sequence(entry.pkl_files)
    
    # 在降采样前对位置、旋转和姿态数据进行平滑（k=3）
    # 平滑关键的连续数据以减少噪声
    for key in ["smplTrans", "smplPose", "objectTrans", "objectRot"]:
        if key in data_np:
            data_np[key] = _moving_avg_timewise(data_np[key], k=3)
    
    # 降采样：60fps -> 30fps (每隔一帧取一帧)
    for key in data_np.keys():
        data_np[key] = data_np[key][::2]
    
    total_frames = data_np["smplPose"].shape[0]
    T = total_frames if frame_trim is None else min(total_frames, frame_trim)
    if T <= 0:
        raise ValueError(f"Sequence {entry.seq_name} has no frames after trimming")

    root_orient = data_np["smplPose"][:T, :3].astype(np.float32)
    pose_body = data_np["smplPose"][:T, 3:].astype(np.float32)
    trans = data_np["smplTrans"][:T].astype(np.float32)

    betas_raw = data_np["smplShape"][:T].astype(np.float32)

    object_angles = data_np["objectRot"][:T].astype(np.float32)
    object_trans = data_np["objectTrans"][:T].astype(np.float32)

    if T > 0:
        root_offset = trans[0].copy()
        trans -= root_offset
        object_trans -= root_offset

    gender = None
    object_name = infer_object_name(entry.seq_name, object_names)
    mesh_path = resolve_object_mesh(object_name, objects_root)
    mesh_dir = os.path.dirname(mesh_path)
    mesh_basename = os.path.splitext(os.path.basename(mesh_path))[0]
    bm = bm_loader.get(gender)

    smpl_init_input = {
            "root_orient": torch.zeros_like(torch.from_numpy(root_orient.reshape(-1, 3)).to(device).float()),
            "pose_body": torch.zeros_like(torch.from_numpy(pose_body.reshape(-1, 63)).to(device).float()),
            "trans": torch.zeros_like(torch.from_numpy(trans.reshape(-1, 3)).to(device).float())
        }
    Jtr_0 = np.asarray(bm(**smpl_init_input).Jtr[:, 0, :].cpu(), dtype=np.float32)

    rot_z90_torch = torch.from_numpy(ROT_Z90).float().to(device)
    rot_z90_repeat = rot_z90_torch.unsqueeze(0)

    root_orient_t = torch.from_numpy(root_orient).float().to(device)
    root_rot_mat = transforms.axis_angle_to_matrix(root_orient_t)
    root_rot_mat = torch.matmul(rot_z90_repeat.expand(root_rot_mat.shape[0], -1, -1), root_rot_mat)
    root_orient_t = transforms.matrix_to_axis_angle(root_rot_mat)
    root_orient = root_orient_t.detach().cpu().numpy().astype(np.float32)

    trans = (trans + Jtr_0) @ ROT_Z90.T - Jtr_0

    object_angles_t = torch.from_numpy(object_angles).float().to(device)
    obj_rot_mat = transforms.axis_angle_to_matrix(object_angles_t)
    obj_rot_mat = torch.matmul(rot_z90_repeat.expand(obj_rot_mat.shape[0], -1, -1), obj_rot_mat)
    object_angles_t = transforms.matrix_to_axis_angle(obj_rot_mat)
    object_angles = object_angles_t.detach().cpu().numpy().astype(np.float32)

    object_trans = object_trans @ ROT_Z90.T

    # ---------- 计算双脚最低点偏移 ----------
    betas_for_floor = pad_betas(betas_raw, bm_loader.num_betas, T)
    foot_floor_y = compute_foot_floor_offset(
        bm, root_orient, pose_body, trans, betas_for_floor, device, batch_size=smpl_batch
    )
    # 修正 trans 和 object_trans，使双脚最低点对齐到 y=0
    trans[:, 1] -= foot_floor_y
    object_trans[:, 1] -= foot_floor_y

    rot_local_chunks: List[torch.Tensor] = []
    rot_global_chunks: List[torch.Tensor] = []
    pos_chunks: List[torch.Tensor] = []
    lfoot_chunks: List[torch.Tensor] = []
    rfoot_chunks: List[torch.Tensor] = []
    lhand_chunks: List[torch.Tensor] = []
    rhand_chunks: List[torch.Tensor] = []
    obj_contact_chunks: List[torch.Tensor] = []
    obj_trans_chunks: List[torch.Tensor] = []
    obj_rot_chunks: List[torch.Tensor] = []
    obj_scale_chunks: List[torch.Tensor] = []

    with torch.no_grad():
        for start, end in iter_slices(T, DEFAULT_CHUNK_LEN):
            betas_chunk = pad_betas(betas_raw[start:end], bm_loader.num_betas, end - start)
            smpl_input = build_smpl_input(
                root_orient[start:end],
                pose_body[start:end],
                trans[start:end],
                betas_chunk,
                device,
            )

            pos_chunk, _ = forward_smpl_in_batches(bm, smpl_input, batch_size=smpl_batch)
            pos_chunks.append(pos_chunk)

            rot_local, rot_global = compute_rotation_features_chunk(
                root_orient[start:end], pose_body[start:end], bm, device
            )
            rot_local_chunks.append(rot_local.float())
            rot_global_chunks.append(rot_global.float())

            obj_rot_mat = R.from_rotvec(object_angles[start:end]).as_matrix().astype(np.float32)
            obj_rot_t = torch.tensor(obj_rot_mat, dtype=torch.float32, device=device)
            obj_trans_t = torch.tensor(object_trans[start:end], dtype=torch.float32, device=device)
            obj_scale_t = torch.ones((end - start,), dtype=torch.float32, device=device)

            pos_dev = pos_chunk.to(device)
            lfoot_c, rfoot_c = compute_foot_contact_labels(pos_dev)
            lhand_c, rhand_c, obj_c = compute_improved_contact_labels(
                obj_trans_t,
                obj_rot_t,
                obj_scale_t,
                pos_dev,
                mesh_dir,
                mesh_basename,
                device,
                end - start,
            )

            lfoot_chunks.append(lfoot_c.cpu())
            rfoot_chunks.append(rfoot_c.cpu())
            lhand_chunks.append(lhand_c.cpu())
            rhand_chunks.append(rhand_c.cpu())
            obj_contact_chunks.append(obj_c.cpu())
            obj_trans_chunks.append(obj_trans_t.cpu())
            obj_rot_chunks.append(obj_rot_t.cpu())
            obj_scale_chunks.append(obj_scale_t.cpu())

            del smpl_input, pos_chunk, pos_dev, obj_rot_t, obj_trans_t, obj_scale_t
            if device.type == "cuda":
                torch.cuda.empty_cache()

    position_global = torch.cat(pos_chunks, dim=0).float()
    rotation_local_full_gt = torch.cat(rot_local_chunks, dim=0).float()
    rotation_global = torch.cat(rot_global_chunks, dim=0).float()
    lfoot_contact = torch.cat(lfoot_chunks, dim=0)
    rfoot_contact = torch.cat(rfoot_chunks, dim=0)
    lhand_contact = torch.cat(lhand_chunks, dim=0)
    rhand_contact = torch.cat(rhand_chunks, dim=0)
    obj_contact = torch.cat(obj_contact_chunks, dim=0)
    obj_trans_cpu = torch.cat(obj_trans_chunks, dim=0)
    obj_rot_cpu = torch.cat(obj_rot_chunks, dim=0)
    obj_scale_cpu = torch.cat(obj_scale_chunks, dim=0)

    output = {
        "seq_name": entry.seq_name,
        "gender": gender,
        "rotation_local_full_gt_list": rotation_local_full_gt,
        "position_global_full_gt_world": position_global,
        "rotation_global": rotation_global,
        "trans": trans,
        "lfoot_contact": lfoot_contact,
        "rfoot_contact": rfoot_contact,
        "betas": torch.from_numpy(betas_raw).float(),
        "obj_name": object_name,
        "obj_mesh_name": mesh_basename,
        "obj_scale": obj_scale_cpu,
        "obj_trans": obj_trans_cpu,
        "obj_rot": obj_rot_cpu,
        "lhand_contact": lhand_contact,
        "rhand_contact": rhand_contact,
        "obj_contact": obj_contact,
        "source_pkls": [os.path.relpath(path, entry.motion_dir) for path in entry.pkl_files],
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{entry.seq_name}.pt")
    torch.save(output, out_path)

    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess IMHD dataset ground truth into PyTorch tensors.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=r"datasets/IMHD/IMHD Dataset/ground_truth",
        help="Root directory containing IMHD ground-truth PKL files.",
    )
    parser.add_argument(
        "--objects_root",
        type=str,
        default=r"datasets/IMHD/IMHD Dataset/object_templates",
        help="Directory that stores IMHD object template meshes.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="process/processed_split_data_IMHD",
        help="Directory where processed sequence tensors will be written.",
    )
    parser.add_argument(
        "--support_dir",
        type=str,
        default="body_models",
        help="Directory with SMPL-H support files.",
    )
    parser.add_argument(
        "--frame_trim",
        type=int,
        default=None,
        help="Optionally limit each sequence to this many frames.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force computation device (cpu or cuda).",
    )
    parser.add_argument(
        "--smpl_batch",
        type=int,
        default=64,
        help="Batch size for SMPL forward passes.",
    )
    parser.add_argument(
        "--num_betas",
        type=int,
        default=16,
        help="Number of body shape coefficients expected by the SMPL-H model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bm_loader = BodyModelLoader(args.support_dir, device, num_betas=args.num_betas)
    sequences = collect_sequences(args.dataset_root)
    if not sequences:
        raise FileNotFoundError(f"No IMHD PKL files found under {args.dataset_root}")

    object_names = [
        name for name in os.listdir(args.objects_root) if os.path.isdir(os.path.join(args.objects_root, name))
    ]
    if not object_names:
        raise FileNotFoundError(f"No object templates found in {args.objects_root}")

    for entry in tqdm(sequences, desc="Processing IMHD sequences"):
        try:
            process_imhd_sequence(
                entry,
                bm_loader,
                args.objects_root,
                object_names,
                device,
                args.output_dir,
                frame_trim=args.frame_trim,
                smpl_batch=args.smpl_batch,
            )
        except Exception as exc:
            print(f"Failed to process {entry.seq_name}: {exc}")


if __name__ == "__main__":
    main()
