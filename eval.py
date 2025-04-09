import os
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from models.DiT_model import MotionDiffusion
from dataloader.dataloader import IMUDataset
from easydict import EasyDict as edict
from human_body_prior.body_model.body_model import BodyModel
import pytorch3d.transforms as transforms

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = edict(config)
    return config

def load_smpl_model(smpl_model_path, device):
    """加载 SMPL 模型 using human_body_prior"""
    smpl_model = BodyModel(
        bm_fname=smpl_model_path,
        num_betas=16
    ).to(device)
    return smpl_model

def load_model(model_path, config, device):
    """加载训练好的 Diffusion 模型"""
    input_length = config['train']['window'] if 'window' in config['train'] else config['test']['window']

    if config.get('model', {}).get('use_dit_model', True):
        print("Loading DiT_model...")
        model = MotionDiffusion(config, input_length=input_length).to(device)
    else:
        print("Loading wrap_model...")
        from models.wrap_model import MotionDiffusion as WrapMotionDiffusion
        model = WrapMotionDiffusion(config, input_length=input_length).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

def compute_frobenius_norm(mat1, mat2):
    """计算两个旋转矩阵批次之间的平均 Frobenius 范数"""
    diff_norm = torch.linalg.norm(mat1 - mat2, ord='fro', dim=(-2, -1))
    mask = ~torch.isnan(diff_norm)
    if mask.sum() == 0:
      return float('nan')
    return diff_norm[mask].mean()

def evaluate_model(model, smpl_model, data_loader, device):
    """评估模型性能，计算 MPJPE, MPJRE, Object Error, Jitter (适配 SMPLH)"""
    metrics = {
        'mpjpe': [], 'mpjre_rot': [],
        'obj_trans_err': [], 'obj_rot_err': [], 'jitter': []
    }
    num_batches = 0
    num_eval_joints = 22

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            gt_root_pos = batch["root_pos"].to(device)
            gt_motion = batch["motion"].to(device)
            gt_human_imu = batch["human_imu"].to(device)
            gt_obj_imu = batch["obj_imu"].to(device) if "obj_imu" in batch else None
            gt_obj_trans = batch["obj_trans"].to(device) if "obj_trans" in batch else None
            gt_obj_rot = batch["obj_rot"].to(device) if "obj_rot" in batch else None

            bs, seq_len, motion_dim = gt_motion.shape

            if motion_dim != 132:
                 print(f"Warning: Batch {batch_idx}: Expected motion dimension 132 for SMPLH (22*6D), got {motion_dim}. Skipping batch.")
                 continue

            has_object = gt_obj_imu is not None
            if not has_object:
                bs, seq_len = gt_human_imu.shape[:2]
                gt_obj_imu = torch.zeros((bs, seq_len, 1, 12), device=device)  # 注意：现在是12D

            data_dict = {
                "human_imu": gt_human_imu,
                "obj_imu": gt_obj_imu
            }

            pred_dict = model.diffusion_reverse(data_dict)
            # pred_root_pos = pred_dict["root_pos"]
            pred_motion = pred_dict["motion"]
            # pred_obj_trans = pred_dict["obj_trans"]
            pred_obj_rot = pred_dict["obj_rot"]

            pred_root_orient_6d = pred_motion[:, :, :6].reshape(bs * seq_len, 6)
            pred_body_pose_6d = pred_motion[:, :, 6:].reshape(bs * seq_len, 21, 6)
            gt_root_orient_6d = gt_motion[:, :, :6].reshape(bs * seq_len, 6)
            gt_body_pose_6d = gt_motion[:, :, 6:].reshape(bs * seq_len, 21, 6)

            pred_root_orient_mat = transforms.rotation_6d_to_matrix(pred_root_orient_6d)
            pred_body_pose_mat = transforms.rotation_6d_to_matrix(pred_body_pose_6d.reshape(-1, 6)).reshape(bs * seq_len, 21, 3, 3)
            gt_root_orient_mat = transforms.rotation_6d_to_matrix(gt_root_orient_6d)
            gt_body_pose_mat = transforms.rotation_6d_to_matrix(gt_body_pose_6d.reshape(-1, 6)).reshape(bs * seq_len, 21, 3, 3)

            pred_root_orient_axis = transforms.matrix_to_axis_angle(pred_root_orient_mat)
            pred_body_pose_axis = transforms.matrix_to_axis_angle(pred_body_pose_mat.reshape(-1, 3, 3)).reshape(bs * seq_len, 21 * 3)
            gt_root_orient_axis = transforms.matrix_to_axis_angle(gt_root_orient_mat)
            gt_body_pose_axis = transforms.matrix_to_axis_angle(gt_body_pose_mat.reshape(-1, 3, 3)).reshape(bs * seq_len, 21 * 3)

            # pred_transl = pred_root_pos.reshape(bs*seq_len, 3)
            # gt_transl = gt_root_pos.reshape(bs*seq_len, 3)

            pred_pose_body_input = {'root_orient': pred_root_orient_axis, 'pose_body': pred_body_pose_axis}
            gt_pose_body_input = {'root_orient': gt_root_orient_axis, 'pose_body': gt_body_pose_axis}

            pred_smplh_out = smpl_model(**pred_pose_body_input)
            gt_smplh_out = smpl_model(**gt_pose_body_input)

            pred_joints_all = pred_smplh_out.Jtr.view(bs, seq_len, -1, 3)
            gt_joints_all = gt_smplh_out.Jtr.view(bs, seq_len, -1, 3)

            pred_joints_eval = pred_joints_all[:, :, :num_eval_joints, :]
            gt_joints_eval = gt_joints_all[:, :, :num_eval_joints, :]

            pred_joints_rel = pred_joints_eval[:, :, 1:, :] - pred_joints_eval[:, :, 0:1, :]
            gt_joints_rel = gt_joints_eval[:, :, 1:, :] - gt_joints_eval[:, :, 0:1, :]
            mpjpe = torch.linalg.norm(pred_joints_rel - gt_joints_rel, dim=-1).mean()
            metrics['mpjpe'].append(mpjpe.item() * 1000)

            pred_body_pose_mat_rs = pred_body_pose_mat.view(bs, seq_len, 21, 3, 3)
            gt_body_pose_mat_rs = gt_body_pose_mat.view(bs, seq_len, 21, 3, 3)
            body_joint_rot_err = torch.linalg.norm(pred_body_pose_mat_rs - gt_body_pose_mat_rs, ord='fro', dim=(-2, -1))
            mpjre_rot = body_joint_rot_err.mean()
            metrics['mpjre_rot'].append(mpjre_rot.item())

            if has_object:
                # obj_trans_err = torch.linalg.norm(pred_obj_trans - gt_obj_trans, dim=-1).mean()
                # metrics['obj_trans_err'].append(obj_trans_err.item() * 1000)

                obj_rot_err = compute_frobenius_norm(pred_obj_rot, gt_obj_rot)
                metrics['obj_rot_err'].append(obj_rot_err.item())
            else:
                metrics['obj_trans_err'].append(float('nan'))
                metrics['obj_rot_err'].append(float('nan'))

            if seq_len >= 3:
                pred_accel = pred_joints_eval[:, 2:] - 2 * pred_joints_eval[:, 1:-1] + pred_joints_eval[:, :-2]
                jitter = torch.linalg.norm(pred_accel, dim=-1).mean()
                metrics['jitter'].append(jitter.item() * 1000)
            else:
                 metrics['jitter'].append(float('nan'))

            num_batches += 1

    avg_metrics = {}
    for key, values in metrics.items():
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            avg_metrics[key] = np.mean(valid_values)
        else:
            avg_metrics[key] = float('nan')

    return avg_metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config_path = 'configs/diffusion_0403.yaml'
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    smpl_model_path = config.get('smpl_model_path', 'body_models/smplh/male/model.npz')
    if not os.path.exists(smpl_model_path):
       print(f"Error: SMPL model path not found: {smpl_model_path}")
       print("Please provide the correct path to your SMPL model file (e.g., SMPL_NEUTRAL.pkl) in the config or script.")
       return
    print(f"Loading SMPL model from: {smpl_model_path}")
    smpl_model = load_smpl_model(smpl_model_path, device)

    model_path = config.get('model_path', None)
    if model_path is None:
        print("Error: Evaluation model path not found in the config.")
        print("Please provide the correct path to your trained model checkpoint in the config (eval_model_path) or script.")
        return
    print(f"Loading trained model from: {model_path}")
    model = load_model(model_path, config, device)

    test_data_dir = config['test'].get('data_path', None)
    if not os.path.exists(test_data_dir):
        print(f"Error: Test dataset path not found: {test_data_dir}")
        print("Please provide the correct path to your test dataset in the config (test.data_path).")
        return
    print(f"Loading test dataset from: {test_data_dir}")
    test_dataset = IMUDataset(
            data_dir=test_data_dir,
            window_size=config['test']['window'],
            window_stride=config['test'].get('window_stride', config['test']['window']),
            normalize=config['test'].get('normalize', False),
            debug=False
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['test'].get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=False
    )

    print("\n开始评估模型...")
    results = evaluate_model(model, smpl_model, test_loader, device)

    print("\n评估结果:")
    print(f"MPJPE (mm, body joints 1-21 relative to root): {results.get('mpjpe', 'N/A'):.4f}")
    print(f"MPJRE (Frob norm, body joints 1-21):          {results.get('mpjre_rot', 'N/A'):.4f}")
    print(f"Obj Trans Error (mm):                         {results.get('obj_trans_err', 'N/A'):.4f}")
    print(f"Obj Rot Error (Frob):                         {results.get('obj_rot_err', 'N/A'):.4f}")
    print(f"Jitter (mm/frame^2, joints 0-21):             {results.get('jitter', 'N/A'):.4f}")

if __name__ == "__main__":
    main() 