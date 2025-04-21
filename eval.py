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
import argparse

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
    """加载训练好的模型"""
    input_length = config.test.get('window', config.train.get('window', 60))

    # 检查模型类型并加载
    if config.get('use_transpose_model', False):
        print("Loading TransPose model...")
        from models.do_train_imu_TransPose import load_transpose_model
        model = load_transpose_model(config, model_path)
    elif config.get('use_transpose_humanOnly_model', False):
        print("Loading TransPose HumanOnly model...")
        from models.do_train_imu_TransPose_humanOnly import load_transpose_model_humanOnly
        model = load_transpose_model_humanOnly(config, model_path)
    elif config.model.get('use_dit_model', True):
        print("Loading DiT_model...")
        model = MotionDiffusion(config, input_length=input_length).to(device)
    else:
        print("Loading wrap_model...")
        from models.wrap_model import MotionDiffusion as WrapMotionDiffusion
        model = WrapMotionDiffusion(config, input_length=input_length).to(device)

    # 如果模型不是通过特定加载函数（如 load_transpose_model）加载的，则加载 state_dict
    if not (config.get('use_transpose_model', False) or config.get('use_transpose_humanOnly_model', False)):
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            # Handle potential keys like 'model_state_dict' or 'model'
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            # Remove 'module.' prefix if saved with DataParallel
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        else:
             print(f"Warning: Model checkpoint not found at {model_path}. Model weights not loaded.")

    model.eval()
    model.to(device)
    return model

def evaluate_model(model, smpl_model, data_loader, device, evaluate_objects=True):
    """评估模型性能，计算 MPJPE, MPJRE (角度), Object Error, Jitter (适配 SMPLH)"""
    metrics = {
        'mpjpe': [], 'mpjre_angle': [],
        'obj_trans_err': [], 'obj_rot_angle': [], 'jitter': []
    }
    num_batches = 0
    num_eval_joints = 22
    num_body_joints = 21

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # --- Ground Truth Data ---
            gt_root_pos = batch["root_pos"].to(device)
            gt_motion = batch["motion"].to(device)
            gt_human_imu = batch["human_imu"].to(device)
            # Add check for head_global_trans - crucial for denormalization
            if "head_global_trans_start" not in batch:
                 print(f"Warning: Batch {batch_idx} missing 'head_global_trans_start'. Skipping batch.")
                 continue
            gt_head_trans_start = batch["head_global_trans_start"].to(device)

            gt_obj_imu = batch.get("obj_imu", None)
            gt_obj_trans = batch.get("obj_trans", None)
            gt_obj_rot_6d = batch.get("obj_rot", None)

            if gt_obj_imu is not None: gt_obj_imu = gt_obj_imu.to(device)
            if gt_obj_trans is not None: gt_obj_trans = gt_obj_trans.to(device)
            if gt_obj_rot_6d is not None: gt_obj_rot_6d = gt_obj_rot_6d.to(device)

            bs, seq_len, motion_dim = gt_motion.shape
            if motion_dim != 132:
                 print(f"Warning: Batch {batch_idx}: Expected motion dimension 132 for SMPLH (22*6D), got {motion_dim}. Skipping batch.")
                 continue

            # Determine if the batch contains object data
            has_object = gt_obj_imu is not None and gt_obj_trans is not None and gt_obj_rot_6d is not None

            # Prepare dummy object IMU if needed by the model but not present in GT
            if has_object:
                obj_imu_input = gt_obj_imu
            else:
                obj_imu_input = None

            # --- Model Prediction ---
            # Prepare input dictionary
            model_input = {"human_imu": gt_human_imu}
            if obj_imu_input is not None:
                model_input["obj_imu"] = obj_imu_input

            try:
                if hasattr(model, 'diffusion_reverse'):
                    pred_dict = model.diffusion_reverse(model_input)
                else:
                    pred_dict = model(model_input)
            except Exception as e:
                 print(f"Error during model inference in batch {batch_idx}: {e}")
                 continue

            # --- Extract Predictions (Normalized) ---
            pred_root_pos_norm = pred_dict.get("root_pos", None)
            pred_motion_norm = pred_dict.get("motion", None)
            pred_obj_trans_norm = pred_dict.get("obj_trans", None)
            pred_obj_rot_norm = pred_dict.get("obj_rot", None)

            if pred_motion_norm is None:
                print(f"Warning: Batch {batch_idx}: Model did not output 'motion'. Skipping batch.")
                continue
            if pred_root_pos_norm is None:
                print(f"Warning: Batch {batch_idx}: Model did not output 'root_pos'. Using GT root position for evaluation.")


            # --- Denormalize Predictions using First Frame Head Pose ---
            head_global_rot_start = gt_head_trans_start[..., :3, :3]
            head_global_pos_start = gt_head_trans_start[..., :3, 3]

            # Denormalize root orientation， 不再需要denormalize
            pred_root_orient_6d_norm = pred_motion_norm[:, :, :6]
            pred_root_orient_mat_norm = transforms.rotation_6d_to_matrix(pred_root_orient_6d_norm)
            # pred_root_orient_mat = head_global_rot_start @ pred_root_orient_mat_norm
            # pred_root_orient_axis = transforms.matrix_to_axis_angle(pred_root_orient_mat).reshape(bs * seq_len, 3)
            pred_root_orient_axis = transforms.matrix_to_axis_angle(pred_root_orient_mat_norm).reshape(bs * seq_len, 3)
            
            if pred_root_pos_norm is not None:
                # pred_root_pos_relative_rotated = torch.matmul(head_global_rot_start, pred_root_pos_norm.unsqueeze(-1))
                # pred_transl = pred_root_pos_relative_rotated.squeeze(-1) @ head_global_rot_start.transpose(-1, -2) + head_global_pos_start
                pred_transl = pred_transl.reshape(bs * seq_len, 3)
            else:
                pred_transl = gt_root_pos.reshape(bs*seq_len, 3)

            # Body pose is local, no denormalization needed
            pred_body_pose_6d = pred_motion_norm[:, :, 6:].reshape(bs * seq_len, 21, 6)
            pred_body_pose_mat = transforms.rotation_6d_to_matrix(pred_body_pose_6d.reshape(-1, 6)).reshape(bs * seq_len, 21, 3, 3)
            pred_body_pose_axis = transforms.matrix_to_axis_angle(pred_body_pose_mat.reshape(-1, 3, 3)).reshape(bs * seq_len, 21 * 3)

            # --- Prepare GT for SMPL and Metrics ---
            gt_root_orient_6d = gt_motion[:, :, :6].reshape(bs * seq_len, 6)
            gt_body_pose_6d = gt_motion[:, :, 6:].reshape(bs * seq_len, 21, 6)
            gt_root_orient_mat = transforms.rotation_6d_to_matrix(gt_root_orient_6d)
            gt_body_pose_mat = transforms.rotation_6d_to_matrix(gt_body_pose_6d.reshape(-1, 6)).reshape(bs * seq_len, 21, 3, 3)
            gt_root_orient_axis = transforms.matrix_to_axis_angle(gt_root_orient_mat)
            gt_body_pose_axis = transforms.matrix_to_axis_angle(gt_body_pose_mat.reshape(-1, 3, 3)).reshape(bs * seq_len, 21 * 3)
            gt_transl = gt_root_pos.reshape(bs*seq_len, 3)

            # --- Get SMPL Joints ---
            pred_pose_body_input = {'root_orient': pred_root_orient_axis, 'pose_body': pred_body_pose_axis, 'trans': pred_transl}
            gt_pose_body_input = {'root_orient': gt_root_orient_axis, 'pose_body': gt_body_pose_axis, 'trans': gt_transl}

            try:
                pred_smplh_out = smpl_model(**pred_pose_body_input)
                gt_smplh_out = smpl_model(**gt_pose_body_input)
            except Exception as e:
                print(f"Error during SMPL forward pass in batch {batch_idx}: {e}")
                continue

            pred_joints_all = pred_smplh_out.Jtr.view(bs, seq_len, -1, 3)
            gt_joints_all = gt_smplh_out.Jtr.view(bs, seq_len, -1, 3)

            # --- Calculate Metrics ---
            # Select joints for evaluation (first 22 for SMPLH compatibility with SMPL eval)
            pred_joints_eval = pred_joints_all[:, :, :num_eval_joints, :]
            gt_joints_eval = gt_joints_all[:, :, :num_eval_joints, :]

            # MPJPE (Mean Per Joint Position Error - Absolute)
            # Calculate Euclidean distance between absolute predicted and GT joint positions
            # Shape of norm: [bs, seq_len, num_eval_joints]
            joint_distances = torch.linalg.norm(pred_joints_eval - gt_joints_eval, dim=-1)
            # Mean over batch, sequence length, and joints
            mpjpe = joint_distances.mean()
            metrics['mpjpe'].append(mpjpe.item() * 1000) # Convert meters to mm

            # MPJRE (Mean Per Joint Rotation Error - Angle in Degrees)
            # Reshape matrices for batch processing: [bs, seq, num_body_joints, 3, 3]
            pred_body_pose_mat_rs = pred_body_pose_mat.view(bs, seq_len, 21, 3, 3)
            gt_body_pose_mat_rs = gt_body_pose_mat.view(bs, seq_len, 21, 3, 3)
            # Calculate relative rotation: R_rel = R_gt^T @ R_pred
            # Transpose gt: [bs, seq, num_body_joints, 3, 3]
            rel_rot_mat = torch.matmul(gt_body_pose_mat_rs.transpose(-1, -2), pred_body_pose_mat_rs)
            # Calculate trace: sum of diagonal elements
            trace = torch.einsum('...ii->...', rel_rot_mat) # Shape: [bs, seq, num_body_joints]
            # Calculate cos(theta) = (trace - 1) / 2
            cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0) # Clamp for numerical stability
            # Calculate angle in radians: theta = arccos(cos_theta)
            angle_rad = torch.acos(cos_theta)
            # Convert angle to degrees
            angle_deg = angle_rad * (180.0 / np.pi)
            # Calculate the mean angle error over batch, sequence, and joints
            mpjre_angle = angle_deg.mean()
            metrics['mpjre_angle'].append(mpjre_angle.item()) # Store mean angle in degrees

            # Object Metrics (Conditional)
            if evaluate_objects and has_object:
                if pred_obj_trans_norm is not None: 
                    # pred_obj_trans_relative_rotated = pred_obj_trans_norm.unsqueeze(-1) @ head_global_rot_start.unsqueeze(1).transpose(-1, -2)
                    # pred_obj_trans = pred_obj_trans_relative_rotated.squeeze(-1) + head_global_pos_start.unsqueeze(1)
                    # Calculate Translation Error
                    obj_trans_err = torch.linalg.norm(pred_obj_trans_norm - gt_obj_trans, dim=-1).mean()
                    metrics['obj_trans_err'].append(obj_trans_err.item() * 1000) # Convert meters to mm
                else:
                    print(f"Warning: Batch {batch_idx}: Missing predicted object translation for error calculation.")
                    metrics['obj_trans_err'].append(float('nan'))
                if pred_obj_rot_norm is not None:
                    # Denormalize rotation
                    pred_obj_rot_6d_norm = pred_obj_rot_norm
                    pred_obj_rot_mat_norm = transforms.rotation_6d_to_matrix(pred_obj_rot_6d_norm.reshape(-1, 6)).reshape(bs, seq_len, 3, 3)
                    # pred_obj_rot_mat = torch.matmul(head_global_rot_start.unsqueeze(1), pred_obj_rot_mat_norm)
                    # Calculate Rotation Error (Angle in Degrees)
                    gt_obj_rot_mat = transforms.rotation_6d_to_matrix(gt_obj_rot_6d.reshape(-1, 6)).reshape(bs, seq_len, 3, 3)
                    # R_rel = R_gt^T @ R_pred
                    obj_rel_rot_mat = torch.matmul(gt_obj_rot_mat.transpose(-1, -2), pred_obj_rot_mat_norm)
                    obj_trace = torch.einsum('...ii->...', obj_rel_rot_mat) # [bs, seq]
                    obj_cos_theta = torch.clamp((obj_trace - 1.0) / 2.0, -1.0, 1.0)
                    obj_angle_rad = torch.acos(obj_cos_theta)
                    obj_angle_deg = obj_angle_rad * (180.0 / np.pi)
                    obj_rot_angle_err = obj_angle_deg.mean()
                    metrics['obj_rot_angle'].append(obj_rot_angle_err.item()) # Store mean angle in degrees
                else:
                    print(f"Warning: Batch {batch_idx}: Missing predicted object rotation for error calculation.")
                    metrics['obj_rot_angle'].append(float('nan'))
            elif evaluate_objects and not has_object:
                 metrics['obj_trans_err'].append(float('nan'))
                 metrics['obj_rot_angle'].append(float('nan'))


            # Jitter (Mean acceleration magnitude of joints)
            if seq_len >= 3:
                # Use world frame joints for jitter calculation
                pred_accel = pred_joints_eval[:, 2:] - 2 * pred_joints_eval[:, 1:-1] + pred_joints_eval[:, :-2]
                jitter = torch.linalg.norm(pred_accel, dim=-1).mean()
                metrics['jitter'].append(jitter.item() * 1000) # Convert m/frame^2 to mm/frame^2
            else:
                 metrics['jitter'].append(float('nan'))

            num_batches += 1
            if batch_idx % 50 == 0:
                print(f"Processed batch {batch_idx + 1} / {len(data_loader)}")

    avg_metrics = {}
    print("\nCalculating average metrics...")
    for key, values in metrics.items():
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            avg_metrics[key] = np.mean(valid_values)
            unit = ""
            if key == 'mpjpe' or key == 'obj_trans_err': unit = "(mm)"
            elif key == 'mpjre_angle' or key == 'obj_rot_angle': unit = "(deg)"
            elif key == 'jitter': unit = "(mm/frame^2)"
            print(f"  {key} {unit}: {avg_metrics[key]:.4f} (from {len(valid_values)}/{len(values)} valid samples)")
        else:
            avg_metrics[key] = float('nan')
            print(f"  {key}: NaN (no valid samples)")

    return avg_metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate EgoMotion Model')
    parser.add_argument('--config', type=str, default='configs/TransPose_train.yaml', help='Path to the configuration file.')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the trained model checkpoint. Overrides config if provided.')
    parser.add_argument('--smpl_model_path', type=str, default=None, help='Path to the SMPL model file (e.g., SMPLH neutral). Overrides config if provided.')
    parser.add_argument('--test_data_dir', type=str, default=None, help='Path to the test dataset directory. Overrides config if provided.')
    parser.add_argument('--batch_size', type=int, default=None, help='Test batch size. Overrides config if provided.')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of dataloader workers. Overrides config if provided.')
    parser.add_argument('--no_eval_objects', action='store_true', help='Do not evaluate object pose errors.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Override config values with command line arguments if provided
    if args.model_path: config.model_path = args.model_path
    if args.smpl_model_path: config.bm_path = args.smpl_model_path
    if args.test_data_dir: config.test.data_path = args.test_data_dir
    if args.batch_size: config.test.batch_size = args.batch_size
    if args.num_workers: config.num_workers = args.num_workers

    # Validate paths from config
    smpl_model_path = config.get('bm_path', 'body_models/smplh/neutral/model.npz')
    if not os.path.exists(smpl_model_path):
       print(f"Error: SMPL model path not found: {smpl_model_path}")
       print("Please provide the correct path in the config (smpl_model_path) or via --smpl_model_path.")
       return
    print(f"Loading SMPL model from: {smpl_model_path}")
    smpl_model = load_smpl_model(smpl_model_path, device)

    model_path = config.get('model_path', None)
    if model_path is None or not os.path.exists(model_path):
        print(f"Error: Evaluation model path not found or invalid: {model_path}")
        print("Please provide the correct path in the config (model_path) or via --model_path.")
        return

    # Determine model type for logging message (can be refined)
    model_type_str = "Unknown"
    if config.get('use_transpose_model', False): model_type_str = "TransPose"
    elif config.get('use_transpose_humanOnly_model', False): model_type_str = "TransPose HumanOnly"
    elif config.model.get('use_dit_model', True): model_type_str = "DiT"
    print(f"Loading {model_type_str} model from: {model_path}")
    model = load_model(model_path, config, device)

    test_data_dir = config.test.get('data_path', None)
    if test_data_dir is None or not os.path.exists(test_data_dir):
        print(f"Error: Test dataset path not found or invalid: {test_data_dir}")
        print("Please provide the correct path in the config (test.data_path) or via --test_data_dir.")
        return
    print(f"Loading test dataset from: {test_data_dir}")

    # Use test window size from config
    test_window_size = config.test.get('window', config.train.get('window', 60))
    test_dataset = IMUDataset(
            data_dir=test_data_dir,
            window_size=test_window_size,
            window_stride=config.test.get('window_stride', test_window_size),
            normalize=config.test.get('normalize', True),
            debug=config.get('debug', False)
        )

    if len(test_dataset) == 0:
         print("Error: Test dataset is empty. Check data path and dataset parameters.")
         return

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test.get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=False
    )

    print(f"\nDataset size: {len(test_dataset)}, Loader size: {len(test_loader)}")
    print(f"Batch size: {config.test.get('batch_size', 32)}, Num workers: {config.get('num_workers', 4)}")
    print(f"Evaluating objects: {not args.no_eval_objects}")

    print("\nStarting model evaluation...")
    results = evaluate_model(model, smpl_model, test_loader, device, evaluate_objects=(not args.no_eval_objects))

    print("\n--- Evaluation Results ---")
    print(f"MPJPE (mm):                                   {results.get('mpjpe', 'N/A'):.4f}")
    print(f"MPJRE (deg):                                  {results.get('mpjre_angle', 'N/A'):.4f}")
    if not args.no_eval_objects:
        print(f"Obj Trans Error (mm):                         {results.get('obj_trans_err', 'N/A'):.4f}")
        print(f"Obj Rot Error (deg):                          {results.get('obj_rot_angle', 'N/A'):.4f}")
    else:
         print("Object metrics skipped.")
    print(f"Jitter (mm/frame^2):                          {results.get('jitter', 'N/A'):.4f}")
    print("--------------------------")

if __name__ == "__main__":
    main()