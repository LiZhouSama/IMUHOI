r"""
    Preprocess DIP-IMU and TotalCapture test dataset.
    Synthesize AMASS dataset.

    Please refer to the `paths` in `config.py` and set the path of each dataset correctly.
"""

import sys
import os
# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加上上层目录到系统路径
root_dir = os.path.abspath(os.path.join(current_dir, "../../tasks/EgoIMU"))
# root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)
print(root_dir) 

import articulate as art
import torch
import pickle
from config import paths, amass_data
import numpy as np
from tqdm import tqdm
import glob
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.rotation_tools import local2global_pose
import pytorch3d.transforms as transforms
import joblib
FRAME_RATE = 1
def process_amass(smooth_n=4):

    def _syn_acc(v):
        r"""
        Synthesize accelerations from vertex positions.
        """
        mid = smooth_n // 2
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        if mid != 0:
            acc[smooth_n:-smooth_n] = torch.stack(
                [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
                 for i in range(0, v.shape[0] - smooth_n * 2)])
        return acc

    vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
    ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])
    body_model = art.ParametricModel(paths.smpl_file)

    data_pose, data_trans, data_beta, length = [], [], [], []
    for ds_name in amass_data:
        print('\rReading', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(paths.raw_amass_dir, ds_name, '*/*_poses.npz'))):
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])
            if framerate == 120: step = 2
            elif framerate == 60 or framerate == 59: step = 1
            else: continue

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])

    assert len(data_pose) != 0, 'AMASS dataset not found. Check config.py or comment the function process_amass()'
    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
    pose[:, 23] = pose[:, 37]     # right hand
    pose = pose[:, :24].clone()   # only use body

    # align AMASS global fame with DIP
    amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
    tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
    pose[:, 0] = art.math.rotation_matrix_to_axis_angle(
        amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0])))

    print('Synthesizing IMU accelerations and orientations')
    b = 0
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
    for i, l in tqdm(list(enumerate(length))):
        if l <= 12: b += l; print('\tdiscard one sequence with length', l); continue
        p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
        grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
        out_pose.append(pose[b:b + l].clone())  # N, 24, 3
        out_tran.append(tran[b:b + l].clone())  # N, 3
        out_shape.append(shape[i].clone())  # 10
        out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
        out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
        out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3
        b += l

    print('Saving')
    os.makedirs(paths.amass_dir, exist_ok=True)
    torch.save(out_pose, os.path.join(paths.amass_dir, 'pose.pt'))
    torch.save(out_shape, os.path.join(paths.amass_dir, 'shape.pt'))
    torch.save(out_tran, os.path.join(paths.amass_dir, 'tran.pt'))
    torch.save(out_joint, os.path.join(paths.amass_dir, 'joint.pt'))
    torch.save(out_vrot, os.path.join(paths.amass_dir, 'vrot.pt'))
    torch.save(out_vacc, os.path.join(paths.amass_dir, 'vacc.pt'))
    print('Synthetic AMASS dataset is saved at', paths.amass_dir)

def process_dipimu():
    imu_mask = [7, 8, 11, 12, 0, 2]
    test_split = ['s_09', 's_10']
    accs, oris, poses, trans = [], [], [], []

    for subject_name in test_split:
        for motion_name in os.listdir(os.path.join(paths.raw_dipimu_dir, subject_name)):
            path = os.path.join(paths.raw_dipimu_dir, subject_name, motion_name)
            data = pickle.load(open(path, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
            ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
            pose = torch.from_numpy(data['gt']).float()

            # fill nan with nearest neighbors
            for _ in range(4):
                acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

            acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
            if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                accs.append(acc.clone())
                oris.append(ori.clone())
                poses.append(pose.clone())
                trans.append(torch.zeros(pose.shape[0], 3))  # dip-imu does not contain translations
            else:
                print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))

    os.makedirs(paths.dipimu_dir, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans}, os.path.join(paths.dipimu_dir, 'test.pt'))
    print('Preprocessed DIP-IMU dataset is saved at', paths.dipimu_dir)

def process_totalcapture():
    inches_to_meters = 0.0254
    file_name = 'gt_skel_gbl_pos.txt'

    accs, oris, poses, trans = [], [], [], []
    for file in sorted(os.listdir(paths.raw_totalcapture_dip_dir)):
        data = pickle.load(open(os.path.join(paths.raw_totalcapture_dip_dir, file), 'rb'), encoding='latin1')
        ori = torch.from_numpy(data['ori']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
        acc = torch.from_numpy(data['acc']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
        pose = torch.from_numpy(data['gt']).float().view(-1, 24, 3)

        # acc/ori and gt pose do not match in the dataset
        if acc.shape[0] < pose.shape[0]:
            pose = pose[:acc.shape[0]]
        elif acc.shape[0] > pose.shape[0]:
            acc = acc[:pose.shape[0]]
            ori = ori[:pose.shape[0]]

        assert acc.shape[0] == ori.shape[0] and ori.shape[0] == pose.shape[0]
        accs.append(acc)    # N, 6, 3
        oris.append(ori)    # N, 6, 3, 3
        poses.append(pose)  # N, 24, 3

    for subject_name in ['S1', 'S2', 'S3', 'S4', 'S5']:
        for motion_name in sorted(os.listdir(os.path.join(paths.raw_totalcapture_official_dir, subject_name))):
            if subject_name == 'S5' and motion_name == 'acting3':
                continue   # no SMPL poses
            f = open(os.path.join(paths.raw_totalcapture_official_dir, subject_name, motion_name, file_name))
            line = f.readline().split('\t')
            index = torch.tensor([line.index(_) for _ in ['LeftFoot', 'RightFoot', 'Spine']])
            pos = []
            while line:
                line = f.readline()
                pos.append(torch.tensor([[float(_) for _ in p.split(' ')] for p in line.split('\t')[:-1]]))
            pos = torch.stack(pos[:-1])[:, index] * inches_to_meters
            pos[:, :, 0].neg_()
            pos[:, :, 2].neg_()
            trans.append(pos[:, 2] - pos[:1, 2])   # N, 3

    # match trans with poses
    for i in range(len(accs)):
        if accs[i].shape[0] < trans[i].shape[0]:
            trans[i] = trans[i][:accs[i].shape[0]]
        assert trans[i].shape[0] == accs[i].shape[0]

    os.makedirs(paths.totalcapture_dir, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans},
               os.path.join(paths.totalcapture_dir, 'test.pt'))
    print('Preprocessed TotalCapture dataset is saved at', paths.totalcapture_dir)

def _syn_acc(v, smooth_n=4):
    """从位置生成加速度"""
    mid = smooth_n // 2
    # 基础二阶差分计算
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * FRAME_RATE ** 2 for i in range(0, v.shape[0] - 2)])
    # 增加边界填充
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    
    # 平滑处理
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * FRAME_RATE ** 2 / smooth_n ** 2
                for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc

def compute_imu_data(position_global, rotation_global, imu_joints, smooth_n=4):
    """
    计算特定关节的IMU数据（加速度和方向）
    
    参数:
        position_global: 全局关节位置 [T, J, 3]
        rotation_global: 全局关节旋转 [T, J, 3, 3]
        imu_joints: IMU关节索引列表
        smooth_n: 平滑窗口大小
    
    返回:
        IMU数据字典，包含加速度和方向信息
    """
    device = position_global.device
    
    # 提取指定关节的位置和旋转
    imu_positions = position_global[:, imu_joints, :]  # [T, num_imus, 3]
    imu_orientations = rotation_global[:, imu_joints, :, :]  # [T, num_imus, 3, 3]
    
    T = imu_positions.shape[0]
    num_imus = len(imu_joints)
    
    
    # 并行计算所有IMU关节的加速度
    imu_accelerations = torch.zeros_like(imu_positions)
    for i in range(num_imus):
        imu_accelerations[:, i, :] = _syn_acc(imu_positions[:, i, :], smooth_n=4)
   
    # 返回IMU数据字典，包含加速度和方向
    imu_data = {
        'accelerations': imu_accelerations,  # [T, num_imus, 3]
        'orientations': imu_orientations   # [T, num_imus, 3, 3]
    }
    
    return imu_data

def process_omomo():
    device = 'cpu'
    try:
        data_dict = joblib.load(paths.raw_omomo_dir)
        os.makedirs(paths.omomo_dir, exist_ok=True)
    except Exception as e:
        print(f"加载数据出错: {e}")
        return
        
    # bm = BodyModel(
    #     bm_fname=paths.bm_fname_male,
    #     num_betas=16,
    # ).to(device)

    body_model = art.ParametricModel(paths.smpl_file)
    
    accs, oris, poses, trans = [], [], [], []
    IMU_JOINTS = torch.tensor([18, 19, 4, 5, 15, 0])
    
    for seq_key in tqdm(data_dict, desc="处理序列"):
        try:
            # if seq_key == 50:
            #     break
            seq_data = data_dict[seq_key]
            # 提取序列数据
            bdata_poses = np.concatenate([seq_data['root_orient'].reshape(-1, 3), 
                                        seq_data['pose_body'].reshape(-1, 63)], axis=1)
            bdata_poses24 = np.concatenate([bdata_poses, np.zeros((bdata_poses.shape[0], 6))], axis=1)
            bdata_trans = seq_data['trans']
            
            # 转换为PyTorch张量
            pose_tensor = torch.tensor(bdata_poses24, device=device).float().reshape(-1, 24, 3)
            trans_tensor = torch.tensor(bdata_trans, device=device).float()

            # 获取形状参数，如果有
            shape_tensor = torch.tensor(seq_data['betas'][:10], device=device).float() if 'betas' in seq_data else torch.zeros(10, device=device)

            # 将轴角表示转换为旋转矩阵
            pose_rotmat = art.math.axis_angle_to_rotation_matrix(pose_tensor).view(-1, 24, 3, 3)

            # 使用body_model.forward_kinematics计算全局旋转矩阵、关节位置和顶点位置
            grot, joint, vert = body_model.forward_kinematics(pose_rotmat, None, trans_tensor, calc_mesh=True)
            
            # 计算IMU数据
            imu_accs = _syn_acc(joint[:, IMU_JOINTS])  # 计算IMU加速度
            
            # 添加到结果列表
            accs.append(imu_accs.cpu())
            oris.append(grot[:, IMU_JOINTS].cpu())  # 提取IMU方向
            poses.append(pose_tensor.cpu())
            trans.append(trans_tensor.cpu())
            
        except Exception as e:
            print(f"处理序列 {seq_key} 时出错: {e}")
    
    # 保存处理后的数据
    if len(accs) > 0:
        torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans},
                  os.path.join(paths.omomo_dir, 'test.pt'))
        print('预处理后的OMOMO数据集保存在', paths.omomo_dir)
    else:
        print('没有找到有效的OMOMO数据！')


if __name__ == '__main__':
    # process_amass()
    # process_dipimu()
    # process_totalcapture()
    process_omomo()  # 添加OMOMO数据集处理
