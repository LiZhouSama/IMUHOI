r"""
    Config for paths, joint set, and normalizing scales.
"""


# datasets (directory names) in AMASS
# e.g., for ACCAD, the path should be `paths.raw_amass_dir/ACCAD/ACCAD/s001/*.npz`
amass_data = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh', 'Transitions_mocap', 'SSM_synced', 'CMU',
              'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BMLmovi', 'EKUT', 'TCD_handMocap', 'ACCAD',
              'BioMotionLab_NTroje', 'BMLhandball', 'MPI_Limits', 'DFaust67']


class paths:
    raw_amass_dir = '/mnt/d/a_WORK/Projects/PhD/datasets/AMASS_SMPL_H'      # raw AMASS dataset path (raw_amass_dir/ACCAD/ACCAD/s001/*.npz)
    amass_dir = 'data/dataset_work/AMASS'         # output path for the synthetic AMASS dataset

    raw_dipimu_dir = '/mnt/d/a_WORK/Projects/PhD/datasets/DIP/DIP_IMU'   # raw DIP-IMU dataset path (raw_dipimu_dir/s_01/*.pkl)
    dipimu_dir = 'data/dataset_work/DIP_IMU'      # output path for the preprocessed DIP-IMU dataset

    # DIP recalculates the SMPL poses for TotalCapture dataset. You should acquire the pose data from the DIP website.
    raw_totalcapture_dip_dir = 'data/dataset_raw/TotalCapture/DIP_recalculate'  # contain ground-truth SMPL pose (*.pkl)
    raw_totalcapture_official_dir = 'data/dataset_raw/TotalCapture/official'    # contain official gt (S1/acting1/gt_skel_gbl_pos.txt)
    totalcapture_dir = 'data/dataset_work/TotalCapture'          # output path for the preprocessed TotalCapture dataset

    # OMOMO dataset paths
    raw_omomo_dir = '/mnt/d/a_WORK/Projects/PhD/tasks/EgoIMU/dataset/test_diffusion_manip_seq_joints24.p'     # raw OMOMO dataset path (raw_omomo_dir/test/*.pkl)
    omomo_dir = 'data/dataset_work/OMOMO'        # output path for the preprocessed OMOMO dataset
    bm_fname_male = '/mnt/d/a_WORK/Projects/PhD/tasks/EgoIMU/body_models/smplh/male/model.npz'
    
    example_dir = 'data/example'                    # example IMU measurements
    smpl_file = 'models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'              # official SMPL model path
    weights_file = 'data/weights.pt'                # network weight file


class joint_set:
    leaf = [7, 8, 12, 20, 21]
    full = list(range(1, 24))
    reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]

    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]

    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)


acc_scale = 30
vel_scale = 3
