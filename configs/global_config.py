# IMU关节索引，可以根据需要修改
IMU_JOINTS_POS = [20, 21, 7, 8, 15, 0]  #左手、右手、左脚、右脚、头部、髋部
IMU_JOINTS_ROT = [18, 19, 4, 5, 15, 0]  #左手、右手、左脚、右脚、头部、髋部
# IMU_JOINT_NAMES = ['left_foot', 'right_foot', 'head', 'left_hand', 'right_hand', 'hip']
IMU_JOINT_NAMES = ['left_hand', 'right_hand', 'left_foot', 'right_foot', 'head', 'hip']
# HEAD_IDX = 2  # 头部在IMU_JOINTS列表中的索引
FRAME_RATE = 30  # 帧率
acc_scale = 1
vel_scale = 1


class joint_set:
    # TransPose关节索引
    leaf = [7, 8, 12, 20, 21]
    full = list(range(1, 22))
    reduced = [1, 2, 3, 6, 9, 12, 13, 14, 16, 17, 18, 19, 4, 5, 15]
    ignored = [0, 7, 8, 10, 11, 20, 21]

    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]
    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)