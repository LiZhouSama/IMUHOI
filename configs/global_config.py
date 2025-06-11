# IMU关节索引，可以根据需要修改
IMU_JOINTS = [20, 21, 7, 8, 15, 0]  # 左手、右手、左脚、右脚、髋部、头部
IMU_JOINT_NAMES = ['left_hand', 'right_hand', 'left_foot', 'right_foot', 'head', 'hip']
HEAD_IDX = 4  # 头部在IMU_JOINTS列表中的索引
FRAME_RATE = 60  # 帧率