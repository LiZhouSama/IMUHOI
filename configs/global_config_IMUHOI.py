import torch

# Sensor indices follow the original DynaIP/Xsens convention
_SENSOR_POS_INDICES = [0, 7, 8, 15, 20, 21]
_SENSOR_ROT_INDICES = [0, 4, 5, 15, 18, 19]
_VEL_SELECTION_INDICES = torch.tensor([0, 7, 8, 15, 20, 21], dtype=torch.long)
_REDUCED_INDICES = [1, 2, 3, 6, 9, 12, 13, 14, 16, 17]
_IGNORED_INDICES = [7, 8, 10, 11, 20, 21, 22, 23]

_SENSOR_NAMES = ['Root', 'LeftLowerLeg', 'RightLowerLeg', 'Head', 'LeftForeArm', 'RightForeArm']
_SENSOR_VEL_NAMES = ['Root', 'LeftFoot', 'RightFoot', 'Head', 'LeftHand', 'RightHand']
_REDUCED_POSE_NAMES = ['LeftHip', 'RightHip', 'Spine1', 'Spine2', 'Spine3', 'Neck', 
                     'LeftCollar', 'RightCollar', 'LeftShoulder', 'RightShoulder']

FRAME_RATE = 30 