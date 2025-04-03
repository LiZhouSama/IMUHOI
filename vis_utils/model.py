import torch
import torch.nn as nn
from vis_utils import articulate as art

class Poser(nn.Module):
    """
    简化版的Poser模型，仅用于可视化
    """
    def __init__(self):
        super(Poser, self).__init__()
        
    def _glb_mat_xsens_to_glb_mat_smpl(self, glb_full_pose_xsens):
        """
        将Xsens格式的全局旋转矩阵转换为SMPL格式
        
        参数:
            glb_full_pose_xsens: Xsens格式的全局旋转矩阵
            
        返回:
            glb_full_pose_smpl: SMPL格式的全局旋转矩阵
        """
        # 检查输入张量的维度，确定是否为SMPLX模型
        if glb_full_pose_xsens.shape[1] > 24:  # SMPLX有更多的关节
            # 直接返回，假设已经是兼容格式
            return glb_full_pose_xsens
        
        # 标准SMPL模型处理
        glb_full_pose_smpl = torch.eye(3, device=glb_full_pose_xsens.device).repeat(glb_full_pose_xsens.shape[0], 24, 1, 1)
        
        # Xsens到SMPL的关节映射
        try:
            indices = [0, 19, 15, 1, 20, 16, 3, 21, 17, 4, 22, 18, 5, 11, 7, 6, 12, 8, 13, 9, 13, 9, 13, 9]
            for idx, i in enumerate(indices):
                if i < glb_full_pose_xsens.shape[1]:
                    glb_full_pose_smpl[:, idx, :] = glb_full_pose_xsens[:, i, :]
        except Exception as e:
            print(f"映射Xsens到SMPL格式时出错: {e}")
            # 如果映射失败，返回原始张量
            return glb_full_pose_xsens
                    
        return glb_full_pose_smpl
    
    def _reduced_glb_6d_to_full_glb_mat_xsens(self, glb_reduced_pose, orientation):
        """
        将简化的6D姿态和朝向转换为Xsens格式的全局旋转矩阵
        
        参数:
            glb_reduced_pose: 简化的6D姿态
            orientation: 朝向矩阵
            
        返回:
            global_full_pose: Xsens格式的全局旋转矩阵
        """
        try:
            joint_set = [19, 15, 1, 2, 3, 4, 5, 11, 7, 12, 8]
            sensor_set = [0, 20, 16, 6, 13, 9]
            ignored = [10, 14, 17, 18, 21, 22]
            parent = [9, 13, 16, 16, 20, 20]
            
            root_rotation = orientation[:, 0].view(-1, 3, 3)
            glb_reduced_pose = art.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, len(joint_set), 3, 3)
            
            # 转换到全局坐标系
            glb_reduced_pose = root_rotation.unsqueeze(1).matmul(glb_reduced_pose)
            orientation[:, 1:] = root_rotation.unsqueeze(1).matmul(orientation[:, 1:])
            
            # 初始化全局姿态
            global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 23, 1, 1)
            
            # 填充旋转矩阵
            global_full_pose[:, joint_set] = glb_reduced_pose
            global_full_pose[:, sensor_set] = orientation
            global_full_pose[:, ignored] = global_full_pose[:, parent]
        except Exception as e:
            print(f"转换6D姿态到Xsens格式时出错: {e}")
            # 如果转换失败，尝试直接使用姿态数据
            return orientation
            
        return global_full_pose
    
    def predict(self, outputs, data):
        """
        处理EgoIMU模型的输出，生成SMPL格式的全局旋转矩阵
        
        参数:
            outputs: 模型输出
            data: 输入数据
            
        返回:
            glb_full_pose_smpl: SMPL格式的全局旋转矩阵
        """
        # 这里假设outputs已经包含了必要的姿态信息
        # 根据实际EgoIMU模型调整此处逻辑
        
        # 尝试多种可能的输出格式
        try:
            # 示例：直接使用outputs作为全局姿态
            if isinstance(outputs, torch.Tensor):
                if outputs.dim() == 4 and outputs.shape[2:] == (3, 3):
                    # 如果outputs已经是旋转矩阵形式
                    glb_full_pose_smpl = outputs
                else:
                    # 假设outputs需要转换为旋转矩阵
                    glb_full_pose_smpl = art.r6d_to_rotation_matrix(outputs).view(outputs.shape[0], -1, 3, 3)
            elif isinstance(outputs, dict):
                # 如果outputs是字典
                pose_key = None
                for key in ['smpl_poses', 'pose', 'poses', 'body_pose', 'full_pose']:
                    if key in outputs:
                        pose_key = key
                        break
                
                if pose_key:
                    glb_full_pose_smpl = outputs[pose_key]
                    
                    # 转换成旋转矩阵如果不是
                    if glb_full_pose_smpl.dim() != 4 or glb_full_pose_smpl.shape[2:] != (3, 3):
                        glb_full_pose_smpl = art.r6d_to_rotation_matrix(glb_full_pose_smpl).view(glb_full_pose_smpl.shape[0], -1, 3, 3)
                else:
                    # 尝试从data中找姿态信息
                    if 'joint' in data and 'full smpl pose' in data['joint']:
                        glb_full_pose_smpl = data['joint']['full smpl pose']
                    else:
                        # 创建默认姿态（单位旋转矩阵）
                        print("无法从输出中找到姿态信息，使用默认姿态")
                        glb_full_pose_smpl = torch.eye(3).repeat(1, 24, 1, 1)
            else:
                # 对象有属性
                pose_attrs = ['smpl_poses', 'pose', 'poses', 'body_pose', 'full_pose']
                for attr in pose_attrs:
                    if hasattr(outputs, attr):
                        glb_full_pose_smpl = getattr(outputs, attr)
                        
                        # 转换成旋转矩阵如果不是
                        if glb_full_pose_smpl.dim() != 4 or glb_full_pose_smpl.shape[2:] != (3, 3):
                            glb_full_pose_smpl = art.r6d_to_rotation_matrix(glb_full_pose_smpl).view(glb_full_pose_smpl.shape[0], -1, 3, 3)
                        break
                else:
                    # 如果没有找到任何姿态属性
                    print("无法从输出对象属性中找到姿态信息，使用默认姿态")
                    glb_full_pose_smpl = torch.eye(3).repeat(1, 24, 1, 1)
        except Exception as e:
            print(f"处理模型输出时出错: {e}")
            # 创建默认姿态
            glb_full_pose_smpl = torch.eye(3).repeat(1, 24, 1, 1)
        
        return glb_full_pose_smpl 