import sys
import os
sys.path.append("../")
import torch.nn
from torch.nn.functional import relu, tanh
import torch
from configs.global_config import FRAME_RATE


class RNN(torch.nn.Module):
    """
    RNN模块，包括线性输入层、RNN和线性输出层。
    """
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=True, dropout=0.2):
        super(RNN, self).__init__()
        self.n_hidden = n_hidden
        self.n_rnn_layer = n_rnn_layer
        self.num_directions = 2 if bidirectional else 1
        # Set batch_first=True for LSTM
        self.rnn = torch.nn.LSTM(n_hidden, n_hidden, n_rnn_layer, bidirectional=bidirectional, batch_first=True)
        self.linear1 = torch.nn.Linear(n_input, n_hidden)
        self.linear2 = torch.nn.Linear(n_hidden * self.num_directions, n_output)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, h=None):
        # Input x is expected to be [batch_size, seq_len, n_input] due to batch_first=True
        # No need to unsqueeze/squeeze if input has the correct shape
        x = self.linear1(self.dropout(x))
        # Pass initial hidden state h (h_0, c_0) if provided
        x, h_out = self.rnn(x, h)
        # Output x is [batch_size, seq_len, n_hidden * num_directions]
        # Apply linear layer to each time step's output
        # Reshape x to [batch_size * seq_len, n_hidden * num_directions] before linear layer
        # Then reshape back to [batch_size, seq_len, n_output]
        batch_size, seq_len, _ = x.shape
        x = x.reshape(-1, self.n_hidden * self.num_directions)
        x = self.linear2(x)
        x = x.reshape(batch_size, seq_len, -1)

        return x, h_out # Return sequence output and final hidden state


class ObjectTransModule(torch.nn.Module):
    """
    模块3: 物体方向向量估计和位置重建
    输入: 物体imu
    输出: obj_trans
    """
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        hidden_dim_multiplier = cfg.hidden_dim_multiplier if hasattr(cfg, 'hidden_dim_multiplier') else 1
        
        # 物体速度估计网络
        self.obj_velocity_net = RNN(3, 3, 128 * hidden_dim_multiplier, bidirectional=False)
        
    
    def forward(self, obj_imu_acc, obj_trans_start):
        
        # 物体速度估计
        pred_obj_vel, _ = self.obj_velocity_net(obj_imu_acc)
        
        pred_obj_trans = torch.cumsum(pred_obj_vel / FRAME_RATE, dim=1) + obj_trans_start.unsqueeze(1).expand(-1, obj_imu_acc.shape[1], -1)
        
        obj_imu_acc_watch = obj_imu_acc.detach().cpu().numpy()
        pred_obj_vel_watch = pred_obj_vel.detach().cpu().numpy()
        pred_obj_trans_watch = pred_obj_trans.detach().cpu().numpy()

        return {
            "pred_obj_vel": pred_obj_vel,                               # [bs, seq, 3] - 物体速度:米/秒
            "pred_obj_trans": pred_obj_trans,                           # [bs, seq, 3] - 物体位置
        }



class TransPoseNet(torch.nn.Module):
    """
    适用于EgoIMU项目的TransPose网络架构，支持分阶段训练和模块化加载
    """
    def __init__(self, cfg, pretrained_modules=None, skip_modules=None):
        """
        """
        super().__init__()
        
        # 从配置中获取参数
        self.num_human_imus = cfg.num_human_imus if hasattr(cfg, 'num_human_imus') else 6
        self.imu_dim = cfg.imu_dim if hasattr(cfg, 'imu_dim') else 9
        self.joint_dim = cfg.joint_dim if hasattr(cfg, 'joint_dim') else 6
        self.num_joints = cfg.num_joints if hasattr(cfg, 'num_joints') else 22
        
        # 设置设备
        if hasattr(cfg, 'device'):
            self.device = torch.device(cfg.device)
        elif hasattr(cfg, 'gpus') and cfg.gpus:
            self.device = torch.device(f"cuda:{cfg.gpus[0]}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始状态输入维度
        n_obj_trans_start = 3
        
        # 初始化默认值
        if pretrained_modules is None:
            pretrained_modules = {}
        if skip_modules is None:
            skip_modules = []
        
        self.object_trans_module = ObjectTransModule(cfg, self.device)
        if 'object_trans' in pretrained_modules:
            print(f"正在加载预训练的object_trans模块: {pretrained_modules['object_trans']}")
            self.object_trans_module = self._load_single_module(
                pretrained_modules['object_trans'], 'object_trans', cfg
            )
        else:
            print("正在初始化新的object_trans模块")
            self.object_trans_module = ObjectTransModule(cfg, self.device)
        
        print(f"TransPoseNet初始化完成:")
        print(f"  - object_trans_module: {'从预训练加载' if 'object_trans' in pretrained_modules else '新初始化' if 'object_trans' not in skip_modules else '跳过初始化'}")
    
    def _load_single_module(self, checkpoint_path, module_name, cfg):
        """加载单个模块的预训练权重"""
        try:
            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 验证模块名称
            if 'module_name' in checkpoint and checkpoint['module_name'] != module_name:
                print(f"警告: 检查点中的模块名称 '{checkpoint['module_name']}' 与请求的模块名称 '{module_name}' 不匹配")
            
            # 创建模块实例
            if module_name == 'object_trans':
                module = ObjectTransModule(cfg, self.device)
            else:
                raise ValueError(f"未知的模块名称: {module_name}")
            
            # 加载权重
            if 'module_state_dict' in checkpoint:
                module.load_state_dict(checkpoint['module_state_dict'])
            elif 'state_dict' in checkpoint:
                module.load_state_dict(checkpoint['state_dict'])
            else:
                # 尝试直接加载整个检查点作为state_dict
                module.load_state_dict(checkpoint)
            
            print(f"成功加载{module_name}模块，来自epoch {checkpoint.get('epoch', 'unknown')}")
            return module
            
        except Exception as e:
            print(f"加载{module_name}模块失败: {e}")
            print(f"将初始化新的{module_name}模块")
            # 如果加载失败，则初始化新模块
            if module_name == 'object_trans':
                return ObjectTransModule(cfg, self.device)



    def get_module_state_dict(self, module_name):
        """获取指定模块的状态字典"""
        if module_name == 'velocity_contact' and self.velocity_contact_module is not None:
            return self.velocity_contact_module.state_dict()
        elif module_name == 'human_pose' and self.human_pose_module is not None:
            return self.human_pose_module.state_dict()
        elif module_name == 'object_trans' and self.object_trans_module is not None:
            return self.object_trans_module.state_dict()
        else:
            return None
    
    def save_module(self, module_name, save_path, epoch, additional_info=None):
        """保存单个模块"""
        module_state_dict = self.get_module_state_dict(module_name)
        if module_state_dict is None:
            print(f"模块 {module_name} 不存在，无法保存")
            return False
        
        checkpoint_data = {
            'module_name': module_name,
            'module_state_dict': module_state_dict,
            'epoch': epoch,
        }
        
        if additional_info:
            checkpoint_data.update(additional_info)
        
        try:
            torch.save(checkpoint_data, save_path)
            print(f"成功保存{module_name}模块到: {save_path}")
            return True
        except Exception as e:
            print(f"保存{module_name}模块失败: {e}")
            return False
    
    def format_input(self, data_dict):
        """格式化输入数据"""
        obj_imu = data_dict.get("obj_imu", None)
        obj_trans = data_dict.get("obj_trans", None)
        obj_imu_acc = obj_imu[...,:3]
        
        batch_size, seq_len = obj_imu.shape[:2]
        
        # 处理IMU数据
        obj_imu_acc_flat_seq = obj_imu_acc.reshape(batch_size, seq_len, -1)
        
        # 处理第一帧状态信息
        obj_trans_start = obj_trans[:, 0]
        
        return obj_imu_acc_flat_seq, obj_trans_start
    
    def forward(self, data_dict, use_object_data=True):
        """
        前向传播
        
        Args:
            data_dict: 包含输入数据的字典，可以包含'use_object_data'键
            use_object_data: 是否使用物体数据（用于分阶段训练），如果为None则从data_dict中获取，默认为True
        
        Returns:
            dict: 包含预测结果的字典
        """
        # 从data_dict中获取use_object_data，如果没有则使用参数值，如果参数也为None则默认为True
        # 格式化输入数据
        obj_imu_data, obj_trans_start = self.format_input(data_dict)
        
        # 初始化结果字典
        results = {}
        object_trans_outputs = self.object_trans_module(
            obj_imu_data,
            obj_trans_start
        )
        results.update(object_trans_outputs)
        
        return results 
