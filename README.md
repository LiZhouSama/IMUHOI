# EgoIMU: 基于IMU的全身姿态估计

EgoIMU是一个利用少量IMU传感器来估计全身姿态的深度学习框架。该框架使用扩散模型和Transformer架构，从IMU数据中直接生成SMPL模型的全身姿态参数。

## 功能特点

- 利用少量IMU传感器数据（仅7个关节点）合成全身动作
- 使用单阶段扩散模型直接生成高质量人体姿态
- 支持物体交互场景，通过优化的BPS (Basis Point Set) 特征表示物体
- 模块化设计，易于扩展和定制
- 高效的数据处理和批量训练

## 项目结构

```
EgoIMU/
├── train.py             # 主训练脚本
├── eval.py              # 评估脚本
├── modules/
│   ├── diffusion.py     # 扩散模型定义
│   ├── transformer.py   # Transformer模型定义
│   ├── model.py         # 完整模型架构
├── utils/
│   ├── data_utils.py    # 数据处理工具
│   ├── trainer.py       # 训练器
├── datasets/
│   ├── imu_dataset.py   # 数据集定义
└── configs/
    └── default.yaml     # 默认配置文件
```

## 安装依赖

```bash
pip install torch numpy pytorch3d tqdm pyyaml
```

## 数据准备

在开始训练之前，需要使用预处理脚本将原始数据转换为训练所需的`.pt`文件：

```bash
python preprocess.py --data_path dataset/train_diffusion_manip_seq_joints24.p --save_dir processed_data_0322/train
```

## 训练模型

### 使用默认配置训练：

```bash
python train.py --train_dir processed_data_0322/train --val_dir processed_data_0322/val --output_dir output
```

### 使用配置文件训练：

```bash
python train.py --config configs/default.yaml
```

### 恢复训练：

```bash
python train.py --config configs/default.yaml --resume output/checkpoints/model_epoch_50.pt
```

## 评估模型

```bash
python eval.py --test_dir processed_data_0322/test --checkpoint output/checkpoints/model_best.pt --config output/config.yaml --output_dir evaluation_results
```

## 主要参数

### 训练参数

- `--train_dir`: 训练数据目录
- `--val_dir`: 验证数据目录
- `--window_size`: 序列窗口大小（默认120帧）
- `--batch_size`: 批次大小
- `--num_epochs`: 训练轮次
- `--lr`: 学习率
- `--use_object`: 是否使用物体信息

### 模型参数

- `--hidden_dim`: 隐藏层维度
- `--num_layers`: Transformer层数
- `--diffusion_steps`: 扩散步数

## 数据格式

输入的`.pt`文件应包含以下键：

- `seq_name`: 序列名称
- `body_parms_list`: 人体SMPL参数
- `imu_global_full_gt`: 全局IMU数据
- `rotation_local_full_gt_list`: 局部旋转数据
- `position_global_full_gt_world`: 全局位置数据
- `obj_trans`, `obj_rot`, `obj_com_pos`: 物体信息（如有）
- `framerate`, `gender`: 元数据

## 引用

如果您在研究中使用了本项目，请引用：

```
@misc{EgoIMU2023,
  author = {Your Name},
  title = {EgoIMU: Full-body Motion Estimation from Sparse IMU Data},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/EgoIMU}}
}
``` 