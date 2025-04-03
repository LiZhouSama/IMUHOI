# EgoIMU: IMU到全身姿态的Diffusion模型

本项目使用Diffusion模型从IMU传感器数据生成全身人体姿态。该模型直接从IMU数据中学习全身姿态，无需额外的VQVAE中间表示。

## 项目结构

```
├── config_diffusion/          # Diffusion模型配置
│   └── imu.yaml               # IMU到全身姿态的配置
├── dataloader/                # 数据加载
│   └── dataloader.py          # 数据集和数据加载器实现
├── diffusion_stage/           # Diffusion模型实现
│   ├── do_train_imu.py        # IMU训练过程
│   ├── parser_util.py         # 配置解析工具
│   └── wrap_model.py          # Diffusion模型封装
├── preprocessed_data_0324/    # 预处理数据
│   ├── train/                 # 训练数据
│   ├── test/                  # 测试数据
│   └── bps_features/          # BPS特征
├── omomo/                     # 原始数据处理代码
│   └── preprocess.py          # 数据预处理
├── outputs/                   # 训练输出
│   └── imu/                   # 模型保存和可视化
├── test_imu.py                # 测试脚本
├── train_imu_diffusion.py     # 训练脚本
└── README.md                  # 项目说明
```

## 数据预处理

在训练前，需要先对原始数据进行预处理，生成IMU数据和BPS特征：

```bash
python omomo/preprocess.py --data_path dataset/train.p --save_dir preprocessed_data_0324/train --obj_mesh_dir dataset/objects --n_bps_points 1024 --num_workers 8
python omomo/preprocess.py --data_path dataset/test.p --save_dir preprocessed_data_0324/test --obj_mesh_dir dataset/objects --n_bps_points 1024 --num_workers 8
```

预处理步骤包括：
1. 计算关节的IMU数据（加速度和角速度）
2. 归一化IMU数据到头部坐标系
3. 计算物体的BPS特征和IMU数据
4. 保存为PT文件

## 训练模型

训练IMU到全身姿态的Diffusion模型：

```bash
python train_imu_diffusion.py --cfg config_diffusion/imu.yaml --batch_size 32
```

训练过程将自动：
1. 加载预处理的数据
2. 创建Diffusion模型
3. 训练并定期保存模型
4. 每5个epoch进行一次测试评估

## 测试模型

```bash
python test_imu.py --cfg config_diffusion/imu.yaml
```

测试脚本将：
1. 加载最佳模型
2. 在测试集上评估性能
3. 计算MPJPE（平均关节位置误差）
4. 生成姿态可视化

## 主要特点

- **直接预测**：从IMU数据直接生成全身姿态，无需中间表示
- **高效训练**：使用加速器和混合精度训练
- **BPS特征**：支持物体BPS特征增强IMU数据
- **可视化**：自动生成骨架可视化结果

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- SMPL模型
- diffusers
- accelerate
- matplotlib
- tqdm
- easydict

## 引用

项目基于SAGE框架改进：
```
@inproceedings{jiang2023sage,
  title={SAGE: Generating VR Hands from Head and Hand Motions for Interactive Mobile Immersion},
  author={Jiang, Feifei and Liu, Weiwei and Zou, Difei and Zhang, Qi and Cai, Zhongyu and Yao, Aishan and Zheng, Fang and Chen, Jian and Yu, Tianyi and Wang, Caiming},
  booktitle={ACM SIGGRAPH},
  year={2023}
}
```

# EgoIMU 可视化工具

本项目实现了基于IMU数据的人体和物体姿态预测，并提供了可视化工具。

## 目录结构

```
├── vis.py                  # 主可视化脚本
├── vis_utils/              # 可视化工具模块
│   ├── __init__.py         # 模块初始化文件
│   ├── articulate.py       # 简化版SMPL人体模型库
│   ├── config.py           # 配置文件
│   └── model.py            # 简化版模型定义
```

## 依赖安装

```bash
pip install numpy torch aitviewer
```

## 可视化工具使用说明

该工具基于aitviewer库实现，可以同时可视化人体和物体的姿态，包括真值和预测值。

### 命令行参数

```bash
python vis.py [--data_path DATA_PATH] [--model_path MODEL_PATH] [--no_objects]
```

- `--data_path`: 指定数据文件路径，如果不指定则从测试目录随机选择一个
- `--model_path`: 指定模型路径，用于在线推理生成预测结果
- `--no_objects`: 添加此参数则不显示物体

### 配置文件

可以通过修改`vis_utils/config.py`来配置常用路径：

```python
# SMPL模型路径
smpl_m = './smpl_models/smpl_male.pkl'
smpl_fm = './smpl_models/smpl_female.pkl'

# 测试数据路径
work_dir = './preprocessed_data_0324'

# 模型权重
model_path = './outputs/imu/checkpoints/best.pt'
```

### 数据格式支持

该工具支持多种数据格式：

1. 真值数据 (`data['joint']`):
   - SMPL姿态格式: `full smpl pose`
   - Xsens姿态格式: `full xsens pose`
   - SMPL标准格式: `global_orient` 和 `body_pose`

2. 预测数据 (`data['predictions']`):
   - 直接的姿态参数: 旋转矩阵或6D旋转表示
   - 字典格式: 包含`smpl_pose`或`pose`键
   - 顶点数据: 包含`vertices`键

3. 物体数据 (`data['objects']` 或 `data['object']`):
   - 顶点数据: `gt_vertices`, `vertices`, `verts` 等
   - 面片数据: `faces`, `face`, `f` 等
   - 预测顶点: `pred_vertices`, `pred_verts` 等

### 示例用法

```bash
# 随机选择一个测试文件进行可视化
python vis.py

# 指定数据文件路径
python vis.py --data_path /path/to/your/data.pt

# 使用模型进行在线推理并可视化
python vis.py --model_path /path/to/your/model.pt

# 仅可视化人体（不显示物体）
python vis.py --no_objects
```

### 可视化效果

- 预测人体姿态: 蓝色
- 真值人体姿态: 红色（向右偏移1米）
- 预测物体: 绿色
- 真值物体: 橙色（向右偏移1米，与真值人体姿态保持一致）

## 数据准备

需要确保您的数据包含以下字段：

```
data = {
    'imu': {
        'imu': [Tensor] # IMU数据
    },
    'joint': {
        'velocity': [Tensor], # 关节速度
        'orientation': [Tensor], # 关节朝向
        'full smpl pose' 或 'full xsens pose': [Tensor] # 完整姿态
    },
    'objects': { # 可选，如果需要可视化物体
        'gt_vertices': [Tensor或Array], # 物体真值顶点
        'pred_vertices': [Tensor或Array], # 物体预测顶点
        'faces': [Array] # 可选，物体面片索引
    }
}
``` 