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