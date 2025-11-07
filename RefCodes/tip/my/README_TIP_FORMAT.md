# TIP格式训练流程

## 概述

本目录包含完全遵循TIP原始实现的训练流程，将OMOMO数据适配到TIP的数据格式和训练方式。

## 主要修改

### 1. **数据格式对齐** (`dataset_omomo_tip_v2.py`)

#### TIP原始数据格式：
- **IMU数据**: `[T, 72]` = 6个IMU × (9维旋转 + 3维加速度)
- **状态数据**: `[T, 131]` = 18×6(关节旋转2axis表示) + 3(根速度) + 20(5个SBP约束×4)
- **返回格式**: 元组 `(x_imu, x_s, y_s_n)`

#### 当前OMOMO适配格式：
- **IMU数据**: `[T, 63]` = 6个人体IMU + 1个物体IMU (使用`--use_object_imu`时) × 9
- **状态数据**: `[T, 129]` = 18×6(关节旋转2axis表示) + 3(根速度) + 3(物体速度)
- **返回格式**: 元组 `(x_imu, x_s, y_s_n)` - **完全匹配TIP**

#### 关键特性：
- ✅ 使用2-axis旋转表示（而非6D rot6d）
- ✅ 返回元组而非字典
- ✅ 支持累积加速度特征（`--with_acc_sum`）
- ✅ 预加载所有序列到内存（匹配TIP的预处理流程）
- ✅ 随机采样或穷举采样模式

### 2. **训练流程对齐** (`train_tip_format.py`)

#### 完全遵循 `train_model.py` 的结构：
- ✅ 使用原始的 `TF_RNN_Past_State` 模型
- ✅ 使用原始的 `loss_q_only_2axis` 损失函数
- ✅ 添加 `loss_jerk` 平滑损失
- ✅ 历史状态噪声增强（`--noise_input_hist`）
- ✅ 梯度裁剪（`--clip`）
- ✅ Cosine学习率调度（`--cosine_lr`）
- ✅ 早停机制（`--patience`）

#### 扩展：
- ➕ 物体速度损失项（可调权重 `--lambda_obj`）
- ➕ 验证集评估
- ➕ 最佳模型保存

### 3. **与原始TIP的对比**

| 特性 | TIP原始 | 当前实现 | 说明 |
|------|---------|----------|------|
| 数据格式 | .pkl → .npy合并 | .pt文件预加载 | 功能等价 |
| 返回格式 | `(x_imu, x_s, y_s_n)` | `(x_imu, x_s, y_s_n)` | ✅ 完全匹配 |
| 旋转表示 | 2-axis | 2-axis | ✅ 完全匹配 |
| 模型架构 | `TF_RNN_Past_State` | `TF_RNN_Past_State` | ✅ 完全匹配 |
| 损失函数 | `loss_q_only_2axis` | `loss_q_only_2axis` + 物体损失 | ✅ 兼容 |
| SBP约束 | 5×4=20维 | 物体速度3维 | ⚠️ 不同（无物理约束） |
| IMU数量 | 6个（人体） | 7个（6人体+1物体） | ➕ 扩展 |

## 使用方法

### 1. 基础训练

```bash
cd /disk2/mmzhou/IMUHOI_1020/RefCodes/transformer-inertial-poser/my
bash train_omomo_tip_format.sh
```

### 2. 自定义参数

```bash
python train_tip_format.py \
    --train_dirs ../../process/processed_data_OMOMO/train \
    --val_dirs ../../process/processed_data_OMOMO/test \
    --save_path output/my_experiment \
    --batch_size 128 \
    --epochs 200 \
    --seq_len 60 \
    --lr 2e-4 \
    --cuda \
    --use_object_imu \
    --cosine_lr
```

### 3. 多数据集训练

```bash
python train_tip_format.py \
    --train_dirs \
        ../../process/processed_data_IMHD_split/train \
        ../../process/processed_data_BEHAVE_split/train \
        ../../process/processed_data_OMOMO/train \
    --val_dirs \
        ../../process/processed_data_IMHD_split/test \
        ../../process/processed_data_BEHAVE_split/test \
        ../../process/processed_data_OMOMO/test \
    --save_path output/multi_dataset \
    --cuda
```

### 4. 热启动（继续训练）

```bash
python train_tip_format.py \
    --warm_start output/tip_omomo_format/best.pt \
    --save_path output/tip_omomo_format_continue \
    --cuda
```

## 参数说明

### 数据参数
- `--train_dirs`: 训练数据目录（可多个）
- `--val_dirs`: 验证数据目录（可多个）
- `--seq_len`: 序列窗口长度（默认60帧）
- `--fps`: 帧率（默认30.0）
- `--use_object_imu`: 使用物体IMU作为额外传感器
- `--with_acc_sum`: 使用累积加速度特征

### 模型参数（匹配TIP默认值）
- `--rnn_nhid`: RNN隐藏层大小（默认512）
- `--tf_nhid`: Transformer FFN大小（默认1024）
- `--tf_in_dim`: Transformer输入维度（默认256）
- `--n_heads`: 注意力头数（默认16）
- `--tf_layers`: Transformer层数（默认4）
- `--past_dropout`: 历史状态dropout率（默认0.8）

### 训练参数
- `--batch_size`: 批次大小（默认128）
- `--epochs`: 训练轮数（默认200）
- `--lr`: 学习率（默认2e-4）
- `--weight_decay`: 权重衰减（默认1e-5）
- `--clip`: 梯度裁剪阈值（默认5.0）
- `--noise_input_hist`: 历史状态噪声强度（默认0.1）
- `--lambda_obj`: 物体损失权重（默认1.0）
- `--cosine_lr`: 使用Cosine学习率调度
- `--patience`: 早停耐心值（默认20）

## 输出文件

训练过程会在`--save_path`目录下生成以下文件：
- `latest.pt`: 最新模型
- `best.pt`: 验证集上最佳模型
- `it{epoch}.pt`: 每10轮的检查点

## 模型评估

### 1. 快速评估

```bash
cd /disk2/mmzhou/IMUHOI_1020/RefCodes/transformer-inertial-poser/my
bash eval_tip_format.sh
```

### 2. 自定义评估

```bash
python eval_tip_format.py \
    --data_dirs ../../process/processed_data_OMOMO/test \
    --weights checkpoints/tip_omomo_format/best.pt \
    --smplh_path ../../smpl_models/smplh/male/model.npz \
    --use_object_imu \
    --eval_contacts
```

### 3. 评估指标说明

脚本会计算以下指标：

| 指标 | 说明 | 单位 |
|------|------|------|
| **MPJPE** | Mean Per Joint Position Error<br>所有关节位置的平均误差 | cm |
| **MPJRE** | Mean Per Joint Rotation Error<br>所有关节旋转的平均误差 | deg |
| **Jitter** | 预测运动的加速度（平滑度）<br>越小越平滑 | mm/frame² |
| **Obj Trans Error** | 物体位置预测误差 | cm |
| **HOI Error** | Hand-Object Interaction Error<br>手部与物体相对位置误差（仅在接触时） | cm |

### 4. 评估输出示例

```
================================================================================
EVALUATION RESULTS
================================================================================
MPJPE (cm):               5.2345 ± 1.2345
MPJRE (deg):              8.7654 ± 2.3456
Jitter (mm/frame²):       12.3456 ± 3.4567
Obj Trans Error (cm):     3.4567 ± 0.8901
HOI Error (cm):           2.1234 ± 0.5678
================================================================================
```

### 5. 多数据集评估

```bash
python eval_tip_format.py \
    --data_dirs \
        ../../process/processed_data_IMHD_split/test \
        ../../process/processed_data_BEHAVE_split/test \
        ../../process/processed_data_OMOMO/test \
    --weights checkpoints/multi_dataset/best.pt \
    --use_object_imu
```

### 6. 鲁棒性测试（加噪声）

```bash
python eval_tip_format.py \
    --weights checkpoints/tip_omomo_format/best.pt \
    --imu_noise_std 0.1 \
    --use_object_imu
```

### 评估注意事项

1. **模型参数必须匹配训练时的配置**，特别是：
   - `--use_object_imu`: 是否使用物体IMU
   - `--with_acc_sum`: 是否使用累积加速度
   - `--rnn_nhid`, `--tf_nhid` 等架构参数

2. **SMPLH模型路径**必须正确，否则无法计算MPJPE等位置相关指标

3. **评估使用完整序列**（`use_full_sequence=True`），不进行随机采样

## 与之前实现的对比

### `dataset_omomo_tip.py` vs `dataset_omomo_tip_v2.py`

| 特性 | 旧版本 (v1) | 新版本 (v2) |
|------|-------------|-------------|
| 返回格式 | 字典 | 元组 `(x_imu, x_s, y_s_n)` |
| 旋转表示 | 6D (rot6d) | 2-axis |
| 数据加载 | 实时加载 | 预加载到内存 |
| 采样方式 | 滑动窗口/随机/完整序列 | 随机采样/穷举采样 |
| 兼容性 | 自定义训练循环 | TIP原始训练代码 |

### `train_tip_omomo.py` vs `train_tip_format.py`

| 特性 | 旧版本 | 新版本 |
|------|--------|--------|
| 模型 | `TIPWithObject` (自定义) | `TF_RNN_Past_State` (TIP原始) |
| 损失函数 | 自定义 | `loss_q_only_2axis` (TIP原始) |
| 数据输入 | 字典 | 元组 |
| 训练循环 | 自定义 | 遵循TIP原始结构 |

## 检查对齐情况

运行以下代码检查数据格式是否正确：

```python
from my.dataset_omomo_tip_v2 import OMOMODatasetTIPFormat

dataset = OMOMODatasetTIPFormat(
    data_dirs=['../../process/processed_data_OMOMO/train'],
    seq_len=60,
    use_object_imu=True,
)

x_imu, x_s, y_s_n = dataset[0]

print(f"x_imu shape: {x_imu.shape}")     # 应该是 [60, 63] (7 IMUs × 9)
print(f"x_s shape: {x_s.shape}")         # 应该是 [60, 129] (18×6 + 3 + 3)
print(f"y_s_n shape: {y_s_n.shape}")     # 应该是 [60, 129]
```

## 已知限制

1. **无SBP约束**: OMOMO数据没有TIP的Sample-based Physics (SBP)约束信息，因此状态维度从131降为129
2. **物体IMU**: 这是对TIP的扩展，原始TIP不包含物体信息
3. **内存占用**: 预加载所有序列会占用较多内存，但提高了训练速度

## 后续改进方向

1. 如果需要完全复现TIP，可以考虑：
   - 添加虚拟SBP约束（全零或基于启发式规则生成）
   - 去除物体IMU，仅使用6个人体IMU
   
2. 如果要充分利用OMOMO数据的特性：
   - 保持当前的物体IMU和物体速度
   - 可以考虑添加接触约束（基于`lhand_contact`, `rhand_contact`）
   
3. 性能优化：
   - 实现真正的数据合并（.pt → .npy）以节省内存
   - 添加数据增强策略

## 总结

本实现在保持TIP核心架构和训练流程的基础上，成功将OMOMO数据适配到TIP格式。主要对齐点：
- ✅ **数据格式**: 元组返回，2-axis旋转表示
- ✅ **模型架构**: 使用原始`TF_RNN_Past_State`
- ✅ **损失函数**: 使用原始`loss_q_only_2axis`
- ✅ **训练流程**: 遵循`train_model.py`的结构
- ➕ **扩展**: 支持物体IMU和物体速度预测

这样既保证了与TIP实现的一致性，又能充分利用OMOMO数据中的物体信息。

