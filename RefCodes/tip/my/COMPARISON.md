# TIP格式实现对比文档

本文档详细对比了原始TIP实现、当前OMOMO实现和新的TIP格式实现。

## 一、数据流程对比

### 1.1 原始TIP数据流程

```
AMASS数据 (*.npz)
    ↓
data-gen-and-viz-bullet-new.py (生成IMU和物理数据)
    ↓
.pkl文件 {imu, nimble_qdq, constrs}
    ↓
preprocess_and_combine_syn_amass.py (合并并预处理)
    ↓
.npy文件 {imu_train, s_train, info_train, sum_imu_train}
    ↓
training_data_loader.py (TrainSubDataset)
    ↓
返回: (x_imu, x_s, y_s_n)
    ↓
train_model.py (训练)
```

**关键数据维度:**
- IMU: `[T, 72]` = 6 IMUs × (9 rot + 3 acc)
- State: `[T, 131]` = 18×6 (rot 2axis) + 3 (root_vel) + 20 (5 SBPs × 4)
- 返回格式: 元组 `(x_imu, x_s, y_s_n)`

### 1.2 原OMOMO实现 (dataset_omomo_tip.py + train_tip_omomo.py)

```
OMOMO数据 (.pt文件)
    ↓
dataset_omomo_tip.py (OMOMODatasetWithObject)
    ↓
返回: 字典 {imu, state_hist, state_target, obj_pos_gt, ...}
    ↓
train_tip_omomo.py (自定义训练循环)
    ↓
model_tip_with_object.py (TIPWithObject自定义模型)
```

**关键数据维度:**
- IMU: `[T, 63]` = 7 IMUs × 9 (6人体 + 1物体)
- State: `[T, 129]` = 18×6 (rot6d) + 3 (root_pos/vel) + 3 (obj_vel)
- 返回格式: 字典

### 1.3 新TIP格式实现 (dataset_omomo_tip_v2.py + train_tip_format.py)

```
OMOMO数据 (.pt文件)
    ↓
dataset_omomo_tip_v2.py (OMOMODatasetTIPFormat)
    ↓ (预加载+转换为2-axis)
内存中的数据 (IMU list, S list)
    ↓
返回: 元组 (x_imu, x_s, y_s_n)
    ↓
train_tip_format.py (遵循TIP训练流程)
    ↓
simple_transformer_with_state.py (TIP原始模型)
```

**关键数据维度:**
- IMU: `[T, 63]` = 7 IMUs × 9 (6人体 + 1物体)
- State: `[T, 129]` = 18×6 (rot 2axis) + 3 (root_vel) + 3 (obj_vel)
- 返回格式: 元组 `(x_imu, x_s, y_s_n)` ✅

## 二、关键差异详解

### 2.1 旋转表示

| 实现 | 表示方式 | 维度 | 说明 |
|------|---------|------|------|
| 原始TIP | 2-axis | 6 per joint | 旋转矩阵的前两列 |
| 原OMOMO实现 | 6D (rot6d) | 6 per joint | 旋转矩阵的前两列（但标记为rot6d） |
| **新TIP格式** | 2-axis | 6 per joint | ✅ 完全匹配TIP |

**技术细节:**
- TIP使用 `batch_to_rot_mat_2axis()` 将轴角转换为2-axis表示
- OMOMO数据已经是旋转矩阵形式，直接取前两列即可
- 两者在数学上等价，但命名和处理流程略有不同

### 2.2 Dataset返回格式

#### 原始TIP (`training_data_loader.py`):
```python
def __getitem__(self, index):
    x_imu = self.IMU[index]              # [T, 72]
    x_s = self.S[index, :-1, :]          # [T, 131]
    y_s_n = self.S[index, 1:, :]         # [T, 131]
    return x_imu, x_s, y_s_n
```

#### 原OMOMO实现 (`dataset_omomo_tip.py`):
```python
def __getitem__(self, index):
    return {
        "imu": ...,                      # [T, 63]
        "state_hist": ...,               # [T, 129]
        "state_target": ...,             # [T, 129]
        "obj_pos_gt": ...,
        # ... 更多字段
    }
```

#### 新TIP格式 (`dataset_omomo_tip_v2.py`):
```python
def __getitem__(self, index):
    x_imu = ...                          # [T, 63]
    x_s = state[:-1]                     # [T, 129]
    y_s_n = state[1:]                    # [T, 129]
    return x_imu, x_s, y_s_n             # ✅ 匹配TIP
```

### 2.3 模型架构

| 实现 | 模型 | 输入 | 输出 |
|------|------|------|------|
| 原始TIP | `TF_RNN_Past_State` | `(x_imu, x_s)` | `y_pred` |
| 原OMOMO实现 | `TIPWithObject` | `(x_imu, x_s)` | `y_pred` |
| **新TIP格式** | `TF_RNN_Past_State` | `(x_imu, x_s)` | `y_pred` ✅ |

**`TIPWithObject` vs `TF_RNN_Past_State`:**
- `TIPWithObject` 包装了 `TF_RNN_Past_State` + 额外的物体refine层
- 新实现直接使用原始 `TF_RNN_Past_State`，通过损失函数处理物体部分

### 2.4 损失函数

#### 原始TIP (`train_model.py`):
```python
loss_q = loss_q_only_2axis(y[:, :-(n_sbps * 4)], y_pred[:, :-(n_sbps * 4)])
loss_c = loss_constr_multi(y[:, -(n_sbps * 4):], y_pred[:, -(n_sbps * 4):])
loss_j = loss_jerk(y_pred[:, :, :-3-(n_sbps * 4)])
loss = loss_c + loss_q + loss_j
```

#### 原OMOMO实现 (`loss_tip_obj.py`):
```python
loss_human = loss_q_only_2axis(human_tgt, human_pred)
loss_obj = torch.mean((obj_pred - obj_tgt) ** 2)
total = loss_human + lambda_obj * loss_obj
```

#### 新TIP格式 (`train_tip_format.py`):
```python
# Human part (rot + root_vel)
loss_q = loss_q_only_2axis(y[:, :human_dim], y_pred[:, :human_dim])

# Object part (velocity)
loss_obj = ((obj_vel_pred - obj_vel_gt) ** 2).mean() * lambda_obj * 100.0

# Jerk regularization
loss_j = loss_jerk(y_pred[:, :, :-3])

loss = loss_q + loss_obj + loss_j  # ✅ 结合TIP和OMOMO
```

**对比:**
- ✅ 使用TIP的 `loss_q_only_2axis`
- ✅ 保留物体损失（OMOMO特有）
- ✅ 添加jerk平滑损失（TIP特有）
- ⚠️ 无SBP约束损失（OMOMO数据不包含）

### 2.5 训练流程

#### 原始TIP特点:
- ✅ 每个epoch重新采样dataset (在新实现中用random_sample=True实现)
- ✅ 历史状态添加噪声增强
- ✅ Cosine学习率调度
- ✅ 梯度裁剪
- ✅ 日志间隔报告

#### 原OMOMO实现特点:
- ✅ 验证集评估
- ✅ 早停机制
- ✅ 最佳模型保存
- ❌ 没有重新采样
- ❌ 没有jerk损失

#### 新TIP格式实现:
- ✅ **融合了两者的优点**
- ✅ 遵循TIP的训练循环结构
- ✅ 保留OMOMO的验证和早停
- ✅ 支持TIP的所有特性

## 三、状态维度详解

### 3.1 原始TIP状态 (131维)

```
[18*6] 关节旋转(2-axis表示)
  + [3] 根速度 (x, y, z)
  + [20] 5个SBP约束 (左脚、右脚、左手、右手、骨盆，各4维)
-------
= 131维
```

### 3.2 OMOMO状态 (129维)

```
[18*6] 关节旋转(2-axis表示)
  + [3] 根速度 (x, y, z)
  + [3] 物体速度 (x, y, z)
-------
= 129维
```

**差异说明:**
- OMOMO没有物理约束信息 → 无SBP维度
- OMOMO有物体交互 → 添加物体速度

### 3.3 根位置 vs 根速度

| 实现 | 根表示 | 说明 |
|------|--------|------|
| 原始TIP | 根速度 | 直接预测速度，积分得位置 |
| 原OMOMO (vel模式) | 根速度 | 同TIP |
| 原OMOMO (pos模式) | 根位置 | 直接预测位置 |
| **新TIP格式** | 根速度 | ✅ 匹配TIP |

## 四、IMU传感器配置

### 4.1 传感器位置

#### 原始TIP (6个IMU):
```python
imu_joints = [
    root,        # 骨盆
    lwrist,      # 左手腕
    rwrist,      # 右手腕
    lknee,       # 左膝 (或lankle左脚踝)
    rknee,       # 右膝 (或rankle右脚踝)
    upperneck    # 颈部
]
```

#### 新实现 (7个IMU):
```python
imu_joints = [
    root,        # 骨盆
    lwrist,      # 左手腕
    rwrist,      # 右手腕
    lknee,       # 左膝
    rknee,       # 右膝
    upperneck,   # 颈部
    # ---- 扩展 ----
    object       # 物体中心 (可选)
]
```

### 4.2 IMU特征

每个IMU传感器输出9维特征:
- **加速度**: 3维 (x, y, z)
- **旋转**: 6维 (2-axis表示 = 旋转矩阵前两列)

总维度:
- TIP: 6 × 9 = 54维 (+ 18维acc_sum可选)
- 新实现: 7 × 9 = 63维 (+ 21维acc_sum可选)

### 4.3 坐标系归一化

两个实现都采用相同的归一化策略:
1. 提取第一帧根关节的位置 `root_pos0` 和旋转 `root_rot0`
2. 所有位置减去 `root_pos0` 并通过 `root_rot0^-1` 变换
3. 所有旋转左乘 `root_rot0^-1`

**目的:** 消除全局位置和朝向的影响，使模型专注于相对运动

## 五、使用场景建议

### 5.1 选择原OMOMO实现 (`dataset_omomo_tip.py` + `train_tip_omomo.py`)

**适用场景:**
- 需要更灵活的数据结构（字典返回）
- 需要额外的GT信息（位置、运动、接触）
- 需要完整序列或特定的采样策略
- 想要使用自定义的模型和损失函数

### 5.2 选择新TIP格式实现 (`dataset_omomo_tip_v2.py` + `train_tip_format.py`)

**适用场景:**
- 想要复现TIP的训练结果
- 想要使用TIP的预训练模型（迁移学习）
- 需要与TIP代码库保持兼容
- 想要使用TIP经过验证的训练策略

### 5.3 混合使用

也可以混合使用两种实现的优点:
- 使用 `dataset_omomo_tip_v2` 保证数据格式一致
- 创建自定义训练脚本结合两者的特性
- 使用TIP的模型但添加自定义的head

## 六、性能对比

### 6.1 内存占用

| 实现 | 数据存储 | 内存占用 |
|------|---------|---------|
| 原始TIP | 预处理为.npy | 高（但I/O快） |
| 原OMOMO | 实时加载.pt | 低（但I/O慢） |
| **新TIP格式** | 预加载到内存 | **最高**（但训练最快） |

### 6.2 训练速度

理论训练速度排序（从快到慢）:
1. **新TIP格式** - 预加载 + 无字典开销
2. 原始TIP - 预处理.npy
3. 原OMOMO - 实时加载 + 字典处理

### 6.3 灵活性

灵活性排序（从高到低）:
1. **原OMOMO** - 字典格式，易扩展
2. 新TIP格式 - 可配置参数多
3. 原始TIP - 固定流程

## 七、迁移指南

### 7.1 从原OMOMO迁移到新TIP格式

**需要修改的代码:**
```python
# 旧代码
from my.dataset_omomo_tip import OMOMODatasetWithObject
from my.model_tip_with_object import TIPWithObject

dataset = OMOMODatasetWithObject(...)
model = TIPWithObject(...)

for batch in loader:
    imu = batch["imu"]
    state = batch["state_hist"]
    target = batch["state_target"]
    pred = model(imu, state)

# 新代码
from my.dataset_omomo_tip_v2 import OMOMODatasetTIPFormat
from simple_transformer_with_state import TF_RNN_Past_State

dataset = OMOMODatasetTIPFormat(...)
model = TF_RNN_Past_State(...)

for x_imu, x_s, y_s_n in loader:
    pred = model(x_imu, x_s)
```

### 7.2 从原始TIP迁移到新TIP格式

**无需修改!** 只需替换数据加载部分:

```python
# 旧TIP代码
from training_data_loader import TrainSubDataset
data = TrainSubDataset(seq_length, info_path, imu_path, s_path)

# 新TIP格式代码
from my.dataset_omomo_tip_v2 import OMOMODatasetTIPFormat
data = OMOMODatasetTIPFormat(data_dirs=[...], seq_len=seq_length)

# 其余代码完全相同！
```

## 八、总结

### 新TIP格式实现的优势

✅ **完全兼容TIP原始代码**
- 数据格式一致（元组返回）
- 使用相同的模型和损失函数
- 遵循相同的训练流程

✅ **保留OMOMO数据的优势**
- 支持物体IMU
- 包含物体运动信息
- 可选的GT数据支持

✅ **融合两者的最佳实践**
- TIP的数据增强策略
- TIP的jerk平滑损失
- OMOMO的验证和早停机制

✅ **易于使用**
- 一键训练脚本
- 详细的参数说明
- 完整的测试代码

### 推荐使用新TIP格式的原因

1. **最接近TIP原始实现** - 便于复现和对比
2. **训练稳定性更好** - 使用TIP验证过的训练策略
3. **可扩展性强** - 易于添加新特性
4. **文档完善** - 有详细的说明和测试

如果你的目标是最大程度地复现TIP的训练结果并在OMOMO数据上训练，**强烈建议使用新TIP格式实现**。


