# TIP格式模型评估指南

## 📋 快速开始

### 最简单的方式

```bash
cd /disk2/mmzhou/IMUHOI_1020/RefCodes/transformer-inertial-poser/my

# 确保环境已激活
conda activate IMUHOI

# 运行评估
bash eval_tip_format.sh
```

## 🎯 评估指标说明

### 1. MPJPE (Mean Per Joint Position Error)
- **含义**: 所有关节的3D位置平均误差
- **单位**: 厘米 (cm)
- **越低越好**: 表示预测的身体姿态位置更准确
- **计算**: 通过SMPLH前向运动学计算3D关节位置，然后与GT比较

### 2. MPJRE (Mean Per Joint Rotation Error)
- **含义**: 所有关节的旋转平均误差
- **单位**: 度 (deg)
- **越低越好**: 表示预测的关节旋转更准确
- **计算**: 直接比较6D旋转表示的差异

### 3. Jitter (运动平滑度)
- **含义**: 预测运动的加速度（抖动程度）
- **单位**: 毫米/帧² (mm/frame²)
- **越低越好**: 表示预测的运动更平滑、更自然
- **计算**: 计算关节位置的二阶差分（加速度）

### 4. Object Translation Error
- **含义**: 物体位置预测误差
- **单位**: 厘米 (cm)
- **越低越好**: 表示物体位置跟踪更准确
- **计算**: 预测的物体轨迹与GT轨迹的L2距离

### 5. HOI Error (Hand-Object Interaction Error)
- **含义**: 手部与物体的相对位置误差
- **单位**: 厘米 (cm)
- **越低越好**: 表示手-物交互预测更准确
- **计算**: 仅在手部接触物体时计算相对位置差异
- **特殊性**: 这个指标专注于交互质量，而非绝对位置

## 🔧 评估参数详解

### 必需参数

```bash
--weights checkpoints/tip_omomo_format/best.pt  # 模型权重路径
--data_dirs ../../process/processed_data_OMOMO/test  # 测试数据路径
```

### 模型架构参数（必须与训练时一致）

```bash
--rnn_nhid 512          # RNN隐藏层大小
--tf_nhid 1024          # Transformer FFN大小
--tf_in_dim 256         # Transformer输入维度
--n_heads 16            # 注意力头数
--tf_layers 4           # Transformer层数
--past_dropout 0.8      # 历史状态dropout（评估时自动设为0）
```

### 数据配置参数（必须与训练时一致）

```bash
--use_object_imu        # 使用物体IMU（7个传感器）
--with_acc_sum          # 使用累积加速度特征
--seq_len 60            # 序列长度（评估时用完整序列）
--fps 30.0              # 帧率
--root_supervision vel  # 根监督类型：vel或pos
```

### 评估配置参数

```bash
--smplh_path ../../smpl_models/smplh/male/model.npz  # SMPLH模型路径
--imu_noise_std 0.0     # IMU噪声标准差（0.0=无噪声）
--eval_contacts         # 启用接触标签（用于HOI误差计算）
```

## 📊 评估模式对比

### 模式1: 清洁评估（默认）
```bash
python eval_tip_format.py \
    --weights checkpoints/tip_omomo_format/best.pt \
    --imu_noise_std 0.0 \
    --use_object_imu
```
- **用途**: 评估模型的最佳性能
- **特点**: 无噪声，完整序列

### 模式2: 鲁棒性测试
```bash
python eval_tip_format.py \
    --weights checkpoints/tip_omomo_format/best.pt \
    --imu_noise_std 0.1 \
    --use_object_imu
```
- **用途**: 测试模型对噪声的鲁棒性
- **特点**: 添加高斯噪声（std=0.1）

### 模式3: 多数据集评估
```bash
python eval_tip_format.py \
    --weights checkpoints/multi_dataset/best.pt \
    --data_dirs \
        ../../process/processed_data_IMHD_split/test \
        ../../process/processed_data_BEHAVE_split/test \
        ../../process/processed_data_OMOMO/test \
    --use_object_imu
```
- **用途**: 在多个数据集上评估泛化能力
- **特点**: 合并多个数据集的结果

## 🐛 常见问题

### Q1: 为什么MPJPE显示不出来？

**原因**: SMPLH模型路径不正确

**解决**:
```bash
# 检查路径是否存在
ls ../../smpl_models/smplh/male/model.npz

# 如果不存在，需要下载SMPLH模型
# 或修改 --smplh_path 指向正确路径
```

### Q2: 模型参数不匹配错误

**错误信息**: `RuntimeError: Error(s) in loading state_dict...`

**原因**: 评估时的模型架构与训练时不一致

**解决**:
```bash
# 确保以下参数与训练时完全一致：
--rnn_nhid 512
--tf_nhid 1024
--tf_in_dim 256
--n_heads 16
--tf_layers 4
--use_object_imu      # 如果训练时用了就加上
--with_acc_sum        # 如果训练时用了就加上
```

### Q3: HOI Error样本数少于其他指标

**原因**: HOI Error只在有接触标签且手部与物体接触时计算

**这是正常的**: 不是所有序列都有接触，不是所有帧都有手-物接触

### Q4: 评估很慢

**原因**: 
1. 使用CPU而非GPU
2. SMPLH前向运动学计算耗时

**解决**:
```bash
# 确保使用GPU
nvidia-smi  # 检查GPU可用性

# 如果只关心旋转误差，可以不加载SMPLH
# （修改代码跳过SMPLH加载）
```

### Q5: 不同模型如何对比？

**方法**: 分别评估并记录结果

```bash
# 评估模型A
python eval_tip_format.py \
    --weights checkpoints/model_A/best.pt \
    --use_object_imu > results_A.txt

# 评估模型B
python eval_tip_format.py \
    --weights checkpoints/model_B/best.pt \
    --use_object_imu > results_B.txt

# 对比结果
diff results_A.txt results_B.txt
```

## 📈 结果解读

### 优秀的结果示例

```
MPJPE (cm):               < 5.0    # 平均误差小于5cm，非常好
MPJRE (deg):              < 10.0   # 旋转误差小于10度，很好
Jitter (mm/frame²):       < 15.0   # 抖动很小，运动平滑
Obj Trans Error (cm):     < 5.0    # 物体跟踪准确
HOI Error (cm):           < 3.0    # 手-物交互准确
```

### 需要改进的结果示例

```
MPJPE (cm):               > 10.0   # 位置误差较大
MPJRE (deg):              > 20.0   # 旋转误差较大
Jitter (mm/frame²):       > 30.0   # 运动抖动明显
Obj Trans Error (cm):     > 10.0   # 物体跟踪不准
HOI Error (cm):           > 5.0    # 手-物交互误差大
```

### 改进建议

如果结果不理想，可以尝试：

1. **训练更久**: 增加epochs或检查是否early stopping太早
2. **调整学习率**: 尝试更小的学习率
3. **增加数据**: 使用更多训练数据
4. **调整损失权重**: 修改 `--lambda_obj` 权重
5. **数据增强**: 增加 `--noise_input_hist` 值
6. **模型容量**: 增加 `--rnn_nhid` 或 `--tf_layers`

## 🔗 相关文档

- **训练指南**: `README_TIP_FORMAT.md`
- **实现对比**: `COMPARISON.md`
- **完整总结**: `IMPLEMENTATION_SUMMARY.md`
- **旧版评估**: `eval_tip_omomo.py` (使用自定义模型)

## 📝 评估报告模板

评估完成后，建议记录以下信息：

```markdown
## 模型评估报告

### 基本信息
- 模型: checkpoints/tip_omomo_format/best.pt
- 数据集: OMOMO test set
- 评估时间: 2024-XX-XX
- 序列数: XX

### 评估结果
| 指标 | 值 | 标准差 |
|------|-----|--------|
| MPJPE (cm) | X.XX | ±X.XX |
| MPJRE (deg) | X.XX | ±X.XX |
| Jitter (mm/frame²) | X.XX | ±X.XX |
| Obj Trans Error (cm) | X.XX | ±X.XX |
| HOI Error (cm) | X.XX | ±X.XX |

### 训练配置
- Epochs: XXX
- Batch size: 128
- Learning rate: 2e-4
- 使用物体IMU: ✅/❌
- 累积加速度: ✅/❌

### 结论
[总结模型表现，与baseline对比，改进方向等]
```

---

**祝评估顺利！如有问题，请查看相关文档或联系开发者。**


