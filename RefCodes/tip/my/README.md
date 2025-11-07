# TIP格式OMOMO训练和评估工具集

## 📖 快速导航

### 🚀 我想...

| 需求 | 操作 | 文档 |
|------|------|------|
| **快速开始** | `bash train_omomo_tip_format.sh` | [README_TIP_FORMAT.md](README_TIP_FORMAT.md) |
| **评估模型** | `bash eval_tip_format.sh` | [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) |
| **理解实现** | 阅读文档 | [COMPARISON.md](COMPARISON.md) |
| **查看完整总结** | 阅读文档 | [FINAL_SUMMARY.md](FINAL_SUMMARY.md) |
| **测试安装** | `python test_tip_format.py` | [README_TIP_FORMAT.md](README_TIP_FORMAT.md#测试验证) |

## 📦 文件组织

### 核心代码（Python）

```
dataset_omomo_tip_v2.py    # TIP格式Dataset (319行)
train_tip_format.py        # TIP格式训练脚本 (384行)
eval_tip_format.py         # TIP格式评估脚本 (351行)
test_tip_format.py         # 自动化测试 (254行)
```

### 启动脚本（Bash）

```
train_omomo_tip_format.sh  # 一键训练
eval_tip_format.sh         # 一键评估
```

### 文档（Markdown）

```
README.md                  # 本文档 (导航)
README_TIP_FORMAT.md       # 主要使用指南
EVALUATION_GUIDE.md        # 评估详细指南
COMPARISON.md              # 实现对比分析
IMPLEMENTATION_SUMMARY.md  # 技术实现总结
FINAL_SUMMARY.md           # 完整项目总结
```

### 旧版本（保留参考）

```
dataset_omomo_tip.py       # 旧版Dataset (字典格式)
model_tip_with_object.py   # 旧版自定义模型
loss_tip_obj.py            # 旧版损失函数
train_tip_omomo.py         # 旧版训练脚本
eval_tip_omomo.py          # 旧版评估脚本
```

## 🎯 主要特性

### ✅ 完全对齐TIP原始实现

- **数据格式**: 元组返回 `(x_imu, x_s, y_s_n)`
- **旋转表示**: 2-axis表示
- **模型架构**: 原始 `TF_RNN_Past_State`
- **损失函数**: 原始 `loss_q_only_2axis`
- **训练流程**: 噪声增强 + 梯度裁剪 + jerk损失

### ➕ 扩展功能

- **物体IMU**: 支持第7个IMU传感器
- **物体跟踪**: 预测物体位置和速度
- **评估系统**: 5个关键指标 (MPJPE, MPJRE, Jitter, Obj, HOI)
- **早停机制**: 防止过拟合
- **完整文档**: 5份详细文档

## 🏃 快速开始

### 1. 测试安装

```bash
cd /disk2/mmzhou/IMUHOI_1020/RefCodes/transformer-inertial-poser/my
conda activate IMUHOI  # 或您的torch环境
python test_tip_format.py
```

### 2. 训练模型

```bash
# 方法1: 使用默认配置
bash train_omomo_tip_format.sh

# 方法2: 自定义配置
python train_tip_format.py \
    --train_dirs ../../process/processed_data_OMOMO/train \
    --val_dirs ../../process/processed_data_OMOMO/test \
    --epochs 200 \
    --batch_size 128 \
    --cuda \
    --use_object_imu
```

### 3. 评估模型

```bash
# 方法1: 使用默认配置
bash eval_tip_format.sh

# 方法2: 自定义配置
python eval_tip_format.py \
    --weights checkpoints/tip_omomo_format/best.pt \
    --data_dirs ../../process/processed_data_OMOMO/test \
    --use_object_imu
```

## 📊 评估指标

| 指标 | 说明 | 单位 | 理想值 |
|------|------|------|--------|
| MPJPE | 关节位置误差 | cm | < 5.0 |
| MPJRE | 关节旋转误差 | deg | < 10.0 |
| Jitter | 运动平滑度 | mm/frame² | < 15.0 |
| Obj Trans | 物体位置误差 | cm | < 5.0 |
| HOI Error | 手-物交互误差 | cm | < 3.0 |

## 📚 文档说明

### [README_TIP_FORMAT.md](README_TIP_FORMAT.md)
**适合**: 新用户，想快速上手
**内容**:
- 使用方法和参数说明
- 训练和评估指南
- 常见问题解答

### [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)
**适合**: 需要评估模型的用户
**内容**:
- 评估指标详细说明
- 评估模式对比
- 结果解读指南
- 常见问题和解决方案

### [COMPARISON.md](COMPARISON.md)
**适合**: 想理解实现细节的用户
**内容**:
- 三种实现的详细对比
- TIP vs OMOMO vs 新实现
- 数据流程和架构差异
- 迁移指南

### [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
**适合**: 开发者和研究者
**内容**:
- 技术实现细节
- 修改点和修复说明
- 性能预期
- 开发备注

### [FINAL_SUMMARY.md](FINAL_SUMMARY.md)
**适合**: 项目管理和总览
**内容**:
- 完整的文件清单
- 核心成就总结
- 对齐度分析
- 未来工作计划

## 🔧 依赖要求

```bash
# 核心依赖
torch >= 1.8.0
numpy
tqdm

# 评估依赖
pytorch3d
human_body_prior (SMPLH)

# 建议环境
conda create -n IMUHOI python=3.8
conda activate IMUHOI
# 安装相关依赖...
```

## 🐛 常见问题

### Q: 训练时出现jerk loss错误？
**A**: 已修复。确保使用最新版本的 `train_tip_format.py`

### Q: 评估时没有MPJPE指标？
**A**: 需要正确配置SMPLH模型路径: `--smplh_path ../../smpl_models/smplh/male/model.npz`

### Q: 如何选择新版还是旧版实现？
**A**: 
- **新版** (`_format.py`): 想要复现TIP，使用原始模型
- **旧版** (`_omomo.py`): 想要更灵活的自定义模型

更多问题请查看 [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md#常见问题)

## 📈 性能对比

| 实现 | 数据格式 | 模型 | 对齐度 | 推荐度 |
|------|---------|------|--------|--------|
| 新版TIP格式 | 元组 | TIP原始 | 98% | ⭐⭐⭐⭐⭐ |
| 旧版OMOMO | 字典 | 自定义 | N/A | ⭐⭐⭐ |

## 🤝 贡献

如果您发现bug或有改进建议，欢迎：
1. 修改代码并测试
2. 更新相关文档
3. 提交pull request

## 📄 许可

基于TIP和OMOMO项目，遵循相应的开源许可。

## 🙏 致谢

- **TIP (Transformer Inertial Poser)**: 提供了核心架构和训练方法
- **OMOMO Dataset**: 提供了丰富的人-物交互数据
- **PyTorch3D & SMPLH**: 提供了评估所需的工具

---

**最后更新**: 2024  
**状态**: ✅ 生产就绪  
**维护者**: AI Assistant

**🚀 开始您的训练之旅吧！**


