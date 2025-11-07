# 快速开始指南

## 5分钟上手DIP-Style训练

### 步骤1：测试数据加载

确保数据能够正确加载：

```bash
cd /disk2/mmzhou/IMUHOI_1020/RefCodes/dip18/my
python test_dip_style_dataset.py
```

如果看到 "✓ All tests passed!"，则数据加载正常。

### 步骤2：开始训练

#### 方式A：使用默认配置（推荐新手）

```bash
python train_dip_style.py
```

#### 方式B：使用配置文件（推荐）

```bash
python train_dip_style.py --config-file config_example.json
```

#### 方式C：使用启动脚本

```bash
bash run_training_dip_style.sh --experiment-name my_first_exp
```

#### 方式D：自定义参数

```bash
python train_dip_style.py \
    --datasets-train processed_data_BEHAVE_split \
    --data-root ../../process \
    --num-epochs 30 \
    --batch-size 128 \
    --experiment-name quick_test
```

### 步骤3：监控训练

在另一个终端启动TensorBoard：

```bash
tensorboard --logdir checkpoints/dip_obj_style/
```

然后在浏览器访问：http://localhost:6006

### 步骤4：查看结果

训练完成后，模型保存在：
```
checkpoints/dip_obj_style/run-TIMESTAMP-EXPERIMENT/
├── model_best.pt      # 最佳模型
├── model_final.pt     # 最终模型
└── config.json        # 完整配置
```

## 常用命令

### 修改学习率
```bash
python train_dip_style.py --learning-rate 1e-4
```

### 修改批大小
```bash
python train_dip_style.py --batch-size 128
```

### 指定GPU
```bash
python train_dip_style.py --device cuda:0
```

### 快速测试（小epoch数）
```bash
python train_dip_style.py --num-epochs 5 --experiment-name quick_test
```

## 常见问题

### Q: 数据路径错误怎么办？
A: 修改 `--data-root` 参数指向正确的数据目录。

### Q: 显存不足怎么办？
A: 减小 `--batch-size`，例如：`--batch-size 64`

### Q: 想要更快的训练？
A: 增大 `--batch-size` 和 `--num-workers`（如果有足够的资源）

### Q: 如何禁用归一化？
A: 修改配置文件，设置 `"normalize_data": false`

## 下一步

- 查看 [README_DIP_STYLE.md](README_DIP_STYLE.md) 了解详细配置
- 查看 [IMPLEMENTATION_COMPARISON.md](IMPLEMENTATION_COMPARISON.md) 了解技术细节
- 查看 [MODIFICATION_SUMMARY.md](MODIFICATION_SUMMARY.md) 了解完整修改内容

## 问题反馈

如遇到问题，请检查：
1. 数据路径是否正确
2. GPU是否可用（`nvidia-smi`）
3. 依赖是否安装完整（PyTorch, numpy, tqdm）
4. 运行测试脚本是否通过

