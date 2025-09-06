#!/bin/bash

# vis2.py 模块化运行示例脚本
# 请根据你的实际路径修改以下变量

# 配置文件路径 - 必需，包含pretrained_modules设置
CONFIG_PATH="configs/TransPose_train.yaml"

# 可选：SimpleObjT的object_trans模块路径
# 如果不提供，SimpleObjT模型将只复用原模型的velocity_contact和human_pose模块
SIMPLE_OBJT_MODULE_PATH="outputs/TransPose/transpose_08031938/modules/object_trans_best.pt"  # 例如: "outputs/TransPose/simple_objt/modules/object_trans_best.pt"

# 数据路径
TEST_DATA_DIR="processed_hoi_data_0803/test"
SMPL_MODEL_PATH="body_models/smplh/neutral/model.npz"
OBJ_GEO_ROOT="./dataset/captured_objects"

# 运行参数
LIMIT_SEQUENCES=500  # 限制加载的序列数量，避免内存问题
NUM_WORKERS=12       # 数据加载器工作进程数

echo "=========================================="
echo "启动模块化三模型对比可视化工具"
echo "=========================================="
echo "配置文件: $CONFIG_PATH"
echo "SimpleObjT模块: ${SIMPLE_OBJT_MODULE_PATH:-'使用原模型共享模块'}"
echo "测试数据: $TEST_DATA_DIR"
echo "限制序列数: $LIMIT_SEQUENCES"
echo "=========================================="

# 检查配置文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误: 配置文件不存在: $CONFIG_PATH"
    echo "请修改脚本中的 CONFIG_PATH 变量"
    exit 1
fi

# 检查配置文件是否包含预训练模块设置
if ! grep -q "pretrained_modules:" "$CONFIG_PATH"; then
    echo "警告: 配置文件可能不包含 pretrained_modules 设置"
    echo "请确认配置文件包含 staged_training.modular_training.pretrained_modules 部分"
    echo "继续运行，但可能遇到模块加载问题..."
fi

if [ ! -d "$TEST_DATA_DIR" ]; then
    echo "错误: 测试数据目录不存在: $TEST_DATA_DIR"
    echo "请修改脚本中的 TEST_DATA_DIR 变量"
    exit 1
fi

# 检查可选的SimpleObjT模块文件
if [ -n "$SIMPLE_OBJT_MODULE_PATH" ] && [ ! -f "$SIMPLE_OBJT_MODULE_PATH" ]; then
    echo "警告: SimpleObjT模块文件不存在: $SIMPLE_OBJT_MODULE_PATH"
    echo "将使用原模型的共享模块"
    SIMPLE_OBJT_MODULE_PATH=""
fi

echo "所有必需文件检查通过，启动可视化工具..."

# 构建运行命令
CMD="python vis2.py \
    --config \"$CONFIG_PATH\" \
    --test_data_dir \"$TEST_DATA_DIR\" \
    --smpl_model_path \"$SMPL_MODEL_PATH\" \
    --obj_geo_root \"$OBJ_GEO_ROOT\" \
    --batch_size 1 \
    --num_workers $NUM_WORKERS \
    --limit_sequences $LIMIT_SEQUENCES"

# 如果提供了SimpleObjT模块路径，则添加到命令中
if [ -n "$SIMPLE_OBJT_MODULE_PATH" ]; then
    CMD="$CMD --simple_objt_module_path \"$SIMPLE_OBJT_MODULE_PATH\""
fi

echo "运行命令:"
echo "$CMD"
echo ""

# 运行可视化工具
eval $CMD

echo "可视化工具已退出" 