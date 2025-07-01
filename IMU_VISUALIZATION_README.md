# IMU数据可视化工具

这个工具可以从dataloader中提取IMU数据并进行三维可视化，支持人体和物体的IMU传感器数据的实时动画显示。

## 功能特性

### 🎯 主要功能
- **3D可视化**: 在三维空间中显示IMU传感器的位置和状态
- **加速度可视化**: 用箭头向量表示加速度的方向和大小
- **旋转可视化**: 用坐标轴表示每个IMU的方向信息
- **时间序列图**: 显示加速度随时间的变化
- **动画播放**: 支持动画播放和交互控制
- **数据导出**: 可以保存动画为GIF文件

### 📊 可视化内容
1. **人体IMU传感器** (6个):
   - 左手 (left_hand)
   - 右手 (right_hand) 
   - 左脚 (left_foot)
   - 右脚 (right_foot)
   - 头部 (head)
   - 髋部 (hip)

2. **物体IMU传感器** (1个):
   - 物体的加速度和方向信息

3. **数据表示**:
   - 每个IMU：9维数据 (3D加速度 + 6D旋转表示)
   - 加速度：用箭头表示方向和大小
   - 旋转：用RGB坐标轴表示 (红=X轴, 绿=Y轴, 蓝=Z轴)

## 安装要求

### Python包依赖
```bash
pip install torch matplotlib pytorch3d numpy
```

### 数据要求
- 确保 `processed_data_0506/test` 目录存在
- 目录中包含 `.pt` 格式的数据文件
- 数据文件应该包含IMU信息

## 使用方法

### 1. 快速演示
```bash
python demo_imu_vis.py
```

### 2. 命令行使用
```bash
# 基本动画播放
python visualize_imu.py

# 指定数据目录和序列
python visualize_imu.py --data_dir processed_data_0506/test --seq_idx 5

# 显示静态帧
python visualize_imu.py --mode static --frame 30

# 保存为GIF
python visualize_imu.py --mode save --output my_imu.gif --fps 15
```

### 3. 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_dir` | str | processed_data_0506/test | 数据目录路径 |
| `--seq_idx` | int | 0 | 要可视化的序列索引 |
| `--mode` | str | animate | 可视化模式 (animate/static/save) |
| `--frame` | int | 0 | 静态模式下显示的帧索引 |
| `--output` | str | imu_visualization.gif | 保存的文件名 |
| `--fps` | int | 10 | 动画帧率 |
| `--interval` | int | 100 | 动画更新间隔(毫秒) |

## 交互控制

### 动画模式下的键盘控制
- **空格键**: 暂停/继续播放
- **左箭头**: 后退一帧
- **右箭头**: 前进一帧  
- **ESC键**: 退出

## 可视化界面说明

### 主要区域
1. **3D可视化区域** (左上，大区域):
   - 显示所有IMU传感器的3D位置
   - 箭头表示加速度向量
   - 坐标轴表示旋转方向
   
2. **人体IMU时间序列** (右上):
   - 显示所有人体IMU的加速度幅值变化
   - 不同颜色代表不同的IMU传感器
   - 黑色虚线表示当前帧位置

3. **物体IMU时间序列** (右下):
   - 显示物体IMU的加速度幅值变化
   - 红色线条表示物体数据

### 颜色编码
- 每个人体IMU传感器有独特的颜色
- 物体IMU使用红色
- RGB坐标轴：红色=X轴，绿色=Y轴，蓝色=Z轴

## 数据格式要求

### 输入数据结构
```python
data = {
    "human_imu": torch.Tensor,  # [seq_len, 6, 9] - 人体IMU数据
    "obj_imu": torch.Tensor,    # [seq_len, 1, 9] - 物体IMU数据  
    "has_object": bool,         # 是否包含物体数据
    "obj_name": str,           # 物体名称
    # ... 其他数据
}
```

### IMU数据格式
- **前3维**: 3D加速度 (m/s²)
- **后6维**: 6D旋转表示 (会自动转换为旋转矩阵)

## 示例用法

### Python脚本中使用
```python
from visualize_imu import IMUVisualizer

# 创建可视化器
visualizer = IMUVisualizer(
    data_dir="processed_data_0506/test", 
    seq_idx=0
)

# 动画播放
visualizer.animate(interval=100)

# 显示特定帧
visualizer.show_static_frame(frame_idx=25)

# 保存动画
visualizer.save_animation("my_imu.gif", fps=15)
```

## 故障排除

### 常见问题

1. **"数据集为空"错误**
   - 检查数据路径是否正确
   - 确保数据文件格式正确
   - 检查数据文件是否包含必要的键

2. **"导入错误"**
   - 安装缺失的Python包
   - 检查pytorch3d是否正确安装

3. **可视化效果不理想**
   - 调整箭头缩放因子 (`acc_scale`)
   - 调整坐标轴长度
   - 修改颜色映射

### 性能优化
- 对于长序列，考虑只可视化部分帧
- 降低动画帧率可以减少计算负担
- 静态模式比动画模式更快

## 扩展功能

### 自定义IMU布局
可以修改 `_get_imu_layout_positions()` 函数来调整IMU在3D空间中的布局位置。

### 添加新的可视化元素
可以在 `update_frame()` 函数中添加新的可视化元素，比如轨迹线、热力图等。

### 数据过滤和预处理
可以在数据加载后添加滤波或平滑处理，以改善可视化效果。

## 技术细节

### 核心算法
- 使用pytorch3d进行6D旋转表示到旋转矩阵的转换
- matplotlib的3D plotting和动画功能
- 实时更新箭头和坐标轴的位置和方向

### 坐标系统
- 使用右手坐标系
- Z轴向上为正方向
- 旋转矩阵表示局部坐标系相对于全局坐标系的方向

---

如有问题或建议，请参考代码注释或联系开发者。 