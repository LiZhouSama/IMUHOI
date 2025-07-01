#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU数据可视化工具
从dataloader中提取人体和物体的IMU数据，并进行三维可视化
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import pytorch3d.transforms as transforms
from dataloader.dataloader import IMUDataset
from configs.global_config import IMU_JOINTS, IMU_JOINT_NAMES, FRAME_RATE
import argparse

# 设置字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class IMUVisualizer:
    def __init__(self, data_dir="processed_data_0612/test", seq_idx=0):
        """
        初始化IMU可视化器
        
        Args:
            data_dir: 数据目录路径
            seq_idx: 要可视化的序列索引
        """
        self.data_dir = data_dir
        self.seq_idx = seq_idx
        
        # 初始化数据集
        print(f"正在加载数据集从 {data_dir}...")
        self.dataset = IMUDataset(
            data_dir=data_dir,
            window_size=120,
            normalize=True,
            debug=False
        )
        
        if len(self.dataset) == 0:
            raise ValueError("数据集为空，请检查数据路径")
        
        print(f"数据集加载完成，共有 {len(self.dataset)} 个序列")
        
        # 加载指定序列的数据
        if seq_idx >= len(self.dataset):
            seq_idx = 0
            print(f"序列索引超出范围，使用索引 {seq_idx}")
        
        self.data = self.dataset[seq_idx]
        self.current_frame = 0
        
        # 提取IMU数据
        self._extract_imu_data()
        
        # 设置可视化参数
        self.setup_visualization()
        
    def _extract_imu_data(self):
        """提取和处理IMU数据"""
        # 人体IMU数据 [seq_len, num_imus, 9]
        self.human_imu = self.data["human_imu"]  # [seq, 6, 9]
        self.seq_len, self.num_human_imus, _ = self.human_imu.shape
        
        # IMU关节全局位置和旋转数据
        self.imu_global_positions = self.data["imu_global_positions"]  # [seq, 6, 3]
        self.imu_global_rotations = self.data["imu_global_rotations"]  # [seq, 6, 3, 3]
        
        # 物体IMU数据 [seq_len, 1, 9] (如果存在)
        self.has_object = self.data["has_object"]
        if self.has_object:
            self.obj_imu = self.data["obj_imu"]  # [seq, 1, 9]
            self.obj_name = self.data.get("obj_name", "unknown_object")
        else:
            self.obj_imu = torch.zeros(self.seq_len, 1, 9)
            self.obj_name = "no_object"
            
        print(f"序列长度: {self.seq_len} 帧")
        print(f"人体IMU数量: {self.num_human_imus}")
        print(f"是否有物体: {self.has_object}")
        if self.has_object:
            print(f"物体名称: {self.obj_name}")
        print(f"IMU关节全局位置范围: [{self.imu_global_positions.min():.3f}, {self.imu_global_positions.max():.3f}]")
        
        # 分离加速度和旋转数据
        self._separate_acc_rot()
        
    def _separate_acc_rot(self):
        """分离加速度和旋转数据"""
        # 人体IMU：前3维是加速度，后6维是6D旋转表示
        self.human_acc = self.human_imu[:, :, :3]  # [seq, 6, 3]
        self.human_rot_6d = self.human_imu[:, :, 3:]  # [seq, 6, 6]
        
        # 将6D旋转表示转换为旋转矩阵
        self.human_rot_matrices = transforms.rotation_6d_to_matrix(
            self.human_rot_6d.reshape(-1, 6)
        ).reshape(self.seq_len, self.num_human_imus, 3, 3)
        
        # 物体IMU
        self.obj_acc = self.obj_imu[:, :, :3]  # [seq, 1, 3]
        self.obj_rot_6d = self.obj_imu[:, :, 3:]  # [seq, 1, 6]
        self.obj_rot_matrices = transforms.rotation_6d_to_matrix(
            self.obj_rot_6d.reshape(-1, 6)
        ).reshape(self.seq_len, 1, 3, 3)
        
        # 计算速度 (通过对加速度积分)
        self._calculate_velocities()
        
        print("IMU数据分离完成")
        print(f"人体加速度范围: [{self.human_acc.min():.3f}, {self.human_acc.max():.3f}]")
        print(f"物体加速度范围: [{self.obj_acc.min():.3f}, {self.obj_acc.max():.3f}]")
        print(f"人体速度范围: [{self.human_vel.min():.3f}, {self.human_vel.max():.3f}]")
        print(f"物体速度范围: [{self.obj_vel.min():.3f}, {self.obj_vel.max():.3f}]")
    
    def _calculate_velocities(self):
        """通过对加速度进行数值积分计算速度"""
        dt = 1.0 / FRAME_RATE  # 时间间隔 = 1/30 秒
        
        # 计算人体IMU速度
        # 使用累积梯形积分: v[t] = v[t-1] + (a[t-1] + a[t]) * dt / 2
        self.human_vel = torch.zeros_like(self.human_acc)
        for t in range(1, self.seq_len):
            # 梯形积分
            self.human_vel[t] = self.human_vel[t-1] + (self.human_acc[t-1] + self.human_acc[t]) * dt / 2
        
        # 计算物体IMU速度
        self.obj_vel = torch.zeros_like(self.obj_acc)
        for t in range(1, self.seq_len):
            # 梯形积分
            self.obj_vel[t] = self.obj_vel[t-1] + (self.obj_acc[t-1] + self.obj_acc[t]) * dt / 2
        
        print(f"速度计算完成，使用帧率: {FRAME_RATE} FPS, dt: {dt:.4f}s")
        
    def setup_visualization(self):
        """设置可视化参数"""
        # 创建图形和子图
        self.fig = plt.figure(figsize=(20, 12))
        
        # 使用 subplot2grid 创建布局
        # 主要的3D可视化 (占据左边大部分区域)
        self.ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=3, projection='3d')
        
        # 加速度时间序列图
        self.ax_acc = plt.subplot2grid((3, 4), (0, 2))
        self.ax_obj_acc = plt.subplot2grid((3, 4), (1, 2))
        
        # 速度时间序列图
        self.ax_vel = plt.subplot2grid((3, 4), (0, 3))
        self.ax_obj_vel = plt.subplot2grid((3, 4), (1, 3))
        
        # 关节位置时间序列图
        self.ax_pos = plt.subplot2grid((3, 4), (2, 2), colspan=2)
        
        # 计算关节位置的范围，用于设置3D坐标轴
        self.pos_min = self.imu_global_positions.min().item()
        self.pos_max = self.imu_global_positions.max().item()
        self.pos_center = (self.pos_min + self.pos_max) / 2
        self.pos_range = (self.pos_max - self.pos_min) / 2 * 1.2  # 增加20%的边距
        
        print(f"关节位置范围: [{self.pos_min:.3f}, {self.pos_max:.3f}], 中心: {self.pos_center:.3f}")
        
        # 颜色映射
        self.colors = plt.cm.tab10(np.linspace(0, 1, self.num_human_imus))
        self.obj_color = 'red'
        
        # 初始化绘图元素
        self._init_plot_elements()
        
    def _init_plot_elements(self):
        """初始化绘图元素"""
        # 3D主图设置
        self.ax_main.set_xlabel('X (m)')
        self.ax_main.set_ylabel('Y (m)')
        self.ax_main.set_zlabel('Z (m)')
        self.ax_main.set_title('IMU Global Positions and Orientations Visualization')
        
        # 设置轴的范围（基于实际关节位置）
        self.ax_main.set_xlim(self.pos_center - self.pos_range, self.pos_center + self.pos_range)
        self.ax_main.set_ylim(self.pos_center - self.pos_range, self.pos_center + self.pos_range)
        self.ax_main.set_zlim(self.pos_center - self.pos_range, self.pos_center + self.pos_range)
        
        # 初始化绘图对象
        self.joint_points = []       # 关节位置点
        self.joint_frames = []       # 关节坐标系
        self.joint_trajectories = [] # 关节轨迹线
        self.quivers_human_acc = []  # 人体加速度箭头
        self.quivers_human_vel = []  # 人体速度箭头
        
        # 创建人体IMU可视化元素
        for i in range(self.num_human_imus):
            pos = self.imu_global_positions[0, i].numpy()
            color = self.colors[i]
            
            # 关节位置点
            point = self.ax_main.scatter(
                pos[0], pos[1], pos[2],
                color=color, s=100, alpha=0.8,
                label=IMU_JOINT_NAMES[i]
            )
            self.joint_points.append(point)
            
            # 关节轨迹线（显示运动轨迹）
            trajectory_data = self.imu_global_positions[:, i, :].numpy()
            line, = self.ax_main.plot(
                trajectory_data[:, 0], 
                trajectory_data[:, 1], 
                trajectory_data[:, 2],
                color=color, alpha=0.3, linewidth=1
            )
            self.joint_trajectories.append(line)
            
            # 加速度箭头（在关节位置显示）
            quiver_acc = self.ax_main.quiver(
                pos[0], pos[1], pos[2],
                0, 0, 0,
                color=color, 
                arrow_length_ratio=0.1,
                alpha=0.8,
                linewidth=2,
                label=f'{IMU_JOINT_NAMES[i]}_acc'
            )
            self.quivers_human_acc.append(quiver_acc)
            
            # 速度箭头 (稍微偏移位置，避免重叠)
            quiver_vel = self.ax_main.quiver(
                pos[0] + 0.05, pos[1] + 0.05, pos[2] + 0.05,
                0, 0, 0,
                color=color, 
                arrow_length_ratio=0.1,
                alpha=0.6,
                linewidth=1,
                linestyle='--'
            )
            self.quivers_human_vel.append(quiver_vel)
            
            # 关节坐标系轴（表示旋转）
            frame_lines = []
            axis_colors = ['red', 'green', 'blue']
            for axis in range(3):
                line, = self.ax_main.plot(
                    [pos[0], pos[0]], 
                    [pos[1], pos[1]], 
                    [pos[2], pos[2]],
                    color=axis_colors[axis],
                    linewidth=2,
                    alpha=0.7
                )
                frame_lines.append(line)
            self.joint_frames.append(frame_lines)
        
        # 物体可视化（如果有）
        if self.has_object:
            # 物体使用不同的位置
            obj_trans_data = self.data.get("obj_trans", torch.zeros(self.seq_len, 3))
            obj_pos = obj_trans_data[0].numpy()
            
            # 物体位置点
            self.obj_point = self.ax_main.scatter(
                obj_pos[0], obj_pos[1], obj_pos[2],
                color=self.obj_color, s=200, alpha=0.8,
                marker='s', label=f'Object: {self.obj_name}'
            )
            
            # 物体轨迹
            obj_trajectory_data = obj_trans_data.numpy()
            self.obj_trajectory, = self.ax_main.plot(
                obj_trajectory_data[:, 0], 
                obj_trajectory_data[:, 1], 
                obj_trajectory_data[:, 2],
                color=self.obj_color, alpha=0.3, linewidth=2
            )
        
        # 添加图例
        self.ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 设置时间序列图
        self.setup_time_series_plots()
        
    def setup_time_series_plots(self):
        """设置时间序列图"""
        # 人体加速度时间序列
        self.ax_acc.set_title('Human IMU Acceleration')
        self.ax_acc.set_xlabel('Time Frame')
        self.ax_acc.set_ylabel('Acceleration (m/s²)')
        
        # 绘制所有IMU的加速度幅值
        time_steps = np.arange(self.seq_len)
        for i in range(self.num_human_imus):
            acc_magnitude = torch.norm(self.human_acc[:, i, :], dim=1).numpy()
            self.ax_acc.plot(time_steps, acc_magnitude, 
                           color=self.colors[i], 
                           label=IMU_JOINT_NAMES[i],
                           alpha=0.7)
        
        # 添加当前帧指示线
        self.frame_line_human = self.ax_acc.axvline(x=0, color='black', 
                                                   linestyle='--', linewidth=2)
        self.ax_acc.legend()
        self.ax_acc.grid(True, alpha=0.3)
        
        # 物体加速度时间序列
        if self.has_object:
            self.ax_obj_acc.set_title(f'Object IMU Acceleration ({self.obj_name})')
            self.ax_obj_acc.set_xlabel('Time Frame')
            self.ax_obj_acc.set_ylabel('Acceleration (m/s²)')
            
            obj_acc_magnitude = torch.norm(self.obj_acc[:, 0, :], dim=1).numpy()
            self.ax_obj_acc.plot(time_steps, obj_acc_magnitude, 
                               color=self.obj_color, linewidth=2)
            
            self.frame_line_obj = self.ax_obj_acc.axvline(x=0, color='black', 
                                                         linestyle='--', linewidth=2)
            self.ax_obj_acc.grid(True, alpha=0.3)
        else:
            self.ax_obj_acc.text(0.5, 0.5, 'No Object Data', 
                               transform=self.ax_obj_acc.transAxes,
                               ha='center', va='center', fontsize=14)
            self.ax_obj_acc.set_xticks([])
            self.ax_obj_acc.set_yticks([])
        
        # 速度时间序列图
        self.ax_vel.set_title('Human IMU Velocity')
        self.ax_vel.set_xlabel('Time Frame')
        self.ax_vel.set_ylabel('Velocity (m/s)')
        
        # 绘制所有IMU的速度幅值
        for i in range(self.num_human_imus):
            vel_magnitude = torch.norm(self.human_vel[:, i, :], dim=1).numpy()
            self.ax_vel.plot(time_steps, vel_magnitude, 
                           color=self.colors[i], 
                           label=IMU_JOINT_NAMES[i],
                           alpha=0.7)
        
        # 添加当前帧指示线
        self.frame_line_human_vel = self.ax_vel.axvline(x=0, color='black', 
                                                       linestyle='--', linewidth=2)
        self.ax_vel.legend()
        self.ax_vel.grid(True, alpha=0.3)
        
        # 物体速度时间序列
        if self.has_object:
            self.ax_obj_vel.set_title(f'Object IMU Velocity ({self.obj_name})')
            self.ax_obj_vel.set_xlabel('Time Frame')
            self.ax_obj_vel.set_ylabel('Velocity (m/s)')
            
            obj_vel_magnitude = torch.norm(self.obj_vel[:, 0, :], dim=1).numpy()
            self.ax_obj_vel.plot(time_steps, obj_vel_magnitude, 
                               color=self.obj_color, linewidth=2)
            
            self.frame_line_obj_vel = self.ax_obj_vel.axvline(x=0, color='black', 
                                                             linestyle='--', linewidth=2)
            self.ax_obj_vel.grid(True, alpha=0.3)
        else:
            self.ax_obj_vel.text(0.5, 0.5, 'No Object Data', 
                               transform=self.ax_obj_vel.transAxes,
                               ha='center', va='center', fontsize=14)
            self.ax_obj_vel.set_xticks([])
            self.ax_obj_vel.set_yticks([])
        
        # 关节位置时间序列图
        self.ax_pos.set_title('IMU Joint Global Positions')
        self.ax_pos.set_xlabel('Time Frame')
        self.ax_pos.set_ylabel('Position (m)')
        
        # 绘制所有IMU关节的位置（X, Y, Z分量）
        for i in range(self.num_human_imus):
            # X分量
            self.ax_pos.plot(time_steps, self.imu_global_positions[:, i, 0].numpy(), 
                           color=self.colors[i], linestyle='-', alpha=0.7,
                           label=f'{IMU_JOINT_NAMES[i]}_X')
            # Y分量
            self.ax_pos.plot(time_steps, self.imu_global_positions[:, i, 1].numpy(), 
                           color=self.colors[i], linestyle='--', alpha=0.7,
                           label=f'{IMU_JOINT_NAMES[i]}_Y')
            # Z分量
            self.ax_pos.plot(time_steps, self.imu_global_positions[:, i, 2].numpy(), 
                           color=self.colors[i], linestyle=':', alpha=0.7,
                           label=f'{IMU_JOINT_NAMES[i]}_Z')
        
        # 添加当前帧指示线
        self.frame_line_pos = self.ax_pos.axvline(x=0, color='black', 
                                                 linestyle='--', linewidth=2)
        self.ax_pos.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax_pos.grid(True, alpha=0.3)
    
    def update_frame(self, frame_idx):
        """更新当前帧的可视化"""
        self.current_frame = frame_idx % self.seq_len
        
        # 更新人体IMU可视化
        for i in range(self.num_human_imus):
            pos = self.imu_global_positions[self.current_frame, i].numpy()
            
            # 更新关节位置点
            self.joint_points[i].remove()
            self.joint_points[i] = self.ax_main.scatter(
                pos[0], pos[1], pos[2],
                color=self.colors[i], s=100, alpha=0.8
            )
            
            # 更新加速度和速度箭头
            acc = self.human_acc[self.current_frame, i].numpy()
            vel = self.human_vel[self.current_frame, i].numpy()
            acc_scale = 0.1  # 缩放因子，使箭头可见
            vel_scale = 0.05  # 速度箭头稍小一些
            
            # 移除旧的加速度箭头并创建新的
            self.quivers_human_acc[i].remove()
            self.quivers_human_acc[i] = self.ax_main.quiver(
                pos[0], pos[1], pos[2],
                acc[0] * acc_scale, acc[1] * acc_scale, acc[2] * acc_scale,
                color=self.colors[i],
                arrow_length_ratio=0.1,
                alpha=0.8,
                linewidth=2
            )
            
            # 移除旧的速度箭头并创建新的
            self.quivers_human_vel[i].remove()
            self.quivers_human_vel[i] = self.ax_main.quiver(
                pos[0] + 0.05, pos[1] + 0.05, pos[2] + 0.05,
                vel[0] * vel_scale, vel[1] * vel_scale, vel[2] * vel_scale,
                color=self.colors[i],
                arrow_length_ratio=0.1,
                alpha=0.6,
                linewidth=1,
                linestyle='--'
            )
            
            # 更新关节坐标系（旋转）
            rot_matrix = self.imu_global_rotations[self.current_frame, i].numpy()
            for axis in range(3):
                axis_vec = rot_matrix[:, axis] * 0.1  # 坐标轴长度
                self.joint_frames[i][axis].set_data_3d(
                    [pos[0], pos[0] + axis_vec[0]],
                    [pos[1], pos[1] + axis_vec[1]],
                    [pos[2], pos[2] + axis_vec[2]]
                )
        
        # 更新物体可视化
        if self.has_object:
            obj_trans_data = self.data.get("obj_trans", torch.zeros(self.seq_len, 3))
            obj_pos = obj_trans_data[self.current_frame].numpy()
            
            # 更新物体位置点
            self.obj_point.remove()
            self.obj_point = self.ax_main.scatter(
                obj_pos[0], obj_pos[1], obj_pos[2],
                color=self.obj_color, s=200, alpha=0.8,
                marker='s'
            )
        
        # 更新时间序列图的当前帧指示线
        self.frame_line_human.set_xdata([self.current_frame, self.current_frame])
        self.frame_line_human_vel.set_xdata([self.current_frame, self.current_frame])
        self.frame_line_pos.set_xdata([self.current_frame, self.current_frame])
        if self.has_object:
            self.frame_line_obj.set_xdata([self.current_frame, self.current_frame])
            self.frame_line_obj_vel.set_xdata([self.current_frame, self.current_frame])
        
        # 更新标题显示当前帧信息
        self.ax_main.set_title(f'IMU Global Positions and Orientations - Frame {self.current_frame+1}/{self.seq_len}')
        
        return []
    
    def animate(self, interval=100):
        """启动动画"""
        print("启动动画播放...")
        print("按空格键暂停/继续，按ESC键退出")
        
        self.anim = FuncAnimation(
            self.fig, 
            self.update_frame,
            frames=self.seq_len,
            interval=interval,
            blit=False,
            repeat=True
        )
        
        # 添加键盘控制
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        plt.tight_layout()
        plt.show()
    
    def _on_key_press(self, event):
        """键盘事件处理"""
        if event.key == ' ':  # 空格键暂停/继续
            if self.anim.running:
                self.anim.pause()
            else:
                self.anim.resume()
        elif event.key == 'escape':  # ESC键退出
            plt.close('all')
        elif event.key == 'left':  # 左箭头后退一帧
            self.current_frame = max(0, self.current_frame - 1)
            self.update_frame(self.current_frame)
            self.fig.canvas.draw()
        elif event.key == 'right':  # 右箭头前进一帧
            self.current_frame = min(self.seq_len - 1, self.current_frame + 1)
            self.update_frame(self.current_frame)
            self.fig.canvas.draw()
    
    def show_static_frame(self, frame_idx=0):
        """显示静态帧"""
        self.update_frame(frame_idx)
        plt.tight_layout()
        plt.show()
    
    def save_animation(self, filename="imu_visualization.gif", fps=10):
        """保存动画为GIF"""
        print(f"正在保存动画到 {filename}...")
        
        anim = FuncAnimation(
            self.fig, 
            self.update_frame,
            frames=self.seq_len,
            interval=1000//fps,
            blit=False,
            repeat=False
        )
        
        anim.save(filename, writer='pillow', fps=fps)
        print(f"动画已保存到 {filename}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='IMU数据可视化工具')
    parser.add_argument('--data_dir', type=str, default='processed_data_0612/debug2',
                       help='数据目录路径')
    parser.add_argument('--seq_idx', type=int, default=0,
                       help='要可视化的序列索引')
    parser.add_argument('--mode', type=str, default='animate', 
                       choices=['animate', 'static', 'save'],
                       help='可视化模式：animate(动画), static(静态), save(保存GIF)')
    parser.add_argument('--frame', type=int, default=0,
                       help='静态模式下要显示的帧索引')
    parser.add_argument('--output', type=str, default='imu_visualization.gif',
                       help='保存动画的文件名')
    parser.add_argument('--fps', type=int, default=1,
                       help='动画帧率')
    parser.add_argument('--interval', type=int, default=10,
                       help='动画更新间隔(毫秒)')
    
    args = parser.parse_args()
    
    try:
        # 创建可视化器
        visualizer = IMUVisualizer(data_dir=args.data_dir, seq_idx=args.seq_idx)
        
        if args.mode == 'animate':
            # 动画模式
            visualizer.animate(interval=args.interval)
        elif args.mode == 'static':
            # 静态模式
            visualizer.show_static_frame(frame_idx=args.frame)
        elif args.mode == 'save':
            # 保存模式
            visualizer.save_animation(filename=args.output, fps=args.fps)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 