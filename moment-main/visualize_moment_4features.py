#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOMENT 4特征可视化脚本
特征：小客车上行、小客车下行、非小客车上行、非小客车下行
Usage: python visualize_moment_4features.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class MOMENTVisualizationRunner:
    """MOMENT 4特征可视化运行器（仿照FaST风格）"""
    
    def __init__(self, predictions_path=None, metrics_path=None, dataset_path=None, 
                 mapping_path=None, start_time="2023-09-01", time_step_hours=1):
        """
        初始化可视化运行器
        
        Args:
            predictions_path: 预测结果路径（npz文件）
            metrics_path: 指标文件路径（json文件）
            dataset_path: 原始数据集路径（用于时间轴映射）
            mapping_path: 站点映射文件路径
            start_time: 数据集起始时间
            time_step_hours: 时间步长（小时）
        """
        self.predictions_path = predictions_path or 'results/predictions.npz'
        self.metrics_path = metrics_path or 'results/metrics.json'
        self.dataset_path = dataset_path or 'dataset/his_data_with_names.csv'
        self.mapping_path = mapping_path or 'dataset/station_mapping.json'
        self.start_time = start_time
        self.time_step_hours = time_step_hours
        
        # 输出目录
        self.output_dir = 'visualization-results'
        self._create_output_directory()
        
        # 特征定义
        self.feature_names = [
            'Passenger Car Up',      # 小客车上行
            'Passenger Car Down',    # 小客车下行
            'Non-Passenger Car Up',  # 非小客车上行
            'Non-Passenger Car Down' # 非小客车下行
        ]
        
        # 配色方案（FaST风格）
        self.feature_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        print("="*70)
        print("MOMENT 4-Feature Visualization Runner")
        print("="*70)
    
    def _create_output_directory(self):
        """创建输出目录结构"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f'{self.output_dir}/metrics', exist_ok=True)
        os.makedirs(f'{self.output_dir}/predictions', exist_ok=True)
        os.makedirs(f'{self.output_dir}/figures', exist_ok=True)
        os.makedirs(f'{self.output_dir}/figures/overview', exist_ok=True)
        os.makedirs(f'{self.output_dir}/figures/per_station', exist_ok=True)
        print(f"✓ Output directory created: {self.output_dir}")
    
    def load_predictions(self):
        """加载预测结果"""
        print("\n📂 Loading predictions...")
        if not os.path.exists(self.predictions_path):
            raise FileNotFoundError(f"Predictions file not found: {self.predictions_path}")
        
        data = np.load(self.predictions_path)
        predictions = data['predictions']
        targets = data['targets'] if 'targets' in data else None
        
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Targets shape: {targets.shape if targets is not None else 'N/A'}")
        
        return predictions, targets
    
    def load_metrics(self):
        """加载评估指标"""
        print("\n📊 Loading metrics...")
        if not os.path.exists(self.metrics_path):
            print(f"  ⚠️  Metrics file not found: {self.metrics_path}")
            return None
        
        with open(self.metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        print(f"  ✓ Metrics loaded")
        if 'overall' in metrics:
            print(f"  Overall MAE: {metrics['overall']['MAE']:.3f}")
            print(f"  Overall RMSE: {metrics['overall']['RMSE']:.3f}")
            print(f"  Overall MAPE: {metrics['overall']['MAPE']:.2f}%")
        
        return metrics
    
    def load_station_mapping(self):
        """加载站点映射"""
        print("\n🗺️  Loading station mapping...")
        if not os.path.exists(self.mapping_path):
            print(f"  ⚠️  Mapping file not found: {self.mapping_path}")
            print(f"  → Using default mapping (Station_001, Station_002, ...)")
            return None
        
        with open(self.mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        print(f"  ✓ Loaded {len(mapping)} station mappings")
        return mapping
    
    def compute_time_axis(self, num_samples, start_sample_idx=0):
        """
        计算时间轴（仿照FaST风格）
        
        Args:
            num_samples: 样本数量
            start_sample_idx: 起始样本索引
        
        Returns:
            times: 时间列表
        """
        start_dt = datetime.strptime(self.start_time, "%Y-%m-%d")
        
        times = []
        for i in range(num_samples):
            # 预测时间 = 起始时间 + (样本索引 + 预测步长) * 时间步长
            time_offset = (start_sample_idx + i + 8) * self.time_step_hours  # +8是预测步长
            current_time = start_dt + timedelta(hours=time_offset)
            times.append(current_time)
        
        return times
    
    def plot_overview(self, predictions, targets, metrics=None):
        """
        绘制4特征总览图（仿照FaST的plot_overview）
        显示所有站点在第一个预测步长的平均趋势
        """
        print("\n📈 Generating overview plots...")
        
        # 提取第一个预测步长的结果
        pred_step0 = predictions[:, 0, :]  # (samples, features)
        target_step0 = targets[:, 0, :] if targets is not None else None
        
        num_samples = pred_step0.shape[0]
        times = self.compute_time_axis(num_samples)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 10))
        axes = axes.flatten()
        
        global_metrics = metrics['overall'] if metrics and 'overall' in metrics else None
        global_title = ""
        if global_metrics:
            global_title = f"Global MAE: {global_metrics['MAE']:.2f} | RMSE: {global_metrics['RMSE']:.2f}"
        
        fig.suptitle(f'MOMENT 4-Feature Overview\n{global_title}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        for idx, (ax, feat_name, color) in enumerate(zip(axes, self.feature_names, self.feature_colors)):
            # 计算所有站点在该特征上的平均值
            pred_all = pred_step0[:, idx * 4:(idx + 1) * 4].mean(axis=1)
            
            if target_step0 is not None:
                target_all = target_step0[:, idx * 4:(idx + 1) * 4].mean(axis=1)
                
                # 真实值：实线
                ax.plot(times, target_all, label='Actual', linewidth=2.5, 
                       color=color, alpha=0.9, zorder=3)
                # 预测值：虚线
                ax.plot(times, pred_all, label='Predicted', linewidth=2.5, 
                       linestyle='--', color=color, alpha=0.7, zorder=2)
            else:
                ax.plot(times, pred_all, label='Predicted', linewidth=2.5, 
                       linestyle='-', color=color)
            
            # 计算该特征的指标
            if target_step0 is not None:
                mae = mean_absolute_error(target_all, pred_all)
                rmse = np.sqrt(mean_squared_error(target_all, pred_all))
                ax.set_title(f'{feat_name}\nMAE: {mae:.3f} | RMSE: {rmse:.3f}', 
                           fontsize=12, fontweight='bold', pad=10)
            else:
                ax.set_title(f'{feat_name}', fontsize=12, fontweight='bold', pad=10)
            
            ax.set_xlabel('Time', fontsize=11)
            ax.set_ylabel('Traffic Flow (veh/h)', fontsize=11)
            
            # 时间轴格式化
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.tick_params(axis='x', rotation=30)
            
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=10, framealpha=0.9)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        save_path = f'{self.output_dir}/figures/overview_4features.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: overview_4features.png")
        plt.close()
    
    def plot_per_station(self, predictions, targets, station_mapping=None, 
                        max_stations=5, max_samples=20):
        """
        绘制单个站点的详细趋势图（仿照FaST风格）
        """
        print("\n📊 Generating per-station plots...")
        
        n_features_per_station = 4
        total_features = predictions.shape[2]
        n_stations = total_features // n_features_per_station
        n_stations_to_plot = min(max_stations, n_stations)
        
        for station_idx in range(n_stations_to_plot):
            # 获取站点信息
            if station_mapping:
                station_info = station_mapping.get(str(station_idx), {})
                station_name = station_info.get('station_name', f'Station_{station_idx:03d}')
                station_code = station_info.get('station_code', f'{station_idx:03d}')
            else:
                station_name = f'Station_{station_idx:03d}'
                station_code = f'{station_idx:03d}'
            
            print(f"\n  Processing: {station_name} ({station_code})")
            
            base_feat_idx = station_idx * n_features_per_station
            n_samples_to_plot = min(max_samples, predictions.shape[0])
            
            # 创建2×2子图
            fig, axes = plt.subplots(2, 2, figsize=(20, 10))
            axes = axes.flatten()
            
            station_title = f'Station: {station_name} ({station_code})'
            fig.suptitle(station_title, fontsize=14, fontweight='bold', y=0.995)
            
            for i, (feat_name, color) in enumerate(zip(self.feature_names, self.feature_colors)):
                ax = axes[i]
                feat_idx = base_feat_idx + i
                
                # 取前N个样本的第一个预测步长
                pred_seq = predictions[:n_samples_to_plot, 0, feat_idx]
                target_seq = targets[:n_samples_to_plot, 0, feat_idx] if targets is not None else None
                
                time_steps = np.arange(n_samples_to_plot)
                
                # 绘图
                if target_seq is not None:
                    ax.plot(time_steps, target_seq, label='Actual', linewidth=2.5, 
                           color=color, alpha=0.9, zorder=3)
                    ax.plot(time_steps, pred_seq, label='Predicted', linewidth=2.5, 
                           linestyle='--', color=color, alpha=0.7, zorder=2)
                    
                    # 计算指标
                    mae = mean_absolute_error(target_seq, pred_seq)
                    rmse = np.sqrt(mean_squared_error(target_seq, pred_seq))
                    
                    # 高误差点标注（FaST风格）
                    self._annotate_high_errors(ax, pred_seq, target_seq, time_steps, 
                                              color, error_threshold=0.3, max_annotations=5)
                    
                    ax.set_title(f'{feat_name}\nMAE: {mae:.3f} | RMSE: {rmse:.3f}', 
                               fontsize=12, fontweight='bold', pad=10)
                else:
                    ax.plot(time_steps, pred_seq, label='Predicted', linewidth=2.5, 
                           color=color)
                    ax.set_title(f'{feat_name}', fontsize=12, fontweight='bold', pad=10)
                
                ax.set_xlabel('Samples', fontsize=11)
                ax.set_ylabel('Traffic Flow (veh/h)', fontsize=11)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(loc='best', fontsize=10, framealpha=0.9)
            
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            save_path = f'{self.output_dir}/figures/per_station/{station_code}.png'
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            print(f"    ✓ Saved: {station_code}.png")
            plt.close()
    
    def _annotate_high_errors(self, ax, pred_seq, target_seq, time_steps, 
                             color, error_threshold=0.3, max_annotations=5):
        """
        标注高误差点（仿照FaST风格）
        """
        epsilon = 1e-8
        abs_error = np.abs(pred_seq - target_seq)
        relative_error = abs_error / (np.abs(target_seq) + epsilon)
        
        # 找出误差超过阈值的点
        high_error_mask = relative_error > error_threshold
        non_zero_mask = np.abs(target_seq) > epsilon
        high_error_mask = high_error_mask & non_zero_mask
        
        high_error_indices = np.where(high_error_mask)[0]
        if len(high_error_indices) == 0:
            return
        
        # 按误差大小排序，取Top N
        error_values = relative_error[high_error_indices]
        sorted_indices = high_error_indices[np.argsort(-error_values)]
        annotated_indices = sorted_indices[:max_annotations]
        
        for idx in annotated_indices:
            error_pct = relative_error[idx] * 100
            
            # 红色圆圈标记真实值
            ax.plot(time_steps[idx], target_seq[idx], 'o', color='red', markersize=10, 
                   markeredgecolor='darkred', markeredgewidth=2, zorder=5)
            # 橙色圆圈标记预测值
            ax.plot(time_steps[idx], pred_seq[idx], 'o', color='orange', markersize=8,
                   markeredgecolor='darkorange', markeredgewidth=1.5, zorder=5)
            
            # 添加误差标注
            ax.annotate(f'Error:{error_pct:.1f}%\nSample {idx}',
                       xy=(time_steps[idx], target_seq[idx]),
                       xytext=(0, 20),
                       textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                                alpha=0.7, edgecolor='orange'),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                       fontsize=8, ha='center')
    
    def save_predictions_csv(self, predictions, targets, station_mapping=None):
        """
        保存所有站点的预测结果为CSV（仿照FaST的save_all_nodes_csv）
        """
        print("\n💾 Saving predictions to CSV...")
        
        n_features_per_station = 4
        total_features = predictions.shape[2]
        n_stations = total_features // n_features_per_station
        n_samples = predictions.shape[0]
        
        # 提取第一个预测步长的结果
        pred_step0 = predictions[:, 0, :]
        target_step0 = targets[:, 0, :] if targets is not None else None
        
        # 生成时间轴
        times = self.compute_time_axis(n_samples)
        
        rows = []
        for sample_idx in range(n_samples):
            time_str = times[sample_idx].strftime("%Y-%m-%d %H:%M:%S")
            
            for station_idx in range(n_stations):
                # 站点名称
                if station_mapping:
                    station_info = station_mapping.get(str(station_idx), {})
                    station_name = station_info.get('station_name', f'Station_{station_idx:03d}')
                    station_code = station_info.get('station_code', f'{station_idx:03d}')
                else:
                    station_name = f'Station_{station_idx:03d}'
                    station_code = f'{station_idx:03d}'
                
                base_feat_idx = station_idx * n_features_per_station
                
                for feat_idx, feat_name in enumerate(self.feature_names):
                    pred_val = round(pred_step0[sample_idx, base_feat_idx + feat_idx], 2)
                    true_val = round(target_step0[sample_idx, base_feat_idx + feat_idx], 2) \
                              if target_step0 is not None else None
                    
                    rows.append([
                        station_name,
                        station_code,
                        time_str,
                        feat_name,
                        true_val if true_val is not None else 'N/A',
                        pred_val
                    ])
        
        # 创建DataFrame
        df = pd.DataFrame(rows, columns=[
            '站点名称', '站点编号', '时间', '特征', '真实值', '预测值'
        ])
        
        # 保存CSV
        save_path = f'{self.output_dir}/predictions/all_predictions.csv'
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"  ✓ Saved: all_predictions.csv")
        print(f"    Total records: {len(df)}")
        print(f"    Stations: {n_stations}")
        print(f"    Samples: {n_samples}")
        
        return save_path
    
    def save_metrics_summary(self, metrics):
        """保存指标摘要到CSV"""
        if metrics is None:
            return
        
        print("\n📊 Saving metrics summary...")
        
        rows = []
        if 'per_feature' in metrics:
            for feat_name, feat_metrics in metrics['per_feature'].items():
                rows.append([
                    feat_name,
                    feat_metrics.get('MAE', 'N/A'),
                    feat_metrics.get('RMSE', 'N/A'),
                    feat_metrics.get('MAPE', 'N/A')
                ])
        
        if rows:
            df = pd.DataFrame(rows, columns=['Feature', 'MAE', 'RMSE', 'MAPE'])
            save_path = f'{self.output_dir}/metrics/feature_metrics.csv'
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"  ✓ Saved: feature_metrics.csv")
    
    def run(self):
        """主运行流程"""
        print("\n" + "="*70)
        print("🚀 START - MOMENT 4-Feature Visualization")
        print("="*70)
        
        # 1. 加载预测结果
        predictions, targets = self.load_predictions()
        
        # 2. 加载指标
        metrics = self.load_metrics()
        
        # 3. 加载站点映射
        station_mapping = self.load_station_mapping()
        
        # 4. 绘制总览图
        self.plot_overview(predictions, targets, metrics)
        
        # 5. 绘制单个站点详细图
        self.plot_per_station(predictions, targets, station_mapping, 
                             max_stations=5, max_samples=20)
        
        # 6. 保存CSV
        self.save_predictions_csv(predictions, targets, station_mapping)
        
        # 7. 保存指标摘要
        self.save_metrics_summary(metrics)
        
        print("\n" + "="*70)
        print("🎉 ALL DONE!")
        print("="*70)
        print(f"\nOutput files saved to: {self.output_dir}/")
        print(f"  - figures/overview_4features.png    : Overview plots")
        print(f"  - figures/per_station/*.png         : Per-station plots")
        print(f"  - predictions/all_predictions.csv   : All predictions")
        print(f"  - metrics/feature_metrics.csv       : Feature metrics")
        print("="*70)


def main():
    """主函数"""
    runner = MOMENTVisualizationRunner(
        predictions_path='results/predictions.npz',
        metrics_path='results/metrics.json',
        dataset_path='dataset/his_data_with_names.csv',
        mapping_path='dataset/station_mapping.json',
        start_time='2023-09-01',
        time_step_hours=1
    )
    runner.run()


if __name__ == '__main__':
    main()