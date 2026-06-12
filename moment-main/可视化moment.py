#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOMENT 4特征可视化脚本
特征：小客车上行、小客车下行、非小客车上行、非小客车下行

使用方法：
python 可视化moment.py

输入：
- results/predictions.npz (预测结果)
- results/metrics.json (评估指标)

输出：
- visualization/figures/ (可视化图表)
- visualization/predictions/ (预测结果CSV)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error
from argparse import ArgumentParser

# ========== 项目路径配置 ==========
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_PATH)

# 配置Matplotlib（使用英文避免字体问题）
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


class MOMENTVisualizationRunner:
    """MOMENT 4特征可视化运行器"""
    
    def __init__(self, predictions_path='results/predictions.npz', 
                 metrics_path='results/metrics.json',
                 station_mapping_path='moment_data/station_mapping.json'):
        self.predictions_path = predictions_path
        self.metrics_path = metrics_path
        self.station_mapping_path = station_mapping_path
        
        # 输出目录
        self.output_dir = 'visualization'
        self.figures_dir = os.path.join(self.output_dir, 'figures')
        self.predictions_dir = os.path.join(self.output_dir, 'predictions')
        
        # 创建输出目录
        self._create_output_directory()
        
        # 加载数据
        self.predictions, self.targets = self._load_predictions()
        self.station_mapping = self._load_station_mapping()
        self.metrics = self._load_metrics()
        
        # 配置参数
        self.n_features_per_station = 4
        self.total_features = self.predictions.shape[2]
        self.n_stations = self.total_features // self.n_features_per_station
        self.pred_len = self.predictions.shape[1]
        
        # 特征名称
        self.feature_names = [
            'Passenger Car Up',      # 小客车上行
            'Passenger Car Down',    # 小客车下行
            'Non-Passenger Car Up',  # 非小客车上行
            'Non-Passenger Car Down' # 非小客车下行
        ]
        
        # FaST风格配色
        self.feature_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        print("="*70)
        print("MOMENT Traffic Flow Forecasting - Visualization")
        print("="*70)
        print(f"  Predictions shape: {self.predictions.shape}")
        print(f"  Targets shape: {self.targets.shape}")
        print(f"  Number of stations: {self.n_stations}")
        print(f"  Prediction length: {self.pred_len}")
        print("="*70)
    
    def _create_output_directory(self):
        """创建输出目录"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(os.path.join(self.figures_dir, 'all_stations'), exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)
    
    def _load_predictions(self):
        """加载预测结果"""
        print(f"\n📂 Loading predictions from {self.predictions_path}...")
        data = np.load(self.predictions_path)
        predictions = data['predictions']
        targets = data['targets']
        print(f"  ✓ Predictions: {predictions.shape}")
        print(f"  ✓ Targets: {targets.shape}")
        return predictions, targets
    
    def _load_station_mapping(self):
        """加载站点映射"""
        if os.path.exists(self.station_mapping_path):
            print(f"\n📂 Loading station mapping from {self.station_mapping_path}...")
            with open(self.station_mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            print(f"  ✓ Loaded {len(mapping)} stations")
            return mapping
        else:
            print(f"\n⚠️  Station mapping not found, using default mapping")
            # 生成默认映射
            mapping = {}
            for i in range(self.n_stations):
                mapping[str(i)] = {
                    'station_name': f'Station {i:03d}',
                    'station_code': f'{i:03d}'
                }
            return mapping
    
    def _load_metrics(self):
        """加载评估指标"""
        if os.path.exists(self.metrics_path):
            print(f"\n📂 Loading metrics from {self.metrics_path}...")
            with open(self.metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            print(f"  ✓ MAE: {metrics['overall']['MAE']:.3f}")
            print(f"  ✓ RMSE: {metrics['overall']['RMSE']:.3f}")
            return metrics
        else:
            print(f"\n⚠️  Metrics not found, will compute on-the-fly")
            return None
    
    def compute_overall_metrics(self):
        """计算整体评估指标"""
        mae = mean_absolute_error(self.targets.flatten(), self.predictions.flatten())
        rmse = np.sqrt(mean_squared_error(self.targets.flatten(), self.predictions.flatten()))
        
        epsilon = 1e-8
        mape = np.mean(np.abs((self.targets - self.predictions) / 
                             (np.abs(self.targets) + epsilon))) * 100
        
        print(f"\n{'='*70}")
        print("Overall Test Set Metrics:")
        print(f"{'='*70}")
        print(f"  MAE:  {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"{'='*70}")
        
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    
    def plot_sample_station_predictions(self, sample_indices=[0, 1, 2], 
                                       station_indices=[0, 1, 2],
                                       error_threshold=0.3,
                                       max_annotations=5):
        """绘制指定样本和站点的预测结果（仿照FaST风格）
        
        Args:
            sample_indices: 要绘制的样本索引列表
            station_indices: 要绘制的站点索引列表
            error_threshold: 相对误差阈值（标注用）
            max_annotations: 最大标注数量
        """
        print(f"\n📊 Generating sample predictions plots...")
        
        for sample_idx in sample_indices:
            if sample_idx >= len(self.predictions):
                continue
                
            for station_idx in station_indices:
                if station_idx >= self.n_stations:
                    continue
                
                # 获取站点信息
                station_info = self.station_mapping.get(str(station_idx), {})
                station_name = station_info.get('station_name', f'Station {station_idx}')
                station_code = station_info.get('station_code', f'{station_idx:03d}')
                
                base_feat_idx = station_idx * self.n_features_per_station
                
                # 创建2×2子图（FaST风格）
                fig, axes = plt.subplots(2, 2, figsize=(20, 10))
                axes = axes.flatten()
                
                # 全局统计
                global_mae = mean_absolute_error(
                    self.targets.flatten(), 
                    self.predictions.flatten()
                )
                global_rmse = np.sqrt(mean_squared_error(
                    self.targets.flatten(), 
                    self.predictions.flatten()
                ))
                
                # 标题
                fig.suptitle(
                    f'Station: {station_name} ({station_code}) | Sample #{sample_idx}\n'
                    f'Global MAE: {global_mae:.2f} | RMSE: {global_rmse:.2f}',
                    fontsize=14, fontweight='bold', y=0.995
                )
                
                for i, feat_name in enumerate(self.feature_names):
                    ax = axes[i]
                    feat_idx = base_feat_idx + i
                    
                    pred = self.predictions[sample_idx, :, feat_idx]
                    target = self.targets[sample_idx, :, feat_idx]
                    time_steps = np.arange(len(pred))
                    
                    # === FaST风格绘图 ===
                    # 真实值：实线
                    ax.plot(time_steps, target, label='Actual', linewidth=2.5,
                           color=self.feature_colors[i], alpha=0.9, zorder=3)
                    # 预测值：虚线
                    ax.plot(time_steps, pred, label='Predicted', linewidth=2.5,
                           linestyle='--', color=self.feature_colors[i], alpha=0.7, zorder=2)
                    
                    # 计算误差
                    mae = mean_absolute_error(target, pred)
                    rmse = np.sqrt(mean_squared_error(target, pred))
                    
                    # 误差标注（FaST风格）
                    epsilon = 1e-8
                    abs_error = np.abs(pred - target)
                    relative_error = abs_error / (np.abs(target) + epsilon)
                    
                    # 找出高误差点
                    high_error_mask = relative_error > error_threshold
                    non_zero_mask = np.abs(target) > epsilon
                    high_error_mask = high_error_mask & non_zero_mask
                    
                    high_error_indices = np.where(high_error_mask)[0]
                    if len(high_error_indices) > 0:
                        error_values = relative_error[high_error_indices]
                        sorted_indices = high_error_indices[np.argsort(-error_values)]
                        annotated_indices = sorted_indices[:max_annotations]
                        
                        for idx in annotated_indices:
                            error_pct = relative_error[idx] * 100
                            # 红色圆圈标记真实值
                            ax.plot(idx, target[idx], 'o', color='red', markersize=10,
                                   markeredgecolor='darkred', markeredgewidth=2, zorder=5)
                            # 橙色圆圈标记预测值
                            ax.plot(idx, pred[idx], 'o', color='orange', markersize=8,
                                   markeredgecolor='darkorange', markeredgewidth=1.5, zorder=5)
                            
                            # 误差标注
                            ax.annotate(f'Error:{error_pct:.1f}%\nStep {idx}',
                                       xy=(idx, target[idx]),
                                       xytext=(0, 20),
                                       textcoords='offset points',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                                                alpha=0.7, edgecolor='orange'),
                                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                                       fontsize=8, ha='center')
                    
                    # 子图标题
                    ax.set_title(f'{feat_name}\nMAE: {mae:.3f} | RMSE: {rmse:.3f}',
                               fontsize=12, fontweight='bold', pad=10)
                    ax.set_xlabel('Time Steps (Hours)', fontsize=11)
                    ax.set_ylabel('Traffic Flow (veh/h)', fontsize=11)
                    
                    # 图例和网格
                    ax.legend(loc='best', fontsize=10, framealpha=0.9)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.set_axisbelow(True)
                
                # 保存
                plt.tight_layout(rect=[0, 0, 1, 0.98])
                save_path = os.path.join(
                    self.figures_dir,
                    f'sample{sample_idx}_station{station_code}.png'
                )
                plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
                print(f"  ✓ Saved: {os.path.basename(save_path)}")
                plt.close()
        
        print(f"\n✓ Sample plots saved to {self.figures_dir}/")
    
    def plot_all_stations_first_step(self, error_threshold=0.3, max_annotations=5):
        """绘制所有站点第1步预测结果（仿照FaST的plot_all_stations）
        
        Args:
            error_threshold: 相对误差阈值
            max_annotations: 最大标注数量
        """
        print(f"\n📊 Generating all stations plots (Step 1 only)...")
        
        # 只绘制第1步预测
        step_idx = 0
        total_samples = self.predictions.shape[0]
        times = np.arange(total_samples)
        
        for feat_idx, feat_name in enumerate(self.feature_names):
            print(f"  Processing {feat_name}...")
            feat_dir = os.path.join(self.figures_dir, 'all_stations', feat_name.replace(' ', '_'))
            os.makedirs(feat_dir, exist_ok=True)
            
            for station_idx in range(self.n_stations):
                base_feat_idx = station_idx * self.n_features_per_station
                feat_idx_global = base_feat_idx + feat_idx
                
                # 获取所有样本在该站点、该特征、第1步的预测
                pred_seq = self.predictions[:, step_idx, feat_idx_global]
                true_seq = self.targets[:, step_idx, feat_idx_global]
                
                # 获取站点信息
                station_info = self.station_mapping.get(str(station_idx), {})
                station_name = station_info.get('station_name', f'Station {station_idx}')
                station_code = station_info.get('station_code', f'{station_idx:03d}')
                
                # 创建图表
                plt.figure(figsize=(14, 4))
                plt.plot(times, true_seq, label='Actual', linewidth=1.8,
                        color='#1f77b4', zorder=3)
                plt.plot(times, pred_seq, label='Predicted', linewidth=1.8,
                        color='#ff4b5c', linestyle='--', zorder=2)
                
                # 误差标注
                epsilon = 1e-8
                abs_error = np.abs(pred_seq - true_seq)
                relative_error = abs_error / (np.abs(true_seq) + epsilon)
                
                high_error_mask = relative_error > error_threshold
                non_zero_mask = np.abs(true_seq) > epsilon
                high_error_mask = high_error_mask & non_zero_mask
                
                high_error_indices = np.where(high_error_mask)[0]
                if len(high_error_indices) > 0:
                    error_values = relative_error[high_error_indices]
                    sorted_indices = high_error_indices[np.argsort(-error_values)]
                    annotated_indices = sorted_indices[:max_annotations]
                    
                    for idx in annotated_indices:
                        time_point = times[idx]
                        true_val = true_seq[idx]
                        pred_val = pred_seq[idx]
                        error_pct = relative_error[idx] * 100
                        
                        plt.plot(time_point, true_val, 'o', color='red', markersize=12,
                                markeredgecolor='darkred', markeredgewidth=2, zorder=5)
                        plt.plot(time_point, pred_val, 'o', color='orange', markersize=8,
                                markeredgecolor='darkorange', markeredgewidth=1.5, zorder=5)
                        
                        plt.annotate(f'Error:{error_pct:.1f}%\nSample {idx}',
                                   xy=(time_point, true_val),
                                   xytext=(0, 20),
                                   textcoords='offset points',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                                            alpha=0.7, edgecolor='orange'),
                                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                                   fontsize=8, ha='center')
                
                # 标题和标签
                plt.title(f'Station {station_code} - {feat_name} (Step 1)')
                plt.xlabel('Sample Index')
                plt.ylabel('Traffic Flow (veh/h)')
                plt.xticks(rotation=30, ha='right')
                plt.grid(alpha=0.3)
                plt.legend()
                plt.tight_layout()
                
                # 保存
                save_path = os.path.join(feat_dir, f'station_{station_code}.png')
                plt.savefig(save_path, dpi=150)
                plt.close()
                
                if station_idx % 20 == 0:
                    print(f"    Processed {station_idx}/{self.n_stations} stations")
        
        print(f"\n✓ All stations plots saved to {os.path.join(self.figures_dir, 'all_stations')}/")
    
    def save_predictions_csv(self):
        """保存所有预测结果为CSV（仿照FaST的save_all_nodes_csv）"""
        print(f"\n💾 Saving predictions to CSV...")
        
        # 提取第1步预测
        pred_step0 = self.predictions[:, 0, :]  # (samples, features)
        true_step0 = self.targets[:, 0, :]
        
        rows = []
        total_samples = pred_step0.shape[0]
        
        for sample_idx in range(total_samples):
            time_str = f"Sample_{sample_idx}"
            
            for station_idx in range(self.n_stations):
                station_info = self.station_mapping.get(str(station_idx), {})
                station_name = station_info.get('station_name', f'Station_{station_idx:03d}')
                station_code = station_info.get('station_code', f'{station_idx:03d}')
                
                base_feat_idx = station_idx * self.n_features_per_station
                
                for feat_idx, feat_name in enumerate(self.feature_names):
                    true_val = round(true_step0[sample_idx, base_feat_idx + feat_idx], 2)
                    pred_val = round(pred_step0[sample_idx, base_feat_idx + feat_idx], 2)
                    
                    rows.append([
                        station_name,
                        station_code,
                        time_str,
                        feat_name,
                        true_val,
                        pred_val
                    ])
        
        # 创建DataFrame
        df = pd.DataFrame(rows, columns=[
            '站点名称', '站点编号', '时间', '特征', '真实值', '预测值'
        ])
        
        # 保存CSV
        save_path = os.path.join(self.predictions_dir, 'all_predictions.csv')
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print(f"  ✓ CSV saved to {save_path}")
        print(f"  Total records: {len(df)}")
        print(f"  Stations: {self.n_stations}")
        print(f"  Samples: {total_samples}")
    
    def save_finetune_data(self):
        """保存大模型微调所需的NPZ文件（仿照FaST的save_finetune_files）"""
        print(f"\n💾 Saving finetune data...")
        
        # 保存预测值
        pred_path = os.path.join(self.predictions_dir, 'finetune_data.npz')
        np.savez(pred_path, prediction=self.predictions)
        print(f"  ✓ Saved predictions: {pred_path}")
        print(f"    Shape: {self.predictions.shape}")
        print(f"    Range: [{self.predictions.min():.2f}, {self.predictions.max():.2f}]")
        
        # 保存真实值
        target_path = os.path.join(self.predictions_dir, 'finetune_real_traffic.npz')
        np.savez(target_path, target=self.targets)
        print(f"  ✓ Saved targets: {target_path}")
        print(f"    Shape: {self.targets.shape}")
        print(f"    Range: [{self.targets.min():.2f}, {self.targets.max():.2f}]")
        
        # 生成时间戳
        timestamps = []
        total_samples = self.predictions.shape[0]
        for i in range(total_samples):
            timestamps.append(f"Sample_{i}")
        
        pd.DataFrame({
            'sample_id': range(total_samples),
            'pred_start_time': timestamps
        }).to_csv(os.path.join(self.predictions_dir, 'timestamps.csv'), index=False)
        
        print(f"  ✓ Saved timestamps: {os.path.join(self.predictions_dir, 'timestamps.csv')}")
        
        # MAE统计
        mae = np.abs(self.predictions - self.targets).mean()
        print(f"  MAE: {mae:.2f}")
    
    def run(self, plot_samples=True, plot_all_stations=True, save_csv=True, save_finetune=True):
        """主运行流程
        
        Args:
            plot_samples: 是否绘制样本预测图
            plot_all_stations: 是否绘制所有站点图
            save_csv: 是否保存CSV
            save_finetune: 是否保存微调数据
        """
        print("\n🚀 START - MOMENT 4-Feature Visualization")
        print("="*70)
        
        # 1. 计算指标
        self.compute_overall_metrics()
        
        # 2. 绘制样本预测图
        if plot_samples:
            self.plot_sample_station_predictions(
                sample_indices=[0, 1, 2],
                station_indices=list(range(min(3, self.n_stations))),
                error_threshold=0.3,
                max_annotations=5
            )
        
        # 3. 绘制所有站点图（耗时较长）
        if plot_all_stations:
            self.plot_all_stations_first_step(
                error_threshold=0.3,
                max_annotations=5
            )
        
        # 4. 保存CSV
        if save_csv:
            self.save_predictions_csv()
        
        # 5. 保存微调数据
        if save_finetune:
            self.save_finetune_data()
        
        print("\n" + "="*70)
        print("🎉 ALL DONE!")
        print("="*70)
        print(f"\nOutput files:")
        print(f"  - {self.figures_dir}/sample*_station*.png")
        print(f"  - {os.path.join(self.figures_dir, 'all_stations')}/")
        print(f"  - {os.path.join(self.predictions_dir, 'all_predictions.csv')}")
        print(f"  - {os.path.join(self.predictions_dir, 'finetune_data.npz')}")
        print(f"  - {os.path.join(self.predictions_dir, 'finetune_real_traffic.npz')}")
        print("="*70)


def main():
    """主函数"""
    parser = ArgumentParser(description='MOMENT Traffic Flow Visualization')
    parser.add_argument('--pred', default='results/predictions.npz',
                       help='Path to predictions NPZ file')
    parser.add_argument('--metrics', default='results/metrics.json',
                       help='Path to metrics JSON file')
    parser.add_argument('--mapping', default='moment_data/station_mapping.json',
                       help='Path to station mapping JSON file')
    parser.add_argument('--no-samples', action='store_true',
                       help='Skip sample predictions plots')
    parser.add_argument('--no-stations', action='store_true',
                       help='Skip all stations plots')
    parser.add_argument('--no-csv', action='store_true',
                       help='Skip CSV export')
    parser.add_argument('--no-finetune', action='store_true',
                       help='Skip finetune data export')
    
    args = parser.parse_args()
    
    runner = MOMENTVisualizationRunner(
        predictions_path=args.pred,
        metrics_path=args.metrics,
        station_mapping_path=args.mapping
    )
    
    runner.run(
        plot_samples=not args.no_samples,
        plot_all_stations=not args.no_stations,
        save_csv=not args.no_csv,
        save_finetune=not args.no_finetune
    )


if __name__ == '__main__':
    main()