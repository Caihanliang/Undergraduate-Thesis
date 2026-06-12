"""
python inference.py

# 推理训练集（你的需求）
python inference.py --mode train

# 推理验证集
python inference.py --mode val

# 推理测试集（默认）
python inference.py --mode test
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from momentfm import MOMENTPipeline
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_absolute_error, mean_squared_error
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

class TrafficFlowDataset(Dataset):
    """Custom Dataset for traffic flow data"""
    
    def __init__(self, input_data, target_data=None):
        self.input_data = input_data
        self.target_data = target_data
        
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        if self.target_data is not None:
            return {
                'x_enc': torch.FloatTensor(self.input_data[idx]),
                'y': torch.FloatTensor(self.target_data[idx])
            }
        else:
            return {
                'x_enc': torch.FloatTensor(self.input_data[idx])
            }

def load_model(checkpoint_path, n_features, forecast_horizon=96, seq_len=8):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
    # Set mirror endpoint for faster download in China
    import os
    if 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print(f"  ✓ Using HuggingFace mirror: {os.environ['HF_ENDPOINT']}")
    
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", 
        model_kwargs={
            'task_name': 'forecasting',
            'forecast_horizon': forecast_horizon,
            'n_channels': n_features,
            'seq_len': seq_len,  # IMPORTANT: Must match training configuration
            'head_dropout': 0.1,
            'weight_decay': 0,
            'freeze_encoder': True,
            'freeze_embedder': True,
            'freeze_head': False,
        },
    )
    
    model.init()
    
    # Load checkpoint - PyTorch 2.6+ compatibility fix
    try:
        # Try with weights_only=True first (safer, default in PyTorch 2.6+)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except Exception as e:
        print(f"  ⚠️  weights_only=True failed: {str(e)[:100]}...")
        print("  Retrying with weights_only=False (ensure checkpoint is from trusted source)")
        # Fallback to weights_only=False for checkpoints with numpy arrays
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    print(f"  Validation loss: {checkpoint['val_loss']:.6f}")
    print(f"  Validation MAE: {checkpoint['val_mae']:.6f}")
    
    return model, checkpoint

def load_test_data(batch_size=32, mode='test'):
    """
    Load dataset for inference
    
    Args:
        batch_size: Batch size for DataLoader
        mode: 'train', 'val', or 'test' - which dataset to load
    """
    print(f"\nLoading {mode.upper()} data...")
    
    # Load dataset based on mode
    dataset_path = f'moment_data/{mode}_dataset.npz'
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}\n"
            f"Please run preprocessing first: python preprocess_data.py"
        )
    
    data = np.load(dataset_path)
    input_data = data['input']
    target_data = data['target']
    
    print(f"{mode.capitalize()} samples: {len(input_data)}")
    print(f"Total features: {input_data.shape[2]}")
    print(f"Number of stations: {input_data.shape[2] // 4}")
    
    # Create dataset and dataloader
    test_dataset = TrafficFlowDataset(input_data, target_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Load station mapping
    mapping_path = 'dataset/station_mapping.json'
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            station_mapping = json.load(f)
        print(f"Station mapping loaded: {len(station_mapping)} stations")
    else:
        station_mapping = {}
        print("⚠️  Warning: Station mapping not found, using default names")
    
    return test_loader, station_mapping, input_data.shape[2]

def evaluate_model(model, dataloader, device, norm_params=None, station_mapping=None):
    """
    Evaluate model on test set with comprehensive metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print("\n" + "="*70)
    print("Evaluating on Test Set...")
    print("="*70)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            x_enc = batch['x_enc'].to(device)
            
            # Transpose to match MOMENT's expected input format: [batch, n_channels, seq_len]
            x_enc_transposed = x_enc.permute(0, 2, 1)
            
            output = model(x_enc=x_enc_transposed)
            predictions = output.forecast  # [batch, n_features, pred_len]
            
            # Transpose back to [batch, pred_len, n_features]
            predictions = predictions.permute(0, 2, 1)
            
            all_predictions.append(predictions.cpu().numpy())
            
            if 'y' in batch:
                all_targets.append(batch['y'].cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets, axis=0)
        
        # === 计算综合指标 ===
        mse = mean_squared_error(all_targets.flatten(), all_predictions.flatten())
        mae = mean_absolute_error(all_targets.flatten(), all_predictions.flatten())
        rmse = np.sqrt(mse)
        
        # MAPE (Mean Absolute Percentage Error)
        mask = np.abs(all_targets) > 1e-8
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((all_targets[mask] - all_predictions[mask]) / all_targets[mask])) * 100
        else:
            mape = 0.0
        
        # === 逐特征指标（重要：分析不同车型和方向的预测性能） ===
        n_features = all_predictions.shape[2]
        feature_names = ['Passenger Car Up', 'Passenger Car Down', 
                        'Non-Passenger Car Up', 'Non-Passenger Car Down']
        
        print("\n" + "="*70)
        print("Overall Test Set Metrics:")
        print("="*70)
        print(f"  MSE:   {mse:.4f}")
        print(f"  RMSE:  {rmse:.4f}")
        print(f"  MAE:   {mae:.4f}  ← 平均绝对误差（车辆数/小时）")
        print(f"  MAPE:  {mape:.2f}%  ← 平均百分比误差")
        print("="*70)
        
        # 逐特征详细指标
        print("\n" + "="*70)
        print("Per-Feature Metrics (Averaged Across All Stations):")
        print("="*70)
        print(f"{'Feature':<25} {'MAE':<12} {'RMSE':<12} {'MAPE':<12}")
        print("-"*70)
        
        per_feature_metrics = {}
        for i in range(min(4, n_features)):
            # 提取该特征在所有站点的数据
            feat_pred = all_predictions[:, :, i::4].flatten()
            feat_true = all_targets[:, :, i::4].flatten()
            
            feat_mae = mean_absolute_error(feat_true, feat_pred)
            feat_rmse = np.sqrt(mean_squared_error(feat_true, feat_pred))
            
            mask_feat = np.abs(feat_true) > 1e-8
            if np.sum(mask_feat) > 0:
                feat_mape = np.mean(np.abs((feat_true[mask_feat] - feat_pred[mask_feat]) / feat_true[mask_feat])) * 100
            else:
                feat_mape = 0.0
            
            print(f"{feature_names[i]:<25} {feat_mae:<12.3f} {feat_rmse:<12.3f} {feat_mape:<12.2f}%")
            
            per_feature_metrics[feature_names[i]] = {
                'MAE': float(feat_mae),
                'RMSE': float(feat_rmse),
                'MAPE': float(feat_mape)
            }
        
        print("="*70)
        
        # === 逐站点指标（Top 10） ===
        print("\n" + "="*70)
        print("Top 10 Stations by MAE (Averaged Across Features):")
        print("="*70)
        
        n_stations = n_features // 4
        station_maes = []
        
        for station_idx in range(n_stations):
            base_idx = station_idx * 4
            station_pred = all_predictions[:, :, base_idx:base_idx+4].flatten()
            station_true = all_targets[:, :, base_idx:base_idx+4].flatten()
            
            station_mae = mean_absolute_error(station_true, station_pred)
            station_maes.append((station_idx, station_mae))
        
        # 按MAE排序（从低到高）
        station_maes.sort(key=lambda x: x[1])
        
        print(f"{'Rank':<6} {'Station ID':<15} {'MAE':<12}")
        print("-"*70)
        for rank, (station_idx, station_mae) in enumerate(station_maes[:10], 1):
            station_info = station_mapping.get(str(station_idx), {})
            station_name = station_info.get('station_name', f'Station {station_idx}')
            print(f"{rank:<6} {station_name:<15} {station_mae:<12.3f}")
        
        print("="*70)
        
        # === 保存指标 ===
        metrics = {
            'overall': {
                'MSE': float(mse),
                'RMSE': float(rmse),
                'MAE': float(mae),
                'MAPE': float(mape)
            },
            'per_feature': per_feature_metrics,
            'per_station': {str(idx): float(mae) for idx, mae in station_maes[:20]}
        }
        
        # 保存到JSON
        metrics_path = 'results/test_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Metrics saved to {metrics_path}")
        
        return all_predictions, all_targets, metrics
    else:
        print("\n⚠️  Warning: No targets available for evaluation")
        return all_predictions, None, None

def plot_comprehensive_predictions(predictions, targets, station_mapping, 
                                   save_dir='results', n_samples=3, n_stations=2):
    """
    绘制综合预测结果，仿照FaST项目可视化风格
    特征：小客车上行、小客车下行、非小客车上行、非小客车下行
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_features_per_station = 4
    total_features = predictions.shape[2]
    actual_n_stations = total_features // n_features_per_station
    
    # 特征名称（英文，避免字体问题）
    feature_names = [
        'Passenger Car Up',      # 小客车上行
        'Passenger Car Down',    # 小客车下行
        'Non-Passenger Car Up',  # 非小客车上行
        'Non-Passenger Car Down' # 非小客车下行
    ]
    feature_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # FaST风格配色
    
    # 全局统计
    global_mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    global_rmse = np.sqrt(mean_squared_error(targets.flatten(), predictions.flatten()))
    
    print(f"\n" + "="*70)
    print("Generating Comprehensive Visualization (FaST Style)...")
    print("="*70)
    print(f"  Global MAE:  {global_mae:.3f}")
    print(f"  Global RMSE: {global_rmse:.3f}")
    print(f"  Samples to plot: {min(n_samples, len(predictions))}")
    print(f"  Stations to plot: {min(n_stations, actual_n_stations)}")
    
    for sample_idx in range(min(n_samples, len(predictions))):
        for station_idx in range(min(n_stations, actual_n_stations)):
            # 获取站点信息
            station_info = station_mapping.get(str(station_idx), {})
            station_name = station_info.get('station_name', f'Station {station_idx}')
            station_code = station_info.get('station_code', f'{station_idx:03d}')
            
            base_feat_idx = station_idx * n_features_per_station
            
            # 创建2×2子图布局（FaST风格）
            fig, axes = plt.subplots(2, 2, figsize=(20, 10))
            axes = axes.flatten()
            
            fig.suptitle(f'Station: {station_name} ({station_code}) | Sample #{sample_idx}\n'
                        f'Global MAE: {global_mae:.2f} | RMSE: {global_rmse:.2f}', 
                        fontsize=14, fontweight='bold', y=0.995)
            
            for i, feat_name in enumerate(feature_names):
                ax = axes[i]
                feat_idx = base_feat_idx + i
                
                pred = predictions[sample_idx, :, feat_idx]
                target = targets[sample_idx, :, feat_idx]
                
                time_steps = np.arange(len(pred))
                
                # === FaST风格绘图 ===
                # 真实值：实线，较粗
                ax.plot(time_steps, target, label='Actual', linewidth=2.5, 
                       color=feature_colors[i], alpha=0.9, zorder=3)
                # 预测值：虚线，稍细
                ax.plot(time_steps, pred, label='Predicted', linewidth=2.5, 
                       linestyle='--', color=feature_colors[i], alpha=0.7, zorder=2)
                
                # 计算误差指标
                mae = mean_absolute_error(target, pred)
                rmse = np.sqrt(mean_squared_error(target, pred))
                
                # 高误差点检测和标注（FaST风格）
                epsilon = 1e-8
                abs_error = np.abs(pred - target)
                relative_error = abs_error / (np.abs(target) + epsilon)
                
                # 找出相对误差>30%的点
                error_threshold = 0.3
                high_error_mask = relative_error > error_threshold
                non_zero_mask = np.abs(target) > epsilon
                high_error_mask = high_error_mask & non_zero_mask
                
                # 标注最多5个最大误差点
                high_error_indices = np.where(high_error_mask)[0]
                if len(high_error_indices) > 0:
                    error_values = relative_error[high_error_indices]
                    sorted_indices = high_error_indices[np.argsort(-error_values)]
                    annotated_indices = sorted_indices[:5]  # 最多标注5个点
                    
                    for idx in annotated_indices:
                        error_pct = relative_error[idx] * 100
                        # 红色圆圈标记真实值
                        ax.plot(idx, target[idx], 'o', color='red', markersize=10, 
                               markeredgecolor='darkred', markeredgewidth=2, zorder=5)
                        # 橙色圆圈标记预测值
                        ax.plot(idx, pred[idx], 'o', color='orange', markersize=8,
                               markeredgecolor='darkorange', markeredgewidth=1.5, zorder=5)
                        
                        # 添加误差标注
                        ax.annotate(f'Error:{error_pct:.1f}%\nStep {idx}',
                                   xy=(idx, target[idx]),
                                   xytext=(0, 20),
                                   textcoords='offset points',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                                            alpha=0.7, edgecolor='orange'),
                                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                                   fontsize=8, ha='center')
                
                # 子图标题和标签
                ax.set_title(f'{feat_name}\nMAE: {mae:.3f} | RMSE: {rmse:.3f}', 
                           fontsize=12, fontweight='bold', pad=10)
                ax.set_xlabel('Time Steps (Hours)', fontsize=11)
                ax.set_ylabel('Traffic Flow (veh/h)', fontsize=11)
                
                # 图例
                ax.legend(loc='best', fontsize=10, framealpha=0.9)
                
                # 网格线（FaST风格）
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_axisbelow(True)
            
            # 保存图表
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            
            station_label = f'{station_code}' if station_code else f'station{station_idx:03d}'
            save_path = os.path.join(save_dir, 
                                    f'sample{sample_idx}_{station_label}.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            print(f"  ✓ Saved: {os.path.basename(save_path)}")
            plt.close()
    
    print(f"\n✓ All plots saved to {save_dir}/")
    print("="*70)

def save_predictions_to_csv(predictions, targets, station_mapping, 
                           save_dir='results', mode='test'):
    """
    保存所有站点的预测结果为CSV（仿照FaST项目）
    """
    print(f"\n💾 Saving predictions to CSV...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    n_features_per_station = 4
    total_features = predictions.shape[2]
    n_stations = total_features // n_features_per_station
    
    feature_names = [
        'Passenger_Car_Up',
        'Passenger_Car_Down', 
        'Non_Passenger_Car_Up',
        'Non_Passenger_Car_Down'
    ]
    
    rows = []
    total_samples = predictions.shape[0]
    
    for sample_idx in range(total_samples):
        # 生成时间戳（基于样本索引）
        time_str = f"Sample_{sample_idx}"
        
        for station_idx in range(n_stations):
            station_info = station_mapping.get(str(station_idx), {})
            station_name = station_info.get('station_name', f'Station_{station_idx:03d}')
            station_code = station_info.get('station_code', f'{station_idx:03d}')
            
            base_feat_idx = station_idx * n_features_per_station
            
            for feat_idx, feat_name in enumerate(feature_names):
                # 取第一个预测步长的结果
                true_val = round(targets[sample_idx, 0, base_feat_idx + feat_idx], 2)
                pred_val = round(predictions[sample_idx, 0, base_feat_idx + feat_idx], 2)
                
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
    save_path = os.path.join(save_dir, f'{mode}_predictions.csv')
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✓ Predictions CSV saved to {save_path}")
    print(f"  Total records: {len(df)}")
    print(f"  Stations: {n_stations}")
    print(f"  Samples: {total_samples}")
    
    return save_path

def main():
    """
    主函数 - 支持推理不同数据集
    
    Usage:
        python inference.py                  # 默认推理测试集
        python inference.py --mode train     # 推理训练集
        python inference.py --mode val       # 推理验证集
        python inference.py --mode test      # 推理测试集
    """
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MOMENT Inference Script')
    parser.add_argument('--mode', type=str, default='test', 
                       choices=['train', 'val', 'test'],
                       help='Dataset mode: train, val, or test (default: test)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference (default: 32)')
    parser.add_argument('--n-samples', type=int, default=3,
                       help='Number of samples to visualize (default: 3)')
    parser.add_argument('--n-stations', type=int, default=3,
                       help='Number of stations to visualize (default: 3)')
    
    args = parser.parse_args()
    
    # Configuration - MUST match preprocess_data.py and train_moment.py settings
    PRED_LEN = 8   # Prediction horizon (hours) - must match training
    BATCH_SIZE = args.batch_size
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODE = args.mode
    
    print("="*70)
    print(f"MOMENT Traffic Flow Forecasting - Inference & Evaluation ({MODE.upper()})")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Mode: {MODE.upper()}")
    print(f"  Device: {DEVICE}")
    print(f"  Prediction horizon: {PRED_LEN}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    # Load normalization parameters
    with open('normalization_params.json', 'r') as f:
        norm_params = json.load(f)
    
    # Load station mapping
    station_mapping = {}
    if os.path.exists('dataset/station_mapping.json'):
        with open('dataset/station_mapping.json', 'r', encoding='utf-8') as f:
            station_mapping = json.load(f)
        print(f"  Station mapping loaded: {len(station_mapping)} stations")
    else:
        print("  ⚠️  Warning: station_mapping.json not found. Using default station names.")
    
    # Load data based on mode
    print(f"\nLoading {MODE.upper()} data...")
    dataset_path = f'moment_data/{MODE}_dataset.npz'
    
    if not os.path.exists(dataset_path):
        print(f"\n❌ Error: Dataset file not found at {dataset_path}")
        print(f"Please run preprocessing first: python preprocess_data.py")
        return
    
    data = np.load(dataset_path)
    input_data = data['input']
    target_data = data['target']
    
    n_features = input_data.shape[2]
    n_stations = n_features // 4
    
    print(f"  {MODE.capitalize()} samples: {len(input_data)}")
    print(f"  Total features: {n_features}")
    print(f"  Number of stations: {n_stations}")
    
    # Create dataset and dataloader
    test_dataset = TrafficFlowDataset(input_data, target_data)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Load model
    checkpoint_path = 'checkpoints/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using: python train_moment.py")
        return
    
    model, checkpoint_info = load_model(checkpoint_path, n_features=n_features, 
                                        forecast_horizon=PRED_LEN, seq_len=8)
    model = model.to(DEVICE)
    
    # Evaluate
    predictions, targets, metrics = evaluate_model(model, test_loader, DEVICE, 
                                                  norm_params, station_mapping)
    
    # Save predictions and metrics
    if predictions is not None:
        print("\nSaving results...")
        
        # Save predictions (NPZ format)
        np.savez_compressed(
            f'results/{MODE}_predictions.npz',
            predictions=predictions,
            targets=targets if targets is not None else np.array([])
        )
        print(f"  ✓ Predictions saved to results/{MODE}_predictions.npz")
        
        # Save metrics (JSON format)
        if metrics:
            with open(f'results/{MODE}_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Metrics saved to results/{MODE}_metrics.json")
        
        # Save predictions to CSV (FaST style)
        csv_path = save_predictions_to_csv(
            predictions, targets, station_mapping,
            save_dir='results',
            mode=MODE
        )
        
        # Generate comprehensive visualizations (FaST style)
        plot_comprehensive_predictions(
            predictions, targets, station_mapping,
            save_dir='results',
            n_samples=args.n_samples,
            n_stations=min(args.n_stations, n_stations)
        )
        
        print("\n" + "="*70)
        print("✓ Inference and evaluation completed successfully!")
        print("="*70)
        print(f"\nOutput files ({MODE} set):")
        print(f"  - results/{MODE}_predictions.npz      : Prediction results (NPZ)")
        print(f"  - results/{MODE}_predictions.csv      : All predictions (CSV, FaST style)")
        print(f"  - results/{MODE}_metrics.json         : Evaluation metrics")
        print(f"  - results/sample*_station*.png        : Visualization plots (FaST style)")
        print("="*70)

if __name__ == "__main__":
    main()