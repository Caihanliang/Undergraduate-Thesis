#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TimesFM 训练集预测与可视化 - 仅第1步预测
功能：
1. 加载训练集数据
2. 使用TimesFM模型进行预测
3. 只保存第1步预测（时间、真实值、预测值、特征、站点）
4. 自动推算时间（从2023-09-01 00:00:00开始，时间粒度1小时）
5. 生成可视化图表（仅第1步）


python visualize_train_predictions.py \
    --model_id google/timesfm-2.5-200m-transformers \
    --adapter_path checkpoints/traffic-lora-32x8 \
    --data_dir dataset/preprocessed
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入配置
import config

# 设置 HuggingFace 镜像加速
os.environ['HF_ENDPOINT'] = config.HF_ENDPOINT

# 特征名称映射
FEATURE_NAMES = {
    0: "Passenger Car Upstream",
    1: "Passenger Car Downstream",
    2: "Non-Passenger Car Upstream",
    3: "Non-Passenger Car Downstream"
}

# 配置
CONFIG = {
    'dataset_start': '2023-09-01 00:00:00',  # 数据集起始时间
    'time_granularity': 1,  # 时间粒度：1小时
    'prediction_step': 1,  # 只使用第1步预测
    'output_dir': 'visualization-results/train_step1',
    'figsize': (18, 6),
    'dpi': 150,
}


def load_model_and_adapter(model_id: str, adapter_path: str, device: str):
    """加载基础模型和 LoRA 适配器"""
    from peft import PeftModel
    from transformers import TimesFm2_5ModelForPrediction
    
    print(f"加载基础模型: {model_id}")
    base_model = TimesFm2_5ModelForPrediction.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    base_model.eval()
    
    print(f"加载 LoRA 适配器: {adapter_path}")
    ft_model = PeftModel.from_pretrained(base_model, adapter_path)
    ft_model.eval()
    
    return ft_model


def load_train_data(data_dir: str):
    """加载训练数据"""
    print(f"\n从 {data_dir} 加载训练数据...")
    
    # 加载训练序列数据
    train_file = os.path.join(data_dir, 'train_series.npz')
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"训练数据文件不存在: {train_file}")
    
    train_data = np.load(train_file, allow_pickle=True)
    train_series = train_data['data']
    
    # 加载元数据
    metadata_file = os.path.join(data_dir, 'metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    print(f"✓ 加载了 {len(train_series)} 个训练序列")
    print(f"特征名称: {metadata.get('feature_names', list(FEATURE_NAMES.values()))}")
    
    return train_series, metadata


@torch.no_grad()
def predict_with_timesfm(model, input_data: np.ndarray, context_len: int = 32, horizon_len: int = 8):
    """
    使用 TimesFM 进行预测
    
    Args:
        model: TimesFM 模型
        input_data: 输入数据，形状 (seq_len,)
        context_len: 上下文长度
        horizon_len: 预测长度
        
    Returns:
        predictions: 预测值，形状 (horizon_len,)
    """
    try:
        # 确保输入长度正确
        if len(input_data) < context_len:
            # 如果数据不足，用最后的数据填充
            padded_input = np.pad(input_data, (context_len - len(input_data), 0), mode='edge')
        else:
            padded_input = input_data[-context_len:]
        
        # 转换为 tensor - TimesFM 期望的形状是 (batch_size, seq_len)
        input_tensor = torch.tensor(padded_input, dtype=torch.float32).unsqueeze(0)  # (1, context_len)
        input_tensor = input_tensor.to(model.device)
        
        # TimesFM 2.5 推理
        outputs = model(
            past_values=input_tensor,
            prediction_length=horizon_len,
        )
        
        # 获取预测值
        raw_predictions = outputs.mean_predictions.float().cpu().numpy()[0]  # (128,)
        
        # 截取前 horizon_len 个作为最终预测
        predictions = raw_predictions[:horizon_len]  # (horizon_len,)
        
        return predictions
        
    except Exception as e:
        print(f"⚠️  预测失败: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros(horizon_len)


@torch.no_grad()
def predict_with_timesfm_batch(model, input_data: np.ndarray, context_len: int = 32, horizon_len: int = 8):
    """
    使用 TimesFM 进行批量预测（优化版）
    
    Args:
        model: TimesFM 模型
        input_data: 输入数据，形状 (batch_size, seq_len) 或 (seq_len,)
        context_len: 上下文长度
        horizon_len: 预测长度
        
    Returns:
        predictions: 预测值，形状 (batch_size, horizon_len) 或 (horizon_len,)
    """
    try:
        # 处理单样本情况
        if input_data.ndim == 1:
            input_data = input_data[np.newaxis, :]  # (1, seq_len)
        
        batch_size = input_data.shape[0]
        
        # 确保输入长度正确
        if input_data.shape[1] < context_len:
            # 如果数据不足，用最后的数据填充
            pad_width = context_len - input_data.shape[1]
            input_data = np.pad(input_data, ((0, 0), (pad_width, 0)), mode='edge')
        else:
            input_data = input_data[:, -context_len:]
        
        # 转换为 tensor - TimesFM 期望的形状是 (batch_size, seq_len)
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        input_tensor = input_tensor.to(model.device)
        
        # TimesFM 2.5 推理
        outputs = model(
            past_values=input_tensor,
            prediction_length=horizon_len,
        )
        
        # 获取预测值
        raw_predictions = outputs.mean_predictions.float().cpu().numpy()  # (batch_size, 128,)
        
        # 截取前 horizon_len 个作为最终预测
        predictions = raw_predictions[:, :horizon_len]  # (batch_size, horizon_len,)
        
        # 如果是单样本，返回一维数组
        if batch_size == 1:
            return predictions[0]
        
        return predictions
        
    except Exception as e:
        print(f"⚠️  预测失败: {e}")
        import traceback
        traceback.print_exc()
        if input_data.ndim == 1:
            return np.zeros(horizon_len)
        else:
            return np.zeros((input_data.shape[0], horizon_len))


def predict_and_save_step1(args):
    """
    对训练集进行预测，只保存第1步预测结果
    
    正确理解：
    - 每个站点有4个特征，每个特征是独立的时间序列
    - 对于长度为N的序列，可以生成 (N - context_len) 个预测样本
    - 每个样本的第1步预测对应的时间 = 起始时间 + (context_len + sample_offset) * 1小时
    - 同一时间点的所有站点和特征的预测，时间戳相同
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "figures" / "all_stations").mkdir(exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)
    
    # 加载模型
    print("\n" + "="*80)
    print("步骤 1/3: 加载模型")
    print("="*80)
    model = load_model_and_adapter(args.model_id, args.adapter_path, device)
    
    # 加载训练数据
    print("\n" + "="*80)
    print("步骤 2/3: 预测训练集（仅第1步）")
    print("="*80)
    train_series, metadata = load_train_data(args.data_dir)
    
    context_len = metadata.get('context_len', 32)
    horizon_len = metadata.get('horizon_len', 8)
    
    print(f"\n配置: {context_len}输入{horizon_len}输出")
    print(f"训练序列数: {len(train_series)}")
    print(f"输出目录: {output_dir}")
    print("="*80)
    
    # 数据集起始时间
    dataset_start = pd.Timestamp(CONFIG['dataset_start'])
    time_granularity = timedelta(hours=CONFIG['time_granularity'])
    
    # 解析序列信息：确定每个序列属于哪个站点和哪个特征
    # 序列组织方式：每4个连续序列对应一个站点（4个特征）
    # 顺序：[station_0_feat_0, station_0_feat_1, station_0_feat_2, station_0_feat_3, 
    #       station_1_feat_0, station_1_feat_1, ...]
    num_stations = 157  # 总站点数
    feature_names_list = ['passenger_car_up', 'passenger_car_down', 
                         'non_passenger_car_up', 'non_passenger_car_down']
    
    print(f"\n🔮 开始预测训练集（仅第1步）...")
    print(f"预期站点数: {num_stations}")
    print(f"特征列表: {feature_names_list}")
    
    # ✅ 动态确定预测范围：根据实际序列长度
    # 找到最短序列的长度，确保所有序列都能预测
    min_seq_length = min(len(series) for series in train_series)
    max_seq_length = max(len(series) for series in train_series)
    avg_seq_length = np.mean([len(series) for series in train_series])
    
    print(f"\n📊 序列长度统计:")
    print(f"  最短序列: {min_seq_length} 小时")
    print(f"  最长序列: {max_seq_length} 小时")
    print(f"  平均长度: {avg_seq_length:.1f} 小时")
    
    # 使用最短序列长度作为预测上限（确保所有序列都能预测）
    TRAIN_HOURS = min_seq_length  # 动态调整
    PREDICT_START_HOUR = context_len  # 从第32小时开始（需要前32小时作为输入）
    PREDICT_END_HOUR = TRAIN_HOURS  # 到最短序列的末尾
    
    print(f"\n⏰ 训练集时间范围:")
    print(f"  起始: {dataset_start}")
    print(f"  结束: {dataset_start + timedelta(hours=TRAIN_HOURS)}")
    print(f"  总时长: {TRAIN_HOURS} 小时")
    print(f"\n🎯 预测范围:")
    print(f"  从第 {PREDICT_START_HOUR} 小时开始 (需要前{context_len}小时作为输入)")
    print(f"  到第 {PREDICT_END_HOUR} 小时结束")
    print(f"  预计预测 {PREDICT_END_HOUR - PREDICT_START_HOUR} 个时间点\n")
    
    # 存储所有预测结果
    all_predictions_step1 = []
    all_targets_step1 = []
    all_station_ids = []
    all_feature_indices = []
    all_prediction_times = []
    
    success_count = 0
    failed_count = 0
    
    # ✅ 批量预测策略：收集所有样本后批量推理
    BATCH_SIZE = 128  # 每批处理128个样本
    
    print(f"\n⚡ 批量预测策略: 每批 {BATCH_SIZE} 个样本\n")
    
    # 收集所有需要预测的样本
    all_input_data = []
    all_target_data = []
    all_meta_info = []  # 存储 (station_idx, feat_idx, pred_hour)
    
    for seq_idx, series in enumerate(train_series):
        # 解析站点ID和特征索引
        station_idx = seq_idx // 4
        feat_idx = seq_idx % 4
        
        # ✅ 使用当前序列的实际长度
        seq_length = len(series)
        
        if seq_length < context_len + horizon_len:
            failed_count += 1
            continue
        
        # 从训练集范围内采样：每隔一段时间预测一次
        SAMPLE_INTERVAL = 24  # 每24小时预测一次，覆盖整个训练集
        
        # ✅ 收集所有样本
        for pred_hour in range(PREDICT_START_HOUR, seq_length, SAMPLE_INTERVAL):
            # 输入：前context_len小时
            input_start = pred_hour - context_len
            input_end = pred_hour
            input_data = series[input_start:input_end]
            
            # 目标：预测时间点及其后续horizon_len小时
            target_start = pred_hour
            target_end = pred_hour + horizon_len
            
            if target_end > seq_length:
                break  # 超出序列长度
            
            target_data = series[target_start:target_end]
            
            all_input_data.append(input_data)
            all_target_data.append(target_data)
            all_meta_info.append((station_idx, feat_idx, pred_hour))
    
    print(f"✓ 样本收集完成: 共 {len(all_input_data)} 个样本")
    print(f"  开始批量预测...\n")
    
    # ✅ 批量预测
    total_samples = len(all_input_data)
    all_predictions_step1 = []
    all_targets_step1 = []
    all_station_ids = []
    all_feature_indices = []
    all_prediction_times = []
    
    for batch_start in range(0, total_samples, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_samples)
        batch_inputs = np.array(all_input_data[batch_start:batch_end])  # (batch_size, context_len)
        batch_targets = np.array(all_target_data[batch_start:batch_end])  # (batch_size, horizon_len)
        
        # 批量预测
        batch_preds = predict_with_timesfm_batch(model, batch_inputs, context_len, horizon_len)
        
        # 处理批量结果
        for i in range(batch_end - batch_start):
            pred = batch_preds[i]
            target = batch_targets[i]
            station_idx, feat_idx, pred_hour = all_meta_info[batch_start + i]
            
            pred_step1 = float(pred[0])
            target_step1 = float(target[0])
            
            # 计算预测时间
            pred_time = dataset_start + (pred_hour * time_granularity)
            
            all_predictions_step1.append(pred_step1)
            all_targets_step1.append(target_step1)
            all_station_ids.append(station_idx)
            all_feature_indices.append(feat_idx)
            all_prediction_times.append(pred_time)
            
            success_count += 1
        
        # 进度显示
        if (batch_end % 1000 == 0) or (batch_end == total_samples):
            print(f"  进度: {batch_end}/{total_samples} ({batch_end*100//total_samples}%)")
    
    failed_count = total_samples - success_count
    
    if not all_predictions_step1:
        raise ValueError("所有预测都失败了！请检查模型和数据。")
    
    predictions_step1 = np.array(all_predictions_step1)
    targets_step1 = np.array(all_targets_step1)
    
    # ✅ 修复：确保是numpy数组类型
    predictions_step1 = np.asarray(predictions_step1, dtype=np.float64)
    targets_step1 = np.asarray(targets_step1, dtype=np.float64)
    
    print(f"\n✓ 预测完成:")
    print(f"  总样本数: {len(predictions_step1)}")
    print(f"  采样策略: 每 {SAMPLE_INTERVAL} 小时采样1次")
    print(f"  成功率: {success_count}/{success_count+failed_count}")
    print(f"  时间范围: {all_prediction_times[0]} 到 {all_prediction_times[-1]}")
    print(f"  唯一时间点数量: {len(set(all_prediction_times))}")
    
    # ============================================================
    # 步骤 3: 保存预测结果为 CSV
    # ============================================================
    print("\n" + "="*80)
    print("步骤 3/3: 保存结果和生成可视化")
    print("="*80)
    print("\n💾 保存第1步预测结果为 CSV...")
    
    rows = []
    for idx in range(len(predictions_step1)):
        station_idx = all_station_ids[idx]
        feat_idx = all_feature_indices[idx]
        
        rows.append({
            'station_id': station_idx,
            'station_name': f'Station_{station_idx:03d}',
            'feature': FEATURE_NAMES[feat_idx],
            'feature_idx': feat_idx,
            'prediction_step': 1,
            'prediction_time': all_prediction_times[idx].strftime('%Y-%m-%d %H:%M:%S'),
            'true_value': round(float(targets_step1[idx]), 2),
            'pred_value': round(float(predictions_step1[idx]), 2),
            'absolute_error': round(float(abs(targets_step1[idx] - predictions_step1[idx])), 2)
        })
    
    df_results = pd.DataFrame(rows)
    csv_path = output_dir / "predictions_step1.csv"
    df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"✓ CSV已保存: {csv_path}")
    print(f"  总记录数: {len(df_results)}")
    print(f"  站点数: {df_results['station_id'].nunique()}")
    print(f"  特征数: {df_results['feature'].nunique()}")
    print(f"  时间范围: {df_results['prediction_time'].min()} 到 {df_results['prediction_time'].max()}")
    print(f"  唯一时间点数量: {df_results['prediction_time'].nunique()}")
    
    # 验证：同一时间点应该有157*4=628条记录
    first_time = df_results['prediction_time'].iloc[0]
    same_time_count = len(df_results[df_results['prediction_time'] == first_time])
    print(f"  验证 - 时间点 {first_time} 的记录数: {same_time_count} (预期: 628)")
    
    # ============================================================
    # 生成可视化图表（仅第1步）
    # ============================================================
    print("\n📈 生成可视化图表...")
    
    # 按站点和特征分组
    stations_data = {}
    for idx in range(len(predictions_step1)):
        station_idx = all_station_ids[idx]
        feat_idx = all_feature_indices[idx]
        
        if station_idx not in stations_data:
            stations_data[station_idx] = {f: {'times': [], 'preds': [], 'targets': []} for f in range(4)}
        
        stations_data[station_idx][feat_idx]['times'].append(all_prediction_times[idx])
        stations_data[station_idx][feat_idx]['preds'].append(float(predictions_step1[idx]))
        stations_data[station_idx][feat_idx]['targets'].append(float(targets_step1[idx]))
    
    total_stations = len(stations_data)
    print(f"总站点数: {total_stations}")
    
    all_metrics = {}
    
    for feat_idx in range(4):
        feature_name = FEATURE_NAMES[feat_idx]
        print(f"\n处理特征: {feature_name}")
        
        # 创建特征子目录
        feat_dir = output_dir / "figures" / "all_stations" / feature_name.replace(' ', '_')
        feat_dir.mkdir(parents=True, exist_ok=True)
        
        feature_metrics = {}
        
        for station_idx in sorted(stations_data.keys()):
            station_data = stations_data[station_idx][feat_idx]
            
            if not station_data['times']:
                continue
            
            # 按时间排序
            sorted_indices = np.argsort(station_data['times'])
            times = [station_data['times'][i] for i in sorted_indices]
            preds = [station_data['preds'][i] for i in sorted_indices]
            targets = [station_data['targets'][i] for i in sorted_indices]
            
            # 计算评估指标
            targets = np.array(station_data['targets'], dtype=np.float64)  # ✅ 修复：确保是numpy数组
            preds = np.array(station_data['preds'], dtype=np.float64)  # ✅ 修复：确保是numpy数组
            
            mae = mean_absolute_error(targets, preds)
            rmse = np.sqrt(mean_squared_error(targets, preds))
            
            # 计算SMAPE - ✅ 修复：确保是numpy数组运算
            denominator = (np.abs(targets) + np.abs(preds)) / 2
            denominator = np.maximum(denominator, 1e-8)
            smape = np.mean(np.abs(targets - preds) / denominator) * 100
            
            feature_metrics[f'Station_{station_idx:03d}'] = {
                'MAE': mae,
                'RMSE': rmse,
                'SMAPE': smape,
                'num_points': len(times)
            }
            
            # 创建图表
            fig, ax = plt.subplots(figsize=CONFIG['figsize'])
            
            ax.plot(times, targets, label='True Traffic', linewidth=1.5, 
                   color='#1f77b4', alpha=0.8, marker='o', markersize=2)
            ax.plot(times, preds, label='Pred Traffic (Step 1)', linewidth=1.5, 
                   color='#ff4b5c', linestyle='--', alpha=0.8, marker='x', markersize=2)
            
            ax.set_title(f"Station_{station_idx:03d} - {feature_name} (Training Set - Step 1)\n"
                        f"MAE={mae:.2f}, RMSE={rmse:.2f}, SMAPE={smape:.1f}%", 
                        fontsize=13, fontweight='bold', pad=15)
            ax.set_xlabel('Prediction Time', fontsize=12)
            ax.set_ylabel('Traffic Flow (veh/h)', fontsize=12)
            
            # 格式化X轴
            time_span = (times[-1] - times[0]).days if len(times) > 1 else 0
            
            if time_span <= 3:
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            elif time_span <= 14:
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            else:
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)
            
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=11)
            
            # 添加数据点数量注释
            ax.text(0.02, 0.98, f"Data Points: {len(times)} | Step: 1", 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            plt.tight_layout()
            
            # 保存
            save_path = feat_dir / f"station_{station_idx:03d}.png"
            plt.savefig(save_path, dpi=CONFIG['dpi'], bbox_inches='tight')
            plt.close()
        
        all_metrics[feature_name] = feature_metrics
        
        if feature_metrics:
            avg_mae = np.mean([m['MAE'] for m in feature_metrics.values()])
            avg_rmse = np.mean([m['RMSE'] for m in feature_metrics.values()])
            avg_smape = np.mean([m['SMAPE'] for m in feature_metrics.values()])
            print(f"  特征 {feature_name} - 平均 MAE: {avg_mae:.3f}, 平均 RMSE: {avg_rmse:.3f}, 平均 SMAPE: {avg_smape:.1f}%")
    
    # 保存评估指标
    metrics_path = output_dir / "metrics.json"
    metrics_summary = {}
    
    for feature_name, stations_metrics in all_metrics.items():
        if stations_metrics:
            mae_list = [m['MAE'] for m in stations_metrics.values()]
            rmse_list = [m['RMSE'] for m in stations_metrics.values()]
            smape_list = [m['SMAPE'] for m in stations_metrics.values()]
            
            metrics_summary[feature_name] = {
                'prediction_step': 1,
                'avg_MAE': round(float(np.mean(mae_list)), 3),
                'avg_RMSE': round(float(np.mean(rmse_list)), 3),
                'avg_SMAPE': round(float(np.mean(smape_list)), 1),
                'std_MAE': round(float(np.std(mae_list)), 3),
                'std_RMSE': round(float(np.std(rmse_list)), 3),
                'min_MAE': round(float(np.min(mae_list)), 3),
                'max_MAE': round(float(np.max(mae_list)), 3),
                'num_stations': len(stations_metrics)
            }
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 评估指标已保存: {metrics_path}")
    
    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "="*80)
    print("🎉 全部完成！")
    print("="*80)
    print(f"输出目录: {output_dir}")
    print(f"  - 预测结果: {csv_path}")
    print(f"  - 站点图: {output_dir / 'figures' / 'all_stations'}/[特征名]/station_XXX.png")
    print(f"  - 评估指标: {metrics_path}")
    print(f"\n预测步长: 第1步")
    print(f"站点数量: {total_stations}")
    print(f"特征数量: 4")
    print(f"时间范围: {all_prediction_times[0]} 到 {all_prediction_times[-1]}")
    print(f"唯一时间点数量: {len(set(all_prediction_times))}")
    print("="*80)


def parse_args():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='TimesFM Training Set Prediction and Visualization (Step 1 only)'
    )
    
    parser.add_argument(
        '--model_id',
        default=config.MODEL_ID,
        help='HuggingFace model ID'
    )
    parser.add_argument(
        '--adapter_path',
        # default=config.CHECKPOINT_DIR,
        default="/home/user/Downloads/cai/timesfm-master/checkpoints/traffic-lora-32x8",

        help='LoRA adapter path'
    )
    parser.add_argument(
        '--data_dir',
        default=config.PREPROCESSED_DIR,
        help='Preprocessed data directory'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    predict_and_save_step1(args)


if __name__ == '__main__':
    main()
