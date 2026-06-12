#!/usr/bin/env python3
"""
TimesFM 交通流量预测和可视化脚本（32输入8输出）

使用微调后的模型进行交通流量预测，并生成完整的可视化结果
参考 FaST-MV 项目风格，为所有站点生成 4 特征预测结果图

时序配置：32输入8输出（使用过去32小时预测未来8小时）
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
    0: "小客车上行",
    1: "小客车下行", 
    2: "非小客车上行",
    3: "非小客车下行"
}

FEATURE_NAMES_EN = {
    0: "Passenger Car Upstream",
    1: "Passenger Car Downstream",
    2: "Non-Passenger Car Upstream", 
    3: "Non-Passenger Car Downstream"
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


def load_test_data(data_dir: str):
    """加载测试数据"""
    print(f"\n从 {data_dir} 加载测试数据...")
    
    # 加载序列数据
    test_data = np.load(os.path.join(data_dir, 'test_series.npz'), allow_pickle=True)
    test_series = test_data['data']
    
    # 加载元数据
    with open(os.path.join(data_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"✓ 加载了 {len(test_series)} 个测试序列")
    print(f"特征名称: {metadata.get('feature_names', [])}")
    
    return test_series, metadata


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
        
        # ✅ 关键修复：mean_predictions 形状为 (batch_size, 128)
        # 我们只需要前 horizon_len 个时间步的预测
        # 根据官方示例：outputs.mean_predictions[0, :horizon_len]
        raw_predictions = outputs.mean_predictions.float().cpu().numpy()[0]  # (128,)
        
        # 截取前 horizon_len 个作为最终预测
        predictions = raw_predictions[:horizon_len]  # (horizon_len,)
        
        return predictions
        
    except Exception as e:
        print(f"⚠️  预测失败: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros(horizon_len)


def generate_prediction_timestamps(sample_idx: int, context_len: int = 32, horizon_len: int = 8, 
                                   dataset_split: str = 'train'):
    """
    根据样本索引生成预测时间戳
    
    Args:
        sample_idx: 样本索引
        context_len: 上下文窗口长度
        horizon_len: 预测窗口长度
        dataset_split: 数据集类型 ('train', 'val', 'test')
        
    Returns:
        start_time: 预测开始时间 (datetime)
        timestamps: 预测时间戳列表 (list of datetime)
    """
    # 数据集起始时间
    dataset_start = pd.Timestamp('2023-09-01 00:00:00')
    
    # 总时长：9月(30天) + 10月(31天) = 61天 = 1464小时
    total_hours = 61 * 24  # 1464小时
    
    # 根据数据集类型计算起始偏移
    if dataset_split == 'train':
        # 训练集：前70%
        split_start_hour = 0
        split_end_hour = int(total_hours * 0.7)  # 1025小时
    elif dataset_split == 'val':
        # 验证集：中间15%
        split_start_hour = int(total_hours * 0.7)  # 1025小时
        split_end_hour = int(total_hours * 0.85)   # 1244小时
    else:  # test
        # 测试集：后15%
        split_start_hour = int(total_hours * 0.85)  # 1244小时
        split_end_hour = total_hours                  # 1464小时
    
    # 每个站点4个特征，计算站点索引和特征索引
    station_idx = sample_idx // 4
    feat_idx = sample_idx % 4
    
    # 假设测试/训练数据均匀采样，每个样本对应一个预测窗口
    # 简化处理：直接根据样本索引计算在整个数据集中的位置
    samples_per_feature = 157  # 站点数
    sample_in_split = sample_idx % samples_per_feature
    
    # 在数据集分割中的偏移（均匀分布）
    split_duration = split_end_hour - split_start_hour - context_len - horizon_len
    offset_in_split = int((sample_in_split / samples_per_feature) * split_duration)
    
    # 预测开始时间的小时偏移
    pred_start_hour = split_start_hour + offset_in_split
    
    # 计算具体时间
    start_time = dataset_start + pd.Timedelta(hours=pred_start_hour)
    
    # 生成8个预测时间戳
    timestamps = [start_time + pd.Timedelta(hours=h) for h in range(horizon_len)]
    
    return start_time, timestamps


def batch_predict_and_visualize(args):
    """批量预测和可视化"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "figures" / "all_stations").mkdir(exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    
    # 加载模型
    model = load_model_and_adapter(args.model_id, args.adapter_path, device)
    
    # 加载数据
    test_series, metadata = load_test_data(args.data_dir)
    
    # 获取配置参数
    context_len = metadata.get('context_len', 32)
    horizon_len = metadata.get('horizon_len', 8)
    
    print(f"\n{'='*80}")
    print(f"开始预测和可视化")
    print(f"{'='*80}")
    print(f"配置: {context_len}输入{horizon_len}输出")
    print(f"测试序列数: {len(test_series)}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}\n")
    
    # ============================================================
    # 步骤 1: 对所有测试序列进行预测
    # ============================================================
    print("🔮 步骤 1/4: 生成预测结果...")
    
    all_predictions = []
    all_targets = []
    station_ids = []
    feature_indices = []
    
    failed_count = 0
    success_count = 0
    
    for idx, series in enumerate(test_series):
        if len(series) < context_len + horizon_len:
            continue
        
        # 提取输入和目标
        input_data = series[:context_len]
        target_data = series[context_len:context_len + horizon_len]
        
        # 预测
        pred = predict_with_timesfm(model, input_data, context_len, horizon_len)
        
        # 严格验证预测结果的形状
        if pred.shape != (horizon_len,):
            print(f"⚠️  样本 {idx}: 预测形状 {pred.shape} != 预期 ({horizon_len},)，跳过")
            failed_count += 1
            continue
        
        # 验证目标数据的形状
        if target_data.shape != (horizon_len,):
            print(f"⚠️  样本 {idx}: 目标形状 {target_data.shape} != 预期 ({horizon_len},)，跳过")
            failed_count += 1
            continue
        
        all_predictions.append(pred)
        all_targets.append(target_data)
        
        # 解析站点ID和特征索引（从元数据或序列索引推断）
        # 这里简化处理，实际应该从metadata中获取
        station_idx = idx // 4  # 假设每4个序列对应一个站点（4个特征）
        feat_idx = idx % 4
        station_ids.append(station_idx)
        feature_indices.append(feat_idx)
        
        success_count += 1
        
        if (idx + 1) % 100 == 0:
            print(f"  已处理 {idx + 1}/{len(test_series)} 个序列 (成功: {success_count}, 失败: {failed_count})")
    
    if not all_predictions:
        raise ValueError("所有预测都失败了！请检查模型和数据。")
    
    all_predictions = np.array(all_predictions)  # (num_samples, horizon_len)
    all_targets = np.array(all_targets)          # (num_samples, horizon_len)
    
    print(f"✓ 预测完成: {len(all_predictions)} 个样本 (成功率: {success_count}/{success_count+failed_count})")
    
    # ============================================================
    # 步骤 2: 计算评估指标
    # ============================================================
    print("\n📊 步骤 2/4: 计算评估指标...")
    
    metrics_by_feature = {}
    
    for feat_idx in range(4):
        # 找到该特征的所有样本
        mask = [i for i, f in enumerate(feature_indices) if f == feat_idx]
        
        if not mask:
            continue
        
        preds_feat = all_predictions[mask]
        targets_feat = all_targets[mask]
        
        # 计算每个预测步长的指标
        mae_per_step = []
        rmse_per_step = []
        mape_per_step = []
        
        for step in range(horizon_len):
            y_pred = preds_feat[:, step]
            y_true = targets_feat[:, step]
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # ✅ 修复MAPE计算：使用对称MAPE(SMAPE)避免真实值接近0时的数值爆炸
            # 传统MAPE: |y_true - y_pred| / |y_true| 会在y_true接近0时爆炸
            # SMAPE: |y_true - y_pred| / ((|y_true| + |y_pred|) / 2)
            denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
            # 避免除以0
            denominator = np.maximum(denominator, 1e-8)
            mape = np.mean(np.abs(y_true - y_pred) / denominator) * 100
            
            mae_per_step.append(mae)
            rmse_per_step.append(rmse)
            mape_per_step.append(mape)
        
        # 整体指标
        overall_mae = np.mean(mae_per_step)
        overall_rmse = np.mean(rmse_per_step)
        overall_mape = np.mean(mape_per_step)
        
        metrics_by_feature[feat_idx] = {
            'per_step': {
                f'step_{s+1}': {
                    'MAE': round(float(mae_per_step[s]), 4),
                    'RMSE': round(float(rmse_per_step[s]), 4),
                    'MAPE': round(float(mape_per_step[s]), 2)
                }
                for s in range(horizon_len)
            },
            'overall': {
                'MAE': round(float(overall_mae), 4),
                'RMSE': round(float(overall_rmse), 4),
                'MAPE': round(float(overall_mape), 2)
            }
        }
        
        print(f"  特征 {feat_idx} ({FEATURE_NAMES_EN[feat_idx]}):")
        print(f"    MAE: {overall_mae:.4f}, RMSE: {overall_rmse:.4f}, MAPE: {overall_mape:.2f}%")
    
    # 保存指标
    metrics_file = output_dir / "metrics" / "evaluation_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump({
            'context_len': context_len,
            'horizon_len': horizon_len,
            'num_samples': len(all_predictions),
            'features': {FEATURE_NAMES_EN[k]: v for k, v in metrics_by_feature.items()},
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 评估指标已保存: {metrics_file}")
    
    # ============================================================
    # 步骤 3: 生成总览图（4个特征的平均趋势）
    # ============================================================
    print("\n📈 步骤 3/4: 生成总览图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    axes = axes.flatten()
    
    # 生成时间轴（简化：使用相对时间）
    time_steps = list(range(horizon_len))
    
    for feat_idx in range(4):
        ax = axes[feat_idx]
        
        # 找到该特征的所有样本
        mask = [i for i, f in enumerate(feature_indices) if f == feat_idx]
        
        if not mask:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(FEATURE_NAMES_EN[feat_idx])
            continue
        
        # 计算平均值
        avg_pred = all_predictions[mask].mean(axis=0)
        avg_true = all_targets[mask].mean(axis=0)
        
        # 绘制
        ax.plot(time_steps, avg_true, linewidth=2, color="#1f77b4", label='True Traffic')
        ax.plot(time_steps, avg_pred, linewidth=2, color="#ff4b5c", linestyle='--', label='Pred Traffic')
        
        ax.set_title(f"{FEATURE_NAMES_EN[feat_idx]} Overview", fontsize=12, fontweight='bold')
        ax.set_xlabel("Prediction Step (hours)", fontsize=10)
        ax.set_ylabel("Traffic Flow (veh/h)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    plt.suptitle(f"TimesFM Prediction Overview ({context_len}-in {horizon_len}-out)", 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = output_dir / "figures" / "overview_4features.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 总览图已保存: {save_path}")
    
    # ============================================================
    # 步骤 4: 为每个站点生成独立图（按站点分组）
    # ============================================================
    print("\n📊 步骤 4/4: 生成所有站点独立图...")
    
    # 按站点分组
    stations_data = {}
    for idx in range(len(all_predictions)):
        station_idx = station_ids[idx]
        if station_idx not in stations_data:
            stations_data[station_idx] = {f: {'preds': [], 'targets': []} for f in range(4)}
        
        feat_idx = feature_indices[idx]
        stations_data[station_idx][feat_idx]['preds'].append(all_predictions[idx])
        stations_data[station_idx][feat_idx]['targets'].append(all_targets[idx])
    
    total_stations = len(stations_data)
    print(f"总站点数: {total_stations}")
    
    for station_idx, features_data in stations_data.items():
        try:
            fig, axes = plt.subplots(2, 2, figsize=(20, 10))
            axes = axes.flatten()
            
            for feat_idx in range(4):
                ax = axes[feat_idx]
                
                feat_data = features_data[feat_idx]
                
                if not feat_data['preds']:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(FEATURE_NAMES_EN[feat_idx])
                    continue
                
                # ✅ 修复：使用np.stack将样本堆叠为2D数组，然后求平均
                # 原代码使用concatenate会导致维度展平
                all_preds = np.stack(feat_data['preds'], axis=0)  # (num_samples, horizon_len)
                all_trues = np.stack(feat_data['targets'], axis=0)  # (num_samples, horizon_len)
                
                # 计算平均趋势
                avg_pred = all_preds.mean(axis=0)  # (horizon_len,)
                avg_true = all_trues.mean(axis=0)  # (horizon_len,)
                
                # 绘制
                ax.plot(time_steps, avg_true, linewidth=1.8, color="#1f77b4", label='True')
                ax.plot(time_steps, avg_pred, linewidth=1.8, color="#ff4b5c", linestyle='--', label='Pred')
                
                ax.set_title(f"Station {station_idx:03d} - {FEATURE_NAMES_EN[feat_idx]}", 
                            fontsize=11, fontweight='bold')
                ax.set_xlabel("Prediction Step", fontsize=10)
                ax.set_ylabel("Traffic Flow", fontsize=10)
                ax.grid(alpha=0.3)
                ax.legend(loc='best', fontsize=9)
            
            plt.suptitle(f"Station {station_idx:03d} - All Features", 
                        fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            save_path = output_dir / "figures" / "all_stations" / f"station_{station_idx:03d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()  # ✅ 关键：关闭figure释放内存
            
            # 每20个站点打印进度
            if (station_idx + 1) % 20 == 0:
                print(f"  已处理 {station_idx + 1}/{total_stations} 个站点")
                
        except Exception as e:
            print(f"  ⚠️  站点 {station_idx} 失败: {e}")
            plt.close()  # 确保出错时也关闭figure

    # ============================================================
    # 步骤 5: 保存预测结果为 CSV
    # ============================================================
    print("\n💾 保存预测结果为 CSV...")
    
    rows = []
    for idx in range(len(all_predictions)):
        station_idx = station_ids[idx]
        feat_idx = feature_indices[idx]
        
        for step in range(horizon_len):
            rows.append({
                'station_id': station_idx,
                'feature': FEATURE_NAMES_EN[feat_idx],
                'prediction_step': step + 1,
                'true_value': round(float(all_targets[idx, step]), 2),
                'pred_value': round(float(all_predictions[idx, step]), 2),
                'absolute_error': round(float(abs(all_targets[idx, step] - all_predictions[idx, step])), 2)
            })
    
    df_results = pd.DataFrame(rows)
    csv_path = output_dir / "predictions" / "all_predictions.csv"
    df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"✓ 预测结果已保存: {csv_path}")
    
    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "="*80)
    print("🎉 全部完成！")
    print("="*80)
    print(f"输出目录: {output_dir}")
    print(f"  - 总览图: {output_dir / 'figures' / 'overview_4features.png'}")
    print(f"  - 站点图: {output_dir / 'figures' / 'all_stations'}/station_XXX.png")
    print(f"  - 评估指标: {output_dir / 'metrics' / 'evaluation_metrics.json'}")
    print(f"  - 预测结果: {output_dir / 'predictions' / 'all_predictions.csv'}")
    print("="*80)


def parse_args():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='TimesFM Prediction and Visualization (32-input 8-output)'
    )
    
    parser.add_argument(
        '--model_id',
        default=config.MODEL_ID,
        help='HuggingFace model ID'
    )
    parser.add_argument(
        '--adapter_path',
        default=config.CHECKPOINT_DIR,
        help='LoRA adapter path'
    )
    parser.add_argument(
        '--data_dir',
        default=config.PREPROCESSED_DIR,
        help='Preprocessed data directory'
    )
    parser.add_argument(
        '--output_dir',
        default=config.PREDICTION_DIR,
        help='Prediction results output directory'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    batch_predict_and_visualize(args)


if __name__ == '__main__':
    main()
