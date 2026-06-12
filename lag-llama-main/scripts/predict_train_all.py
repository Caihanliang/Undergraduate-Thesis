"""
预测训练集所有时间窗口的脚本

功能:
1. 加载训练集数据 (train.jsonl)
2. 使用Lag-Llama模型对所有样本进行预测
3. 按特征分别保存预测结果
4. 生成合并后的评估文件

使用说明:
python scripts/predict_train_all.py
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions
from tqdm import tqdm

# ==================== PyTorch 2.6+ 安全限制修复 ====================
try:
    from gluonts.torch.distributions import StudentTOutput, NegativeBinomialOutput
    from gluonts.torch.modules.loss import NegativeLogLikelihood
    
    torch.serialization.add_safe_globals([
        StudentTOutput,
        NegativeBinomialOutput,
        NegativeLogLikelihood
    ])
    print("✓ 已注册PyTorch安全全局对象")
except Exception as e:
    print(f"警告: 注册安全全局对象失败: {e}")

# 设置HuggingFace镜像(国内加速)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from lag_llama.gluon.estimator import LagLlamaEstimator

# ==================== 配置参数 ====================
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
RESULTS_DIR = PROJECT_ROOT / "prediction_results"

# 模型配置 - 8输入8输出
MODEL_NAME = "time-series-foundation-models/Lag-Llama"
CONTEXT_LEN = 8   # 8小时输入
PREDICTION_LEN = 8  # 8小时输出
NUM_SAMPLES = 100  # 采样次数

# 数据文件 - 使用训练集
TRAIN_FILE = PROCESSED_DATA_DIR / "train.jsonl"

# 特征定义
FEATURES = {
    'passenger_up': {'name': '小客车上行', 'direction': '上行'},
    'passenger_down': {'name': '小客车下行', 'direction': '下行'},
    'non_passenger_up': {'name': '非小客车上行', 'direction': '上行'},
    'non_passenger_down': {'name': '非小客车下行', 'direction': '下行'}
}


def load_dataset(file_path):
    """加载JSONL格式的数据集"""
    print(f"正在加载数据集: {file_path}")
    
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            sample['start'] = pd.Timestamp(sample['start'])
            samples.append(sample)
    
    dataset = ListDataset(samples, freq='H')
    print(f"  加载完成,样本数: {len(dataset)}")
    return dataset


def load_model():
    """加载Lag-Llama预训练模型"""
    print("=" * 80)
    print("正在加载Lag-Llama预训练模型...")
    print("=" * 80)
    
    try:
        from huggingface_hub import hf_hub_download
        
        checkpoint_path = hf_hub_download(
            repo_id=MODEL_NAME,
            filename="lag-llama.ckpt"
        )
        
        print(f"Checkpoint下载至: {checkpoint_path}")
        
        # 从checkpoint提取完整配置
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        model_kwargs = checkpoint['hyper_parameters']['model_kwargs']
        lags_seq_from_ckpt = model_kwargs['lags_seq']
        scaling = model_kwargs.get('scaling', 'mean')
        context_length = checkpoint['hyper_parameters'].get('context_length', CONTEXT_LEN)
        
        n_layer = model_kwargs.get('n_layer', 8)
        n_head = model_kwargs.get('n_head', 9)
        n_embd_per_head = model_kwargs.get('n_embd_per_head', 16)
        embed_dim = n_head * n_embd_per_head
        time_feat = model_kwargs.get('time_feat', True)
        
        print(f"  模型架构参数:")
        print(f"    context_length: {context_length}")
        print(f"    n_layer: {n_layer}, n_head: {n_head}, embed_dim: {embed_dim}")
        print(f"    time_feat: {time_feat}")
        
        # 关键修复: 如果lags_seq是整数列表,直接使用
        if len(lags_seq_from_ckpt) > 0 and isinstance(lags_seq_from_ckpt[0], int):
            lags_seq_to_use = lags_seq_from_ckpt
            print(f"  ✓ 使用checkpoint中的整数lags_seq (长度: {len(lags_seq_to_use)})")
        else:
            lags_seq_to_use = ["Q", "M", "W", "D", "H", "T", "S"]
            print(f"  ✓ 使用默认频率字符串lags_seq")
        
        # 创建estimator
        estimator = LagLlamaEstimator(
            ckpt_path=checkpoint_path,
            prediction_length=PREDICTION_LEN,
            context_length=context_length,
            num_parallel_samples=NUM_SAMPLES,
            scaling=scaling,
            use_single_pass_sampling=True,
            time_feat=time_feat,
            n_layer=n_layer,
            n_head=n_head,
            n_embd_per_head=n_embd_per_head,
            lags_seq=lags_seq_to_use,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # 训练estimator(实际上只是加载模型)
        predictor = estimator.train(training_data=ListDataset([], freq='H'))
        
        print("✓ 模型加载成功!")
        return predictor
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def predict_and_save(predictor, dataset, feature_key, feature_name):
    """对数据集进行预测并保存结果"""
    print(f"\n{'='*80}")
    print(f"正在预测: {feature_name}")
    print(f"{'='*80}")
    
    # 生成预测
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=NUM_SAMPLES
    )
    
    forecasts = list(tqdm(forecast_it, total=len(dataset), desc="生成预测"))
    tss = list(ts_it)
    
    print(f"预测完成!共 {len(forecasts)} 个序列")
    
    # 提取预测结果
    results = []
    
    for i, (forecast, ts) in enumerate(zip(forecasts, tss)):
        item_id = forecast.item_id
        parts = item_id.split('_')
        
        if len(parts) >= 3:
            station_id = parts[0]
            window_idx = parts[-1] if parts[-1].startswith('window') else '0'
        else:
            station_id = item_id
            window_idx = '0'
        
        # 真实值提取
        if hasattr(ts, 'values'):
            ts_values = ts.values
        else:
            ts_values = np.array(ts)
        
        if hasattr(ts_values, 'flatten'):
            ts_values = ts_values.flatten()
        
        # 取最后PREDICTION_LEN个作为真实值
        if len(ts_values) >= PREDICTION_LEN:
            true_values = ts_values[-PREDICTION_LEN:].tolist()
        else:
            true_values = ts_values.tolist() + [None] * (PREDICTION_LEN - len(ts_values))
        
        # 预测值(取中位数)
        pred_values = forecast.median.tolist()
        
        # 置信区间
        pred_lower = forecast.quantile(0.1).tolist()
        pred_upper = forecast.quantile(0.9).tolist()
        
        # 时间戳计算 - 关键修复
        start_time = forecast.start_date
        if hasattr(start_time, 'to_timestamp'):
            start_time = start_time.to_timestamp()
        
        # 预测的实际开始时间 = start + CONTEXT_LEN小时
        pred_start_time = start_time + pd.Timedelta(hours=CONTEXT_LEN)
        
        timestamps = [pred_start_time + pd.Timedelta(hours=j) for j in range(PREDICTION_LEN)]
        
        # 构建结果记录
        for j in range(PREDICTION_LEN):
            ts_j = timestamps[j]
            if hasattr(ts_j, 'to_timestamp'):
                ts_j = ts_j.to_timestamp()
            
            results.append({
                'station_id': station_id,
                'feature_key': feature_key,
                'feature_name': feature_name,
                'window_idx': window_idx,
                'timestamp': ts_j.isoformat(),
                'hour_offset': j,
                'true_value': true_values[j] if j < len(true_values) else None,
                'pred_value': pred_values[j],
                'pred_lower_10': pred_lower[j],
                'pred_upper_90': pred_upper[j]
            })
    
    # 转换为DataFrame
    df_results = pd.DataFrame(results)
    
    # 保存结果
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / f"{feature_key}_predictions.csv"
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"结果已保存至: {output_file}")
    print(f"  总记录数: {len(df_results)}")
    
    # 计算评估指标
    if df_results['true_value'].notna().any():
        valid_mask = df_results['true_value'].notna()
        mae = np.mean(np.abs(df_results.loc[valid_mask, 'true_value'] - df_results.loc[valid_mask, 'pred_value']))
        rmse = np.sqrt(np.mean((df_results.loc[valid_mask, 'true_value'] - df_results.loc[valid_mask, 'pred_value']) ** 2))
        
        true_vals = df_results.loc[valid_mask, 'true_value']
        pred_vals = df_results.loc[valid_mask, 'pred_value']
        mask_nonzero = true_vals != 0
        if mask_nonzero.any():
            mape = np.mean(np.abs((true_vals[mask_nonzero] - pred_vals[mask_nonzero]) / true_vals[mask_nonzero])) * 100
        else:
            mape = np.nan
        
        print(f"\n评估指标:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.2f}%")
    
    return df_results


def main():
    """主函数: 执行完整的预测流程"""
    print("=" * 80)
    print("Lag-Llama 高速公路交通流量预测 - 训练集全量预测")
    print("配置: 8输入8输出")
    print("=" * 80)
    
    # Step 1: 检查数据文件是否存在
    if not TRAIN_FILE.exists():
        print("错误: 未找到预处理后的训练集文件!")
        print("请先运行: python scripts/preprocess_traffic_data.py")
        return
    
    print(f"\n使用训练集: {TRAIN_FILE}")
    dataset = load_dataset(TRAIN_FILE)
    
    # Step 2: 加载模型
    predictor = load_model()
    
    # Step 3: 为每个特征进行预测
    all_results = {}
    
    for feature_key, feature_info in FEATURES.items():
        # 筛选当前特征的样本
        feature_samples = []
        with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                sample['start'] = pd.Timestamp(sample['start'])
                if f"_{feature_key}_" in sample['item_id']:
                    feature_samples.append(sample)
        
        if not feature_samples:
            print(f"警告: 未找到特征 {feature_key} 的样本")
            continue
        
        # 创建临时数据集
        feature_dataset = ListDataset(feature_samples, freq='H')
        print(f"\n特征 {feature_key} 的样本数: {len(feature_dataset)}")
        
        # 预测并保存
        df_results = predict_and_save(
            predictor, 
            feature_dataset, 
            feature_key, 
            feature_info['name']
        )
        all_results[feature_key] = df_results
    
    # Step 4: 汇总统计
    print(f"\n{'='*80}")
    print("预测完成!汇总统计")
    print(f"{'='*80}")
    
    for feature_key, df in all_results.items():
        print(f"\n{FEATURES[feature_key]['name']}:")
        print(f"  预测记录数: {len(df)}")
        print(f"  涉及站点数: {df['station_id'].nunique()}")
        print(f"  时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    
    print(f"\n所有结果已保存至: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
