# Copyright 2024 Arjun Ashok
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
使用Lag-Llama预训练模型进行零样本预测

说明:
1. 从HuggingFace加载预训练权重
2. 对测试集进行预测
3. 按特征分别保存预测结果
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
# 添加允许的全局对象,解决 weights_only=True 导致的加载失败
try:
    from gluonts.torch.distributions import StudentTOutput, NegativeBinomialOutput
    from gluonts.torch.modules.loss import NegativeLogLikelihood
    
    # 将这些类添加到安全白名单
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

# 模型配置
MODEL_NAME = "time-series-foundation-models/Lag-Llama"
CONTEXT_LEN = 8   # 修改为8，与预处理脚本保持一致（实现8输入8输出）
PREDICTION_LEN = 8  # 预测长度
NUM_SAMPLES = 100  # 采样次数(概率预测)

# 数据文件
TRAIN_FILE = PROCESSED_DATA_DIR / "train.jsonl"
TEST_FILE = PROCESSED_DATA_DIR / "test.jsonl"
STATION_MAPPING_FILE = PROCESSED_DATA_DIR / "station_mapping.csv"

# 特征定义(与预处理脚本保持一致)
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
            # 将start字符串转换回Timestamp
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
        # 直接从checkpoint加载配置
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
        
        # 提取模型架构参数
        n_layer = model_kwargs.get('n_layer', 8)
        n_head = model_kwargs.get('n_head', 9)
        n_embd_per_head = model_kwargs.get('n_embd_per_head', 16)
        
        # 计算嵌入维度 (关键!)
        embed_dim = n_head * n_embd_per_head
        
        # 提取更多配置参数
        time_feat = model_kwargs.get('time_feat', True)  # 默认为True
        
        print(f"  lags_seq长度: {len(lags_seq_from_ckpt)}")
        print(f"  lags_seq前10个值: {lags_seq_from_ckpt[:10]}")
        print(f"  scaling: {scaling}")
        print(f"  context_length: {context_length}")
        print(f"  time_feat: {time_feat} (重要!)")
        print(f"  n_layer: {n_layer}")
        print(f"  n_head: {n_head}")
        print(f"  n_embd_per_head: {n_embd_per_head}")
        print(f"  embed_dim: {embed_dim} (重要!)")
        
        # 计算预期的feature_size
        if time_feat:
            expected_feature_size = 1 * len(lags_seq_from_ckpt) + 2 * 1 + 6
        else:
            expected_feature_size = 1 * len(lags_seq_from_ckpt) + 2 * 1
        print(f"  预期feature_size: {expected_feature_size} (应该是92)")
        
        # 关键修复: 如果lags_seq是整数列表,直接使用;否则使用默认频率字符串
        # 检查lags_seq的元素类型
        if len(lags_seq_from_ckpt) > 0 and isinstance(lags_seq_from_ckpt[0], int):
            # checkpoint中已经是整数索引列表,直接使用
            lags_seq_to_use = lags_seq_from_ckpt
            print(f"  ✓ 使用checkpoint中的整数lags_seq")
        else:
            # 使用默认频率字符串
            lags_seq_to_use = ["Q", "M", "W", "D", "H", "T", "S"]
            print(f"  ✓ 使用默认频率字符串lags_seq")
        
        # 创建estimator - 使用从checkpoint提取的正确参数
        estimator = LagLlamaEstimator(
            ckpt_path=checkpoint_path,
            prediction_length=PREDICTION_LEN,
            context_length=context_length,  # 使用checkpoint的context_length
            num_parallel_samples=NUM_SAMPLES,
            scaling=scaling,
            use_single_pass_sampling=True,
            time_feat=time_feat,  # 关键: 必须与预训练模型一致!
            # 关键: 必须使用与预训练模型一致的架构参数
            n_layer=n_layer,
            n_head=n_head,
            n_embd_per_head=n_embd_per_head,
            lags_seq=lags_seq_to_use,  # 使用正确的lags_seq
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # 训练estimator(实际上只是加载模型,不进行训练)
        predictor = estimator.train(training_data=ListDataset([], freq='H'))
        
        print("✓ 模型加载成功!")
        return predictor
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def predict_and_save(predictor, test_dataset, feature_key, feature_name):
    """
    对测试集进行预测并保存结果
    
    Args:
        predictor: 训练好的predictor
        test_dataset: 测试数据集
        feature_key: 特征键名
        feature_name: 特征中文名
    """
    print(f"\n{'='*80}")
    print(f"正在预测: {feature_name}")
    print(f"{'='*80}")
    
    # 生成预测
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_dataset,
        predictor=predictor,
        num_samples=NUM_SAMPLES
    )
    
    forecasts = list(tqdm(forecast_it, total=len(test_dataset), desc="生成预测"))
    tss = list(ts_it)
    
    print(f"预测完成!共 {len(forecasts)} 个序列")
    
    # 提取预测结果
    results = []
    
    for i, (forecast, ts) in enumerate(zip(forecasts, tss)):
        # 解析item_id获取站点信息
        item_id = forecast.item_id
        parts = item_id.split('_')
        
        # item_id格式: {station_id}_{feature_key}_window_{idx}
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
        
        # GluonTS的ts包含完整序列(context_len + prediction_len),需要取最后PREDICTION_LEN个作为真实值
        if len(ts_values) >= PREDICTION_LEN:
            true_values = ts_values[-PREDICTION_LEN:].tolist()
        else:
            print(f"  ⚠️ 警告: ts长度不足 {len(ts_values)} < {PREDICTION_LEN}")
            true_values = ts_values.tolist() + [None] * (PREDICTION_LEN - len(ts_values))
        
        # 预测值(取中位数作为点预测)
        pred_values = forecast.median.tolist()
        
        # 置信区间(10%和90%分位数)
        pred_lower = forecast.quantile(0.1).tolist()
        pred_upper = forecast.quantile(0.9).tolist()
        
        if i < 2:  # 只打印前2个样本的调试信息
            print(f"  样本{i}: ts长度={len(ts_values)}, 真实值前3个={true_values[:3]}")
            print(f"           预测值前3个={pred_values[:3]}")
        
        # 时间戳 - 关键修复:预测时间起点应该是 start + context_length
        # forecast.start_date 是样本的起始时间(包含输入序列)
        # 预测应该从输入序列之后开始
        start_time = forecast.start_date
        # 如果start_time是Period对象,先转换为Timestamp
        if hasattr(start_time, 'to_timestamp'):
            start_time = start_time.to_timestamp()
        
        # 预测的实际开始时间 = start + CONTEXT_LEN小时
        pred_start_time = start_time + pd.Timedelta(hours=CONTEXT_LEN)
        
        print(f"  样本ID: {item_id}")
        print(f"  样本起始时间: {start_time}")
        print(f"  预测起始时间: {pred_start_time}")
        if i == 0:  # 只打印第一个样本的调试信息
            print(f"  前3个预测时间戳:")
        
        timestamps = [pred_start_time + pd.Timedelta(hours=j) for j in range(PREDICTION_LEN)]
        
        # 构建结果记录
        for j in range(PREDICTION_LEN):
            # 确保timestamp是Timestamp对象后再调用isoformat
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
        
        # MAPE (避免除以0)
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Lag-Llama交通流量预测')
    parser.add_argument('--dataset', type=str, default='train', 
                       choices=['train', 'test'],
                       help='选择要预测的数据集: train(训练集) 或 test(测试集)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Lag-Llama 高速公路交通流量预测")
    print(f"目标数据集: {args.dataset.upper()}")
    print("=" * 80)
    
    # Step 1: 检查数据文件是否存在
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        print("错误: 未找到预处理后的数据文件!")
        print("请先运行: python scripts/preprocess_traffic_data.py")
        return
    
    # Step 2: 根据参数选择数据集
    if args.dataset == 'train':
        data_file = TRAIN_FILE
        print(f"\n使用训练集: {TRAIN_FILE}")
    else:
        data_file = TEST_FILE
        print(f"\n使用测试集: {TEST_FILE}")
    
    dataset = load_dataset(data_file)
    
    # Step 3: 加载模型
    predictor = load_model()
    
    # Step 4: 为每个特征进行预测
    all_results = {}
    
    for feature_key, feature_info in FEATURES.items():
        # 筛选当前特征的样本
        feature_samples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                sample['start'] = pd.Timestamp(sample['start'])
                # 检查item_id是否包含当前特征
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
    
    # Step 5: 汇总所有特征的结果
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
