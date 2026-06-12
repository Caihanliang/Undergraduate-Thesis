#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time-MoE 推理和可视化脚本
支持8输入8输出的时序预测
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from transformers import AutoModelForCausalLM
import json
import os
from datetime import datetime, timedelta
from pathlib import Path


class TimeMoEPredictor:
    """Time-MoE 预测器"""
    
    def __init__(self, model_path='Maple728/TimeMoE-50M', device='cpu'):
        """
        初始化预测器
        
        Args:
            model_path: 模型路径或HuggingFace模型ID
            device: 设备 ('cpu' 或 'cuda')
        """
        print(f"🔄 正在加载模型: {model_path}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        
        self.device = device
        self.model.eval()
        
        # 加载元数据
        self.metadata = None
        self.feature_cols = None
        
        print("✅ 模型加载完成")
    
    def load_metadata(self, metadata_path='./processed_data/metadata.json'):
        """加载元数据（标准化参数等）"""
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            self.feature_cols = self.metadata['feature_columns']
            print(f"✅ 元数据加载完成，特征列: {self.feature_cols}")
        else:
            print(f"⚠️  警告: 未找到元数据文件 {metadata_path}")
    
    def normalize(self, data):
        """标准化数据"""
        if self.metadata is None:
            return data
        
        mean = np.array(self.metadata['scaler_mean'])
        scale = np.array(self.metadata['scaler_scale'])
        
        return (data - mean) / scale
    
    def inverse_normalize(self, data):
        """反标准化数据"""
        if self.metadata is None:
            return data
        
        mean = np.array(self.metadata['scaler_mean'])
        scale = np.array(self.metadata['scaler_scale'])
        
        return data * scale + mean
    
    def predict(self, input_seq, prediction_length=8):
        """
        进行预测
        
        Args:
            input_seq: 输入序列，形状为 [context_length] 或 [batch_size, context_length]
            prediction_length: 预测长度
            
        Returns:
            predictions: 预测结果
        """
        # 确保输入是二维的
        if input_seq.ndim == 1:
            input_seq = input_seq.reshape(1, -1)
        
        # 转换为tensor并移到设备
        input_tensor = torch.FloatTensor(input_seq).to(self.device)
        
        # 生成预测
        with torch.no_grad():
            output = self.model.generate(
                input_tensor,
                max_new_tokens=prediction_length
            )
        
        # 提取预测部分
        predictions = output[:, -prediction_length:].cpu().numpy()
        
        return predictions
    
    def predict_and_inverse(self, input_seq, prediction_length=8):
        """预测并反标准化"""
        # 标准化输入
        if self.metadata is not None:
            input_normalized = self.normalize(input_seq)
        else:
            input_normalized = input_seq
        
        # 预测
        pred_normalized = self.predict(input_normalized, prediction_length)
        
        # 反标准化
        if self.metadata is not None:
            pred_original = self.inverse_normalize(pred_normalized)
        else:
            pred_original = pred_normalized
        
        return pred_original


def load_test_data(test_path='./processed_data/test.pkl'):
    """加载测试数据"""
    import pickle
    
    if os.path.exists(test_path):
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
        print(f"✅ 加载了 {len(test_data)} 个测试样本")
        return test_data
    else:
        raise FileNotFoundError(f"测试数据文件不存在: {test_path}")


def visualize_predictions(predictions, targets, timestamps, feature_names, 
                         save_dir='./visualization_results', sample_indices=None):
    """
    可视化预测结果
    
    Args:
        predictions: 预测值数组 [n_samples, prediction_length, n_features]
        targets: 真实值数组 [n_samples, prediction_length, n_features]
        timestamps: 时间戳列表
        feature_names: 特征名称列表
        save_dir: 保存目录
        sample_indices: 要可视化的样本索引列表，None表示全部
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if sample_indices is None:
        sample_indices = list(range(min(10, len(predictions))))  # 默认可视化前10个样本
    
    print(f"\n📊 正在可视化 {len(sample_indices)} 个样本...")
    
    for idx in sample_indices:
        fig, axes = plt.subplots(len(feature_names), 1, figsize=(14, 4*len(feature_names)))
        
        if len(feature_names) == 1:
            axes = [axes]
        
        start_time = pd.to_datetime(timestamps[idx]['start'])
        time_points = [start_time + timedelta(hours=i) for i in range(predictions.shape[1])]
        
        for feat_idx, (ax, feat_name) in enumerate(zip(axes, feature_names)):
            true_vals = targets[idx, :, feat_idx]
            pred_vals = predictions[idx, :, feat_idx]
            
            ax.plot(time_points, true_vals, 'b-o', label='True', linewidth=2, markersize=6)
            ax.plot(time_points, pred_vals, 'r--s', label='Predicted', linewidth=2, markersize=6)
            
            ax.set_title(f'{feat_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # 格式化x轴时间
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # 计算误差指标
        mae = np.mean(np.abs(predictions[idx] - targets[idx]))
        rmse = np.sqrt(np.mean((predictions[idx] - targets[idx]) ** 2))
        
        fig.suptitle(f'Sample {idx} | MAE: {mae:.4f} | RMSE: {rmse:.4f}', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        save_path = os.path.join(save_dir, f'sample_{idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ 样本 {idx} 已保存 (MAE={mae:.4f}, RMSE={rmse:.4f})")
    
    print(f"✅ 可视化完成，结果保存在: {save_dir}")


def calculate_metrics(predictions, targets):
    """计算评估指标"""
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


def save_metrics_to_json(metrics, save_path='./evaluation_metrics.json'):
    """保存评估指标到JSON文件"""
    # 确保所有值都是Python原生类型（可JSON序列化）
    metrics_serializable = {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in metrics.items()
    }
    
    # 添加时间戳
    metrics_serializable['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 评估指标已保存到: {save_path}")
    return metrics_serializable


def save_results_to_csv(predictions, targets, timestamps, feature_names, 
                       save_path='./prediction_results.csv'):
    """保存预测结果到CSV"""
    records = []
    
    for idx in range(len(predictions)):
        start_time = pd.to_datetime(timestamps[idx]['start'])
        
        for step in range(predictions.shape[1]):
            current_time = start_time + timedelta(hours=step)
            
            for feat_idx, feat_name in enumerate(feature_names):
                true_val = float(targets[idx, step, feat_idx])
                pred_val = float(predictions[idx, step, feat_idx])
                
                records.append({
                    'sample_index': idx,
                    'prediction_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'feature': feat_name,
                    'true_value': true_val,
                    'predicted_value': pred_val,
                    'error': abs(true_val - pred_val),
                    'absolute_percentage_error': abs((true_val - pred_val) / (true_val + 1e-8)) * 100
                })
    
    df_results = pd.DataFrame(records)
    df_results.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 预测结果已保存到: {save_path}")
    
    return df_results


def main():
    """主函数"""
    print("="*60)
    print("Time-MoE 推理和可视化")
    print("="*60)
    
    # 配置
    config = {
        'model_path': 'Maple728/TimeMoE-50M',  # 可以替换为你训练好的模型路径
        'device': 'cpu',  # 如果有GPU改为 'cuda'
        'prediction_length': 8,
        'test_data_path': './processed_data/test.pkl',
        'metadata_path': './processed_data/metadata.json',
        'save_dir': './visualization_results',
        'results_csv_path': './prediction_results.csv'
    }
    
    # 1. 初始化预测器
    predictor = TimeMoEPredictor(
        model_path=config['model_path'],
        device=config['device']
    )
    
    # 2. 加载元数据
    predictor.load_metadata(config['metadata_path'])
    
    # 3. 加载测试数据
    test_data = load_test_data(config['test_data_path'])
    
    # 4. 进行预测
    print(f"\n🔮 正在进行预测...")
    
    all_predictions = []
    all_targets = []
    all_timestamps = []
    
    # 取前100个样本进行测试（避免太慢）
    n_samples = min(100, len(test_data))
    
    for i in range(n_samples):
        sample = test_data[i]
        input_seq = np.array(sample['input'])  # [context_length, n_features]
        target_seq = np.array(sample['target'])  # [prediction_length, n_features]
        
        # 注意：Time-MoE处理的是单变量序列，需要对每个特征分别预测
        # 这里我们简化处理，假设已经转换好格式
        
        # 对于多变量情况，需要特殊处理
        # 这里演示单变量的方式
        if input_seq.ndim == 2:
            # 如果是多变量，取第一个特征作为示例
            input_1d = input_seq[:, 0]
            target_1d = target_seq[:, 0]
        else:
            input_1d = input_seq
            target_1d = target_seq
        
        pred = predictor.predict_and_inverse(
            input_1d,
            prediction_length=config['prediction_length']
        )
        
        all_predictions.append(pred)
        all_targets.append(target_1d.reshape(1, -1))
        all_timestamps.append({
            'start': sample['timestamp_start'],
            'end': sample['timestamp_end']
        })
        
        if (i + 1) % 10 == 0:
            print(f"   进度: {i+1}/{n_samples}")
    
    # 转换为numpy数组
    predictions = np.array(all_predictions)  # [n_samples, prediction_length, 1]
    targets = np.array(all_targets)  # [n_samples, prediction_length, 1]
    
    # 5. 计算指标
    metrics = calculate_metrics(predictions, targets)
    print(f"\n📊 评估指标:")
    print(f"   MAE:  {metrics['MAE']:.4f}")
    print(f"   RMSE: {metrics['RMSE']:.4f}")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")
    
    # 5.1 保存评估指标到JSON
    save_metrics_to_json(metrics, save_path='./evaluation_metrics.json')
    
    # 6. 可视化
    feature_names = predictor.feature_cols if predictor.feature_cols else ['Feature_0']
    visualize_predictions(
        predictions,
        targets,
        all_timestamps,
        feature_names,
        save_dir=config['save_dir']
    )
    
    # 7. 保存结果到CSV（包含详细误差信息）
    save_results_to_csv(
        predictions,
        targets,
        all_timestamps,
        feature_names,
        save_path=config['results_csv_path']
    )
    
    print("\n" + "="*60)
    print("✅ 推理和可视化完成！")
    print("="*60)
    print(f"\n📁 输出文件:")
    print(f"   - evaluation_metrics.json: 评估指标 (MAE, RMSE, MAPE)")
    print(f"   - prediction_results.csv: 详细预测结果和误差")
    print(f"   - visualization_results/: 可视化图表")
    print("="*60)


if __name__ == '__main__':
    main()
