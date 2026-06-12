#!/usr/bin/env python3
"""
干净的数据生成 - 不训练，只生成数据
"""
import os
import json
import numpy as np
import pandas as pd
from chinese_calendar import is_workday
import random

PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"

print("="*60)
print("🧹 生成干净的训练数据")
print("="*60)

# 1. 加载基础数据
print("📥 加载NPZ数据...")
npz_true = np.load(os.path.join(PROJECT_ROOT, "finetune_real_traffic.npz"))
npz_pred = np.load(os.path.join(PROJECT_ROOT, "finetune_data.npz"))

true_array = npz_true['target'] if 'target' in npz_true else npz_true['arr_0']
pred_array = npz_pred['prediction'] if 'prediction' in npz_pred else npz_pred['arr_0']

# 维度对齐
if true_array.ndim == 4 and true_array.shape[1] == 8 and true_array.shape[3] == 4:
    true_array = true_array.transpose(0, 2, 3, 1)
    pred_array = pred_array.transpose(0, 2, 3, 1)

num_samples, num_stations, num_features, seq_len = true_array.shape
print(f"数据形状: {true_array.shape}")

# 2. 简化：只处理高误差样本
print(f"\n🔍 筛选高误差样本 (MAE > 50)...")

generated_data = []
high_error_count = 0
total_count = 0

for j in range(min(num_stations, 10)):  # 只处理前10个站点
    station_name = f"Station_{j}"
    
    for feat_idx in range(num_features):
        for i in range(min(num_samples, 100)):  # 只处理前100个样本
            total_count += 1
            
            try:
                pred_vals = pred_array[i, j, feat_idx]
                true_vals = true_array[i, j, feat_idx]
                mae = np.mean(np.abs(pred_vals - true_vals))
                
                if mae > 50:  # 高误差样本
                    high_error_count += 1
                    
                    # 生成简单但有效的训练样本
                    gnn_str = ",".join(str(int(v)) for v in pred_vals)
                    truth_str = ",".join(str(int(v)) for v in true_vals)
                    
                    # 构建训练文本
                    text = f"""
Station: {station_name}
Feature: {feat_idx}
GNN prediction: [{gnn_str}]
True values: [{truth_str}]
Final Correction: [{truth_str}] <|eot_id|>
"""
                    
                    # 简化token化（实际训练时会用真实tokenizer）
                    generated_data.append({
                        "text": text.strip(),
                        "gnn_values": pred_vals.tolist(),
                        "true_values": true_vals.tolist(),
                        "mae": float(mae),
                        "station_idx": j,
                        "feature_idx": feat_idx,
                        "sample_idx": i
                    })
                    
                    if len(generated_data) % 100 == 0:
                        print(f"生成 {len(generated_data)} 条数据...")
                        
            except Exception as e:
                continue

# 3. 保存
output_path = os.path.join(PROJECT_ROOT, "clean_training_data.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(generated_data, f, ensure_ascii=False, indent=2)

print(f"\n✅ 数据生成完成！")
print(f"处理样本: {total_count}")
print(f"高误差样本: {high_error_count}")
print(f"生成数据: {len(generated_data)} 条")
print(f"保存到: {output_path}")