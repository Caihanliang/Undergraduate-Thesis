#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 FaST 4特征模型的预测结果
读取 metrics、predictions 和 error_annotation 数据
生成论文可用的性能总结
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# 结果目录
RESULT_DIR = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/visulization-result-4FEAT-train"

print("=" * 80)
print(" FaST-4FEAT 模型性能分析")
print("=" * 80)

# 1. 读取指标文件
print("\n【1】读取评估指标")
metrics_dir = os.path.join(RESULT_DIR, "metrics")
if os.path.exists(metrics_dir):
    for file in os.listdir(metrics_dir):
        if file.endswith('.json') or file.endswith('.txt'):
            file_path = os.path.join(metrics_dir, file)
            print(f"\n📄 {file}:")
            try:
                if file.endswith('.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        print(json.dumps(data, indent=2, ensure_ascii=False))
                else:
                    with open(file_path, 'r') as f:
                        print(f.read())
            except Exception as e:
                print(f"  读取失败: {e}")

# 2. 读取预测结果
print("\n【2】读取预测结果统计")
predictions_dir = os.path.join(RESULT_DIR, "predictions")
if os.path.exists(predictions_dir):
    for file in sorted(os.listdir(predictions_dir)):
        if file.endswith('.csv') or file.endswith('.npy') or file.endswith('.npz'):
            file_path = os.path.join(predictions_dir, file)
            print(f"\n📊 {file}:")
            try:
                if file.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    print(f"  形状: {df.shape}")
                    print(f"  列名: {list(df.columns)[:10]}...")
                    if 'MAE' in df.columns or 'mae' in df.columns:
                        mae_col = 'MAE' if 'MAE' in df.columns else 'mae'
                        print(f"  MAE 统计:")
                        print(f"    均值: {df[mae_col].mean():.4f}")
                        print(f"    中位数: {df[mae_col].median():.4f}")
                        print(f"    标准差: {df[mae_col].std():.4f}")
                        print(f"    最小值: {df[mae_col].min():.4f}")
                        print(f"    最大值: {df[mae_col].max():.4f}")
                elif file.endswith('.npz'):
                    data = np.load(file_path)
                    print(f"  包含的数组: {list(data.keys())}")
                    for key in data.keys():
                        print(f"    {key}: {data[key].shape}")
            except Exception as e:
                print(f"  读取失败: {e}")

# 3. 分析误差标注
print("\n【3】分析误差分布")
error_dir = os.path.join(RESULT_DIR, "error_annotation")
if os.path.exists(error_dir):
    error_files = [f for f in os.listdir(error_dir) if f.endswith('.json')]
    print(f"  误差标注文件数: {len(error_files)}")
    
    if len(error_files) > 0:
        # 读取几个样本文件
        sample_files = error_files[:min(5, len(error_files))]
        all_errors = []
        
        for file in sample_files:
            file_path = os.path.join(error_dir, file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if 'error' in data or 'mae' in data:
                        error_val = data.get('error', data.get('mae', 0))
                        all_errors.append(error_val)
            except:
                pass
        
        if all_errors:
            all_errors = np.array(all_errors)
            print(f"\n  误差统计 (样本 {len(all_errors)} 个):")
            print(f"    均值: {all_errors.mean():.4f}")
            print(f"    标准差: {all_errors.std():.4f}")
            print(f"    中位数: {np.median(all_errors):.4f}")

# 4. 读取所有节点预测结果
print("\n【4】全局预测结果分析")
all_nodes_file = os.path.join(RESULT_DIR, "all_nodes_prediction.csv")
if os.path.exists(all_nodes_file):
    print(f"  正在加载 {all_nodes_file} ...")
    try:
        df_all = pd.read_csv(all_nodes_file)
        print(f"  数据形状: {df_all.shape}")
        print(f"  列名: {list(df_all.columns)}")
        
        # 寻找误差相关列
        error_cols = [col for col in df_all.columns if 'MAE' in col.upper() or 'ERROR' in col.upper() or 'MAPE' in col.upper()]
        if error_cols:
            print(f"\n  误差列: {error_cols}")
            for col in error_cols:
                print(f"    {col}:")
                print(f"      均值: {df_all[col].mean():.4f}")
                print(f"      中位数: {df_all[col].median():.4f}")
                print(f"      标准差: {df_all[col].std():.4f}")
    except Exception as e:
        print(f"  读取失败: {e}")

# 5. 读取图表
print("\n【5】可视化结果")
figures_dir = os.path.join(RESULT_DIR, "figures")
if os.path.exists(figures_dir):
    figures = os.listdir(figures_dir)
    print(f"  图表文件: {figures}")

print("\n" + "=" * 80)
print(" 分析完成")
print("=" * 80)
