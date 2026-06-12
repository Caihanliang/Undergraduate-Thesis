#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 his_data.csv 中每个站点的零值个数（分特征讨论）
特征：小客车上行、小客车下行、非小客车上行、非小客车下行
"""

import pandas as pd
import numpy as np
import os

# ========== 配置参数 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# CSV_PATH = os.path.join(BASE_DIR, "main-master", "datasets", "HNGS_4FEAT", "his_data.csv")
CSV_PATH = "main-master/datasets/HNGS_4FEAT/his_data.csv"

# OUTPUT_CSV = os.path.join(BASE_DIR, "main-master", "datasets", "HNGS_4FEAT", "zero_value_statistics_simple.csv")
OUTPUT_CSV = "main-master/datasets/HNGS_4FEAT/zero_value_statistics_simple.csv"



def analyze_zero_values():
    """分析每个站点各特征的零值情况（分特征讨论）"""
    print("=" * 80)
    print("📊 开始统计零值分布（按4个特征分别统计）")
    print("=" * 80)
    
    # 加载数据
    print(f"\n📂 加载数据: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  ✓ 数据量: {len(df)} 行")
    print(f"  ✓ 列名: {list(df.columns)}")
    
    # 定义特征列
    feature_cols = ['小客车上行', '小客车下行', '非小客车上行', '非小客车下行']
    
    # 统计每个站点每个特征的零值个数
    print(f"\n🔍 正在统计零值...")
    results = []
    
    stations = sorted(df['站点编号'].unique())
    total_stations = len(stations)
    
    for idx, station in enumerate(stations):
        station_data = df[df['站点编号'] == station]
        
        row = {'站点编号': station}
        
        for feat in feature_cols:
            zero_count = (station_data[feat] == 0).sum()
            total_count = len(station_data)
            zero_ratio = zero_count / total_count * 100
            
            row[f'{feat}_零值个数'] = int(zero_count)
            row[f'{feat}_总样本数'] = int(total_count)
            row[f'{feat}_零值占比(%)'] = round(zero_ratio, 2)
        
        results.append(row)
        
        # 进度显示
        if (idx + 1) % 20 == 0 or idx == total_stations - 1:
            print(f"  处理进度: {idx + 1}/{total_stations} ({(idx+1)/total_stations*100:.1f}%)")
    
    # 创建结果 DataFrame
    result_df = pd.DataFrame(results)
    
    print(f"\n✅ 统计完成！共 {len(result_df)} 个站点")
    
    # 保存结果
    print(f"\n💾 保存统计结果: {OUTPUT_CSV}")
    result_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    file_size = os.path.getsize(OUTPUT_CSV) / 1024
    print(f"  ✓ 文件大小: {file_size:.2f} KB")
    
    # ==================== 按特征分别统计 ====================
    print(f"\n{'=' * 80}")
    print("📈 按特征分别统计零值情况")
    print(f"{'=' * 80}")
    
    for feat_idx, feat in enumerate(feature_cols, 1):
        print(f"\n{'─' * 80}")
        print(f"【特征 {feat_idx}】{feat}")
        print(f"{'─' * 80}")
        
        # 总体统计
        total_zeros = result_df[f'{feat}_零值个数'].sum()
        total_samples = result_df[f'{feat}_总样本数'].sum()
        avg_ratio = total_zeros / total_samples * 100
        
        print(f"\n  📊 总体统计:")
        print(f"    总零值数: {total_zeros:,}")
        print(f"    总样本数: {total_samples:,}")
        print(f"    平均零值率: {avg_ratio:.2f}%")
        
        # 零值最多的前10个站点
        print(f"\n  🔝 零值最多的前10个站点:")
        top_stations = result_df.nlargest(10, f'{feat}_零值个数')
        
        for i, (_, row) in enumerate(top_stations.iterrows()):
            station = row['站点编号']
            zeros = int(row[f'{feat}_零值个数'])
            ratio = row[f'{feat}_零值占比(%)']
            print(f"    {i+1}. 站点 {station}: {zeros:,} 个零值 (占比 {ratio:.2f}%)")
        
        # 零值率最高的前10个站点
        print(f"\n  📉 零值率最高的前10个站点:")
        top_ratio_stations = result_df.nlargest(10, f'{feat}_零值占比(%)')
        
        for i, (_, row) in enumerate(top_ratio_stations.iterrows()):
            station = row['站点编号']
            zeros = int(row[f'{feat}_零值个数'])
            ratio = row[f'{feat}_零值占比(%)']
            print(f"    {i+1}. 站点 {station}: {zeros:,} 个零值 (占比 {ratio:.2f}%)")
    
    # ==================== 对比分析 ====================
    print(f"\n{'=' * 80}")
    print("📊 4个特征对比分析")
    print(f"{'=' * 80}")
    
    print(f"\n{'特征名称':<15} {'总零值数':<15} {'总样本数':<15} {'平均零值率':<15}")
    print(f"{'─' * 60}")
    
    for feat in feature_cols:
        total_zeros = result_df[f'{feat}_零值个数'].sum()
        total_samples = result_df[f'{feat}_总样本数'].sum()
        avg_ratio = total_zeros / total_samples * 100
        
        print(f"{feat:<15} {total_zeros:<15,} {total_samples:<15,} {avg_ratio:<15.2f}%")
    
    # 打印详细示例
    print(f"\n{'=' * 80}")
    print("📋 详细数据示例（前5个站点）")
    print(f"{'=' * 80}")
    print(result_df.head(5).to_string(index=False))
    
    return result_df


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🚀 开始分析 his_data.csv 零值分布（分特征讨论）")
    print("=" * 80 + "\n")
    
    try:
        result_df = analyze_zero_values()
        
        print("\n" + "=" * 80)
        print("🎉 分析完成！")
        print("=" * 80)
        print(f"\n输出文件: {OUTPUT_CSV}")
        print(f"\n使用建议:")
        print(f"  1. 每个特征的零值分布可能不同，建议分别分析")
        print(f"  2. 重点关注零值率 > 50% 的站点/特征，可能存在数据采集问题")
        print(f"  3. 对比4个特征的零值模式，识别系统性异常")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
