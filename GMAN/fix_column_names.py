#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复GMAN预测结果文件的列名
将韩文"값"改为中文"值"
"""

import pandas as pd
import os

def fix_gman_column_names():
    """修复GMAN CSV文件的列名"""
    
    print("=" * 80)
    print("修复GMAN预测结果文件列名")
    print("=" * 80)
    
    # GMAN预测结果路径
    gman_file = '/home/user/Downloads/cai/GMAN/results/train_predictions.csv'
    output_file = '/home/user/Downloads/cai/GMAN/results/train_predictions_fixed.csv'
    
    if not os.path.exists(gman_file):
        print(f"❌ 文件不存在: {gman_file}")
        return
    
    print(f"\n 加载文件: {gman_file}")
    
    # 读取CSV文件
    df = pd.read_csv(gman_file, encoding='utf-8-sig')
    
    print(f"   原始列名: {df.columns.tolist()}")
    
    # 定义列名映射（韩文 -> 中文）
    column_mapping = {
        '小客车上行_预测값': '小客车上行_预测值',
        '小客车上行_真实값': '小客车上行_真实值',
        '小客车下行_预测값': '小客车下行_预测值',
        '小客车下行_真实값': '小客车下行_真实值',
        '非小客车上行_预测값': '非小客车上行_预测值',
        '非小客车上行_真实값': '非小客车上行_真实值',
        '非小客车下行_预测값': '非小客车下行_预测值',
        '非小客车下行_真实값': '非小客车下行_真实值'
    }
    
    # 重命名列
    df.rename(columns=column_mapping, inplace=True)
    
    print(f"\n✅ 列名已修复")
    print(f"   新列名: {df.columns.tolist()}")
    
    # 保存修复后的文件
    print(f"\n💾 保存到: {output_file}")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 修复完成!")
    print(f"   文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    print(f"   行数: {len(df)}")
    
    # 备份原文件
    backup_file = gman_file + '.backup'
    if not os.path.exists(backup_file):
        print(f"\n📦 备份原文件到: {backup_file}")
        os.rename(gman_file, backup_file)
    
    # 替换原文件
    print(f"\n🔄 替换原文件: {gman_file}")
    os.rename(output_file, gman_file)
    
    print(f"\n{'='*80}")
    print(f"✅ 所有操作完成！")
    print(f"{'='*80}")

if __name__ == '__main__':
    fix_gman_column_names()
