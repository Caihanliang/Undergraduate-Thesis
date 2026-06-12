#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
删除指定站点编号的数据行
目标文件：
- 观测站小时交通量-9.csv
- 观测站小时交通量-10.csv
要删除的站点：G60L003430281, S80L030430521, G55L160431126, S80L020430421
python DataPipeline/remove_specific_stations.py
FaST-main-8_8MO4Delete/FaST-main/DataPipeline/remove_specific_stations.py
"""

import pandas as pd
import os
import shutil
from datetime import datetime

# ========== 配置参数 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 输入文件
INPUT_FILES = [
    "观测站小时交通量-9.csv",
    "观测站小时交通量-10.csv"
]

# 要删除的站点编号
STATIONS_TO_DELETE = [
    "G60L003430281",
    "S80L030430521",
    "G55L160431126",
    "S80L020430421"
]


def remove_stations_from_csv(file_path, stations_to_delete):
    """从CSV文件中删除指定站点的数据"""
    print(f"\n{'=' * 80}")
    print(f"📂 处理文件: {os.path.basename(file_path)}")
    print(f"{'=' * 80}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"  ❌ 文件不存在，跳过")
        return None
    
    # 加载原始数据
    print(f"\n  1️⃣  加载原始数据...")
    df_original = pd.read_csv(file_path)
    original_rows = len(df_original)
    print(f"     原始数据量: {original_rows:,} 行")
    
    # 检查是否有"观测站编号"列
    station_column = None
    possible_columns = ['观测站编号', '站点编号', 'station_id', 'StationID', '站编号']
    
    for col in possible_columns:
        if col in df_original.columns:
            station_column = col
            print(f"     找到站点列: {col}")
            break
    
    if station_column is None:
        print(f"     ❌ 未找到站点编号列！当前列名: {list(df_original.columns)}")
        return None
    
    # 统计要删除的站点数据量
    print(f"\n  2️⃣  统计要删除的站点数据...")
    stations_found = []
    
    for station in stations_to_delete:
        count = (df_original[station_column] == station).sum()
        if count > 0:
            stations_found.append((station, count))
            print(f"     ✓ 站点 {station}: {count:,} 行")
        else:
            print(f"     - 站点 {station}: 未找到")
    
    if not stations_found:
        print(f"\n  ⚠️  警告: 未找到任何要删除的站点数据")
        return df_original
    
    # 执行删除
    print(f"\n  3️⃣  删除指定站点数据...")
    mask = ~df_original[station_column].isin(stations_to_delete)
    df_filtered = df_original[mask].copy()
    
    removed_rows = original_rows - len(df_filtered)
    print(f"     删除数据量: {removed_rows:,} 行")
    print(f"     剩余数据量: {len(df_filtered):,} 行")
    print(f"     删除比例: {removed_rows / original_rows * 100:.2f}%")
    
    return df_filtered


def save_filtered_data(df, original_file_path, stations_to_delete):
    """保存过滤后的数据（带备份）"""
    print(f"\n  4️⃣  保存过滤后的数据...")
    
    # 创建备份
    backup_path = original_file_path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy2(original_file_path, backup_path)
        print(f"     ✓ 已创建备份: {os.path.basename(backup_path)}")
    else:
        print(f"     - 备份已存在，跳过创建")
    
    # 保存过滤后的数据
    output_path = original_file_path.replace(".csv", "_filtered.csv")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"     ✓ 已保存: {os.path.basename(output_path)} ({file_size:.2f} MB)")
    
    return output_path


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🚀 开始删除指定站点数据")
    print("=" * 80)
    print(f"\n要删除的站点编号 ({len(STATIONS_TO_DELETE)} 个):")
    for i, station in enumerate(STATIONS_TO_DELETE, 1):
        print(f"  {i}. {station}")
    
    results = []
    
    try:
        for file_name in INPUT_FILES:
            file_path = os.path.join(BASE_DIR, file_name)
            
            # 删除指定站点数据
            df_filtered = remove_stations_from_csv(file_path, STATIONS_TO_DELETE)
            
            if df_filtered is not None:
                # 保存结果
                output_path = save_filtered_data(df_filtered, file_path, STATIONS_TO_DELETE)
                results.append({
                    'original': file_name,
                    'filtered': output_path,
                    'original_rows': len(pd.read_csv(file_path)),
                    'filtered_rows': len(df_filtered)
                })
        
        # 打印总结
        print(f"\n{'=' * 80}")
        print("📊 处理总结")
        print(f"{'=' * 80}")
        
        print(f"\n{'原始文件':<30} {'原始行数':<15} {'过滤后行数':<15} {'删除行数':<15}")
        print(f"{'─' * 75}")
        
        total_original = 0
        total_filtered = 0
        
        for result in results:
            original = result['original']
            original_rows = result['original_rows']
            filtered_rows = result['filtered_rows']
            removed = original_rows - filtered_rows
            
            total_original += original_rows
            total_filtered += filtered_rows
            
            print(f"{original:<30} {original_rows:<15,} {filtered_rows:<15,} {removed:<15,}")
        
        print(f"{'─' * 75}")
        print(f"{'合计':<30} {total_original:<15,} {total_filtered:<15,} {total_original - total_filtered:<15,}")
        
        print(f"\n{'=' * 80}")
        print("🎉 处理完成！")
        print(f"{'=' * 80}")
        print(f"\n输出文件:")
        for result in results:
            print(f"  - {result['filtered']}")
        
        print(f"\n⚠️  重要提示:")
        print(f"  1. 原始文件已备份为 .backup 后缀")
        print(f"  2. 过滤后的文件名包含 _filtered 后缀")
        print(f"  3. 请检查过滤结果后再替换原始文件")
        print(f"  4. 如需替换原始文件，可手动执行:")
        print(f"     mv {os.path.basename(INPUT_FILES[0]).replace('.csv', '_filtered.csv')} {INPUT_FILES[0]}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
