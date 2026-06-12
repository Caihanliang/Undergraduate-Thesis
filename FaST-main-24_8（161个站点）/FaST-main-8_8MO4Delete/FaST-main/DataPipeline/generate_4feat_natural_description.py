#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为高速公路站点4特征数据生成自然语言描述
用于大模型微调数据集构建

输入：his_data_with_index.csv（包含时间、站点编号、4个特征、站点索引）
输出：station_natural_list_4feat.txt（每个站点的4个特征分别描述）

特征定义：
  - 小客车上行
  - 小客车下行
  - 非小客车上行
  - 非小客车下行

输出格式（每行一个特征）：
Workday: traffic flow range X-Y, peak time HH:MM:SS, average peak flow Z, low-peak time HH:MM:SS, average low-peak flow W; 
Off-day: traffic flow range A-B, peak time HH:MM:SS, average peak flow C, low-peak time HH:MM:SS, average low-peak flow D.
python DataPipeline/generate_4feat_natural_description.py
"""

import pandas as pd
import os
from datetime import datetime

# ========== 配置参数 ==========
# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 项目根目录
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 输入文件
INPUT_CSV = os.path.join(PROJECT_ROOT, "config", "cai", "his_data_with_index.csv")

# 输出目录和文件
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "config", "cai")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "station_natural_list_4feat.txt")

# 特征配置
FEATURES = [
    {"name": "小客车上行", "col": "小客车上行"},
    {"name": "小客车下行", "col": "小客车下行"},
    {"name": "非小客车上行", "col": "非小客车上行"},
    {"name": "非小客车下行", "col": "非小客车下行"}
]


def load_data():
    """加载并预处理数据"""
    print("=" * 60)
    print("步骤1: 加载数据")
    print("=" * 60)
    
    print(f"📂 读取文件: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    print(f"  ✓ 加载完成: {len(df)} 行")
    print(f"  列名: {list(df.columns)}")
    
    # 转换时间列
    df['时间'] = pd.to_datetime(df['时间'])
    
    # 提取日期和时间
    df['日期'] = df['时间'].dt.date
    df['时刻'] = df['时间'].dt.time
    
    # 判断工作日（简单规则：周一到周五为工作日）
    df['is_workday'] = df['时间'].dt.weekday < 5
    
    print(f"  时间范围: {df['时间'].min()} ~ {df['时间'].max()}")
    print(f"  站点数: {df['站点编号'].nunique()}")
    print(f"  工作日数据: {df['is_workday'].sum()} 行")
    print(f"  休息日数据: {(~df['is_workday']).sum()} 行")
    
    return df


def analyze_feature(df, station_id, feature_name, feature_col):
    """分析单个站点的单个特征"""
    
    # 提取该站点该特征的数据
    station_data = df[df['站点编号'] == station_id].copy()
    
    result = {
        "station": station_id,
        "feature": feature_name
    }
    
    # 工作日分析
    workday_data = station_data[station_data['is_workday'] == True]
    if not workday_data.empty:
        w_avg = workday_data.groupby('时刻')[feature_col].mean()
        
        w_min = int(workday_data[feature_col].min())
        w_max = int(workday_data[feature_col].max())
        w_range = f"{w_min}-{w_max}"
        
        # 高峰：平均值最大的时刻
        w_peak_times = w_avg[w_avg == w_avg.max()].index.tolist()
        w_peak_values = [int(round(w_avg.max()))] * len(w_peak_times)
        
        # 低峰：平均值最小的时刻
        w_low_times = w_avg[w_avg == w_avg.min()].index.tolist()
        w_low_values = [int(round(w_avg.min()))] * len(w_low_times)
        
        result['workday'] = {
            'range': w_range,
            'peak_times': w_peak_times,
            'peak_values': w_peak_values,
            'low_times': w_low_times,
            'low_values': w_low_values
        }
    else:
        result['workday'] = None
    
    # 休息日分析
    holiday_data = station_data[station_data['is_workday'] == False]
    if not holiday_data.empty:
        h_avg = holiday_data.groupby('时刻')[feature_col].mean()
        
        h_min = int(holiday_data[feature_col].min())
        h_max = int(holiday_data[feature_col].max())
        h_range = f"{h_min}-{h_max}"
        
        # 高峰
        h_peak_times = h_avg[h_avg == h_avg.max()].index.tolist()
        h_peak_values = [int(round(h_avg.max()))] * len(h_peak_times)
        
        # 低峰
        h_low_times = h_avg[h_avg == h_avg.min()].index.tolist()
        h_low_values = [int(round(h_avg.min()))] * len(h_low_times)
        
        result['holiday'] = {
            'range': h_range,
            'peak_times': h_peak_times,
            'peak_values': h_peak_values,
            'low_times': h_low_times,
            'low_values': h_low_values
        }
    else:
        result['holiday'] = None
    
    return result


def format_description(result, feature_name):
    """将分析结果格式化为自然语言描述"""
    
    # 工作日描述
    if result['workday']:
        w = result['workday']
        peak_times_str = ", ".join([str(t) for t in w['peak_times']])
        peak_values_str = ", ".join([str(v) for v in w['peak_values']])
        low_times_str = ", ".join([str(t) for t in w['low_times']])
        low_values_str = ", ".join([str(v) for v in w['low_values']])
        
        workday_desc = (
            f"Workday: traffic flow range {w['range']}, "
            f"peak time {peak_times_str}, average peak flow {peak_values_str}, "
            f"low-peak time {low_times_str}, average low-peak flow {low_values_str}"
        )
    else:
        workday_desc = "Workday: no data"
    
    # 休息日描述
    if result['holiday']:
        h = result['holiday']
        peak_times_str = ", ".join([str(t) for t in h['peak_times']])
        peak_values_str = ", ".join([str(v) for v in h['peak_values']])
        low_times_str = ", ".join([str(t) for t in h['low_times']])
        low_values_str = ", ".join([str(v) for v in h['low_values']])
        
        holiday_desc = (
            f"Off-day: traffic flow range {h['range']}, "
            f"peak time {peak_times_str}, average peak flow {peak_values_str}, "
            f"low-peak time {low_times_str}, average low-peak flow {low_values_str}"
        )
    else:
        holiday_desc = "Off-day: no data"
    
    # 添加特征名称标识（关键！）
    return f"[{feature_name}] {workday_desc}; {holiday_desc}."


def generate_natural_descriptions(df):
    """为所有站点的所有特征生成自然语言描述"""
    print("\n" + "=" * 60)
    print("步骤2: 生成自然语言描述")
    print("=" * 60)
    
    stations = sorted(df['站点编号'].unique())
    print(f"\n  处理站点数: {len(stations)}")
    print(f"  每个站点4个特征，共 {len(stations) * 4} 行描述\n")
    
    all_descriptions = []
    
    for idx, station in enumerate(stations):
        for feature in FEATURES:
            # 分析该站点该特征
            result = analyze_feature(df, station, feature['name'], feature['col'])
            
            # 格式化描述（传入特征名称）
            desc = format_description(result, feature['name'])
            
            # 添加站点和特征信息作为注释（可选）
            all_descriptions.append(desc)
        
        # 进度显示
        if (idx + 1) % 10 == 0 or idx == len(stations) - 1:
            print(f"  进度: {idx + 1}/{len(stations)} 站点 ({(idx+1)/len(stations)*100:.1f}%)")
    
    return all_descriptions


def save_descriptions(descriptions):
    """保存描述到文件"""
    print("\n" + "=" * 60)
    print("步骤3: 保存结果")
    print("=" * 60)
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for desc in descriptions:
            f.write(desc + '\n')
    
    file_size = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"\n  ✓ 保存成功: {OUTPUT_FILE}")
    print(f"  文件大小: {file_size:.2f} KB")
    print(f"  总行数: {len(descriptions)}")
    
    # 打印前几行示例
    print(f"\n  前3行示例:")
    for i, desc in enumerate(descriptions[:3]):
        print(f"  [{i+1}] {desc[:100]}...")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("🚀 高速公路站点4特征自然语言描述生成器")
    print("=" * 60 + "\n")
    
    try:
        # 检查输入文件
        if not os.path.exists(INPUT_CSV):
            print(f"❌ 错误: 找不到输入文件 {INPUT_CSV}")
            return
        
        # 加载数据
        df = load_data()
        
        # 生成描述
        descriptions = generate_natural_descriptions(df)
        
        # 保存结果
        save_descriptions(descriptions)
        
        print("\n" + "=" * 60)
        print("🎉 处理完成！")
        print("=" * 60)
        print(f"\n输出文件: {OUTPUT_FILE}")
        print(f"\n用途说明:")
        print(f"  该文件包含 {len(descriptions)} 行自然语言描述")
        print(f"  每行对应一个站点的一个特征（工作日/休息日的流量模式）")
        print(f"  可用于大模型微调数据集构建")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
