#!/usr/bin/env python3
"""
检查三个数据集中站点的顺序和索引是否一致
"""

import pandas as pd
import os

# 定义三个数据集的路径
FAST_DATASET_PATH = "/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main/DataPipeline"
MOMENT_DATASET_PATH = "/home/user/Downloads/cai/moment-main/moment-main/dataset"
LAG_LLAMA_DATASET_PATH = "/home/user/Downloads/cai/lag-llama-main/dataset"

def load_and_analyze_dataset(file_path, dataset_name):
    """加载并分析数据集"""
    print(f"\n{'='*80}")
    print(f"📊 分析 {dataset_name} 数据集")
    print(f"{'='*80}")
    
    # 读取9月份的数据
    file_9 = os.path.join(file_path, "观测站小时交通量-9.csv")
    df = pd.read_csv(file_9)
    
    print(f"文件路径: {file_9}")
    print(f"总行数: {len(df)}")
    print(f"列名: {list(df.columns)}")
    
    # 提取唯一的站点信息
    stations = df[['观测站编号', '观测站名称']].drop_duplicates()
    stations = stations.sort_values('观测站编号').reset_index(drop=True)
    
    print(f"\n站点总数: {len(stations)}")
    print("\n站点列表 (前20个):")
    print("-" * 80)
    for idx, row in stations.head(20).iterrows():
        print(f"索引 {idx:3d}: 编号={row['观测站编号']:20s} | 名称={row['观测站名称']}")
    
    if len(stations) > 20:
        print(f"... (共 {len(stations)} 个站点)")
        print("\n最后10个站点:")
        print("-" * 80)
        for idx, row in stations.tail(10).iterrows():
            print(f"索引 {idx:3d}: 编号={row['观测站编号']:20s} | 名称={row['观测站名称']}")
    
    # 检查10月份的数据是否包含相同的站点
    file_10 = os.path.join(file_path, "观测站小时交通量-10.csv")
    if os.path.exists(file_10):
        df_10 = pd.read_csv(file_10)
        stations_10 = df_10[['观测站编号', '观测站名称']].drop_duplicates()
        stations_10 = stations_10.sort_values('观测站编号').reset_index(drop=True)
        
        print(f"\n✓ 10月份数据站点数: {len(stations_10)}")
        
        # 比较两个月份的站点是否一致
        if len(stations) == len(stations_10):
            # 检查站点顺序是否一致
            match = (stations['观测站编号'].values == stations_10['观测站编号'].values).all()
            if match:
                print("✓ 9月和10月的站点顺序完全一致")
            else:
                print("⚠️ 9月和10月的站点顺序不一致!")
                # 找出不一致的站点
                mismatches = stations[stations['观测站编号'] != stations_10['观测站编号']]
                print(f"不匹配的站点数: {len(mismatches)}")
        else:
            print(f"⚠️ 9月({len(stations)})和10月({len(stations_10)})站点数不一致!")
    
    return stations

def compare_datasets(fast_stations, moment_stations, lag_llama_stations):
    """比较三个数据集的站点"""
    print(f"\n{'='*80}")
    print("🔍 跨数据集站点一致性检查")
    print(f"{'='*80}")
    
    # 检查站点数量
    print(f"\n站点数量对比:")
    print(f"  FaST:      {len(fast_stations)} 个站点")
    print(f"  MOMENT:    {len(moment_stations)} 个站点")
    print(f"  Lag-Llama: {len(lag_llama_stations)} 个站点")
    
    # 检查站点编号是否一致
    fast_ids = set(fast_stations['观测站编号'].values)
    moment_ids = set(moment_stations['观测站编号'].values)
    lag_llama_ids = set(lag_llama_stations['观测站编号'].values)
    
    print(f"\n站点编号集合对比:")
    print(f"  FaST ∩ MOMENT:    {len(fast_ids & moment_ids)} 个共同站点")
    print(f"  FaST ∩ Lag-Llama: {len(fast_ids & lag_llama_ids)} 个共同站点")
    print(f"  MOMENT ∩ Lag-Llama: {len(moment_ids & lag_llama_ids)} 个共同站点")
    print(f"  三者交集:          {len(fast_ids & moment_ids & lag_llama_ids)} 个共同站点")
    
    # 检查是否有差异
    if fast_ids != moment_ids:
        print(f"\n⚠️ FaST 和 MOMENT 站点不完全一致:")
        only_in_fast = fast_ids - moment_ids
        only_in_moment = moment_ids - fast_ids
        if only_in_fast:
            print(f"  仅在FaST中: {len(only_in_fast)} 个站点")
            for sid in sorted(list(only_in_fast))[:5]:
                print(f"    - {sid}")
        if only_in_moment:
            print(f"  仅在MOMENT中: {len(only_in_moment)} 个站点")
            for sid in sorted(list(only_in_moment))[:5]:
                print(f"    - {sid}")
    
    if fast_ids != lag_llama_ids:
        print(f"\n⚠️ FaST 和 Lag-Llama 站点不完全一致:")
        only_in_fast = fast_ids - lag_llama_ids
        only_in_lag_llama = lag_llama_ids - fast_ids
        if only_in_fast:
            print(f"  仅在FaST中: {len(only_in_fast)} 个站点")
            for sid in sorted(list(only_in_fast))[:5]:
                print(f"    - {sid}")
        if only_in_lag_llama:
            print(f"  仅在Lag-Llama中: {len(only_in_lag_llama)} 个站点")
            for sid in sorted(list(only_in_lag_llama))[:5]:
                print(f"    - {sid}")
    
    # 检查站点顺序是否一致
    if len(fast_stations) == len(moment_stations) == len(lag_llama_stations):
        fast_order = fast_stations['观测站编号'].values
        moment_order = moment_stations['观测站编号'].values
        lag_llama_order = lag_llama_stations['观测站编号'].values
        
        fast_moment_match = (fast_order == moment_order).all()
        fast_lag_match = (fast_order == lag_llama_order).all()
        
        print(f"\n站点顺序一致性:")
        print(f"  FaST vs MOMENT:    {'✓ 完全一致' if fast_moment_match else '✗ 不一致'}")
        print(f"  FaST vs Lag-Llama: {'✓ 完全一致' if fast_lag_match else '✗ 不一致'}")
        
        if not fast_moment_match:
            print("\n  前10个站点顺序对比:")
            print("  " + "-" * 70)
            print(f"  {'索引':<5} {'FaST编号':<25} {'MOMENT编号':<25}")
            print("  " + "-" * 70)
            for i in range(min(10, len(fast_stations))):
                match_symbol = "✓" if fast_order[i] == moment_order[i] else "✗"
                print(f"  {i:<5} {fast_order[i]:<25} {moment_order[i]:<25} {match_symbol}")
    
    return fast_ids == moment_ids == lag_llama_ids

def check_finetune_config():
    """检查微调代码中的配置"""
    print(f"\n{'='*80}")
    print("🔧 检查微调代码配置")
    print(f"{'='*80}")
    
    finetune_script = "/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main/微调513.py"
    
    if not os.path.exists(finetune_script):
        print(f"❌ 微调脚本不存在: {finetune_script}")
        return
    
    with open(finetune_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取关键配置
    print("\n📋 微调脚本关键配置:")
    
    # 查找站点列表文件路径
    import re
    
    carpark_pattern = r'CARPARK_DES_PATH\s*=\s*os\.path\.join\([^)]+\)'
    match = re.search(carpark_pattern, content)
    if match:
        print(f"  站点列表配置: {match.group()}")
    
    # 查找站点数量相关配置
    num_stations_patterns = [
        r'num_stations\s*=',
        r'len\(carpark_des_list\)',
        r'station_list_hngs',
        r'_98\.txt',
        r'_161\.txt',
    ]
    
    print("\n  站点数量相关配置:")
    for pattern in num_stations_patterns:
        matches = re.findall(pattern + r'\s*[^,\n]*', content)
        if matches:
            for m in matches[:2]:  # 只显示前2个匹配
                print(f"    - {m.strip()}")
    
    # 检查实际使用的站点列表文件
    station_files = [
        "station_list_hngs.txt",
        "station_list_hngs_98.txt",
        "station_list_hngs_161.txt",
    ]
    
    config_dir = "/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main/config/cai/"
    print(f"\n  配置文件目录: {config_dir}")
    print("  存在的站点列表文件:")
    for sf in station_files:
        sf_path = os.path.join(config_dir, sf)
        if os.path.exists(sf_path):
            with open(sf_path, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]
            print(f"    ✓ {sf}: {len(lines)} 个站点")
        else:
            print(f"    ✗ {sf}: 不存在")

if __name__ == "__main__":
    print("="*80)
    print("🚀 开始检查三个数据集的站点一致性")
    print("="*80)
    
    # 分析每个数据集
    fast_stations = load_and_analyze_dataset(FAST_DATASET_PATH, "FaST")
    moment_stations = load_and_analyze_dataset(MOMENT_DATASET_PATH, "MOMENT")
    lag_llama_stations = load_and_analyze_dataset(LAG_LLAMA_DATASET_PATH, "Lag-Llama")
    
    # 比较三个数据集
    all_consistent = compare_datasets(fast_stations, moment_stations, lag_llama_stations)
    
    # 检查微调配置
    check_finetune_config()
    
    # 总结
    print(f"\n{'='*80}")
    print("📝 总结")
    print(f"{'='*80}")
    if all_consistent:
        print("✅ 三个数据集的站点完全一致（数量和顺序都相同）")
    else:
        print("⚠️ 三个数据集的站点存在差异，请检查上述详细信息")
    
    print("\n💡 建议:")
    print("  1. 确保所有模型使用相同的数据集文件")
    print("  2. 微调代码中的站点列表文件必须与数据集实际包含的站点匹配")
    print("  3. 如果站点数量不同，需要重新生成或调整数据集")
