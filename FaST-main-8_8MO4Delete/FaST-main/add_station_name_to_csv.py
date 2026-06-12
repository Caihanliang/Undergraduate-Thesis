#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 his_data.csv 中添加站点索引列
根据 station_mapping.json，将站点编号替换为对应的索引（0,1,2...）
输出：[时间, 站点编号, 站点索引, 小客车上行, ...]
"""

import pandas as pd
import json
import os

# ========== 直接使用绝对路径（100% 不报错） ==========
CSV_PATH = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/main-master/datasets/HNGS_4FEAT/his_data.csv"
MAPPING_PATH = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/main-master/datasets/HNGS_4FEAT/station_mapping.json"
OUTPUT_CSV = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/main-master/datasets/HNGS_4FEAT/his_data_with_index.csv"


def load_station_mapping():
    """加载站点编号到索引的映射"""
    print("=" * 60)
    print("加载站点映射关系")
    print("=" * 60)

    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    station_to_idx = mapping.get("station_to_idx", {})
    print(f"✓ 加载 {len(station_to_idx)} 个站点映射关系")

    return station_to_idx


def add_station_index_column():
    """添加站点索引列（编号 → 索引）"""
    station_to_idx = load_station_mapping()

    # 加载 CSV
    print(f"\n📂 加载数据: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"✓ 数据规模: {len(df)} 行")

    if '站点编号' not in df.columns:
        raise ValueError("未找到'站点编号'列")

    # ===== 核心：站点编号 → 索引 =====
    print("\n正在将站点编号转换为索引...")
    df['站点索引'] = df['站点编号'].map(station_to_idx)

    # 把“站点名称”列直接变成索引（用户要求）
    df['站点名称'] = df['站点索引']

    print(f"✓ 转换完成！新增列：站点索引")
    print(f"  示例前5行：")
    print(df[['时间', '站点编号', '站点名称']].head())

    # 保存
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n💾 已保存: {OUTPUT_CSV}")

    return df


def main():
    print("\n🎉 开始执行：站点编号 → 索引")
    df = add_station_index_column()
    print("\n✅ 处理完成！")


if __name__ == "__main__":
    main()