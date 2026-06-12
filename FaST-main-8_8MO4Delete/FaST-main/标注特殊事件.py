# python 标注特殊事件.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ===================== 配置区域 =====================
# 输入文件路径
INPUT_CSV = "prediction_with_section_name.csv"
OUTPUT_DIR = "error_analysis_results"

# 误差阈值配置（百分比）
TOP_ERROR_PERCENTILES = [5, 10]  # 分析前5%和前10%的误差样本

# 绝对误差和相对误差阈值（用于筛选显著误差）
ABS_ERROR_THRESHOLD = 30  # 绝对误差阈值
REL_ERROR_THRESHOLD = 50  # 相对误差阈值(%)

# ===================== 1. 加载数据 & 格式校验 =====================
print("="*80)
print("📂 正在加载数据...")
print("="*80)

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"找不到输入文件: {INPUT_CSV}")

# 先尝试读取文件查看列名
df_temp = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", nrows=5)
print(f"\n📋 CSV文件前5行数据预览:")
print(df_temp.head())
print(f"\n📋 CSV文件实际列名:")
for i, col in enumerate(df_temp.columns):
    print(f"  列{i}: '{col}'")
print()

# 判断是否有正确的列名
has_header = all(col in df_temp.columns for col in ["站点", "断面名", "时间"])

if has_header:
    # 有标准列名，直接读取
    print("✅ 检测到标准列名，使用列名映射模式")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    
    # 自动识别列名映射关系
    column_mapping = {
        "站点": ["站点", "station", "node", "node_name", "收费站", "收费站名"],
        "断面名": ["断面名", "断面", "section", "section_name", "断面名称"],
        "时间": ["时间", "time", "datetime", "timestamp", "日期时间"],
        "小客车真实值": ["小客车真实值", "小客车_真实值", "car_true", "car_actual", "客车真实值", "car_real"],
        "小客车预测值": ["小客车预测值", "小客车_预测值", "car_pred", "car_predicted", "客车预测值", "car_predict"],
        "非小客车真实值": ["非小客车真实值", "非小客车_真实值", "truck_true", "truck_actual", "货车真实值", "truck_real", "非客车真实值"],
        "非小客车预测值": ["非小客车预测值", "非小客车_预测值", "truck_pred", "truck_predicted", "货车预测值", "truck_predict", "非客车预测值"],
    }
    
    # 自动匹配列名
    actual_columns = {}
    for std_name, variants in column_mapping.items():
        matched = False
        for variant in variants:
            if variant in df.columns:
                actual_columns[std_name] = variant
                matched = True
                break
        
        if not matched:
            print(f"⚠️  警告: 未找到 '{std_name}' 对应的列")
            print(f"   支持的列名变体: {variants}")
    
    # 检查是否所有必需列都找到了
    missing_cols = set(column_mapping.keys()) - set(actual_columns.keys())
    if missing_cols:
        print(f"\n❌ 错误: 缺少以下必需列: {missing_cols}")
        print(f"\n请检查CSV文件，确保包含所有必需的列。")
        print(f"当前找到的列映射:")
        for std_name, actual_name in actual_columns.items():
            print(f"  {std_name} <- {actual_name}")
        raise ValueError("CSV列名不匹配！")
    
    # 使用标准列名创建副本
    df = df.rename(columns={v: k for k, v in actual_columns.items()})
    print(f"✅ 列名映射成功！")
    
else:
    # 没有标准列名，按列位置读取
    print("⚠️  未检测到标准列名，使用列位置模式")
    print("请确认CSV文件的列顺序如下：")
    print("  列0: 站点")
    print("  列1: 断面名")
    print("  列2: 时间")
    print("  列3: 小客车真实值")
    print("  列4: 小客车预测值")
    print("  列5: 非小客车真实值")
    print("  列6: 非小客车预测值")
    print()
    
    # 按位置读取，指定列名
    df = pd.read_csv(
        INPUT_CSV, 
        encoding="utf-8-sig",
        header=None,  # 没有表头
        names=["站点", "断面名", "时间", "小客车真实值", "小客车预测值", "非小客车真实值", "非小客车预测值"]
    )
    print("✅ 已按列位置读取数据")

print()
# 转换时间列为datetime格式
df["时间_dt"] = pd.to_datetime(df["时间"])

print(f"✅ 数据加载完成！")
print(f"总样本数：{len(df)}")
print(f"时间范围：{df['时间_dt'].min()} ~ {df['时间_dt'].max()}")
print(f"站点数：{df['站点'].nunique()}")
print(f"断面数：{df['断面名'].nunique()}")
print("="*80 + "\n")

# ===================== 2. 创建输出目录 =====================
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📁 输出目录: {OUTPUT_DIR}\n")

# ===================== 3. 计算误差（小客车/非小客车分开计算） =====================
print("="*80)
print("🔢 正在计算误差...")
print("="*80)

# 小客车误差
df["小客车_绝对误差"] = np.abs(df["小客车真实值"] - df["小客车预测值"])
df["小客车_相对误差(%)"] = (df["小客车_绝对误差"] / (df["小客车真实值"] + 1e-8)) * 100

# 非小客车（货车）误差
df["非小客车_绝对误差"] = np.abs(df["非小客车真实值"] - df["非小客车预测值"])
df["非小客车_相对误差(%)"] = (df["非小客车_绝对误差"] / (df["非小客车真实值"] + 1e-8)) * 100

# 综合误差（取两种车型中较大的相对误差）
df["最大相对误差(%)"] = df[["小客车_相对误差(%)", "非小客车_相对误差(%)"]].max(axis=1)
df["最大绝对误差"] = df[["小客车_绝对误差", "非小客车_绝对误差"]].max(axis=1)

print(f"✅ 误差计算完成！")
print(f"小客车平均相对误差: {df['小客车_相对误差(%)'].mean():.2f}%")
print(f"非小客车平均相对误差: {df['非小客车_相对误差(%)'].mean():.2f}%")
print(f"最大相对误差: {df['最大相对误差(%)'].max():.2f}%")
print("="*80 + "\n")

# ===================== 4. 保存完整误差数据 =====================
print("💾 正在保存完整误差数据...")
full_error_df = df[[
    "站点", "断面名", "时间", "时间_dt",
    "小客车真实值", "小客车预测值", "小客车_绝对误差", "小客车_相对误差(%)",
    "非小客车真实值", "非小客车预测值", "非小客车_绝对误差", "非小客车_相对误差(%)",
    "最大相对误差(%)", "最大绝对误差"
]].copy()

full_error_df.to_csv(f"{OUTPUT_DIR}/all_errors.csv", index=False, encoding="utf-8-sig")
print(f"✅ 完整误差数据已保存: {OUTPUT_DIR}/all_errors.csv\n")

# ===================== 5. 定义特殊事件数据库 =====================
print("="*80)
print("📅 正在构建特殊事件数据库...")
print("="*80)

def build_special_events_db(year=2023):
    """
    构建特殊事件数据库
    包括：节假日、免费通行、交通管制、施工、事故多发期、恶劣天气、政策整治等
    注意：数据时间粒度为每小时（00:00-23:00），事件边界需对齐到整点
    """
    events = []
    
    # ========== 5.1 国家法定节假日（高速免费/限行）==========
    holidays = {
        "元旦": [(f"{year}-01-01 00:00:00", f"{year}-01-03 23:00:00")],
        "春节": [(f"{year}-01-21 00:00:00", f"{year}-01-27 23:00:00")],  # 2023年春节
        "清明节": [(f"{year}-04-05 00:00:00", f"{year}-04-07 23:00:00")],
        "劳动节": [(f"{year}-05-01 00:00:00", f"{year}-05-03 23:00:00")],
        "端午节": [(f"{year}-06-22 00:00:00", f"{year}-06-24 23:00:00")],
        "中秋节": [(f"{year}-09-29 00:00:00", f"{year}-10-01 23:00:00")],  # 2023年中秋国庆连休
        "国庆节": [(f"{year}-10-01 00:00:00", f"{year}-10-07 23:00:00")],  # 国庆黄金周
    }
    
    for holiday_name, date_ranges in holidays.items():
        for start_time, end_time in date_ranges:
            events.append({
                "事件类型": "节假日",
                "事件名称": f"{holiday_name}假期",
                "开始时间": pd.to_datetime(start_time),
                "结束时间": pd.to_datetime(end_time),
                "影响说明": f"{holiday_name}期间，车流量大幅增加，可能出现拥堵"
            })
    
    # ========== 5.2 高速免费通行政策 ==========
    free_pass_periods = [
        ("春节免费", f"{year}-01-21 00:00:00", f"{year}-01-27 23:00:00"),
        ("清明节免费", f"{year}-04-05 00:00:00", f"{year}-04-07 23:00:00"),
        ("劳动节免费", f"{year}-05-01 00:00:00", f"{year}-05-03 23:00:00"),
        ("国庆节免费", f"{year}-10-01 00:00:00", f"{year}-10-07 23:00:00"),
    ]
    
    for event_name, start, end in free_pass_periods:
        events.append({
            "事件类型": "免费通行",
            "事件名称": event_name,
            "开始时间": pd.to_datetime(start),
            "结束时间": pd.to_datetime(end),
            "影响说明": "高速免费通行期间，车流量激增，可能导致拥堵和预测偏差"
        })
    
    # ========== 5.3 货车限行时段 ==========
    truck_restrictions = [
        ("中秋国庆货车限行", f"{year}-09-29 00:00:00", f"{year}-10-07 23:00:00", "中秋国庆连休期间，部分地区实施货车限行"),
        ("春节货车限行", f"{year}-01-20 00:00:00", f"{year}-01-28 23:00:00", "春节期间部分路段货车限行"),
    ]
    
    for event_name, start, end, desc in truck_restrictions:
        events.append({
            "事件类型": "交通管制",
            "事件名称": event_name,
            "开始时间": pd.to_datetime(start),
            "结束时间": pd.to_datetime(end),
            "影响说明": desc
        })
    
    # ========== 5.4 恶劣天气季节 ==========
    weather_events = [
        ("梅雨季节", f"{year}-06-01 00:00:00", f"{year}-07-15 23:00:00", "梅雨季节，持续降雨影响行车速度和车流量"),
        ("台风季节", f"{year}-07-01 00:00:00", f"{year}-09-30 23:00:00", "台风季节，可能出现暴雨、大风等极端天气"),
        ("冬季雾霾", f"{year}-11-01 00:00:00", f"{year+1}-02-28 23:00:00", "冬季雾霾天气，能见度低，影响交通"),
        ("高温酷暑", f"{year}-07-01 00:00:00", f"{year}-08-31 23:00:00", "夏季高温，可能引发车辆故障和交通事故"),
    ]
    
    for event_name, start, end, desc in weather_events:
        events.append({
            "事件类型": "天气影响",
            "事件名称": event_name,
            "开始时间": pd.to_datetime(start),
            "结束时间": pd.to_datetime(end),
            "影响说明": desc
        })
    
    # ========== 5.5 重大活动期间 ==========
    major_events = [
        ("春运期间", f"{year}-01-07 00:00:00", f"{year}-02-15 23:00:00", "春运期间，返乡和返程车流集中"),
        ("高考期间", f"{year}-06-07 00:00:00", f"{year}-06-09 23:00:00", "高考期间，部分路段交通管制"),
        ("双十一物流高峰", f"{year}-11-01 00:00:00", f"{year}-11-20 23:00:00", "双十一购物节，物流运输高峰，货车流量增加"),
    ]
    
    for event_name, start, end, desc in major_events:
        events.append({
            "事件类型": "重大活动",
            "事件名称": event_name,
            "开始时间": pd.to_datetime(start),
            "结束时间": pd.to_datetime(end),
            "影响说明": desc
        })
    
    # ========== 5.6 道路施工期（示例，需要根据实际情况调整）==========
    construction_periods = [
        ("春季道路养护", f"{year}-03-01 00:00:00", f"{year}-04-30 23:00:00", "春季道路养护施工，部分路段封闭或限速"),
        ("秋季道路维修", f"{year}-09-01 00:00:00", f"{year}-10-31 23:00:00", "秋季道路维修，可能影响通行能力"),
    ]
    
    for event_name, start, end, desc in construction_periods:
        events.append({
            "事件类型": "道路施工",
            "事件名称": event_name,
            "开始时间": pd.to_datetime(start),
            "结束时间": pd.to_datetime(end),
            "影响说明": desc
        })
    
    # ========== 5.7 政策整治期 ==========
    policy_events = [
        ("环保督察", f"{year}-03-01 00:00:00", f"{year}-03-31 23:00:00", "环保督察期间，货车运输受限"),
        ("交通安全整治", f"{year}-05-01 00:00:00", f"{year}-05-31 23:00:00", "交通安全专项整治，加强执法检查"),
        ("超限超载治理", f"{year}-08-01 00:00:00", f"{year}-08-31 23:00:00", "超限超载专项治理，货车检查严格"),
    ]
    
    for event_name, start, end, desc in policy_events:
        events.append({
            "事件类型": "政策整治",
            "事件名称": event_name,
            "开始时间": pd.to_datetime(start),
            "结束时间": pd.to_datetime(end),
            "影响说明": desc
        })
    
    # 转换为DataFrame
    events_df = pd.DataFrame(events)
    return events_df

events_db = build_special_events_db(year=2023)
print(f"✅ 特殊事件数据库构建完成！共 {len(events_db)} 个事件")
print(f"事件类型分布:\n{events_db['事件类型'].value_counts()}")
print("="*80 + "\n")

# ===================== 6. 为每个样本标注特殊事件 =====================
print("="*80)
print("🏷️  正在为样本标注特殊事件...")
print("="*80)

def find_matching_events(sample_time, events_df):
    """查找某个时间点匹配的所有特殊事件"""
    matched_events = events_df[
        (events_df["开始时间"] <= sample_time) & 
        (events_df["结束时间"] >= sample_time)
    ]
    
    if len(matched_events) == 0:
        return "无特殊事件"
    
    # 合并多个事件
    event_list = []
    for _, event in matched_events.iterrows():
        event_str = f"{event['事件类型']}-{event['事件名称']}"
        event_list.append(event_str)
    
    return "; ".join(event_list)

def find_event_details(sample_time, events_df):
    """获取详细的事件信息"""
    matched_events = events_df[
        (events_df["开始时间"] <= sample_time) & 
        (events_df["结束时间"] >= sample_time)
    ]
    
    if len(matched_events) == 0:
        return {
            "事件类型": "无",
            "事件名称": "无",
            "影响说明": "无特殊事件"
        }
    
    # 返回第一个匹配事件的详细信息
    first_event = matched_events.iloc[0]
    return {
        "事件类型": first_event["事件类型"],
        "事件名称": first_event["事件名称"],
        "影响说明": first_event["影响说明"]
    }

# 为所有样本标注事件（由于数据量大，这里采用批量处理）
print("正在标注事件（这可能需要几分钟）...")
df["特殊事件"] = df["时间_dt"].apply(lambda x: find_matching_events(x, events_db))

# 提取主要事件类型和名称
event_details = df["时间_dt"].apply(lambda x: find_event_details(x, events_db))
df["主要事件类型"] = event_details.apply(lambda x: x["事件类型"])
df["主要事件名称"] = event_details.apply(lambda x: x["事件名称"])
df["事件影响说明"] = event_details.apply(lambda x: x["影响说明"])

print(f"✅ 事件标注完成！")
print(f"有事件的样本数: {(df['主要事件类型'] != '无').sum()}")
print(f"无事件的样本数: {(df['主要事件类型'] == '无').sum()}")
print("="*80 + "\n")

# ===================== 7. 按误差百分位筛选高误差样本 =====================
print("="*80)
print("📊 正在分析高误差样本...")
print("="*80)

for percentile in TOP_ERROR_PERCENTILES:
    print(f"\n{'='*80}")
    print(f"🔍 分析前{percentile}%高误差样本")
    print(f"{'='*80}")
    
    # 计算误差阈值
    error_threshold = np.percentile(df["最大相对误差(%)"], 100 - percentile)
    print(f"误差阈值（前{percentile}%）: {error_threshold:.2f}%")
    
    # 筛选高误差样本
    high_error_samples = df[df["最大相对误差(%)"] >= error_threshold].copy()
    print(f"高误差样本数: {len(high_error_samples)} (占总样本 {len(high_error_samples)/len(df)*100:.2f}%)")
    
    # 按误差降序排序
    high_error_samples = high_error_samples.sort_values(by="最大相对误差(%)", ascending=False)
    
    # 统计事件分布
    print(f"\n📋 特殊事件分布（前{percentile}%高误差样本）:")
    event_dist = high_error_samples["主要事件类型"].value_counts()
    for event_type, count in event_dist.items():
        pct = count / len(high_error_samples) * 100
        print(f"  - {event_type}: {count} 个 ({pct:.2f}%)")
    
    # 统计站点分布
    print(f"\n📍 高误差站点Top 20:")
    station_dist = high_error_samples["站点"].value_counts().head(20)
    for station, count in station_dist.items():
        print(f"  - {station}: {count} 个高误差样本")
    
    # 统计时间段分布（按小时）
    print(f"\n⏰ 高误差时间段分布（按小时）:")
    high_error_samples["小时"] = high_error_samples["时间_dt"].dt.hour
    hour_dist = high_error_samples["小时"].value_counts().sort_index()
    for hour, count in hour_dist.items():
        bar = "█" * (count // max(hour_dist.max() // 20, 1))
        print(f"  {hour:02d}:00 - {count:4d} 个 {bar}")
    
    # 保存结果
    output_file = f"{OUTPUT_DIR}/top_{percentile}percent_high_errors.csv"
    high_error_output = high_error_samples[[
        "站点", "断面名", "时间", "时间_dt",
        "小客车真实值", "小客车预测值", "小客车_绝对误差", "小客车_相对误差(%)",
        "非小客车真实值", "非小客车预测值", "非小客车_绝对误差", "非小客车_相对误差(%)",
        "最大相对误差(%)", "最大绝对误差",
        "主要事件类型", "主要事件名称", "特殊事件", "事件影响说明"
    ]].copy()
    
    high_error_output.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n✅ 结果已保存: {output_file}")

print("\n" + "="*80)

# ===================== 8. 生成综合分析报表 =====================
print("\n" + "="*80)
print("📈 生成综合分析报表...")
print("="*80)

# 8.1 按事件类型统计误差
print("\n📊 不同事件类型下的平均误差:")
event_error_stats = df.groupby("主要事件类型").agg({
    "小客车_相对误差(%)": ["mean", "median", "std"],
    "非小客车_相对误差(%)": ["mean", "median", "std"],
    "最大相对误差(%)": ["mean", "median", "std"]
}).round(2)

print(event_error_stats.to_string())

# 8.2 保存综合统计报表
stats_report = []
for event_type in df["主要事件类型"].unique():
    mask = df["主要事件类型"] == event_type
    subset = df[mask]
    
    stats_report.append({
        "事件类型": event_type,
        "样本数": len(subset),
        "小客车平均误差(%)": subset["小客车_相对误差(%)"].mean(),
        "小客车中位数误差(%)": subset["小客车_相对误差(%)"].median(),
        "非小客车平均误差(%)": subset["非小客车_相对误差(%)"].mean(),
        "非小客车中位数误差(%)": subset["非小客车_相对误差(%)"].median(),
        "最大平均误差(%)": subset["最大相对误差(%)"].mean(),
        "最大中位数误差(%)": subset["最大相对误差(%)"].median(),
    })

stats_df = pd.DataFrame(stats_report)
stats_df.to_csv(f"{OUTPUT_DIR}/error_statistics_by_event.csv", index=False, encoding="utf-8-sig")
print(f"\n✅ 综合统计报表已保存: {OUTPUT_DIR}/error_statistics_by_event.csv")

# 8.3 生成高误差站点清单
print("\n🎯 高误差站点清单（误差最大的前50个站点）:")
station_error = df.groupby("站点").agg({
    "最大相对误差(%)": ["mean", "max", "count"],
    "小客车_相对误差(%)": "mean",
    "非小客车_相对误差(%)": "mean"
}).round(2)

station_error.columns = ["平均最大误差(%)", "最大误差(%)", "样本数", "小客车平均误差(%)", "非小客车平均误差(%)"]
station_error = station_error.sort_values(by="平均最大误差(%)", ascending=False).head(50)

print(station_error.to_string())
station_error.to_csv(f"{OUTPUT_DIR}/high_error_stations.csv", encoding="utf-8-sig")
print(f"\n✅ 高误差站点清单已保存: {OUTPUT_DIR}/high_error_stations.csv")

# ===================== 9. 生成总结报告 =====================
print("\n" + "="*80)
print("📝 生成总结报告...")
print("="*80)

summary_report = f"""
{'='*80}
                    车流量预测误差分析报告
{'='*80}

一、数据概况
-----------
- 总样本数: {len(df)}
- 时间范围: {df['时间_dt'].min()} ~ {df['时间_dt'].max()}
- 站点数量: {df['站点'].nunique()}
- 断面数量: {df['断面名'].nunique()}

二、整体误差统计
--------------
- 小客车平均相对误差: {df['小客车_相对误差(%)'].mean():.2f}%
- 小客车中位数相对误差: {df['小客车_相对误差(%)'].median():.2f}%
- 非小客车平均相对误差: {df['非小客车_相对误差(%)'].mean():.2f}%
- 非小客车中位数相对误差: {df['非小客车_相对误差(%)'].median():.2f}%

三、高误差样本分析
----------------
"""

for percentile in TOP_ERROR_PERCENTILES:
    threshold = np.percentile(df["最大相对误差(%)"], 100 - percentile)
    high_error_count = (df["最大相对误差(%)"] >= threshold).sum()
    summary_report += f"- 前{percentile}%高误差样本数: {high_error_count} (阈值: {threshold:.2f}%)\n"

summary_report += f"""
四、特殊事件影响分析
------------------
"""

event_types = df["主要事件类型"].value_counts()
for event_type, count in event_types.items():
    subset = df[df["主要事件类型"] == event_type]
    avg_error = subset["最大相对误差(%)"].mean()
    summary_report += f"- {event_type}: {count} 个样本, 平均误差 {avg_error:.2f}%\n"

summary_report += f"""
五、输出文件清单
--------------
1. all_errors.csv - 所有样本的完整误差数据
2. top_5percent_high_errors.csv - 前5%高误差样本及事件标注
3. top_10percent_high_errors.csv - 前10%高误差样本及事件标注
4. error_statistics_by_event.csv - 按事件类型统计的误差
5. high_error_stations.csv - 高误差站点清单
6. analysis_summary.txt - 分析总结报告

六、建议
-------
1. 重点关注节假日、免费通行期间的预测偏差
2. 对高误差站点进行单独建模或特征工程
3. 考虑引入天气、事故等实时数据作为模型输入
4. 针对特殊事件建立专门的修正机制

{'='*80}
报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

# 保存总结报告
with open(f"{OUTPUT_DIR}/analysis_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary_report)

print(summary_report)
print(f"\n✅ 总结报告已保存: {OUTPUT_DIR}/analysis_summary.txt")

print("\n" + "="*80)
print("🎉 分析完成！所有结果已保存到:", OUTPUT_DIR)
print("="*80)