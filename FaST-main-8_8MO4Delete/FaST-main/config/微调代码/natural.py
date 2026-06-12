import pandas as pd
from chinese_calendar import is_workday

# 1. 加载数据
print("正在读取数据...")
df = pd.read_csv('subway.csv', index_col=0, parse_dates=True)

# 筛选时间段 (根据需要修改)
df = df.loc['2022-02-01 08:00:00':'2022-06-7 19:30:00']

# 2. 识别节假日
print("正在识别节假日...")
unique_dates = pd.Series(df.index.date).unique()
workday_map = {d: is_workday(d) for d in unique_dates}
df['is_actual_workday'] = [workday_map[d] for d in df.index.date]

# 3. 分割工作日和休息日
workday_df = df[df['is_actual_workday'] == True].copy()
holiday_df = df[df['is_actual_workday'] == False].copy()

# 4. 逐个站点分析
stations = [col for col in df.columns if col != 'is_actual_workday']
txt_lines = []

print(f"开始分析共 {len(stations)} 个站点并生成结果...")

for station in stations:
    # --- 工作日分析 ---
    if not workday_df.empty:
        w_data = workday_df[[station]].copy()
        w_data['Time'] = w_data.index.time
        w_avg = w_data.groupby('Time')[station].mean()

        w_range = f"{int(workday_df[station].min())}-{int(workday_df[station].max())}"

        # 高峰数据
        w_peak_t = w_avg.idxmax()
        w_peak_v = int(round(w_avg.max()))

        # 低峰数据 (新增平均值)
        w_low_t = w_avg.idxmin()
        w_low_v = int(round(w_avg.min()))
    else:
        w_range, w_peak_t, w_peak_v, w_low_t, w_low_v = "0-0", "N/A", 0, "N/A", 0

    # --- 休息日分析 ---
    if not holiday_df.empty:
        h_data = holiday_df[[station]].copy()
        h_data['Time'] = h_data.index.time
        h_avg = h_data.groupby('Time')[station].mean()

        h_range = f"{int(holiday_df[station].min())}-{int(holiday_df[station].max())}"

        # 高峰数据
        h_peak_t = h_avg.idxmax()
        h_peak_v = int(round(h_avg.max()))

        # 低峰数据 (新增平均值)
        h_low_t = h_avg.idxmin()
        h_low_v = int(round(h_avg.min()))
    else:
        h_range, h_peak_t, h_peak_v, h_low_t, h_low_v = "0-0", "N/A", 0, "N/A", 0

    # --- 整合文本 (加入 average off-peak flow) ---
    line = (f"Workday: passenger flow range {w_range}, peak time {w_peak_t}, "
            f"average peak flow {w_peak_v}, low-peak time {w_low_t}, average low-peak flow {w_low_v}; "
            f"Off-day: passenger flow range {h_range}, peak time {h_peak_t}, "
            f"average peak flow {h_peak_v}, low-peak time {h_low_t}, average low-peak flow {h_low_v}.")
    txt_lines.append(line)

# 5. 写入 TXT 文件
with open('station_analysis_report.txt', 'w', encoding='utf-8') as f:
    for line in txt_lines:
        f.write(line + '\n')

print("\n分析成功完成！结果已包含低峰时刻平均值并保存至: station_analysis_report.txt")