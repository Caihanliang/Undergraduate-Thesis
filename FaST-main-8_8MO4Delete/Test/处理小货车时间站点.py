#  station 格式
import pandas as pd
import numpy as np
from datetime import datetime

def process_traffic_to_wide(excel_path, output_path="小客车流量_宽表格式.csv", year=2023):
    """
    把分月/日/小时的长表流量数据，转换成你要的宽表格式：
    行：时间（2023-09-01 00:00:00）
    列：每个断面（Station_001, Station_002...）
    值：对应断面的小客车流量
    
    Args:
        excel_path: 输入Excel文件路径（长表：断面名、月、日、小时、节假日、小客车流量、非小客车流量）
        output_path: 输出宽表CSV路径
        year: 数据年份（默认2023）
    """
    # ===================== 1. 读取原始长表数据 =====================
    print("正在读取原始Excel文件...")
    df = pd.read_excel(excel_path)
    print(f"原始长表形状：{df.shape}")
    print(f"原始列名：{list(df.columns)}")

    # ===================== 2. 合并月/日/小时为完整时间列 =====================
    print("\n正在生成完整时间列...")
    # 自动匹配列名
    month_col = [col for col in df.columns if "月" in str(col)][0]
    day_col = [col for col in df.columns if "日" in str(col)][0]
    hour_col = [col for col in df.columns if "小时" in str(col)][0]
    section_col = [col for col in df.columns if "断面" in str(col)][0]
    car_col = [col for col in df.columns if "小客车流量" in str(col)][0]

    # 生成完整时间（格式：2023-09-01 00:00:00，和你截图一致）
    df["Time"] = pd.to_datetime(
        {
            "year": year,
            "month": df[month_col],
            "day": df[day_col],
            "hour": df[hour_col]
        },
        errors="coerce"
    )

    # ===================== 3. 数据清洗 =====================
    print("\n正在清洗数据...")
    # 删除缺失值
    df = df.dropna(subset=["Time", section_col, car_col])
    # 转换流量为数值型
    df[car_col] = pd.to_numeric(df[car_col], errors="coerce")
    # 删除负流量异常值
    df = df[df[car_col] >= 0]
    # 按时间排序
    df = df.sort_values(by="Time").reset_index(drop=True)

    # ===================== 4. 长表转宽表（核心步骤） =====================
    print("\n正在长表转宽表...")
    # 透视表：行=时间，列=断面名，值=小客车流量
    df_wide = df.pivot_table(
        index="Time",          # 行：时间
        columns=section_col,   # 列：断面名
        values=car_col,        # 值：小客车流量
        aggfunc="first"        # 同一时间同一断面取第一个值（避免重复）
    ).reset_index()

    # ===================== 5. 格式化列名（Station_001 格式） =====================
    print("\n正在格式化列名...")
    # 给断面列统一命名为 Station_XXX 格式（和你截图完全一致）
    # 先获取所有断面列（排除Time列）
    station_cols = [col for col in df_wide.columns if col != "Time"]
    # 按顺序重命名：Station_001, Station_002, ...
    rename_map = {
        old_col: f"Station_{str(i+1).zfill(3)}" 
        for i, old_col in enumerate(station_cols)
    }
    df_wide = df_wide.rename(columns=rename_map)

    # ===================== 6. 格式化时间列 =====================
    # 时间格式：2023-09-01 00:00:00（和你截图完全一致）
    df_wide["Time"] = df_wide["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # ===================== 7. 保存宽表 =====================
    print("\n正在保存宽表文件...")
    df_wide.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"宽表已保存到：{output_path}")

    # ===================== 8. 输出处理报告 =====================
    print("\n" + "="*60)
    print("✅ 数据处理完成！最终宽表格式如下：")
    print(f"1. 原始长表：{df.shape[0]}行，{df.shape[1]}列")
    print(f"2. 最终宽表：{df_wide.shape[0]}行，{df_wide.shape[1]}列")
    print(f"3. 时间范围：{pd.to_datetime(df_wide['Time']).min()} ~ {pd.to_datetime(df_wide['Time']).max()}")
    print(f"4. 断面数量：{len(station_cols)}个（Station_001 ~ Station_{str(len(station_cols)).zfill(3)}）")
    print(f"5. 输出文件：{output_path}")
    print("="*60)

    # 打印前5行预览
    print("\n📊 最终宽表预览（前5行）：")
    print(df_wide.head())

    return df_wide

# ===================== 运行代码 =====================
if __name__ == "__main__":
    # 请修改为你的文件路径
    INPUT_EXCEL_PATH = "LittleCar.xlsx"  # 替换为你的Excel文件路径
    OUTPUT_CSV_PATH = "LittleCar.csv"
    DATA_YEAR = 2023  # 你的数据年份

    try:
        processed_df = process_traffic_to_wide(INPUT_EXCEL_PATH, OUTPUT_CSV_PATH, year=DATA_YEAR)
    except Exception as e:
        print(f"❌ 数据处理失败：{str(e)}")
# python 处理小货车时间站点.py