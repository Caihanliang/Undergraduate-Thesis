# # python 处理小货车时间断面号.py
# # 真正断面号形式
# import pandas as pd
# import numpy as np

# def process_traffic_to_wide(excel_path, output_path="小客车流量_宽表.csv", year=2023):
#     """
#     最终版：
#     行 = 时间
#     列 = 原始断面名（不修改，直接用）
#     值 = 小客车流量
#     自动删除节假日、非小客车列
#     """
#     # 1. 读取Excel
#     print("正在读取数据...")
#     df = pd.read_excel(excel_path)

#     # 2. 自动匹配列
#     section_col = [c for c in df.columns if "断面" in str(c)][0]
#     month_col = [c for c in df.columns if "月" in str(c)][0]
#     day_col = [c for c in df.columns if "日" in str(c)][0]
#     hour_col = [c for c in df.columns if "小时" in str(c)][0]
#     # car_col = [c for c in df.columns if "小客车" in str(c)][0]
#     car_col = [c for c in df.columns if "非小客车" in str(c)][0]


#     # 3. 生成标准时间列
#     print("正在生成时间...")
#     df["Time"] = pd.to_datetime(
#         {"year": year, "month": df[month_col], "day": df[day_col], "hour": df[hour_col]}
#     )

#     # 4. 只保留需要的3列
#     df = df[["Time", section_col, car_col]].copy()

#     # 5. 清洗
#     df = df.dropna()
#     df[car_col] = pd.to_numeric(df[car_col], errors="coerce")
#     df = df[df[car_col] >= 0]

#     # 6. 长表转宽表（核心）
#     print("正在生成宽表...")
#     df_wide = df.pivot_table(
#         index="Time",
#         columns=section_col,
#         values=car_col,
#         aggfunc="first"
#     ).reset_index()

#     # 7. 时间格式化
#     df_wide["Time"] = df_wide["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")

#     # 8. 保存
#     df_wide.to_csv(output_path, index=False, encoding="utf-8-sig")

#     # 输出结果
#     print("="*60)
#     print("✅ 处理完成！最终格式如下：")
#     print(f"行数(时间点)：{len(df_wide)}")
#     print(f"列数(断面数+时间)：{len(df_wide.columns)}")
#     print("最终列名：", list(df_wide.columns[:5]), "...")
#     print("文件已保存至：", output_path)
#     print("="*60)

#     return df_wide


# # ===================== 直接在这里改你的文件路径 =====================
# if __name__ == "__main__":
#     INPUT_EXCEL = "LittleCar.xlsx"   # 改成你自己的文件路径
#     OUTPUT_CSV = "NLittleCar真正名.csv"
#     DATA_YEAR = 2023

#     df_result = process_traffic_to_wide(INPUT_EXCEL, OUTPUT_CSV, DATA_YEAR)



# # ===================== 剔除6月数据 =====================
import pandas as pd

def filter_june_data(input_path, output_path="LittleCar真正名9_10.csv"):
    """
    从宽表流量数据中，自动剔除所有6月的数据，保留其他月份
    输入格式：Time列 + 各断面流量列（和你截图一致）
    输出格式：完全相同的宽表，仅删除6月的行
    """
    # 1. 读取数据
    print("正在读取数据...")
    df = pd.read_csv(input_path)
    print(f"原始数据行数：{len(df)}")
    print(f"原始时间范围：{df['Time'].min()} ~ {df['Time'].max()}")

    # 2. 转换时间列并筛选
    print("\n正在剔除6月数据...")
    # 转换为datetime类型
    df["Time_dt"] = pd.to_datetime(df["Time"])
    # 筛选：月份 != 6
    df_filtered = df[df["Time_dt"].dt.month != 6].copy()

    # 3. 恢复原时间格式，删除辅助列
    df_filtered["Time"] = df_filtered["Time_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_filtered = df_filtered.drop(columns=["Time_dt"])

    # 4. 保存结果
    print("\n正在保存结果...")
    df_filtered.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 5. 输出处理报告
    print("\n" + "="*60)
    print("✅ 数据处理完成！")
    print(f"原始数据行数：{len(df)}")
    print(f"剔除6月后行数：{len(df_filtered)}")
    print(f"新时间范围：{df_filtered['Time'].min()} ~ {df_filtered['Time'].max()}")
    print(f"文件已保存至：{output_path}")
    print("="*60)

    return df_filtered

# ===================== 直接修改这里的路径 =====================
if __name__ == "__main__":
    # 输入文件：你当前的宽表CSV路径
    INPUT_CSV = "NLittleCar真正名.csv"
    # 输出文件：剔除6月后的新CSV
    OUTPUT_CSV = "NLittleCar真正名9_10.csv"

    df_result = filter_june_data(INPUT_CSV, OUTPUT_CSV)
    # 打印前5行预览
    print("\n📊 处理后数据预览（前5行）：")
    print(df_result.head())