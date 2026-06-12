# 统计CSV文件中数据为0的个数及对应站点
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

def count_zero_values_in_csv(file_path, output_detail=False):
    """
    统计单个CSV文件中各站点的0值个数
    :param file_path: CSV文件路径（需包含Time列和站点列）
    :param output_detail: 是否输出详细统计结果
    :return: 0值统计字典（站点: 0值个数）
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(file_path, encoding='gbk')
    
    # 分离时间列和站点列（假设第一列为Time，其余为站点）
    time_col = df.columns[0]
    station_cols = df.columns[1:]  # 所有站点列
    
    print(f"{'='*60}")
    print(f"📊 正在分析文件：{os.path.basename(file_path)}")
    print(f"⏰ 时间列：{time_col}")
    print(f"🏁 站点总数：{len(station_cols)}")
    print(f"📅 数据总行数（时间粒度）：{len(df)}")
    
    # 统计每个站点的0值个数
    zero_count_dict = {}
    total_zero_count = 0  # 全局0值总数
    
    for station in station_cols:
        # 统计当前站点的0值个数
        zero_count = (df[station] == 0).sum()
        zero_count_dict[station] = zero_count
        total_zero_count += zero_count
        
        # 计算0值占比
        zero_ratio = (zero_count / len(df)) * 100
        
        # 输出详细信息（按需求开关）
        if output_detail:
            print(f"  {station:<10} | 0值个数: {zero_count:>4} | 占比: {zero_ratio:>5.2f}%")
    
    # 输出汇总信息
    total_data_count = len(df) * len(station_cols)  # 总数据单元格数
    global_zero_ratio = (total_zero_count / total_data_count) * 100
    
    print(f"\n📈 汇总统计：")
    print(f"  全局0值总数：{total_zero_count:,}")
    print(f"  总数据单元格数：{total_data_count:,}")
    print(f"  全局0值占比：{global_zero_ratio:.2f}%")
    
    # 找出0值最多的前10个站点
    sorted_zero_stations = sorted(zero_count_dict.items(), key=lambda x: x[1], reverse=True)
    print(f"\n🔴 0值最多的前10个站点：")
    for i, (station, count) in enumerate(sorted_zero_stations[:10], 1):
        ratio = (count / len(df)) * 100
        print(f"  {i:2d}. {station:<10} | 0值个数: {count:>4} | 占比: {ratio:>5.2f}%")
    
    # 找出无0值的站点
    no_zero_stations = [station for station, count in zero_count_dict.items() if count == 0]
    if no_zero_stations:
        print(f"\n🟢 无0值的站点（共{len(no_zero_stations)}个）：")
        # 按每10个站点换行显示
        for i in range(0, len(no_zero_stations), 10):
            print(f"  {' | '.join(no_zero_stations[i:i+10])}")
    else:
        print(f"\n🟢 无0值的站点：无")
    
    return zero_count_dict

def count_zero_values_in_folder(folder_path, output_detail=False):
    """
    统计文件夹中所有CSV文件的0值情况
    :param folder_path: 文件夹路径
    :param output_detail: 是否输出详细统计结果
    """
    # 遍历文件夹中的CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"❌ 在文件夹 {folder_path} 中未找到CSV文件")
        return
    
    print(f"{'='*80}")
    print(f"📂 开始分析文件夹：{folder_path}")
    print(f"📄 找到CSV文件总数：{len(csv_files)}")
    print(f"{'='*80}\n")
    
    # 逐个分析CSV文件
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        count_zero_values_in_csv(file_path, output_detail=output_detail)
        print(f"\n")  # 分隔不同文件的结果

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    # --------------------------
    # 配置区域（请根据实际情况修改）
    # --------------------------
    # 方式1：分析单个CSV文件（推荐，针对你上传的"小客车上行流量.csv"）
    TARGET_FILE = "小客车上行流量.csv"  # 你的CSV文件路径（当前目录直接写文件名）
    # TARGET_FILE = "小客车下行流量.csv"  # 你的CSV文件路径（当前目录直接写文件名）
    # TARGET_FILE = "(汽车自然数-小客车)上行流量.csv"  # 你的CSV文件路径（当前目录直接写文件名）
    # TARGET_FILE = "(汽车自然数-小客车)下行流量.csv"  # 你的CSV文件路径（当前目录直接写文件名）
    # TARGET_FILE = "观测站小时交通量-9.csv"  # 你的CSV文件路径（当前目录直接写文件名）

    # FaST-main-8_8MO4/Test/观测站小时交通量-9.csv
    
    
    # 方式2：分析整个文件夹中的所有CSV（如需批量处理，启用下面两行）
    # TARGET_FOLDER = "./"  # 目标文件夹路径（当前目录为"./"）
    # output_detail_mode = False  # 是否输出每个站点的详细0值统计（True/False）
    
    # --------------------------
    # 执行统计（二选一）
    # --------------------------
    # 1. 统计单个文件（推荐）
    if os.path.exists(TARGET_FILE):
        count_zero_values_in_csv(TARGET_FILE, output_detail=True)
    else:
        print(f"❌ 找不到文件：{TARGET_FILE}，请检查文件路径是否正确")
    
    # 2. 统计整个文件夹（批量处理时启用）
    # count_zero_values_in_folder(TARGET_FOLDER, output_detail=output_detail_mode)
    # python 统计站点的缺失值.py