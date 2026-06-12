import pandas as pd
import os

def remove_zero_columns(input_csv_path, output_csv_path="DataPipeline/202309_10_512数据集_cleaned.csv", zero_threshold=0.99):
    """
    删除CSV文件中全0列/几乎全0列，并打印被删除的列名
    
    Args:
        input_csv_path: 输入CSV文件路径
        output_csv_path: 清洗后数据的输出路径
        zero_threshold: 零值占比阈值，超过该比例的列会被删除（默认99%，即99%为0就删）
    """
    # 1. 读取CSV文件（自动处理编码问题）
    print(f"正在读取数据：{input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path, encoding="utf-8-sig")
    except:
        df = pd.read_csv(input_csv_path, encoding="gbk")
    
    # 2. 查看原始数据基本信息
    print(f"\n原始数据形状：{df.shape}（行数：{df.shape[0]}, 列数：{df.shape[1]}）")
    print(f"前5个列名：{df.columns[:5].tolist()}...（共{len(df.columns)}列）")
    
    # 3. 识别需要删除的列（全0列 + 几乎全0列）
    delete_columns = []
    total_rows = len(df)  # 总数据行数
    
    for col in df.columns:
        # 排除时间列（第一列是Time，不参与删除判断）
        if col == "Time" or df[col].dtype == "datetime64[ns]":
            continue
        
        # 填充NaN为0，转float
        col_data = df[col].fillna(0).astype(float)
        # 计算零值数量和零值占比
        zero_count = (col_data == 0).sum()
        zero_ratio = zero_count / total_rows
        
        # 条件1：严格全0列（100%为0）
        # 条件2：几乎全0列（超过阈值，比如99%为0）
        if zero_ratio == 1.0 or zero_ratio >= zero_threshold:
            delete_columns.append(col)
            print(f"  列【{col}】：零值占比 {zero_ratio*100:.2f}% → 标记删除")
    
    # 4. 打印删除信息
    print(f"\n{'='*60}")
    if len(delete_columns) > 0:
        print(f"检测到 {len(delete_columns)} 个无效列（全0/几乎全0），将删除以下列：")
        # 分批打印（避免列太多刷屏）
        for i in range(0, len(delete_columns), 10):
            batch = delete_columns[i:i+10]
            print(f"  第{i//10+1}组：{' | '.join(batch)}")
    else:
        print("未检测到无效列，无需删除！")
    print(f"{'='*60}")
    
    # 5. 删除无效列
    df_cleaned = df.drop(columns=delete_columns, errors="ignore")
    
    # 6. 保存清洗后的数据（不覆盖原始文件，生成新文件）
    df_cleaned.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 清洗后数据已保存至：{output_csv_path}")
    print(f"清洗后数据形状：{df_cleaned.shape}（行数：{df_cleaned.shape[0]}, 列数：{df_cleaned.shape[1]}）")
    print(f"清洗后前5个列名：{df_cleaned.columns[:5].tolist()}...（共{len(df_cleaned.columns)}列）")
    
    return df_cleaned, delete_columns

# ===================== 主程序运行 =====================
if __name__ == "__main__":
    # 你的原始CSV文件路径（根据实际路径修改！）
    input_file = "DataPipeline/202309_10_512数据集.csv"
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 错误：文件 {input_file} 不存在！")
        print(f"请确认文件路径是否正确，当前目录下的CSV文件有：")
        for file in os.listdir():
            if file.endswith(".csv"):
                print(f"  - {file}")
    else:
        # 执行清洗（阈值99%：99%为0就删，可根据需求调整，比如0.95=95%为0）
        cleaned_df, deleted_cols = remove_zero_columns(input_file, zero_threshold=0.99)
# python 删除数据全为0的列.py