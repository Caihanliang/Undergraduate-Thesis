import pandas as pd
import chardet  # 用于自动检测文件编码
import os

def detect_csv_encoding(file_path: str) -> str:
    """
    自动检测CSV文件的编码格式（解决“不知道原始编码”的问题）
    :param file_path: CSV文件路径
    :return: 检测到的编码（如'gbk'、'utf-8'、'latin-1'）
    """
    with open(file_path, 'rb') as f:
        # 读取文件前10000字节（足够检测编码，避免读取大文件耗时）
        raw_data = f.read(10000)
        # 自动检测编码
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']  # 检测置信度
        print(f"✅ 检测到文件编码：{encoding}（置信度：{confidence:.2f}）")
        return encoding

def fix_csv_encoding(
    input_file: str, 
    output_file: str, 
    target_encoding: str = 'utf-8-sig',
    sep: str = ','  # CSV分隔符，默认逗号，若为制表符可改为'\t'
) -> None:
    """
    修复CSV文件编码，转为目标编码（默认UTF-8-SIG，兼容所有系统）
    :param input_file: 原始CSV文件路径
    :param output_file: 修复后输出文件路径
    :param target_encoding: 目标编码（默认utf-8-sig）
    :param sep: CSV文件分隔符
    """
    # 1. 检测原始编码
    raw_encoding = detect_csv_encoding(input_file)
    
    # 2. 读取原始文件（用检测到的编码，避免乱码）
    try:
        df = pd.read_csv(
            input_file,
            encoding=raw_encoding,
            sep=sep,
            dtype=str  # 避免读取时数据类型自动转换（如时间、数值格式改变）
        )
        print(f"✅ 成功读取文件，共 {len(df)} 行数据，{len(df.columns)} 列")
    except Exception as e:
        print(f"❌ 读取文件失败：{str(e)}")
        print("💡 尝试用'gbk'或'latin-1'编码读取（常见兼容编码）")
        # 备选编码：若自动检测失败，尝试常见编码
        for backup_encoding in ['gbk', 'latin-1', 'utf-8']:
            try:
                df = pd.read_csv(input_file, encoding=backup_encoding, sep=sep, dtype=str)
                print(f"✅ 用备份编码 {backup_encoding} 成功读取")
                break
            except:
                continue
        else:
            raise Exception("❌ 所有编码尝试失败，请检查文件是否损坏")
    
    # 3. 保存为目标编码（UTF-8-SIG）
    try:
        df.to_csv(
            output_file,
            encoding=target_encoding,
            index=False,  # 不保存行索引
            sep=sep
        )
        print(f"✅ 编码修复完成！输出文件：{output_file}")
        print(f"🔍 目标编码：{target_encoding}（兼容Excel/Python/Notepad）")
    except Exception as e:
        raise Exception(f"❌ 保存文件失败：{str(e)}")

def verify_fixed_csv(output_file: str) -> None:
    """
    验证修复后的文件是否正常（无乱码、数据完整）
    :param output_file: 修复后的CSV文件路径
    """
    print("\n" + "="*50)
    print("🔍 开始验证修复后的文件")
    # 1. 读取修复后的文件（UTF-8-SIG编码）
    df = pd.read_csv(output_file, encoding='utf-8-sig')
    # 2. 打印基本信息
    print(f"✅ 验证通过：文件可正常读取")
    print(f"📊 数据规模：{len(df)} 行 × {len(df.columns)} 列")
    print(f"🏷️  列名：{list(df.columns)}")
    # 3. 打印前3行数据（直观查看是否有乱码）
    print("\n📄 前3行数据预览：")
    print(df.head(3).to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    # -------------------------- 配置参数（根据你的文件修改） --------------------------
    INPUT_CSV_PATH = "cai-config/ture/station_events_list.csv"    # 原始文件路径
    OUTPUT_CSV_PATH = "cai-config/ture/station_events_list.csv"  # 输出文件路径
    # ---------------------------------------------------------------------------------
    
    # 执行编码修复流程
    fix_csv_encoding(
        input_file=INPUT_CSV_PATH,
        output_file=OUTPUT_CSV_PATH,
        target_encoding='utf-8-sig'  # 固定目标编码为UTF-8-SIG
    )
    
    # 验证修复结果
    verify_fixed_csv(output_file=OUTPUT_CSV_PATH)
    # python cai-config/ture/修正编码.py