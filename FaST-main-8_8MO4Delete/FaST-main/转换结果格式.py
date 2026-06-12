# python 转换结果格式.py
import pandas as pd

# ---------------------- 1. 读取原始CSV文件 ----------------------
# 请将这里替换成你真实的文件路径
input_path = "prediction_with_section_name.csv"
df = pd.read_csv(input_path)

# 打印原始数据结构，方便你核对
print("✅ 原始数据前10行：")
print(df.head(10))
print("\n✅ 原始列名：", df.columns.tolist())

# ---------------------- 2. 数据透视转换（核心步骤） ----------------------
# 检测实际列名
if "特征" not in df.columns:
    raise KeyError(f"❌ 未找到'特征'列！当前列名: {df.columns.tolist()}")

print(f"\n📊 检测到的特征类型: {df['特征'].unique().tolist()}")

# 执行数据透视转换
df_pivot = df.pivot_table(
    index=["站点名称", "断面名", "时间"],  # 行：站点 + 断面名 + 时间（唯一组合）
    columns="特征",                        # 列：按特征类型拆分
    values=["真实值", "预测值"],           # 要展开的值
    aggfunc="first"                       # 每个组合只有一个值，直接取第一个
)

# 重命名列名，构建清晰的列名结构
# 原始格式: ('真实值', 'LittleCar_Up') -> 新格式: '小客车上行_真实值'
new_columns = []
for value_type, feature_name in df_pivot.columns:
    # 映射英文特征名到中文
    feature_mapping = {
        'LittleCar_Up': '小客车上行',
        'LittleCar_Down': '小客车下行',
        'NonLittleCar_Up': '非小客车上行',
        'NonLittleCar_Down': '非小客车下行'
    }
    
    cn_feature = feature_mapping.get(feature_name, feature_name)
    new_col_name = f"{cn_feature}_{value_type}"
    new_columns.append(new_col_name)

df_pivot.columns = new_columns

# 重置索引，把站点、断面名、时间变回普通列
df_final = df_pivot.reset_index()

# 调整列顺序，组织为更清晰的格式
final_columns = [
    "站点名称",
    "断面名",
    "时间",
    # 小客车特征
    "小客车上行_真实值",
    "小客车上行_预测值",
    "小客车下行_真实值",
    "小客车下行_预测值",
    # 非小客车特征
    "非小客车上行_真实值",
    "非小客车上行_预测值",
    "非小客车下行_真实值",
    "非小客车下行_预测值"
]

# 验证所有列都存在
missing_cols = [col for col in final_columns if col not in df_final.columns]
if missing_cols:
    print(f"\n⚠️  警告：以下列不存在: {missing_cols}")
    # 只保留实际存在的列
    final_columns = [col for col in final_columns if col in df_final.columns]

df_final = df_final[final_columns]

# ---------------------- 3. 保存新文件 ----------------------
output_path = "prediction_formatted.csv"
df_final.to_csv(output_path, index=False, encoding="utf-8-sig")

# ---------------------- 4. 打印结果确认 ----------------------
print("\n" + "="*100)
print("✅ 转换完成！新文件已保存为：", output_path)
print("\n✅ 转换后数据前5行：")
print(df_final.head())
print("\n✅ 最终列名：")
for i, col in enumerate(df_final.columns, 1):
    print(f"  {i}. {col}")
print(f"\n✅ 数据形状: {df_final.shape}")
print(f"   - 总行数: {df_final.shape[0]}")
print(f"   - 总列数: {df_final.shape[1]}")