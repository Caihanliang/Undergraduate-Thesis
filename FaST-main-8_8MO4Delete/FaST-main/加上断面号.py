# python 加上断面号.py
import pandas as pd

# ---------------------- 1. 读取两个文件 ----------------------
# 第一个文件：预测结果CSV
pred_path = "all_nodes_prediction.csv"  # 替换为你的第一个文件路径
pred_df = pd.read_csv(pred_path)

# 第二个文件：原始断面数据CSV
section_path = "观测站小时交通量-9.csv"  # 替换为你的第二个文件路径
section_df = pd.read_csv(section_path)

# 打印原始列名，方便核对
print("✅ 第一个文件（预测结果）列名：", pred_df.columns.tolist())
print("✅ 第二个文件（断面数据）列名：", section_df.columns.tolist())

# ---------------------- 2. 自适应检测列名 ----------------------
# 检测预测结果文件中的站点列名
if "站点名称" in pred_df.columns:
    station_col = "站点名称"
elif "站点" in pred_df.columns:
    station_col = "站点"
else:
    raise KeyError(f"❌ 未找到站点列！当前列名: {pred_df.columns.tolist()}")

print(f"\n✅ 检测到预测结果中的站点列为: '{station_col}'")

# ---------------------- 3. 构建 node → 断面名 的映射关系 ----------------------
# 步骤1：从第二个文件中提取【唯一的断面名列表】（按出现顺序）
# 注意：需要排除已删除的4个站点
deleted_stations = [
    "G60L003430281",  # 零值最多的站点
    "S80L030430521",
    "G55L160431126",
    "S80L020430421"
]

print(f"\n🗑️  将从断面数据中排除已删除的 {len(deleted_stations)} 个站点:")
for station in deleted_stations:
    print(f"    - {station}")

# 过滤掉已删除的站点
section_df_filtered = section_df[~section_df["观测站编号"].isin(deleted_stations)].copy()
print(f"\n✅ 过滤后的断面数据: {len(section_df_filtered)} 行 (原始 {len(section_df)} 行)")

# 提取唯一的断面名列表（按出现顺序）
unique_sections = section_df_filtered["观测站名称"].unique().tolist()

print(f"\n✅ 第二个文件中共有 {len(unique_sections)} 个唯一断面名（排除已删除站点后）：")
print(unique_sections[:10])  # 打印前10个，方便核对

# 步骤2：从第一个文件中提取【唯一的站点列表】（按出现顺序）
unique_nodes = pred_df[station_col].unique().tolist()
print(f"\n✅ 第一个文件中共有 {len(unique_nodes)} 个唯一站点：")
print(unique_nodes[:10])  # 打印前10个，方便核对

# 步骤3：按顺序一一对应
if len(unique_nodes) != len(unique_sections):
    print(f"\n⚠️  警告：站点数({len(unique_nodes)})与断面名数({len(unique_sections)})仍然不一致！")
    print(f"    请检查数据是否正确处理。")
    # 创建空映射以避免后续报错
    node_section_map = {}
else:
    # 构建映射字典
    node_section_map = dict(zip(unique_nodes, unique_sections))
    print(f"\n✅ 已构建 {len(node_section_map)} 个 node 与 断面名 的对应关系")
    print("✅ 映射示例（前5个）：")
    for i, (node, section) in enumerate(node_section_map.items()):
        if i < 5:
            print(f"  {node} → {section}")

# ---------------------- 4. 给第一个文件新增「断面名」列 ----------------------
# 用映射字典匹配站点列，生成「断面名」
pred_df["断面名"] = pred_df[station_col].map(node_section_map)

# 检查是否有未匹配的站点
unmatched_nodes = pred_df[pred_df["断面名"].isna()][station_col].unique()
if len(unmatched_nodes) > 0:
    print(f"\n⚠️  注意：以下 {len(unmatched_nodes)} 个站点未匹配到断面名：")
    print(unmatched_nodes[:5])
else:
    print("\n✅ 所有站点都成功匹配到断面名！")

# ---------------------- 5. 调整列顺序（把「断面名」放在站点列后面） ----------------------
current_columns = pred_df.columns.tolist()
station_index = current_columns.index(station_col)
# 插入「断面名」到站点列之后
new_columns = (
    current_columns[:station_index+1]  # 站点及之前的列
    + ["断面名"]  # 新增的断面名列
    + current_columns[station_index+1:]  # 站点之后的其他列
)
pred_df = pred_df[new_columns]

# ---------------------- 6. 保存最终结果 ----------------------
output_path = "prediction_with_section_name.csv"
pred_df.to_csv(output_path, index=False, encoding="utf-8-sig")

# ---------------------- 7. 打印最终结果确认 ----------------------
print("\n" + "="*100)
print(f"✅ 任务完成！带断面名的文件已保存为：{output_path}")
print("\n✅ 最终文件前5行：")
print(pred_df.head())
print("\n✅ 最终文件列名顺序：")
print(pred_df.columns.tolist())