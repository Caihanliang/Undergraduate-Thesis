""""
python prepare_finetune_98_files.py
/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main/prepare_98_metadata.py

"""
import os
import json
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main/config/cai/"

# =========================
# 原始文件路径
# =========================
OLD_YTRUE_PATH = os.path.join(PROJECT_ROOT, "finetune_real_traffic.npz")
OLD_YPRED_PATH = os.path.join(PROJECT_ROOT, "finetune_data.npz")

OLD_STATION_LIST = os.path.join(PROJECT_ROOT, "station_list_hngs.txt")
OLD_PATTERN_LIST = os.path.join(PROJECT_ROOT, "station_natural_list_4feat.txt")
OLD_HIS_DATA = os.path.join(PROJECT_ROOT, "his_data_with_index.csv")
OLD_WEATHER_DIR = os.path.join(PROJECT_ROOT, "160站点天气信息")

# =========================
# 新文件路径
# =========================
NEW_YTRUE_PATH = os.path.join(PROJECT_ROOT, "finetune_real_traffic_98.npz")
NEW_YPRED_PATH = os.path.join(PROJECT_ROOT, "finetune_data_98.npz")

NEW_STATION_LIST = os.path.join(PROJECT_ROOT, "station_list_hngs_98.txt")
NEW_PATTERN_LIST = os.path.join(PROJECT_ROOT, "station_natural_list_4feat_98.txt")
NEW_HIS_DATA = os.path.join(PROJECT_ROOT, "his_data_with_index_98.csv")
NEW_WEATHER_DIR = os.path.join(PROJECT_ROOT, "98站点天气信息")

KEEP_ID_PATH = os.path.join(PROJECT_ROOT, "keep_station_ids_98.json")
REMOVE_ID_PATH = os.path.join(PROJECT_ROOT, "removed_station_ids_59.json")

NUM_ORIGINAL_STATIONS = 157
NUM_FEATURES = 4

# =========================
# 你删除的 59 个原始站点
# =========================
REMOVED_STATION_IDS = {
    60, 148, 111, 117, 133, 29, 17, 134, 152, 62,
    122, 61, 56, 115, 147, 151, 1, 76, 153, 140,
    15, 55, 89, 127, 121, 88, 35, 138, 149, 143,
    129, 116, 72, 126, 108, 16, 155, 95, 75, 73,
    40, 57, 86, 96, 66, 87, 41, 23, 74, 67,
    98, 118, 120, 105, 114, 144, 137, 136, 146
}

KEEP_STATION_IDS = [
    i for i in range(NUM_ORIGINAL_STATIONS)
    if i not in REMOVED_STATION_IDS
]

print("=" * 100)
print("开始生成 98 站点微调相关文件")
print("=" * 100)
print(f"删除站点数: {len(REMOVED_STATION_IDS)}")
print(f"保留站点数: {len(KEEP_STATION_IDS)}")
print(f"保留站点ID: {KEEP_STATION_IDS}")

assert len(REMOVED_STATION_IDS) == 59, f"删除站点数应为59，当前={len(REMOVED_STATION_IDS)}"
assert len(KEEP_STATION_IDS) == 98, f"保留站点数应为98，当前={len(KEEP_STATION_IDS)}"


# =========================
# 工具函数
# =========================
def load_npz_array(npz_path, preferred_key):
    data = np.load(npz_path)

    if preferred_key in data:
        return data[preferred_key]
    elif "arr_0" in data:
        return data["arr_0"]
    else:
        raise KeyError(f"{npz_path} 未找到有效键，可用键: {list(data.keys())}")


def align_4d_arrays(true_data, pred_data):
    """
    对齐到 [num_samples, num_stations, num_features, seq_len]
    """
    if true_data.ndim != 4 or pred_data.ndim != 4:
        raise ValueError(f"数据维度错误: true={true_data.shape}, pred={pred_data.shape}")

    if true_data.shape[1] == 8 and true_data.shape[3] == 4:
        true_data = true_data.transpose(0, 2, 3, 1)
        pred_data = pred_data.transpose(0, 2, 3, 1)

    elif true_data.shape[1] == 4 and true_data.shape[3] == 8:
        true_data = true_data.transpose(0, 2, 1, 3)
        pred_data = pred_data.transpose(0, 2, 1, 3)

    return true_data, pred_data


# =========================
# 1. 生成 98 站点 npz
# =========================
print("\n[1/5] 处理 NPZ 流量数据...")

true_array = load_npz_array(OLD_YTRUE_PATH, "target")
pred_array = load_npz_array(OLD_YPRED_PATH, "prediction")

true_array, pred_array = align_4d_arrays(true_array, pred_array)

print(f"对齐后 true_array shape: {true_array.shape}")
print(f"对齐后 pred_array shape: {pred_array.shape}")

num_samples, num_stations, num_features, seq_len = true_array.shape

if num_stations == NUM_ORIGINAL_STATIONS:
    print("检测到原始 157 站点数据，正在裁剪为 98 站点...")

    true_98 = true_array[:, KEEP_STATION_IDS, :, :]
    pred_98 = pred_array[:, KEEP_STATION_IDS, :, :]

elif num_stations == 98:
    print("检测到数据已经是 98 站点，直接复制保存为 _98 文件...")

    true_98 = true_array
    pred_98 = pred_array

else:
    raise ValueError(
        f"站点数既不是157也不是98，当前 num_stations={num_stations}，请检查数据"
    )

np.savez_compressed(NEW_YTRUE_PATH, target=true_98)
np.savez_compressed(NEW_YPRED_PATH, prediction=pred_98)

print(f"✅ 已保存: {NEW_YTRUE_PATH}, shape={true_98.shape}")
print(f"✅ 已保存: {NEW_YPRED_PATH}, shape={pred_98.shape}")


# =========================
# 2. 生成 98 站点 station_list
# =========================
print("\n[2/5] 处理 station_list_hngs.txt...")

with open(OLD_STATION_LIST, "r", encoding="utf-8") as f:
    old_stations = [line.strip() for line in f if line.strip()]

assert len(old_stations) >= NUM_ORIGINAL_STATIONS, (
    f"原始 station_list 行数不足: {len(old_stations)}"
)

new_stations = [old_stations[old_sid] for old_sid in KEEP_STATION_IDS]

with open(NEW_STATION_LIST, "w", encoding="utf-8") as f:
    for line in new_stations:
        f.write(line + "\n")

print(f"✅ 已保存: {NEW_STATION_LIST}, 行数={len(new_stations)}")


# =========================
# 3. 生成 98 站点 pattern_list
# =========================
print("\n[3/5] 处理 station_natural_list_4feat.txt...")

with open(OLD_PATTERN_LIST, "r", encoding="utf-8") as f:
    old_patterns = [line.strip() for line in f if line.strip()]

assert len(old_patterns) >= NUM_ORIGINAL_STATIONS * NUM_FEATURES, (
    f"原始 pattern 行数不足: {len(old_patterns)}, "
    f"至少需要 {NUM_ORIGINAL_STATIONS * NUM_FEATURES}"
)

new_patterns = []

for old_sid in KEEP_STATION_IDS:
    for feat in range(NUM_FEATURES):
        old_pattern_idx = old_sid * NUM_FEATURES + feat
        new_patterns.append(old_patterns[old_pattern_idx])

with open(NEW_PATTERN_LIST, "w", encoding="utf-8") as f:
    for line in new_patterns:
        f.write(line + "\n")

print(f"✅ 已保存: {NEW_PATTERN_LIST}, 行数={len(new_patterns)}")


# =========================
# 4. 生成 98 站点 his_data_with_index.csv
# =========================
print("\n[4/5] 处理 his_data_with_index.csv...")

his_df = pd.read_csv(OLD_HIS_DATA)
his_df["时间"] = pd.to_datetime(his_df["时间"])

# 和你微调代码保持一致：按时间排序
parking_data = his_df.set_index("时间").sort_index(kind="stable")

expected_157_rows = true_98.shape[0] * NUM_ORIGINAL_STATIONS
expected_98_rows = true_98.shape[0] * 98

print(f"his_data 行数: {len(parking_data)}")
print(f"num_samples * 157 = {expected_157_rows}")
print(f"num_samples * 98  = {expected_98_rows}")

if len(parking_data) >= expected_157_rows:
    print("检测到 his_data 仍然是原始 157 站点结构，正在裁剪为 98 站点...")

    selected_rows = []

    for sample_idx in tqdm(range(true_98.shape[0]), desc="裁剪 his_data"):
        base = sample_idx * NUM_ORIGINAL_STATIONS

        for old_sid in KEEP_STATION_IDS:
            row_idx = base + old_sid
            if row_idx < len(parking_data):
                selected_rows.append(row_idx)

    parking_98 = parking_data.iloc[selected_rows].reset_index()

elif len(parking_data) >= expected_98_rows:
    print("检测到 his_data 可能已经是 98 站点结构，直接截取前 num_samples*98 行...")

    parking_98 = parking_data.iloc[:expected_98_rows].reset_index()

else:
    raise ValueError(
        f"his_data 行数不足，当前={len(parking_data)}, "
        f"至少需要 {expected_98_rows}"
    )

parking_98.to_csv(NEW_HIS_DATA, index=False, encoding="utf-8-sig")
print(f"✅ 已保存: {NEW_HIS_DATA}, 行数={len(parking_98)}")


# =========================
# 5. 生成 98 站点天气文件夹，重新编号为 000~097
# =========================
print("\n[5/5] 处理天气文件夹...")

os.makedirs(NEW_WEATHER_DIR, exist_ok=True)

old_weather_files = os.listdir(OLD_WEATHER_DIR)

missing_weather = []

for new_sid, old_sid in tqdm(
    list(enumerate(KEEP_STATION_IDS)),
    desc="复制并重编号天气文件"
):
    old_prefix = f"{old_sid:03d}"
    new_prefix = f"{new_sid:03d}"

    matches = [f for f in old_weather_files if f.startswith(old_prefix)]

    if not matches:
        missing_weather.append(old_sid)
        continue

    # 如果一个站点有多个天气文件，全部复制
    for fname in matches:
        old_path = os.path.join(OLD_WEATHER_DIR, fname)

        # 新文件名：把原来的前三位编号替换成新编号
        new_fname = new_prefix + fname[3:]
        new_path = os.path.join(NEW_WEATHER_DIR, new_fname)

        shutil.copy2(old_path, new_path)

print(f"✅ 已保存天气目录: {NEW_WEATHER_DIR}")
if missing_weather:
    print(f"⚠️ 以下原始站点没有找到天气文件: {missing_weather}")
else:
    print("✅ 所有保留站点均找到天气文件")


# =========================
# 保存映射
# =========================
with open(KEEP_ID_PATH, "w", encoding="utf-8") as f:
    json.dump(KEEP_STATION_IDS, f, ensure_ascii=False, indent=2)

with open(REMOVE_ID_PATH, "w", encoding="utf-8") as f:
    json.dump(sorted(list(REMOVED_STATION_IDS)), f, ensure_ascii=False, indent=2)

print("\n✅ 映射文件已保存:")
print(f"保留站点映射: {KEEP_ID_PATH}")
print(f"删除站点列表: {REMOVE_ID_PATH}")

print("\n" + "=" * 100)
print("✅ 98 站点微调相关文件全部生成完成")
print("=" * 100)
print(f"NPZ true:     {NEW_YTRUE_PATH}")
print(f"NPZ pred:     {NEW_YPRED_PATH}")
print(f"station list: {NEW_STATION_LIST}")
print(f"pattern list: {NEW_PATTERN_LIST}")
print(f"his data:     {NEW_HIS_DATA}")
print(f"weather dir:  {NEW_WEATHER_DIR}")
print("=" * 100)
