"""
python fine_gnn_vals.py
"""
import os
# 环境变量依然保留作为基础设置
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
os.environ["UNSLOTH_SKIP_INIT_CHECK"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import logging
import random
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from chinese_calendar import is_workday
import re
import sys

class Logger(object):
    def __init__(self, filename="3training_0_011.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 实时刷新，防止程序崩溃导致日志丢失

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 将这行放在代码最前面（imports 之后）
sys.stdout = Logger("3training_0_011.txt.txt")
# 1. 禁用 Unsloth 的报错拦截
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
print("✓ 正在启动【隐状态截获版】MAE 混合训练...")
# --- 2. 路径配置 ---
BASE_DATA_PATH = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"
NOTTINGHAM_H5_PATH = BASE_DATA_PATH + "hngs.h5"

# ✅ 使用可视化脚本生成的微调数据（包含8步、4特征、157站点）
FINETUNE_OUTPUT_PATH = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"
YTRUE_TRAIN_PATH = FINETUNE_OUTPUT_PATH + "finetune_real_traffic.npz"  # 真实值
YPRED_TRAIN_PATH = FINETUNE_OUTPUT_PATH + "finetune_data.npz"          # GNN预测值

TEXT_FILES_PATH = BASE_DATA_PATH
CARPARK_DES_PATH = TEXT_FILES_PATH + "station_list_hngs.txt"
NATURAL_PATTERN_PATH = TEXT_FILES_PATH + "station_natural_list_4feat.txt"
WEATHER_DATA_PATH = TEXT_FILES_PATH + "160站点天气信息/"

# ✅ 使用英文版事件文件（站点名称已转换为英文）
EVENTS_CSV_PATH_EN = TEXT_FILES_PATH + "events_list_quan_en.csv"
EVENTS_CSV_PATH_CN = TEXT_FILES_PATH + "events_list_quan.csv"

# 优先使用英文版，如果不存在则使用中文版
if os.path.exists(EVENTS_CSV_PATH_EN):
    EVENTS_CSV_PATH = EVENTS_CSV_PATH_EN
    print(f"✓ 使用英文版事件文件: {EVENTS_CSV_PATH}")
else:
    EVENTS_CSV_PATH = EVENTS_CSV_PATH_CN
    print(f"⚠️  英文版事件文件不存在，使用中文版: {EVENTS_CSV_PATH}")
    print(f"   建议先运行: python convert_events_to_english.py")

HIS_DATA_CSV_PATH = TEXT_FILES_PATH + "his_data_with_index.csv"

# --- 3. 数据加载与天气预处理 (仅保留天气代码) ---
print("正在加载数据并精简天气信息...")

# 加载历史数据CSV获取时间索引
his_df = pd.read_csv(HIS_DATA_CSV_PATH)
his_df['时间'] = pd.to_datetime(his_df['时间'])
parking_data = his_df.set_index('时间').sort_index()

# 加载NPZ格式的预测和真实值（使用正确的键名）
npz_true = np.load(YTRUE_TRAIN_PATH)
npz_pred = np.load(YPRED_TRAIN_PATH)

print(f"✓ NPZ文件键名检测:")
print(f"  ytrue_train.npz 键名: {list(npz_true.keys())}")
print(f"  ypred_train.npz 键名: {list(npz_pred.keys())}")

# 根据实际键名加载数据
if 'target' in npz_true:
    true_array = npz_true['target']
elif 'arr_0' in npz_true:
    true_array = npz_true['arr_0']
else:
    raise KeyError(f"ytrue_train.npz 中未找到预期的键名，可用键名: {list(npz_true.keys())}")

if 'prediction' in npz_pred:
    pred_array = npz_pred['prediction']
elif 'arr_0' in npz_pred:
    pred_array = npz_pred['arr_0']
else:
    raise KeyError(f"ypred_train.npz 中未找到预期的键名，可用键名: {list(npz_pred.keys())}")

print(f"✓ 数据加载成功:")
print(f"  true_array shape: {true_array.shape}")
print(f"  pred_array shape: {pred_array.shape}")

# ✅ 验证预测值和真实值是否不同
if np.allclose(true_array, pred_array):
    print("\n⚠️  ⚠️  ⚠️  严重警告：预测值和真实值完全相同！ ⚠️  ⚠️  ⚠️")
    print("  可能原因:")
    print("  1. YTRUE_TRAIN_PATH 和 YPRED_TRAIN_PATH 指向了同一个文件")
    print("  2. GNN预测结果被错误地同时用作预测值和真实值")
    print("  3. 数据文件损坏或为空")
    print(f"\n  当前配置:")
    print(f"  YTRUE_TRAIN_PATH = {YTRUE_TRAIN_PATH}")
    print(f"  YPRED_TRAIN_PATH = {YPRED_TRAIN_PATH}")
    print("\n  请检查路径配置，确保两者指向不同的文件！")
    raise ValueError("预测值和真实值不能相同，否则LLM无法学习修正能力！")
else:
    diff = np.abs(true_array - pred_array)
    print(f"\n✓ 数据验证通过:")
    print(f"  - 预测值与真实值差异:")
    print(f"    平均绝对误差 (MAE): {diff.mean():.2f}")
    print(f"    最大绝对误差: {diff.max():.2f}")
    print(f"    误差标准差: {diff.std():.2f}")
    print(f"  - 这表明GNN预测存在误差，LLM可以学习修正这些误差")

# 加载站点描述
with open(CARPARK_DES_PATH, "r", encoding='utf-8') as f:
    carpark_des_list = [l.strip() for l in f if l.strip()]

# 构建中英文站点名称映射（从站点描述中提取英文名作为key）
station_name_mapping = {}
for desc in carpark_des_list:
    # 提取第一个单词作为英文站点名（如"Huangxing"）
    english_name = desc.split(' ')[0].strip()
    station_name_mapping[english_name] = desc

print(f"✓ 已构建 {len(station_name_mapping)} 个站点的英文名称映射")

# 加载自然语言模式描述（4个特征）
with open(NATURAL_PATTERN_PATH, "r", encoding='utf-8') as f:
    natural_pattern_list = [l.strip() for l in f if l.strip()]

# 调整数组维度：确保格式为 (样本数, 站点数, 特征数, 时间步)
print(f"\n原始数据形状分析:")
print(f"  true_array shape: {true_array.shape}")
print(f"  pred_array shape: {pred_array.shape}")

# 根据实际数据结构调整
# 预期格式: (样本数, 序列长度, 站点数, 特征数) -> 需要转换为 (样本数, 站点数, 特征数, 序列长度)
if true_array.ndim == 4:
    # 假设原始格式是 (样本数, 序列长度=8, 站点数, 特征数=4)
    # 需要转置为 (样本数, 站点数, 特征数, 序列长度)
    if true_array.shape[1] == 8 and true_array.shape[3] == 4:
        print("✓ 检测到标准4维格式: (样本数, 序列长度=8, 站点数, 特征数=4)")
        print("  正在转置为: (样本数, 站点数, 特征数, 序列长度)")
        true_array = true_array.transpose(0, 2, 3, 1)  # (N, T, S, C) -> (N, S, C, T)
        pred_array = pred_array.transpose(0, 2, 3, 1)
    elif true_array.shape[1] == 4 and true_array.shape[3] == 8:
        print("✓ 检测到另一种4维格式: (样本数, 特征数=4, 站点数, 序列长度=8)")
        print("  正在转置为: (样本数, 站点数, 特征数, 序列长度)")
        true_array = true_array.transpose(0, 2, 1, 3)  # (N, C, S, T) -> (N, S, C, T)
        pred_array = pred_array.transpose(0, 2, 1, 3)
    else:
        print(f"⚠️ 未知的4维格式，尝试直接使用")
else:
    raise ValueError(f"不支持的数据维度: {true_array.ndim}D，期望4维数据")

num_samples, num_stations, num_features, seq_len = true_array.shape
print(f"\n✓ 维度对齐完成:")
print(f"  样本数 (Samples): {num_samples}")
print(f"  站点数 (Stations): {num_stations}")
print(f"  特征数 (Features): {num_features}")
print(f"  序列长度 (Sequence Length): {seq_len}")

# 验证维度合理性
if num_features != 4:
    print(f"\n⚠️ 警告: 检测到 {num_features} 个特征，但预期为 4 个特征")
    print(f"  请检查数据是否正确转置")
if seq_len != 8:
    print(f"\n⚠️ 警告: 检测到序列长度为 {seq_len}，但预期为 8")

# 定义特征名称映射（严格按照顺序）
FEATURE_NAMES = {
    0: "小客车上行",
    1: "小客车下行", 
    2: "非小客车上行",
    3: "非小客车下行"
}

def get_weather_at_time(station_idx, target_time):
    """获取指定站点在指定时间的天气"""
    try:
        # 构建天气文件路径
        weather_files = os.listdir(WEATHER_DATA_PATH)
        # 查找对应站点的天气文件
        station_prefix = f"{station_idx:03d}"
        matched_file = None
        for wf in weather_files:
            if wf.startswith(station_prefix):
                matched_file = wf
                break
        
        if not matched_file:
            return "Unknown"
        
        # 读取天气文件
        weather_path = os.path.join(WEATHER_DATA_PATH, matched_file)
        if matched_file.endswith('.csv'):
            weather_df_local = pd.read_csv(weather_path)
        elif matched_file.endswith('.xlsx'):
            weather_df_local = pd.read_excel(weather_path)
        else:
            return "Unknown"
        
        # 查找日期列
        date_col = None
        for col in weather_df_local.columns:
            if '日期' in col or 'date' in col.lower():
                date_col = col
                break
        
        if not date_col:
            return "Unknown"
        
        weather_df_local['日期'] = pd.to_datetime(weather_df_local['日期'])
        weather_df_local = weather_df_local.set_index('日期').sort_index()
        
        # 查找天气代码列
        code_col = None
        for col in weather_df_local.columns:
            if "天气代码" in col or "weather" in col.lower():
                code_col = col
                break
        
        if not code_col:
            return "Unknown"
        
        idx = weather_df_local.index.get_indexer([target_time], method='pad')[0]
        return str(weather_df_local.iloc[idx][code_col]) if idx != -1 else "Unknown"
    except Exception as e:
        return "Unknown"

def load_precise_events(path, station_name_mapping=None):
    """
    加载事件数据,并自动转换站点名称为英文
    
    Args:
        path: 事件文件路径
        station_name_mapping: 中文->英文的站点名称映射字典
    """
    if not os.path.exists(path):
        print(f"⚠️ 未找到事件文件 {path},将按无事件模式运行。")
        return {}
    try:
        # 尝试多种编码方式读取CSV文件 (GB18030是GBK超集,支持更多字符)
        df_ev = None
        for encoding in ['utf-8', 'gb18030', 'gbk', 'latin1']:
            try:
                # 先尝试直接读取
                df_ev = pd.read_csv(path, encoding=encoding)
                print(f"✓ 使用 {encoding} 编码成功加载事件文件")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # 如果所有编码都失败,尝试用ignore策略逐行读取
        if df_ev is None:
            print(f"  ⚠️ 标准读取失败,尝试逐行容错读取...")
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
                
                # 解析CSV
                import io
                csv_content = ''.join(lines)
                df_ev = pd.read_csv(io.StringIO(csv_content))
                print(f"✓ 使用容错模式成功读取 {len(df_ev)} 行")
            except Exception as e:
                raise ValueError(f"无法读取事件文件: {e}")
        
        if df_ev is None or df_ev.empty:
            raise ValueError(f"无法使用任何编码读取事件文件: {path}")
        
        # 打印列名以便调试
        print(f"  事件文件列名: {list(df_ev.columns)}")
        print(f"  总行数: {len(df_ev)}")
        
        # 标准化列名(处理可能的编码问题)
        df_ev.columns = [col.strip() for col in df_ev.columns]
        
        # 查找正确的列名
        date_col = None
        station_col = None
        event_col = None
        
        for col in df_ev.columns:
            if '日期' in col or 'date' in col.lower():
                date_col = col
            elif '站' in col or 'station' in col.lower():
                station_col = col
            elif '事件' in col or 'event' in col.lower():
                event_col = col
        
        if not all([date_col, station_col, event_col]):
            print(f"⚠️ 无法识别事件文件的列名,使用默认列名")
            date_col = df_ev.columns[0]
            station_col = df_ev.columns[1]
            event_col = df_ev.columns[2]
        
        print(f"  使用的列: 日期={date_col}, 站点={station_col}, 事件={event_col}")
        
        df_ev['日期'] = pd.to_datetime(df_ev[date_col]).dt.strftime('%Y-%m-%d')
        
        # ✅ 检测事件文件中的站点名称是中文还是英文
        sample_stations = df_ev[station_col].dropna().head(10).astype(str).tolist()
        is_chinese = any('\u4e00' <= char <= '\u9fff' for station in sample_stations for char in station)
        
        if is_chinese:
            print(f"  ✓ 检测到事件文件使用中文站点名称，将进行转换")
            needs_conversion = True
        else:
            print(f"  ✓ 检测到事件文件已使用英文站点名称，无需转换")
            needs_conversion = False
        
        # 构建 (日期, 站点名称) -> 事件内容 的映射
        event_dict = {}
        converted_count = 0
        skipped_count = 0
        
        for _, row in df_ev.iterrows():
            original_station = str(row[station_col]).strip()
            event_desc = str(row[event_col]).strip()
            date_key = row['日期']
            
            # 跳过无效数据
            if not original_station or original_station == 'nan':
                skipped_count += 1
                continue
            
            # ✅ 根据检测结果决定是否转换
            final_station = original_station
            if needs_conversion and station_name_mapping and original_station in station_name_mapping:
                final_station = station_name_mapping[original_station]
                converted_count += 1
            # 如果已经是英文，直接使用
            
            key = (date_key, final_station)
            event_dict[key] = event_desc
        
        print(f"✓ 已加载 {len(event_dict)} 条精准事件数据")
        if skipped_count > 0:
            print(f"  跳过 {skipped_count} 条无效记录")
        if converted_count > 0:
            print(f"  其中 {converted_count} 条站点名称已从中文转换为英文")
        
        # 打印样例用于调试
        if event_dict:
            print(f"\n  事件数据样例(前3条):")
            for i, (key, val) in enumerate(list(event_dict.items())[:3]):
                print(f"    {key}: {val[:60]}...")
        
        return event_dict
    except Exception as e:
        print(f"❌ 加载事件失败: {e}")
        import traceback
        traceback.print_exc()
        print(f"  将继续运行,但所有样本的事件信息将设为 'None'")
        return {}

# 构建中文->英文的站点名称映射
# 优先从JSON文件加载,其次使用硬编码映射
mapping_json_path = os.path.join(BASE_DATA_PATH, 'station_name_mapping.json')

if os.path.exists(mapping_json_path):
    import json
    with open(mapping_json_path, 'r', encoding='utf-8') as f:
        chinese_to_english_station = json.load(f)
    print(f"✓ 从JSON文件加载 {len(chinese_to_english_station)} 个站点映射")
    print(f"  文件路径: {mapping_json_path}")
else:
    # Fallback: 手动维护的映射表(需要根据实际情况补充)
    print(f"⚠️ 未找到映射JSON文件: {mapping_json_path}")
    print(f"  建议运行: python build_station_mapping.py 自动生成映射文件")
    
    chinese_to_english_station = {
        # 示例映射(需要根据实际数据补充完整)
        "黄兴": "Huangxing",
        "榔梨": "Langli", 
        "红旗": "Hongqi",
        "洞井": "Dongjing",
        "星城": "Xingcheng",
        "岳阳东": "Yueyang East",
        # ... 请补充剩余的154个站点映射
    }
    print(f"  使用硬编码映射表,共 {len(chinese_to_english_station)} 个条目")

print(f"✓ 已构建 {len(chinese_to_english_station)} 个中英文站点名称映射")

# 加载事件数据(传入映射表)
precise_events_map = load_precise_events(EVENTS_CSV_PATH, chinese_to_english_station)

# 打印事件统计信息
if precise_events_map:
    print(f"\n📊 事件数据统计:")
    # 统计涉及的日期范围
    dates = set(key[0] for key in precise_events_map.keys())
    stations_with_events = set(key[1] for key in precise_events_map.keys())
    print(f"  涉及日期数: {len(dates)}")
    print(f"  涉及站点数: {len(stations_with_events)}")
    print(f"  总事件记录数: {len(precise_events_map)}")
    
    # 检查是否有匹配成功的样例
    sample_matches = 0
    for key in list(precise_events_map.keys())[:5]:
        if key[1] in [v for v in chinese_to_english_station.values()]:
            sample_matches += 1
    print(f"  前5条中英文名匹配数: {sample_matches}/5")
else:
    print(f"\n⚠️ 未加载到任何事件数据,所有样本将标记为无事件")

# --- 4. 自定义混合 MAE Loss Trainer (修正缩进与参数) ---
class NoLogitMAEHybridTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 适配新版警告，优先使用 processing_class
        local_tokenizer = getattr(self, "processing_class", self.tokenizer)

        # 1. 动态扫描词表，找出纯阿拉伯数字 Token
        self.id_to_val = {}
        all_digit_ids = []

        print("🔍 正在扫描 Llama-3.1 词表（严格过滤版）...")
        for i in range(local_tokenizer.vocab_size):
            token_text = local_tokenizer.decode([i]).strip()

            # 严格过滤：必须是非空，且每个字符都在 0-9 之间
            if token_text and all('0' <= char <= '9' for char in token_text):
                try:
                    val = float(token_text)
                    self.id_to_val[i] = val
                    all_digit_ids.append(i)
                except ValueError:
                    continue

        # 存储为 Tensor
        self.digit_ids = torch.tensor(all_digit_ids).to(self.args.device)
        # 锁定锚点：维持 510
        self.target_bracket_id = 510

        print(f"✅ 成功捕获 {len(all_digit_ids)} 个纯数字变体 Token")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        # A. 隐状态截获
        inputs["output_hidden_states"] = True
        outputs = model(**inputs)
        last_hidden_state = outputs.hidden_states[-1]
        target_dtype = next(model.lm_head.parameters()).dtype
        manual_logits = model.lm_head(last_hidden_state.to(target_dtype))

        # B. 基础 CE Loss
        ce_loss = outputs.loss if isinstance(outputs, dict) else outputs[0]

        # C. 维度对齐
        labels = inputs.get("labels")
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = manual_logits[..., :-1, :].contiguous()

        # 1. 全场扫描所有数字 ID
        all_digit_mask = torch.isin(shift_labels, self.digit_ids)

        # 2. 定位逻辑：锁定 Final Correction: [ 之后的数字位
        digit_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        for b in range(shift_labels.shape[0]):
            bracket_pos = (shift_labels[b] == self.target_bracket_id).nonzero(as_tuple=True)[0]
            if len(bracket_pos) > 0:
                start_from = bracket_pos[-1].item() + 1
                digit_mask[b, start_from:] = all_digit_mask[b, start_from:]

        # 排除填充位
        digit_mask &= (shift_labels != -100)

        # D. 计算 MAE 混合损失
        mae_loss = torch.tensor(0.0).to(self.args.device)
        total_loss = ce_loss
        expected_values = torch.tensor([]).to(self.args.device)

        if digit_mask.any():
            digit_logits = shift_logits[digit_mask][:, self.digit_ids]
            probs = torch.softmax(digit_logits.to(torch.float32), dim=-1)

            val_vector = torch.tensor(
                [self.id_to_val[int(tid)] for tid in self.digit_ids.tolist()]
            ).to(self.args.device).to(torch.float32)

            expected_values = (probs * val_vector).sum(dim=-1)

            target_labels = shift_labels[digit_mask]
            target_values = torch.zeros_like(expected_values)
            for i in range(len(target_labels)):
                tid = int(target_labels[i])
                target_values[i] = self.id_to_val.get(tid, 0.0)

            # MAE 计算
            mask = torch.not_equal(target_values, 0.0).float()
            mask_mean = torch.mean(mask)
            if mask_mean > 0: mask = mask / mask_mean

            abs_error = torch.abs(expected_values - target_values)
            mae_loss = torch.mean(abs_error * mask)
            total_loss = ce_loss + 0.11 * mae_loss

            # --- E. 实时数值对账单 ---
            if self.state.global_step % 1 == 0:
                sample_idx = 0
                s_labels = shift_labels[sample_idx].cpu().numpy()
                b_indices = (s_labels == self.target_bracket_id).nonzero()[0]

                if len(b_indices) > 0:
                    start_preview = max(0, b_indices[-1] - 150)
                    relevant_ids = s_labels[start_preview:]
                    safe_ids = [tid for tid in relevant_ids if tid != -100]
                    raw_text_only = self.tokenizer.decode(safe_ids, skip_special_tokens=False).split('<|eot_id|>')[
                                        0] + " <|eot_id|>"

                    prev_mae_bits = digit_mask[:sample_idx].sum().item()
                    curr_mae_bits = digit_mask[sample_idx].sum().item()

                    if curr_mae_bits > 0:
                        batch_preds = expected_values[prev_mae_bits: prev_mae_bits + curr_mae_bits].cpu().tolist()
                        start_ptr = b_indices[-1].item() + 1
                        raw_token_seq = shift_labels[sample_idx, start_ptr:].cpu().tolist()

                        final_pred_res = []
                        final_true_res = []

                        current_p_str = ""
                        current_t_str = ""

                        pred_ptr = 0
                        for tid in raw_token_seq:
                            if tid == -100 or tid == 128009: break

                            if tid in self.id_to_val:
                                token_char = self.tokenizer.decode([tid]).strip()
                                current_t_str += token_char
                                if pred_ptr < len(batch_preds):
                                    p_val = int(round(batch_preds[pred_ptr]))
                                    current_p_str += str(p_val)
                                    pred_ptr += 1

                            elif tid == 11 or tid == 60:
                                if current_t_str:
                                    final_true_res.append(int(current_t_str) if current_t_str.isdigit() else 0)
                                    final_pred_res.append(int(current_p_str) if current_p_str.isdigit() else 0)
                                    current_t_str = ""
                                    current_p_str = ""

                        print(f"\n" + "=" * 15 + " 核心客流对账单 (硬核对齐版) " + "=" * 15)
                        print(f"Step: {self.state.global_step}")
                        print(f"原文预览:\n{raw_text_only}")
                        print(f"还原预测数值: {final_pred_res[:8]}")
                        print(f"还原真实数值: {final_true_res[:8]}")
                        print("-" * 60)
                        print(
                            f"Loss Total: {total_loss.item():.4f} | CE: {ce_loss.item():.4f} | MAE: {mae_loss.item():.4f}")
                        print("=" * 65 + "\n")

        return (total_loss, outputs) if return_outputs else total_loss

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/user/Llama-3.1-8B",
    max_seq_length=1024,
    load_in_4bit=True,
    dtype=torch.bfloat16,
    device_map={"": 0},
)

# --- 4. 生成训练数据集 ---
# 适配 Llama-3.1 标准格式 - 增加特征标识
instruction_text = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "Role: Traffic Flow Prediction Refiner Objective: Fine-tune GNN simulation based on external factors.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Input Data: "
    "(1) Station: {} "
    "(2) Feature Type: {} "
    "(3) The prediction period: {} to {} (Day Type: {}). "
    "(4) Flow Patterns: {}. "
    "(5) Weather: {}. "
    "(6) GNN simulation: {}. "
    "(7) {}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)

print("正在生成数据集：4特征多变量模式 + 数值取整 + 日期切换检测...")
generated_dataset = []

# 设置挑选阈值
ERROR_THRESHOLD = 40.0  # 显著误差阈值
NORMAL_SAMPLE_RATIO = 0.1  # 平常样本抽取比例 (10%)

print(f"开始处理 {num_stations} 个站点，每个站点 {num_features} 个特征...")

# 遍历所有站点和所有特征
for j in range(num_stations):
    full_description = carpark_des_list[j]
    station_name = full_description.split(' ')[0].strip() if ' ' in full_description else full_description.strip()
    
    print(f"\n{'='*60}")
    print(f"正在处理站点 [{j+1}/{num_stations}]: {station_name}")
    print(f"{'='*60}")
    
    # 为该站点的所有4个特征分别处理
    for feat_idx in range(num_features):
        feature_name = FEATURE_NAMES[feat_idx]
        print(f"  → 处理特征: [{feature_name}]")
        
        # 从natural_pattern_list中获取该特征的Pattern
        pattern_idx = j * num_features + feat_idx
        if pattern_idx >= len(natural_pattern_list):
            print(f"    ⚠️ Pattern索引越界: {pattern_idx}，跳过")
            continue
            
        pattern_str = natural_pattern_list[pattern_idx]
        
        # 解析Pattern中的范围信息
        ranges = re.findall(r"range 0-(\d+)", pattern_str)
        work_max = ranges[0] if len(ranges) > 0 else "0"
        off_max = ranges[1] if len(ranges) > 1 else work_max
        
        # 只循环这一个站点+特征的样本
        for i in range(num_samples):
            try:
                s2 = 8 + i
                if s2 + 7 >= len(parking_data.index): 
                    break

                # 过滤平稳重复数据
                if i > 0 and np.array_equal(true_array[i, j, feat_idx, :], true_array[i - 1, j, feat_idx, :]):
                    continue

                t3 = parking_data.index[s2]
                date_key = t3.strftime('%Y-%m-%d')
                t4 = parking_data.index[s2+7]
                
                # 查询事件（尝试多种名称匹配策略）
                event_info = "None"
                
                # 策略1: 直接使用英文名称查询
                event_info = precise_events_map.get((date_key, station_name), "None")
                
                # 策略2: 如果没找到，尝试从中文名映射
                if event_info == "None" and station_name in chinese_to_english_station:
                    english_name = chinese_to_english_station[station_name]
                    event_info = precise_events_map.get((date_key, english_name), "None")
                
                # 策略3: 反向查找（遍历所有可能的中文名称）
                if event_info == "None":
                    for cn_name, en_name in chinese_to_english_station.items():
                        if en_name == station_name:
                            event_info = precise_events_map.get((date_key, cn_name), "None")
                            if event_info != "None":
                                break
                
                # 获取该特征的预测和真实值
                pred_raw = pred_array[i, j, feat_idx, :]
                truth_raw = true_array[i, j, feat_idx, :]
                
                current_mae = np.mean(np.abs(pred_raw - truth_raw))
                
                # 判定跨天与日期属性切换
                is_work_start = is_workday(t3)
                is_work_end = is_workday(t4)
                is_type_transition = (is_work_start != is_work_end)
                is_cross_day = (t3.date() != t4.date())

                # 动态描述 Day Type
                if is_type_transition:
                    day_type_desc = f"{'Workday' if is_work_start else 'Off-day'} to {'Workday' if is_work_end else 'Off-day'}"
                else:
                    day_type_desc = "Workday" if is_work_start else "Off-day"

                # 逐点检查8个时间步
                gnn_time_points = [parking_data.index[s2 + k] for k in range(8)]
                gnn_vals = [int(round(v)) for v in pred_raw]
                peak_hits = []
                low_peak_hits = []

                for k in range(8):
                    curr_t = gnn_time_points[k]
                    curr_is_work = is_workday(curr_t)
                    curr_time_str = curr_t.strftime('%H:%M:%S')

                    # 选择对应日期的 Pattern
                    pattern_part = pattern_str.split(';')[0] if curr_is_work else pattern_str.split(';')[1]

                    # 提取规则：高峰与低峰
                    p_match = re.search(
                        r"peak time ([\d:,\s]+), average peak flow ([\d:,\s]+), low-peak time ([\d:,\s]+), average low-peak flow ([\d:,\s]+)",
                        pattern_part)

                    if p_match:
                        p_times = [t.strip() for t in p_match.group(1).split(',')]
                        p_flows = [f.strip() for f in p_match.group(2).split(',')]
                        l_times = [t.strip() for t in p_match.group(3).split(',')]
                        l_flows = [f.strip() for f in p_match.group(4).split(',')]

                        # 检测高峰
                        if curr_time_str in p_times:
                            peak_hits.append({"step": k, "time": curr_time_str[:5],
                                              "ref_val": int(p_flows[p_times.index(curr_time_str)]),
                                              "is_work": curr_is_work})

                        # 检测低峰
                        if curr_time_str in l_times:
                            low_peak_hits.append({"step": k, "time": curr_time_str[:5],
                                                  "ref_val": int(l_flows[l_times.index(curr_time_str)]),
                                                  "is_work": curr_is_work})
                
                start_hm, end_hm = t3.strftime('%H:%M'), t4.strftime('%H:%M')
                cross_tag = "CROSS-DAY " if is_cross_day else ""

                reason = ""
                is_effective_event = False
                if event_info != "None":
                    has_time_in_event = bool(re.search(r"\d{2}:\d{2}", event_info))
                    has_peak_hit = len(peak_hits) > 0

                    if has_time_in_event or has_peak_hit:
                        is_effective_event = True

                # 判断样本类型
                if is_effective_event:
                    reason = "Event"
                elif event_info != "None":
                    event_info = "None"
                    if current_mae > ERROR_THRESHOLD:
                        reason = "Error"
                    elif random.random() < NORMAL_SAMPLE_RATIO:
                        reason = "Normal"
                elif current_mae > ERROR_THRESHOLD:
                    reason = "Error"
                elif random.random() < NORMAL_SAMPLE_RATIO:
                    reason = "Normal"

                if reason:
                    # 抓取特征 (站点属性)
                    matched_types = []
                    desc_lower = full_description.lower()
                    if "residential" in desc_lower: matched_types.append("Residential Area")
                    if "commercial" in desc_lower: matched_types.append("Commercial Area")
                    if "hub" in desc_lower: matched_types.append("Transportation Hub")
                    if "business" in desc_lower: matched_types.append("Business Area")
                    if "scenic" in desc_lower: matched_types.append("Scenic Area")
                    area_type = "/".join(matched_types) if matched_types else "General Area"
                    is_interchange = "Interchange Station" if "interchange" in full_description.lower() else "Non-interchange Station"

                    # --- 构造分析文本 ---
                    analysis = []

                    # (1) 站点属性
                    analysis.append(f"1. This station is a {area_type} and {is_interchange}.")
                    
                    # (2) 时间和日期属性
                    weekday_str = t3.strftime('%A')
                    has_event_label = "Yes" if event_info != "None" else "No"
                    analysis.append(f"2. The prediction period is {cross_tag}{day_type_desc} {start_hm} to {end_hm} (Special Event: {has_event_label}).")
                    
                    hybrid_targets = [float(v) for v in gnn_vals]
                    
                    # --- 特殊事件判定逻辑 ---
                    event_handled = False
                    if event_info != "None":
                        hit_event_steps = []
                        event_info_cleaned = event_info.replace("：", ":")
                        
                        range_match = re.search(r"(?:surge|inflow|between)\s+(\d{2}:\d{2})\s+and\s+(\d{2}:\d{2})",
                                                event_info_cleaned, re.I)
                        single_point_matches = re.findall(r"(?:surge\s+at|inflow\s+at|flow\s+at)\s+(\d{2}:\d{2})",
                                                          event_info_cleaned, re.I)

                        if range_match:
                            start_str, end_str = range_match.groups()
                            start_min = int(start_str[:2]) * 60 + int(start_str[3:])
                            end_min = int(end_str[:2]) * 60 + int(end_str[3:])

                            for k in range(8):
                                curr_t = gnn_time_points[k]
                                curr_min = curr_t.hour * 60 + curr_t.minute
                                if start_min <= curr_min <= end_min:
                                    hit_event_steps.append(k)

                        if single_point_matches:
                            for et in single_point_matches:
                                for k in range(8):
                                    if et == gnn_time_points[k].strftime('%H:%M'):
                                        hit_event_steps.append(k)

                        hit_event_steps = sorted(list(set(hit_event_steps)))

                        if hit_event_steps:
                            event_handled = True
                            event_details = []
                            for step in hit_event_steps:
                                g_val = gnn_vals[step]
                                t_val = truth_raw[step]
                                ratio = ((t_val - g_val) / g_val * 100) if g_val > 0 else 100.0
                                hybrid_targets[step] = float(t_val)
                                event_details.append(
                                    f"at {gnn_time_points[step].strftime('%H:%M')}, the GNN prediction is {g_val}, should be adjusted by {ratio:+.2f}%")
                            analysis.append(
                                f"3. Strategy: Special Event Mode. Although this is typically a Non-peak period, due to {event_info_cleaned}, " + ", ".join(
                                    event_details) + ".")
                        else:
                            if peak_hits and any(h['is_work'] for h in peak_hits):
                                event_handled = True
                                hit = [h for h in peak_hits if h['is_work']][0]
                                ref_val = hit["ref_val"]
                                gnn_at_peak = gnn_vals[hit['step']]
                                t_val = truth_raw[hit['step']]
                                ratio = ((t_val - gnn_at_peak) / gnn_at_peak * 100) if gnn_at_peak > 0 else 100.0
                                hybrid_targets[hit['step']] = float(t_val)
                                analysis.append(
                                    f"3. Strategy: Special Event Mode. Detected historical peak point at {hit['time']}. "
                                    f"But due to {event_info_cleaned}, the peak sequence needs to be adjusted by {ratio:+.2f}%.")

                    # --- 原本的 Strategy 逻辑 ---
                    if not event_handled:
                        if low_peak_hits:
                            hit = low_peak_hits[0]
                            ref_val = hit["ref_val"]
                            gnn_at_low = gnn_vals[hit["step"]]
                            hybrid_targets[hit["step"]] = float(ref_val)
                            analysis.append(
                                f"3. Strategy: Low-peak Mode. Detected historical low-peak point at {hit['time']}. "
                                f"The GNN prediction at this point is {gnn_at_low}, while the historical average low-peak flow baseline is {ref_val}. "
                                f"Conclusion: adjust the GNN prediction value {gnn_at_low} to {ref_val}."
                            )
                        elif peak_hits and any(h['is_work'] for h in peak_hits):
                            work_hits = [h for h in peak_hits if h['is_work']]
                            hit = work_hits[0]
                            ref_val = hit["ref_val"]
                            gnn_at_peak = gnn_vals[hit["step"]]
                            truth_at_peak = int(round(truth_raw[hit["step"]]))
                            hybrid_targets[hit["step"]] = float(truth_at_peak)

                            if gnn_at_peak > 0:
                                adj_ratio = ((truth_at_peak - gnn_at_peak) / gnn_at_peak) * 100
                                ratio_str = f"{adj_ratio:+.2f}%"
                            else:
                                ratio_str = "+100% (Baseline correction)"

                            if gnn_at_peak < ref_val:
                                judge = f"the GNN prediction ({gnn_at_peak}) is lower than the historical baseline ({ref_val}), and should be significantly increased. "
                            else:
                                judge = f"the GNN prediction ({gnn_at_peak}) has exceeded the historical baseline ({ref_val}), and should be moderately increased. "

                            analysis.append(
                                f"3. Strategy: Workday Peak Mode. Detected historical peak point at {hit['time']}. "
                                f"The historical average peak flow (baseline) for this period is {ref_val}. "
                                f"At this point, GNN predicts {gnn_at_peak}. "
                                f"Conclusion: {judge} The GNN sequence needs to be adjusted by {ratio_str}."
                            )
                        elif peak_hits:
                            hit = peak_hits[0]
                            analysis.append(
                                f"3. Strategy: Off-day Mode. Detected peak point at {hit['time']}. "
                                f"However, peak adjustments are not applied on Off-days, so the GNN simulation is followed exactly."
                            )
                        else:
                            mode = "Transition" if is_type_transition else "Routine"
                            analysis.append(
                                f"3. Strategy: {day_type_desc} {mode} Mode. No historical peak or low-peak points detected, so the GNN simulation is followed exactly.")

                    # --- 天气核心修改 ---
                    weather_input = get_weather_at_time(j, t3)
                    has_off_day = (not is_work_start) or (not is_work_end)

                    if not has_off_day:
                        weather_logic = f"4. Weather Factor: Weather correction is ignored in Workday mode."
                    else:
                        if any(w in weather_input for w in ['Moderate Rain', 'Heavy Rain', 'Moderate Snow']):
                            weather_logic = f"4. Weather Factor: {weather_input}. Adverse weather on Off-days reduces travel intention, suggesting a slight downward adjustment."
                        else:
                            weather_logic = f"4. Weather Factor: {weather_input}. Normal weather, no adjustment needed."

                    analysis.append(weather_logic)

                    # --- 精准混合数值逻辑 ---
                    if event_handled:
                        for step in hit_event_steps:
                            hybrid_targets[step] = truth_raw[step]
                    elif not event_handled:
                        if low_peak_hits:
                            hit = low_peak_hits[0]
                            hybrid_targets[hit['step']] = hit['ref_val']
                        elif peak_hits and any(h['is_work'] for h in peak_hits):
                            work_hits = [h for h in peak_hits if h['is_work']]
                            hit = work_hits[0]
                            hybrid_targets[hit['step']] = truth_raw[hit['step']]

                    # 将数值取整并封装
                    final_correction_list = [int(round(v)) for v in hybrid_targets]

                    analysis.append("Final Correction: [")
                    cot_text = " ".join(analysis)
                    final_values = ", ".join(map(str, final_correction_list)) + "] <|eot_id|>"

                    # 填充最终 Prompt (注意增加了特征类型字段)
                    prompt = instruction_text.format(
                        full_description, 
                        feature_name,  # 新增：特征类型
                        t3, t4, day_type_desc,
                        pattern_str, (work_max if is_work_start else off_max), 
                        weather_input,
                        gnn_vals, event_info
                    )

                    # --- 手动分词并屏蔽 Prompt ---
                    full_content = prompt + cot_text + final_values

                    full_enc = tokenizer(full_content, truncation=True, max_length=1024, add_special_tokens=False)
                    full_ids = full_enc["input_ids"]

                    prompt_enc = tokenizer(prompt, truncation=True, max_length=1024, add_special_tokens=False)
                    prompt_len = len(prompt_enc["input_ids"])

                    labels = [-100] * prompt_len + full_ids[prompt_len:]

                    if len(labels) > len(full_ids):
                        labels = labels[:len(full_ids)]
                    elif len(labels) < len(full_ids):
                        labels = labels + [-100] * (len(full_ids) - len(labels))

                    # 保存处理后的 ID
                    generated_dataset.append({
                        "decoded_text": full_content,
                        "input_ids": full_ids,
                        "labels": labels,
                        "attention_mask": [1] * len(full_ids),
                        "reason": reason,
                        "station_idx": j,
                        "feature_idx": feat_idx,
                        "feature_name": feature_name
                    })

            except Exception as e:
                print(f"    Error processing sample {i} for feature {feat_idx}: {e}")
                continue
    
    print(f"  ✓ 站点 {j+1} 已完成，累计样本数: {len(generated_dataset)}")

print(f"\n{'='*60}")
print(f"✓ 筛选完成。总样本数: {len(generated_dataset)}")

# --- 统计输出逻辑 ---
event_count = sum(1 for d in generated_dataset if d['reason'] == "Event")
error_count = sum(1 for d in generated_dataset if d['reason'] == "Error")
normal_count = sum(1 for d in generated_dataset if d['reason'] == "Normal")

print(f"  - 特殊事件样本 (Event): {event_count}")
print(f"  - 显著误差样本 (Error): {error_count}")
print(f"  - 随机平常样本 (Normal): {normal_count}")
print(f"  - 平常样本占比: {(normal_count / len(generated_dataset) * 100):.2f}%")
print(f"{'='*60}\n")

# 保存数据集
with open("3training_0_011.json", "w", encoding="utf-8") as f:
    json.dump(generated_dataset, f, ensure_ascii=False, indent=2)
print(f"✓ 训练数据集已保存至 3training_0_011.json (共 {len(generated_dataset)} 条)")

# --- 5. 训练部分 ---
random.shuffle(generated_dataset)
dataset = Dataset.from_list(generated_dataset)

# --- 全量 Token 统计 ---
print(f"正在启动全量统计，共 {len(dataset)} 条数据...")

def count_tokens(example):
    return {"token_length": len(example["input_ids"])}

token_stats = dataset.map(
    count_tokens,
    num_proc=os.cpu_count(),
    remove_columns=dataset.column_names,
    desc="正在全量计算 Token 长度"
)

all_lengths = np.array(token_stats["token_length"])

max_len = all_lengths.max()
avg_len = all_lengths.mean()
over_1024 = (all_lengths > 1024).sum()

print(f"\n" + "="*50)
print(f"全量统计报告 (总计: {len(all_lengths)} 条):")
print(f"  - 最大长度: {max_len}")
print(f"  - 平均长度: {avg_len:.2f}")
print(f"  - 超过 1024 Token 的样本数: {over_1024}")
print(f"  - 超过 1024 的比例: {(over_1024 / len(all_lengths) * 100):.2f}%")
print(f"  - 建议 max_seq_length: {1024 if max_len <= 1024 else 2048 if max_len <= 2048 else 4096}")
print("="*50 + "\n")

if over_1024 > 0:
    print(f"⚠️ 注意：由于存在 {over_1024} 条超长样本，建议将 max_seq_length 调整为符合最大长度的值。")

model.config.output_hidden_states = True
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    optim="paged_adamw_8bit",
    learning_rate=5e-5,
    bf16=True,
    logging_steps=10,
    report_to="none",
)

trainer = NoLogitMAEHybridTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field=None,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

print("🚀 开始微调训练...")
trainer.train()
model.save_pretrained("llama-3-1-8b-subway-0-mae011")
print("✓ 任务圆满完成！")

# --- 显存清理逻辑 ---
print("正在清理显存...")

del model
del trainer
if 'dataset' in locals(): del dataset
if 'token_stats' in locals(): del token_stats

import gc
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()

print("✓ 显存已释放，任务圆满完成！")