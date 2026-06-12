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
    def __init__(self, filename="3training_true0_011.txt"):
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
sys.stdout = Logger("3training_true0_011.txt.txt")
# 1. 禁用 Unsloth 的报错拦截
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
print("✓ 正在启动【隐状态截获版】MAE 混合训练...")
# --- 2. 路径配置 ---
BASE_DATA_PATH = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"
NOTTINGHAM_H5_PATH = BASE_DATA_PATH + "hngs.h5"
YTRUE_TRAIN_PATH = BASE_DATA_PATH + "ytrue_train.npz"
YPRED_TRAIN_PATH = BASE_DATA_PATH + "ypred_train.npz"
TEXT_FILES_PATH = BASE_DATA_PATH
CARPARK_DES_PATH = TEXT_FILES_PATH + "station_list_hngs.txt"
NATURAL_PATTERN_PATH = TEXT_FILES_PATH + "station_natural_list_4feat.txt"
WEATHER_DATA_PATH = TEXT_FILES_PATH + "160站点天气信息/"
EVENTS_CSV_PATH = TEXT_FILES_PATH + "events_list_quan.csv"
HIS_DATA_CSV_PATH = TEXT_FILES_PATH + "his_data_with_index.csv"

# --- 3. 数据加载与天气预处理 (仅保留天气代码) ---
print("正在加载数据并精简天气信息...")

# 加载历史数据CSV获取时间索引
his_df = pd.read_csv(HIS_DATA_CSV_PATH)
his_df['时间'] = pd.to_datetime(his_df['时间'])
parking_data = his_df.set_index('时间').sort_index()

# 加载NPZ格式的预测和真实值
true_array = np.load(YTRUE_TRAIN_PATH)['arr_0']
pred_array = np.load(YPRED_TRAIN_PATH)['arr_0']

# 加载站点描述
with open(CARPARK_DES_PATH, "r", encoding='utf-8') as f:
    carpark_des_list = [l.strip() for l in f if l.strip()]

# 加载自然语言模式描述（4个特征）
with open(NATURAL_PATTERN_PATH, "r", encoding='utf-8') as f:
    natural_pattern_list = [l.strip() for l in f if l.strip()]

# 调整数组维度：确保格式为 (样本数, 站点数, 特征数, 时间步)
if true_array.ndim == 3:
    # 如果已经是 (样本, 站点, 时间)，需要添加特征维度
    # 假设原始数据是 (样本, 站点*特征, 时间) 或 (样本, 站点, 时间)
    print(f"原始数据形状: {true_array.shape}")
    # 根据实际数据结构调整，这里假设需要转置
    if true_array.shape[0] < true_array.shape[1]:
        true_array = true_array.transpose(1, 2, 0)
        pred_array = pred_array.transpose(1, 2, 0)
elif true_array.ndim == 4:
    # 如果已经是4维 (样本, 站点, 特征, 时间)
    print(f"4维数据形状: {true_array.shape}")
else:
    raise ValueError(f"不支持的数据维度: {true_array.ndim}")

num_samples, num_stations, num_features, seq_len = true_array.shape
print(f"✓ 维度对齐完成: 样本数 {num_samples}, 站点数 {num_stations}, 特征数 {num_features}, 序列长度 {seq_len}")

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

def load_precise_events(path):
    if not os.path.exists(path):
        print(f"⚠️ 未找到事件文件 {path}，将按无事件模式运行。")
        return {}
    try:
        df_ev = pd.read_csv(path)
        df_ev['日期'] = pd.to_datetime(df_ev['日期']).dt.strftime('%Y-%m-%d')
        # 构建 (日期, 站点名称) -> 事件内容 的映射
        event_dict = {}
        for _, row in df_ev.iterrows():
            key = (row['日期'], str(row['站点名称']).strip())
            event_dict[key] = str(row['事件描述']).strip()
        print(f"✓ 已加载 {len(event_dict)} 条精准事件数据。")
        return event_dict
    except Exception as e:
        print(f"❌ 加载事件失败: {e}")
        return {}

precise_events_map = load_precise_events(EVENTS_CSV_PATH)

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
            # 这样可以排除 '²' (Unicode digit), '½' (Unicode numeric) 等干扰
            if token_text and all('0' <= char <= '9' for char in token_text):
                try:
                    val = float(token_text)
                    self.id_to_val[i] = val
                    all_digit_ids.append(i)
                except ValueError:
                    continue  # 双重保险

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

            # --- E. 实时数值对账单 (硬核 Token 对齐版) ---
            if self.state.global_step % 1 == 0:
                sample_idx = 0
                s_labels = shift_labels[sample_idx].cpu().numpy()
                b_indices = (s_labels == self.target_bracket_id).nonzero()[0]

                if len(b_indices) > 0:
                    # 1. 解码全文预览
                    start_preview = max(0, b_indices[-1] - 150)
                    relevant_ids = s_labels[start_preview:]
                    safe_ids = [tid for tid in relevant_ids if tid != -100]
                    raw_text_only = self.tokenizer.decode(safe_ids, skip_special_tokens=False).split('<|eot_id|>')[
                                        0] + " <|eot_id|>"

                    # 2. 提取当前样本在 digit_mask 中的预测和标签
                    prev_mae_bits = digit_mask[:sample_idx].sum().item()
                    curr_mae_bits = digit_mask[sample_idx].sum().item()

                    if curr_mae_bits > 0:
                        batch_preds = expected_values[prev_mae_bits: prev_mae_bits + curr_mae_bits].cpu().tolist()
                        # 关键：我们要拿到方括号后所有的原始 Token ID，包括逗号
                        # 这样我们才能知道哪个预测碎片属于哪个数字
                        start_ptr = b_indices[-1].item() + 1
                        raw_token_seq = shift_labels[sample_idx, start_ptr:].cpu().tolist()

                        final_pred_res = []
                        final_true_res = []

                        current_p_str = ""
                        current_t_str = ""

                        pred_ptr = 0
                        for tid in raw_token_seq:
                            if tid == -100 or tid == 128009: break  # 结束

                            if tid in self.id_to_val:  # 如果是数字碎片
                                token_char = self.tokenizer.decode([tid]).strip()
                                current_t_str += token_char
                                # 对应的预测值
                                if pred_ptr < len(batch_preds):
                                    p_val = int(round(batch_preds[pred_ptr]))
                                    current_p_str += str(p_val)
                                    pred_ptr += 1

                            elif tid == 11 or tid == 60:  # 如果是逗号或右方括号，说明数字结束
                                if current_t_str:
                                    # 还原并转回整数，防止出现 9486 这种非法拼接
                                    # 如果 current_p_str 为空则填 0
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
    "(3) The prediction period: {} to {} (Day Type: {}, {}). "
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
        # 每个站点有4行pattern，对应4个特征
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
                
                # 查询事件（使用站点名称）
                event_info = precise_events_map.get((date_key, station_name), "None")
                
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
                    analysis.append(f"2. The prediction period is {weekday_str}, {cross_tag}{day_type_desc} {start_hm} to {end_hm} (Special Event: {has_event_label}).")
                    
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
                                analysis.append(
                                    f"3. Strategy: Special Event Mode. Detected historical peak point at {hit['time']}. "
                                    f"BUT due to {event_info_cleaned}, the peak sequence needs to be adjusted by {ratio:+.2f}%.")

                    # --- 原本的 Strategy 逻辑 ---
                    if not event_handled:
                        if low_peak_hits:
                            hit = low_peak_hits[0]
                            ref_val = hit["ref_val"]
                            gnn_at_low = gnn_vals[hit["step"]]
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
                                f"However, peak adjustments are not applied on Off-days, only minor adjustments are needed. "
                            )
                        else:
                            mode = "Transition" if is_type_transition else "Routine"
                            analysis.append(
                                f"3. Strategy: {day_type_desc} {mode} Mode. No historical peak or low-peak points detected, only minor adjustments are needed. ")

                    # --- 天气核心修改 ---
                    weather_input = get_weather_at_time(j, t3)
                    has_off_day = (not is_work_start) or (not is_work_end)

                    if not has_off_day:
                        weather_logic = f"4. Weather Factor: Weather correction is ignored in Workday mode."
                    else:
                        if any(w in weather_input for w in ['Moderate Rain', 'Heavy Rain', 'Moderate Snow']):
                            weather_logic = f"4. Weather Factor: {weather_input}. Adverse weather on Off-days reduces travel intention, suggesting a slight downward adjustment."
                        else:
                            weather_logic = f"4. Weather Factor: {weather_input}. Normal weather."

                    analysis.append(weather_logic)
                    
                    # --- 详细数值步长分析 ---
                    step_analysis = []
                    for k in range(8):
                        g_val = gnn_vals[k]
                        t_val = truth_raw[k]

                        if g_val > 0:
                            ratio = ((t_val - g_val) / g_val) * 100
                        else:
                            ratio = 100.0 if t_val > 0 else 0.0

                        time_str = gnn_time_points[k].strftime('%H:%M')

                        if abs(ratio) < 0.01:
                            step_analysis.append(
                                f"Step {k} ({time_str}): GNN {g_val} is accurate, remaining {int(round(t_val))}."
                            )
                        else:
                            action = "up-adjusted" if ratio > 0 else "down-adjusted"
                            step_analysis.append(
                                f"Step {k} ({time_str}): GNN {g_val} is {action} by {abs(ratio):.2f}% to become {int(round(t_val))}."
                            )

                    analysis.append("5. Detailed Step-by-Step Adjustment: " + " ".join(step_analysis))
                    analysis.append("Final Correction: [")
                    
                    # 拼接CoT文本
                    cot_text = " ".join(analysis)
                    final_values = ", ".join(map(str, [int(round(v)) for v in truth_raw])) + "] <|eot_id|>"

                    # 填充最终 Prompt (注意增加了特征类型字段)
                    prompt = instruction_text.format(
                        full_description, 
                        feature_name,  # 新增：特征类型
                        t3, t4, weekday_str, day_type_desc,
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
with open("3training_true0_011.json", "w", encoding="utf-8") as f:
    json.dump(generated_dataset, f, ensure_ascii=False, indent=2)
print(f"✓ 训练数据集已保存至 3training_true0_011.json (共 {len(generated_dataset)} 条)")

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
model.save_pretrained("llama-3-1-8b-subway-true0-mae011")
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