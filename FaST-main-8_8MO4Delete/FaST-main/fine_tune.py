# 微调代码
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

# 1. 禁用 Unsloth 的报错拦截
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
print("✓ 正在启动【隐状态截获版】MAE 混合训练...")
# --- 2. 路径配置 ---
# BASE_DATA_PATH = "/home/user/FSTLLM/own/data/FSTLLM-main/FSTLLM_STGNN/data/"
# NOTTINGHAM_H5_PATH = BASE_DATA_PATH + "subway.h5"
# YTRUE_TRAIN_PATH = BASE_DATA_PATH + "ytrue_train.npz"
# YPRED_TRAIN_PATH = BASE_DATA_PATH + "ypred_train.npz"
# TEXT_FILES_PATH = "/home/user/FSTLLM/own/data/FSTLLM-main/"
# CARPARK_DES_PATH = TEXT_FILES_PATH + "station_list.txt"
# NATURAL_PATTERN_PATH = TEXT_FILES_PATH + "station_natural_list_quan.txt"
# WEATHER_DATA_PATH = TEXT_FILES_PATH + "weather_data.csv"
# EVENTS_CSV_PATH = TEXT_FILES_PATH + "events_list_quan.csv"
BASE_DATA_PATH = "/home/user/Downloads/cai/FaST-main/cai-config/ture/"
NOTTINGHAM_H5_PATH = BASE_DATA_PATH + "hngs_his_2023.h5"  # 要改 √
YTRUE_TRAIN_PATH = BASE_DATA_PATH + "ytrue_train.npz"     # 要改 √
YPRED_TRAIN_PATH = BASE_DATA_PATH + "ypred_train.npz"     # 要改 √    验证机
TEXT_FILES_PATH = "/home/user/Downloads/cai/FaST-main/cai-config/ture/"
CARPARK_DES_PATH = TEXT_FILES_PATH + "station_list_hngs.txt" # 站点的信息描述    不用改
NATURAL_PATTERN_PATH = TEXT_FILES_PATH + "station_traffic_list.txt"  #站点的交通信息描述 不用改
WEATHER_DATA_PATH = TEXT_FILES_PATH + "station_weather_data.csv"  #湖南天气描述  天气还要改
EVENTS_CSV_PATH = TEXT_FILES_PATH + "station_events_list.csv"  #特殊事件描述  要核对 

# --- 3. 数据加载与天气预处理 (仅保留天气代码) ---
print("正在加载数据并精简天气信息...")
parking_data = pd.read_hdf(NOTTINGHAM_H5_PATH)

# true_array = np.load(YTRUE_TRAIN_PATH)['arr_0']
# data = np.load(YPRED_TRAIN_PATH)
# print("文件里包含的所有键：", list(data.keys()))  # 打印所有可用的键
true_array = np.load(YTRUE_TRAIN_PATH)['ytrue_train']

# pred_array = np.load(YPRED_TRAIN_PATH)['arr_0']
# data = np.load(YPRED_TRAIN_PATH)
# print("文件里包含的所有键：", list(data.keys()))  # 打印所有可用的键
pred_array = np.load(YPRED_TRAIN_PATH)['prediction']

weather_df = pd.read_csv(WEATHER_DATA_PATH, encoding="utf-8-sig")

# 只匹配天气代码列
try:
    code_col = [c for c in weather_df.columns if "天气代码" in c][0]
    print(f"✓ 成功匹配天气列: {code_col}")
except IndexError:
    print("❌ 错误：未找到'天气代码'列，请检查 CSV 标题。")
    exit()

weather_df['日期'] = pd.to_datetime(weather_df['日期'])
# 天气描述只保留代码
weather_df['weather_desc'] = weather_df[code_col].astype(str)
weather_df = weather_df.set_index('日期').sort_index()

with open(CARPARK_DES_PATH, "r", encoding='utf-8') as f:
    carpark_des_list = [l.strip() for l in f if l.strip()]
with open(NATURAL_PATTERN_PATH, "r", encoding='utf-8') as f:
    natural_pattern_list = [l.strip() for l in f if l.strip()]

if true_array.shape[0] < true_array.shape[1]:
    true_array = true_array.transpose(1, 2, 0)
    pred_array = pred_array.transpose(1, 2, 0)

# print("true_array 真实形状:", true_array.shape)   # 验证维度是多少   改
# num_samples, num_stations, _ = true_array.shape
# 只取前 3 个维度
num_samples, num_stations, _ = true_array.shape[:3]
print(f"✓ 维度对齐完成: 样本数 {num_samples}, 站点数 {num_stations}")

def get_weather_at_time(target_time):
    try:
        idx = weather_df.index.get_indexer([target_time], method='pad')[0]
        return weather_df.iloc[idx]['weather_desc'] if idx != -1 else "Unknown"
    except:
        return "Unknown"
# def load_precise_events(path):
#     if not os.path.exists(path):
#         print(f"⚠️ 未找到事件文件 {path}，将按无事件模式运行。")
#         return {}
#     try:
#         df_ev = pd.read_csv(path, encoding="gbk")   # 编码问题
#         df_ev['日期'] = pd.to_datetime(df_ev['日期']).dt.strftime('%Y-%m-%d')
#         # 构建 (日期, 站点名称) -> 事件内容 的映射
#         event_dict = {}
#         for _, row in df_ev.iterrows():
#             key = (row['日期'], str(row['站点名称']).strip())
#             event_dict[key] = str(row['事件描述']).strip()
#         print(f"✓ 已加载 {len(event_dict)} 条精准事件数据。")
#         return event_dict
#     except Exception as e:
#         print(f"❌ 加载事件失败: {e}")
#         return {}
def load_precise_events(path):
    if not os.path.exists(path):
        print(f"⚠️ 未找到事件文件 {path}")
        return {}

    try:
        # 1. 用 GBK 读取（你的文件是中文）
        df_ev = pd.read_csv(path, encoding="gbk")

        # 2. 强制统一日期格式为 %Y-%m-%d
        df_ev['日期'] = pd.to_datetime(df_ev['日期']).dt.strftime("%Y-%m-%d")

        # 3. 站点名统一去空格 + 转字符串
        df_ev['站点名称'] = df_ev['站点名称'].astype(str).str.strip()

        # 4. 打印前3条事件 → 让你确认格式是否正确
        print("📌 事件示例：", list(zip(df_ev['日期'].head(3), df_ev['站点名称'].head(3))))

        event_dict = {}
        for _, row in df_ev.iterrows():
            key = (row['日期'], row['站点名称'])
            event_dict[key] = str(row['事件描述']).strip()

        return event_dict

    except Exception as e:
        print(f"❌ 加载事件失败: {e}")
        return {}
precise_events_map = load_precise_events(EVENTS_CSV_PATH)

# --- 4. 自定义混合 MAE Loss Trainer (修正缩进与参数) ---
class NoLogitMAEHybridTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 1. 获取 0-9 数字在词表中的 ID 及其带空格的变体
        raw_digits = [str(i) for i in range(10)]
        space_digits = [f" {i}" for i in range(10)]

        all_digit_list = []
        self.id_to_val = {}
        for i in range(10):
            tid_raw = self.tokenizer.convert_tokens_to_ids(raw_digits[i])
            if tid_raw != self.tokenizer.unk_token_id:
                all_digit_list.append(tid_raw)
                self.id_to_val[tid_raw] = float(i)

            tid_space = self.tokenizer.convert_tokens_to_ids(space_digits[i])
            if tid_space != self.tokenizer.unk_token_id:
                all_digit_list.append(tid_space)
                self.id_to_val[tid_space] = float(i)

        # 存储所有可能的数字 Token ID
        self.digit_ids = torch.tensor(list(set(all_digit_list))).to(self.args.device)
        # 锁定答案框左括号 [ 的 ID (根据扫描结果为 518)
        self.target_bracket_id = 518

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        # A. 显式要求返回隐状态并计算 Logits
        inputs["output_hidden_states"] = True
        outputs = model(**inputs)

        # 截获最后一层隐状态并手动投影
        last_hidden_state = outputs.hidden_states[-1]
        target_dtype = next(model.lm_head.parameters()).dtype
        # 这就是模型预测词的原始分数值，完全等同于 logits
        manual_logits = model.lm_head(last_hidden_state.to(target_dtype))

        # B. 基础交叉熵损失 (CE)
        ce_loss = outputs.loss if isinstance(outputs, dict) else outputs[0]

        # C. 精准定位：只在 [/INST] [ 之后提取数字
        labels = inputs.get("labels")
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = manual_logits[..., :-1, :].contiguous()

        # 1. 先扫描全场所有 0-9 的位置
        all_digit_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        for d_id in self.digit_ids:
            all_digit_mask |= (shift_labels == d_id)

        # 2. 基于 [/INST] [ 锚点寻找起点 (锁定最后一个 ID 518)
        digit_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        for b in range(shift_labels.shape[0]):
            # 找到序列中最后出现的那个 '[' (ID 518)
            bracket_pos = (shift_labels[b] == self.target_bracket_id).nonzero(as_tuple=True)[0]
            if len(bracket_pos) > 0:
                # 从最后一个 '[' 的下一位开始抓数字
                start_from = bracket_pos[-1].item() + 1
                digit_mask[b, start_from:] = all_digit_mask[b, start_from:]

        # 排除填充位 -100
        digit_mask &= (shift_labels != -100)

        # D. 计算 MAE
        mae_loss = torch.tensor(0.0).to(self.args.device)
        total_loss = ce_loss
        expected_values = torch.tensor([]).to(self.args.device)

        if digit_mask.any():
            digit_logits = shift_logits[digit_mask][:, self.digit_ids]
            probs = torch.softmax(digit_logits.to(torch.float32), dim=-1)

            val_vector = torch.tensor(
                [self.id_to_val[int(tid)] for tid in self.digit_ids]
            ).to(self.args.device).to(torch.float32)

            expected_values = (probs * val_vector).sum(dim=-1)

            target_labels = shift_labels[digit_mask]
            target_values = torch.zeros_like(expected_values)
            for tid, val in self.id_to_val.items():
                target_values[target_labels == tid] = val

            # --- 核心：复刻 masked_mae_np 逻辑 ---
            null_val = 0.0
            # 5.1 创建 Mask (np.not_equal)
            mask = torch.not_equal(target_values, null_val).float()
            # 5.2 权重归一化 (mask /= np.mean(mask))
            mask_mean = torch.mean(mask)
            if mask_mean > 0: mask = mask / mask_mean
            # 5.3 计算绝对误差并应用掩码
            abs_error = torch.abs(expected_values - target_values)
            mae_loss = torch.mean(abs_error * mask)

            total_loss = ce_loss + 3.0 * mae_loss

        # --- E. 还原 8 个数值 + 修复 IndexError 的原文预览 ---
        if self.state.global_step % 10 == 0:
            sample_idx = 0
            s_labels = shift_labels[sample_idx].cpu().numpy()
            b_indices = (s_labels == self.target_bracket_id).nonzero()[0]

            final_true_nums, final_pred_nums = [], []
            raw_text_only = ""

            if len(b_indices) > 0:
                start = b_indices[-1]
                # 核心修复：剔除 -100 以防止 decode 报错
                relevant_ids = s_labels[start:]
                safe_ids_for_decode = [tid for tid in relevant_ids if tid != -100]

                # 解码原文预览
                raw_text_only = self.tokenizer.decode(safe_ids_for_decode, skip_special_tokens=False).split('</s>')[
                                    0] + " </s>"

                # 数值聚合逻辑 (与之前一致)
                temp_true, temp_pred = [], []
                val_ptr = 0
                all_prev_bits = digit_mask[:sample_idx].sum().item()

                for tid in relevant_ids[1:]:  # 跳过 [
                    if tid == 2 or tid == 29962: break

                    if tid in self.digit_ids:
                        temp_true.append(self.id_to_val[int(tid)])
                        if (all_prev_bits + val_ptr) < len(expected_values):
                            temp_pred.append(expected_values[all_prev_bits + val_ptr].item())
                            val_ptr += 1
                    elif (tid == 29892 or tid == 29871) and len(temp_true) > 0:
                        t_num = sum(d * (10 ** (len(temp_true) - 1 - j)) for j, d in enumerate(temp_true))
                        p_num = sum(d * (10 ** (len(temp_pred) - 1 - j)) for j, d in enumerate(temp_pred))
                        final_true_nums.append(int(t_num));
                        final_pred_nums.append(round(p_num, 1))
                        temp_true, temp_pred = [], []

                if len(temp_true) > 0:
                    t_num = sum(d * (10 ** (len(temp_true) - 1 - j)) for j, d in enumerate(temp_true))
                    p_num = sum(d * (10 ** (len(temp_pred) - 1 - j)) for j, d in enumerate(temp_pred))
                    final_true_nums.append(int(t_num));
                    final_pred_nums.append(round(p_num, 1))

            print(f"\n" + "=" * 12 + " 核心客流对账单 " + "=" * 12)
            print(f"Step: {self.state.global_step}")
            print(f"原文预览: {raw_text_only}")
            print(f"预测客流: {final_pred_nums[:8]}")
            print(f"真实客流: {final_true_nums[:8]}")
            print("-" * 60)
            print(f"Loss Total: {total_loss.item():.4f} | CE: {ce_loss.item():.4f} | MAE: {mae_loss.item():.4f}")
            print("=" * 30 + "\n")

        return (total_loss, outputs) if return_outputs else total_loss
        # --- 4. 生成训练数据集 ---
'''
instruction_text = (
    '<s>[INST] Role: Subway Prediction Refiner '
    'Objective: Fine-tune GNN simulation (5) based on external factors. '
    'Input Data: '
    '(1) Station: {} '
    '(2) Period: {} to {} (Day Type: {}). '
    '(3) Flow Patterns: {} (Max Range: {}). '
    '(4) Weather: {}. '
    '(5) GNN simulation: {}. '
    '(6) {}. '
    'Refinement Rules (Strict Priority):'
    '1. **EVENT SURGE (Highest Priority)**: If (6) is Special Event: '
    '- If (6) contains "(low occupancy)": Identify the morning peak hours defined in (3) and scale DOWN the corresponding values in (5). '
    '- If (6) mentions specific magnitudes, replace the corresponding time steps in (5) with these exact values, regardless of the GNN trend. '
    '- If (6) mentions specific times (e.g., "surge at 22:15") but NO specific values, LIFT the flow for those time steps in (5) up to the **Max Range** defined in (3). '
    '- If (6) has NO "( )" note: Identify the typical peak hours in (3) and scale UP the corresponding values in (5) to the Max Range. '
    '2. If (6) is "None":'
    '- Lift any value in (5) that is below the historical average in (3) during peak hours. '
    '- Ensure no value exceeds the Max Range in (3).'
    '3. If (4) is "Severe" (中雨/大雨), evaluate the potential reduction in flow and slightly scale DOWN the simulation (5). '
    '4. If no conditions in Rules 1-3 are met, you MUST output (5) exactly without any changes. '
    'Final Requirement: Output exactly 8 integers inside []. No commentary. [/INST] [ '
)
'''
instruction_text = (
    '<s>[INST] Role: Subway Prediction Refiner '
    'Objective: Fine-tune GNN simulation (5) based on external factors. '
    'Input Data: '
    '(1) Station: {} '
    '(2) Period: {} to {} (Day Type: {}). '
    '(3) Flow Patterns: {} (Max Range: {}). '
    '(4) Weather: {}. '
    '(5) GNN simulation: {}. '
    '(6) {}. '
    'Please synthesize all the above factors to accurately optimize the GNN simulation results. '  
    'Final Requirement: Output exactly 8 integers inside []. No commentary. [/INST] [ '
)
print("正在生成数据集：仅保留天气代码 + 数值取整 + 日期切换检测...")
generated_dataset = []
# num_samples, num_stations, _ = true_array.shape
num_samples, num_stations, _ = true_array.shape[:3]
# 我们只取前一部分真正有变化的数据
actual_valid_samples = num_samples
# --- 设置挑选阈值 ---
# MAE 阈值：如果 8 个时间步的平均误差超过此值，则认为 GNN 预测错误，加入训练集
ERROR_THRESHOLD = 40.0  # 显著误差阈值
NORMAL_SAMPLE_RATIO = 0.015  # 平常样本抽取比例 (10%)

for j in range(min(34, num_stations)):
    pattern_str = natural_pattern_list[j]
    full_description = carpark_des_list[j]
    station_name = full_description.split('-')[0].strip() if '-' in full_description else full_description.strip()

    ranges = re.findall(r"range 0-(\d+)", pattern_str)
    work_max = ranges[0] if len(ranges) > 0 else "0"
    off_max = ranges[1] if len(ranges) > 1 else work_max

    for i in range(num_samples):
        try:
            s1, s2 = i, 8 + i
            if s2 + 7 >= len(parking_data.index): break

            # 过滤掉数据源末尾的平稳填充区
            if i > 0 and np.array_equal(true_array[i, j, :], true_array[i - 1, j, :]):
                continue

            t3 = parking_data.index[s2]
            date_key = t3.strftime('%Y-%m-%d')
            event_info = precise_events_map.get((date_key, station_name), "None")

            pred_raw = pred_array[i, j, :]
            truth_raw = true_array[i, j, :]
            current_mae = np.mean(np.abs(pred_raw - truth_raw))

            # --- 样本类型判定 ---
            is_special_event = (event_info != "None")
            is_prediction_error = (current_mae > ERROR_THRESHOLD)

            # 决定是否保留该样本
            keep_sample = False
            reason = ""

            if is_special_event:
                keep_sample = True
                reason = "Event"
            elif is_prediction_error:
                keep_sample = True
                reason = "Error"
            elif random.random() < NORMAL_SAMPLE_RATIO:
                # 只有当前两项都不满足时，才按比例随机抽取平常样本
                keep_sample = True
                reason = "Normal"

            if keep_sample:
                p_is_work = is_workday(t3)
                day_type = "Workday" if p_is_work else "Off-day"
                current_max = work_max if p_is_work else off_max
                p_wea = get_weather_at_time(t3)

                pred = [int(round(float(v))) for v in pred_raw]
                truth = [int(round(float(v))) for v in truth_raw]

                prompt = instruction_text.format(
                    carpark_des_list[j], t3, parking_data.index[s2 + 7], day_type,
                    pattern_str, current_max, p_wea, pred, event_info
                )

                answer_part = ", ".join([str(v) for v in truth]) + "] </s>"
                generated_dataset.append({"text": prompt + answer_part, "reason": reason})

        except Exception:
            continue

# --- 统计输出逻辑 ---
event_count = sum(1 for d in generated_dataset if d['reason'] == "Event")
error_count = sum(1 for d in generated_dataset if d['reason'] == "Error")
normal_count = sum(1 for d in generated_dataset if d['reason'] == "Normal")

print(f"\n" + "=" * 50)
print(f"✓ 筛选完成。总样本: {len(generated_dataset)}")
print(f"  - 特殊事件样本 (Event): {event_count}")
print(f"  - 显著误差样本 (Error): {error_count}")
print(f"  - 随机平常样本 (Normal): {normal_count}")
print(f"  - 平常样本占比: {(normal_count / len(generated_dataset) * 100):.2f}%")
print("=" * 50 + "\n")
with open("training_data_check.json", "w", encoding="utf-8") as f:
    json.dump(generated_dataset, f, ensure_ascii=False, indent=2)
print(f"✓ 训练数据集已保存至 training_data_check.json (共 {len(generated_dataset)} 条)")
# --- 5. 训练部分 ---
random.shuffle(generated_dataset)
dataset = Dataset.from_list(generated_dataset)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/user/models--NousResearch--Llama-2-7b-chat-hf",
    max_seq_length=1024,
    load_in_4bit=False,
    dtype=torch.bfloat16,
    device_map={"": 0},
)

# --- 5. 全量 Token 统计 ---
print(f"正在启动全量统计，共 {len(dataset)} 条数据...")

# 定义一个只计算长度的函数
def count_tokens(example):
    return {"token_length": len(tokenizer.encode(example["text"]))}

# 使用多进程进行全量映射统计
token_stats = dataset.map(
    count_tokens,
    num_proc=os.cpu_count(), # 使用所有 CPU 核心（即你之前看到的 196）
    remove_columns=dataset.column_names, # 只保留 token_length 列
    desc="正在全量计算 Token 长度"
)

# 转换为 numpy 进行快速计算
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
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
    packing=False, # 必须设为 False 才能让 shift_labels 的数值位置对齐
)

print("🚀 开始微调训练...")
trainer.train()
model.save_pretrained("llama-2-7b-subway-quan2")
print("✓ 任务圆满完成！")
# --- 核心：显存清理逻辑 ---
print("正在清理显存...")

# 2. 移除引用
del model
del trainer
if 'dataset' in locals(): del dataset
if 'token_stats' in locals(): del token_stats

# 3. 强制垃圾回收
import gc
gc.collect()

# 4. 清空 PyTorch 缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect() # 清理进程间通信残留
    torch.cuda.synchronize() # 强制同步等待所有 CUDA 核心停止任务

print("✓ 显存已释放，任务圆满完成！")
# python fine_tune.py