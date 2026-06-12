# 推理代码
import os
import logging
import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
from peft import PeftModel
from chinese_calendar import is_workday # 必须与微调保持一致

# 1. 环境变量与日志拦截
os.environ["UNSLOTH_SKIP_INIT_CHECK"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

class UnslothCrashInterceptor(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return False if ("weights" in msg or "inv_freq" in msg) else True

logging.getLogger("transformers.modeling_utils").addFilter(UnslothCrashInterceptor())

# 2. 路径配置
# BASE_DATA_PATH = "/home/user/FSTLLM/own/data/FSTLLM-main/FSTLLM_STGNN/data/"
# TEXT_FILES_PATH = "/home/user/FSTLLM/own/data/FSTLLM-main/"
# model_name = "/home/user/models--NousResearch--Llama-2-7b-chat-hf"
# lora_model_path = "llama-2-7b-subway-quan2"  # 指向最新的微调模型
# WEATHER_DATA_PATH = TEXT_FILES_PATH + "weather_data.csv"
# EVENTS_CSV_PATH = TEXT_FILES_PATH + "events_list_quan.csv"
BASE_DATA_PATH = "/home/user/Downloads/cai/FaST-main-8_8/FaST-main/cai-config/ture/"  FaST-main-8_8
TEXT_FILES_PATH = "/home/user/Downloads/cai/FaST-main-8_8/FaST-main/cai-config/ture/"
model_name = "/home/user/models--NousResearch--Llama-2-7b-chat-hf"
lora_model_path = "/home/user/FSTLLM/own/data/FSTLLM-main/llama-2-7b-subway-quan2"  # 指向最新的微调模型
WEATHER_DATA_PATH = TEXT_FILES_PATH + "station_weather_data.csv"
EVENTS_CSV_PATH = TEXT_FILES_PATH + "station_events_list.csv"

# 3. 加载描述文件
with open(TEXT_FILES_PATH + "station_list_hngs.txt", "r", encoding='utf-8') as f:
    carpark_des_list = [l.strip() for l in f if l.strip()]
with open(TEXT_FILES_PATH + "station_traffic_list.txt", "r", encoding='utf-8') as f:
    natural_pattern_list = [l.strip() for l in f if l.strip()]

# --- 天气数据预处理 (仅保留天气代码) ---
weather_df = pd.read_csv(WEATHER_DATA_PATH)
code_col = [c for c in weather_df.columns if "天气代码" in c][0]
weather_df['日期'] = pd.to_datetime(weather_df['日期'])
weather_df['weather_desc'] = weather_df[code_col].astype(str)
weather_df = weather_df.set_index('日期').sort_index()

def get_weather_at_time(target_time):
    try:
        idx = weather_df.index.get_indexer([target_time], method='pad')[0]
        return weather_df.iloc[idx]['weather_desc'] if idx != -1 else "Unknown"
    except:
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
# 4. 初始化模型
print("🚀 正在初始化推理引擎并加载 LoRA 权重...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=1024,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, lora_model_path)
FastLanguageModel.for_inference(model)
tokenizer.padding_side = "left"

# 5. 加载客流数据
# parking_data = pd.read_hdf(BASE_DATA_PATH + 'subway.h5')
parking_data = pd.read_hdf(BASE_DATA_PATH + 'hngs_his_2023.h5')

# pred_data = np.load(BASE_DATA_PATH + 'ypred.npy.npz')['arr_0'] # 仿真预测数据
pred_data = np.load(BASE_DATA_PATH + 'ypred.npy.npz')['arr_0'] # 仿真预测数据

timestamps = parking_data.index.tolist()
'''
# 同步微调时的指令模板 (包含 Shift 逻辑描述占位符)
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
)'''
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
# --- 修改后的推理数据填充逻辑 ---
prompts = []
meta_data = []
print("📊 正在处理推理数据 (精简模式：对齐微调逻辑)...")

# 确保推理脚本中也导入了 re 模块
import re
num_stations = len(carpark_des_list)
test_start_offset = 10421    #2184
for j in range(len(carpark_des_list)):
    pattern_str = natural_pattern_list[j]
    full_description = carpark_des_list[j]
    station_name = full_description.split('-')[0].strip() if '-' in full_description else full_description.strip()

    # 1. 预先解析该站点的两个最大值 (逻辑与微调完全一致)
    # 匹配 Workday 后的 range 数值 和 Off-day 后的 range 数值
    ranges = re.findall(r"range 0-(\d+)", pattern_str)
    work_max = int(ranges[0]) if len(ranges) > 0 else 0
    off_max = int(ranges[1]) if len(ranges) > 1 else work_max

    for i in range(2603):#544
        try:
            # 这里的计算逻辑维持您原推理脚本的索引方式
            idx_start = test_start_offset - 8 + i
            t1 = timestamps[idx_start]
            t3, t4 = timestamps[idx_start + 9], timestamps[idx_start + 16]

            # 精准匹配：只有该日期+该站点匹配上，才有值
            date_key = t3.strftime('%Y-%m-%d')
            event_info = precise_events_map.get((date_key, station_name), "None")

            # 2. 判定预测日期的类型 (Workday/Off-day)
            p_is_work = is_workday(t3)
            day_type = "Workday" if p_is_work else "Off-day"

            # 3. 选择对应的 Max Range
            current_max = work_max if p_is_work else off_max
            current_limit = current_max if event_info == "None" else current_max * 1.5
            # 4. 获取天气代码
            hist_weather = get_weather_at_time(t1)
            pred_weather = get_weather_at_time(t3)

            # 5. 仿真预测数值取整
            # 注意：此处 pred_data 维度对应关系根据您的原始脚本 [horizon, sample, station]
            pred = [int(round(float(v))) for v in pred_data[:, i, j]]

            # 6. 填充精简指令 (严格匹配微调模板的 9 个占位符)
            # (1)Station, (2)t3, (2)t4, (2)day_type, (3)pattern, (3)max, (4)h_wea, (4)p_wea, (5)pred
            prompts.append(instruction_text.format(
                carpark_des_list[j],  # (1) Station
                t3, t4, day_type,  # (2) Period (开始, 结束, 类型)
                pattern_str, current_max,  # (3) Flow Patterns & Max Range
                pred_weather,  # (4) Weather (预测)
                pred , # (5) GNN Baseline
                event_info
            ))
            meta_data.append({"limit": current_limit, "baseline": pred, "has_event": event_info != "None"})
        except IndexError:
            break

# 6. 执行批量推理
print(f"✅ 准备就绪，开始推理 {len(prompts)} 条数据...")
answer_list = []
batch_size = 32


def post_process_safe_value(resp_text, meta, has_event):
    try:
        nums = [int(n) for n in re.findall(r'\d+', resp_text)]
        if len(nums) != 8: return str(meta["baseline"])

        final_nums = []
        limit_val = meta["limit"]

        for k in range(8):
            val = nums[k]
            baseline_val = meta["baseline"][k]

            # 1. 物理上限拦截（不论是否有事件，都不允许超过闸机极限）
            if val > limit_val:
                final_nums.append(min(val, int(limit_val)))
                continue

            # 2. 特殊事件放行逻辑：如果是事件样本，跳过倍数拦截
            if has_event:
                final_nums.append(val)
                continue

            # 3. 常规样本拦截逻辑
            if baseline_val <= 50:
                if val > baseline_val * 5 and val > 20:
                    final_nums.append(baseline_val)
                else:
                    final_nums.append(val)
            else:
                if val > baseline_val * 4 or val < baseline_val * 0.3:
                    final_nums.append(baseline_val)
                else:
                    final_nums.append(val)

        return str(final_nums)
    except:
        return str(meta["baseline"])
for i in tqdm(range(0, len(prompts), batch_size)):
    batch = prompts[i: i + batch_size]
    batch_meta = meta_data[i: i + batch_size]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            use_cache=True,
            do_sample=False, # 数值预测通常建  False 以获得确定性结果
            num_beams=1,
            #temperature=0.1,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        decoded_answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for idx, resp in enumerate(decoded_answers):
            # 执行安全截断逻辑
            safe_resp = post_process_safe_value(resp.strip(), batch_meta[idx], batch_meta[idx]["has_event"])
            answer_list.append(batch[idx] + safe_resp + " </s>")

# 7. 保存结果
with open("inference_results_quan2.json", "w", encoding="utf-8") as file:
    json.dump(answer_list, file, ensure_ascii=False, indent=2)

print(f"🎉 推理完成！结果保存至 inference_results_quan2.json")
# python inference.py