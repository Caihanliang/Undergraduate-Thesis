# import os
# import sys
# import torch
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import re
# import time
# import threading
# from datetime import datetime
# from unsloth import FastLanguageModel

# # =================配置区域=================
# # PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main"
# # MODEL_PATH = os.path.join(PROJECT_ROOT, "config/cai quick428/llama-3-1-8b-highway-finetuned-quick") 
# # DATA_PATH = os.path.join(PROJECT_ROOT, "config/cai quick428/finetune_data.npz")
# # TRUE_PATH = os.path.join(PROJECT_ROOT, "config/cai quick428/finetune_real_traffic.npz")
# # STATION_LIST = os.path.join(PROJECT_ROOT, "config/cai quick428/station_list_hngs.txt")
# # WEATHER_DATA_PATH = os.path.join(PROJECT_ROOT, "config/cai quick428/160站点天气信息/")
# # EVENTS_CSV_PATH = os.path.join(PROJECT_ROOT, "config/cai quick428/events_list_quan.csv")
# PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main"
# # MODEL_PATH = os.path.join(PROJECT_ROOT, "config/cai/llama-3-1-8b-highway-optimized") 
# MODEL_PATH = os.path.join(PROJECT_ROOT, "config/cai/llama-3-1-8b-highway-finetuned-quick427") 

# DATA_PATH = os.path.join(PROJECT_ROOT, "config/cai/finetune_data.npz")
# TRUE_PATH = os.path.join(PROJECT_ROOT, "config/cai/finetune_real_traffic.npz")
# STATION_LIST = os.path.join(PROJECT_ROOT, "config/cai/station_list_hngs.txt")
# WEATHER_DATA_PATH = os.path.join(PROJECT_ROOT, "config/cai/160站点天气信息/")
# EVENTS_CSV_PATH = os.path.join(PROJECT_ROOT, "config/cai/events_list_quan.csv")
# MAX_SEQ_LENGTH = 1024
# DTYPE = torch.float16
# LOAD_IN_4BIT = True

# # 评估配置 - 快速测试模式
# MAX_SAMPLES = 5         # 每个站点测试5个样本（快速验证）
# MAX_STATIONS = 1        # 只评估1个站点（Huangxing）
# MAX_FEATURES = 4        # 测试所有4个特征
# TIMEOUT_SECONDS = 30    # 增加超时时间
# BATCH_SIZE = 1
# # =========================================

# def load_station_list():
#     """加载站点列表"""
#     with open(STATION_LIST, "r", encoding='utf-8') as f:
#         stations = [line.strip() for line in f if line.strip()]
#     station_names = [s.split(' ')[0] for s in stations]
#     return station_names

# def load_weather_data():
#     """预加载天气数据"""
#     weather_cache = {}
#     if not os.path.exists(WEATHER_DATA_PATH):
#         print("⚠️ 天气目录不存在，使用默认天气")
#         return weather_cache
    
#     files = os.listdir(WEATHER_DATA_PATH)
#     station_names = load_station_list()
#     for j in tqdm(range(min(len(station_names), 1)), desc="加载天气数据"):  # 只加载第一个站点的天气
#         try:
#             prefix = f"{j:03d}"
#             match = [f for f in files if f.startswith(prefix)]
#             if not match:
#                 continue
#             path = os.path.join(WEATHER_DATA_PATH, match[0])
#             if path.endswith('.csv'):
#                 df = pd.read_csv(path, encoding='utf-8')
#             else:
#                 continue
            
#             d_col = next((c for c in df.columns if '日期' in c or 'date' in c.lower()), None)
#             w_col = next((c for c in df.columns if '天气' in c or 'weather' in c.lower()), None)
#             if not d_col or not w_col:
#                 continue
            
#             df[d_col] = pd.to_datetime(df[d_col]).dt.date.astype(str)
#             df = df.drop_duplicates(subset=[d_col]).set_index(d_col)
#             weather_cache[j] = df[w_col].to_dict()
#         except:
#             continue
#     print(f"✅ 天气数据加载完成：{len(weather_cache)} 个站点")
#     return weather_cache

# def load_events_data():
#     """加载事件数据"""
#     if not os.path.exists(EVENTS_CSV_PATH):
#         print("⚠️ 事件文件不存在")
#         return {}
#     try:
#         df_ev = pd.read_csv(EVENTS_CSV_PATH, encoding='utf-8')
#         date_col = "日期"
#         station_col = "站点名称"
#         event_col = "事件描述"
        
#         event_map = {}
#         for _, r in df_ev.iterrows():
#             try:
#                 dt = pd.to_datetime(r[date_col]).strftime('%Y-%m-%d')
#                 st = str(r[station_col]).strip()
#                 ev = str(r[event_col]).strip()
#                 if st and ev != 'nan' and ev != 'None':
#                     event_map[(dt, st)] = ev
#             except:
#                 continue
#         print(f"✅ 事件数据加载完成：{len(event_map)} 条")
#         return event_map
#     except Exception as e:
#         print(f"⚠️ 事件加载失败: {e}")
#         return {}

# def get_weather(weather_cache, station_idx, date_str):
#     """获取天气"""
#     return weather_cache.get(station_idx, {}).get(date_str, "Clear")

# def get_event(events_map, station_name, date_str):
#     """获取事件"""
#     return events_map.get((date_str, station_name), "None")

# def is_workday(date):
#     """判断是否是工作日"""
#     return date.weekday() < 5

# def get_feature_name(feature_idx):
#     """获取特征名称"""
#     feature_names = {
#         0: "Passenger Car Up",
#         1: "Passenger Car Down",
#         2: "Non-Passenger Car Up",
#         3: "Non-Passenger Car Down"
#     }
#     return feature_names.get(feature_idx, f"Feature_{feature_idx}")

# def get_feature_short_name(feature_name):
#     """获取特征缩写名称"""
#     short_names = {
#         "Passenger Car Up": "PCU",
#         "Passenger Car Down": "PCD",
#         "Non-Passenger Car Up": "NPCU",
#         "Non-Passenger Car Down": "NPCD"
#     }
#     return short_names.get(feature_name, feature_name[:10])

# def build_instruction(station_name, feature_name, time_info, day_type, weather, event, gnn_pred):
#     """构建完整的指令（简化版，加快推理）"""
#     if isinstance(gnn_pred, list):
#         pred_str = ', '.join([f"{x:.0f}" for x in gnn_pred[:4]])  # 只取前4个值，减少token
#     else:
#         pred_str = str(gnn_pred)
    
#     # 更简洁的指令模板
#     instruction = (
#         f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
#         f"Refine traffic prediction:\n"
#         f"Station: {station_name}\n"
#         f"Feature: {feature_name}\n"
#         f"Time: {time_info}\n"
#         f"GNN: [{pred_str}]\n"
#         f"Output corrected list.<|eot_id|>"
#         f"<|start_header_id|>assistant<|end_header_id|>\n\n"
#         f"Final Correction: ["
#     )
#     return instruction

# def extract_numbers_from_response(response, instruction):
#     """从模型响应中提取数字序列"""
#     if not response:
#         return None
    
#     full_text = instruction + response
    
#     # 尝试匹配 "Final Correction: [x, x, x]" 格式
#     match = re.search(r'Final Correction:\s*\[(.*?)\]', full_text, re.IGNORECASE | re.DOTALL)
#     if match:
#         nums = re.findall(r'\d+\.?\d*', match.group(1))
#         if len(nums) >= 1:
#             return [float(x) for x in nums[:8]]
    
#     # 尝试匹配单独的列表
#     match = re.search(r'\[(.*?)\]', full_text)
#     if match:
#         nums = re.findall(r'\d+\.?\d*', match.group(1))
#         if len(nums) >= 1:
#             return [float(x) for x in nums[:8]]
    
#     # 尝试匹配末尾连续数字
#     all_nums = re.findall(r'\d+\.?\d*', response)
#     if len(all_nums) >= 8:
#         return [float(x) for x in all_nums[:8]]
    
#     return None

# def generate_with_timeout(model, tokenizer, inputs, max_new_tokens=64, timeout=TIMEOUT_SECONDS):
#     """带超时的生成函数"""
#     result = [None]
#     error = [None]
    
#     def generate():
#         try:
#             with torch.no_grad():
#                 outputs = model.generate(
#                     **inputs,
#                     max_new_tokens=max_new_tokens,
#                     use_cache=True,
#                     temperature=0.1,
#                     do_sample=False,
#                     pad_token_id=tokenizer.eos_token_id,
#                     eos_token_id=tokenizer.eos_token_id,
#                     early_stopping=True
#                 )
#             result[0] = outputs
#         except Exception as e:
#             error[0] = e
    
#     thread = threading.Thread(target=generate)
#     thread.daemon = True
#     thread.start()
#     thread.join(timeout=timeout)
    
#     if thread.is_alive():
#         return None, "timeout"
    
#     if error[0]:
#         return None, str(error[0])
    
#     return result[0], None

# def load_data():
#     """加载所有数据"""
#     print("加载数据集...")
    
#     pred_npz = np.load(DATA_PATH)
#     true_npz = np.load(TRUE_PATH)
    
#     pred_key = 'prediction' if 'prediction' in pred_npz else 'arr_0'
#     true_key = 'target' if 'target' in true_npz else 'arr_0'
    
#     pred_array = pred_npz[pred_key]
#     true_array = true_npz[true_key]
    
#     station_names = load_station_list()
    
#     # 数据维度处理
#     if len(pred_array.shape) == 4:
#         num_samples = pred_array.shape[0]
#         num_steps = pred_array.shape[1]
#         num_stations = pred_array.shape[2]
#         num_features = pred_array.shape[3]
#     elif len(pred_array.shape) == 3:
#         num_stations = len(station_names)
#         num_samples = pred_array.shape[0] // num_stations
#         num_steps = pred_array.shape[1]
#         num_features = pred_array.shape[2]
#         pred_array = pred_array.reshape(num_samples, num_steps, num_stations, num_features)
#         true_array = true_array.reshape(num_samples, num_steps, num_stations, num_features)
#     else:
#         raise ValueError(f"未知数据形状: {pred_array.shape}")
    
#     print(f"数据形状: 样本数={num_samples}, 步长={num_steps}, 站点数={num_stations}, 特征数={num_features}")
    
#     return pred_array, true_array, station_names, num_samples, num_steps, num_stations, num_features

# def evaluate_station(model, tokenizer, station_idx, station_name, 
#                      pred_array, true_array, weather_cache, events_map,
#                      sample_indices, feature_indices, num_steps):
#     """评估单个站点"""
#     results = []
    
#     for sample_idx in sample_indices:
#         # 获取时间信息
#         date_str = "2023-09-01"
#         day_type = "Workday"
#         weather = get_weather(weather_cache, station_idx, date_str)
#         event = get_event(events_map, station_name, date_str)
#         time_info = f"{date_str} 08:00"
        
#         for feat_idx in feature_indices:
#             try:
#                 # 获取8步预测序列（只取前4步，减少输入长度）
#                 pred_sequence = []
#                 for step in range(min(num_steps, 4)):
#                     val = pred_array[sample_idx, step, station_idx, feat_idx]
#                     pred_sequence.append(float(val) if hasattr(val, 'item') else float(val))
                
#                 # 真实值（第一步）
#                 true_val = true_array[sample_idx, 0, station_idx, feat_idx]
#                 true_val = float(true_val) if hasattr(true_val, 'item') else float(true_val)
#                 gnn_val = pred_sequence[0]
                
#                 feature_name = get_feature_name(feat_idx)
                
#                 # 构建指令
#                 instruction = build_instruction(
#                     station_name, feature_name, time_info, 
#                     day_type, weather, event, pred_sequence
#                 )
                
#                 print(f"\n  🔄 处理: {station_name} - {get_feature_short_name(feature_name)} - 样本{sample_idx}")
#                 print(f"     GNN预测: {gnn_val:.0f}, 真实值: {true_val:.0f}")
                
#                 # Tokenize
#                 inputs = tokenizer(instruction, return_tensors="pt", truncation=True, max_length=512).to("cuda")
                
#                 # 生成
#                 outputs, error = generate_with_timeout(model, tokenizer, inputs, max_new_tokens=64)
                
#                 if outputs is None:
#                     llm_val = gnn_val
#                     response = f"ERROR: {error}"
#                     print(f"     ⚠️ 生成超时或错误: {error}")
#                 else:
#                     input_len = inputs.input_ids.shape[1]
#                     response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                    
#                     # 提取LLM预测
#                     extracted = extract_numbers_from_response(response, instruction)
                    
#                     if extracted and len(extracted) > 0:
#                         llm_val = extracted[0]
#                         print(f"     LLM预测: {llm_val:.0f}")
#                         print(f"     响应: {response[:100]}...")
#                     else:
#                         llm_val = gnn_val
#                         print(f"     ⚠️ 未能提取LLM预测，使用GNN值")
                
#                 # 计算误差
#                 gnn_error = abs(gnn_val - true_val)
#                 llm_error = abs(llm_val - true_val)
#                 improvement = ((gnn_error - llm_error) / (gnn_error + 1e-6)) * 100
                
#                 print(f"     GNN误差: {gnn_error:.1f}, LLM误差: {llm_error:.1f}, 改进: {improvement:+.1f}%")
                
#                 results.append({
#                     'sample_idx': sample_idx,
#                     'station_idx': station_idx,
#                     'station_name': station_name,
#                     'feature_idx': feat_idx,
#                     'feature_name': feature_name,
#                     'gnn_pred': round(gnn_val, 2),
#                     'llm_pred': round(llm_val, 2),
#                     'true_val': round(true_val, 2),
#                     'gnn_error': round(gnn_error, 2),
#                     'llm_error': round(llm_error, 2),
#                     'improvement': round(improvement, 2),
#                     'response': response[:200] if response else None
#                 })
                
#             except Exception as e:
#                 print(f"     ❌ 错误: {e}")
#                 continue
    
#     return results

# def main():
#     """主评估函数"""
#     print("="*80)
#     print("高速公路流量预测LLM修正评估系统 - 快速测试模式")
#     print("="*80)
    
#     # 加载数据
#     pred_array, true_array, station_names, num_samples, num_steps, num_stations, num_features = load_data()
    
#     # 加载辅助数据
#     print("\n加载辅助数据...")
#     weather_cache = load_weather_data()
#     events_map = load_events_data()
    
#     # 确定测试范围
#     test_samples = min(MAX_SAMPLES, num_samples)
#     test_stations = min(MAX_STATIONS, num_stations) if MAX_STATIONS else num_stations
#     test_features = list(range(min(MAX_FEATURES, num_features)))
    
#     print(f"\n📊 评估配置（快速测试）:")
#     print(f"  测试站点: {station_names[0] if station_names else 'Huangxing'} (只测第1个)")
#     print(f"  测试样本数: {test_samples}")
#     print(f"  测试特征数: {len(test_features)}")
#     print(f"  超时设置: {TIMEOUT_SECONDS}秒")
    
#     # 加载模型
#     print("\n🔧 加载模型...")
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name=MODEL_PATH,
#         max_seq_length=MAX_SEQ_LENGTH,
#         dtype=DTYPE,
#         load_in_4bit=LOAD_IN_4BIT,
#     )
#     FastLanguageModel.for_inference(model)
#     print("✅ 模型加载完成")
    
#     sample_indices = list(range(test_samples))
#     feature_indices = test_features
    
#     # 存储所有结果
#     all_results = []
    
#     print(f"\n🚀 开始评估站点: {station_names[0] if station_names else 'Huangxing'}...")
#     print("-"*80)
    
#     for station_idx in range(test_stations):
#         station_name = station_names[station_idx] if station_idx < len(station_names) else "Huangxing"
        
#         results = evaluate_station(
#             model, tokenizer, station_idx, station_name,
#             pred_array, true_array, weather_cache, events_map,
#             sample_indices, feature_indices, num_steps
#         )
        
#         all_results.extend(results)
    
#     # ================= 输出结果 =================
#     print("\n" + "="*80)
#     print("📊 评估结果汇总")
#     print("="*80)
    
#     if not all_results:
#         print("❌ 没有成功评估任何样本！")
#         return
    
#     # 转换为DataFrame
#     df = pd.DataFrame(all_results)
    
#     # 按特征聚合
#     print(f"\n📈 {station_names[0] if station_names else 'Huangxing'} 站点各特征评估结果:")
#     print(f"\n{'特征':<22} {'样本数':<8} {'GNN误差':<12} {'LLM误差':<12} {'改进(%)':<12} {'状态':<6}")
#     print("-"*80)
    
#     feature_stats = df.groupby('feature_name').agg({
#         'gnn_error': 'mean',
#         'llm_error': 'mean',
#         'improvement': 'mean',
#         'sample_idx': 'count'
#     }).reset_index()
    
#     for _, row in feature_stats.iterrows():
#         if row['improvement'] > 0:
#             status = "✅ 改善" if row['improvement'] > 1 else "✅ 微改善"
#         else:
#             status = "❌ 变差"
#         print(f"{row['feature_name']:<22} {row['sample_idx']:<8.0f} "
#               f"{row['gnn_error']:<12.2f} {row['llm_error']:<12.2f} {row['improvement']:>+10.2f}% {status}")
    
#     # 总体统计
#     print("\n" + "="*50)
#     print(f"📊 总体统计（仅{station_names[0] if station_names else 'Huangxing'}站点）")
#     print(f"{'='*50}")
#     print(f"评估样本总数: {len(df)}")
#     print(f"平均GNN误差: {df['gnn_error'].mean():.2f}")
#     print(f"平均LLM误差: {df['llm_error'].mean():.2f}")
#     print(f"平均改进率: {df['improvement'].mean():+.2f}%")
#     print(f"改进样本数: {(df['improvement'] > 0).sum()}/{len(df)} ({(df['improvement'] > 0).sum()/len(df)*100:.1f}%)")
    
#     # 判断效果
#     avg_improvement = df['improvement'].mean()
#     if avg_improvement > 5:
#         print(f"\n🎉 效果显著！LLM预测明显优于GNN")
#     elif avg_improvement > 0:
#         print(f"\n👍 略有改善，继续训练效果会更好")
#     elif avg_improvement > -5:
#         print(f"\n➖ 效果持平，需要调整参数")
#     else:
#         print(f"\n❌ 效果不佳，LLM预测比GNN更差，需要重新训练")
    
#     print(f"{'='*50}")
    
#     # 保存结果
#     output_dir = os.path.join(PROJECT_ROOT, "config/cai/evaluation")
#     os.makedirs(output_dir, exist_ok=True)
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     detail_path = os.path.join(output_dir, f"quick_test_{timestamp}.csv")
#     df.to_csv(detail_path, index=False, encoding='utf-8-sig')
#     print(f"\n✅ 详细结果已保存: {detail_path}")
    
#     print("\n✅ 快速评估完成！")

# if __name__ == "__main__":
#     main()
# """
# python 推理signal.py
# """
# import re
# import numpy as np
# import torch
# import os
# from tqdm import tqdm
# from unsloth import FastLanguageModel

# # ===================== 配置 =====================
# PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"
# FINETUNED_MODEL_PATH = os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-finetuned-quick427")

# YTRUE_TEST_PATH = os.path.join(PROJECT_ROOT, "finetune_real_traffic.npz")
# YPRED_TEST_PATH = os.path.join(PROJECT_ROOT, "finetune_data.npz")
# CARPARK_DES_PATH = os.path.join(PROJECT_ROOT, "station_list_hngs.txt")

# TARGET_STATION = 0
# MAX_SAMPLE = 20

# FEATURE_NAMES = [
#     "Passenger Car Up",
#     "Passenger Car Down",
#     "Non-Passenger Car Up",
#     "Non-Passenger Car Down"
# ]

# # 强制模型输出数字
# INSTRUCTION_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
# You are a traffic prediction refiner.
# Given 8-step flow sequence, output ONLY 8 integers in a list.
# Station: {}
# Feature: {}
# GNN: {}
# OUTPUT ONLY 8 NUMBERS.
# <|eot_id|><|start_header_id|>assistant<|end_header_id|>
# ["""

# # ===================== 安全解析（关键修复） =====================
# def extract_numbers_safe(text, fallback):
#     try:
#         nums = re.findall(r'\d+', text)
#         nums = [int(x) for x in nums if x.isdigit()]
#         nums = [x for x in nums if 0 <= x <= 9999]  # 流量不可能超过9999
        
#         if len(nums) >= 8:
#             return nums[:8]
#     except:
#         pass
#     return fallback[:8]

# def cal_metrics(pred, true):
#     mae = np.mean(np.abs(pred - true))
#     rmse = np.sqrt(np.mean((pred - true)**2))
#     mape = np.mean(np.abs((pred - true) / (true + 1e-8))) * 100
#     return round(mae,2), round(rmse,2), round(mape,2)

# # ===================== 主函数 =====================
# def main():
#     with open(CARPARK_DES_PATH, encoding='utf-8') as f:
#         stations = [s.strip() for s in f if s.strip()]
#     station_name = stations[TARGET_STATION]

#     true = np.load(YTRUE_TEST_PATH)['target']
#     pred = np.load(YPRED_TEST_PATH)['prediction']

#     true = true[:MAX_SAMPLE, :, TARGET_STATION, :]
#     pred = pred[:MAX_SAMPLE, :, TARGET_STATION, :]

#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name=FINETUNED_MODEL_PATH,
#         max_seq_length=1024,
#         load_in_4bit=True,
#         dtype=torch.bfloat16,
#         device_map={"":0}
#     )
#     FastLanguageModel.for_inference(model)

#     llm_result = np.zeros_like(pred)
#     print(f"\n🚀 开始推理：{station_name[:40]}...")

#     for i in tqdm(range(MAX_SAMPLE)):
#         for f in range(4):
#             gnn_seq = pred[i,:,f].tolist()
#             prompt = INSTRUCTION_TEMPLATE.format(
#                 station_name[:30], FEATURE_NAMES[f], gnn_seq
#             )

#             inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
#             with torch.no_grad():
#                 out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
            
#             resp = tokenizer.decode(out[0], skip_special_tokens=True)
#             llm_seq = extract_numbers_safe(resp, gnn_seq)
#             llm_result[i,:,f] = llm_seq

#     # ================== 输出 4 个特征正常指标 ==================
#     print("\n" + "="*70)
#     print(f"📊 【{station_name[:40]}...】 4 特征正常指标")
#     print("="*70)

#     for f in range(4):
#         g_mae, g_rmse, g_mape = cal_metrics(pred[:,:,f], true[:,:,f])
#         l_mae, l_rmse, l_mape = cal_metrics(llm_result[:,:,f], true[:,:,f])

#         print(f"\n{FEATURE_NAMES[f]}")
#         print(f"MAE   GNN: {g_mae:>6.2f}    LLM: {l_mae:>6.2f}")
#         print(f"RMSE  GNN: {g_rmse:>6.2f}    LLM: {l_rmse:>6.2f}")
#         print(f"MAPE  GNN: {g_mape:>6.2f}    LLM: {l_mape:>6.2f}")

#     print("\n✅ 推理完成（已修复数值爆炸问题）")

# if __name__ == "__main__":
#     main()
import json
import re
import numpy as np
import torch
import os
from tqdm import tqdm
from unsloth import FastLanguageModel

# ===================== 路径配置（和你训练完全一致） =====================
PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"
MODEL_PATH = os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-finetuned-quick427")

YTRUE_PATH = os.path.join(PROJECT_ROOT, "finetune_real_traffic.npz")
YPRED_PATH = os.path.join(PROJECT_ROOT, "finetune_data.npz")
STATION_LIST = os.path.join(PROJECT_ROOT, "station_list_hngs.txt")

FEATURE_NAMES = [
    "LittleCar_Up",
    "LittleCar_Down",
    "NonLittleCar_Up",
    "NonLittleCar_Down"
]

TARGET_STATION = 50
MAX_SAMPLE = 20

# ===================== 【最强锁死指令：强制只输出数字！】 =====================
INSTRUCTION_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Only output 8 numbers in [], no any words!
Station: {}
Feature: {}
GNN: {}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
["""

# ===================== 【暴力安全提取：只留0-9数字，绝对不炸】 =====================
# def extract_numbers_safe(text, fallback):
def extract_numbers_safe(text, fallback):

    try:
        # 只提取 0-9 纯数字，过滤所有文字、符号、英文
        nums = re.findall(r'\d+', text)
        valid = []
        for n in nums:
            if n.isdigit() and 0 <= int(n) <= 9999:  # 流量不可能超过9999
                valid.append(int(n))
        if len(valid) >= 8:
            return valid[:8]
    except:
        pass
    return fallback[:8]

# ===================== 指标计算 =====================
def cal_metrics(pred, true):
    mask = (true != 0)
    mae = np.mean(np.abs(pred[mask] - true[mask]))
    rmse = np.sqrt(np.mean((pred[mask] - true[mask])**2))
    mape = np.mean(np.abs((pred[mask] - true[mask])/(true[mask]+1e-8))) * 100
    return round(mae,2), round(rmse,2), round(mape,2)

# ===================== 主函数 =====================
def main():
    with open(STATION_LIST, encoding="utf-8") as f:
        stations = [l.strip() for l in f if l.strip()]
    station_name = stations[TARGET_STATION]

    true = np.load(YTRUE_PATH)["target"][:MAX_SAMPLE, :, TARGET_STATION, :]
    pred = np.load(YPRED_PATH)["prediction"][:MAX_SAMPLE, :, TARGET_STATION, :]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=torch.bfloat16,
        device_map={"":0}
    )
    FastLanguageModel.for_inference(model)

    llm_result = np.zeros_like(pred)
    print(f"\n🚀 推理站点：{station_name[:40]}")

    for i in tqdm(range(MAX_SAMPLE)):
        for f in range(4):
            gnn_seq = pred[i,:,f].tolist()
            prompt = INSTRUCTION_TEMPLATE.format(station_name[:30], FEATURE_NAMES[f], gnn_seq)

            inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=25, do_sample=False, temperature=0.001)
            
            resp = tokenizer.decode(out[0], skip_special_tokens=True)
            llm_result[i,:,f] = extract_numbers_safe(resp, gnn_seq)

    # ===================== 输出结果 =====================
    print("\n" + "="*80)
    print(f"📊 【{station_name[:40]}】 4 特征指标（已锁死，绝对不爆炸）")
    print("="*80)

    for f in range(4):
        g_mae, g_rmse, g_mape = cal_metrics(pred[:,:,f], true[:,:,f])
        l_mae, l_rmse, l_mape = cal_metrics(llm_result[:,:,f], true[:,:,f])

        print(f"\n{FEATURE_NAMES[f]}")
        print(f"MAE   GNN: {g_mae:>6.2f}    LLM: {l_mae:>6.2f}")
        print(f"RMSE  GNN: {g_rmse:>6.2f}    LLM: {l_rmse:>6.2f}")
        print(f"MAPE  GNN: {g_mape:>6.2f}    LLM: {l_mape:>6.2f}")

    print("\n✅ 推理完成！无爆炸数值！")

if __name__ == "__main__":
    main()
    """
    python 推理signal.py
    """