"""
LLM微调模型诊断脚本
用于检查微调后的模型是否正确输出了修正值
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import re
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# =================配置区域=================
PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main"
MODEL_PATH = os.path.join(PROJECT_ROOT, "config/cai/llama-3-1-8b-highway-finetuned-quick") 
DATA_PATH = os.path.join(PROJECT_ROOT, "config/cai/finetune_data.npz")
TRUE_PATH = os.path.join(PROJECT_ROOT, "config/cai/finetune_real_traffic.npz")
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "config/cai/quick.json")
STATION_LIST = os.path.join(PROJECT_ROOT, "config/cai/station_list_hngs.txt")

MAX_SEQ_LENGTH = 1024
DTYPE = torch.float16
LOAD_IN_4BIT = True
# =========================================

def load_station_names():
    """加载站点名称"""
    with open(STATION_LIST, "r", encoding='utf-8') as f:
        stations = [line.strip().split(' ')[0] for line in f if line.strip()]
    return stations

def load_data():
    """加载数据"""
    print("加载数据...")
    pred_npz = np.load(DATA_PATH)
    true_npz = np.load(TRUE_PATH)
    
    pred_key = 'prediction' if 'prediction' in pred_npz else 'arr_0'
    true_key = 'target' if 'target' in true_npz else 'arr_0'
    
    pred_array = pred_npz[pred_key]
    true_array = true_npz[true_key]
    
    # 处理维度
    if len(pred_array.shape) == 4:
        num_samples = pred_array.shape[0]
        num_steps = pred_array.shape[1]
        num_stations = pred_array.shape[2]
        num_features = pred_array.shape[3]
    else:
        # 尝试重塑
        stations = load_station_names()
        num_stations = len(stations)
        num_samples = pred_array.shape[0] // num_stations
        num_steps = pred_array.shape[1]
        num_features = pred_array.shape[2]
        pred_array = pred_array.reshape(num_samples, num_steps, num_stations, num_features)
        true_array = true_array.reshape(num_samples, num_steps, num_stations, num_features)
    
    return pred_array, true_array, num_samples, num_steps, num_stations, num_features

def load_training_samples():
    """加载训练样本用于对比"""
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"⚠️ 训练数据不存在: {TRAIN_DATA_PATH}")
        return None
    
    with open(TRAIN_DATA_PATH, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ 加载训练样本: {len(data)} 条")
    return data[:5]  # 只取前5条用于对比

def extract_from_training_sample(sample, tokenizer):
    """从训练样本中提取GNN值和修正值"""
    try:
        input_ids = sample.get('input_ids', [])
        if not input_ids:
            return None, None
        
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        
        # 提取GNN值
        gnn_match = re.search(r'GNN:\s*\[(.*?)\]', text, re.IGNORECASE)
        if not gnn_match:
            return None, None
        
        gnn_str = gnn_match.group(1)
        gnn_values = [float(x.strip()) for x in gnn_str.split(',')]
        
        # 提取修正值
        corr_match = re.search(r'Final Correction:\s*\[(.*?)\]', text, re.IGNORECASE)
        if corr_match:
            corr_str = corr_match.group(1)
            corr_values = [float(x.strip()) for x in corr_str.split(',')]
        else:
            corr_values = None
        
        return gnn_values, corr_values
    except Exception as e:
        return None, None

def build_test_prompts(pred_array, true_array, station_names, num_stations, num_features):
    """构建测试用的prompt"""
    test_prompts = []
    
    # 选择几个有代表性的站点
    test_stations = [0, 1, 2, 50, 100]  # 前几个和中间的几个
    test_stations = [s for s in test_stations if s < num_stations]
    
    for station_idx in test_stations:
        station_name = station_names[station_idx] if station_idx < len(station_names) else f"S{station_idx}"
        
        for feature_idx in [0, 1, 2, 3]:  # 测试所有特征
            if feature_idx >= num_features:
                continue
            
            # 取第一个样本
            sample_idx = 0
            
            # 获取8步预测值
            pred_sequence = []
            for step in range(min(8, pred_array.shape[1])):
                val = pred_array[sample_idx, step, station_idx, feature_idx]
                pred_sequence.append(float(val) if hasattr(val, 'item') else float(val))
            
            # 真实值
            true_val = true_array[sample_idx, 0, station_idx, feature_idx]
            true_val = float(true_val) if hasattr(true_val, 'item') else float(true_val)
            
            # 特征名称
            feature_names = {
                0: "Passenger Car Up",
                1: "Passenger Car Down", 
                2: "Non-Passenger Car Up",
                3: "Non-Passenger Car Down"
            }
            feature_name = feature_names.get(feature_idx, f"Feature_{feature_idx}")
            
            # 构建prompt
            prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Refine traffic prediction:
Station: {station_name}
Feature: {feature_name}
Time: 2023-09-01 00:00 to 2023-09-01 08:00 | Friday | Workday
GNN prediction: {pred_sequence}
Output corrected list.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

Final Correction: ["""
            
            test_prompts.append({
                'station_idx': station_idx,
                'station_name': station_name,
                'feature_idx': feature_idx,
                'feature_name': feature_name,
                'gnn_sequence': pred_sequence,
                'true_val': true_val,
                'gnn_first': pred_sequence[0],
                'prompt': prompt
            })
    
    return test_prompts

def test_model(model, tokenizer, test_prompts):
    """测试模型输出"""
    results = []
    
    print("\n" + "="*80)
    print("开始测试模型输出")
    print("="*80)
    
    for i, test in enumerate(test_prompts):
        print(f"\n{'='*60}")
        print(f"测试 {i+1}/{len(test_prompts)}")
        print(f"站点: {test['station_name']}")
        print(f"特征: {test['feature_name']}")
        print(f"GNN输入: {test['gnn_sequence'][:4]}... (共{len(test['gnn_sequence'])}个)")
        print(f"真实值(第1步): {test['true_val']:.2f}")
        print(f"GNN误差(第1步): {abs(test['gnn_first'] - test['true_val']):.2f}")
        print("-"*60)
        
        # Tokenize
        inputs = tokenizer(test['prompt'], return_tensors="pt", truncation=True, max_length=512).to("cuda")
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # 解码
        input_len = inputs.input_ids.shape[1]
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        print(f"\n模型完整输出:")
        print("-"*40)
        print(full_output)
        print("-"*40)
        
        # 提取数字
        # 方法1: 从响应中提取
        numbers1 = re.findall(r'\d+\.?\d*', response)
        
        # 方法2: 从完整输出中提取Final Correction
        corr_match = re.search(r'Final Correction:\s*\[(.*?)\]', full_output, re.IGNORECASE | re.DOTALL)
        if corr_match:
            numbers2 = re.findall(r'\d+\.?\d*', corr_match.group(1))
        else:
            numbers2 = []
        
        # 方法3: 提取所有数字
        all_numbers = re.findall(r'\d+\.?\d*', full_output)
        
        print(f"\n提取结果:")
        print(f"  从响应提取: {[float(x) for x in numbers1[:8]] if numbers1 else 'None'}")
        print(f"  从Final Correction提取: {[float(x) for x in numbers2[:8]] if numbers2 else 'None'}")
        print(f"  所有数字: {[float(x) for x in all_numbers[:10]] if all_numbers else 'None'}")
        
        # 判断是否成功提取
        extracted_values = None
        if numbers2:
            extracted_values = [float(x) for x in numbers2[:8]]
        elif numbers1 and len(numbers1) >= 8:
            extracted_values = [float(x) for x in numbers1[:8]]
        elif all_numbers and len(all_numbers) >= 8:
            extracted_values = [float(x) for x in all_numbers[:8]]
        
        if extracted_values:
            llm_first = extracted_values[0]
            llm_error = abs(llm_first - test['true_val'])
            gnn_error = abs(test['gnn_first'] - test['true_val'])
            improvement = ((gnn_error - llm_error) / (gnn_error + 1e-6)) * 100
            
            print(f"\n评估结果:")
            print(f"  GNN预测: {test['gnn_first']:.2f}")
            print(f"  LLM预测: {llm_first:.2f}")
            print(f"  真实值: {test['true_val']:.2f}")
            print(f"  GNN误差: {gnn_error:.2f}")
            print(f"  LLM误差: {llm_error:.2f}")
            print(f"  改进率: {improvement:+.2f}%")
        else:
            print(f"\n⚠️ 未能提取到有效的预测值！")
        
        results.append({
            'station': test['station_name'],
            'feature': test['feature_name'],
            'gnn_pred': test['gnn_first'],
            'true_val': test['true_val'],
            'llm_pred': extracted_values[0] if extracted_values else None,
            'gnn_error': abs(test['gnn_first'] - test['true_val']),
            'llm_error': abs(extracted_values[0] - test['true_val']) if extracted_values else None,
            'full_output': full_output,
            'response': response
        })
        
        print("="*60)
    
    return results

def compare_with_training_data(tokenizer, training_samples):
    """对比训练数据中的格式"""
    if not training_samples:
        return
    
    print("\n" + "="*80)
    print("训练数据格式分析")
    print("="*80)
    
    for i, sample in enumerate(training_samples):
        print(f"\n训练样本 {i+1}:")
        print("-"*40)
        
        gnn_vals, corr_vals = extract_from_training_sample(sample, tokenizer)
        
        if gnn_vals:
            print(f"GNN输入: {gnn_vals[:5]}...")
        if corr_vals:
            print(f"修正值: {corr_vals[:5]}...")
        
        # 解码部分内容
        input_ids = sample.get('input_ids', [])
        if input_ids:
            text = tokenizer.decode(input_ids[:500], skip_special_tokens=True)
            print(f"文本预览: {text[:300]}...")
        
        print("-"*40)

def analyze_model_config():
    """分析模型配置"""
    print("\n" + "="*80)
    print("模型配置分析")
    print("="*80)
    
    # 检查模型路径
    if os.path.exists(MODEL_PATH):
        print(f"✅ 模型路径存在: {MODEL_PATH}")
        
        # 列出模型文件
        files = os.listdir(MODEL_PATH)
        print(f"模型文件: {[f for f in files if f.endswith('.safetensors') or f.endswith('.bin')]}")
        
        # 检查adapter_config.json
        adapter_config = os.path.join(MODEL_PATH, "adapter_config.json")
        if os.path.exists(adapter_config):
            with open(adapter_config, 'r') as f:
                config = json.load(f)
            print(f"LoRA配置: r={config.get('r', 'N/A')}, alpha={config.get('lora_alpha', 'N/A')}")
    else:
        print(f"❌ 模型路径不存在: {MODEL_PATH}")

def main():
    print("="*80)
    print("LLM微调模型诊断工具")
    print("="*80)
    
    # 分析模型配置
    analyze_model_config()
    
    # 加载数据
    pred_array, true_array, num_samples, num_steps, num_stations, num_features = load_data()
    station_names = load_station_names()
    
    print(f"\n数据统计:")
    print(f"  样本数: {num_samples}")
    print(f"  步长: {num_steps}")
    print(f"  站点数: {num_stations}")
    print(f"  特征数: {num_features}")
    
    # 加载训练样本
    training_samples = load_training_samples()
    
    # 加载模型
    print(f"\n加载模型: {MODEL_PATH}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    print("✅ 模型加载完成")
    
    # 对比训练数据格式
    if training_samples:
        compare_with_training_data(tokenizer, training_samples)
    
    # 构建测试prompts
    test_prompts = build_test_prompts(pred_array, true_array, station_names, num_stations, num_features)
    
    # 测试模型
    results = test_model(model, tokenizer, test_prompts)
    
    # 总结
    print("\n" + "="*80)
    print("诊断总结")
    print("="*80)
    
    successful = sum(1 for r in results if r['llm_pred'] is not None)
    print(f"成功提取LLM预测: {successful}/{len(results)}")
    
    if successful > 0:
        improvements = [r['llm_error'] - r['gnn_error'] for r in results if r['llm_pred'] is not None]
        avg_improvement = np.mean(improvements) if improvements else 0
        print(f"平均误差变化: {avg_improvement:+.2f} (负值表示改善)")
        
        if avg_improvement > 0:
            print("\n⚠️ 警告: LLM预测平均误差大于GNN，模型效果不佳")
            print("可能原因:")
            print("  1. 微调轮数不足")
            print("  2. 训练数据质量差")
            print("  3. 修正系数设置不当")
            print("  4. 推理格式与训练不匹配")
        else:
            print("\n✅ LLM预测有所改善，但需要更多测试")
    
    # 保存诊断结果
    output_path = os.path.join(PROJECT_ROOT, "config/cai/diagnosis_results.json")
    with open(output_path, "w", encoding='utf-8') as f:
        # 简化结果以便保存
        save_results = []
        for r in results:
            save_results.append({
                'station': r['station'],
                'feature': r['feature'],
                'gnn_pred': r['gnn_pred'],
                'true_val': r['true_val'],
                'llm_pred': r['llm_pred'],
                'gnn_error': r['gnn_error'],
                'llm_error': r['llm_error'],
                'response': r['response'][:500] if r['response'] else None
            })
        json.dump(save_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 诊断结果已保存: {output_path}")
    print("="*80)

if __name__ == "__main__":
    main()
"""
python 诊断LLM输出值.py
/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/诊断LLM输出值
"""