"""
深度诊断脚本：检查微调和推理的完整链路
运行方式：python 深度诊断_微推理链路.py
"""
import os
import json
import numpy as np
import re

PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"

print("="*80)
print("🔍 FaST项目微推理链路深度诊断工具")
print("="*80)

# ==================== 1. 检查数据集缓存 ====================
print("\n【步骤1】检查数据集缓存")
print("-" * 80)
dataset_cache = os.path.join(PROJECT_ROOT, "quick30.json")
if os.path.exists(dataset_cache):
    file_size = os.path.getsize(dataset_cache) / (1024 * 1024)
    import datetime
    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(dataset_cache))
    print(f"⚠️  发现数据集缓存: quick30.json")
    print(f"   文件大小: {file_size:.1f} MB")
    print(f"   修改时间: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 读取前几个样本检查站点信息
    with open(dataset_cache, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            print(f"   样本总数: {len(data)}")
            
            if len(data) > 0:
                # 解码第一个样本来检查包含的站点
                sample = data[0]
                input_ids = sample['input_ids']
                
                # 需要加载tokenizer来解码
                print(f"\n   ⚠️  警告: 此数据集是旧版本生成的！")
                print(f"   建议: 删除此文件后重新运行微调代码")
                
                response = input("\n   是否立即删除旧缓存？(y/n): ").strip().lower()
                if response == 'y':
                    os.remove(dataset_cache)
                    print("   ✅ 已删除旧缓存")
                else:
                    print("   ❌ 保留旧缓存，这可能导致微调无效！")
        except Exception as e:
            print(f"   ❌ 读取缓存失败: {e}")
else:
    print("✅ 未发现数据集缓存，可以安全生成新数据")

# ==================== 2. 检查微调代码配置 ====================
print("\n【步骤2】检查微调代码配置")
print("-" * 80)
finetune_file = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/微调430solo.py"
with open(finetune_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 提取站点配置
match = re.search(r'stations\s*=\s*\[(.*?)\]', content)
if match:
    stations_str = match.group(1).strip()
    print(f"📋 微调代码配置的站点: [{stations_str}]")
    
    # 检查是否是注释掉的
    if '#' in content.split('stations =')[0].split('\n')[-1]:
        print("   ⚠️  注意: 请确认此行未被注释")
else:
    print("❌ 未找到站点配置")

# 检查训练输出目录
output_dir = os.path.join(PROJECT_ROOT, "results-quick430")
if os.path.exists(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        latest = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        print(f"📂 发现训练checkpoint: {latest}")
        
        # 检查checkpoint的时间
        checkpoint_path = os.path.join(output_dir, latest)
        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
        print(f"   修改时间: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if mod_time < datetime.datetime.now() - datetime.timedelta(hours=1):
            print("   ⚠️  警告: Checkpoint较旧，可能需要重新训练")
    else:
        print("📂 输出目录存在但无checkpoint")
else:
    print("✅ 输出目录不存在，将创建新目录")

# ==================== 3. 检查推理代码配置 ====================
print("\n【步骤3】检查推理代码配置")
print("-" * 80)
inference_file = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/推理430.py"
with open(inference_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 提取测试站点
match = re.search(r'TEST_STATION_ID\s*=\s*(\d+)', content)
if match:
    test_station = int(match.group(1))
    print(f"📋 推理代码测试的站点: {test_station}")
else:
    print("❌ 未找到测试站点配置")

# 检查模型路径
model_path_match = re.search(r'MODEL_PATH\s*=\s*os\.path\.join\(PROJECT_ROOT,\s*"([^"]+)"\)', content)
if model_path_match:
    model_path = model_path_match.group(1)
    full_model_path = os.path.join(PROJECT_ROOT, model_path)
    print(f"📋 推理使用的模型路径: {model_path}")
    
    if os.path.exists(full_model_path):
        print(f"   ✅ 模型目录存在")
        
        # 检查模型文件
        adapter_files = [f for f in os.listdir(full_model_path) if f.endswith('.safetensors') or f.endswith('.bin')]
        if adapter_files:
            print(f"   📦 找到LoRA适配器文件: {len(adapter_files)} 个")
        else:
            print(f"   ⚠️  警告: 未找到LoRA适配器文件")
    else:
        print(f"   ❌ 模型目录不存在！请先运行微调代码")
else:
    print("❌ 未找到模型路径配置")

# ==================== 4. 检查数据文件 ====================
print("\n【步骤4】检查数据文件")
print("-" * 80)
ytrue_path = os.path.join(PROJECT_ROOT, "finetune_real_traffic.npz")
ypred_path = os.path.join(PROJECT_ROOT, "finetune_data.npz")

if os.path.exists(ytrue_path):
    true_npz = np.load(ytrue_path)
    print(f"✅ 真实值文件存在: finetune_real_traffic.npz")
    print(f"   可用键名: {list(true_npz.keys())}")
    if 'target' in true_npz:
        true_data = true_npz['target']
    elif 'arr_0' in true_npz:
        true_data = true_npz['arr_0']
    print(f"   数据形状: {true_data.shape}")
    print(f"   站点数: {true_data.shape[1]}")
else:
    print("❌ 真实值文件不存在")

if os.path.exists(ypred_path):
    pred_npz = np.load(ypred_path)
    print(f"✅ 预测值文件存在: finetune_data.npz")
    print(f"   可用键名: {list(pred_npz.keys())}")
    if 'prediction' in pred_npz:
        pred_data = pred_npz['prediction']
    elif 'arr_0' in pred_npz:
        pred_data = pred_npz['arr_0']
    print(f"   数据形状: {pred_data.shape}")
    
    # 检查GNN预测质量
    if os.path.exists(ytrue_path):
        mae = np.mean(np.abs(true_data - pred_data))
        print(f"   GNN基线MAE: {mae:.2f}")
else:
    print("❌ 预测值文件不存在")

# ==================== 5. 一致性检查 ====================
print("\n【步骤5】一致性检查")
print("-" * 80)

# 重新读取配置进行对比
with open(finetune_file, 'r', encoding='utf-8') as f:
    ft_content = f.read()
with open(inference_file, 'r', encoding='utf-8') as f:
    inf_content = f.read()

ft_match = re.search(r'stations\s*=\s*\[(.*?)\]', ft_content)
inf_match = re.search(r'TEST_STATION_ID\s*=\s*(\d+)', inf_content)

if ft_match and inf_match:
    ft_stations = [int(x.strip()) for x in ft_match.group(1).split(',') if x.strip().isdigit()]
    inf_station = int(inf_match.group(1))
    
    print(f"微调训练站点: {ft_stations}")
    print(f"推理测试站点: {inf_station}")
    
    if inf_station in ft_stations:
        print("✅ 站点配置一致！推理站点已参与训练")
    else:
        print(f"❌ 站点配置不一致！站点{inf_station}未参与训练")
        print(f"   解决方案: 修改微调代码的stations列表，或修改推理代码的TEST_STATION_ID")
else:
    print("❌ 无法读取配置进行对比")

# ==================== 6. 诊断总结 ====================
print("\n" + "="*80)
print("📊 诊断总结与建议")
print("="*80)

issues = []

# 检查数据集缓存
if os.path.exists(dataset_cache):
    issues.append("⚠️  存在旧数据集缓存，可能导致使用了错误的训练数据")

# 检查模型是否存在
model_path_match = re.search(r'MODEL_PATH\s*=\s*os\.path\.join\(PROJECT_ROOT,\s*"([^"]+)"\)', inf_content)
if model_path_match:
    model_path = os.path.join(PROJECT_ROOT, model_path_match.group(1))
    if not os.path.exists(model_path):
        issues.append("❌ 推理使用的模型目录不存在，需要先运行微调")

# 检查站点一致性
if ft_match and inf_match:
    ft_stations = [int(x.strip()) for x in ft_match.group(1).split(',') if x.strip().isdigit()]
    inf_station = int(inf_match.group(1))
    if inf_station not in ft_stations:
        issues.append(f"❌ 站点不匹配: 推理站点{inf_station}不在训练站点{ft_stations}中")

if issues:
    print("\n发现的问题:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print("\n🎯 建议操作顺序:")
    print("  1. 删除旧数据集缓存: rm config/cai/quick30.json")
    print("  2. 确认微调和推理的站点ID一致")
    print("  3. 运行微调: python 微调430solo.py")
    print("  4. 等待训练完成后，运行推理: python 推理430.py")
else:
    print("\n✅ 未发现明显配置问题")
    print("\n如果推理仍然无效，可能的原因:")
    print("  1. 模型训练不充分（Loss未收敛）")
    print("  2. Prompt格式与微调时不完全对齐")
    print("  3. 采样参数设置不当（temperature/do_sample）")
    print("  4. 数据本身质量问题（GNN已经很好，LLM难以改进）")

print("\n" + "="*80)
