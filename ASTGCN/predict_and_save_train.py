#!/usr/bin/env python
# coding: utf-8
"""
ASTGCN 训练集预测结果保存脚本
将训练集的预测结果保存为CSV格式，便于后续分析
"""

import torch
import numpy as np
import pandas as pd
import os
import argparse
import configparser
from model.ASTGCN_r import make_model
from lib.utils import load_adj, generate_data, re_normalization
from torch.utils.data import DataLoader, TensorDataset

print("=" * 60)
print("ASTGCN 训练集预测结果生成")
print("=" * 60)

# 解析配置文件
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/highway_traffic.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()

config = configparser.ConfigParser()
print(f'Read configuration file: {args.config}')
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

# 读取配置参数
adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']

ctx = training_config['ctx']
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:' + str(ctx))
print(f"CUDA: {USE_CUDA} {DEVICE}")

in_channels = int(training_config['in_channels'])
nb_block = int(training_config['nb_block'])
K = int(training_config['K'])
nb_chev_filter = int(training_config['nb_chev_filter'])
nb_time_filter = int(training_config['nb_time_filter'])
time_strides = int(training_config['time_strides'])
batch_size = int(training_config['batch_size'])

# 加载数据
print(f"\n📂 加载训练数据...")
train_x, train_target, val_x, val_target, test_x, test_target, _mean, _std = generate_data(
    graph_signal_matrix_filename
)

print(f"   训练集 x: {train_x.shape}")
print(f"   训练集 y: {train_target.shape}")

# 加载邻接矩阵
adj_mx = load_adj(adj_filename, num_of_vertices=num_of_vertices)
print(f"✅ 邻接矩阵加载成功: shape={adj_mx.shape}")

# 创建模型
print(f"\n🔧 创建模型...")
net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, 
                nb_time_filter, time_strides, adj_mx,
                num_for_predict, len_input, num_of_vertices, 
                num_features_out=in_channels)

# 查找最佳模型
params_path = f'experiments/{dataset_name}/astgcn_r_h1d0w0_channel4_1.000000e-03'

# 自动查找最佳的epoch文件
best_epoch = None
best_loss = float('inf')
for filename in os.listdir(params_path):
    if filename.startswith('epoch_') and filename.endswith('.params'):
        try:
            epoch_num = int(filename.split('_')[1].split('.')[0])
            # 这里简化处理，假设最后一个保存的epoch就是最好的
            # 如果需要精确匹配，可以读取验证损失日志
            best_epoch = epoch_num
        except:
            continue

if best_epoch is None:
    print("❌ 未找到训练好的模型文件")
    exit(1)

params_filename = os.path.join(params_path, f'epoch_{best_epoch}.params')
print(f"📦 加载模型: {params_filename}")
net.load_state_dict(torch.load(params_filename))
net.to(DEVICE)
net.eval()

print(f"✅ 模型加载完成 (Epoch {best_epoch})")

# 准备训练集数据
print(f"\n🔄 准备训练集数据...")
train_x_tensor = torch.FloatTensor(train_x).to(DEVICE)
train_y_tensor = torch.FloatTensor(train_target).to(DEVICE)

train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

print(f"   Batch数量: {len(train_loader)}")

# 进行预测
print(f"\n🚀 开始预测训练集...")
predictions = []
targets = []
inputs = []

with torch.no_grad():
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        output = net(batch_x)  # (B, N, F, T_out)
        
        predictions.append(output.cpu().numpy())
        targets.append(batch_y.cpu().numpy())
        inputs.append(batch_x.cpu().numpy())
        
        if (batch_idx + 1) % 10 == 0:
            print(f'   进度: {batch_idx + 1}/{len(train_loader)} batches')

# 合并所有batch的结果
predictions = np.concatenate(predictions, axis=0)  # (B, N, F, T_out)
targets = np.concatenate(targets, axis=0)          # (B, N, F, T_out)
inputs = np.concatenate(inputs, axis=0)            # (B, N, F, T_in)

print(f"\n✅ 预测完成!")
print(f"   预测值形状: {predictions.shape}")
print(f"   真实值形状: {targets.shape}")

# 反归一化
print(f"\n📊 反归一化...")
# 注意：预处理脚本已对 x 和 y 都进行了归一化，所以都需要反归一化
predictions_denorm = re_normalization(predictions, _mean, _std)
targets_denorm = re_normalization(targets, _mean, _std)  # ✅ targets 也需要反归一化

print(f"   Predictions 范围: [{predictions_denorm.min():.2f}, {predictions_denorm.max():.2f}]")
print(f"   Targets 范围: [{targets_denorm.min():.2f}, {targets_denorm.max():.2f}]")

# 转换为DataFrame
print(f"\n💾 转换为CSV格式...")

# 特征名称映射
feature_names = ['Passenger_Car_Up', 'Passenger_Car_Down', 
                 'Non_Passenger_Car_Up', 'Non_Passenger_Car_Down']
feature_names_cn = ['小客车上行', '小客车下行', '非小客车上行', '非小客车下行']

# 获取站点信息（如果有的话）
station_mapping_file = os.path.join(graph_signal_matrix_filename, 'station_mapping.csv')
if os.path.exists(station_mapping_file):
    station_df = pd.read_csv(station_mapping_file)
    print(f"✅ 加载站点映射文件: {len(station_df)} 个站点")
    print(f"   可用列: {list(station_df.columns)}")
    
    # 自动检测列名（兼容不同格式）
    station_name_col = None
    station_code_col = None
    
    # 尝试常见的列名
    for col in station_df.columns:
        col_lower = col.lower()
        if 'name' in col_lower or '名称' in col:
            station_name_col = col
        if 'code' in col_lower or '编号' in col or 'id' in col_lower:
            # 优先选择包含'code'的列作为站点代码
            if 'code' in col_lower:
                station_code_col = col
            elif station_code_col is None:
                station_code_col = col
    
    # 如果没有找到站点名称列，使用station_id或默认命名
    if station_name_col is None:
        # 检查是否有station_id列
        if 'station_id' in station_df.columns:
            station_name_col = 'station_id'
            print(f"   ℹ️  未找到站点名称列，使用station_id作为站点名称")
        else:
            station_name_col = station_df.columns[0] if len(station_df.columns) > 0 else None
            print(f"   ⚠️  使用第一列作为站点名称: {station_name_col}")
    
    # 如果没有找到站点编号列，使用station_code或第二列
    if station_code_col is None:
        if 'station_code' in station_df.columns:
            station_code_col = 'station_code'
        else:
            station_code_col = station_df.columns[1] if len(station_df.columns) > 1 else station_name_col
    
    print(f"   使用列 - 站点名称: {station_name_col}, 站点编号: {station_code_col}")
else:
    print(f"⚠️  未找到站点映射文件，使用默认编号")
    station_df = None
    station_name_col = None
    station_code_col = None

# 构建DataFrame
rows = []
B, N, F, T_out = predictions_denorm.shape

print(f"   正在处理 {B} 个样本 × {N} 个站点 × {F} 个特征 × {T_out} 个时间步...")

for b in range(B):
    for n in range(N):
        # 获取站点信息
        if station_df is not None and n < len(station_df):
            station_info = station_df.iloc[n]
            station_name = str(station_info[station_name_col]) if station_name_col else f'Station_{n:03d}'
            station_code = str(station_info[station_code_col]) if station_code_col else f'{n:03d}'
        else:
            station_name = f'Station_{n:03d}'
            station_code = f'{n:03d}'
        
        for f_idx in range(F):
            for t in range(T_out):
                # 时间标识：Sample_{batch_index}_{time_step}
                time_label = f'Sample_{b}_T{t}'
                
                row = {
                    '站点名称': str(station_info[station_name_col]) if station_df is not None and n < len(station_df) and station_name_col else f'{n}',
                    '站点编号': str(station_info[station_code_col]) if station_df is not None and n < len(station_df) and station_code_col else f'{n:03d}',
                    '时间': time_label,
                    '特征': feature_names[f_idx],
                    '真实值': round(float(targets_denorm[b, n, f_idx, t]), 2),
                    '预测值': round(float(predictions_denorm[b, n, f_idx, t]), 2)
                }
                rows.append(row)

# 创建DataFrame
df_predictions = pd.DataFrame(rows)

print(f"✅ DataFrame创建完成: {len(df_predictions)} 行")

# 保存为CSV
output_dir = params_path
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'train_predictions.csv')

print(f"\n💾 保存到: {output_file}")
df_predictions.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✅ 保存成功!")
print(f"   文件路径: {output_file}")
print(f"   文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

# 显示统计信息
print(f"\n{'='*60}")
print(f"📊 预测结果统计")
print(f"{'='*60}")

print(f"\n总体统计:")
print(f"  样本数: {B}")
print(f"  站点数: {N}")
print(f"  特征数: {F}")
print(f"  时间步: {T_out}")
print(f"  总行数: {len(df_predictions)}")

print(f"\n按特征分组统计:")
for feat_name in feature_names:
    feat_data = df_predictions[df_predictions['特征'] == feat_name]
    mae = np.mean(np.abs(feat_data['真实值'] - feat_data['预测值']))
    rmse = np.sqrt(np.mean((feat_data['真实值'] - feat_data['预测值']) ** 2))
    
    # SMAPE
    epsilon = 1e-8
    denominator = (np.abs(feat_data['真实值']) + np.abs(feat_data['预测值']) + epsilon) / 2.0
    smape = np.mean(np.abs(feat_data['预测值'] - feat_data['真实值']) / denominator) * 100
    
    print(f"  {feat_name}:")
    print(f"    MAE: {mae:.2f}")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    SMAPE: {smape:.2f}%")

print(f"\n前5行预览:")
print(df_predictions.head())

print(f"\n{'='*60}")
print(f"✅ 全部完成!")
print(f"{'='*60}")
