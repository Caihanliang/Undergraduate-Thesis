"""
GMAN预测结果保存脚本（终极稳定版）
功能: 加载训练好的模型，对训练集、验证集、测试集进行预测，并保存真实值和预测值到CSV文件
使用NumPy权重格式，避免Keras序列化问题
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
import time
from datetime import datetime, timedelta

# 导入模型类
from train_highway_4feat_v2 import GMANModel

# ==================== 配置参数 ====================
MODEL_DIR = '/home/user/Downloads/cai/GMAN/models/highway_4feat'
DATA_FILE = '/home/user/Downloads/cai/GMAN/data/highway_4feat/highway_data.npz'
OUTPUT_CSV_DIR = '/home/user/Downloads/cai/GMAN/results'

os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

# 时间配置（与preprocess_highway_data.py保持一致）
START_TIME = datetime(2023, 9, 1, 0, 0, 0)  # 起始时间
TIME_GRANULARITY = 1  # 时间粒度：1小时
P = 8  # 输入步长
Q = 8  # 预测步长


def load_data():
    """加载预处理数据"""
    data = np.load(DATA_FILE)
    
    # 注意：键名是小写的，没有下划线
    trainX = data['trainX']
    trainY = data['trainY']
    trainTE = data['trainTE']
    
    valX = data['valX']
    valY = data['valY']
    valTE = data['valTE']
    
    testX = data['testX']
    testY = data['testY']
    testTE = data['testTE']
    
    SE = data['SE']
    mean = float(data['mean'])
    std = float(data['std'])
    
    print(f"   训练集: X={trainX.shape}, Y={trainY.shape}")
    print(f"   验证集: X={valX.shape}, Y={valY.shape}")
    print(f"   测试集: X={testX.shape}, Y={testY.shape}")
    print(f"   归一化参数: mean={mean:.4f}, std={std:.4f}")
    
    return (trainX, trainY, trainTE), (valX, valY, valTE), (testX, testY, testTE), SE, mean, std


def create_dataset(X, Y, TE, batch_size=32, shuffle=False):
    """创建TensorFlow Dataset"""
    dataset = tf.data.Dataset.from_tensor_slices((
        {'X': X, 'TE': TE},
        Y
    ))
    
    if shuffle:
        dataset = dataset.shuffle(1000)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def generate_spatial_embedding_tensor(SE, N_times_F):
    """生成空间嵌入张量"""
    N_se, D_se = SE.shape
    N = N_se  # 站点数
    F = N_times_F // N  # 特征数
    
    # 重复每个站点的嵌入F次（对应F个特征）
    SE_expanded = np.repeat(SE, F, axis=0)  # (N*F, D_se)
    
    return tf.constant(SE_expanded, dtype=tf.float32)


def load_model_weights(model, weights_path):
    """
    从TensorFlow Checkpoint或NumPy文件加载权重
    
    Args:
        model: Keras模型实例
        weights_path: 权重文件路径（支持.tf checkpoint或.npz）
    
    Returns:
        bool: 是否成功加载
    """
    # 方式1: TensorFlow Checkpoint格式（优先）
    ckpt_dir = os.path.join(MODEL_DIR, 'best_model_tf')
    if os.path.exists(ckpt_dir) and os.path.exists(weights_path + '.index'):
        try:
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(tf.train.latest_checkpoint(os.path.dirname(weights_path)))
            print(f"   ✅ 已从TF Checkpoint加载: {weights_path}")
            
            # 验证：检查所有变量是否已赋值
            uninit_vars = [v for v in model.variables if not v._has_valid_value()]
            if len(uninit_vars) == 0:
                print(f"   ✅ 所有{len(model.variables)}个变量已成功加载")
                return True
            else:
                print(f"   ⚠️  仍有{len(uninit_vars)}个变量未初始化")
                return False
        except Exception as e:
            print(f"   ⚠️  TF Checkpoint加载失败: {str(e)[:100]}")
    
    # 方式2: NumPy格式（备份）
    if weights_path.endswith('.npz'):
        try:
            weights_file = np.load(weights_path)
            loaded_count = 0
            
            for var in model.variables:
                # 将变量名转换为存储时的键名格式
                key = var.name.replace('/', '_').replace(':', '_')
                
                if key in weights_file:
                    weight_value = weights_file[key]
                    if weight_value.shape == var.shape:
                        var.assign(weight_value)
                        loaded_count += 1
                    else:
                        print(f"   ⚠️  形状不匹配: {key} 期望{var.shape}, 实际{weight_value.shape}")
                else:
                    print(f"   ⚠️  未找到权重: {key}")
            
            total_vars = len(model.variables)
            print(f"   ✅ 加载权重: {loaded_count}/{total_vars} 个变量 ({loaded_count/total_vars*100:.1f}%)")
            
            return loaded_count == total_vars
        except Exception as e:
            print(f"   ❌ NumPy加载失败: {e}")
            return False
    
    return False


def predict_dataset(model, dataset, SE_tensor):
    """
    对整个数据集进行预测
    
    Returns:
        predictions: 所有样本的预测值 (samples, Q, N*F)
        actuals: 所有样本的真实值 (samples, Q, N*F)
    """
    all_preds = []
    all_actuals = []
    
    for batch_x, batch_y in dataset:
        # 添加SE到输入（SE不需要batch维度，模型内部会处理）
        batch_input = {
            'X': batch_x['X'],
            'TE': batch_x['TE'],
            'SE': SE_tensor  # 直接使用 [N*F, D_se] 形状
        }
        
        pred = model(batch_input, training=False)
        all_preds.append(pred.numpy())
        all_actuals.append(batch_y.numpy())
    
    predictions = np.concatenate(all_preds, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    
    return predictions, actuals


def generate_timestamps(num_samples, P, Q):
    """
    生成每个样本的时间戳
    
    Args:
        num_samples: 样本数量
        P: 输入序列长度
        Q: 预测序列长度
        
    Returns:
        timestamps: 形状为 (num_samples, Q) 的时间戳数组
    """
    timestamps = []
    for i in range(num_samples):
        # 预测目标的起始时间
        pred_start = START_TIME + timedelta(hours=i)
        # 生成Q个时间戳
        sample_timestamps = [pred_start + timedelta(hours=j) for j in range(Q)]
        timestamps.append(sample_timestamps)
    
    return np.array(timestamps)  # (num_samples, Q)


def save_predictions_to_csv(predictions, actuals, timestamps, dataset_name, mean, std):
    """
    将预测结果保存为CSV格式
    
    Args:
        predictions: 预测值 (归一化空间), shape: (samples, Q, N*F)
        actuals: 真实值 (归一化空间), shape: (samples, Q, N*F)
        timestamps: 时间戳, shape: (samples, Q)
        dataset_name: 数据集名称 (train/val/test)
        mean: 归一化均值
        std: 归一化标准差
    """
    # 反归一化
    predictions_denorm = predictions * std + mean
    actuals_denorm = actuals * std + mean
    
    # 重塑数据: (samples, Q, N*F) -> (samples, Q, N, F)
    samples, Q, NF = predictions_denorm.shape
    N = 98  # 站点数
    F = NF // N  # 特征数 = 4
    
    print(f"   📊 数据维度: samples={samples}, Q={Q}, N={N}, F={F}")
    
    # Reshape为4维
    pred_reshaped = predictions_denorm.reshape(samples, Q, N, F)
    actual_reshaped = actuals_denorm.reshape(samples, Q, N, F)
    
    # 展平为长格式 (samples*Q*N, F)
    pred_flat = pred_reshaped.reshape(-1, F)
    actual_flat = actual_reshaped.reshape(-1, F)
    
    # 创建索引
    total_rows = samples * Q * N
    sample_indices = np.repeat(np.arange(samples), Q * N)
    time_steps = np.tile(np.repeat(np.arange(Q), N), samples)
    station_indices = np.tile(np.arange(N), samples * Q)
    
    # 生成时间戳列（展平）
    timestamps_flat = timestamps.reshape(-1)  # (samples*Q,)
    timestamps_expanded = np.repeat(timestamps_flat, N)  # (samples*Q*N,)
    
    # 格式化时间戳为字符串
    timestamp_strs = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps_expanded]
    
    print(f"   📊 展平后形状: pred_flat={pred_flat.shape}, indices={sample_indices.shape}")
    
    # 创建DataFrame
    df = pd.DataFrame({
        'sample_index': sample_indices,
        'time_step': time_steps,
        'station_index': station_indices,
        'timestamp': timestamp_strs,
        '小客车上行_预测值': pred_flat[:, 0],
        '小客车上行_真实值': actual_flat[:, 0],
        '小客车下行_预测值': pred_flat[:, 1],
        '小客车下行_真实值': actual_flat[:, 1],
        '非小客车上行_预测值': pred_flat[:, 2],
        '非小客车上行_真实值': actual_flat[:, 2],
        '非小客车下行_预测值': pred_flat[:, 3],
        '非小客车下行_真实值': actual_flat[:, 3]
    })
    
    # 计算每个特征的MAE
    mae_dict = {}
    feature_names = ['小客车上行', '小客车下行', '非小客车上行', '非小客车下行']
    for i, feat in enumerate(feature_names):
        mae = np.mean(np.abs(pred_flat[:, i] - actual_flat[:, i]))
        mae_dict[feat] = mae
        print(f"   {feat} MAE: {mae:.2f}")
    
    # 保存CSV
    csv_path = os.path.join(OUTPUT_CSV_DIR, f'{dataset_name}_predictions.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 已保存到: {csv_path}")
    print(f"   📊 数据形状: {df.shape}")
    
    return mae_dict


def main():
    print("="*60)
    print("GMAN 预测结果保存")
    print("="*60)
    
    # 1. 加载数据
    print("\n📊 加载数据...")
    (trainX, trainY, trainTE), (valX, valY, valTE), (testX, testY, testTE), SE, mean, std = load_data()
    
    # 2. 创建Dataset
    print("\n🔧 创建Dataset...")
    train_dataset = create_dataset(trainX, trainY, trainTE, batch_size=32, shuffle=False)
    val_dataset = create_dataset(valX, valY, valTE, batch_size=32, shuffle=False)
    test_dataset = create_dataset(testX, testY, testTE, batch_size=32, shuffle=False)
    
    # 3. 生成空间嵌入
    print("\n🌐 生成空间嵌入...")
    SE_tensor = generate_spatial_embedding_tensor(SE, trainX.shape[-1])
    print(f"   SE shape: {SE.shape}")
    
    # 4. 构建模型
    print("\n🔧 构建模型...")
    P, Q = 8, 8
    L, K, d = 3, 8, 16
    T_dim = trainTE.shape[-1]  # 时间嵌入维度 = 2
    D_se = SE.shape[1]  # 空间嵌入维度 = 64
    
    model = GMANModel(P=P, Q=Q, T=T_dim, L=L, K=K, d=d, D_se=D_se)
    
    # Build模型
    dummy_X = tf.constant(trainX[:1], dtype=tf.float32)  # (1, P, N*F)
    dummy_TE = tf.constant(trainTE[:1], dtype=tf.float32)  # (1, P+Q, 2)
    
    # SE应该直接传入原始形状 [N*F, D_se]，不需要batch维度
    dummy_SE = SE_tensor  # (392, 64)，不是 (1, 392, 64)
    
    _ = model({'X': dummy_X, 'TE': dummy_TE, 'SE': dummy_SE}, training=False)
    print(f"   ✅ 模型已构建")
    
    # 编译模型
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer)
    
    # 5. 加载权重（优先级: TF Checkpoint > NumPy > weights.h5）
    print("\n🔧 加载权重...")
    
    ckpt_path = os.path.join(MODEL_DIR, 'best_model_tf')
    npz_path = os.path.join(MODEL_DIR, 'best_model_weights.npz')
    h5_path = os.path.join(MODEL_DIR, 'best_model.weights.h5')
    
    model_loaded = False
    
    # 方式1: TensorFlow Checkpoint（最可靠）
    if os.path.exists(ckpt_path + '.index'):
        print(f"   📂 检测到TF Checkpoint格式")
        success = load_model_weights(model, ckpt_path)
        if success:
            model_loaded = True
        else:
            print(f"   ⚠️  TF Checkpoint加载不完整，尝试其他方式...")
    
    # 方式2: NumPy格式
    if not model_loaded and os.path.exists(npz_path):
        print(f"   📂 检测到NumPy格式")
        success = load_model_weights(model, npz_path)
        if success:
            model_loaded = True
        else:
            print(f"   ⚠️  NumPy加载不完整，尝试weights.h5...")
    
    # 方式3: H5格式（最后尝试）
    if not model_loaded and os.path.exists(h5_path):
        try:
            model.load_weights(h5_path)
            print(f"   ✅ 已从h5加载: {h5_path}")
            model_loaded = True
        except Exception as e:
            print(f"   ❌ h5加载失败: {str(e)[:100]}")
    
    if not model_loaded:
        print(f"   ❌ 所有权重文件均加载失败")
        print(f"   💡 建议: 重新训练模型")
        return
    
    # 6. 验证权重加载
    print("\n🔍 验证权重...")
    test_pred = model({'X': dummy_X, 'TE': dummy_TE, 'SE': dummy_SE}, training=False)
    pred_std = np.std(test_pred.numpy())
    print(f"   📊 预测值标准差: {pred_std:.4f}")
    
    if pred_std < 0.01:
        print(f"   ❌ 严重错误: 预测值标准差过小，权重未正确加载！")
        return
    else:
        print(f"   ✅ 权重验证通过")
    
    # 7. 生成时间戳
    print("\n🕒 生成时间戳...")
    train_timestamps = generate_timestamps(len(trainX), P, Q)
    val_timestamps = generate_timestamps(len(valX), P, Q)
    test_timestamps = generate_timestamps(len(testX), P, Q)
    
    print(f"   训练集时间范围: {train_timestamps[0, 0]} ~ {train_timestamps[-1, -1]}")
    print(f"   验证集时间范围: {val_timestamps[0, 0]} ~ {val_timestamps[-1, -1]}")
    print(f"   测试集时间范围: {test_timestamps[0, 0]} ~ {test_timestamps[-1, -1]}")
    
    # 8. 对三个数据集进行预测
    datasets = {
        'train': (train_dataset, trainY, train_timestamps),
        'val': (val_dataset, valY, val_timestamps),
        'test': (test_dataset, testY, test_timestamps)
    }
    
    all_mae_results = {}
    
    for dataset_name, (dataset, original_Y, timestamps) in datasets.items():
        print(f"\n{'='*60}")
        print(f"🚀 预测 {dataset_name} 集...")
        print(f"{'='*60}")
        
        start_time = time.time()
        predictions, actuals = predict_dataset(model, dataset, SE_tensor)
        elapsed = time.time() - start_time
        
        print(f"   ⏱️  耗时: {elapsed:.2f}s")
        print(f"   📊 预测形状: {predictions.shape}")
        print(f"   📊 真实值形状: {actuals.shape}")
        
        # 计算归一化空间的MAE
        norm_mae = np.mean(np.abs(predictions - actuals))
        print(f"   📈 归一化空间 MAE: {norm_mae:.4f}")
        
        # 保存CSV
        print(f"\n💾 保存 {dataset_name} 集预测结果...")
        mae_dict = save_predictions_to_csv(predictions, actuals, timestamps, dataset_name, mean, std)
        all_mae_results[dataset_name] = mae_dict
    
    # 9. 保存汇总结果
    print(f"\n{'='*60}")
    print("📊 预测结果汇总")
    print(f"{'='*60}")
    
    summary_df = pd.DataFrame(all_mae_results)
    print(summary_df)
    
    summary_path = os.path.join(OUTPUT_CSV_DIR, 'mae_summary.csv')
    summary_df.to_csv(summary_path, encoding='utf-8-sig')
    print(f"\n✅ 汇总结果已保存到: {summary_path}")
    
    print(f"\n{'='*60}")
    print("✅ 所有预测结果已保存完成!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()