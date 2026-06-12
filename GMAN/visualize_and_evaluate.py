"""
GMAN预测结果可视化与分析脚本
功能: 
1. 加载训练好的模型进行预测
2. 可视化4个特征的预测结果
3. 计算并输出评估指标
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# ==================== 配置参数 ====================
DATA_DIR = '/home/user/Downloads/cai/GMAN/data/highway_4feat'
MODEL_DIR = '/home/user/Downloads/cai/GMAN/models/highway_4feat'
OUTPUT_DIR = '/home/user/Downloads/cai/GMAN/results/highway_4feat'

# 特征名称
FEATURE_NAMES = {
    0: '小客车上行',
    1: '小客车下行',
    2: '非小客车上行',
    3: '非小客车下行'
}

P = 8  # 输入步长
Q = 8  # 预测步长


def load_data():
    """加载数据"""
    print("📊 加载数据...")
    data = np.load(os.path.join(DATA_DIR, 'highway_data.npz'))
    
    testX = data['testX']
    testTE = data['testTE']
    testY = data['testY']
    SE = data['SE']
    mean = float(data['mean'])
    std = float(data['std'])
    
    # 加载站点映射
    station_mapping = pd.read_csv(os.path.join(DATA_DIR, 'station_mapping.csv'))
    
    print(f"   测试集: X={testX.shape}, Y={testY.shape}")
    print(f"   站点数: {len(station_mapping)}")
    
    return testX, testTE, testY, SE, mean, std, station_mapping


def predict(model_dir, testX, testTE, SE, mean, std):
    """使用训练好的模型进行预测"""
    print("\n🔮 进行预测...")
    
    num_test, P, N = testX.shape
    
    # 重建模型图
    X = tf.placeholder(shape=(None, P, N), dtype=tf.float32, name='X')
    TE = tf.placeholder(shape=(None, P + Q, 2), dtype=tf.int32, name='TE')
    is_training = tf.placeholder(shape=(), dtype=tf.bool, name='is_training')
    
    # 这里需要重新导入GMAN模型函数
    # 简化版：直接加载保存的预测结果
    # 实际使用时需要从train_highway_4feat.py导出预测函数
    
    saver = tf.train.import_meta_graph(model_dir + '.meta')
    
    with tf.Session() as sess:
        saver.restore(sess, model_dir)
        
        graph = tf.get_default_graph()
        pred_tensor = graph.get_tensor_by_name('pred:0')
        X_tensor = graph.get_tensor_by_name('X:0')
        TE_tensor = graph.get_tensor_by_name('TE:0')
        is_training_tensor = graph.get_tensor_by_name('is_training:0')
        
        testPred = []
        batch_size = 32
        num_batch = int(np.ceil(num_test / batch_size))
        
        for batch_idx in range(num_batch):
            start_idx = batch_idx * batch_size
            end_idx = min(num_test, (batch_idx + 1) * batch_size)
            
            feed_dict = {
                X_tensor: testX[start_idx:end_idx],
                TE_tensor: testTE[start_idx:end_idx],
                is_training_tensor: False
            }
            
            pred_batch = sess.run(pred_tensor, feed_dict=feed_dict)
            testPred.append(pred_batch)
        
        testPred = np.concatenate(testPred, axis=0)
    
    # 反归一化
    testPred = testPred * std + mean
    testY_original = testY * std + mean
    
    print(f"   预测完成! 形状: {testPred.shape}")
    
    return testPred, testY_original


def calculate_metrics(pred, true):
    """计算评估指标"""
    # MAE
    mae = np.mean(np.abs(pred - true))
    
    # RMSE
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    
    # MAPE (排除零值)
    mask = true != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100
    else:
        mape = 0.0
    
    # SMAPE
    smape = np.mean(2 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8)) * 100
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'SMAPE': smape}


def visualize_predictions(pred, true, station_mapping, output_dir, max_samples=5):
    """可视化预测结果"""
    print("\n📈 生成可视化图表...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_test, Q, N = pred.shape
    num_stations = len(station_mapping)
    num_features = 4
    
    # 重塑为 (samples, Q, stations, features)
    pred_reshaped = pred.reshape(num_test, Q, num_stations, num_features)
    true_reshaped = true.reshape(num_test, Q, num_stations, num_features)
    
    # 随机选择几个样本和站点进行可视化
    sample_indices = np.random.choice(num_test, min(max_samples, num_test), replace=False)
    station_indices = np.random.choice(num_stations, min(3, num_stations), replace=False)
    
    for sample_idx in sample_indices:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for feat_idx in range(num_features):
            ax = axes[feat_idx]
            
            for station_idx in station_indices:
                station_name = station_mapping.iloc[station_idx]['station_code']
                
                # 提取该站点的预测和真实值
                pred_values = pred_reshaped[sample_idx, :, station_idx, feat_idx]
                true_values = true_reshaped[sample_idx, :, station_idx, feat_idx]
                
                time_steps = range(Q)
                ax.plot(time_steps, true_values, 'o-', label=f'{station_name} (真实)', linewidth=2)
                ax.plot(time_steps, pred_values, 's--', label=f'{station_name} (预测)', linewidth=2)
            
            ax.set_title(f'{FEATURE_NAMES[feat_idx]}', fontsize=12, fontweight='bold')
            ax.set_xlabel('预测步长 (小时)', fontsize=10)
            ax.set_ylabel('流量 (辆)', fontsize=10)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Sample {sample_idx} - 4特征预测对比', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'sample_{sample_idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   已保存: {save_path}")
    
    print(f"✅ 可视化完成! 共生成 {len(sample_indices)} 张图表")


def save_results_to_csv(pred, true, station_mapping, output_dir):
    """保存预测结果到CSV"""
    print("\n💾 保存预测结果...")
    
    num_test, Q, N = pred.shape
    num_stations = len(station_mapping)
    num_features = 4
    
    # 重塑数据
    pred_reshaped = pred.reshape(num_test, Q, num_stations, num_features)
    true_reshaped = true.reshape(num_test, Q, num_stations, num_features)
    
    # 创建DataFrame
    records = []
    
    for sample_idx in range(num_test):
        for q in range(Q):
            for station_idx in range(num_stations):
                station_code = station_mapping.iloc[station_idx]['station_code']
                
                for feat_idx in range(num_features):
                    records.append({
                        'sample_id': sample_idx,
                        'prediction_step': q,
                        'station_code': station_code,
                        'feature': FEATURE_NAMES[feat_idx],
                        'feature_id': feat_idx,
                        'true_value': true_reshaped[sample_idx, q, station_idx, feat_idx],
                        'pred_value': pred_reshaped[sample_idx, q, station_idx, feat_idx]
                    })
    
    df_results = pd.DataFrame(records)
    
    csv_path = os.path.join(output_dir, 'all_predictions.csv')
    df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"   已保存到: {csv_path}")
    print(f"   总记录数: {len(df_results)}")


def main():
    """主函数"""
    print("=" * 60)
    print("GMAN 高速公路交通流量预测 - 结果分析")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载数据
    testX, testTE, testY, SE, mean, std, station_mapping = load_data()
    
    # 2. 预测 (需要先运行训练脚本)
    try:
        pred, true = predict(MODEL_DIR, testX, testTE, SE, mean, std)
    except Exception as e:
        print(f"\n⚠️  预测失败: {e}")
        print("   请先运行 train_highway_4feat.py 进行训练")
        return
    
    # 3. 计算整体指标
    print("\n📊 整体评估指标:")
    metrics = calculate_metrics(pred, true)
    print(f"   MAE:  {metrics['MAE']:.2f}")
    print(f"   RMSE: {metrics['RMSE']:.2f}")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")
    print(f"   SMAPE:{metrics['SMAPE']:.2f}%")
    
    # 4. 按特征分别计算指标
    print("\n📊 各特征评估指标:")
    num_test, Q, N = pred.shape
    num_stations = len(station_mapping)
    num_features = 4
    
    pred_reshaped = pred.reshape(num_test, Q, num_stations, num_features)
    true_reshaped = true.reshape(num_test, Q, num_stations, num_features)
    
    for feat_idx in range(num_features):
        feat_pred = pred_reshaped[:, :, :, feat_idx].flatten()
        feat_true = true_reshaped[:, :, :, feat_idx].flatten()
        
        feat_metrics = calculate_metrics(feat_pred, feat_true)
        print(f"\n   {FEATURE_NAMES[feat_idx]}:")
        print(f"      MAE:  {feat_metrics['MAE']:.2f}")
        print(f"      RMSE: {feat_metrics['RMSE']:.2f}")
        print(f"      MAPE: {feat_metrics['MAPE']:.2f}%")
        print(f"      SMAPE:{feat_metrics['SMAPE']:.2f}%")
    
    # 5. 可视化
    visualize_predictions(pred, true, station_mapping, OUTPUT_DIR)
    
    # 6. 保存结果
    save_results_to_csv(pred, true, station_mapping, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("✅ 分析完成!")
    print(f"   结果目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
