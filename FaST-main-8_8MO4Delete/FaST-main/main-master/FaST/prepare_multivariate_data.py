"""
多变量数据准备工具
将小客车和非小客车流量合并为多变量数据集

当前格式：
- HNGS_LC: [T, 160, 1]  ← 小客车流量
- HNGS_NLC: [T, 160, 1] ← 非小客车流量

目标格式：
- HNGS_MULTI: [T, 160, 2] ← [小客车, 非小客车]
"""
import os
import sys
import numpy as np
import json
from pathlib import Path

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_multivariate_dataset():
    """
    创建多变量数据集：[T, N, C]
    C = 2: [小客车流量, 非小客车流量]
    """
    base_path = Path(__file__).parent.parent / "datasets"
    
    print("=" * 70)
    print("多变量数据集准备工具")
    print("=" * 70)
    
    # 1. 加载小客车数据
    lc_path = base_path / "HNGS_LC"
    lc_data_file = lc_path / "data.dat"
    lc_desc_file = lc_path / "desc.json"
    
    with open(lc_desc_file, 'r') as f:
        lc_desc = json.load(f)
    
    lc_shape = tuple(lc_desc["shape"])
    lc_data = np.memmap(lc_data_file, dtype='float32', mode='r', shape=lc_shape)
    print(f"✅ 加载小客车数据 (HNGS_LC): {lc_data.shape}")
    print(f"   特征: {lc_desc['feature_description']}")
    
    # 2. 加载非小客车数据
    nlc_path = base_path / "HNGS_NLC"
    nlc_data_file = nlc_path / "data.dat"
    nlc_desc_file = nlc_path / "desc.json"
    
    with open(nlc_desc_file, 'r') as f:
        nlc_desc = json.load(f)
    
    nlc_shape = tuple(nlc_desc["shape"])
    nlc_data = np.memmap(nlc_data_file, dtype='float32', mode='r', shape=nlc_shape)
    print(f"✅ 加载非小客车数据 (HNGS_NLC): {nlc_data.shape}")
    print(f"   特征: {nlc_desc['feature_description']}")
    
    # 3. 验证数据一致性
    assert lc_shape[0] == nlc_shape[0], "❌ 时间步不一致！"
    assert lc_shape[1] == nlc_shape[1], "❌ 站点数不一致！"
    
    T, N, _ = lc_shape
    print(f"\n📊 数据验证:")
    print(f"   - 时间步: {T}")
    print(f"   - 站点数: {N}")
    
    # 4. 合并为多变量格式 [T, N, C]
    # 提取流量特征（第0维）
    lc_flow = lc_data[:, :, 0:1]   # [T, N, 1] 小客车
    nlc_flow = nlc_data[:, :, 0:1] # [T, N, 1] 非小客车
    
    # 沿特征维度拼接
    multi_data = np.concatenate([lc_flow, nlc_flow], axis=2)  # [T, N, 2]
    print(f"\n✅ 合并后形状: {multi_data.shape}")
    print(f"   C=2: [小客车流量, 非小客车流量]")
    
    # 5. 创建输出目录
    output_name = "HNGS_MULTI"
    output_path = base_path / output_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 6. 保存多变量数据
    output_data_file = output_path / "data.dat"
    multi_memmap = np.memmap(
        output_data_file,
        dtype='float32',
        mode='w+',
        shape=multi_data.shape
    )
    multi_memmap[:] = multi_data[:]
    multi_memmap.flush()
    print(f"💾 数据已保存: {output_data_file}")
    
    # 7. 创建描述文件
    multi_desc = {
        "name": "hngs_multi",
        "domain": "traffic flow",
        "shape": list(multi_data.shape),
        "num_time_steps": T,
        "num_nodes": N,
        "num_features": 2,  # 🔥 关键：2个特征
        "feature_description": [
            "light vehicle flow",    # 小客车流量
            "non-light vehicle flow"  # 非小客车流量
        ],
        "has_graph": True,
        "frequency (minutes)": 60,
        "description": "Hunan Highway Multi-Vehicle Dataset (Light + Non-Light)",
        "time_range": "2023-09-01 to 2023-10-31",
        "mean": [0.0, 0.0],
        "std": [1.0, 1.0],
        "regular_settings": {
            "INPUT_LEN": 8,
            "OUTPUT_LEN": 8,
            "TRAIN_VAL_TEST_RATIO": [0.6, 0.2, 0.2],
            "NORM_EACH_CHANNEL": False,
            "RESCALE": False,
            "METRICS": ["MAE", "RMSE", "MAPE"],
            "NULL_VAL": 0.0
        }
    }
    
    output_desc_file = output_path / "desc.json"
    with open(output_desc_file, 'w') as f:
        json.dump(multi_desc, f, indent=2)
    print(f"💾 描述文件已保存: {output_desc_file}")
    
    # 8. 复制索引文件（与单变量相同）
    for idx_file in ['idx_train.npy', 'idx_val.npy', 'idx_test.npy']:
        src = lc_path / idx_file
        dst = output_path / idx_file
        if src.exists():
            import shutil
            shutil.copy(src, dst)
            print(f"💾 索引文件已复制: {idx_file}")
    
    print("\n" + "=" * 70)
    print("✅ 多变量数据集创建完成！")
    print("=" * 70)
    print(f"\n📁 输出路径: {output_path}")
    print(f"📊 数据格式: [T, N, C] = {multi_data.shape}")
    print(f"   - T: {T} 时间步")
    print(f"   - N: {N} 站点")
    print(f"   - C: 2 特征 [小客车, 非小客车]")
    print(f"\n💡 使用方法:")
    print(f"   CFG.DATASET.NAME = '{output_name}'")
    print(f"   num_nodes = {N}  # 站点数不变")
    print(f"   CFG.MODEL.FORWARD_FEATURES = [0, 1]  # 使用两个特征")
    print(f"   CFG.MODEL.TARGET_FEATURES = [0, 1]    # 预测两个特征")
    
    return output_path


def verify_multivariate_dataset():
    """验证多变量数据集"""
    base_path = Path(__file__).parent.parent / "datasets" / "HNGS_MULTI"
    
    print("\n" + "=" * 70)
    print("验证多变量数据集")
    print("=" * 70)
    
    data_file = base_path / "data.dat"
    desc_file = base_path / "desc.json"
    
    if not data_file.exists():
        print("❌ 数据集不存在，请先运行 create_multivariate_dataset()")
        return
    
    # 加载描述
    with open(desc_file, 'r') as f:
        desc = json.load(f)
    
    shape = tuple(desc["shape"])
    data = np.memmap(data_file, dtype='float32', mode='r', shape=shape)
    
    print(f"✅ 数据集形状: {data.shape}")
    print(f"✅ 特征描述: {desc['feature_description']}")
    
    # 检查数据范围
    print(f"\n📊 数据统计:")
    for i, feat_name in enumerate(desc['feature_description']):
        feat_data = data[:, :, i]
        print(f"   {feat_name}:")
        print(f"      均值: {feat_data.mean():.2f}")
        print(f"      标准差: {feat_data.std():.2f}")
        print(f"      最小值: {feat_data.min():.2f}")
        print(f"      最大值: {feat_data.max():.2f}")
    
    # 检查第一个站点的前3个时间步
    print(f"\n📋 示例数据（中方站，前3小时）:")
    station_idx = 0
    for t in range(3):
        lc = data[t, station_idx, 0]
        nlc = data[t, station_idx, 1]
        print(f"   t={t}: 小客车={lc:.0f}, 非小客车={nlc:.0f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify_multivariate_dataset()
    else:
        # 创建数据集
        create_multivariate_dataset()
        
        # 验证
        print("\n")
        verify_multivariate_dataset()
