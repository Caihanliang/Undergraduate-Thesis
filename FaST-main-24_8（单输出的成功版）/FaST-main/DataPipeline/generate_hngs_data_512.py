import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

"""
湖南省高速公路流量数据集转换脚本
将原始数据转换为 FaST 项目所需的标准格式

输入数据要求:
- 纵向：时间序列（2023 年 9 月 1 日 - 10 月 31 日）
- 横向：512 个收费站点（黄兴到芙蓉镇）
- 数据频率：1小时间隔

输出文件格式:
- hngs_his_2023_512.h5: 历史流量数据
- hngs_meta_512.csv: 站点元数据
- hngs_rn_adj_512.npy: 邻接矩阵
  把原始 Excel/CSV 数据，转换成 GNN 模型（FaST）能直接训练的标准格式
"""

def load_and_preprocess_data(input_file, output_dir='HNGS_512'):
    """
    加载并预处理原始数据
    
    Args:
        input_file: 原始数据文件路径 (Excel 或 CSV)
        output_dir: 输出目录名
    """
    print(f"正在加载数据：{input_file}")
    
    # 读取数据
    if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
        df = pd.read_excel(input_file)
    elif input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        raise ValueError("不支持的文件格式，请使用 Excel 或 CSV")
    
    print(f"原始数据形状：{df.shape}")
    print(f"列名：{df.columns.tolist()}")
    
    # 数据清洗和格式化
    # 假设第一列是时间，其余列是站点名称
    time_col = df.columns[0]
    station_cols = df.columns[1:]
    
    print(f"时间列：{time_col}")
    print(f"站点数量：{len(station_cols)}")
    
    # 转换时间列
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)
    
    # 检查数据频率
    time_diff = df.index[1] - df.index[0]
    print(f"数据时间间隔：{time_diff}")
    
    # ===================== 重要修改 =====================
    # 你的数据是 1小时一条，直接按 1小时 处理，不转15分钟！
    if time_diff != timedelta(hours=1):
        print("正在重采样为 1 小时间隔...")
        df = df.resample('1H').mean().round(0)
        print(f"重采样后数据形状：{df.shape}")
    # ====================================================
    
    # 填充缺失值
    print("正在填充缺失值...")
    df = df.fillna(0)
    
    # 保存历史数据
    os.makedirs(output_dir, exist_ok=True)
    his_file = f"{output_dir}/hngs_his_2023.h5"
    df.to_hdf(his_file, key="t", mode="w")
    print(f"已保存历史数据：{his_file}")
    
    return df, station_cols


def generate_meta_data(station_names, output_dir='HNGS_512'):
    """
    生成站点元数据
    
    Args:
        station_names: 站点名称列表
        output_dir: 输出目录
    """
    print("正在生成站点元数据...")
    
    # 创建元数据 DataFrame
    meta_data = []
    for i, station_name in enumerate(station_names):
        # 如果没有经纬度信息，使用占位值
        # 实际应用中应该填入真实的经纬度
        meta_entry = {
            'ID': str(i + 1),
            'StationName': station_name,
            'Lat': 28.0 + i * 0.01,  # 占位纬度，需要替换为真实值
            'Lng': 110.0 + i * 0.01,  # 占位经度，需要替换为真实值
            'District': 'Hunan',
            'County': 'Unknown',
            'Type': 'TollStation',
            'ID2': i
        }
        meta_data.append(meta_entry)
    
    meta_df = pd.DataFrame(meta_data)
    
    # 保存元数据
    meta_file = f"{output_dir}/hngs_meta.csv"
    meta_df.to_csv(meta_file, index=False)
    print(f"已保存元数据：{meta_file}")
    
    return meta_df


def generate_adjacency_matrix(num_nodes, method='distance', meta_df=None, output_dir='HNGS'):
    """
    生成邻接矩阵
    
    Args:
        num_nodes: 节点数量（站点数）
        method: 生成方法 ('distance' 基于距离，'sequence' 基于序列顺序，'custom' 自定义)
        meta_df: 元数据 DataFrame（包含经纬度信息）
        output_dir: 输出目录
    """
    print(f"正在生成邻接矩阵（方法：{method}）...")
    
    if method == 'distance' and meta_df is not None:
        # 基于欧氏距离计算邻接矩阵
        lats = meta_df['Lat'].values
        lngs = meta_df['Lng'].values
        
        # 计算距离矩阵
        dist_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                lat_diff = lats[i] - lats[j]
                lng_diff = lngs[i] - lats[j]
                dist_matrix[i, j] = np.sqrt(lat_diff**2 + lng_diff**2)
        
        # 使用高斯核函数转换为邻接矩阵
        sigma = np.std(dist_matrix)
        adj_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2))
        
        # 稀疏化：只保留最近的 k 个邻居
        k = min(10, num_nodes // 2)
        for i in range(num_nodes):
            indices = np.argsort(adj_matrix[i])[::-1]
            for j in range(k+1, num_nodes):
                adj_matrix[i, indices[j]] = 0
    
    elif method == 'sequence':
        # 基于序列顺序（假设站点按顺序排列在高速公路上）
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            # 连接到前后相邻的站点
            if i > 0:
                adj_matrix[i, i-1] = 1
            if i < num_nodes - 1:
                adj_matrix[i, i+1] = 1
            # 自连接
            adj_matrix[i, i] = 1
    
    elif method == 'custom':
        # 自定义邻接矩阵
        # 这里可以根据实际的高速公路网络拓扑结构来定义
        adj_matrix = np.eye(num_nodes)  # 默认只有自连接
        print("警告：使用自定义邻接矩阵，请根据实际情况修改代码")
    
    else:
        raise ValueError("不支持的邻接矩阵生成方法")
    
    # 保存邻接矩阵
    adj_file = f"{output_dir}/hngs_rn_adj.npy"
    np.save(adj_file, adj_matrix)
    print(f"已保存邻接矩阵：{adj_file}")
    print(f"邻接矩阵形状：{adj_matrix.shape}")
    print(f"非零元素比例：{np.count_nonzero(adj_matrix) / (num_nodes * num_nodes):.4f}")
    
    return adj_matrix


if __name__ == "__main__":
    # 设置工作目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # ========== 第一步：加载和预处理数据 ==========
    # 请修改为你的原始数据文件路径
    input_file = "202309_10_512数据集.csv"  # 或 "hunan_highway_traffic_test.xlsx"
    
    try:
        df, station_names = load_and_preprocess_data(input_file, output_dir='HNGS_512')
    except FileNotFoundError:
        print(f"\n错误：找不到文件 '{input_file}'")
        print("请将您的原始数据文件放在 DataPipeline 目录下，并修改 input_file 变量")
        exit(1)
    
    # ========== 第二步：生成元数据 ==========
    meta_df = generate_meta_data(station_names, output_dir='HNGS_512')
    
    # ========== 第三步：生成邻接矩阵 ==========
    # 选择邻接矩阵生成方法：
    # 'distance' - 基于经纬度距离（如果有真实经纬度）
    # 'sequence' - 基于站点顺序（假设站点按顺序排列）
    # 'custom' - 自定义（需要根据实际路网修改代码）
    adj_method = 'sequence'  # 推荐使用 sequence，因为站点在高速公路上是线性排列的
    adj_matrix = generate_adjacency_matrix(
        num_nodes=len(station_names),
        method=adj_method,
        meta_df=meta_df,
        output_dir='HNGS_512'
    )
    
    print("\n" + "="*50)
    print("数据转换完成！")
    print("="*50)
    print(f"输出目录：HNGS_512/")
    print(f"  - hngs_his_2023.h5: 历史流量数据")    # 历史车流量时间序列（主数据）
    print(f"  - hngs_meta.csv: 站点元数据")         # 161 个高速站点信息
    print(f"  - hngs_rn_adj.npy: 邻接矩阵")         # 高速路网图结构（GNN 依赖）

    print("\n下一步：将这些文件移动到 DataPipeline 目录")
    # python DataPipeline/generate_hngs_data_512.py