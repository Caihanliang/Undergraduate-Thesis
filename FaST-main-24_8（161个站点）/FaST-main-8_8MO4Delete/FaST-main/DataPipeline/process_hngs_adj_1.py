import numpy as np
import pickle
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 处理湖南高速数据集的邻接矩阵
data_name = "hngs"
npy_file_path = f"{data_name}_rn_adj.npy"
data = np.load(npy_file_path)

pkl_file_path = f"../main-master/datasets/{data_name.upper()}/adj_mx.pkl"

with open(pkl_file_path, "wb") as f:
    pickle.dump(data, f)

print(f"邻接矩阵已保存到：{pkl_file_path}")
