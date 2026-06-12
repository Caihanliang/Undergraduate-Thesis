import pandas as pd

# 1. 读取你的坐标距离文件
# 假设三列分别是：起点ID, 终点ID, 距离数值
dist_df = pd.read_csv('/home/user/FSTLLM/contrast/Traffic-Benchmark-master/methods/GMAN/METR/data/subway/distance.csv', header=None)

# 2. 将距离转为权重（因为 GMAN 预训练需要权重，距离越小权重越大）
# 我们用 1 除以距离，并保留 3 位小数
dist_df[2] = (1.0 / (dist_df[2] + 1e-5)).round(3)

# 3. 保存为 Adj.txt (这是 GMAN 要求的 node2vec 输入格式)
dist_df.to_csv('/home/user/FSTLLM/contrast/Traffic-Benchmark-master/methods/GMAN/METR/data/subway/Adj.txt', sep=' ', header=False, index=False)

print("第一步完成：已生成 /home/user/FSTLLM/contrast/Traffic-Benchmark-master/methods/GMAN/METR/data/subway/Adj.txt")