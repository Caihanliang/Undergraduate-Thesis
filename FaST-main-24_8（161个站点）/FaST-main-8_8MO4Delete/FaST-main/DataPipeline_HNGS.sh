# 湖南高速公路数据集处理流程  

echo "开始处理湖南高速公路流量数据..."

# 第一步：生成原始数据（需要先将你的 Excel/CSV 文件放在 DataPipeline 目录）
python DataPipeline/generate_hngs_data.py

# 第二步：生成训练数据
python DataPipeline/generate_hngs_training_data.py --dataset hngs --years 2023

# 第三步：处理邻接矩阵  按照位置的邻接情况  下次试试经纬度
python DataPipeline/process_hngs_adj.py

# 第四步：生成训练/验证/测试索引
python DataPipeline/generate_hngs_idx.py

echo "湖南高速数据处理完成！"
echo "数据集已保存到：main-master/datasets/HNGS/"
