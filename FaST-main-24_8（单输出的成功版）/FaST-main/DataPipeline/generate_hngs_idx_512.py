# 这个只是生成索引文件
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def generate_and_split_indices_sequential(data_name, input_step, output_step):
    """生成样本索引并划分训练/验证/测试集"""

    data_npz = np.load(f"../main-master/datasets/{data_name}/his.npz")

    data = data_npz["data"]
    T, N, D = data.shape
    total_steps = input_step + output_step

    indices = np.arange(total_steps - 1, T)
    num_samples = len(indices)

    train_end = int(0.6 * num_samples)
    val_end = train_end + int(0.2 * num_samples)
    idx_train = indices[:train_end]
    idx_val = indices[train_end:val_end]
    idx_test = indices[val_end:]

    target_dir = os.path.join(
        "..", "main-master", "datasets", data_name, f"{input_step}_{output_step}"
    )
    os.makedirs(target_dir, exist_ok=True)

    np.save(
        f"../main-master/datasets/{data_name}/{input_step}_{output_step}/idx_train.npy",
        idx_train,
    )
    np.save(
        f"../main-master/datasets/{data_name}/{input_step}_{output_step}/idx_val.npy",
        idx_val,
    )
    np.save(
        f"../main-master/datasets/{data_name}/{input_step}_{output_step}/idx_test.npy",
        idx_test,
    )
    
    print(f"已生成 {data_name} 数据集的索引：输入={input_step}, 输出={output_step}")
    print(f"  训练集：{len(idx_train)} 样本")
    print(f"  验证集：{len(idx_val)} 样本")
    print(f"  测试集：{len(idx_test)} 样本")


if __name__ == "__main__":
    # 湖南高速数据集配置（1小时数据版本 ✅）
    data_name = "HNGS_512"
    input_time_step = 24      # 输入：过去 24 小时（1小时一步，所以是24）
    output_time_steps = [8]   # 输出：预测未来 8 小时 → 完美匹配大模型LLM

    print(f"开始为 HNGS_512 数据集生成索引...")
    print(f"输入长度：{input_time_step} (24小时)")
    print(f"预测长度：{output_time_steps}")
    
    for output_time_step in output_time_steps:
        generate_and_split_indices_sequential(
            data_name, input_time_step, output_time_step
        )
    
    print("\n索引生成完成！")


    # python DataPipeline/generate_hngs_idx_512.py