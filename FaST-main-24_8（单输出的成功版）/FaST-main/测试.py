# import h5py
# with h5py.File('DataPipeline/hngs_his_2023.h5', 'r') as f:
#     # 打印所有组
#     print(list(f.keys()))
#     # 查看特定数据集
#     # print(f['dataset_name'][:])
##########################查看.h5文件##########################
##########################.h5就是数据集的压缩存储方式##########################
import h5py
import numpy as np

# 👇 这里改成你的 .h5 文件路径
H5_FILE_PATH = "DataPipeline/hngs_his_2023.h5"  # 改成你真实的文件路径！
# H5_FILE_PATH = "DataPipeline/HNGS_LC/hngs_his_2023.h5" 
#  FaST-main-24_8/FaST-main/DataPipeline/ca_his_2019.h5
def print_h5_structure(name, obj):
    """递归打印 h5 文件的所有结构和信息"""
    if isinstance(obj, h5py.Dataset):
        print(f"\n📊 数据集: {name}")
        print(f"   形状 (shape): {obj.shape}")
        print(f"   数据类型 (dtype): {obj.dtype}")
        # 打印前5个数据（如果是一维/二维）
        try:
            data = obj[:]
            if len(data.shape) == 1:
                print(f"   前5个值: {data[:5]}")
            elif len(data.shape) == 2:
                print(f"   前5行5列值:\n{data[:5, :5]}")
        except:
            print(f"   无法预览数据")

# 打开并查看文件
with h5py.File(H5_FILE_PATH, "r") as f:
    print("="*50)
    print("🔍 根目录 keys:", list(f.keys()))  # 你看到的 ['t'] 就在这里
    print("="*50)
    
    # 遍历所有内容
    print("\n📂 完整 H5 文件结构：")
    f.visititems(print_h5_structure)


# ##########################查看.npy文件##########################
# ##########################存交通时序数据 里面存的是：交通流量 / 车速 + 时间特征 形状是：(总时间步, 节点数, 特征数)##########################
# # 导入 numpy
# import numpy as np

# # 1. 加载 .npy 文件
# # 把这里换成你自己的 data.npy 路径！
# file_path = "main-master/datasets/HNGS/24_8/idx_train.npy" # gla_rn_adj.npy
# #  DataPipeline/HNGS_512/hngs_rn_adj.npy
# # main-master/datasets/HNGS_512_归一化/24_8/idx_test.npy
# data = np.load(file_path)

# # 2. 查看数据形状（最重要！）
# print("数据形状 (总时间步, 节点数, 特征数)：")
# print(data.shape)

# # 3. 查看数据类型
# print("\n数据类型：")
# print(data.dtype)

# # 4. 看前 2 个时间步的数据（快速预览）
# print("\n前 2 个时间步的前 5 个节点数据：")
# # print(data[:2, :5, :])
# # print(data[:5, :5]) #只用2个维度
# print(data[:5]) #只用2个维度


# # # ##########################查看.npz文件##########################

# import numpy as np

# # 1. 换成你的 .npz 路径
# file_path = "main-master/datasets/HNGS_LC/his.npz" #FaST-main-24_8/FaST-main/main-master/datasets/HNGS_512/his.npz
# # /home/user/Downloads/cai/FaST-main-24_8/FaST-main/visulization-result/predictions/finetune_data.npz  微调的两个数据就是一样的
# # /home/user/Downloads/cai/FaST-main-24_8/FaST-main/visulization-result/predictions/finetune_real_traffic.npz 
# # 都是保存的真实值和预测值 预测值都是从9/2/0开始的  但其实这个应该是作为样本的
# # /home/user/Downloads/cai/FaST-main-24_8/FaST-main/main-master/datasets/HNGS_LC/his.npz

# # 2. 加载 npz 文件
# data = np.load(file_path)

# # 3. 查看里面 包含哪些数组（最重要！）
# print("npz 里面的所有键名：")
# print(data.files)

# # 4. 遍历查看每个数据的形状
# print("\n每个数据的形状：")
# for key in data.files:
#     print(f"{key}:  {data[key].shape}")
# # 查看时序数据
# print("\n查看 prediction：")
# print(data['prediction'][:2, :5, :])
# print("\n查看 target：")
# print(data['target'][:2, :5, :])

##########################################测试生成数据的代码
# import os
# import argparse
# import numpy as np
# import pandas as pd
# import json

# def generate_data(df, add_time_of_day, add_day_of_week):
#     _, num_nodes = df.shape
#     data = np.expand_dims(df.values, axis=-1)
#     feature_list = [data]

#     if add_time_of_day:
#         idx = df.index.values
#         time_ind = (idx % 24) / 24.0
#         time_ind = time_ind.reshape((-1, 1, 1))
#         time_of_day = np.tile(time_ind, (1, num_nodes, 1))
#         feature_list.append(time_of_day)

#     if add_day_of_week:
#         idx = df.index.values
#         dow = ((idx // 24) % 7) / 7.0
#         dow = dow.reshape((-1, 1, 1))  # 🔥 修复维度错误
#         day_of_week = np.tile(dow, (1, num_nodes, 1))
#         feature_list.append(day_of_week)

#     data = np.concatenate(feature_list, axis=-1)
#     return data

# def generate_train_val_test(args):
#     # 🔥【核心修复】获取当前脚本绝对路径 → 保证目录一定生成
#     script_path = os.path.dirname(os.path.abspath(__file__))
#     os.chdir(script_path)

#     years = args.years.split("_")
#     df = pd.DataFrame()
    
#     for y in years:
#         df_tmp = pd.read_hdf(args.dataset + "_his_" + y + ".h5")
#         print(f"✅ 数据集 {y} 已加载")
#         df = pd.concat([df, df_tmp], axis=0, ignore_index=True)

#     data = generate_data(df, args.tod, args.dow)

#     # 🔥【终极修复】强制生成绝对路径 → 绝对不会找不到目录
#     out_dir = os.path.abspath("../main-master/datasets/HNGS_LC/")
#     os.makedirs(out_dir, exist_ok=True)

#     # 保存 npz
#     npz_file = os.path.join(out_dir, "his.npz")
#     np.savez_compressed(npz_file, data=data)

#     # 保存 desc
#     desc = {
#         "name": "hngs",
#         "domain": "traffic flow",
#         "shape": list(data.shape),
#         "num_time_steps": data.shape[0],
#         "num_nodes": data.shape[1],
#         "num_features": 3,
#         "feature_description": ["traffic flow", "time of day", "day of week"],
#         "has_graph": True,
#         "frequency (minutes)": 60,
#         "description": "Hunan Highway 1h Dataset",
#         "time_range": "2023-09-01 to 2023-10-31",
#         "mean": [0.0, 0.0, 0.0],
#         "std": [1.0, 1.0, 1.0],
#         "regular_settings": {
#             "INPUT_LEN": 24,
#             "OUTPUT_LEN": 8,
#             "TRAIN_VAL_TEST_RATIO": [0.6, 0.2, 0.2],
#             "NORM_EACH_CHANNEL": False,
#             "RESCALE": False,
#             "METRICS": ["MAE", "RMSE", "MAPE"],
#             "NULL_VAL": 0.0
#         }
#     }

#     json_file = os.path.join(out_dir, "desc.json")
#     with open(json_file, "w", encoding="utf-8") as f:
#         json.dump(desc, f, indent=2)

#     # 🔥 最终打印真实路径 → 你直接点开就能看到文件
#     print("\n" * 2)
#     print("=" * 60)
#     print("✅ 全部生成成功！")
#     print(f"📂 输出目录：{out_dir}")
#     print(f"✅ his.npz 已生成")
#     print(f"✅ desc.json 已生成")
#     print(f"✅ 数据形状：{data.shape}")
#     print("=" * 60)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", default="hngs")
#     parser.add_argument("--years", default="2023")
#     parser.add_argument("--tod", type=int, default=1)
#     parser.add_argument("--dow", type=int, default=1)
#     args = parser.parse_args()
#     generate_train_val_test(args)

