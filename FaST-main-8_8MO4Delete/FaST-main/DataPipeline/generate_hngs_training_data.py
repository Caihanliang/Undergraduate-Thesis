# # 自动归一化、能解决你所有问题的版本！
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
#         dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
#         day_of_week = dow_tiled
#         feature_list.append(day_of_week)

#     data = np.concatenate(feature_list, axis=-1)
#     return data

# def generate_train_val_test(args):
#     os.chdir(os.path.dirname(os.path.abspath(__file__)))
#     years = args.years.split("_")
#     df = pd.DataFrame()
    
#     for y in years:
#         df_tmp = pd.read_hdf(args.dataset + "_his_" + y + ".h5")
#         df = pd.concat([df, df_tmp], axis=0, ignore_index=True)

#     data = generate_data(df, args.tod, args.dow)

#     # ===================== 【自动归一化】 =====================
#     # 对流量特征（第0维）做标准化
#     flow_data = data[..., 0]
#     mean = flow_data.mean()
#     std  = flow_data.std()
#     data[..., 0] = (flow_data - mean) / std
#     # ==========================================================

#     out_dir = "../main-master/datasets/" + args.dataset.upper() + "/"
#     os.makedirs(out_dir, exist_ok=True)
#     np.savez_compressed(os.path.join(out_dir, "his.npz"), data=data)

#     # 保存正确的 desc.json
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
#         "mean": [float(mean), 0.0, 0.0],
#         "std": [float(std), 1.0, 1.0],
#         "regular_settings": {
#             "INPUT_LEN": 24,
#             "OUTPUT_LEN": 8,
#             "TRAIN_VAL_TEST_RATIO": [0.6, 0.2, 0.2],
#             "NORM_EACH_CHANNEL": False,
#             "RESCALE": True,
#             "METRICS": ["MAE", "RMSE", "MAPE"],
#             "NULL_VAL": 0.0
#         }
#     }

#     with open(os.path.join(out_dir, "desc.json"), "w", encoding="utf-8") as f:
#         json.dump(desc, f, indent=2)

#     print("✅ 已生成 归一化 his.npz + desc.json")
#     print(f"✅ 流量均值: {mean:.2f}")
#     print(f"✅ 流量标准差: {std:.2f}")
#     print(f"✅ 数据已归一化为 0 附近，训练将完全正常！")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", default="hngs")
#     parser.add_argument("--years", default="2023")
#     parser.add_argument("--tod", type=int, default=1)
#     parser.add_argument("--dow", type=int, default=1)
#     args = parser.parse_args()
#     generate_train_val_test(args)


import os
import argparse
import numpy as np
import pandas as pd
import json

def generate_data(df, add_time_of_day, add_day_of_week):
    _, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]

    if add_time_of_day:
        idx = df.index.values
        time_ind = (idx % 24) / 24.0
        time_ind = time_ind.reshape((-1, 1, 1))
        time_of_day = np.tile(time_ind, (1, num_nodes, 1))
        feature_list.append(time_of_day)

    if add_day_of_week:
        idx = df.index.values
        dow = ((idx // 24) % 7) / 7.0
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        day_of_week = dow_tiled
        feature_list.append(day_of_week)

    data = np.concatenate(feature_list, axis=-1)
    return data

def generate_train_val_test(args):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    years = args.years.split("_")
    df = pd.DataFrame()
    
    for y in years:
        # df_tmp = pd.read_hdf(args.dataset + "_his_" + y +"_lc" +".h5")
        df_tmp = pd.read_hdf(args.dataset + "_his_" + y +"_nlc" +".h5")
        print(f"数据集 {y} 已加载")
        df = pd.concat([df, df_tmp], axis=0, ignore_index=True)

    data = generate_data(df, args.tod, args.dow)

    # ===================== 【已删除：归一化代码】 =====================

    # out_dir = "../main-master/datasets/HNGS_LC/" 
    out_dir = "../main-master/datasets/HNGS_NLC/" 
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "his.npz"), data=data)

    # 保存 desc.json（不使用归一化参数）
    desc = {
        "name": "hngs",
        "domain": "traffic flow",
        "shape": list(data.shape),
        "num_time_steps": data.shape[0],
        "num_nodes": data.shape[1],
        "num_features": 3,
        "feature_description": ["traffic flow", "time of day", "day of week"],
        "has_graph": True,
        "frequency (minutes)": 60,
        "description": "Hunan Highway 1h Dataset",
        "time_range": "2023-09-01 to 2023-10-31",
        # ===================== 关键修改：mean/std 设为空 =====================
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "regular_settings": {
            "INPUT_LEN": 24,
            "OUTPUT_LEN": 8,
            "TRAIN_VAL_TEST_RATIO": [0.6, 0.2, 0.2],
            "NORM_EACH_CHANNEL": False,
            "RESCALE": False,    # 关闭自动缩放
            "METRICS": ["MAE", "RMSE", "MAPE"],
            "NULL_VAL": 0.0
        }
    }

    with open(os.path.join(out_dir, "desc.json"), "w", encoding="utf-8") as f:
        json.dump(desc, f, indent=2)

    print("✅ 已生成 【无归一化】 his.npz + desc.json")
    print(f"✅ 数据形状: {data.shape}")
    print(f"✅ 数据保持原始流量值，未做任何缩放！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hngs")
    parser.add_argument("--years", default="2023")
    parser.add_argument("--tod", type=int, default=1)
    parser.add_argument("--dow", type=int, default=1)
    args = parser.parse_args()
    generate_train_val_test(args)

    # python DataPipeline/generate_hngs_training_data.py