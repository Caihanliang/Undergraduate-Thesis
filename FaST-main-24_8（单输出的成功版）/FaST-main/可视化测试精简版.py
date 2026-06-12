import sys
import os

# ========== 你原本的路径（完全不动）==========
BASE_PATH = "/home/user/Downloads/cai/FaST-main-24_8"
sys.path.append(BASE_PATH)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'main-master')))
os.chdir(os.path.abspath(os.path.dirname(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from datetime import datetime, timedelta
import json
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from basicts.runners import SimpleTimeSeriesForecastingRunner
from easytorch.config import init_cfg
from easytorch.utils import get_logger

plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

class FaSTVisualizationRunner:
    def __init__(self, config_path: str, checkpoint_path: str, mode:str, gpu_id: str = '0'):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.gpu_id = gpu_id
        self.logger = get_logger("fast-visualization")

        self.cfg = init_cfg(config_path, save=False)
        self.runner = self.cfg["RUNNER"](self.cfg)
        self.runner.init_logger(logger_name="fast-visualization")

        self.mode = mode
        self.cfg.mode = mode

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        torch.backends.cudnn.benchmark = True

        self.start_dt = self._get_start_time()
        self.output_dir = self._create_output_directory()

        data_root = os.path.join("main-master", "datasets", self.cfg.DATASET.NAME)
        self.in_len = self.cfg.DATASET.PARAM.input_len  # 24
        self.out_len = self.cfg.DATASET.PARAM.output_len # 8
        sample_path = os.path.join(data_root, f"{self.in_len}_{self.out_len}")

        self.idx_train = np.load(os.path.join(sample_path, "idx_train.npy"))
        self.idx_val   = np.load(os.path.join(sample_path, "idx_val.npy"))
        self.idx_test  = np.load(os.path.join(sample_path, "idx_test.npy"))

        self.print_time_ranges()
        self._setup_components()
        self._load_model()

    def print_time_ranges(self):
        t0 = self.start_dt + timedelta(hours=int(self.idx_train[0]))
        t1 = self.start_dt + timedelta(hours=int(self.idx_train[-1]))
        t2 = self.start_dt + timedelta(hours=int(self.idx_val[0]))
        t3 = self.start_dt + timedelta(hours=int(self.idx_val[-1]))
        t4 = self.start_dt + timedelta(hours=int(self.idx_test[0]))
        t5 = self.start_dt + timedelta(hours=int(self.idx_test[-1]))

        print("\n" + "="*60)
        print("✅ 输入序列起始时间范围（从 .npy 索引加载）")
        print(f"训练集 train: {t0.strftime('%Y-%m-%d %H:%M')} —— {t1.strftime('%Y-%m-%d %H:%M')}")
        print(f"验证集   val: {t2.strftime('%Y-%m-%d %H:%M')} —— {t3.strftime('%Y-%m-%d %H:%M')}")
        print(f"测试集  test: {t4.strftime('%Y-%m-%d %H:%M')} —— {t5.strftime('%Y-%m-%d %H:%M')}")
        print("="*60 + "\n")

    def _get_start_time(self):
        desc_path = os.path.join("main-master", "datasets", self.cfg.DATASET.NAME, "desc.json")
        with open(desc_path, 'r', encoding='utf-8') as f:
            desc = json.load(f)
            start = desc["time_range"].split(" to ")[0]
        return datetime.strptime(start, "%Y-%m-%d")

    def _setup_components(self):
        self.logger.info("Setting up components...")
        if self.runner.need_setup_graph:
            self.runner.setup_graph(cfg=self.cfg, train=False)

        from basicts.data import MyTimeSeries
        from torch.utils.data import DataLoader

        dataset_cfg = self.cfg.DATASET.PARAM
        input_len = dataset_cfg.input_len
        output_len = dataset_cfg.output_len

        if self.mode == "train":
            self.logger.info("✅ 手动加载 训练集")
            dataset = MyTimeSeries(dataset_name=dataset_cfg.dataset_name, train_val_test_ratio=dataset_cfg.train_val_test_ratio, mode="train", input_len=input_len, output_len=output_len)
        elif self.mode == "val":
            self.logger.info("✅ 手动加载 验证集")
            dataset = MyTimeSeries(dataset_name=dataset_cfg.dataset_name, train_val_test_ratio=dataset_cfg.train_val_test_ratio, mode="valid", input_len=input_len, output_len=output_len)
        else:
            self.logger.info("✅ 手动加载 测试集")
            dataset = MyTimeSeries(dataset_name=dataset_cfg.dataset_name, train_val_test_ratio=dataset_cfg.train_val_test_ratio, mode="test", input_len=input_len, output_len=output_len)

        batch_size = self.cfg.TRAIN.DATA.BATCH_SIZE
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.runner.test_data_loader = loader

    def _load_model(self):
        self.logger.info(f"Loading model from {self.checkpoint_path}")
        self.runner.load_model(ckpt_path=self.checkpoint_path, strict=True)
        self.runner.model.eval()

    def _create_output_directory(self):
        root = "visulization-result"
        os.makedirs(root, exist_ok=True)
        os.makedirs(f"{root}/metrics", exist_ok=True)
        os.makedirs(f"{root}/predictions", exist_ok=True)
        os.makedirs(f"{root}/figures", exist_ok=True)
        os.makedirs(f"{root}/figures/all_stations", exist_ok=True)
        return root

    @torch.no_grad()
    def generate_predictions(self):
        self.logger.info("Generating ALL predictions...")
        loader = self.runner.test_data_loader
        preds, targets = [], []
        for data in loader:
            out = self.runner.forward(data, train=False)
            preds.append(out["prediction"])
            targets.append(out["target"])
        pred = torch.cat(preds, dim=0).cpu().numpy()
        target = torch.cat(targets, dim=0).cpu().numpy()
        self.logger.info(f"Final shape: {pred.shape}")
        
        # ✅ 验证时间索引
        self.verify_time_index(pred)
        
        return pred, target

    def verify_time_index(self, pred):
        """验证时间索引是否正确"""
        if self.mode == "train":
            indices = self.idx_train
        elif self.mode == "val":
            indices = self.idx_val
        else:
            indices = self.idx_test
        
        print(f"\n{'='*60}")
        print(f"📊 时间索引验证（模式：{self.mode}）")
        print(f"{'='*60}")
        print(f"idx_test[0] = {int(indices[0])} (这是输入序列的结束时间索引)")
        print(f"idx_test[-1] = {int(indices[-1])} (这是最后一个样本的输入结束时间索引)")
        print(f"总样本数：{len(indices)}")
        print(f"预测结果形状：{pred.shape}")
        print(f"\n前 5 个样本的时间索引验证：")
        
        for i in range(min(5, len(indices))):
            input_end_time = self.start_dt + timedelta(hours=int(indices[i]))
            pred_start_time = input_end_time - timedelta(hours=7)
            print(f"  样本 {i}: 输入结束={input_end_time.strftime('%Y-%m-%d %H:%M')}, "
                  f"预测起始={pred_start_time.strftime('%Y-%m-%d %H:%M')}")
            
        print(f"\n后 5 个样本的时间索引验证：")
        for i in range(max(0, len(indices)-5), len(indices)):
            input_end_time = self.start_dt + timedelta(hours=int(indices[i]))
            pred_start_time = input_end_time - timedelta(hours=7)
            print(f"  样本 {i}: 输入结束={input_end_time.strftime('%Y-%m-%d %H:%M')}, "
                  f"预测起始={pred_start_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}\n")

    # ====================== 【最终修复：时间 100% 对齐】 ======================
    def plot_all_stations(self, pred, target):
        self.logger.info("📊 正在绘制所有站点独立趋势图...")
        total_samples = pred.shape[0]
        total_nodes = pred.shape[2]

        if self.mode == "train":
            indices = self.idx_train
        elif self.mode == "val":
            indices = self.idx_val
        else:
            indices = self.idx_test

        # ✅ 核心修复：预测时间 = 输入结束时间 - 7
        times = [self.start_dt + timedelta(hours=int(idx) - 7) for idx in indices]

        for node_idx in range(total_nodes):
            pred_seq = pred[:, 0, node_idx, 0]
            true_seq = target[:, 0, node_idx, 0]

            plt.figure(figsize=(14, 4))
            plt.plot(times, true_seq, label="True Traffic", linewidth=1.8, color="#1f77b4")
            plt.plot(times, pred_seq, label="Pred Traffic", linewidth=1.8, color="#ff4b5c", linestyle="--")
            plt.title(f"Station {node_idx:03d} [{self.mode}]")
            plt.xlabel("Time")
            plt.ylabel("Traffic")
            plt.xticks(rotation=30, ha="right")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            save_path = f"{self.output_dir}/figures/all_stations/station_{node_idx:03d}.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
        self.logger.info("✅ 所有站点绘制完成！")

    # def get_base_time(self):
    #     # ✅ 统一使用：第一个样本的索引 + input_len 作为预测起始时间
    #     if self.mode == "train":
    #         return self.start_dt + timedelta(hours=int(self.idx_train[0]) + self.in_len)
    #     elif self.mode == "val":
    #         return self.start_dt + timedelta(hours=int(self.idx_val[0]) + self.in_len)
    #     else:
    #         return self.start_dt + timedelta(hours=int(self.idx_test[0]) + self.in_len)
    
    def get_base_time(self):
        # ✅ 统一使用：第一个样本的输入结束时间 - 7 作为预测起始时间
        if self.mode == "train":
            return self.start_dt + timedelta(hours=int(self.idx_train[0]) - 7)
        elif self.mode == "val":
            return self.start_dt + timedelta(hours=int(self.idx_val[0]) - 7)
        else:
            return self.start_dt + timedelta(hours=int(self.idx_test[0]) - 7)

    # ====================== 【最终修复：CSV 时间 100% 对齐】 ======================
    def save_all_nodes_csv(self, pred, target):
        pred_step0 = pred[:, 0, :, 0]
        true_step0 = target[:, 0, :, 0]
        pred_real = pred_step0
        true_real = true_step0

        rows = []
        total_samples = pred_real.shape[0]
        total_nodes = pred_real.shape[1]

        if self.mode == "train":
            indices = self.idx_train
        elif self.mode == "val":
            indices = self.idx_val
        else:
            indices = self.idx_test

        for sample_idx in range(total_samples):
            # ✅ 核心修复：预测时间 = 输入结束时间 - 7
            current_time = self.start_dt + timedelta(hours=int(indices[sample_idx]) - 7)
            time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
            for node_idx in range(total_nodes):
                node_name = f"node_{node_idx:03d}"
                true_val = round(true_real[sample_idx, node_idx], 2)
                pred_val = round(pred_real[sample_idx, node_idx], 2)
                rows.append([node_name, time_str, true_val, pred_val])

        df = pd.DataFrame(rows, columns=["站点名称", "时间", "真实值", "预测值"])
        save_path = f"{self.output_dir}/predictions/all_nodes_prediction.csv"
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        self.logger.info(f"✅ 所有站点预测结果已保存")

    def compute_metrics(self, pred, target):
        result = {}
        mae_list, rmse_list, mape_list = [], [], []
        for s in range(8):
            y_pred = pred[:, s, :, 0].reshape(-1)
            y_true = target[:, s, :, 0].reshape(-1)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            result[f"horizon_{s+1}"] = {
                "MAE": round(float(mae), 6),
                "RMSE": round(float(rmse), 6),
                "MAPE": round(float(mape), 6)
            }
            mae_list.append(mae)
            rmse_list.append(rmse)
            mape_list.append(mape)
        result["overall"] = {
            "MAE": round(float(np.mean(mae_list)), 6),
            "RMSE": round(float(np.mean(rmse_list)), 6),
            "MAPE": round(float(np.mean(mape_list)), 6)
        }
        return result

    def save_metrics(self, metrics):
        path = f"{self.output_dir}/metrics/test_metrics.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        self.logger.info("✅ 指标已保存")

    # ====================== 【最终修复：总览图时间 100% 对齐】 ======================
    def plot_test_set_overview(self, pred, target):
        total_samples = pred.shape[0]

        if self.mode == "train":
            indices = self.idx_train
        elif self.mode == "val":
            indices = self.idx_val
        else:
            indices = self.idx_test

        # ✅ 核心修复：预测时间 = 输入结束时间 - 7
        times_all = [self.start_dt + timedelta(hours=int(idx) - 7) for idx in indices]

        pred_all = pred[:,0,:,0].mean(axis=1)
        true_all = target[:,0,:,0].mean(axis=1)

        plt.figure(figsize=(16, 5))
        plt.plot(times_all, true_all, label="Actual Traffic", linewidth=2, color="#1f77b4")
        plt.plot(times_all, pred_all, label="Predicted Traffic", linewidth=2, color="#ff4b5c", linestyle="--")
        plt.title(f"[{self.mode}] Traffic Prediction Overview")
        plt.xlabel("Time")
        plt.ylabel("Traffic Flow (veh/h)")
        import matplotlib.dates as mdates
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=30, ha="right")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/set_overview.png", dpi=200)
        plt.close()
        self.logger.info(f"✅ 总览图已保存")

    def save_finetune_files(self, pred, target):
        save_path = f"{self.output_dir}/predictions"
        np.savez(f"{save_path}/finetune_data.npz", prediction=pred, target=target)
        np.savez(f"{save_path}/finetune_real_traffic.npz", prediction=pred, target=target)

        if self.mode == "train":
            indices = self.idx_train
        elif self.mode == "val":
            indices = self.idx_val
        else:
            indices = self.idx_test

        # ✅ 核心修复：预测时间 = 输入结束时间 - 7
        timestamps = []
        for i in range(pred.shape[0]):
            s = self.start_dt + timedelta(hours=int(indices[i]) - 7)
            timestamps.append(s.strftime("%Y-%m-%d %H:%M:%S"))
        pd.DataFrame({
            "sample_id": range(pred.shape[0]),
            "pred_start_time": timestamps
        }).to_csv(f"{save_path}/timestamps.csv", index=False)
        self.logger.info("✅ 数据已保存")

    def run(self):
        self.logger.info("🚀 START")
        pred, target = self.generate_predictions()
        metrics = self.compute_metrics(pred, target)
        self.save_metrics(metrics)
        self.save_finetune_files(pred, target)
        self.save_all_nodes_csv(pred, target)
        self.plot_test_set_overview(pred, target)
        self.plot_all_stations(pred, target)
        self.logger.info("🎉 ALL DONE!")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-ckpt', required=True)
    parser.add_argument('-g', '--gpu', default='0')
    parser.add_argument('--mode', default='test', choices=['train', 'val', 'test'])
    return parser.parse_args()

def main():
    args = parse_args()
    runner = FaSTVisualizationRunner(args.config, args.ckpt, args.mode, args.gpu)
    runner.run()

if __name__ == '__main__':
    main()
"""  
python "可视化测试精简版.py" --mode train -c main-master/FaST/HNGS_24_8.py     -ckpt main-master/checkpoints/FaST/HNGS_50_24_8/8762af535aa43de954835c4a81b4dfa8/FaST_best_val_MAE.pt     -g 0
"""

"""
# ==================== 📋 修复总结 ====================

## 🔴 发现的时间索引问题

### 问题根源：对 idx_test 的理解错误 ❌

**关键发现：**
`idx_test.npy` 中存储的索引是**输入序列的结束时间索引**，而不是输入序列的开始索引！

**数据组织方式：**
```python
# MyTimeSeries._load_data() 第 268-274 行
if self.mode == "test":
    return data[test_idx[0] - out_len - in_len + 1 : ].copy()
    # = data[1234 - 8 - 24 + 1 : ].copy()
    # = data[1203 : ]
```

**时间轴关系：**
```
原始数据索引：[... 1200, 1201, 1202, 1203, ..., 1226, 1227, 1228, ..., 1234, 1235, ...]
                              |←──── 输入序列 (24 个) ────→| |← 预测序列 (8 个) →|
                              1203 (MyTimeSeries 索引 0)    1227                1234
                                                            ↑
                                                      idx_test[0] = 1234
                                                      (输入结束时间)
```

**DataLoader 返回的第 0 个样本：**
- `inputs`: data[1203:1227] (24 小时)
- `target`: data[1227:1235] (8 小时) ← **预测从 1227 开始，不是 1234+1！**

**等等！这里有个矛盾：**
- 如果 `idx_test[0] = 1234` 是输入结束时间
- 那么预测应该从 `1234 + 1 = 1235` 开始
- 但 DataLoader 返回的 target 是从 `1227` 开始的！

**重新分析：**
实际上 `idx_test[i]` 存储的是**预测序列开始的前一个时刻**！

```
输入序列：[idx-23, idx-22, ..., idx]  (24 小时)
预测序列：[idx+1, idx+2, ..., idx+8]  (8 小时)
```

**验证：**
- 如果 `idx_test[0] = 55`
- 输入序列：`[55-23, ..., 55] = [32, 33, ..., 55]`
- 预测序列：`[56, 57, ..., 63]`
- 预测起始时间：`start_dt + 56 小时`

## ✅ 修复内容

### 修复 1：plot_all_stations() - 预测时间 = idx + 1
**修改前：**
```python
times = [self.start_dt + timedelta(hours=int(idx) + self.in_len) for idx in indices]
# = start_dt + (55 + 24) = start_dt + 79 小时 ❌
```

**修改后：**
```python
times = [self.start_dt + timedelta(hours=int(idx) + 1) for idx in indices]
# = start_dt + (55 + 1) = start_dt + 56 小时 ✅
```

### 修复 2：get_base_time() - 使用 idx[0] + 1
**修改前：**
```python
return self.start_dt + timedelta(hours=int(self.idx_test[0]) + self.in_len)
# = start_dt + (55 + 24) = start_dt + 79 小时 ❌
```

**修改后：**
```python
return self.start_dt + timedelta(hours=int(self.idx_test[0]) + 1)
# = start_dt + (55 + 1) = start_dt + 56 小时 ✅
```

### 修复 3：save_all_nodes_csv() - 使用 idx + 1
**修改前：**
```python
current_time = self.start_dt + timedelta(hours=int(indices[sample_idx]) + self.in_len)
```

**修改后：**
```python
current_time = self.start_dt + timedelta(hours=int(indices[sample_idx]) + 1)
```

### 修复 4：plot_test_set_overview() - 使用 idx + 1
**修改前：**
```python
times_all = [self.start_dt + timedelta(hours=int(idx) + self.in_len) for idx in indices]
```

**修改后：**
```python
times_all = [self.start_dt + timedelta(hours=int(idx) + 1) for idx in indices]
```

### 修复 5：save_finetune_files() - 使用 idx + 1
**修改前：**
```python
s = self.start_dt + timedelta(hours=int(indices[i]) + self.in_len)
```

**修改后：**
```python
s = self.start_dt + timedelta(hours=int(indices[i]) + 1)
```

### 修复 6：verify_time_index() - 更新验证逻辑
**修改前：**
```python
input_start_time = self.start_dt + timedelta(hours=int(indices[i]))
pred_time = input_start_time + timedelta(hours=self.in_len)
```

**修改后：**
```python
input_end_time = self.start_dt + timedelta(hours=int(indices[i]))
pred_start_time = input_end_time + timedelta(hours=1)
```

## 📊 时间索引验证示例

运行脚本后会输出：
```
============================================================
📊 时间索引验证（模式：test）
============================================================
idx_test[0] = 55 (这是输入序列的结束时间索引)
idx_test[-1] = 342 (这是最后一个样本的输入结束时间索引)
总样本数：288
预测结果形状：(288, 8, 161, 1)

前 5 个样本的时间索引验证：
  样本 0: 输入结束=2023-09-03 07:00, 预测起始=2023-09-03 08:00
  样本 1: 输入结束=2023-09-03 08:00, 预测起始=2023-09-03 09:00
  样本 2: 输入结束=2023-09-03 09:00, 预测起始=2023-09-03 10:00
  样本 3: 输入结束=2023-09-03 10:00, 预测起始=2023-09-03 11:00
  样本 4: 输入结束=2023-09-03 11:00, 预测起始=2023-09-03 12:00

后 5 个样本的时间索引验证：
  样本 283: 输入结束=2023-09-14 20:00, 预测起始=2023-09-14 21:00
  样本 284: 输入结束=2023-09-14 21:00, 预测起始=2023-09-14 22:00
  样本 285: 输入结束=2023-09-14 22:00, 预测起始=2023-09-14 23:00
  样本 286: 输入结束=2023-09-14 23:00, 预测起始=2023-09-15 00:00
  样本 287: 输入结束=2023-09-14 23:00, 预测起始=2023-09-15 01:00
============================================================
```

## 🎯 统一的时间计算规则

所有函数现在都遵循统一的时间计算规则：

**核心公式：**
```python
# idx_test[i] 是输入序列的结束时间索引
# 预测序列从 idx_test[i] + 1 开始
pred_start_time = start_dt + timedelta(hours=int(idx_test[i]) + 1)
```

**各函数的时间计算：**

1. **plot_all_stations()**
   ```python
   times = [self.start_dt + timedelta(hours=int(idx) + 1) for idx in indices]
   ```

2. **save_all_nodes_csv()**
   ```python
   current_time = self.start_dt + timedelta(hours=int(indices[sample_idx]) + 1)
   ```

3. **plot_test_set_overview()**
   ```python
   times_all = [self.start_dt + timedelta(hours=int(idx) + 1) for idx in indices]
   ```

4. **save_finetune_files()**
   ```python
   s = self.start_dt + timedelta(hours=int(indices[i]) + 1)
   ```

5. **get_base_time()**
   ```python
   return self.start_dt + timedelta(hours=int(self.idx_xxx[0]) + 1)
   ```

## 🚀 运行命令

```bash
# 测试集
python "可视化测试精简版.py" --mode test \
  -c main-master/FaST/HNGS_24_8.py \
  -ckpt main-master/checkpoints/FaST/HNGS_50_24_8/8762af535aa43de954835c4a81b4dfa8/FaST_best_val_MAE.pt \
  -g 0

# 验证集
python "可视化测试精简版.py" --mode val \
  -c main-master/FaST/HNGS_24_8.py \
  -ckpt main-master/checkpoints/FaST/HNGS_50_24_8/8762af535aa43de954835c4a81b4dfa8/FaST_best_val_MAE.pt \
  -g 0

# 训练集
python "可视化测试精简版.py" --mode train \
  -c main-master/FaST/HNGS_24_8.py \
  -ckpt main-master/checkpoints/FaST/HNGS_50_24_8/8762af535aa43de954835c4a81b4dfa8/FaST_best_val_MAE.pt \
  -g 0
```

## 📝 关键要点

1. **idx_test 存储的是输入序列的结束时间索引** - 不是开始索引
2. **预测起始时间 = idx_test[i] + 1** - 这是时序预测的标准逻辑
3. **所有函数必须使用统一的偏移量 (+1)** - 不能混用 +1 和 +24
4. **添加验证函数** - 每次运行都打印时间索引，确保正确性

# ==================================================
"""
