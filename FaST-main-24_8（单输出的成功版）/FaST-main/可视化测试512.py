import sys
import os

BASE_PATH = "/home/user/Downloads/cai/FaST-main-24_8"
sys.path.append(BASE_PATH)
sys.path.append(os.path.join(BASE_PATH, "main-master"))
os.chdir(BASE_PATH)

import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from datetime import datetime, timedelta
import json
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'main-master')))
os.chdir(os.path.abspath(os.path.dirname(__file__)))

from basicts.runners import SimpleTimeSeriesForecastingRunner
from easytorch.config import init_cfg
from easytorch.utils import get_logger

# 修复matplotlib中文警告
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

        # ===================== 【核心修复】mode 提前固定 =====================
        self.mode = mode
        self.cfg.mode = mode

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        torch.backends.cudnn.benchmark = True

        self.mean, self.std = self._load_mean_std()
        self.start_dt = self._get_start_time()
        self.output_dir = self._create_output_directory()

        data_root = os.path.join("main-master", "datasets", self.cfg.DATASET.NAME)
        in_len = self.cfg.DATASET.PARAM.input_len
        out_len = self.cfg.DATASET.PARAM.output_len
        sample_path = os.path.join(data_root, f"{in_len}_{out_len}")

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
        print("✅ 真实数据集时间范围（从 .npy 索引加载）")
        print(f"训练集 train: {t0.strftime('%Y-%m-%d %H:%M')} —— {t1.strftime('%Y-%m-%d %H:%M')}")
        print(f"验证集   val: {t2.strftime('%Y-%m-%d %H:%M')} —— {t3.strftime('%Y-%m-%d %H:%M')}")
        print(f"测试集  test: {t4.strftime('%Y-%m-%d %H:%M')} —— {t5.strftime('%Y-%m-%d %H:%M')}")
        print("="*60 + "\n")

    def _load_mean_std(self):
        desc_path = os.path.join("main-master", "datasets", self.cfg.DATASET.NAME, "desc.json")
        with open(desc_path, 'r', encoding='utf-8') as f:
            desc = json.load(f)
        mean = desc["mean"][0]
        std = desc["std"][0]
        # ✅ 保留 mean 和 std 用于其他可能需要的场景，但模型输出已反归一化
        self.logger.info(f"Mean: {mean:.2f}, Std: {std:.2f} (仅用于参考，模型输出已自动反归一化)")
        return mean, std

    def _get_start_time(self):
        desc_path = os.path.join("main-master", "datasets", self.cfg.DATASET.NAME, "desc.json")
        with open(desc_path, 'r', encoding='utf-8') as f:
            desc = json.load(f)
            start = desc["time_range"].split(" to ")[0]
        return datetime.strptime(start, "%Y-%m-%d")

    # ===================== 【终极修复】手动强制加载数据集 =====================
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
            self.logger.info("✅ 手动加载 训练集 (859 样本)")
            dataset = MyTimeSeries(dataset_name=dataset_cfg.dataset_name, train_val_test_ratio=dataset_cfg.train_val_test_ratio, mode="train", input_len=input_len, output_len=output_len)
        elif self.mode == "val":
            self.logger.info("✅ 手动加载 验证集 (286 样本)")
            dataset = MyTimeSeries(dataset_name=dataset_cfg.dataset_name, train_val_test_ratio=dataset_cfg.train_val_test_ratio, mode="valid", input_len=input_len, output_len=output_len)
        else:
            self.logger.info("✅ 手动加载 测试集 (288 样本)")
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
        total_samples = len(loader.dataset)
        print(f"\n======================================")
        print(f"当前运行模式：{self.mode}")
        print(f"批次数量（batch数）：{len(loader)}")
        print(f"总样本数：{total_samples}")
        print(f"======================================\n")
        preds, targets = [], []
        for data in loader:
            out = self.runner.forward(data, train=False)
            preds.append(out["prediction"])
            targets.append(out["target"])
        pred = torch.cat(preds, dim=0).cpu().numpy()
        target = torch.cat(targets, dim=0).cpu().numpy()
        self.logger.info(f"Final shape: {pred.shape}")
        return pred, target

    def get_base_time(self):
        if self.mode == "train":
            return self.start_dt + timedelta(hours=int(self.idx_train[0]))
        elif self.mode == "val":
            return self.start_dt + timedelta(hours=int(self.idx_val[0]))
        else:
            return self.start_dt + timedelta(hours=int(self.idx_test[0]))

    # ===================== ✅ 修复：模型输出已包含反归一化，直接使用即可 =====================
    def plot_all_stations(self, pred, target):
        self.logger.info("📊 正在绘制所有站点独立趋势图（真实值来自 DataLoader，无滞后）...")

        total_samples = pred.shape[0]
        total_nodes = pred.shape[2]

        if self.mode == "test":
            indices = self.idx_test
        elif self.mode == "val":
            indices = self.idx_val
        else:
            indices = self.idx_train

        # ✅ 从配置文件动态获取 input_len，避免硬编码
        input_len = self.cfg.DATASET.PARAM.input_len
        
        # ✅ 真正正确的时间：样本索引 + input_len（输入长度）
        pred_times = [self.start_dt + timedelta(hours=int(idx) + input_len) for idx in indices]

        for node_idx in range(total_nodes):
            # ✅ 模型输出已经是原始尺度，直接使用（无需再次反归一化）
            # target shape: [batch, horizon, nodes, features]
            pred_seq = pred[:, 0, node_idx, 0]
            true_seq = target[:, 0, node_idx, 0]

            plt.figure(figsize=(14, 4))
            plt.plot(pred_times, true_seq, label="True Traffic", linewidth=1.8, color="#1f77b4")
            plt.plot(pred_times, pred_seq, label="Pred Traffic", linewidth=1.8, color="#ff4b5c", linestyle="--")
            plt.title(f"Station {node_idx:03d} [{self.mode}]")
            plt.xlabel("Time")
            plt.ylabel("Traffic")
            import matplotlib.dates as mdates
            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
            plt.xticks(rotation=30, ha="right")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            save_path = f"{self.output_dir}/figures/all_stations/station_{node_idx:03d}.png"
            plt.savefig(save_path, dpi=150)
            plt.close()

        self.logger.info("✅ 所有站点独立图绘制完成（真实值与预测值 100% 对齐，无滞后）！")

    def save_all_nodes_csv(self, pred, target):
        pred_step0 = pred[:, 0, :, 0]
        true_step0 = target[:, 0, :, 0]
        # ✅ 模型输出已经是原始尺度，无需反归一化
        pred_real = pred_step0
        true_real = true_step0

        rows = []
        total_samples = pred_real.shape[0]
        total_nodes = pred_real.shape[1]

        if self.mode == "test":
            indices = self.idx_test
        elif self.mode == "val":
            indices = self.idx_val
        else:
            indices = self.idx_train

        for sample_idx in range(total_samples):
            # ✅ 从配置文件动态获取 input_len
            input_len = self.cfg.DATASET.PARAM.input_len
            # ✅ 时间对齐：+input_len 小时
            current_time = self.start_dt + timedelta(hours=int(indices[sample_idx]) + input_len)
            time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
            for node_idx in range(total_nodes):
                node_name = f"node_{node_idx:03d}"
                true_val = round(true_real[sample_idx, node_idx], 2)
                pred_val = round(pred_real[sample_idx, node_idx], 2)
                rows.append([node_name, time_str, true_val, pred_val])

        df = pd.DataFrame(rows, columns=["站点名称", "时间", "真实值", "预测值"])
        save_path = f"{self.output_dir}/predictions/all_nodes_prediction.csv"
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        self.logger.info(f"✅ 所有站点预测结果已保存到：{save_path}")

    def compute_metrics(self, pred, target):
        result = {}
        mae_list, rmse_list, mape_list = [], [], []
        # ✅ 模型输出已经是原始尺度，直接计算指标
        for step in range(8):
            y_pred = pred[:, step, :, 0].reshape(-1)
            y_true = target[:, step, :, 0].reshape(-1)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            result[f"horizon_{step+1}"] = {
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
        self.logger.info("✅ 测试集指标 JSON 已保存")

    def save_finetune_files(self, pred, target):
        save_path = f"{self.output_dir}/predictions"
        # ✅ 保存归一化和原始尺度的数据
        np.savez(f"{save_path}/finetune_data.npz", prediction=pred, target=target)
        # ✅ 模型输出已经是原始尺度，无需再次反归一化
        pred_real = pred
        target_real = target
        np.savez(f"{save_path}/finetune_real_traffic.npz", prediction=pred_real, target=target_real)

        base_time = self.get_base_time()
        timestamps = []
        for i in range(pred.shape[0]):
            s = base_time + timedelta(hours=i)
            timestamps.append(s.strftime("%Y-%m-%d %H:%M:%S"))
        pd.DataFrame({
            "sample_id": range(pred.shape[0]),
            "pred_start_time": timestamps
        }).to_csv(f"{save_path}/timestamps.csv", index=False)
        self.logger.info("✅ 大模型微调文件已保存")

    def run(self):
        self.logger.info("🚀 START")
        pred, target = self.generate_predictions()
        metrics = self.compute_metrics(pred, target)
        self.save_metrics(metrics)
        self.save_finetune_files(pred, target)
        self.save_all_nodes_csv(pred, target)
        # self.plot_test_set_overview(pred, target)
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
    # ===================== 【关键】把 mode 直接传入构造函数 =====================
    runner = FaSTVisualizationRunner(args.config, args.ckpt, args.mode, args.gpu)
    runner.run()

if __name__ == '__main__':
    main()


"""
# 脚本运行命令（你提供的实际执行命令）：
# -c：指定模型配置文件路径
# -ckpt：指定模型权重文件路径
# -g：指定使用的GPU编号（0号GPU）
python "可视化测试.py"     -c main-master/FaST/HNGS_24_8.py     -ckpt main-master/checkpoints/FaST/HNGS_512_50_24_8/93bcc7fd7df0a188ae3f5bceb16153ce/FaST_best_val_MAE.pt     -g 0

FaST-main-24_8（161个站点）/FaST-main

python "可视化测试512.py" --mode test -c main-master/FaST/HNGS_24_8_512.py     -ckpt main-master/checkpoints/FaST/HNGS_512_50_24_8/93bcc7fd7df0a188ae3f5bceb16153ce/FaST_best_val_MAE.pt     -g 0
python "可视化测试512.py" --mode val -c main-master/FaST/HNGS_24_8_512.py     -ckpt main-master/checkpoints/FaST/HNGS_512_50_24_8/93bcc7fd7df0a188ae3f5bceb16153ce/FaST_best_val_MAE.pt     -g 0
python "可视化测试512.py" --mode train -c main-master/FaST/HNGS_24_8_512.py     -ckpt main-master/checkpoints/FaST/HNGS_512_50_24_8/93bcc7fd7df0a188ae3f5bceb16153ce/FaST_best_val_MAE.pt     -g 0
/home/user/Downloads/cai/FaST-main-24_8/FaST-main/main-master/checkpoints/FaST/HNGS_512_50_24_8/93bcc7fd7df0a188ae3f5bceb16153ce/FaST_best_val_MAE.pt


python "可视化测试512.py" --mode train -c main-master/FaST/HNGS_24_8.py     -ckpt main-master/checkpoints/FaST/HNGS_50_24_8/8762af535aa43de954835c4a81b4dfa8/FaST_best_val_MAE.pt
"""