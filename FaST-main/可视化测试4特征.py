#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FaST-MV 4特征可视化脚本
特征：小客车上行、小客车下行、非小客车上行、非小客车下行
python 可视化测试4特征.py
FaST-main-8_8MO4Delete/FaST-main/可视化测试4特征.py
"""

import sys
import os

# ========== 项目路径配置 ==========
BASE_PATH = "/home/user/Downloads/cai/FaST-main-8_8MO4/FaST-main"
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

# 注意：本脚本全部使用英文标题和标签，无需配置中文字体

class FaST4FEATVisualizationRunner:
    """FaST-MV 4特征可视化运行器"""
    
    # 加载配置、模型、数据集索引
    def __init__(self, config_path: str, checkpoint_path: str, mode: str, gpu_id: str = '0'):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.gpu_id = gpu_id
        self.logger = get_logger("fast-4feat-visualization")

        self.cfg = init_cfg(config_path, save=False)
        self.runner = self.cfg["RUNNER"](self.cfg)
        self.runner.init_logger(logger_name="fast-4feat-visualization")

        self.mode = mode
        self.cfg.mode = mode

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        torch.backends.cudnn.benchmark = True

        self.start_dt = self._get_start_time()
        self.output_dir = self._create_output_directory()

        # 加载索引文件
        data_root = os.path.join("main-master", "datasets", self.cfg.DATASET.NAME)
        self.in_len = self.cfg.DATASET.PARAM.input_len
        self.out_len = self.cfg.DATASET.PARAM.output_len
        sample_path = os.path.join(data_root, f"{self.in_len}_{self.out_len}")

        self.idx_train = np.load(os.path.join(sample_path, "idx_train.npy"))
        self.idx_val = np.load(os.path.join(sample_path, "idx_val.npy"))
        self.idx_test = np.load(os.path.join(sample_path, "idx_test.npy"))

        self.print_time_ranges()
        self._setup_components()
        self._load_model()

    def print_time_ranges(self):
        """打印时间范围"""
        t0 = self.start_dt + timedelta(hours=int(self.idx_train[0]))
        t1 = self.start_dt + timedelta(hours=int(self.idx_train[-1]))
        t2 = self.start_dt + timedelta(hours=int(self.idx_val[0]))
        t3 = self.start_dt + timedelta(hours=int(self.idx_val[-1]))
        t4 = self.start_dt + timedelta(hours=int(self.idx_test[0]))
        t5 = self.start_dt + timedelta(hours=int(self.idx_test[-1]))

        print("\n" + "=" * 60)
        print("✅ 输入序列起始时间范围")
        print(f"训练集 train: {t0.strftime('%Y-%m-%d %H:%M')} —— {t1.strftime('%Y-%m-%d %H:%M')}")
        print(f"验证集   val: {t2.strftime('%Y-%m-%d %H:%M')} —— {t3.strftime('%Y-%m-%d %H:%M')}")
        print(f"测试集  test: {t4.strftime('%Y-%m-%d %H:%M')} —— {t5.strftime('%Y-%m-%d %H:%M')}")
        print("=" * 60 + "\n")

    # 解析数据集时间范围
    def _get_start_time(self):
        """获取数据集起始时间"""
        desc_path = os.path.join("main-master", "datasets", self.cfg.DATASET.NAME, "desc.json")
        with open(desc_path, 'r', encoding='utf-8') as f:
            desc = json.load(f)
            start = desc["time_range"].split(" to ")[0]
        return datetime.strptime(start, "%Y-%m-%d")

    # 设置测试/验证数据加载器
    def _setup_components(self):
        """初始化数据加载器"""
        self.logger.info("Setting up components...")
        if self.runner.need_setup_graph:
            self.runner.setup_graph(cfg=self.cfg, train=False)

        from basicts.data import MyTimeSeries
        from torch.utils.data import DataLoader

        dataset_cfg = self.cfg.DATASET.PARAM
        input_len = dataset_cfg.input_len
        output_len = dataset_cfg.output_len

        if self.mode == "train":
            dataset = MyTimeSeries(dataset_name=dataset_cfg.dataset_name, 
                                   train_val_test_ratio=dataset_cfg.train_val_test_ratio, 
                                   mode="train", input_len=input_len, output_len=output_len)
        elif self.mode == "val":
            dataset = MyTimeSeries(dataset_name=dataset_cfg.dataset_name, 
                                   train_val_test_ratio=dataset_cfg.train_val_test_ratio, 
                                   mode="valid", input_len=input_len, output_len=output_len)
        else:
            dataset = MyTimeSeries(dataset_name=dataset_cfg.dataset_name, 
                                   train_val_test_ratio=dataset_cfg.train_val_test_ratio, 
                                   mode="test", input_len=input_len, output_len=output_len)

        batch_size = self.cfg.TRAIN.DATA.BATCH_SIZE
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.runner.test_data_loader = loader

    #  加载训练好的PyTorch模型
    def _load_model(self):
        """加载模型"""
        self.logger.info(f"Loading model from {self.checkpoint_path}")
        self.runner.load_model(ckpt_path=self.checkpoint_path, strict=True)
        self.runner.model.eval()
    # 创建输出目录
    def _create_output_directory(self):
        """创建输出目录"""
        root = "visulization-result-4FEAT"
        os.makedirs(root, exist_ok=True)
        os.makedirs(f"{root}/metrics", exist_ok=True)
        os.makedirs(f"{root}/predictions", exist_ok=True)
        os.makedirs(f"{root}/figures", exist_ok=True)
        os.makedirs(f"{root}/figures/all_stations", exist_ok=True)
        return root

    # 运行模型生成所有预测
    @torch.no_grad()
    def generate_predictions(self):
        """生成预测结果"""
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
        return pred, target

    def compute_metrics(self, pred, target):
        """计算4个特征的评估指标"""
        result = {}
        feature_names = ["LittleCar_Up", "LittleCar_Down", "NonLittleCar_Up", "NonLittleCar_Down"]
        
        for feat_idx, feat_name in enumerate(feature_names):
            result[feat_name] = {}
            mae_list, rmse_list, mape_list = [], [], []
            
            for s in range(8):
                y_pred = pred[:, s, :, feat_idx].reshape(-1)
                y_true = target[:, s, :, feat_idx].reshape(-1)
                
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
                
                result[feat_name][f"horizon_{s+1}"] = {
                    "MAE": round(float(mae), 6),
                    "RMSE": round(float(rmse), 6),
                    "MAPE": round(float(mape), 6)
                }
                mae_list.append(mae)
                rmse_list.append(rmse)
                mape_list.append(mape)
            
            result[feat_name]["overall"] = {
                "MAE": round(float(np.mean(mae_list)), 6),
                "RMSE": round(float(np.mean(rmse_list)), 6),
                "MAPE": round(float(np.mean(mape_list)), 6)
            }
        
        return result

    def save_metrics(self, metrics):
        """保存指标"""
        path = f"{self.output_dir}/metrics/test_metrics.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        self.logger.info("✅ Metrics saved")

    def plot_overview(self, pred, target):
        """绘制4特征总览图"""
        total_samples = pred.shape[0]

        if self.mode == "train":
            indices = self.idx_train
        elif self.mode == "val":
            indices = self.idx_val
        else:
            indices = self.idx_test

        # ✅ 修复：动态生成时间轴，处理索引不足的情况
        times_all = []
        for i in range(total_samples):
            if i < len(indices):
                current_idx = int(indices[i])
            else:
                last_idx = int(indices[-1])
                current_idx = last_idx + (i - len(indices) + 1)
            # 预测时间 = 输入结束时间 - 7 + 8 = 输入结束时间 + 1
            adjusted_idx = max(0, current_idx + 8)
            times_all.append(self.start_dt + timedelta(hours=adjusted_idx))

        feature_names = ["LittleCar Up", "LittleCar Down", "NonLittleCar Up", "NonLittleCar Down"]
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, (ax, feat_name) in enumerate(zip(axes, feature_names)):
            # 计算所有站点在该特征、该预测步长上的平均值
            pred_all = pred[:, 0, :, idx].mean(axis=1)
            true_all = target[:, 0, :, idx].mean(axis=1)

            ax.plot(times_all, true_all, label="True", linewidth=2, color="#1f77b4")
            ax.plot(times_all, pred_all, label="Pred", linewidth=2, color="#ff4b5c", linestyle="--")
            ax.set_title(f"[{self.mode}] {feat_name} Traffic Overview")
            ax.set_xlabel("Time")
            ax.set_ylabel("Traffic Flow (veh/h)")
            
            import matplotlib.dates as mdates
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.tick_params(axis='x', rotation=30)
            ax.grid(alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/overview_4features.png", dpi=200)
        plt.close()
        self.logger.info("✅ Overview plot saved")

    def plot_all_stations(self, pred, target, error_threshold=0.3, max_annotations=10):
        """绘制所有站点独立趋势图（仅第1步预测）"""
        self.logger.info("📊 Generating all stations plots...")
        total_samples = pred.shape[0]
        total_nodes = pred.shape[2]

        if self.mode == "train":
            indices = self.idx_train
        elif self.mode == "val":
            indices = self.idx_val
        else:
            indices = self.idx_test

        # ✅ 修复：动态生成时间轴，处理索引不足的情况
        times = []
        for i in range(total_samples):
            if i < len(indices):
                current_idx = int(indices[i])
            else:
                last_idx = int(indices[-1])
                current_idx = last_idx + (i - len(indices) + 1)
            # 预测时间 = 输入结束时间 + 1
            adjusted_idx = max(0, current_idx + 8)
            times.append(self.start_dt + timedelta(hours=adjusted_idx))

        feature_names = ["LittleCar_Up", "LittleCar_Down", "NonLittleCar_Up", "NonLittleCar_Down"]
        
        for feat_idx, feat_name in enumerate(feature_names):
            feat_dir = f"{self.output_dir}/figures/all_stations/{feat_name}"
            os.makedirs(feat_dir, exist_ok=True)
            
            for node_idx in range(total_nodes):
                pred_seq = pred[:, 0, node_idx, feat_idx]
                true_seq = target[:, 0, node_idx, feat_idx]

                plt.figure(figsize=(14, 4))
                plt.plot(times, true_seq, label="True Traffic", linewidth=1.8, color="#1f77b4")
                plt.plot(times, pred_seq, label="Pred Traffic", linewidth=1.8, color="#ff4b5c", linestyle="--")
                 # 🔴 新增：检测并标注高误差点
                # 计算相对误差（避免除以0）
                epsilon = 1e-8
                abs_error = np.abs(pred_seq - true_seq)
                relative_error = abs_error / (np.abs(true_seq) + epsilon)
                
                # 找出误差超过阈值的点
                high_error_mask = relative_error > error_threshold
                
                # 排除真实值为0的情况（此时相对误差无意义）
                non_zero_mask = np.abs(true_seq) > epsilon
                high_error_mask = high_error_mask & non_zero_mask
                
                # 限制标注数量，只标注误差最大的前N个点
                high_error_indices = np.where(high_error_mask)[0]
                if len(high_error_indices) > 0:
                    # 按误差大小排序
                    error_values = relative_error[high_error_indices]
                    sorted_indices = high_error_indices[np.argsort(-error_values)]
                    # 只取前max_annotations个
                    annotated_indices = sorted_indices[:max_annotations]
                    
                    # 在图上标注高误差点
                    for idx in annotated_indices:
                        time_point = times[idx]
                        true_val = true_seq[idx]
                        pred_val = pred_seq[idx]
                        error_pct = relative_error[idx] * 100
                        
                        # 绘制红色圆圈标记
                        plt.plot(time_point, true_val, 'o', color='red', markersize=12, 
                                markeredgecolor='darkred', markeredgewidth=2, zorder=5)
                        plt.plot(time_point, pred_val, 'o', color='orange', markersize=8,
                                markeredgecolor='darkorange', markeredgewidth=1.5, zorder=5)
                        
                        # 添加误差百分比标注
                        plt.annotate(f'er:{error_pct:.1f}%\n{time_point.strftime("%m-%d %H:%M")}',
                                   xy=(time_point, true_val),
                                   xytext=(0, 20),
                                   textcoords='offset points',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                                            alpha=0.7, edgecolor='orange'),
                                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                                   fontsize=8,
                                   ha='center')
            
 

                plt.title(f"Station {node_idx:03d} [{self.mode}] - {feat_name}")
                plt.xlabel("Time")
                plt.ylabel("Traffic")
                plt.xticks(rotation=30, ha="right")
                plt.grid(alpha=0.3)
                plt.legend()
                plt.tight_layout()
                save_path = f"{feat_dir}/station_{node_idx:03d}.png"
                plt.savefig(save_path, dpi=150)
                plt.close()
        
        self.logger.info("✅ All stations plots saved")

    def save_all_nodes_csv(self, pred, target):
        """保存所有站点的预测结果为CSV"""
        self.logger.info("💾 Saving all nodes CSV...")
        
        # 提取第1步预测的结果
        pred_step0 = pred[:, 0, :, :]  # (samples, nodes, 4 features)
        true_step0 = target[:, 0, :, :]

        rows = []
        total_samples = pred_step0.shape[0]
        total_nodes = pred_step0.shape[1]
        feature_names = ["LittleCar_Up", "LittleCar_Down", "NonLittleCar_Up", "NonLittleCar_Down"]

        if self.mode == "train":
            indices = self.idx_train
        elif self.mode == "val":
            indices = self.idx_val
        else:
            indices = self.idx_test

        for sample_idx in range(total_samples):
            # ✅ 修复：动态计算时间，处理索引不足的情况
            if sample_idx < len(indices):
                current_idx = int(indices[sample_idx])
            else:
                last_idx = int(indices[-1])
                current_idx = last_idx + (sample_idx - len(indices) + 1)
            
            # 预测时间 = 输入结束时间 + 1
            adjusted_idx = max(0, current_idx + 8)
            current_time = self.start_dt + timedelta(hours=adjusted_idx)
            time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
            
            for node_idx in range(total_nodes):
                node_name = f"node_{node_idx:03d}"
                for feat_idx, feat_name in enumerate(feature_names):
                    true_val = round(true_step0[sample_idx, node_idx, feat_idx], 2)
                    pred_val = round(pred_step0[sample_idx, node_idx, feat_idx], 2)
                    rows.append([node_name, time_str, feat_name, true_val, pred_val])

        df = pd.DataFrame(rows, columns=["站点名称", "时间", "特征", "真实值", "预测值"])
        save_path = f"{self.output_dir}/predictions/all_nodes_prediction.csv"
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        self.logger.info(f"✅ All nodes CSV saved to {save_path}")

    def save_finetune_files(self, pred, target):
        """保存大模型微调所需的NPZ文件
        
        Args:
            pred: GNN模型的预测值 (已反归一化)，形状: (samples, steps, nodes, features)
            target: 数据集的真实值 (已反归一化)，形状: (samples, steps, nodes, features)
        """
        self.logger.info("💾 Saving finetune files...")
        save_path = f"{self.output_dir}/predictions"
        
        # ✅ 正确保存逻辑：
        # finetune_data.npz: 保存GNN预测值（用于LLM输入）
        np.savez(f"{save_path}/finetune_data.npz", 
                 prediction=pred)  # 只保存预测值
        
        # finetune_real_traffic.npz: 保存真实值（用于LLM学习目标）
        np.savez(f"{save_path}/finetune_real_traffic.npz", 
                 target=target)    # 只保存真实值
        
        self.logger.info(f"✅ Finetune files saved:")
        self.logger.info(f"  - finetune_data.npz: prediction shape = {pred.shape}")
        self.logger.info(f"  - finetune_real_traffic.npz: target shape = {target.shape}")
        
        # 验证数据范围（确保是原始尺度）
        self.logger.info(f"  - Prediction range: [{pred.min():.2f}, {pred.max():.2f}]")
        self.logger.info(f"  - Target range: [{target.min():.2f}, {target.max():.2f}]")
        self.logger.info(f"  - MAE: {np.abs(pred - target).mean():.2f}")

        if self.mode == "train":
            indices = self.idx_train
        elif self.mode == "val":
            indices = self.idx_val
        else:
            indices = self.idx_test

        # ✅ 修复：处理预测样本数与索引数不一致的情况
        num_samples = pred.shape[0]
        timestamps = []
        
        for i in range(num_samples):
            if i < len(indices):
                # 如果索引存在，使用索引计算时间
                current_idx = int(indices[i])
            else:
                # 如果索引不够（例如预测样本比索引多），则基于最后一个索引往后推
                last_idx = int(indices[-1])
                current_idx = last_idx + (i - len(indices) + 1)
            
            # 预测时间 = 输入结束时间 + 1
            adjusted_idx = max(0, current_idx + 8)
            s = self.start_dt + timedelta(hours=adjusted_idx)
            timestamps.append(s.strftime("%Y-%m-%d %H:%M:%S"))
        
        pd.DataFrame({
            "sample_id": range(num_samples),
            "pred_start_time": timestamps
        }).to_csv(f"{save_path}/timestamps.csv", index=False)
        
        self.logger.info("✅ Finetune files saved")

    def run(self):
        """主运行流程"""
        self.logger.info("🚀 START - FaST-MV 4-Feature Visualization")
        
        
        # 1. 生成预测
        pred, target = self.generate_predictions()
        # """
        # 2. 计算并保存指标
        self.logger.info("📊 Computing metrics...")
        metrics = self.compute_metrics(pred, target)
        self.save_metrics(metrics)
        # """
        
        # # 3. 保存大模型微调文件
        # self.logger.info("💾 Saving finetune files...")
        # self.save_finetune_files(pred, target)
        
       
        # # 4. 保存所有站点CSV
        # self.logger.info("💾 Saving all nodes CSV...")
        # self.save_all_nodes_csv(pred, target)
        
        # # 5. 生成总览图
        # self.logger.info("📈 Generating overview plot...")
        # self.plot_overview(pred, target)
        
        # # 6. 生成所有站点独立图
        # self.logger.info("📈 Generating all stations plots...")
        # self.plot_all_stations(pred, target, error_threshold=0.3, max_annotations=10)
        
        # self.logger.info("🎉 ALL DONE!")
        

def parse_args():
    """解析命令行参数"""
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-ckpt', required=True)
    parser.add_argument('-g', '--gpu', default='0')
    parser.add_argument('--mode', default='test', choices=['train', 'val', 'test'])
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    runner = FaST4FEATVisualizationRunner(args.config, args.ckpt, args.mode, args.gpu)
    runner.run()


if __name__ == '__main__':
    main()


"""
使用示例：

# 测试集可视化
python "可视化测试4特征.py" --mode test \
  -c main-master/FaST/HNGS_8_8_4FEAT.py \
  -ckpt main-master/checkpoints/FaST_MV_4FEAT/HNGS_4FEAT_50_8_8/34b0a701b1df1ace2f9d476ccff0bcb8/FaST_MV_best_val_MAE.pt \
  -g 0

# 验证集可视化
python "可视化测试4特征.py" --mode train \
  -c main-master/FaST/HNGS_8_8_4FEAT.py \
  -ckpt main-master/checkpoints/FaST_MV_4FEAT/HNGS_4FEAT_50_8_8/ddfb7b9aab4682869a56ef40d9afeb34/FaST_MV_best_val_MAE.pt \
  -g 0



  python "可视化测试4特征.py" --mode train \
  -c main-master/FaST/HNGS_8_8_4FEAT.py \
  -ckpt main-master/checkpoints/FaST_MV_4FEAT/HNGS_4FEAT_50_8_8/4ba0f00c556d5c4f92ad2de1b8962d50/FaST_MV_best_val_MAE.pt \
  -g 0
"""
