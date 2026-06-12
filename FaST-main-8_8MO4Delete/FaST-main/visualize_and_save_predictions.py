# # =========== 【固定路径：直接复制这一整块】 ===========
# import sys
# import os

# # 强制添加 main-master 路径（100% 生效）
# BASE_PATH = "/home/user/Downloads/cai/FaST-main-8_8"
# sys.path.append(BASE_PATH)
# sys.path.append(os.path.join(BASE_PATH, "main-master"))
# os.chdir(BASE_PATH)
# # ======================================================
# import os
# import sys
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from argparse import ArgumentParser
# from typing import Dict, Tuple, Optional
# import pandas as pd
# from datetime import datetime

# # 添加项目路径
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'main-master')))
# os.chdir(os.path.abspath(os.path.dirname(__file__)))


# from basicts.runners import SimpleTimeSeriesForecastingRunner
# from easytorch.config import init_cfg
# from easytorch.utils import get_logger


# class FaSTVisualizationRunner:
#     """FaST 模型预测结果可视化和保存类"""
    
#     def __init__(self, config_path: str, checkpoint_path: str, gpu_id: str = '0'):
#         """
#         初始化可视化运行器
        
#         Args:
#             config_path: 配置文件路径
#             checkpoint_path: 模型检查点路径
#             gpu_id: GPU 设备 ID
#         """
#         self.config_path = config_path
#         self.checkpoint_path = checkpoint_path
#         self.gpu_id = gpu_id
#         self.logger = get_logger("fast-visualization")
        
#         # 初始化配置
#         self.cfg = init_cfg(config_path, save=False)
#         self.runner = self.cfg["RUNNER"](self.cfg)
#         self.runner.init_logger(logger_name="fast-visualization", log_file_name="visualization_log")
        
#         # 设置 GPU
#         os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
#         torch.backends.cudnn.benchmark = True
        
#         # 设置 graph 和初始化数据加载器
#         self._setup_components()
        
#         # 加载模型
#         self._load_model()
        
#         # 创建输出目录
#         self.output_dir = self._create_output_directory()
    
#     def _setup_components(self):
#         """设置组件（graph 和数据加载器）"""
#         self.logger.info("Setting up components...")
        
#         # 设置 graph（如果需要）
#         if self.runner.need_setup_graph:
#             self.logger.info("Setting up graph...")
#             self.runner.setup_graph(cfg=self.cfg, train=False)
        
#         # 准备数据加载器 - 使用正确的方法名
#         self.logger.info("Preparing data loaders...")
#         try:
#             # 方法 1: 尝试使用 init_test (推荐)
#             self.runner.init_test(self.cfg)
#             self.logger.info("Data loaders initialized via init_test.")
#         except Exception as e:
#             self.logger.warning(f"init_test failed ({e}), trying alternative method...")
#             # 方法 2: 直接调用 build_test_data_loader
#             try:
#                 self.runner.test_data_loader = self.runner.build_test_data_loader(self.cfg)
#                 self.logger.info("Data loader built successfully via build_test_data_loader.")
#             except Exception as e2:
#                 self.logger.error(f"Both methods failed: {e2}")
#                 raise
        
#         self.logger.info("Components setup completed.")
        
#     def _load_model(self):
#         """加载模型检查点"""
#         self.logger.info(f"Loading model from {self.checkpoint_path}")
#         self.runner.load_model(ckpt_path=self.checkpoint_path, strict=True)
#         self.runner.model.eval()
        
#     def _create_output_directory(self) -> str:
#         """创建输出目录"""
#         dataset_name = self.cfg.DATASET.NAME
#         model_name = self.cfg.MODEL.NAME
#         output_len = self.cfg.DATASET.PARAM.output_len
        
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_dir = os.path.join(
#             "visualization_results",
#             f"{model_name}_{dataset_name}_{output_len}_{timestamp}"
#         )
        
#         os.makedirs(output_dir, exist_ok=True)
#         os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
#         os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
#         os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
        
#         self.logger.info(f"Output directory created: {output_dir}")
#         return output_dir
    
#     @torch.no_grad()
#     def generate_predictions(self) -> Dict[str, torch.Tensor]:
#         """
#         生成预测结果
        
#         Returns:
#             包含 prediction, target, inputs 的字典
#         """
#         self.logger.info("Generating predictions...")
        
#         prediction_list = []
#         target_list = []
#         inputs_list = []
        
#         for data in self.runner.test_data_loader:
#             forward_return = self.runner.forward(data, epoch=None, iter_num=None, train=False)
            
#             prediction_list.append(forward_return["prediction"])
#             target_list.append(forward_return["target"])
#             inputs_list.append(forward_return["inputs"])
        
#         # 合并所有批次的数据
#         results = {
#             "prediction": torch.cat(prediction_list, dim=0),
#             "target": torch.cat(target_list, dim=0),
#             "inputs": torch.cat(inputs_list, dim=0)
#         }
        
#         self.logger.info(f"Predictions generated. Shape: {results['prediction'].shape}")
#         return results
    
#     def compute_metrics(self, results: Dict[str, torch.Tensor]) -> Dict:
#         """
#         计算评估指标
        
#         Args:
#             results: 包含 prediction, target, inputs 的字典
            
#         Returns:
#             指标结果字典
#         """
#         self.logger.info("Computing evaluation metrics...")
        
#         prediction = results["prediction"]
#         target = results["target"]
        
#         metrics_results = {}
        
#         # 按时间步计算指标
#         output_len = prediction.shape[1]
#         for i in range(output_len):
#             pred_i = prediction[:, i, :, :]
#             real_i = target[:, i, :, :]
            
#             metrics_results[f"horizon_{i+1}"] = {}
            
#             # MAE
#             mae = torch.abs(pred_i - real_i).mean()
#             metrics_results[f"horizon_{i+1}"]["MAE"] = mae.item()
            
#             # RMSE
#             rmse = torch.sqrt(((pred_i - real_i) ** 2).mean())
#             metrics_results[f"horizon_{i+1}"]["RMSE"] = rmse.item()
            
#             # MAPE
#             mape = torch.abs((pred_i - real_i) / (real_i + 1e-8)) * 100
#             metrics_results[f"horizon_{i+1}"]["MAPE"] = mape.mean().item()
        
#         # 整体指标
#         metrics_results["overall"] = {
#             "MAE": torch.abs(prediction - target).mean().item(),
#             "RMSE": torch.sqrt(((prediction - target) ** 2).mean()).item(),
#             "MAPE": (torch.abs((prediction - target) / (target + 1e-8)) * 100).mean().item()
#         }
        
#         self.logger.info(f"Overall MAE: {metrics_results['overall']['MAE']:.4f}")
#         self.logger.info(f"Overall RMSE: {metrics_results['overall']['RMSE']:.4f}")
#         self.logger.info(f"Overall MAPE: {metrics_results['overall']['MAPE']:.4f}%")
        
#         return metrics_results
    
#     def save_predictions(self, results: Dict[str, torch.Tensor], 
#                         format: str = 'npz') -> str:
#         """
#         保存预测结果
        
#         Args:
#             results: 包含 prediction, target, inputs 的字典
#             format: 保存格式 ('npz', 'csv', 'pt')
            
#         Returns:
#             保存的文件路径
#         """
#         self.logger.info(f"Saving predictions in {format} format...")
        
#         # 转换为 numpy
#         results_np = {k: v.cpu().numpy() for k, v in results.items()}
        
#         if format == 'npz':
#             save_path = os.path.join(self.output_dir, "predictions", "test_results.npz")
#             np.savez(save_path, **results_np)
            
#         elif format == 'pt':
#             save_path = os.path.join(self.output_dir, "predictions", "test_results.pt")
#             torch.save(results_np, save_path)
            
#         elif format == 'csv':
#             # 保存为多个 CSV 文件
#             save_dir = os.path.join(self.output_dir, "predictions", "csv_files")
#             os.makedirs(save_dir, exist_ok=True)
            
#             # 保存预测值
#             pred_df = pd.DataFrame(results_np["prediction"].reshape(-1, results_np["prediction"].shape[-1]))
#             pred_df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)
            
#             # 保存真实值
#             target_df = pd.DataFrame(results_np["target"].reshape(-1, results_np["target"].shape[-1]))
#             target_df.to_csv(os.path.join(save_dir, "targets.csv"), index=False)
            
#             save_path = save_dir
            
#         self.logger.info(f"Predictions saved to {save_path}")
#         return save_path
    
#     def plot_prediction_vs_actual(self, results: Dict[str, torch.Tensor], 
#                                  sample_indices: list = None,
#                                  node_indices: list = None) -> str:
#         """
#         绘制预测值与真实值对比图
        
#         Args:
#             results: 包含 prediction, target, inputs 的字典
#             sample_indices: 要绘制的样本索引列表
#             node_indices: 要绘制的节点索引列表
            
#         Returns:
#             保存的图表路径
#         """
#         self.logger.info("Plotting prediction vs actual curves...")
        
#         prediction = results["prediction"].cpu().numpy()
#         target = results["target"].cpu().numpy()
        
#         # 默认绘制前 5 个样本，每个样本的前 3 个节点
#         if sample_indices is None:
#             sample_indices = list(range(min(5, prediction.shape[0])))
#         if node_indices is None:
#             node_indices = list(range(min(3, prediction.shape[2])))
        
#         fig, axes = plt.subplots(len(sample_indices), len(node_indices), 
#                                 figsize=(15, 5*len(sample_indices)))
        
#         if len(sample_indices) == 1:
#             axes = np.array([[axes]])
#         elif len(node_indices) == 1:
#             axes = axes.reshape(-1, 1)
        
#         for i, sample_idx in enumerate(sample_indices):
#             for j, node_idx in enumerate(node_indices):
#                 ax = axes[i, j]
                
#                 # 获取该样本和节点的数据
#                 pred = prediction[sample_idx, :, node_idx, 0]
#                 actual = target[sample_idx, :, node_idx, 0]
                
#                 # 绘制曲线
#                 ax.plot(pred, 'r--', label='Prediction', linewidth=2)
#                 ax.plot(actual, 'b-', label='Actual', linewidth=2)
#                 ax.set_xlabel('Time Step')
#                 ax.set_ylabel('Value')
#                 ax.set_title(f'Sample {sample_idx}, Node {node_idx}')
#                 ax.legend()
#                 ax.grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         save_path = os.path.join(self.output_dir, "figures", "prediction_vs_actual.png")
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         plt.close()
        
#         self.logger.info(f"Plot saved to {save_path}")
#         return save_path
    
#     def plot_error_distribution(self, results: Dict[str, torch.Tensor]) -> str:
#         """
#         绘制误差分布图
        
#         Args:
#             results: 包含 prediction, target, inputs 的字典
            
#         Returns:
#             保存的图表路径
#         """
#         self.logger.info("Plotting error distribution...")
        
#         prediction = results["prediction"].cpu().numpy()
#         target = results["target"].cpu().numpy()
        
#         error = prediction - target
        
#         fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
#         # 1. 误差直方图
#         axes[0].hist(error.flatten(), bins=100, edgecolor='black', alpha=0.7)
#         axes[0].set_xlabel('Error')
#         axes[0].set_ylabel('Frequency')
#         axes[0].set_title('Error Distribution Histogram')
#         axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
#         axes[0].grid(True, alpha=0.3)
        
#         # 2. 误差箱线图（按时间步）
#         error_by_step = error.transpose(1, 0, 2, 3).reshape(error.shape[1], -1)
#         sample_steps = min(20, error.shape[1])  # 只显示前 20 个时间步
#         step_indices = np.linspace(0, error.shape[1]-1, sample_steps).astype(int)
        
#         box_data = [error_by_step[i] for i in step_indices]
#         bp = axes[1].boxplot(box_data, patch_artist=True, labels=[f"{i+1}" for i in step_indices])
#         axes[1].set_xlabel('Time Step')
#         axes[1].set_ylabel('Error')
#         axes[1].set_title('Error Box Plot by Time Step')
#         axes[1].grid(True, alpha=0.3)
#         axes[1].tick_params(axis='x', rotation=45)
        
#         # 3. 误差热力图（单个样本）
#         sample_idx = 0
#         error_sample = error[sample_idx, :, :, 0]
#         im = axes[2].imshow(error_sample, cmap='RdBu_r', aspect='auto')
#         axes[2].set_xlabel('Node')
#         axes[2].set_ylabel('Time Step')
#         axes[2].set_title(f'Error Heatmap (Sample {sample_idx})')
#         plt.colorbar(im, ax=axes[2], label='Error')
        
#         plt.tight_layout()
#         save_path = os.path.join(self.output_dir, "figures", "error_distribution.png")
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         plt.close()
        
#         self.logger.info(f"Plot saved to {save_path}")
#         return save_path
    
#     def plot_metrics_heatmap(self, metrics: Dict) -> str:
#         """
#         绘制指标热力图
        
#         Args:
#             metrics: 指标结果字典
            
#         Returns:
#             保存的图表路径
#         """
#         self.logger.info("Plotting metrics heatmap...")
        
#         # 提取各时间步的指标
#         horizons = sorted([k for k in metrics.keys() if k.startswith('horizon_')])
        
#         data = {
#             'Horizon': [],
#             'Metric': [],
#             'Value': []
#         }
        
#         for horizon in horizons:
#             for metric_name in ['MAE', 'RMSE', 'MAPE']:
#                 data['Horizon'].append(horizon.replace('horizon_', ''))
#                 data['Metric'].append(metric_name)
#                 data['Value'].append(metrics[horizon][metric_name])
        
#         df = pd.DataFrame(data)
        
#         # 透视表
#         pivot_data = df.pivot(index='Horizon', columns='Metric', values='Value')
        
#         # 绘制热力图
#         fig, ax = plt.subplots(figsize=(10, max(6, len(horizons) * 0.3)))
#         sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax)
#         ax.set_title('Metrics Across Different Forecasting Horizons')
#         ax.set_xlabel('Metric')
#         ax.set_ylabel('Forecasting Horizon')
        
#         plt.tight_layout()
#         save_path = os.path.join(self.output_dir, "figures", "metrics_heatmap.png")
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         plt.close()
        
#         self.logger.info(f"Plot saved to {save_path}")
#         return save_path
    
#     def save_metrics(self, metrics: Dict) -> str:
#         """
#         保存指标结果
        
#         Args:
#             metrics: 指标结果字典
            
#         Returns:
#             保存的文件路径
#         """
#         self.logger.info("Saving metrics...")
        
#         # 保存为 JSON
#         import json
#         save_path = os.path.join(self.output_dir, "metrics", "evaluation_metrics.json")
#         with open(save_path, 'w') as f:
#             json.dump(metrics, f, indent=4)
        
#         # 同时保存为 CSV（方便查看）
#         csv_path = os.path.join(self.output_dir, "metrics", "evaluation_metrics.csv")
        
#         horizons = sorted([k for k in metrics.keys() if k.startswith('horizon_')])
        
#         df_data = []
#         for horizon in horizons:
#             row = {'Horizon': horizon.replace('horizon_', '')}
#             for metric_name in ['MAE', 'RMSE', 'MAPE']:
#                 row[metric_name] = metrics[horizon][metric_name]
#             df_data.append(row)
        
#         # 添加整体指标
#         row_overall = {'Horizon': 'Overall'}
#         for metric_name in ['MAE', 'RMSE', 'MAPE']:
#             row_overall[metric_name] = metrics['overall'][metric_name]
#         df_data.append(row_overall)
        
#         df = pd.DataFrame(df_data)
#         df.to_csv(csv_path, index=False)
        
#         self.logger.info(f"Metrics saved to {save_path} and {csv_path}")
#         return save_path
    
#     def run_full_pipeline(self, save_formats: list = ['npz', 'csv']) -> Dict:
#         """
#         运行完整的可视化和保存流程
        
#         Args:
#             save_formats: 保存格式列表
            
#         Returns:
#             包含所有输出路径的字典
#         """
#         self.logger.info("="*60)
#         self.logger.info("Starting full visualization and saving pipeline")
#         self.logger.info("="*60)
        
#         output_paths = {}
        
#         # 1. 生成预测
#         results = self.generate_predictions()
        
#         # 2. 计算指标
#         metrics = self.compute_metrics(results)
        
#         # 3. 保存预测结果
#         for fmt in save_formats:
#             path = self.save_predictions(results, format=fmt)
#             output_paths[f'predictions_{fmt}'] = path
        
#         # 4. 保存指标
#         output_paths['metrics'] = self.save_metrics(metrics)
        
#         # 5. 绘制预测 vs 实际值
#         output_paths['prediction_vs_actual'] = self.plot_prediction_vs_actual(results)
        
#         # 6. 绘制误差分布
#         output_paths['error_distribution'] = self.plot_error_distribution(results)
        
#         # 7. 绘制指标热力图
#         output_paths['metrics_heatmap'] = self.plot_metrics_heatmap(metrics)
        
#         self.logger.info("="*60)
#         self.logger.info("Pipeline completed successfully!")
#         self.logger.info(f"All results saved to: {self.output_dir}")
#         self.logger.info("="*60)
        
#         return output_paths


# def parse_args():
#     """解析命令行参数"""
#     parser = ArgumentParser(description='Visualize FaST model predictions and save results!')
    
#     # 配置文件路径
#     parser.add_argument('-c', '--config', 
#                        default='main-master/FaST/SD_96_48.py',
#                        help='Configuration file path')
    
#     # 检查点路径
#     parser.add_argument('-ckpt', '--checkpoint',
#                        default='checkpoints/FaST/SD/SD_50_96_48/best_model.pt',
#                        help='Checkpoint file path')
    
#     # GPU 设置
#     parser.add_argument('-g', '--gpu', 
#                        default='0',
#                        help='GPU device ID')
    
#     # 保存格式
#     parser.add_argument('--formats',
#                        nargs='+',
#                        default=['npz', 'csv'],
#                        choices=['npz', 'csv', 'pt'],
#                        help='Output formats for predictions')
    
#     # 仅执行特定步骤
#     parser.add_argument('--step',
#                        default='all',
#                        choices=['all', 'predict', 'visualize', 'save'],
#                        help='Run specific step only')
    
#     return parser.parse_args()


# def main():
#     """主函数"""
#     args = parse_args()
    
#     # 创建可视化运行器
#     runner = FaSTVisualizationRunner(
#         config_path=args.config,
#         checkpoint_path=args.checkpoint,
#         gpu_id=args.gpu
#     )
    
#     # 运行完整流程
#     output_paths = runner.run_full_pipeline(save_formats=args.formats)
    
#     # 打印输出路径
#     print("\n" + "="*60)
#     print("Output files:")
#     print("="*60)
#     for key, value in output_paths.items():
#         print(f"{key}: {value}")
#     print("="*60)


# if __name__ == '__main__':
#     main()




# =========== 【固定路径：直接复制这一整块】 ===========
import sys
import os

# 强制添加 main-master 路径（100% 生效）
BASE_PATH = "/home/user/Downloads/cai/FaST-main-8_8"
sys.path.append(BASE_PATH)
sys.path.append(os.path.join(BASE_PATH, "main-master"))
os.chdir(BASE_PATH)
# ======================================================
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from typing import Dict, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import json

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'main-master')))
os.chdir(os.path.abspath(os.path.dirname(__file__)))

from basicts.runners import SimpleTimeSeriesForecastingRunner
from easytorch.config import init_cfg
from easytorch.utils import get_logger


class FaSTVisualizationRunner:
    """FaST 模型预测结果可视化和保存类"""

    def __init__(self, config_path: str, checkpoint_path: str, gpu_id: str = '0'):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.gpu_id = gpu_id
        self.logger = get_logger("fast-visualization")

        self.cfg = init_cfg(config_path, save=False)
        self.runner = self.cfg["RUNNER"](self.cfg)
        self.runner.init_logger(logger_name="fast-visualization", log_file_name="visualization_log")

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        torch.backends.cudnn.benchmark = True

        # ========== 加载均值和标准差，用于还原真实流量 ==========
        self.mean, self.std = self._load_mean_std()
        # ========== 加载数据集时间范围（用于标注预测时间段） ==========
        self.data_start_time = self._get_data_start_time()

        self._setup_components()
        self._load_model()
        self.output_dir = self._create_output_directory()

    # ========== 从 desc.json 加载均值、标准差 ==========
    def _load_mean_std(self):
        desc_path = os.path.join("main-master", "datasets", self.cfg.DATASET.NAME, "desc.json")
        with open(desc_path, 'r', encoding='utf-8') as f:
            desc = json.load(f)
        mean = desc["mean"][0]
        std = desc["std"][0]
        self.logger.info(f"Mean: {mean:.2f}, Std: {std:.2f}")
        return mean, std

    # ========== 新增：从 desc.json 获取数据起始时间（用于计算预测时间段） ==========
    def _get_data_start_time(self):
        desc_path = os.path.join("main-master", "datasets", self.cfg.DATASET.NAME, "desc.json")
        with open(desc_path, 'r', encoding='utf-8') as f:
            desc = json.load(f)
        time_range = desc["time_range"].split(" to ")[0]  # 取数据起始时间（如 "2023-09-01"）
        return datetime.strptime(time_range, "%Y-%m-%d")  # 转换为 datetime 对象

    def _setup_components(self):
        self.logger.info("Setting up components...")
        if self.runner.need_setup_graph:
            self.runner.setup_graph(cfg=self.cfg, train=False)
        self.logger.info("Preparing data loaders...")
        try:
            self.runner.init_test(self.cfg)
        except:
            self.runner.test_data_loader = self.runner.build_test_data_loader(self.cfg)

    def _load_model(self):
        self.logger.info(f"Loading model from {self.checkpoint_path}")
        self.runner.load_model(ckpt_path=self.checkpoint_path, strict=True)
        self.runner.model.eval()

    def _create_output_directory(self) -> str:
        output_dir = "visulization-result"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures", "all_nodes"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures", "all_nodes_real"), exist_ok=True)
        self.logger.info(f"Output directory: {output_dir}")
        return output_dir

    @torch.no_grad()
    def generate_predictions(self) -> Dict[str, torch.Tensor]:
        self.logger.info("Generating predictions...")
        prediction_list, target_list, inputs_list = [], [], []
        for data in self.runner.test_data_loader:
            forward_return = self.runner.forward(data, epoch=None, iter_num=None, train=False)
            prediction_list.append(forward_return["prediction"])
            target_list.append(forward_return["target"])
            inputs_list.append(forward_return["inputs"])

        results = {
            "prediction": torch.cat(prediction_list, dim=0),
            "target": torch.cat(target_list, dim=0),
            "inputs": torch.cat(inputs_list, dim=0)
        }
        self.logger.info(f"Shape: {results['prediction'].shape}")
        return results

    def compute_metrics(self, results: Dict[str, torch.Tensor]) -> Dict:
        self.logger.info("Computing metrics...")
        prediction = results["prediction"]
        target = results["target"]
        metrics_results = {}
        output_len = prediction.shape[1]

        for i in range(output_len):
            pred_i = prediction[:, i, :, :]
            real_i = target[:, i, :, :]
            metrics_results[f"horizon_{i+1}"] = {
                "MAE": torch.abs(pred_i - real_i).mean().item(),
                "RMSE": torch.sqrt(((pred_i - real_i) ** 2).mean()).item(),
                "MAPE": (torch.abs((pred_i - real_i) / (real_i + 1e-8)) * 100).mean().item()
            }

        metrics_results["overall"] = {
            "MAE": torch.abs(prediction - target).mean().item(),
            "RMSE": torch.sqrt(((prediction - target) ** 2).mean()).item(),
            "MAPE": (torch.abs((prediction - target) / (target + 1e-8)) * 100).mean().item()
        }
        return metrics_results

    def save_predictions(self, results: Dict[str, torch.Tensor]):
        self.logger.info("Saving FULL 161 nodes predictions & targets...")
        pred = results["prediction"].cpu().numpy()
        target = results["target"].cpu().numpy()

        np.savez(os.path.join(self.output_dir, "predictions", "all_nodes_results.npz"),
                 prediction=pred, target=target)

        csv_dir = os.path.join(self.output_dir, "predictions", "all_nodes_csv")
        os.makedirs(csv_dir, exist_ok=True)
        num_nodes = pred.shape[2]

        for node in range(num_nodes):
            pred_node = pred[:, :, node, 0].mean(axis=0)
            target_node = target[:, :, node, 0].mean(axis=0)
            df = pd.DataFrame({
                "time_step": list(range(1, len(pred_node)+1)),
                "prediction": pred_node,
                "actual": target_node
            })
            df.to_csv(os.path.join(csv_dir, f"node_{node:03d}.csv"), index=False)

        self.logger.info("✅ All 161 nodes saved successfully!")

    def plot_all_nodes(self, results: Dict[str, torch.Tensor]):
        self.logger.info("Plotting ALL 161 nodes prediction vs actual...")
        pred = results["prediction"].cpu().numpy()
        target = results["target"].cpu().numpy()
        num_nodes = pred.shape[2]

        save_dir = os.path.join(self.output_dir, "figures", "all_nodes")

        for node in range(num_nodes):
            pred_mean = pred[:, :, node, 0].mean(axis=0)
            target_mean = target[:, :, node, 0].mean(axis=0)

            plt.figure(figsize=(10, 4))
            plt.plot(target_mean, label="Actual", linewidth=2.5)
            plt.plot(pred_mean, label="Prediction", linewidth=2.5, linestyle="--")
            plt.title(f"Node {node:03d} (Normalized)")
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"node_{node:03d}.png"), dpi=150)
            plt.close()

        self.logger.info(f"✅ All 161 node figures saved to {save_dir}")

    # ========== 【修改：横坐标显示具体预测时间段】 ==========
    def plot_real_traffic_all_nodes(self, results: Dict[str, torch.Tensor]):
        plt.rcParams['axes.unicode_minus'] = False
        self.logger.info("Plotting real traffic flow (veh/h) vs prediction...")
        pred = results["prediction"].cpu().numpy()
        target = results["target"].cpu().numpy()
        num_nodes = pred.shape[2]
        save_dir = os.path.join(self.output_dir, "figures", "all_nodes_real")

        # 计算预测的具体时间范围（以测试集第一个样本为例，标注平均趋势对应的时间段）
        # 假设输入窗口为24小时，第一个预测时间 = 数据起始时间 + 24小时
        first_pred_time = self.data_start_time + timedelta(hours=24)
        # 生成8个预测时间点（每小时一个）
        pred_times = [first_pred_time + timedelta(hours=i) for i in range(8)]
        # 格式化为 "HH:MM"（如 "00:00"）用于横坐标
        pred_time_labels = [t.strftime("%m-%d %H:00") for t in pred_times]

        for node_idx in range(num_nodes):
            pred_mean = pred[:, :, node_idx, 0].mean(axis=0)
            target_mean = target[:, :, node_idx, 0].mean(axis=0)

            # 还原真实流量
            pred_real = pred_mean * self.std + self.mean
            target_real = target_mean * self.std + self.mean

            plt.figure(figsize=(12, 5))
            plt.plot(pred_time_labels, target_real, 'b-', linewidth=3, label="Actual (veh/h)")
            plt.plot(pred_time_labels, pred_real, 'r--', linewidth=3, label="Prediction (veh/h)")
            
            # 标题标注完整预测时间段
            start_label = first_pred_time.strftime("%Y-%m-%d %H:00")
            end_label = (first_pred_time + timedelta(hours=7)).strftime("%Y-%m-%d %H:00")
            plt.title(f"Node {node_idx:03d} - 8h Traffic Forecast ({start_label} ~ {end_label})", fontsize=14)
            
            plt.xlabel("Prediction Time", fontsize=12)
            plt.ylabel("Traffic Flow (veh/h)", fontsize=12)
            plt.xticks(rotation=45)  # 旋转横坐标标签，避免重叠
            plt.legend(fontsize=12)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"node_{node_idx:03d}_real.png"), dpi=200)
            plt.close()

        self.logger.info(f"✅ All real traffic figures saved!")

    # ==========================================================

    def plot_basic_figures(self, results, metrics):
        prediction = results["prediction"].cpu().numpy()
        target = results["target"].cpu().numpy()
        error = prediction - target

        fig, ax = plt.subplots(figsize=(10,5))
        ax.hist(error.flatten(), bins=100, alpha=0.7)
        ax.axvline(0, color='red', linestyle='--')
        plt.savefig(os.path.join(self.output_dir, "figures", "error_dist.png"), dpi=200)
        plt.close()

    def save_metrics(self, metrics):
        with open(os.path.join(self.output_dir, "metrics", "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    def run_full_pipeline(self):
        self.logger.info("🚀 START FULL VISUALIZATION PIPELINE")
        res = self.generate_predictions()
        metrics = self.compute_metrics(res)
        self.save_predictions(res)
        self.plot_all_nodes(res)
        self.plot_real_traffic_all_nodes(res)
        self.plot_basic_figures(res, metrics)
        self.save_metrics(metrics)
        self.logger.info("✅ ALL DONE!")
        return self.output_dir

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-ckpt', '--checkpoint', required=True)
    parser.add_argument('-g', '--gpu', default='0')
    return parser.parse_args()

def main():
    args = parse_args()
    runner = FaSTVisualizationRunner(args.config, args.checkpoint, args.gpu)
    out = runner.run_full_pipeline()
    print("\n✅ ALL RESULTS SAVED TO:", out)

if __name__ == '__main__':
    main()

"""
python visualize_and_save_predictions.py \
    -c main-master/FaST/HNGS_96_48.py \
    -ckpt main-master/checkpoints/FaST/HNGS_50_96_48/log/FaST_best_val_MAE.pt\
    -g 0
"""


"""
python FaST-main/visualize_and_save_predictions.py \
    -c main-master/FaST/HNGS_24_8.py \
    -ckpt checkpoints/FaST/HNGS_50_24_8/8762af535aa43de954835c4a81b4dfa8/FaST_best_val_MAE.pt
-g 0

"""
"""
python visualize_and_save_predictions.py \
    -c main-master/FaST/HNGS_24_8.py \
    -ckpt main-master/checkpoints/FaST/HNGS_50_24_8/8762af535aa43de954835c4a81b4dfa8/FaST_best_val_MAE.pt
-g 0

"""