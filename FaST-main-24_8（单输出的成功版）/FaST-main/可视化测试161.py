# # =========== 【脚本核心功能】生成测试集预览图 + 大模型微调所需数据 ===========
# # 导入系统操作库：用于路径配置、环境变量设置
# import sys
# import os

# # 配置项目根路径：指向FaST模型的主目录（根据你的实际路径设置）
# BASE_PATH = "/home/user/Downloads/cai/FaST-main-24_8"
# # 将根路径加入Python解释器的路径列表，确保能导入main-master下的模块
# sys.path.append(BASE_PATH)
# # 将main-master子目录加入路径，确保能导入basicts等核心库
# sys.path.append(os.path.join(BASE_PATH, "main-master"))
# # 切换工作目录到项目根路径，避免路径引用混乱
# os.chdir(BASE_PATH)

# # 导入核心依赖库：
# import torch  # PyTorch框架：用于模型加载和张量运算
# import numpy as np  # 数值计算库：处理预测结果的数组运算
# import matplotlib.pyplot as plt  # 绘图库：生成测试集趋势图
# from argparse import ArgumentParser  # 命令行参数解析库：接收外部传入的配置/模型路径
# from datetime import datetime, timedelta  # 时间处理库：生成预测结果对应的时间戳
# import json  # JSON处理库：保存模型评估指标（MAE/RMSE/MAPE）
# import pandas as pd  # 数据处理库：生成161个站点的CSV结果文件
# # 从sklearn导入回归指标：计算模型的MAE、RMSE（MAPE自定义计算）
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# # 再次补充路径：确保能导入main-master下的basicts库（双重保险，避免路径问题）
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'main-master')))
# # 切换到当前脚本所在目录，确保相对路径正确
# os.chdir(os.path.abspath(os.path.dirname(__file__)))

# # 从basicts库导入核心组件：
# from basicts.runners import SimpleTimeSeriesForecastingRunner  # 时序预测运行器：封装模型测试逻辑
# from easytorch.config import init_cfg  # 配置初始化函数：加载模型的配置文件（如HNGS_24_8.py）
# from easytorch.utils import get_logger  # 日志工具：打印运行过程中的关键信息（如加载模型、保存文件）


# # 定义FaST模型可视化与数据导出的核心类（封装所有功能）
# class FaSTVisualizationRunner:
#     # 类初始化：接收配置文件路径、模型权重路径、GPU编号
#     def __init__(self, config_path: str, checkpoint_path: str, gpu_id: str = '0'):
#         # 保存外部传入的参数到类属性
#         self.config_path = config_path  # 模型配置文件路径（如HNGS_24_8.py）
#         self.checkpoint_path = checkpoint_path  # 模型权重路径（如FaST_best_val_MAE.pt）
#         self.gpu_id = gpu_id  # 使用的GPU编号（默认0号GPU）
#         # 初始化日志对象：命名为"fast-visualization"，用于打印运行日志
#         self.logger = get_logger("fast-visualization")

#         # 加载模型配置文件：从config_path读取配置（如输入步24、输出步8、站点数161等）
#         self.cfg = init_cfg(config_path, save=False)
#         # 根据配置文件中的"RUNNER"参数，创建时序预测运行器实例（此处为SimpleTimeSeriesForecastingRunner）
#         self.runner = self.cfg["RUNNER"](self.cfg)
#         # 初始化运行器的日志：确保运行器内部的日志也统一命名
#         self.runner.init_logger(logger_name="fast-visualization")

#         # 配置GPU环境：指定使用的GPU编号（如gpu_id=0则仅使用第0块GPU）
#         os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
#         # 开启CuDNN自动优化：加速PyTorch的GPU运算（针对固定形状的张量优化）
#         torch.backends.cudnn.benchmark = True

#         # 加载数据的均值和标准差（从desc.json读取，用于反归一化预测结果）
#         self.mean, self.std = self._load_mean_std()
#         # 获取数据集的起始时间（从desc.json读取，用于生成预测结果的时间戳）
#         self.start_dt = self._get_start_time()
#         # 创建输出目录（用于保存指标、预测数据、图表）
#         self.output_dir = self._create_output_directory()

#         # 初始化模型测试所需的组件（如测试数据集加载器）
#         self._setup_components()
#         # 加载预训练的FaST模型权重，并设置为评估模式
#         self._load_model()

#     # 私有方法：加载数据集的均值和标准差（用于反归一化）
#     def _load_mean_std(self):
#         # 构建desc.json的路径：desc.json保存数据集的统计信息（均值、标准差、时间范围等）
#         desc_path = os.path.join("main-master", "datasets", self.cfg.DATASET.NAME, "desc.json")
#         # 打开desc.json文件并读取内容
#         with open(desc_path, 'r', encoding='utf-8') as f:
#             desc = json.load(f)
#         # 提取第0个特征（流量特征）的均值和标准差（因模型仅预测流量，对应TARGET_FEATURES=[0]）
#         mean = desc["mean"][0]
#         std = desc["std"][0]
#         # 打印均值和标准差到日志（方便验证数据是否正确）
#         self.logger.info(f"Mean: {mean:.2f}, Std: {std:.2f}")
#         # 返回均值和标准差，用于后续反归一化
#         return mean, std

#     # 私有方法：获取数据集的起始时间（用于生成时间戳）
#     def _get_start_time(self):
#         # 同_load_mean_std，构建desc.json的路径
#         desc_path = os.path.join("main-master", "datasets", self.cfg.DATASET.NAME, "desc.json")
#         # 打开文件并读取时间范围（格式如"2023-09-02 to 2023-09-14"）
#         with open(desc_path, 'r', encoding='utf-8') as f:
#             desc = json.load(f)
#             # 分割时间范围字符串，取起始日期（如"2023-09-02"）
#             start = desc["time_range"].split(" to ")[0]
#         # 将起始日期字符串转换为datetime对象（用于后续计算每个预测样本的时间）
#         return datetime.strptime(start, "%Y-%m-%d")

#     # 私有方法：初始化模型测试的组件（主要是测试数据集加载器）
#     def _setup_components(self):
#         # 打印日志：提示正在初始化组件
#         self.logger.info("Setting up components...")
#         # 如果运行器需要构建图结构（如GNN模型的图拓扑），则初始化图
#         if self.runner.need_setup_graph:
#             self.runner.setup_graph(cfg=self.cfg, train=False)  # train=False表示测试模式
#         try:
#             # 尝试通过runer的init_test方法初始化测试组件（如测试数据集加载器）
#             self.runner.init_test(self.cfg)
#         except:
#             # 若init_test方法失败（部分版本兼容问题），则直接构建测试数据集加载器
#             self.runner.test_data_loader = self.runner.build_test_data_loader(self.cfg)

#     # 私有方法：加载预训练模型权重，并设置为评估模式
#     def _load_model(self):
#         # 打印日志：提示正在加载模型，显示模型权重路径
#         self.logger.info(f"Loading model from {self.checkpoint_path}")
#         # 加载模型权重：strict=True表示严格匹配模型参数（确保权重与模型结构一致）
#         self.runner.load_model(ckpt_path=self.checkpoint_path, strict=True)
#         # 将模型设置为评估模式（禁用Dropout等训练时的随机操作，确保预测结果稳定）
#         self.runner.model.eval()

#     # 私有方法：创建输出目录（用于保存指标、预测数据、图表）
#     def _create_output_directory(self):
#         # 定义根输出目录名称：visulization-result
#         root = "visulization-result"
#         # 创建根目录：exist_ok=True表示若目录已存在则不报错
#         os.makedirs(root, exist_ok=True)
#         # 创建子目录：保存模型评估指标（JSON文件）
#         os.makedirs(f"{root}/metrics", exist_ok=True)
#         # 创建子目录：保存预测数据（CSV、NPZ文件，用于大模型微调）
#         os.makedirs(f"{root}/predictions", exist_ok=True)
#         # 创建子目录：保存可视化图表（测试集趋势图）
#         os.makedirs(f"{root}/figures", exist_ok=True)
#         # 返回根目录路径，用于后续保存文件
#         return root

#     # 装饰器@torch.no_grad()：禁用梯度计算（测试阶段无需反向传播，节省显存并加速）
#     @torch.no_grad()
#     # 方法：生成测试集的所有预测结果（核心功能1）
#     def generate_predictions(self):
#         # 打印日志：提示正在生成所有预测结果
#         self.logger.info("Generating ALL predictions...")
#         # 初始化两个列表：分别存储预测值和真实值
#         preds, targets = [], []
#         # 遍历测试数据集加载器的每个批次
#         for data in self.runner.test_data_loader:
#             # 调用模型进行前向传播（train=False表示测试模式），获取预测结果
#             out = self.runner.forward(data, train=False)
#             # 将当前批次的预测值添加到preds列表
#             preds.append(out["prediction"])
#             # 将当前批次的真实值添加到targets列表
#             targets.append(out["target"])
#         # 将所有批次的预测值拼接为一个numpy数组（先拼接张量，再转移到CPU，最后转为numpy）
#         pred = torch.cat(preds, dim=0).cpu().numpy()
#         # 同理，拼接所有批次的真实值
#         target = torch.cat(targets, dim=0).cpu().numpy()
#         # 打印日志：显示最终预测结果的形状（如(288, 8, 161, 1)：288个样本、8步预测、161个站点、1个特征）
#         self.logger.info(f"Final shape: {pred.shape}")
#         # 返回预测值和真实值数组（后续用于计算指标、生成图表、导出数据）
#         return pred, target

#     # ==================== ✅ 新增功能1：导出161个站点的真实值&预测值CSV ====================
#     def save_all_nodes_csv(self, pred, target):
#         pred_step0 = pred[:, 0, :, 0]
#         true_step0 = target[:, 0, :, 0]
#         # ✅ 模型输出已经是原始尺度，无需反归一化
#         pred_real = pred_step0
#         true_real = true_step0

#         rows = []
#         total_samples = pred_real.shape[0]
#         total_nodes = pred_real.shape[1]
        
#         # ✅ 根据 mode 选择正确的索引数组
#         if self.mode == "train":
#             indices = self.idx_train
#         elif self.mode == "val":
#             indices = self.idx_val
#         else:
#             indices = self.idx_test
            
#         base_time = self.get_base_time()

#         for sample_idx in range(total_samples):
#             # ✅ 使用真实的时间索引
#             current_time = base_time + timedelta(hours=int(indices[sample_idx]))
#             time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
#             for node_idx in range(total_nodes):
#                 node_name = f"node_{node_idx:03d}"
#                 true_val = round(true_real[sample_idx, node_idx], 2)
#                 pred_val = round(pred_real[sample_idx, node_idx], 2)
#                 rows.append([node_name, time_str, true_val, pred_val])

#         df = pd.DataFrame(rows, columns=["站点名称", "时间", "真实值", "预测值"])
#         save_path = f"{self.output_dir}/predictions/all_nodes_prediction.csv"
#         df.to_csv(save_path, index=False, encoding="utf-8-sig")
#         self.logger.info(f"✅ 161 个站点预测结果已保存到：{save_path}")

#     # ==================== ✅ 新增功能2：计算8步长+总体的MAE/RMSE/MAPE，保存为JSON ====================
#     def compute_metrics(self, pred, target):
#         # 初始化字典：存储最终的指标结果
#         result = {}
#         # 初始化列表：分别存储8个步长的MAE、RMSE、MAPE，用于后续计算总体指标
#         mae_list, rmse_list, mape_list = [], [], []

#         # 遍历8个预测步长（horizon_1到horizon_8）
#         for step in range(8):
#             # 提取当前步长的所有预测值：shape从(288,8,161,1)→(288×161,)（展平为一维数组）
#             y_pred = pred[:, step, :, 0].reshape(-1)
#             # 同理，提取当前步长的所有真实值
#             y_true = target[:, step, :, 0].reshape(-1)

#             # 计算当前步长的MAE（平均绝对误差）
#             mae = mean_absolute_error(y_true, y_pred)
#             # 计算当前步长的RMSE（均方根误差：对MSE开平方）
#             rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#             # 计算当前步长的MAPE（平均绝对百分比误差：避免分母为0，加1e-8）
#             mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

#             # 将当前步长的指标存入result字典（格式与你要求的完全一致）
#             result[f"horizon_{step+1}"] = {
#                 "MAE": round(float(mae), 6),  # 保留6位小数，转为float类型（避免numpy类型）
#                 "RMSE": round(float(rmse), 6),
#                 "MAPE": round(float(mape), 6)
#             }
#             # 将当前步长的指标添加到列表，用于计算总体指标
#             mae_list.append(mae)
#             rmse_list.append(rmse)
#             mape_list.append(mape)

#         # 计算总体指标：8个步长指标的平均值
#         result["overall"] = {
#             "MAE": round(float(np.mean(mae_list)), 6),  # 总体MAE
#             "RMSE": round(float(np.mean(rmse_list)), 6),  # 总体RMSE
#             "MAPE": round(float(np.mean(mape_list)), 6)   # 总体MAPE
#         }
#         # 返回指标字典（后续保存为JSON）
#         return result

#     # 方法：将指标字典保存为JSON文件
#     def save_metrics(self, metrics):
#         # 定义JSON文件的保存路径
#         path = f"{self.output_dir}/metrics/test_metrics.json"
#         # 打开文件并写入指标字典：indent=4表示格式化显示（易读），ensure_ascii=False支持中文（若有）
#         with open(path, 'w', encoding='utf-8') as f:
#             json.dump(metrics, f, indent=4, ensure_ascii=False)
#         # 打印日志：提示JSON指标文件已保存
#         self.logger.info("✅ 测试集指标 JSON 已保存")

#     # ==================== 🔥 核心功能2：生成测试集总览图（真实值vs预测值趋势） ====================
#     def plot_test_set_overview(self, pred, target):
#         # 初始化列表：存储所有样本的平均预测值、平均真实值、对应的时间
#         pred_all = []
#         true_all = []
#         times_all = []

#         # 遍历每个样本（288个样本）
#         for i in range(pred.shape[0]):
#             # 计算当前样本所有站点的平均预测值（第1步预测，因误差最小）
#             pred_all.append(pred[i, 0, :, :].mean())
#             # 计算当前样本所有站点的平均真实值
#             true_all.append(target[i, 0, :, :].mean())
#             # 计算当前样本对应的时间（同save_all_nodes_csv的时间逻辑）
#             times_all.append(self.start_dt + timedelta(hours=24 + i))

#         # 反归一化：将平均预测值/真实值还原为真实流量值
#         pred_all = np.array(pred_all) * self.std + self.mean
#         true_all = np.array(true_all) * self.std + self.mean

#         # 创建画布：尺寸16×5（宽×高），适合展示长时间序列趋势
#         plt.figure(figsize=(16, 5))
#         # 绘制真实值折线：蓝色实线，线宽2，标签"Actual Traffic"
#         plt.plot(times_all, true_all, label="Actual Traffic", linewidth=2, color="#1f77b4")
#         # 绘制预测值折线：红色虚线，线宽2，标签"Predicted Traffic"
#         plt.plot(times_all, pred_all, label="Predicted Traffic", linewidth=2, color="#ff4b5c", linestyle="--")
#         # 设置图表标题：说明是测试集288个预测样本的真实值vs预测值
#         plt.title("Test Set: 288 Predictions (Actual vs Prediction)")
#         # 设置x轴标签：时间
#         plt.xlabel("Time")
#         # 设置y轴标签：流量单位（veh/h，与你的业务场景一致）
#         plt.ylabel("Traffic Flow (veh/h)")

#         # 导入matplotlib的日期处理模块：优化x轴时间显示
#         import matplotlib.dates as mdates
#         # 获取当前坐标轴对象
#         ax = plt.gca()
#         # 设置x轴主刻度：按天显示（每1天一个刻度）
#         ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
#         # 设置x轴主刻度格式：年-月-日（如2023-09-02）
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#         # 设置x轴次刻度：按6小时显示（补充细节，不拥挤）
#         ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
#         # 旋转x轴刻度标签：30度倾斜，避免重叠
#         plt.xticks(rotation=30, ha='right')
#         # 添加网格：透明度0.3，主次刻度都显示（方便读取数值）
#         plt.grid(alpha=0.3, which='both')

#         # 添加图例：显示真实值和预测值的标签（默认在右上角）
#         plt.legend()
#         # 自动调整布局：避免标签被截断
#         plt.tight_layout()
#         # 保存图表：路径为figures目录下的test_set_overview.png，分辨率200dpi（高清）
#         plt.savefig(f"{self.output_dir}/figures/test_set_overview.png", dpi=200)
#         # 关闭画布：释放内存（避免多图绘制时内存泄漏）
#         plt.close()
#         # 打印日志：提示图表已保存
#         self.logger.info("✅ 测试集总览图已保存")

#     # ==================== 🎯 核心功能3：生成大模型微调所需的NPZ和时间戳文件 ====================
#     def save_finetune_files(self, pred, target):
#         # 定义大模型微调数据的保存路径（predictions子目录）
#         save_path = f"{self.output_dir}/predictions"
#         # 保存归一化后的预测值和真实值：NPZ格式（方便大模型加载，保留原始维度）
#         np.savez(f"{save_path}/finetune_data.npz", prediction=pred, target=target)
        
#         # 反归一化：生成真实流量值的NPZ（大模型微调时可能需要真实流量参考）
#         pred_real = pred * self.std + self.mean
#         target_real = target * self.std + self.mean
#         # 保存反归一化后的预测值和真实值
#         np.savez(f"{save_path}/finetune_real_traffic.npz", prediction=pred_real, target=target_real)

#         # 初始化列表：存储每个样本的预测起始时间
#         timestamps = []
#         # 遍历每个样本，生成对应的时间戳
#         for i in range(pred.shape[0]):
#             # 计算当前样本的预测起始时间（同之前的时间逻辑）
#             s = self.start_dt + timedelta(hours=24 + i)
#             # 将时间转换为字符串格式（如"2023-09-02 00:00:00"）
#             timestamps.append(s.strftime("%Y-%m-%d %H:%M:%S"))
#         # 创建DataFrame：包含样本ID和预测起始时间（方便大模型匹配样本与时间）
#         pd.DataFrame({
#             "sample_id": range(pred.shape[0]),  # 样本ID（0~287）
#             "pred_start_time": timestamps       # 预测起始时间
#         }).to_csv(f"{save_path}/timestamps.csv", index=False)  # 保存为CSV，不保留行索引
#         # 打印日志：提示大模型微调文件已保存
#         self.logger.info("✅ 大模型微调文件已保存")

#     # 方法：脚本主运行逻辑（串联所有功能）
#     def run(self):
#         # 打印日志：提示脚本开始运行
#         self.logger.info("🚀 START")
#         # 1. 生成测试集的预测值和真实值
#         pred, target = self.generate_predictions()

#         # 2. 计算8步长+总体指标，并保存为JSON
#         metrics = self.compute_metrics(pred, target)
#         self.save_metrics(metrics)

#         # 3. 保存大模型微调所需的NPZ和时间戳文件
#         self.save_finetune_files(pred, target)
#         # 4. 保存161个站点的真实值&预测值CSV
#         self.save_all_nodes_csv(pred, target)  # 🟢 新增：导出CSV
#         # 5. 生成测试集总览图（真实值vs预测值）
#         self.plot_test_set_overview(pred, target)

#         # 打印日志：提示所有功能执行完成
#         self.logger.info("🎉 ALL DONE!")

# # 函数：解析命令行参数（接收外部传入的配置文件、模型权重、GPU编号）
# def parse_args():
#     # 创建参数解析器对象
#     parser = ArgumentParser()
#     # 添加-c/--config参数：必填，模型配置文件路径（如main-master/FaST/HNGS_24_8.py）
#     parser.add_argument('-c', '--config', required=True)
#     # 添加-ckpt参数：必填，模型权重文件路径（如FaST_best_val_MAE.pt）
#     parser.add_argument('-ckpt', required=True)
#     # 添加-g/--gpu参数：可选，GPU编号，默认0
#     parser.add_argument('-g', '--gpu', default='0')
#     # 解析命令行参数并返回
#     return parser.parse_args()

# # 函数：主函数（脚本入口）
# def main():
#     # 1. 解析命令行参数
#     args = parse_args()
#     # 2. 创建FaSTVisualizationRunner实例，传入参数
#     runner = FaSTVisualizationRunner(args.config, args.ckpt, args.gpu)
#     # 3. 调用run方法，执行所有功能
#     runner.run()

# # 脚本入口：当脚本被直接运行时，执行main函数
# if __name__ == '__main__':
#     main()

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
# 脚本运行命令（你提供的实际执行命令）：
# -c：指定模型配置文件路径
# -ckpt：指定模型权重文件路径
# -g：指定使用的GPU编号（0号GPU）
python "可视化测试.py"     -c main-master/FaST/HNGS_24_8.py     -ckpt main-master/checkpoints/FaST/HNGS_50_24_8/8762af535aa43de954835c4a81b4dfa8/FaST_best_val_MAE.pt     -g 0

FaST-main-24_8（161个站点）/FaST-main

python "可视化测试161.py" --mode test -c main-master/FaST/HNGS_24_8.py     -ckpt main-master/checkpoints/FaST/HNGS_50_24_8/8762af535aa43de954835c4a81b4dfa8/FaST_best_val_MAE.pt     -g 0
python "可视化测试.py" --mode val -c main-master/FaST/HNGS_24_8.py     -ckpt main-master/checkpoints/FaST/HNGS_50_24_8/8762af535aa43de954835c4a81b4dfa8/FaST_best_val_MAE.pt     -g 0
python "可视化测试.py" --mode train -c main-master/FaST/HNGS_24_8.py     -ckpt main-master/checkpoints/FaST/HNGS_50_24_8/8762af535aa43de954835c4a81b4dfa8/FaST_best_val_MAE.pt     -g 0

python "可视化测试161.py" --mode train -c main-master/FaST/HNGS_24_8.py     -ckpt main-master/checkpoints/FaST/HNGS_50_24_8/8762af535aa43de954835c4a81b4dfa8/FaST_best_val_MAE.pt     -g 0

"""


