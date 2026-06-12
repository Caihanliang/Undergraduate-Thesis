# 导入系统操作相关库（文件路径、系统参数）
import os
import sys
# 导入EasyDict：将字典转换为可通过"."访问属性的对象，方便配置参数调用
from easydict import EasyDict

# 把当前文件的上上级目录加入Python路径（解决跨目录导入模块的问题，确保能找到basicts库）
sys.path.append(os.path.abspath(__file__ + "/../.."))
# 从basicts库导入核心组件：
# 1. 评估指标（掩码MAE、掩码APE、掩码SE，适配流量数据中的缺失值）
from basicts.metrics import masked_ae, masked_ape, masked_se
# 2. 时间序列数据集类（处理交通流量时序数据的加载和预处理）
from basicts.data import MyTimeSeries
# 3. 简单时序预测运行器（封装模型训练、验证、测试的完整流程）
from basicts.runners import SimpleTimeSeriesForecastingRunner
# 4. 通用配置获取函数（读取数据集默认配置，如训练验证测试比例、归一化方式）
from basicts.utils import get_regular_settings
# 5. PyTorch的神经网络模块（用于定义损失函数）
import torch.nn as nn


# 自定义Smooth L1损失函数包装器（相比普通L1损失，对异常值更鲁棒，适合流量预测中的突发波动）
def smooth_l1_loss_wrapper(prediction, target):
    # 初始化Smooth L1损失，beta=1表示损失在误差<=1时为二次函数，>1时为线性函数
    loss_fn = nn.SmoothL1Loss(beta=1)
    # 计算预测值与真实值的损失并返回
    return loss_fn(prediction, target)


# 从当前目录的arch.py文件中导入FaST模型（你毕设的核心预测模型）
from .arch import FaST

############################## 热门参数配置（核心可调参数，与你的毕设强相关） ##############################
# 数据集名称：指定为湖南高速数据集（HNGS），需与数据集文件夹名一致
DATA_NAME = 'HNGS'             # 改成你的湖南高速数据集 ✅
# 站点数量：湖南高速161个监测站点，与你数据集中的站点数匹配
num_nodes = 161                # 161 个高速站点 ✅
# 输入序列长度：输入24小时数据（因时间粒度为1h，对应24个时间步），用于模型学习历史规律
INPUT_LEN = 24                 # 输入：24 小时（1h一步 → 24步）✅
# 输出序列长度：输出8小时预测结果（与你大模型微调的8步预测对应，形成"GNN预测+LLM修正"闭环）
OUTPUT_LEN = 8                 # 输出：8 小时（匹配LLM大模型 8 个数值）✅
# 训练轮数：模型训练50轮，平衡训练效果与时间成本（50轮足够模型收敛，避免过拟合）
NUM_EPOCHS = 50
# 批次大小：每批处理32个样本（161个站点数据量较大，小批次可降低显存占用，避免OOM）
BATCH_SIZE = 32                # 161个站点适合小batch ✅

# 获取数据集的通用配置（从basicts库读取HNGS数据集的默认设置，无需手动定义）
regular_settings = get_regular_settings(DATA_NAME)
# 训练/验证/测试集比例（如默认7:2:1，按时序数据特性划分，避免数据泄露）
TRAIN_VAL_TEST_RATIO = regular_settings["TRAIN_VAL_TEST_RATIO"]
# 每个通道是否单独归一化（流量数据可能含多个特征，单独归一化避免特征间量级干扰）
NORM_EACH_CHANNEL = regular_settings["NORM_EACH_CHANNEL"]
# 是否重缩放数据（将数据缩放到指定范围，提升模型训练稳定性）
RESCALE = regular_settings["RESCALE"]
# 空值标记（流量数据中可能存在的缺失值标识，用于掩码指标计算时跳过缺失值）
NULL_VAL = regular_settings["NULL_VAL"]
# 模型架构：指定为FaST模型（你毕设使用的核心时空预测模型）
MODEL_ARCH = FaST

# FaST模型的核心架构参数（定义模型的网络结构，需与arch.py中的FaST类参数匹配）
MODEL_PARAM = {
    "num_nodes": num_nodes,          # 站点数量（161），用于构建站点间的空间关联
    "input_len": INPUT_LEN,          # 输入序列长度（24），匹配输入数据的时间步
    "output_len": OUTPUT_LEN,        # 输出序列长度（8），匹配预测目标的时间步
    "layers": 3,                     # 模型层数（3层 encoder-decoder 结构，平衡复杂度与效果）
    "num_experts": 8,                # 专家网络数量（8个专家分别学习不同时段的流量规律）
    "daily_steps": 24,               # 每日时间步（1h粒度下一天24步，用于捕捉日周期规律）✅
    "weekly_days": 7,                # 每周天数（7天，用于捕捉周周期规律，如工作日/周末差异）
    "hidden_dim": 64,                # 隐藏层维度（64维特征向量，平衡表达能力与显存）
    "num_agent": 32,                 # 智能体数量（32个智能体负责学习不同站点的局部特征）
}

############################## 通用配置（模型整体运行的基础设置） ##############################
# 初始化全局配置对象（所有配置参数都封装在这里，方便后续调用）
CFG = EasyDict()
# 配置描述：说明该配置的用途（湖南高速1h流量预测的FaST模型）
CFG.DESCRIPTION = "FaST for Hunan Highway 1h Traffic Forecasting"
# 是否使用FP16混合精度训练（True：使用半精度浮点数，减少显存占用，提升训练速度）
CFG["fp16"] = True
# 使用的GPU数量（1张GPU，适配大多数单机环境）
CFG.GPU_NUM = 1
# 模型运行器：指定为简单时序预测运行器（封装了训练、验证、测试的完整逻辑）
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

############################## 数据集配置（定义数据加载和预处理规则） ##############################
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME                # 数据集名称（HNGS），用于定位数据集路径
CFG.DATASET.TYPE = MyTimeSeries             # 数据集类（使用basicts定义的时序数据集类）
# 数据集参数（传递给MyTimeSeries类的初始化参数）
CFG.DATASET.PARAM = EasyDict({
    "dataset_name": DATA_NAME,                      # 数据集名称（与上文一致）
    "train_val_test_ratio": TRAIN_VAL_TEST_RATIO,    # 训练/验证/测试集比例（默认7:2:1）
    "input_len": INPUT_LEN,                          # 输入序列长度（24），切割数据时的输入步长
    "output_len": OUTPUT_LEN                        # 输出序列长度（8），切割数据时的预测步长
})

############################## 模型配置（定义模型的结构和输入输出特征） ##############################
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__        # 模型名称（自动获取FaST类的名字，用于日志和保存路径）
CFG.MODEL.ARCH = MODEL_ARCH                  # 模型架构（指定为FaST类，用于创建模型实例）
CFG.MODEL.PARAM = MODEL_PARAM                # 模型参数（传递给FaST类的初始化参数，如站点数、层数）
# 模型输入特征索引（[0,1,2]表示使用数据的第0、1、2列特征，如流量、速度、密度，需与数据集特征对应）
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
# 模型目标特征索引（[0]表示仅预测第0列特征，即流量，与你毕设的预测目标一致）
CFG.MODEL.TARGET_FEATURES = [0]

############################## 评估指标配置（定义模型性能的评价标准） ##############################
CFG.METRICS = EasyDict()
# 评估函数字典（指定训练和测试时使用的指标，均为掩码版本，适配缺失值）
CFG.METRICS.FUNCS = EasyDict({
    "MAE": masked_ae,    # 掩码平均绝对误差（核心指标，衡量预测值与真实值的平均偏差）
    "RMSE": masked_se,   # 掩码均方根误差（先计算均方误差再开方，对大误差更敏感）
    "MAPE": masked_ape  # 掩码平均绝对百分比误差（衡量相对误差，注意低流量时可能偏大）
})
# 模型优化的目标指标（以MAE为核心优化目标，与你之前的模型评估指标一致）
CFG.METRICS.TARGET = "MAE"
# 指标计算的空值标记（与数据集的NULL_VAL一致，计算时跳过缺失值，避免影响结果）
CFG.METRICS.NULL_VAL = NULL_VAL

############################## 训练配置（定义模型训练的具体规则） ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS            # 训练轮数（50轮，与上文热门参数一致）
# 模型 checkpoint 保存路径（按"模型名/数据集_轮数_输入步_输出步"命名，方便区分不同实验）
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",                          # 根目录：checkpoints（所有模型的保存目录）
    MODEL_ARCH.__name__,                    # 二级目录：模型名（FaST）
    "_".join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])  # 三级目录：HNGS_50_24_8
)

CFG.TRAIN.LOSS = smooth_l1_loss_wrapper      # 训练使用的损失函数（自定义的Smooth L1损失）
# 优化器配置（模型参数更新的算法）
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"               # 优化器类型：Adam（常用优化器，收敛稳定）
# 优化器参数（学习率和权重衰减，平衡收敛速度与过拟合）
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,             # 学习率：0.002（FaST模型的经验值，过大易震荡，过小收敛慢）
    "weight_decay": 0.0001   # 权重衰减：0.0001（L2正则化，防止模型过拟合）
}
# 学习率调度器配置（训练过程中动态调整学习率，提升后期收敛效果）
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"  # 调度器类型：MultiStepLR（按指定轮次衰减学习率）
# 调度器参数（在第10、20、30、40、50轮时，学习率乘以0.5，逐步降低学习率）
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [10, 20, 30, 40, 50], "gamma": 0.5}

# 训练数据集加载配置（控制训练时数据的读取方式）
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = BATCH_SIZE       # 批次大小（32，与上文热门参数一致）
CFG.TRAIN.DATA.SHUFFLE = True                # 是否打乱训练数据（True：打乱，避免模型学习时序顺序偏差）
CFG.TRAIN.DATA.PREFETCH = True               # 是否预加载数据（True：提前加载下一批数据，提升训练速度）
CFG.TRAIN.DATA.NUM_WORKERS = 4               # 数据加载的线程数（4个线程，平衡CPU占用与加载速度）
CFG.TRAIN.DATA.PIN_MEMORY = True             # 是否锁定内存（True：将数据固定在内存，减少GPU与内存的数据传输耗时）

# 梯度裁剪参数（限制梯度的最大范数为5.0，防止梯度爆炸，保证训练稳定）
CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}

############################## 验证配置（定义模型验证的规则） ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1                         # 验证间隔（每训练1轮后进行1次验证，及时监控模型性能）
# 验证数据集加载配置（与训练数据加载逻辑类似，但不打乱数据）
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = BATCH_SIZE         # 验证批次大小（32，与训练一致，保证显存稳定）
CFG.VAL.DATA.PREFETCH = True                 # 是否预加载（True：提升验证速度）
CFG.VAL.DATA.NUM_WORKERS = 4                 # 验证数据加载线程数（4个，与训练一致）
CFG.VAL.DATA.PIN_MEMORY = True               # 是否锁定内存（True：减少数据传输耗时）

############################## 测试配置（定义模型测试的规则） ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 200                      # 测试间隔（每训练200个迭代步骤进行1次测试，可根据数据量调整）
# 测试数据集加载配置（与验证一致，保证测试结果的稳定性）
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = BATCH_SIZE        # 测试批次大小（32，与训练/验证一致）
CFG.TEST.DATA.PREFETCH = True                # 是否预加载（True：提升测试速度）
CFG.TEST.DATA.NUM_WORKERS = 4                # 测试数据加载线程数（4个，与训练/验证一致）
CFG.TEST.DATA.PIN_MEMORY = True              # 是否锁定内存（True：减少数据传输耗时）

############################## 评估配置（定义模型最终评估的规则） ##############################
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = []                       # 评估的时间步范围（空列表表示评估所有输出步，即8个预测步）
CFG.EVAL.USE_GPU = True                      # 评估时是否使用GPU（True：利用GPU加速评估，提升效率）