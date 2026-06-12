# 深度学习预测模型
# 导入依赖和初始化
import os
import sys
from easydict import EasyDict  # 更方便的字典写法

sys.path.append(os.path.abspath(__file__ + "/../.."))
# 导入项目模块
from basicts.metrics import masked_ae, masked_ape, masked_se  # 带掩码的评估指标（处理缺失值）
from basicts.data import MyTimeSeries # 自定义时间序列数据集类
from basicts.runners import SimpleTimeSeriesForecastingRunner # 时序预测的运行器
from basicts.utils import get_regular_settings # 获取数据集的常规设置
import torch.nn as nn

#  定义损失函数
def smooth_l1_loss_wrapper(prediction, target):
    loss_fn = nn.SmoothL1Loss(beta=1)
    return loss_fn(prediction, target)


from .arch import FaST

############################## Hot Parameters ##############################
DATA_NAME = 'HNGS_512'
num_nodes = 506
INPUT_LEN = 24
OUTPUT_LEN = 8
NUM_EPOCHS = 50
BATCH_SIZE = 32

# 获取数据集的通用配置（从basicts库读取HNGS数据集的默认设置，无需手动定义）
regular_settings = get_regular_settings(DATA_NAME)
TRAIN_VAL_TEST_RATIO = regular_settings["TRAIN_VAL_TEST_RATIO"]
NORM_EACH_CHANNEL = regular_settings["NORM_EACH_CHANNEL"]
RESCALE = regular_settings["RESCALE"]
NULL_VAL = regular_settings["NULL_VAL"]
MODEL_ARCH = FaST

MODEL_PARAM = {
    "num_nodes": num_nodes,
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN,
    "layers": 3,
    "num_experts": 8,
    "daily_steps": 24,
    "weekly_days": 7,
    "hidden_dim": 64,
    "num_agent": 32,
}

############################## General Configuration ##############################
# 初始化全局配置对象（所有配置参数都封装在这里，方便后续调用）
CFG = EasyDict()   #创建一个空的字典 之后所有数据和参数都可以放进去
CFG.DESCRIPTION = "FaST for Hunan Highway 1h Traffic Forecasting"
CFG["fp16"] = True
CFG.GPU_NUM = 1
# 模型运行器：指定为简单时序预测运行器（封装了训练、验证、测试的完整逻辑）
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = MyTimeSeries
# ===================== 【修复】删除多余参数 =====================
CFG.DATASET.PARAM = EasyDict({
    "dataset_name": DATA_NAME,
    "train_val_test_ratio": TRAIN_VAL_TEST_RATIO,
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]  # 输入特征 数据格式(样本数, 时间步, 节点数, 特征数)
CFG.MODEL.TARGET_FEATURES = [0]   # 目标特征 输出特征

############################## Metrics Configuration ##############################
CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({
    "MAE": masked_ae,
    "RMSE": masked_se,
    "MAPE": masked_ape,
})
# 模型优化的目标指标（以MAE为核心优化目标，与你之前的模型评估指标一致）
CFG.METRICS.TARGET = "MAE"
# 指标计算的空值标记（与数据集的NULL_VAL一致，计算时跳过缺失值，避免影响结果）
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
# 模型 checkpoint 保存路径（按"模型名/数据集_轮数_输入步_输出步"命名，方便区分不同实验）
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    MODEL_ARCH.__name__,
    "_".join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)

# 训练使用的损失函数（自定义的Smooth L1损失）
CFG.TRAIN.LOSS = smooth_l1_loss_wrapper
# 优化器配置（模型参数更新的算法）
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam" # 优化器类型：Adam
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,            # 学习率：0.002
    "weight_decay": 0.0001, # 权重衰减：0.0001
}
# 学习率调度器配置（训练过程中动态调整学习率，提升后期收敛效果）
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"   # 调度器类型：MultiStepLR
# 调度器参数（在第10、20、30、40、50轮时，学习率乘以0.5，逐步降低学习率）
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [10, 20, 30, 40, 50], "gamma": 0.5}
# 训练数据集加载配置（控制训练时数据的读取方式）
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = BATCH_SIZE # 批次大小
CFG.TRAIN.DATA.SHUFFLE = True # 是否打乱训练数据（True：打乱，避免模型学习时序顺序偏差）  训练数据这个地方就是有所打乱
CFG.TRAIN.DATA.PREFETCH = True # 是否预加载数据（True：提前加载下一批数据，提升训练速度）
CFG.TRAIN.DATA.NUM_WORKERS = 4 # 数据加载的线程数（4个线程，平衡CPU占用与加载速度）
CFG.TRAIN.DATA.PIN_MEMORY = True # 是否锁定内存（True：将数据固定在内存，减少GPU与内存的数据传输耗时）

# 梯度裁剪参数（限制梯度的最大范数为5.0，防止梯度爆炸，保证训练稳定）
CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = BATCH_SIZE
CFG.VAL.DATA.PREFETCH = True
CFG.VAL.DATA.NUM_WORKERS = 4
CFG.VAL.DATA.PIN_MEMORY = True

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 200
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = BATCH_SIZE
CFG.TEST.DATA.PREFETCH = True
CFG.TEST.DATA.NUM_WORKERS = 4
CFG.TEST.DATA.PIN_MEMORY = True

############################## Evaluation Configuration ##############################
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = []
CFG.EVAL.USE_GPU = True
#python main-master/experiments/train_seed.py -c FaST/HNGS_24_8_512.py -g 0
# FaST-main-24_8/FaST-main/main-master/FaST/HNGS_24_8_512.py