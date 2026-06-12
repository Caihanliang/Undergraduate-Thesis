import os
import sys
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../.."))

from basicts.metrics import masked_ae, masked_ape, masked_se
from basicts.data import MyTimeSeries
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.utils import get_regular_settings
import torch.nn as nn


def smooth_l1_loss_wrapper(prediction, target):
    loss_fn = nn.SmoothL1Loss(beta=1)
    return loss_fn(prediction, target)


# 🔥 导入新的 FaST-MV 模型
from .arch.fast_arch_mv import FaST_MV

############################## Hot Parameters ##############################
DATA_NAME = 'HNGS_MULTI'  # 多变量数据集
num_nodes = 160  # 站点数（不变）
num_features = 2  # 🔥 特征数：[小客车, 非小客车]

INPUT_LEN = 8
OUTPUT_LEN = 8
NUM_EPOCHS = 50
BATCH_SIZE = 32

regular_settings = get_regular_settings(DATA_NAME)
TRAIN_VAL_TEST_RATIO = regular_settings["TRAIN_VAL_TEST_RATIO"]
NORM_EACH_CHANNEL = regular_settings["NORM_EACH_CHANNEL"]
RESCALE = regular_settings["RESCALE"]
NULL_VAL = regular_settings["NULL_VAL"]
MODEL_ARCH = FaST_MV

MODEL_PARAM = {
    "num_nodes": num_nodes,
    "num_features": num_features,  # 🔥 新增：特征数
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN,
    "layers": 3,
    "num_experts": 8,
    "hidden_dim": 64,
    "num_agent": 32,
    "use_revIN": True,
    "channel_independent": True,  # 🔥 使用 Channel Independence
}

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = "FaST-MV for Multi-Vehicle Traffic Forecasting (Light + Non-Light)"
CFG["fp16"] = True
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = MyTimeSeries
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
# 🔥 使用所有特征进行预测
CFG.MODEL.FORWARD_FEATURES = [0, 1]  # [小客车, 非小客车]
CFG.MODEL.TARGET_FEATURES = [0, 1]    # 同时预测两个特征

############################## Metrics Configuration ##############################
CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({
    "MAE": masked_ae,
    "RMSE": masked_se,
    "MAPE": masked_ape,
})
CFG.METRICS.TARGET = "MAE"
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    MODEL_ARCH.__name__,
    "_".join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)

CFG.TRAIN.LOSS = smooth_l1_loss_wrapper
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
    "weight_decay": 0.0001,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [10, 20, 30, 40, 50], "gamma": 0.5}

# 🔥 添加缺失的 DATA 配置
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = BATCH_SIZE
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.PREFETCH = True
CFG.TRAIN.DATA.NUM_WORKERS = 4
CFG.TRAIN.DATA.PIN_MEMORY = True

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
