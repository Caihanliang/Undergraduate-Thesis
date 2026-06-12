# ==================== 【FaST 配置文件 - 4特征版本】 ====================
"""
数据集: HNGS_4FEAT (4特征: 小客车上行、小客车下行、非小客车上行、非小客车下行)
输入: 8小时
输出: 8小时
"""

import os
import sys
from easydict import EasyDict

# 添加项目路径
sys.path.append(os.path.abspath(__file__ + "/../.."))

from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.data import MyTimeSeries
from basicts.metrics import masked_ae, masked_ape, masked_se
import torch.nn as nn

# 导入多变量模型架构
from .arch import FaST_MV

def smooth_l1_loss_wrapper(prediction, target):
    """Smooth L1 Loss 包装器"""
    loss_fn = nn.SmoothL1Loss(beta=1)
    return loss_fn(prediction, target)

# ==================== CFG 配置字典 ====================
CFG = EasyDict()

# ==================== 基本配置 ====================
CFG.DESCRIPTION = "FaST-MV for 4-Feature Traffic Forecasting (LittleCar Up/Down + NonLittleCar Up/Down)"
CFG.fp16 = True
CFG.GPU_NUM = 1

# ==================== Runner配置 ====================
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

# ==================== 数据集配置 ====================
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = "HNGS_4FEAT"
CFG.DATASET.TYPE = MyTimeSeries
CFG.DATASET.PARAM = EasyDict({
    "dataset_name": "HNGS_4FEAT",
    "train_val_test_ratio": [0.6, 0.2, 0.2],
    "input_len": 8,
    "output_len": 8,
})

# ==================== 模型配置 ====================
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "FaST_MV"
CFG.MODEL.ARCH = FaST_MV  # 使用类对象，不是字符串
CFG.MODEL.PARAM = EasyDict({
    "num_nodes": 157,           # 站点数量（删除4个零值最多的站点后）
    "num_features": 4,           # 特征数量：小客车上行、小客车下行、非小客车上行、非小客车下行
    "input_len": 8,
    "output_len": 8,
    "layers": 3,
    "num_experts": 8,
    "hidden_dim": 64,
    "num_agent": 32,
    "use_revIN": True,
    "channel_independent": True,
})
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]  # 使用前向特征（所有4个特征）
CFG.MODEL.TARGET_FEATURES = [0, 1, 2, 3]    # 预测目标（所有4个特征）

# ==================== 评估指标配置 ====================
CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({
    "MAE": masked_ae,
    "RMSE": masked_se,
    "MAPE": masked_ape,
})
CFG.METRICS.TARGET = "MAE"
CFG.METRICS.NULL_VAL = 0.0

# ==================== 训练配置 ====================
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = 50
CFG.TRAIN.CKPT_SAVE_DIR = "checkpoints/FaST_MV_4FEAT/HNGS_4FEAT_50_8_8"
CFG.TRAIN.LOSS = smooth_l1_loss_wrapper
CFG.TRAIN.OPTIM = EasyDict({
    "TYPE": "Adam",
    "PARAM": EasyDict({
        "lr": 0.001,
        "weight_decay": 0.0001,
    })
})
CFG.TRAIN.LR_SCHEDULER = EasyDict({
    "TYPE": "MultiStepLR",
    "PARAM": EasyDict({
        "milestones": [10, 20, 30, 40, 50],
        "gamma": 0.5,
    })
})
CFG.TRAIN.DATA = EasyDict({
    "BATCH_SIZE": 32,
    "SHUFFLE": True,
    "PREFETCH": True,
    "NUM_WORKERS": 4,
    "PIN_MEMORY": True,
})
CFG.TRAIN.CLIP_GRAD_PARAM = EasyDict({
    "max_norm": 5.0,
})

# ==================== 验证配置 ====================
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict({
    "BATCH_SIZE": 32,
    "PREFETCH": True,
    "NUM_WORKERS": 4,
    "PIN_MEMORY": True,
})

# ==================== 测试配置 ====================
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 200
CFG.TEST.DATA = EasyDict({
    "BATCH_SIZE": 32,
    "PREFETCH": True,
    "NUM_WORKERS": 4,
    "PIN_MEMORY": True,
})

# ==================== 评估配置 ====================
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = []  # 空列表表示评估所有horizon
CFG.EVAL.USE_GPU = True
"""
python main-master/experiments/train_seed.py \ -c FaST/HNGS_8_8_4FEAT.py \ -g 0
"""
