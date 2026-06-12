# TimesFM 项目配置文件
# 用于集中管理所有路径和参数配置

import os

# ============================================================================
# 路径配置
# ============================================================================

# 项目根目录
PROJECT_ROOT = '/home/user/Downloads/cai/timesfm-master'

# 数据集目录
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')

# 原始数据文件（自动检测）
# 如果需要指定特定文件，取消下面的注释并修改文件名
# RAW_DATA_FILE = os.path.join(DATASET_DIR, '观测站小时交通量-9.csv')
RAW_DATA_FILE = None  # None 表示自动检测第一个CSV文件

# 预处理数据输出目录
PREPROCESSED_DIR = os.path.join(DATASET_DIR, 'preprocessed')

# 模型检查点目录
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints', 'traffic-lora-32x8')

# 预测结果输出目录
PREDICTION_DIR = os.path.join(PROJECT_ROOT, 'prediction_results_8x8')

# 训练日志文件
TRAINING_LOG = os.path.join(PROJECT_ROOT, 'training.log')

# ============================================================================
# 模型配置
# ============================================================================

# HuggingFace 模型 ID
MODEL_ID = 'google/timesfm-2.5-200m-transformers'

# 时序配置
CONTEXT_LEN = 32   # 输入窗口长度（小时）- TimesFM 2.5 要求 >= 32 且是 32 的倍数
HORIZON_LEN = 8    # 预测窗口长度（小时）

# ============================================================================
# 训练配置
# ============================================================================

# 训练参数
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_SAMPLES = 5000  # 每epoch采样的训练窗口数

# LoRA 配置
LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.05

# 其他
SEED = 42
DEVICE = 'cuda'  # 'cuda' 或 'cpu'

# ============================================================================
# 特征配置
# ============================================================================

# 目标特征名称
FEATURE_NAMES = [
    'passenger_car_up',       # 小客车上行
    'passenger_car_down',     # 小客车下行
    'non_passenger_car_up',   # 非小客车上行
    'non_passenger_car_down'  # 非小客车下行
]

# 特征显示名称（用于可视化）
FEATURE_DISPLAY_NAMES = {
    'passenger_car_up': 'Passenger Car Upstream',
    'passenger_car_down': 'Passenger Car Downstream',
    'non_passenger_car_up': 'Non-Passenger Car Upstream',
    'non_passenger_car_down': 'Non-Passenger Car Downstream'
}

# ============================================================================
# 数据处理配置
# ============================================================================

# 数据划分比例
TRAIN_RATIO = 0.70  # 70% 训练集
VAL_RATIO = 0.15    # 15% 验证集
TEST_RATIO = 0.15   # 15% 测试集

# 缺失值处理
FILL_METHOD = 'ffill'  # 'ffill', 'bfill', 或 'zero'

# ============================================================================
# HuggingFace 配置
# ============================================================================

# 镜像加速
HF_ENDPOINT = 'https://hf-mirror.com'

# ============================================================================
# 辅助函数
# ============================================================================

def get_raw_data_file():
    """获取原始数据文件路径（自动检测）"""
    if RAW_DATA_FILE and os.path.exists(RAW_DATA_FILE):
        return RAW_DATA_FILE
    
    # 自动检测 CSV 文件
    import glob
    csv_files = glob.glob(os.path.join(DATASET_DIR, '*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(
            f"在 {DATASET_DIR} 目录下未找到CSV文件\n"
            f"请确保数据文件存在，或在 config.py 中设置 RAW_DATA_FILE"
        )
    
    # 返回第一个找到的CSV文件
    selected_file = csv_files[0]
    print(f"自动检测到数据文件: {os.path.basename(selected_file)}")
    
    return selected_file


def ensure_directories():
    """确保所有必要的目录存在"""
    directories = [
        PREPROCESSED_DIR,
        CHECKPOINT_DIR,
        PREDICTION_DIR,
        os.path.dirname(TRAINING_LOG)
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    
    print("✓ 所有必要目录已创建/确认")


def print_config():
    """打印当前配置"""
    print("=" * 60)
    print("TimesFM 配置信息")
    print("=" * 60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据集目录: {DATASET_DIR}")
    print(f"原始数据文件: {get_raw_data_file()}")
    print(f"预处理目录: {PREPROCESSED_DIR}")
    print(f"模型检查点: {CHECKPOINT_DIR}")
    print(f"预测结果: {PREDICTION_DIR}")
    print("-" * 60)
    print(f"模型ID: {MODEL_ID}")
    print(f"时序配置: {CONTEXT_LEN}输入{HORIZON_LEN}输出")
    print(f"训练轮数: {EPOCHS}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"LoRA配置: r={LORA_R}, alpha={LORA_ALPHA}")
    print("=" * 60)


if __name__ == '__main__':
    # 测试配置
    print_config()
    ensure_directories()
