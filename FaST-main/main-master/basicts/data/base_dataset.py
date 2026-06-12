from dataclasses import dataclass
from typing import List

import numpy as np
from torch.utils.data import Dataset


@dataclass
class BaseDataset(Dataset):  
    """
    An abstract base class for creating datasets for time series forecasting in PyTorch.

    This class provides a structured template for defining custom datasets by specifying methods
    to load data and descriptions, and to access individual samples. It is designed to be subclassed
    with specific implementations for different types of time series data.

    Attributes:
        dataset_name (str): The name of the dataset which is used for identifying the dataset uniquely.
        train_val_test_ratio (List[float]): Ratios for splitting the dataset into training, validation,
            and testing sets respectively. Each value in the list should sum to 1.0.
        mode (str): Operational mode of the dataset. Valid values are "train", "valid", or "test".
        input_len (int): The length of the input sequence, i.e., the number of historical data points used.
        output_len (int): The length of the output sequence, i.e., the number of future data points predicted.
        overlap (bool): Flag to indicate whether the splits between training, validation, and testing can overlap. 
            Defaults to False to enforce non-overlapping data in different sets, but can be set to True to allow overlap.
    【抽象基类】用于时间序列预测任务的 PyTorch 数据集基类
    所有具体的数据集（如交通数据集、气象数据集）都需要继承这个类
    
    作用：
    1. 定义统一接口
    2. 规范训练/验证/测试数据加载流程
    3. 适配 PyTorch 的 Dataset 标准
    
    """
    # ===================== 数据集参数 =====================
    dataset_name: str   # 数据集唯一名称
    train_val_test_ratio: List[float]  # 训练/验证/测试划分比例，和为1.0，如 [0.6,0.2,0.2]
    mode: str  # 当前数据集模式：train / valid / test
    input_len: int  # 输入序列长度（历史多少步）
    output_len: int # 输出序列长度（预测未来多少步）
    overlap: bool = False  # 不同数据集是否允许重叠，默认不重叠

    def _load_description(self) -> dict:  # 加载数据集元信息
        """
        Abstract method to load a dataset's description from a file or source.

        This method should be implemented by subclasses to load and return the dataset's metadata, 
        such as its shape, range, or other relevant properties, typically from a JSON or similar file.

        Returns:
            dict: A dictionary containing the dataset's metadata.

        Raises:
            NotImplementedError: If the method has not been implemented by a subclass.
        【抽象方法】加载数据集描述信息（元信息）
        比如：数据均值、方差、维度、时间范围、特征名称等
        
        返回：
            dict: 数据集元信息
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def _load_data(self) -> np.ndarray:  # 加载真实数据
        """
        Abstract method to load the dataset and organize it based on the specified mode.

        This method should be implemented by subclasses to load actual time series data into an array,
        handling any necessary preprocessing and partitioning according to the specified `mode`.

        Returns:
            np.ndarray: The loaded and appropriately split dataset array.

        Raises:
            NotImplementedError: If the method has not been implemented by a subclass.
        【抽象方法】加载并划分数据集
        根据 mode（train/valid/test）返回对应数据
        
        返回：
            np.ndarray: 加载好的时序数据
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def __len__(self) -> int: # 告诉框架有多少条样本
        """
        Abstract method to get the total number of samples available in the dataset.

        This method should be implemented by subclasses to calculate and return the total number of valid
        samples available for training, validation, or testing based on the configuration and dataset size.

        Returns:
            int: The total number of samples.

        Raises:
            NotImplementedError: If the method has not been implemented by a subclass.
        
        【抽象方法】返回数据集总样本数
        PyTorch Dataset 必须实现的方法
        
        返回：
            int: 样本数量
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def __getitem__(self, idx: int) -> dict:  #  根据索引取一条（输入 + 标签）
        """
        Abstract method to retrieve a single sample from the dataset.

        This method should be implemented by subclasses to access and return a specific sample from the dataset,
        given an index. It should handle the slicing of input and output sequences according to the defined
        `input_len` and `output_len`.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the input sequence ('inputs') and output sequence ('target').

        Raises:
            NotImplementedError: If the method has not been implemented by a subclass.
        
        【抽象方法】根据索引获取单条样本
        PyTorch Dataset 必须实现的方法
        根据 idx 切分：
            历史 input_len 步 → 输入
            未来 output_len 步 → 标签（目标）
        
        参数：
            idx: 样本索引
        
        返回：
            dict: {
                "inputs": 输入序列,
                "target": 输出序列（标签）
            }
        """

        raise NotImplementedError("Subclasses must implement this method.")
