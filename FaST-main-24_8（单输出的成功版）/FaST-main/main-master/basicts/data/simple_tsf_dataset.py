import inspect
import json
import logging
import os
from typing import List

import numpy as np

from .base_dataset import BaseDataset


class TimeSeriesForecastingDataset(BaseDataset):
    """
    A dataset class for time series forecasting problems, handling the loading, parsing, and partitioning
    of time series data into training, validation, and testing sets based on provided ratios.

    This class supports configurations where sequences may or may not overlap, accommodating scenarios
    where time series data is drawn from continuous periods or distinct episodes, affecting how
    the data is split into batches for model training or evaluation.

    Attributes:
        data_file_path (str): Path to the file containing the time series data.
        description_file_path (str): Path to the JSON file containing the description of the dataset.
        data (np.ndarray): The loaded time series data array, split according to the specified mode.
        description (dict): Metadata about the dataset, such as shape and other properties.
    时间序列预测标准数据集类
    功能：自动加载数据、按比例划分训练/验证/测试集、支持滑动窗口采样
    适用于常规时序预测任务（交通、电量、天气等）
    """

    def __init__(
        self,
        dataset_name: str,   # 数据集名称
        train_val_test_ratio: List[float], # 训练/验证/测试比例
        mode: str, # 当前模式：train/valid/test
        input_len: int,  # 输入序列长度
        output_len: int,  # 预测序列长度
        overlap: bool = False, # 样本是否允许重叠
        logger: logging.Logger = None, # 日志
    ) -> None:
        """
        Initializes the TimeSeriesForecastingDataset by setting up paths, loading data, and
        preparing it according to the specified configurations.

        Args:
            dataset_name (str): The name of the dataset.
            train_val_test_ratio (List[float]): Ratios for splitting the dataset into train, validation, and test sets.
                Each value should be a float between 0 and 1, and their sum should ideally be 1.
            mode (str): The operation mode of the dataset. Valid values are 'train', 'valid', or 'test'.
            input_len (int): The length of the input sequence (number of historical points).
            output_len (int): The length of the output sequence (number of future points to predict).
            overlap (bool): Flag to determine if training/validation/test splits should overlap.
                Defaults to False for strictly non-overlapping periods. Set to True to allow overlap.
            logger (logging.Logger): logger.

        Raises:
            AssertionError: If `mode` is not one of ['train', 'valid', 'test'].
        """
        # 检查模式是否合法
        assert mode in [
            "train",
            "valid",
            "test",
        ], f"Invalid mode: {mode}. Must be one of ['train', 'valid', 'test']."
        # 调用父类构造
        super().__init__(
            dataset_name, train_val_test_ratio, mode, input_len, output_len, overlap
        )
        self.logger = logger

        # 数据路径与描述文件路径
        self.data_file_path = f"datasets/{dataset_name}/data.dat"
        self.description_file_path = f"datasets/{dataset_name}/desc.json"
        # 加载描述信息 & 数据
        self.description = self._load_description()
        self.data = self._load_data()

    def _load_description(self) -> dict:
        """
        加载数据集描述文件（JSON）：包含数据形状、归一化信息等
        Loads the description of the dataset from a JSON file.

        Returns:
            dict: A dictionary containing metadata about the dataset, such as its shape and other properties.

        Raises:
            FileNotFoundError: If the description file is not found.
            json.JSONDecodeError: If there is an error decoding the JSON data.
        """

        try:
            with open(self.description_file_path, "r") as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Description file not found: {self.description_file_path}"
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error decoding JSON file: {self.description_file_path}"
            ) from e

    def _load_data(self) -> np.ndarray:
        """
        Loads the time series data from a file and splits it according to the selected mode.

        Returns:
            np.ndarray: The data array for the specified mode (train, validation, or test).

        Raises:
            ValueError: If there is an issue with loading the data file or if the data shape is not as expected.
         核心功能：
        1. 从磁盘加载大数据（memmap 内存映射，不占内存）
        2. 按比例切分 train/valid/test
        3. 处理 overlap（数据太短时自动开启重叠采样
        """
         # 内存映射读取大型数据
        try:
            data = np.memmap(
                self.data_file_path,
                dtype="float32",
                mode="r",
                shape=tuple(self.description["shape"]),
            )
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Error loading data file: {self.data_file_path}") from e

        # 按比例计算各段长度
        total_len = len(data)
        valid_len = int(total_len * self.train_val_test_ratio[1])
        test_len = int(total_len * self.train_val_test_ratio[2])
        train_len = total_len - valid_len - test_len

        # Automatically configure the overlap parameter
        # 如果数据过短，自动开启 overlap（避免样本数为0）
        minimal_len = self.input_len + self.output_len
        if (
            minimal_len
            > {"train": train_len, "valid": valid_len, "test": test_len}[self.mode]
        ):
            self.overlap = True  # Enable overlap when the train, validation, or test set is too short
            current_frame = inspect.currentframe()
            file_name = inspect.getfile(current_frame)
            line_number = current_frame.f_lineno - 7
            dataset = {"train": "Training", "valid": "Validation", "test": "Test"}[
                self.mode
            ]
            if self.logger is not None:
                self.logger.info(
                    f"{dataset} dataset is too short, enabling overlap. See details in {file_name} at line {line_number}."
                )
            else:
                print(
                    f"{dataset} dataset is too short, enabling overlap. See details in {file_name} at line {line_number}."
                )
        # 按模式返回对应数据段
        if self.mode == "train":
            offset = self.output_len if self.overlap else 0
            return data[: train_len + offset].copy()
        elif self.mode == "valid":
            offset_left = self.input_len - 1 if self.overlap else 0
            offset_right = self.output_len if self.overlap else 0
            return data[
                train_len - offset_left : train_len + valid_len + offset_right
            ].copy()
        else:  # self.mode == 'test'
            offset = self.input_len - 1 if self.overlap else 0
            return data[train_len + valid_len - offset :].copy()

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a sample from the dataset at the specified index, considering both the input and output lengths.

        Args:
            index (int): The index of the desired sample in the dataset.

        Returns:
            dict: A dictionary containing 'inputs' and 'target', where both are slices of the dataset corresponding to
                  the historical input data and future prediction data, respectively.
        根据索引获取单条样本：
        history_data = 历史输入
        future_data  = 预测目标
        """
        history_data = self.data[index : index + self.input_len]
        future_data = self.data[
            index + self.input_len : index + self.input_len + self.output_len
        ]
        return {"inputs": history_data, "target": future_data}

    def __len__(self) -> int:
        """
        计算总样本数量：
        样本数 = 数据长度 - 输入长度 - 输出长度 + 1
        这是时序预测标准公式
        Calculates the total number of samples available in the dataset, adjusted for the lengths of input and output sequences.

        Returns:
            int: The number of valid samples that can be drawn from the dataset, based on the configurations of input and output lengths.
        """
        return len(self.data) - self.input_len - self.output_len + 1


import torch


class MyTimeSeries(BaseDataset):
    """
    自定义时序数据集（适配你的项目）
    从 .npz / .npy 读取预划分好的训练/验证/测试时间索引
    直接按索引截取数据，不自动划分
    """
    def __init__(
        self,
        dataset_name: str,
        train_val_test_ratio: List[float],
        mode: str,
        input_len: int,
        output_len: int,
        overlap: bool = False,
        logger: logging.Logger = None,
    ) -> None:
        assert mode in [
            "train",
            "valid",
            "test",
        ], f"Invalid mode: {mode}. Must be one of ['train', 'valid', 'test']."
        super().__init__(
            dataset_name, train_val_test_ratio, mode, input_len, output_len, overlap
        )
        self.logger = logger

        self.data_file_path = f"datasets/{dataset_name}"
        self.description = self._load_description()
        self.data = self._load_data()

    def _load_description(self) -> dict:
        # 空实现，无需描述文件
        pass

    def _load_data(self) -> np.ndarray:
        """
        从预先生成的 .npy 索引文件中读取训练/验证/测试的时间区间
        直接截取对应时间段的数据
        """
        sample_path = "/" + str(self.input_len) + "_" + str(self.output_len)
        # data = np.load("main-master/"+self.data_file_path + "/his.npz")  #/home/user/Downloads/cai/FaST-main-24_8/FaST-main/main-master/datasets/{dataset_name}
        data = np.load("/home/user/Downloads/cai/FaST-main-24_8/FaST-main/main-master/datasets/HNGS_LC/his.npz")
        
        Traffic = data["data"]
        data = Traffic
        
        # print(self.data_file_path + sample_path + "/idx_train.npy")
        # 加载预划分的索引
        # 可视化过程
        train_idx = np.load('main-master/'+self.data_file_path + sample_path + "/idx_train.npy")
        val_idx = np.load('main-master/'+self.data_file_path + sample_path + "/idx_val.npy")
        test_idx = np.load('main-master/'+self.data_file_path + sample_path + "/idx_test.npy")
        # 训练过程
        # train_idx = np.load(self.data_file_path + sample_path + "/idx_train.npy")
        # val_idx = np.load(self.data_file_path + sample_path + "/idx_val.npy")
        # test_idx = np.load(self.data_file_path + sample_path + "/idx_test.npy")
        # 根据 mode 截取对应数据段
        if self.mode == "train":
            return data[
                train_idx[0] - self.output_len - self.input_len + 1 : train_idx[-1] + 1
            ].copy()
        elif self.mode == "valid":
            return data[val_idx[0] - self.output_len - self.input_len + 1 : val_idx[-1] + 1].copy()
        else:
            return data[test_idx[0] - self.output_len - self.input_len + 1 :].copy()

    def __getitem__(self, index: int) -> dict:
        """获取单条样本，并转为 torch.Tensor"""
        history_data = self.data[index : index + self.input_len].astype(np.float32)
        future_data = self.data[
            index + self.input_len : index + self.input_len + self.output_len
        ].astype(np.float32)

        history_data = torch.from_numpy(history_data)
        future_data = torch.from_numpy(future_data)

        return {"inputs": history_data, "target": future_data}

    def __len__(self) -> int:
        """标准时序样本数计算"""
        return len(self.data) - self.input_len - self.output_len + 1


class MyTimeSeries2(BaseDataset):
    """
    高级自定义数据集：支持【分组节点】+【补零对齐】
    适用于多节点时空预测（如多传感器、多路段交通预测）
    可以把节点分成组，自动补齐到固定数量
    """
    def __init__(
        self,
        dataset_name: str,
        train_val_test_ratio: List[float],
        mode: str,
        input_len: int,
        output_len: int,
        group_size: int | None = None,     # ← ① 可选，默认 None 表示“全部节点”
        overlap: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(dataset_name, train_val_test_ratio,
                         mode, input_len, output_len, overlap)
        self.logger = logger
        self.data_file_path = f"datasets/{dataset_name}"

        self.description = self._load_description()
        raw = self._load_data()            # raw.shape == (T, N, D)

        # ② 若用户未指定 group_size，则用 N 全部节点
        if group_size is None or group_size <= 0:
            group_size = raw.shape[1]       # N
        self.group_size = group_size

        N = raw.shape[1]                       # 原始节点数
        S = (self.group_size - N % self.group_size) % self.group_size

        if S:                                   # 需要补节点
            # ① 生成要复制的节点索引，例如 N=5, S=8 -> [0,1,2,3,4,0,1,2]
            pad_idx = np.arange(S) % N
            # ② 取出对应节点形成补丁，并拼接在节点维度
            pad = raw[:, pad_idx, :]            # (T, S, D)
            raw = np.concatenate([raw, pad], axis=1)  # (T, N+S, D)

        self.data = raw
        self.num_groups   = (N + S) // group_size
        self.time_samples = len(self.data) - input_len - output_len + 1

    # --------------------------------------------------------

    def _load_data(self) -> np.ndarray:
        sample_path = "/" + str(self.input_len) + "_" + str(self.output_len)
        data = np.load(self.data_file_path + "/his.npz")
        Traffic = data["data"]
        data = Traffic

        train_idx = np.load(self.data_file_path + sample_path + "/idx_train.npy")
        val_idx = np.load(self.data_file_path + sample_path + "/idx_val.npy")
        test_idx = np.load(self.data_file_path + sample_path + "/idx_test.npy")

        if self.mode == "train":
            return data[
                train_idx[0] - self.output_len - self.input_len + 1 : train_idx[-1] + 1
            ].copy()
        elif self.mode == "valid":
            return data[
                val_idx[0] - self.output_len - self.input_len + 1 : val_idx[-1] + 1
            ].copy()
        


        
        else:
            return data[test_idx[0] - self.output_len - self.input_len + 1 :].copy()

    def __len__(self) -> int:
        return self.time_samples * self.num_groups
    
    def _load_description(self) -> dict:
        pass

    def __getitem__(self, index: int) -> dict:
        # 将平面 index 拆成 “时间窗索引 + 节点组索引”
        # 输出：该时间窗 + 该组节点 的历史与未来数据
        time_idx = index // self.num_groups
        group_idx = index % self.num_groups

        node_start = group_idx * self.group_size
        node_end = node_start + self.group_size
        t0, t1, t2 = time_idx, time_idx + self.input_len, time_idx + self.input_len + self.output_len

        history = self.data[t0:t1, node_start:node_end].astype(np.float32)
        future  = self.data[t1:t2, node_start:node_end].astype(np.float32)

        return {
            "inputs": torch.from_numpy(history),   # (Input, M, D)
            "target": torch.from_numpy(future),    # (Output, M, D)
            # 可选：mask = 0/1，区分真实节点与 padding 节点
        }
