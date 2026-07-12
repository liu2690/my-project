#!/usr/bin/env python3
"""
dataset_schema.py — 阶段一：统一数据对象定义

定义 VOCDatasetBundle 和辅助类型，确保数据接口一致。
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

# 当前协议版本
SCHEMA_VERSION = "1.0.0"

# 主任务标签映射
LABEL_MAPPING = {1: 0, 2: 0, 3: 1}

# Unknown 删除规则版本
UNKNOWN_REMOVAL_VERSION = "v1_strict_equal"


@dataclass(frozen=True)
class VOCDatasetBundle:
    """不可变 VOC 数据集对象。

    Attributes:
        X: 特征矩阵 (n_samples, n_features)，已删除 Unknown 特征，未做任何数据驱动筛选
        y_binary: 主任务二分类标签 {1:0, 2:0, 3:1}
        y_original: 原始三分类标签 [1, 2, 3]
        sample_ids: 不可变样本 ID，基于原始 Excel 行号
        feature_names: 已知 VOC 特征名称
        feature_ids: VOC ID 编号
        metadata: 元数据 DataFrame
        source_path: 数据源路径
        dataset_fingerprint: SHA-256 fingerprint
    """
    X: np.ndarray
    y_binary: np.ndarray
    y_original: np.ndarray
    sample_ids: np.ndarray
    feature_names: np.ndarray
    feature_ids: Optional[np.ndarray]
    metadata: pd.DataFrame
    source_path: str
    dataset_fingerprint: str

    def __post_init__(self):
        """验证数据一致性"""
        n = self.X.shape[0]
        # 样本数必须一致
        if len(self.y_binary) != n:
            raise ValueError(
                f"y_binary length ({len(self.y_binary)}) != X rows ({n})"
            )
        if len(self.y_original) != n:
            raise ValueError(
                f"y_original length ({len(self.y_original)}) != X rows ({n})"
            )
        if len(self.sample_ids) != n:
            raise ValueError(
                f"sample_ids length ({len(self.sample_ids)}) != X rows ({n})"
            )
        # 特征名数量必须匹配
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError(
                f"feature_names ({len(self.feature_names)}) != X cols ({self.X.shape[1]})"
            )
        # y_original 只能包含 1, 2, 3
        valid_original = set(np.unique(self.y_original))
        if valid_original != {1, 2, 3}:
            raise ValueError(
                f"y_original must contain only {{1,2,3}}, got {valid_original}"
            )
        # y_binary 只能包含 0, 1
        valid_binary = set(np.unique(self.y_binary))
        if not valid_binary.issubset({0, 1}):
            raise ValueError(
                f"y_binary must contain only {{0,1}}, got {valid_binary}"
            )
        # 标签映射一致性
        for orig, expected_bin in LABEL_MAPPING.items():
            mask = self.y_original == orig
            if not np.all(self.y_binary[mask] == expected_bin):
                raise ValueError(
                    f"Label mapping inconsistent: y_original={orig} "
                    f"expected y_binary={expected_bin}"
                )

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def class_counts_binary(self) -> dict:
        """返回二分类计数 {0: n_neg, 1: n_pos}"""
        return {0: int((self.y_binary == 0).sum()),
                1: int((self.y_binary == 1).sum())}

    @property
    def class_counts_original(self) -> dict:
        """返回原始三分类计数 {1: n, 2: n, 3: n}"""
        return {1: int((self.y_original == 1).sum()),
                2: int((self.y_original == 2).sum()),
                3: int((self.y_original == 3).sum())}

    def check_consistency(self) -> bool:
        """全面一致性检查，通过则返回 True，否则 raise"""
        self.__post_init__()
        # 检查行顺序：feature_names 和 X 列对齐
        # 已经通过 __post_init__ 验证
        return True