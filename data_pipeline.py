#!/usr/bin/env python3
"""
data_pipeline.py — 阶段一：统一数据读取接口

从原始 Excel 读取数据，生成 VOCDatasetBundle。
所有数据驱动步骤 (标准化、丰度筛选、IQR 筛选) 均不在此阶段执行。
"""

import hashlib
import os
from typing import Optional
import numpy as np
import pandas as pd

from dataset_schema import (
    VOCDatasetBundle,
    LABEL_MAPPING,
    UNKNOWN_REMOVAL_VERSION,
    SCHEMA_VERSION,
)

# 默认数据路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(PROJECT_ROOT, 'data', '1、2、3.xlsx')


# ============================================================
# 配置
# ============================================================
# Excel 文件中数据起始行 (0-indexed)
DATA_START_ROW = 2  # Row 0 = VOC ID header, Row 1 = VOC name, Rows 2-160 = data

# 特征名规范化：去除前后空格
def normalize_feature_name(name: str) -> str:
    """规范化特征名称：去除前后空格"""
    if not isinstance(name, str):
        return str(name)
    return name.strip()


def is_unknown_feature(name: str) -> bool:
    """判断特征是否为 Unknown。
    
    规则: 规范化后的名称严格等于 'Unknown'。
    不会误删类似 'Unknown compound candidate' 的名称。
    """
    return normalize_feature_name(name) == 'Unknown'


# ============================================================
# 数据读取
# ============================================================
def load_raw_excel(excel_path: str) -> dict:
    """从原始 Excel 读取数据，返回原始结构化数据。
    
    Args:
        excel_path: Excel 文件路径
    
    Returns:
        dict with keys: classes, data_raw, voc_ids, voc_names, sample_ids_raw
    """
    df = pd.read_excel(excel_path, header=None)

    # 类别标签 (Row 2-160, Col 0)
    classes = df.iloc[DATA_START_ROW:, 0].values.astype(int)

    # 数据矩阵 (Row 2-160, Col 1-1734)
    data_raw = df.iloc[DATA_START_ROW:, 1:].values.astype(np.float64)

    # VOC IDs (Row 0, Col 1-1734)
    voc_ids = np.array(df.iloc[0, 1:].values.tolist())

    # VOC names (Row 1, Col 1-1734)
    voc_names = np.array([normalize_feature_name(str(v)) for v in df.iloc[1, 1:].values])

    n_samples = data_raw.shape[0]
    n_features_raw = data_raw.shape[1]

    # 基于原始 Excel 物理行号创建 sample_id
    # DATA_START_ROW=2 → 第一行数据在 Excel 第 3 行 (0-indexed: row 2)
    sample_ids = np.array([
        f"sample_row_{i + DATA_START_ROW + 1:04d}"
        for i in range(n_samples)
    ])

    return {
        'classes': classes,
        'data_raw': data_raw,
        'voc_ids': voc_ids,
        'voc_names': voc_names,
        'n_samples': n_samples,
        'n_features_raw': n_features_raw,
        'sample_ids': sample_ids,
        'excel_path': excel_path,
    }


def build_dataset(
    excel_path: str,
    label_mapping: Optional[dict] = None,
) -> VOCDatasetBundle:
    """从原始 Excel 构建 VOCDatasetBundle。
    
    处理流程:
    1. 读取原始 Excel
    2. 删除 Unknown 特征 (基于特征名称)
    3. 应用标签映射
    4. 生成 fingerprint
    5. 验证一致性
    
    Args:
        excel_path: Excel 文件路径
        label_mapping: 标签映射 dict，默认 {1:0, 2:0, 3:1}
    
    Returns:
        VOCDatasetBundle
    """
    if label_mapping is None:
        label_mapping = LABEL_MAPPING

    raw = load_raw_excel(excel_path)

    # 删除 Unknown 特征
    unknown_mask = np.array([is_unknown_feature(n) for n in raw['voc_names']])
    known_mask = ~unknown_mask
    n_unknown_removed = int(unknown_mask.sum())

    X = raw['data_raw'][:, known_mask].astype(np.float64)
    feature_names = raw['voc_names'][known_mask]
    feature_ids = raw['voc_ids'][known_mask]

    # 验证
    if X.shape[1] == 0:
        raise ValueError("All features were removed as Unknown")

    # 标签映射
    classes = raw['classes']
    y_original = classes.copy()
    y_binary = np.array([label_mapping[c] for c in classes], dtype=int)

    # 验证映射
    for c in np.unique(classes):
        if c not in label_mapping:
            raise ValueError(f"Class {c} not in label_mapping {label_mapping}")

    # 元数据 DataFrame (当前为空)
    metadata = pd.DataFrame({'sample_id': raw['sample_ids']})

    # 生成 fingerprint
    fingerprint = compute_dataset_fingerprint(
        X=X,
        y_binary=y_binary,
        y_original=y_original,
        sample_ids=raw['sample_ids'],
        feature_names=feature_names,
        feature_ids=feature_ids,
        source_path=excel_path,
        label_mapping=label_mapping,
        n_unknown_removed=n_unknown_removed,
    )

    bundle = VOCDatasetBundle(
        X=X,
        y_binary=y_binary,
        y_original=y_original,
        sample_ids=raw['sample_ids'],
        feature_names=feature_names,
        feature_ids=feature_ids,
        metadata=metadata,
        source_path=os.path.abspath(excel_path),
        dataset_fingerprint=fingerprint,
    )

    return bundle


# ============================================================
# Dataset Fingerprint
# ============================================================
def compute_dataset_fingerprint(
    X: np.ndarray,
    y_binary: np.ndarray,
    y_original: np.ndarray,
    sample_ids: np.ndarray,
    feature_names: np.ndarray,
    feature_ids: Optional[np.ndarray],
    source_path: str,
    label_mapping: dict,
    n_unknown_removed: int,
    fingerprint_version: str = "voc_dataset_fingerprint_v1",
) -> str:
    """计算数据集 fingerprint (SHA-256)。
    
    包含的内容:
    - sample_ids
    - y_original
    - y_binary
    - feature_names
    - feature_ids (若存在)
    - X 的 shape
    - X 的 dtype
    - X 的规范化二进制内容
    - 数据源文件内容哈希
    - 标签映射版本
    - Unknown 删除规则版本
    - fingerprint 版本
    """
    hasher = hashlib.sha256()

    # 版本
    hasher.update(fingerprint_version.encode('utf-8'))
    hasher.update(b'\n')

    # Schema 版本
    hasher.update(SCHEMA_VERSION.encode('utf-8'))
    hasher.update(b'\n')

    # Unknown 删除规则版本
    hasher.update(UNKNOWN_REMOVAL_VERSION.encode('utf-8'))
    hasher.update(b'\n')

    # 已删除 Unknown 数量
    hasher.update(str(n_unknown_removed).encode('utf-8'))
    hasher.update(b'\n')

    # 标签映射
    hasher.update(str(sorted(label_mapping.items())).encode('utf-8'))
    hasher.update(b'\n')

    # sample_ids
    for sid in sample_ids:
        hasher.update(sid.encode('utf-8'))
    hasher.update(b'\n')

    # y_original
    hasher.update(y_original.tobytes())
    hasher.update(b'\n')

    # y_binary
    hasher.update(y_binary.tobytes())
    hasher.update(b'\n')

    # feature_names
    for fn in feature_names:
        hasher.update(fn.encode('utf-8'))
    hasher.update(b'\n')

    # feature_ids
    if feature_ids is not None:
        hasher.update(feature_ids.tobytes())
    hasher.update(b'\n')

    # X shape 和 dtype
    hasher.update(str(X.shape).encode('utf-8'))
    hasher.update(b'\n')
    hasher.update(str(X.dtype).encode('utf-8'))
    hasher.update(b'\n')

    # X 规范化二进制内容 (使用 round 到 8 位小数避免浮点精度问题)
    hasher.update(np.round(X, 8).tobytes())

    # 数据源文件哈希
    if os.path.exists(source_path):
        with open(source_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        hasher.update(file_hash.encode('utf-8'))
    hasher.update(b'\n')

    return hasher.hexdigest()