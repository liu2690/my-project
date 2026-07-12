#!/usr/bin/env python3
"""
voc_preprocessing.py — 阶段一：fold-wise VOC 预处理器

实现 sklearn 兼容的 VOCAbundanceIQRFilter。
所有 fit 行为仅基于训练数据，transform 不重新计算统计量。
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array


class VOCAbundanceIQRFilter(BaseEstimator, TransformerMixin):
    """VOC 丰度 + IQR 筛选器。

    按照 preprocessing_data.ipynb 中的逻辑：
    1. 丰度筛选: 保留训练折平均丰度 > P(abundance_percentile) 的特征
    2. log1p 变换
    3. IQR 筛选: 保留训练折 IQR >= P(iqr_percentile) 的特征

    Parameters:
        abundance_percentile: 丰度百分位阈值 (默认 40.0)
        iqr_percentile: IQR 百分位阈值 (默认 25.0)
        apply_log1p: 是否应用 log1p 变换 (默认 True)
    """

    def __init__(
        self,
        abundance_percentile: float = 40.0,
        iqr_percentile: float = 25.0,
        apply_log1p: bool = True,
    ):
        self.abundance_percentile = abundance_percentile
        self.iqr_percentile = iqr_percentile
        self.apply_log1p = apply_log1p

    def fit(self, X, y=None):
        """根据训练数据计算筛选阈值。

        Args:
            X: 训练特征矩阵 (n_samples, n_features)，未经 log1p 的原始数据
            y: 忽略

        Returns:
            self
        """
        X = check_array(
            X, accept_sparse=False, force_all_finite=True,
            ensure_min_features=1, dtype=np.float64,
        )
        self.n_features_in_ = X.shape[1]

        # 检查 NaN / inf
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("Input contains NaN or inf values")

        # Step 1: 丰度筛选
        mean_abundance = np.mean(X, axis=0)
        self.abundance_threshold_ = float(
            np.percentile(mean_abundance, self.abundance_percentile)
        )
        self.abundance_support_ = mean_abundance > self.abundance_threshold_

        if self.abundance_support_.sum() == 0:
            raise ValueError(
                "Abundance filter removed all features. "
                f"Threshold={self.abundance_threshold_:.4f}, "
                f"max mean={mean_abundance.max():.4f}"
            )

        # Step 2: log1p 变换 (仅对丰度保留的特征)
        X_abundance = X[:, self.abundance_support_]

        if self.apply_log1p:
            # 检查负值
            if np.any(X_abundance < 0):
                raise ValueError(
                    "Negative values detected in abundance-filtered features. "
                    "Cannot apply log1p. Set apply_log1p=False or handle negative values."
                )
            X_log = np.log1p(X_abundance)
        else:
            X_log = X_abundance

        # Step 3: IQR 筛选
        q75 = np.percentile(X_log, 75, axis=0)
        q25 = np.percentile(X_log, 25, axis=0)
        iqr_values = q75 - q25

        self.iqr_threshold_ = float(
            np.percentile(iqr_values, self.iqr_percentile)
        )

        # 在丰度保留的特征中，保留 IQR >= threshold 的
        iqr_support_after_abundance = iqr_values >= self.iqr_threshold_
        self.iqr_support_after_abundance_ = iqr_support_after_abundance

        if iqr_support_after_abundance.sum() == 0:
            raise ValueError(
                "IQR filter removed all remaining features. "
                f"Threshold={self.iqr_threshold_:.4f}, "
                f"max IQR={iqr_values.max():.4f}"
            )

        # 最终 support: 将 IQR support 映射回原始特征空间
        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        abundance_indices = np.where(self.abundance_support_)[0]
        final_indices = abundance_indices[iqr_support_after_abundance]
        self.support_[final_indices] = True

        self.n_features_out_ = int(self.support_.sum())

        # 保存 feature_names_in_ (如果通过 set_feature_names_in 设置)
        if hasattr(self, 'feature_names_in_'):
            self.feature_names_out_ = np.array(self.feature_names_in_)[self.support_]

        return self

    def transform(self, X):
        """应用已拟合的筛选器。

        Args:
            X: 特征矩阵 (n_samples, n_features_in_)

        Returns:
            筛选后的特征矩阵 (n_samples, n_features_out_)
        """
        check_is_fitted(self, ['support_', 'n_features_in_'])

        X = check_array(
            X, accept_sparse=False, force_all_finite=True,
            dtype=np.float64,
        )

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        # 筛选特征
        X_selected = X[:, self.support_]

        if self.apply_log1p:
            # 检查负值
            if np.any(X_selected < 0):
                raise ValueError(
                    "Negative values detected in selected features. "
                    "Cannot apply log1p."
                )
            X_selected = np.log1p(X_selected)

        return X_selected

    def fit_transform(self, X, y=None):
        """Fit + transform"""
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices=False):
        """获取特征 support mask。

        Args:
            indices: 若 True，返回索引列表；否则返回布尔 mask

        Returns:
            np.ndarray of bool or int
        """
        check_is_fitted(self, 'support_')
        if indices:
            return np.where(self.support_)[0]
        return self.support_.copy()

    def get_feature_names_out(self, input_features=None):
        """获取输出特征名称。

        Args:
            input_features: 输入特征名称列表

        Returns:
            np.ndarray of str
        """
        check_is_fitted(self, 'support_')
        if input_features is None:
            if hasattr(self, 'feature_names_in_'):
                input_features = self.feature_names_in_
            else:
                input_features = np.array(
                    [f"x{i}" for i in range(self.n_features_in_)]
                )
        input_features = np.asarray(input_features)
        return input_features[self.support_]

    def _more_tags(self):
        return {
            'allow_nan': False,
            'requires_y': False,
        }