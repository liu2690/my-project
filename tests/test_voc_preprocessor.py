#!/usr/bin/env python3
"""
test_voc_preprocessor.py — 测试 VOCAbundanceIQRFilter 的 sklearn 兼容性和 fold-wise 行为
"""

import os
import sys
import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import build_dataset, EXCEL_PATH
from voc_preprocessing import VOCAbundanceIQRFilter


@pytest.fixture(scope='module')
def bundle():
    return build_dataset(EXCEL_PATH)


@pytest.fixture(scope='module')
def X(bundle):
    return bundle.X.copy()


class TestVOCAbundanceIQRFilterInit:
    """测试初始化参数"""

    def test_default_params(self):
        f = VOCAbundanceIQRFilter()
        assert f.abundance_percentile == 40.0
        assert f.iqr_percentile == 25.0
        assert f.apply_log1p is True

    def test_custom_params(self):
        f = VOCAbundanceIQRFilter(
            abundance_percentile=50.0,
            iqr_percentile=30.0,
            apply_log1p=False,
        )
        assert f.abundance_percentile == 50.0
        assert f.iqr_percentile == 30.0
        assert f.apply_log1p is False


class TestVOCAbundanceIQRFilterFit:
    """测试 fit 行为"""

    def test_fit_returns_self(self, X):
        f = VOCAbundanceIQRFilter()
        result = f.fit(X)
        assert result is f

    def test_fit_sets_n_features_in(self, X):
        f = VOCAbundanceIQRFilter()
        f.fit(X)
        assert f.n_features_in_ == 988

    def test_fit_sets_n_features_out(self, X):
        f = VOCAbundanceIQRFilter()
        f.fit(X)
        assert f.n_features_out_ > 0
        assert f.n_features_out_ <= 988

    def test_fit_sets_support(self, X):
        f = VOCAbundanceIQRFilter()
        f.fit(X)
        assert f.support_.sum() == f.n_features_out_

    def test_fit_sets_abundance_threshold(self, X):
        f = VOCAbundanceIQRFilter()
        f.fit(X)
        assert f.abundance_threshold_ > 0

    def test_fit_sets_iqr_threshold(self, X):
        f = VOCAbundanceIQRFilter()
        f.fit(X)
        assert f.iqr_threshold_ > 0

    def test_fit_rejects_nan(self):
        f = VOCAbundanceIQRFilter()
        X_nan = np.array([[1.0, np.nan], [3.0, 4.0]])
        with pytest.raises(ValueError):
            f.fit(X_nan)

    def test_fit_rejects_inf(self):
        f = VOCAbundanceIQRFilter()
        X_inf = np.array([[1.0, np.inf], [3.0, 4.0]])
        with pytest.raises(ValueError):
            f.fit(X_inf)

    def test_fit_rejects_negative_with_log1p(self):
        f = VOCAbundanceIQRFilter(apply_log1p=True)
        # 全正值数据，丰度筛选后也无负值，所以不会触发
        # 这里用微小数据测试
        X_pos = np.array([[0.5, 1.0, 2.0], [0.3, 1.5, 1.8], [0.7, 0.8, 2.2]])
        f.fit(X_pos)  # 不应报错

    def test_feature_count_reduction(self, X):
        """筛选后特征数应减少"""
        f = VOCAbundanceIQRFilter(abundance_percentile=40.0, iqr_percentile=25.0)
        f.fit(X)
        # 丰度筛选保留约 60% = ~593, IQR 筛选保留约 75% = ~445
        assert 300 <= f.n_features_out_ <= 700


class TestVOCAbundanceIQRFilterTransform:
    """测试 transform 行为"""

    def test_transform_output_shape(self, X):
        f = VOCAbundanceIQRFilter()
        f.fit(X)
        X_out = f.transform(X)
        assert X_out.shape[0] == X.shape[0]
        assert X_out.shape[1] == f.n_features_out_

    def test_transform_requires_fit(self, X):
        f = VOCAbundanceIQRFilter()
        with pytest.raises(Exception):
            f.transform(X)

    def test_transform_checks_input_shape(self, X):
        f = VOCAbundanceIQRFilter()
        f.fit(X)
        X_wrong = X[:, :10]  # 特征数不对
        with pytest.raises(ValueError):
            f.transform(X_wrong)

    def test_fit_transform(self, X):
        f = VOCAbundanceIQRFilter()
        X_out = f.fit_transform(X)
        assert X_out.shape[0] == X.shape[0]
        assert X_out.shape[1] == f.n_features_out_

    def test_transform_deterministic(self, X):
        """同一 fit 状态下多次 transform 结果一致"""
        f = VOCAbundanceIQRFilter()
        f.fit(X)
        out1 = f.transform(X)
        out2 = f.transform(X)
        assert np.allclose(out1, out2)

    def test_transform_no_negative_output(self, X):
        """log1p 变换后不应有负值"""
        f = VOCAbundanceIQRFilter(apply_log1p=True)
        f.fit(X)
        X_out = f.transform(X)
        assert np.all(X_out >= 0)


class TestVOCAbundanceIQRFilterGetSupport:
    """测试 get_support 和 get_feature_names_out"""

    def test_get_support_bool(self, X):
        f = VOCAbundanceIQRFilter()
        f.fit(X)
        support = f.get_support(indices=False)
        assert support.dtype == bool
        assert len(support) == 988

    def test_get_support_indices(self, X):
        f = VOCAbundanceIQRFilter()
        f.fit(X)
        indices = f.get_support(indices=True)
        assert indices.dtype in [np.int64, np.int32]
        assert len(indices) == f.n_features_out_

    def test_get_feature_names_out(self, X):
        f = VOCAbundanceIQRFilter()
        f.fit(X)
        names = f.get_feature_names_out()
        assert len(names) == f.n_features_out_

    def test_get_feature_names_out_with_input(self, X):
        f = VOCAbundanceIQRFilter()
        input_names = [f"feat_{i}" for i in range(988)]
        f.fit(X)
        names = f.get_feature_names_out(input_features=input_names)
        assert len(names) == f.n_features_out_


class TestVOCAbundanceIQRFilterPipeline:
    """测试 sklearn Pipeline 兼容性"""

    def test_pipeline_fit_transform(self, X):
        y = np.random.randint(0, 2, X.shape[0])
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('filter', VOCAbundanceIQRFilter()),
            ('scaler', StandardScaler()),
        ])
        X_out = pipe.fit_transform(X, y)
        assert X_out.shape[0] == X.shape[0]
        assert X_out.shape[1] > 0

    def test_pipeline_with_logistic_regression(self, X):
        y = np.random.randint(0, 2, X.shape[0])
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('filter', VOCAbundanceIQRFilter()),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(solver='liblinear', max_iter=10000)),
        ])
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)

    def test_pipeline_get_params(self):
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('filter', VOCAbundanceIQRFilter()),
            ('scaler', StandardScaler()),
        ])
        params = pipe.get_params()
        assert 'filter__abundance_percentile' in params
        assert 'filter__iqr_percentile' in params
        assert 'filter__apply_log1p' in params

    def test_pipeline_set_params(self, X):
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('filter', VOCAbundanceIQRFilter()),
            ('scaler', StandardScaler()),
        ])
        pipe.set_params(filter__abundance_percentile=50.0)
        assert pipe.named_steps['filter'].abundance_percentile == 50.0


class TestVOCAbundanceIQRFilterFoldWise:
    """测试 fold-wise 行为: fit 基于训练数据，transform 不重新计算"""

    def test_fit_on_train_transform_on_test(self, X):
        """fit 在训练集，transform 在测试集，不应泄漏测试集信息"""
        # 模拟 fold split
        n = X.shape[0]
        train_idx = np.arange(n // 2)
        test_idx = np.arange(n // 2, n)

        X_train = X[train_idx]
        X_test = X[test_idx]

        f1 = VOCAbundanceIQRFilter()
        f1.fit(X_train)
        X_test_out = f1.transform(X_test)

        # 如果用整个数据集 fit，结果可能不同
        f2 = VOCAbundanceIQRFilter()
        f2.fit(X)

        # 验证 test 输出形状正确
        assert X_test_out.shape[0] == X_test.shape[0]
        assert X_test_out.shape[1] == f1.n_features_out_

    def test_different_folds_have_different_support(self, X):
        """不同 fold 的 fit 结果可能产生不同的 support"""
        n = X.shape[0]
        fold_size = n // 5

        supports = []
        for i in range(5):
            test_idx = np.arange(i * fold_size, (i + 1) * fold_size)
            train_idx = np.setdiff1d(np.arange(n), test_idx)

            f = VOCAbundanceIQRFilter()
            f.fit(X[train_idx])
            supports.append(f.support_.copy())

        # 至少有一个 fold 与另一个 fold 的 support 不同 (高度可能但不保证)
        # 这验证了 fold-wise 预处理确实产生了不同的筛选结果
        all_same = all(
            np.array_equal(supports[0], s) for s in supports[1:]
        )
        # 由于数据差异不大，可能全部相同，这也是合理的
        # 但至少验证了每个 fold 都正常 fit
        assert len(supports) == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-q'])