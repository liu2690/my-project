#!/usr/bin/env python3
"""
test_metrics.py — 测试 metrics.py 的统一指标计算
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import (
    compute_binary_metrics,
    compute_pooled_oof_metrics,
    compute_foldwise_mean_std,
    safe_metric,
    _compute_specificity,
)


class TestSafeMetric:
    def test_returns_value_on_success(self):
        assert safe_metric(lambda: 42) == 42

    def test_returns_none_on_exception(self):
        assert safe_metric(lambda: 1 / 0) is None


class TestSpecificity:
    def test_normal(self):
        assert _compute_specificity(80, 20) == 0.8

    def test_zero_denom(self):
        assert _compute_specificity(0, 0) == 0.0


class TestBinaryMetricsBasic:
    def test_perfect_classification(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        m = compute_binary_metrics(y_true, y_pred)
        assert m['f1'] == 1.0
        assert m['precision'] == 1.0
        assert m['recall'] == 1.0
        assert m['sensitivity'] == 1.0
        assert m['specificity'] == 1.0
        assert m['accuracy'] == 1.0
        assert m['balanced_accuracy'] == 1.0
        assert m['mcc'] == 1.0
        assert m['tn'] == 2
        assert m['fp'] == 0
        assert m['fn'] == 0
        assert m['tp'] == 2

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        m = compute_binary_metrics(y_true, y_pred)
        assert m['f1'] == 0.0
        assert m['tp'] == 0
        assert m['tn'] == 0
        assert m['fp'] == 2
        assert m['fn'] == 2

    def test_all_positive_prediction(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        m = compute_binary_metrics(y_true, y_pred)
        assert m['recall'] == 1.0
        assert m['specificity'] == 0.0
        assert m['predicted_positive_count'] == 4
        assert m['predicted_positive_rate'] == 1.0

    def test_all_negative_prediction(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        m = compute_binary_metrics(y_true, y_pred)
        assert m['recall'] == 0.0
        assert m['specificity'] == 1.0
        assert m['predicted_positive_count'] == 0
        assert m['predicted_positive_rate'] == 0.0

    def test_zero_division_handling(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        m = compute_binary_metrics(y_true, y_pred)
        assert m['f1'] == 0.0
        assert m['precision'] == 0.0
        assert m['recall'] == 0.0

    def test_single_class(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 1, 1])
        m = compute_binary_metrics(y_true, y_pred)
        assert m['tp'] == 3
        assert m['tn'] == 0
        assert m['fp'] == 0
        assert m['fn'] == 0


class TestBinaryMetricsWithScores:
    def test_roc_auc_normal(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        scores = np.array([0.1, 0.4, 0.6, 0.9])
        m = compute_binary_metrics(y_true, y_pred, scores)
        assert m['roc_auc'] is not None
        assert m['pr_auc'] is not None

    def test_roc_auc_single_class(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        scores = np.array([0.1, 0.2, 0.3])
        m = compute_binary_metrics(y_true, y_pred, scores)
        assert m['roc_auc'] is None
        assert m['roc_auc_reason'] is not None

    def test_no_scores(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        m = compute_binary_metrics(y_true, y_pred, scores=None)
        assert m['roc_auc'] is None
        assert m['pr_auc'] is None

    def test_nan_in_scores(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        scores = np.array([0.5, np.nan])
        m = compute_binary_metrics(y_true, y_pred, scores)
        assert m['roc_auc'] is None
        assert m['roc_auc_reason'] == 'NaN_in_scores'


class TestMCC:
    def test_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        m = compute_binary_metrics(y_true, y_pred)
        assert m['mcc'] == 1.0

    def test_random(self):
        np.random.seed(42)
        y_true = np.array([0] * 100 + [1] * 100)
        y_pred = np.random.randint(0, 2, 200)
        m = compute_binary_metrics(y_true, y_pred)
        assert -1.0 <= m['mcc'] <= 1.0


class TestFoldwiseMeanStd:
    def test_single_fold(self):
        m1 = compute_binary_metrics(
            np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])
        )
        mean_d, std_d = compute_foldwise_mean_std([m1])
        assert mean_d['f1'] == 1.0
        assert std_d['f1'] == 0.0

    def test_multiple_folds(self):
        m1 = compute_binary_metrics(
            np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])
        )
        m2 = compute_binary_metrics(
            np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0])
        )
        mean_d, std_d = compute_foldwise_mean_std([m1, m2])
        assert mean_d['f1'] == 0.5
        assert std_d['f1'] > 0

    def test_handles_none_values(self):
        m1 = {'f1': 0.8, 'roc_auc': None, 'precision': None}
        m2 = {'f1': 0.6, 'roc_auc': None, 'precision': 0.5}
        mean_d, std_d = compute_foldwise_mean_std([m1, m2])
        assert mean_d['f1'] == 0.7
        assert mean_d['roc_auc'] is None
        assert mean_d['precision'] == 0.5


class TestPooledOOFMetrics:
    def test_pooled(self):
        y_true_list = [np.array([0, 0]), np.array([1, 1])]
        y_pred_list = [np.array([0, 0]), np.array([1, 1])]
        pooled = compute_pooled_oof_metrics(y_true_list, y_pred_list)
        assert pooled['f1'] == 1.0

    def test_pooled_with_scores(self):
        y_true_list = [np.array([0, 0]), np.array([1, 1])]
        y_pred_list = [np.array([0, 0]), np.array([1, 1])]
        scores_list = [np.array([0.1, 0.2]), np.array([0.8, 0.9])]
        pooled = compute_pooled_oof_metrics(y_true_list, y_pred_list, scores_list)
        assert pooled['roc_auc'] is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-q'])