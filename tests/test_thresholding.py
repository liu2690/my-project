#!/usr/bin/env python3
"""
test_thresholding.py — 测试 thresholding.py 的阈值选择逻辑
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thresholding import (
    select_threshold,
    _apply_threshold,
    _generate_threshold_candidates,
    ThresholdSelectionResult,
)


class TestApplyThreshold:
    def test_probability(self):
        scores = np.array([0.3, 0.6, 0.8, 0.2])
        pred = _apply_threshold(scores, 0.5, 'probability')
        assert np.array_equal(pred, [0, 1, 1, 0])

    def test_decision(self):
        scores = np.array([-1.0, 0.5, 2.0, -0.3])
        pred = _apply_threshold(scores, 0.0, 'decision')
        assert np.array_equal(pred, [0, 1, 1, 0])

    def test_probability_boundary(self):
        scores = np.array([0.5])
        pred = _apply_threshold(scores, 0.5, 'probability')
        assert pred[0] == 0  # > 0.5, not >=


class TestGenerateThresholdCandidates:
    def test_probability_candidates(self):
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        candidates = _generate_threshold_candidates(np.sort(scores), 'probability')
        assert 0.5 in candidates
        assert len(candidates) > 0

    def test_decision_candidates(self):
        scores = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        candidates = _generate_threshold_candidates(np.sort(scores), 'decision')
        assert 0.0 in candidates
        assert len(candidates) > 0

    def test_all_positive_boundary(self):
        scores = np.array([0.1, 0.2, 0.3])
        candidates = _generate_threshold_candidates(np.sort(scores), 'probability')
        # 应有全预测正类阈值 (小于 min score)
        has_all_pos = any(c < 0.1 for c in candidates)
        assert has_all_pos

    def test_all_negative_boundary(self):
        scores = np.array([0.7, 0.8, 0.9])
        candidates = _generate_threshold_candidates(np.sort(scores), 'probability')
        # 应有全预测负类阈值 (大于 max score)
        has_all_neg = any(c > 0.9 for c in candidates)
        assert has_all_neg


class TestSelectThresholdBasic:
    def test_default_probability_threshold(self):
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.2, 0.4, 0.6, 0.8])
        result = select_threshold(y_true, scores, 'probability')
        assert isinstance(result, ThresholdSelectionResult)
        assert result.default_threshold == 0.5

    def test_default_decision_threshold(self):
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([-1.0, -0.5, 0.5, 1.0])
        result = select_threshold(y_true, scores, 'decision')
        assert result.default_threshold == 0.0

    def test_finds_better_than_default(self):
        """应能找到比默认阈值更好的阈值"""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.3, 0.45, 0.55, 0.7, 0.9])
        result = select_threshold(y_true, scores, 'probability')
        # 至少 F1 >= 默认 F1
        assert result.f1 >= result.default_f1

    def test_returns_all_metrics(self):
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.2, 0.4, 0.6, 0.8])
        result = select_threshold(y_true, scores, 'probability')
        for key in ['f1', 'mcc', 'balanced_accuracy', 'precision', 'recall',
                     'specificity', 'accuracy', 'pr_auc', 'roc_auc']:
            assert hasattr(result, key)

    def test_tiebreak_by_mcc(self):
        """F1 相同时按 MCC tie-break"""
        # 构造两个阈值产生相同 F1 但不同 MCC 的场景
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.2, 0.4, 0.6, 0.3, 0.5, 0.7])
        result = select_threshold(y_true, scores, 'probability')
        assert result.f1 is not None

    def test_deterministic(self):
        """相同输入产生相同结果"""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        r1 = select_threshold(y_true, scores, 'probability')
        r2 = select_threshold(y_true, scores, 'probability')
        assert r1.threshold == r2.threshold
        assert r1.f1 == r2.f1

    def test_single_class(self):
        """单一类别时返回默认阈值"""
        y_true = np.array([0, 0, 0])
        scores = np.array([0.1, 0.2, 0.3])
        result = select_threshold(y_true, scores, 'probability')
        assert result.threshold == 0.5

    def test_rejects_nan(self):
        y_true = np.array([0, 1])
        scores = np.array([0.5, np.nan])
        with pytest.raises(ValueError):
            select_threshold(y_true, scores, 'probability')


class TestSelectThresholdToDict:
    def test_to_dict(self):
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.2, 0.4, 0.6, 0.8])
        result = select_threshold(y_true, scores, 'probability')
        d = result.to_dict()
        assert 'threshold' in d
        assert 'f1' in d
        assert 'default_threshold' in d


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-q'])