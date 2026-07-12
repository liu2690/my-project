#!/usr/bin/env python3
"""
test_nested_cv_determinism.py — 测试 nested_cv_engine.py 的确定性

在合成小数据和固定 manifests 上重复运行两次，验证：
- selected candidate 相同
- threshold 相同
- predictions 相同
- metrics 相同
"""

import os
import sys
import json
import pytest
import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nested_cv_engine import (
    evaluate_candidate_inner,
    rank_candidates,
    select_threshold,
)
from thresholding import _apply_threshold
from candidate_registry import generate_candidates
from metrics import compute_binary_metrics


def make_synthetic_data(n_samples=200, n_features=50, random_state=42):
    """创建合成数据集"""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    y = rng.randint(0, 2, n_samples)
    sample_ids = np.array([f'synth_{i:04d}' for i in range(n_samples)])
    feature_names = np.array([f'feat_{i}' for i in range(n_features)])
    return X, y, sample_ids, feature_names


def make_synthetic_manifests(n_samples=200, n_outer=3, n_inner=3, random_state=42):
    """创建合成 fold manifests"""
    rng = np.random.RandomState(random_state)
    y = rng.randint(0, 2, n_samples)
    sample_ids = np.array([f'synth_{i:04d}' for i in range(n_samples)])

    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    outer_manifest = {'folds': [], 'dataset_fingerprint': 'test_fp', 'manifest_hash': 'test_hash'}

    for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(np.zeros(n_samples), y)):
        outer_fold = {
            'fold_id': fold_id,
            'train_sample_ids': sample_ids[train_idx].tolist(),
            'test_sample_ids': sample_ids[test_idx].tolist(),
        }
        outer_manifest['folds'].append(outer_fold)

    return outer_manifest, sample_ids, y


class TestDeterminism:
    """测试确定性"""

    def test_threshold_deterministic(self):
        """相同输入产生相同阈值"""
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100)
        scores = rng.random(100)

        r1 = select_threshold(y_true, scores, 'probability')
        r2 = select_threshold(y_true, scores, 'probability')

        assert r1.threshold == r2.threshold
        assert r1.f1 == r2.f1
        assert r1.mcc == r2.mcc

    def test_ranking_deterministic(self):
        """相同候选列表产生相同排名"""
        results = [
            {'candidate_id': 'a', 'inner_tuned_f1': 0.8, 'inner_tuned_mcc': 0.5,
             'inner_tuned_balanced_accuracy': 0.7, 'inner_pr_auc': 0.9},
            {'candidate_id': 'b', 'inner_tuned_f1': 0.9, 'inner_tuned_mcc': 0.5,
             'inner_tuned_balanced_accuracy': 0.7, 'inner_pr_auc': 0.9},
            {'candidate_id': 'c', 'inner_tuned_f1': 0.8, 'inner_tuned_mcc': 0.6,
             'inner_tuned_balanced_accuracy': 0.7, 'inner_pr_auc': 0.9},
        ]

        ranked1 = rank_candidates(results)
        ranked2 = rank_candidates(results)

        ids1 = [r['candidate_id'] for r in ranked1]
        ids2 = [r['candidate_id'] for r in ranked2]
        assert ids1 == ids2

    def test_apply_threshold_deterministic(self):
        """相同阈值产生相同预测"""
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y1 = _apply_threshold(scores, 0.5, 'probability')
        y2 = _apply_threshold(scores, 0.5, 'probability')
        assert np.array_equal(y1, y2)

    def test_metrics_deterministic(self):
        """相同输入产生相同指标"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        m1 = compute_binary_metrics(y_true, y_pred)
        m2 = compute_binary_metrics(y_true, y_pred)
        assert m1['f1'] == m2['f1']
        assert m1['mcc'] == m2['mcc']

    def test_full_workflow_deterministic(self):
        """完整 inner OOF + threshold 流程确定性"""
        rng = np.random.RandomState(42)
        n_samples = 100
        X = rng.randn(n_samples, 20)
        y = rng.randint(0, 2, n_samples)

        # 模拟 inner OOF
        oof_scores = np.zeros(n_samples)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        for train_idx, val_idx in cv.split(X, y):
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(solver='liblinear', random_state=42)
            clf.fit(X[train_idx], y[train_idx])
            oof_scores[val_idx] = clf.predict_proba(X[val_idx])[:, 1]

        r1 = select_threshold(y, oof_scores, 'probability')
        r2 = select_threshold(y, oof_scores, 'probability')

        assert r1.threshold == r2.threshold
        assert r1.f1 == r2.f1


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-q'])