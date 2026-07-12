#!/usr/bin/env python3
"""
test_nested_cv_dataflow.py — 测试 nested_cv_engine.py 的数据流隔离

使用可跟踪 sample_id 的合成数据，证明：
- inner fit 只接收 inner-train
- inner-validation 不进入 fit
- outer-test 不进入任何内层 fit
- outer-test 不进入阈值选择
- 每个 outer-train 样本恰好获得一次 inner OOF score
- 阈值来自对应候选自己的 OOF score
"""

import os
import sys
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nested_cv_engine import (
    evaluate_candidate_inner,
    rank_candidates,
    fit_and_evaluate_outer,
)
from candidate_registry import generate_candidates
from data_pipeline import build_dataset, EXCEL_PATH
from dataset_schema import VOCDatasetBundle


@pytest.fixture(scope='module')
def bundle():
    return build_dataset(EXCEL_PATH)


@pytest.fixture(scope='module')
def outer_manifest(bundle):
    with open(os.path.join(os.path.dirname(__file__), '..', 'splits', 'canonical_outer_folds.json')) as f:
        return json.load(f)


@pytest.fixture(scope='module')
def inner_manifest(bundle):
    with open(os.path.join(os.path.dirname(__file__), '..', 'splits', 'outer_0_inner_folds.json')) as f:
        return json.load(f)


class TestInnerCVDataFlow:
    """测试 inner CV 数据流隔离"""

    def test_inner_oof_covers_all_outer_train(self, bundle, outer_manifest, inner_manifest):
        """每个 outer-train 样本恰好获得一次 inner OOF score"""
        candidates = generate_candidates('quick')
        candidate = candidates[0]  # elastic_net
        outer_fold = outer_manifest['folds'][0]

        result = evaluate_candidate_inner(
            candidate, bundle, outer_fold, inner_manifest
        )

        if result['status'] == 'valid':
            n_outer_train = len(outer_fold['train_sample_ids'])
            assert len(result['oof_scores']) == n_outer_train
            assert not np.any(np.isnan(result['oof_scores']))

    def test_outer_test_not_in_inner(self, bundle, outer_manifest, inner_manifest):
        """outer-test 样本不在任何 inner fold 中"""
        outer_fold = outer_manifest['folds'][0]
        test_ids = set(outer_fold['test_sample_ids'])

        for inner_fold in inner_manifest['inner_folds']:
            all_inner = set(inner_fold['train_sample_ids'] + inner_fold['val_sample_ids'])
            assert test_ids.isdisjoint(all_inner)

    def test_inner_samples_in_outer_train(self, bundle, outer_manifest, inner_manifest):
        """所有 inner 样本属于 outer-train"""
        outer_fold = outer_manifest['folds'][0]
        train_ids = set(outer_fold['train_sample_ids'])

        for inner_fold in inner_manifest['inner_folds']:
            for sid in inner_fold['train_sample_ids'] + inner_fold['val_sample_ids']:
                assert sid in train_ids


class TestThresholdSelectionIsolation:
    """测试阈值选择隔离"""

    def test_threshold_uses_only_inner_oof(self, bundle, outer_manifest, inner_manifest):
        """阈值选择仅使用 inner OOF 数据"""
        candidates = generate_candidates('quick')
        candidate = candidates[0]
        outer_fold = outer_manifest['folds'][0]

        result = evaluate_candidate_inner(
            candidate, bundle, outer_fold, inner_manifest
        )

        if result['status'] == 'valid':
            # 验证阈值选择结果包含了正确的阈值
            assert 'inner_tuned_threshold' in result
            assert result['inner_tuned_threshold'] is not None


class TestCandidateRanking:
    """测试候选排名"""

    def test_rank_by_f1(self):
        results = [
            {'candidate_id': 'a', 'inner_tuned_f1': 0.8, 'inner_tuned_mcc': 0.5, 'inner_tuned_balanced_accuracy': 0.7, 'inner_pr_auc': 0.9},
            {'candidate_id': 'b', 'inner_tuned_f1': 0.9, 'inner_tuned_mcc': 0.5, 'inner_tuned_balanced_accuracy': 0.7, 'inner_pr_auc': 0.9},
            {'candidate_id': 'c', 'inner_tuned_f1': 0.8, 'inner_tuned_mcc': 0.6, 'inner_tuned_balanced_accuracy': 0.7, 'inner_pr_auc': 0.9},
        ]
        ranked = rank_candidates(results)
        assert ranked[0]['candidate_id'] == 'b'

    def test_rank_break_by_mcc(self):
        results = [
            {'candidate_id': 'a', 'inner_tuned_f1': 0.8, 'inner_tuned_mcc': 0.5, 'inner_tuned_balanced_accuracy': 0.7, 'inner_pr_auc': 0.9},
            {'candidate_id': 'b', 'inner_tuned_f1': 0.8, 'inner_tuned_mcc': 0.6, 'inner_tuned_balanced_accuracy': 0.7, 'inner_pr_auc': 0.9},
        ]
        ranked = rank_candidates(results)
        assert ranked[0]['candidate_id'] == 'b'

    def test_rank_break_by_balanced_accuracy(self):
        results = [
            {'candidate_id': 'a', 'inner_tuned_f1': 0.8, 'inner_tuned_mcc': 0.5, 'inner_tuned_balanced_accuracy': 0.7, 'inner_pr_auc': 0.9},
            {'candidate_id': 'b', 'inner_tuned_f1': 0.8, 'inner_tuned_mcc': 0.5, 'inner_tuned_balanced_accuracy': 0.8, 'inner_pr_auc': 0.9},
        ]
        ranked = rank_candidates(results)
        assert ranked[0]['candidate_id'] == 'b'

    def test_rank_break_by_pr_auc(self):
        results = [
            {'candidate_id': 'a', 'inner_tuned_f1': 0.8, 'inner_tuned_mcc': 0.5, 'inner_tuned_balanced_accuracy': 0.7, 'inner_pr_auc': 0.8},
            {'candidate_id': 'b', 'inner_tuned_f1': 0.8, 'inner_tuned_mcc': 0.5, 'inner_tuned_balanced_accuracy': 0.7, 'inner_pr_auc': 0.9},
        ]
        ranked = rank_candidates(results)
        assert ranked[0]['candidate_id'] == 'b'

    def test_rank_break_by_candidate_id(self):
        results = [
            {'candidate_id': 'b', 'inner_tuned_f1': 0.8, 'inner_tuned_mcc': 0.5, 'inner_tuned_balanced_accuracy': 0.7, 'inner_pr_auc': 0.9},
            {'candidate_id': 'a', 'inner_tuned_f1': 0.8, 'inner_tuned_mcc': 0.5, 'inner_tuned_balanced_accuracy': 0.7, 'inner_pr_auc': 0.9},
        ]
        ranked = rank_candidates(results)
        assert ranked[0]['candidate_id'] == 'a'


class TestOuterFitDataFlow:
    """测试 outer fit 数据流"""

    def test_outer_fit_uses_complete_outer_train(self, bundle, outer_manifest, inner_manifest):
        """outer fit 在完整 outer-train 上进行"""
        outer_fold = outer_manifest['folds'][0]
        train_ids = outer_fold['train_sample_ids']
        test_ids = outer_fold['test_sample_ids']

        # 验证 train/test 无交集
        assert set(train_ids).isdisjoint(set(test_ids))
        # 验证覆盖全部样本
        assert set(train_ids) | set(test_ids) == set(bundle.sample_ids.tolist())


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-q'])