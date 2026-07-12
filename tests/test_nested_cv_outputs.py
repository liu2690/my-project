#!/usr/bin/env python3
"""
test_nested_cv_outputs.py — 测试 nested_cv_engine.py 的输出结构
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import build_dataset, EXCEL_PATH
from dataset_schema import VOCDatasetBundle


class TestOutputSchema:
    """测试输出目录和文件结构 (在 quick 运行后)"""

    @pytest.fixture(scope='class')
    def output_dir(self):
        d = os.path.join(
            os.path.dirname(__file__), '..',
            'result', 'nested_cv', 'traditional_selector', 'quick'
        )
        if not os.path.isdir(d):
            pytest.skip("quick 模式尚未运行")
        return d

    def test_output_dir_exists(self, output_dir):
        assert os.path.isdir(output_dir)

    def test_oof_predictions_exists(self, output_dir):
        assert os.path.exists(os.path.join(output_dir, 'oof_predictions.csv'))

    def test_outer_fold_metrics_exists(self, output_dir):
        assert os.path.exists(os.path.join(output_dir, 'outer_fold_metrics.csv'))

    def test_aggregate_metrics_exists(self, output_dir):
        assert os.path.exists(os.path.join(output_dir, 'aggregate_metrics.json'))

    def test_inner_candidate_results_exists(self, output_dir):
        assert os.path.exists(os.path.join(output_dir, 'inner_candidate_results.csv'))

    def test_selected_configs_exists(self, output_dir):
        assert os.path.exists(os.path.join(output_dir, 'selected_configs.json'))

    def test_thresholds_exists(self, output_dir):
        assert os.path.exists(os.path.join(output_dir, 'thresholds.csv'))

    def test_selected_features_exists(self, output_dir):
        assert os.path.exists(os.path.join(output_dir, 'selected_features.csv'))

    def test_subgroup_errors_exists(self, output_dir):
        assert os.path.exists(os.path.join(output_dir, 'subgroup_errors.csv'))

    def test_run_metadata_exists(self, output_dir):
        assert os.path.exists(os.path.join(output_dir, 'run_metadata.json'))


class TestOofPredictionsSchema:
    """测试 oof_predictions.csv schema"""

    @pytest.fixture(scope='class')
    def df(self):
        path = os.path.join(
            os.path.dirname(__file__), '..',
            'result', 'nested_cv', 'traditional_selector', 'quick',
            'oof_predictions.csv'
        )
        if not os.path.exists(path):
            pytest.skip("quick 模式尚未运行")
        return pd.read_csv(path)

    def test_required_columns(self, df):
        required = [
            'sample_id', 'excel_row', 'y_original', 'y_binary',
            'outer_fold', 'selected_candidate_id', 'selected_model_family',
            'selected_threshold', 'score_type', 'score',
            'prediction_tuned', 'prediction_default', 'default_threshold',
            'dataset_fingerprint', 'outer_manifest_hash',
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_duplicate_sample_ids(self, df):
        # quick mode: outer folds 0,1 → 2*32=64 or 63+32=95
        assert df['sample_id'].nunique() == len(df)

    def test_y_original_preserved(self, df):
        assert set(df['y_original'].unique()).issubset({1, 2, 3})

    def test_tuned_and_default_present(self, df):
        assert 'prediction_tuned' in df.columns
        assert 'prediction_default' in df.columns

    def test_predictions_are_binary(self, df):
        assert set(df['prediction_tuned'].unique()).issubset({0, 1})
        assert set(df['prediction_default'].unique()).issubset({0, 1})


class TestAggregateMetricsSchema:
    """测试 aggregate_metrics.json schema"""

    @pytest.fixture(scope='class')
    def agg(self):
        path = os.path.join(
            os.path.dirname(__file__), '..',
            'result', 'nested_cv', 'traditional_selector', 'quick',
            'aggregate_metrics.json'
        )
        if not os.path.exists(path):
            pytest.skip("quick 模式尚未运行")
        with open(path) as f:
            return json.load(f)

    def test_pooled_oof_metrics(self, agg):
        assert 'pooled_oof_metrics' in agg
        pom = agg['pooled_oof_metrics']
        assert 'tuned' in pom
        assert 'default' in pom

    def test_foldwise_mean_metrics(self, agg):
        assert 'foldwise_mean_metrics' in agg
        assert 'foldwise_std_metrics' in agg

    def test_baseline_metrics(self, agg):
        assert 'baseline_pooled_oof_metrics' in agg

    def test_is_quick_run(self, agg):
        assert agg.get('is_quick_run') is True

    def test_evidence_limitations(self, agg):
        el = agg.get('evidence_limitations', {})
        assert el.get('subject_independence_unverified') is True
        assert el.get('internal_validation_only') is True


class TestSubgroupErrors:
    """测试 subgroup_errors.csv"""

    @pytest.fixture(scope='class')
    def df(self):
        path = os.path.join(
            os.path.dirname(__file__), '..',
            'result', 'nested_cv', 'traditional_selector', 'quick',
            'subgroup_errors.csv'
        )
        if not os.path.exists(path):
            pytest.skip("quick 模式尚未运行")
        return pd.read_csv(path)

    def test_has_all_classes(self, df):
        classes = set(df['original_class'].unique())
        assert classes == {1, 2, 3}

    def test_columns(self, df):
        required = ['outer_fold', 'original_class', 'n',
                     'tuned_pred_pos_count', 'tuned_pred_pos_rate',
                     'default_pred_pos_count', 'default_pred_pos_rate']
        for col in required:
            assert col in df.columns


class TestRunMetadata:
    """测试 run_metadata.json"""

    @pytest.fixture(scope='class')
    def meta(self):
        path = os.path.join(
            os.path.dirname(__file__), '..',
            'result', 'nested_cv', 'traditional_selector', 'quick',
            'run_metadata.json'
        )
        if not os.path.exists(path):
            pytest.skip("quick 模式尚未运行")
        with open(path) as f:
            return json.load(f)

    def test_has_fingerprint(self, meta):
        assert 'dataset_fingerprint' in meta
        assert len(meta['dataset_fingerprint']) == 64

    def test_has_mode(self, meta):
        assert meta['mode'] == 'quick'
        assert meta['is_quick_run'] is True


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-q'])