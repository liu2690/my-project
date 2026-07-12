#!/usr/bin/env python3
"""
test_holdout_dataflow_isolation.py — 测试 legacy holdout 数据流隔离
"""

import os
import sys
import json
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import build_dataset, EXCEL_PATH
from split_manager import (
    generate_legacy_holdout,
    validate_legacy_holdout,
    SPLITS_DIR,
)
from run_legacy_holdout import dry_run, final_evaluation_rejected


@pytest.fixture(scope='module')
def bundle():
    return build_dataset(EXCEL_PATH)


@pytest.fixture(scope='module')
def legacy_manifest(bundle):
    return generate_legacy_holdout(bundle, force_regenerate=True)


class TestLegacyHoldoutDataFlowIsolation:
    """测试 legacy holdout 数据流隔离"""

    def test_train_test_no_overlap(self, bundle, legacy_manifest):
        """Train/test 绝对无交集"""
        train_set = set(legacy_manifest['train_sample_ids'])
        test_set = set(legacy_manifest['test_sample_ids'])
        assert train_set.isdisjoint(test_set)

    def test_covers_all_samples(self, bundle, legacy_manifest):
        """Train ∪ test = 全部样本"""
        train_set = set(legacy_manifest['train_sample_ids'])
        test_set = set(legacy_manifest['test_sample_ids'])
        all_ids = set(bundle.sample_ids.tolist())
        assert train_set | test_set == all_ids

    def test_train_test_partition_is_complete(self, bundle, legacy_manifest):
        """分割是完整的 partition"""
        train_set = set(legacy_manifest['train_sample_ids'])
        test_set = set(legacy_manifest['test_sample_ids'])
        all_ids = set(bundle.sample_ids.tolist())

        assert len(train_set) + len(test_set) == len(all_ids)
        assert train_set.union(test_set) == all_ids

    def test_legacy_manifest_has_status(self, legacy_manifest):
        """Manifest 包含状态标记"""
        assert 'status' in legacy_manifest
        assert legacy_manifest['status'] == 'legacy_v1_reconstructed_unverified'

    def test_legacy_manifest_has_status_evidence(self, legacy_manifest):
        """Manifest 包含状态证据说明"""
        assert 'status_evidence' in legacy_manifest
        assert 'torch.manual_seed(42)' in legacy_manifest['status_evidence']

    def test_legacy_manifest_has_expected_class_counts(self, legacy_manifest):
        """Manifest 包含预期类别计数"""
        assert 'expected_class_counts' in legacy_manifest
        ecc = legacy_manifest['expected_class_counts']
        assert 'test_total' in ecc
        assert 'test_negative' in ecc
        assert 'test_positive' in ecc

    def test_legacy_manifest_fingerprint_matches(self, bundle, legacy_manifest):
        """Legacy manifest fingerprint 与 bundle 一致"""
        assert legacy_manifest['dataset_fingerprint'] == bundle.dataset_fingerprint


class TestDryRun:
    """测试 dry_run() 函数"""

    def test_dry_run_returns_status(self, bundle):
        result = dry_run(bundle)
        assert result['status'] == 'dry_run_completed'

    def test_dry_run_checks_data_loaded(self, bundle):
        result = dry_run(bundle)
        assert result['checks']['data_loaded'] is True
        assert result['checks']['n_samples'] == 159
        assert result['checks']['n_features'] == 988

    def test_dry_run_checks_fingerprint(self, bundle):
        result = dry_run(bundle)
        assert 'fingerprint' in result['checks']
        assert bundle.dataset_fingerprint[:32] in result['checks']['fingerprint']

    def test_dry_run_checks_legacy_status(self, bundle):
        result = dry_run(bundle)
        assert result['checks']['legacy_status'] == 'legacy_v1_reconstructed_unverified'

    def test_dry_run_no_train_test_overlap(self, bundle):
        result = dry_run(bundle)
        assert result['checks']['train_test_no_overlap'] is True

    def test_dry_run_covers_all_samples(self, bundle):
        result = dry_run(bundle)
        assert result['checks']['covers_all_samples'] is True

    def test_dry_run_train_test_counts(self, bundle):
        result = dry_run(bundle)
        assert result['checks']['n_train'] == 128
        assert result['checks']['n_test'] == 31

    def test_dry_run_pipeline_builds(self, bundle):
        result = dry_run(bundle)
        assert result['checks']['pipeline_builds'] is True

    def test_dry_run_splits_dir_exists(self, bundle):
        result = dry_run(bundle)
        assert result['checks']['splits_dir_exists'] is True

    def test_dry_run_no_metrics_computed(self, bundle):
        result = dry_run(bundle)
        assert result['checks']['no_metrics_computed'] is True
        assert result['checks']['no_model_trained'] is True
        assert result['checks']['no_thresholding'] is True
        assert result['checks']['no_test_predictions_generated'] is True


class TestFinalEvaluationRejected:
    """测试正式评价被安全拒绝"""

    def test_final_evaluation_rejected_raises_system_exit(self):
        with pytest.raises(SystemExit) as exc_info:
            final_evaluation_rejected()
        assert exc_info.value.code == 1


class TestDataFlowIsolation:
    """测试数据流隔离：预处理器不会泄漏测试集信息"""

    def test_filter_fit_on_train_only(self, bundle, legacy_manifest):
        """预处理器只应在训练集上 fit"""
        from voc_preprocessing import VOCAbundanceIQRFilter

        train_ids = set(legacy_manifest['train_sample_ids'])
        train_mask = np.array([sid in train_ids for sid in bundle.sample_ids])

        X_train = bundle.X[train_mask]
        X_test = bundle.X[~train_mask]

        # Fit on train only
        f = VOCAbundanceIQRFilter()
        f.fit(X_train)

        # Transform test
        X_test_out = f.transform(X_test)

        assert X_test_out.shape[0] == X_test.shape[0]
        assert X_test_out.shape[1] == f.n_features_out_

    def test_imputer_fit_on_train_only(self, bundle, legacy_manifest):
        """SimpleImputer 只应在训练集上 fit"""
        from sklearn.impute import SimpleImputer

        train_ids = set(legacy_manifest['train_sample_ids'])
        train_mask = np.array([sid in train_ids for sid in bundle.sample_ids])

        X_train = bundle.X[train_mask]
        X_test = bundle.X[~train_mask]

        imputer = SimpleImputer(strategy='median')
        imputer.fit(X_train)
        X_test_imputed = imputer.transform(X_test)

        assert X_test_imputed.shape == X_test.shape

    def test_scaler_fit_on_train_only(self, bundle, legacy_manifest):
        """StandardScaler 只应在训练集上 fit"""
        from sklearn.preprocessing import StandardScaler

        train_ids = set(legacy_manifest['train_sample_ids'])
        train_mask = np.array([sid in train_ids for sid in bundle.sample_ids])

        X_train = bundle.X[train_mask]
        X_test = bundle.X[~train_mask]

        # 先做 imputation
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        scaler = StandardScaler()
        scaler.fit(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        assert X_test_scaled.shape == X_test_imp.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-q'])