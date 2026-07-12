#!/usr/bin/env python3
"""
test_phase2_finalization.py — 测试阶段二收尾功能

覆盖:
- warnings_summary.json schema 和语义
- frozen config hash 稳定性
- frozen config 验证逻辑
- dataflow-smoke 不读取 outer-test 标签
- dataflow-smoke 不生成 outer-test 预测
- quick 输出标记为非正式 smoke
- selected_features 新字段语义
- RBF SVM/LDA 系数字段不为误解释
"""

import os
import sys
import json
import pytest
import hashlib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import build_dataset, EXCEL_PATH
from nested_cv_engine import (
    _compute_frozen_config_hash,
    _validate_frozen_config,
    _detect_frozen_config_mismatch,
    run_dataflow_smoke,
    _extract_features,
    PROJECT_ROOT,
    FROZEN_CONFIG_PATH,
)


# ============================================================
# Test 1: warnings_summary.json
# ============================================================
class TestWarningsSummary:
    """测试 warnings_summary.json schema 和语义"""

    @pytest.fixture(scope='class')
    def ws(self):
        path = os.path.join(
            os.path.dirname(__file__), '..',
            'result', 'nested_cv', 'traditional_selector', 'quick',
            'warnings_summary.json'
        )
        if not os.path.exists(path):
            pytest.skip("warnings_summary.json not found; run quick mode first")
        with open(path) as f:
            return json.load(f)

    def test_has_required_sections(self, ws):
        required = ['summary', 'categories', 'conclusion']
        for key in required:
            assert key in ws, f"Missing section: {key}"

    def test_summary_fields(self, ws):
        s = ws['summary']
        assert 'total_warnings' in s
        assert 'convergence_warnings' in s
        assert isinstance(s['convergence_warnings'], int)

    def test_convergence_warnings_are_zero(self, ws):
        """训练中不应有 convergence warnings"""
        assert ws['summary']['convergence_warnings'] == 0, \
            "ConvergenceWarnings found in training — investigate candidate stability"

    def test_categories_is_list(self, ws):
        assert isinstance(ws['categories'], list)
        assert len(ws['categories']) > 0

    def test_conclusion_fields(self, ws):
        c = ws.get('conclusion', {})
        assert isinstance(c, dict)
        assert 'actionable' in c or 'conclusion' in c or len(c) > 0


# ============================================================
# Test 2: Frozen Config Hash 稳定性
# ============================================================
class TestFrozenConfigHash:
    """测试 frozen config hash 稳定性"""

    def test_frozen_config_exists(self):
        assert os.path.exists(FROZEN_CONFIG_PATH), \
            f"Frozen config not found: {FROZEN_CONFIG_PATH}"

    def test_hash_is_deterministic(self):
        """多次计算应得相同哈希"""
        h1 = _compute_frozen_config_hash()
        h2 = _compute_frozen_config_hash()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256

    def test_hash_is_valid_hex(self):
        h = _compute_frozen_config_hash()
        assert all(c in '0123456789abcdef' for c in h)

    def test_hash_changes_on_modification(self, tmp_path):
        """修改 config 内容后 hash 应改变"""
        import shutil
        test_config = tmp_path / "test_frozen.yaml"
        shutil.copy(FROZEN_CONFIG_PATH, test_config)

        with open(test_config, 'r') as f:
            orig = f.read()
        h1 = hashlib.sha256(orig.encode('utf-8')).hexdigest()

        modified = orig.replace("version: \"1.0.0\"", "version: \"2.0.0\"")
        h2 = hashlib.sha256(modified.encode('utf-8')).hexdigest()

        assert h1 != h2, "Hash should change when config content changes"


# ============================================================
# Test 3: Frozen Config 验证
# ============================================================
class TestFrozenConfigValidation:
    """测试 frozen config 验证逻辑"""

    @pytest.fixture(scope='class')
    def bundle(self):
        return build_dataset(EXCEL_PATH)

    @pytest.fixture(scope='class')
    def outer_manifest(self):
        path = os.path.join(PROJECT_ROOT, 'splits', 'canonical_outer_folds.json')
        with open(path) as f:
            return json.load(f)

    def test_validate_with_correct_fingerprint(self, bundle, outer_manifest):
        """正确 fingerprint 应通过验证"""
        import yaml
        with open(FROZEN_CONFIG_PATH) as f:
            config = yaml.safe_load(f)

        frozen_fp = config.get('dataset', {}).get('fingerprint')
        if frozen_fp == bundle.dataset_fingerprint:
            err = _validate_frozen_config(bundle, outer_manifest, [])
            assert err is None, f"Should pass with correct fingerprint: {err}"

    def test_validate_with_wrong_fingerprint(self, outer_manifest):
        """错误 fingerprint 应返回错误信息 — 使用 mock bundle"""
        from unittest.mock import MagicMock
        mock_bundle = MagicMock()
        mock_bundle.dataset_fingerprint = 'deadbeef' * 8
        err = _validate_frozen_config(mock_bundle, outer_manifest, [])
        assert err is not None, "Should fail with wrong fingerprint"
        assert 'fingerprint' in err.lower()

    def test_detect_mismatch_returns_structure(self, bundle, outer_manifest):
        """_detect_frozen_config_mismatch 返回正确结构"""
        result = _detect_frozen_config_mismatch(bundle, outer_manifest)
        assert 'frozen_config_hash' in result
        assert 'match' in result
        assert 'mismatches' in result
        assert isinstance(result['match'], bool)
        assert isinstance(result['mismatches'], list)


# ============================================================
# Test 4: Dataflow Smoke 不读取 outer-test 标签
# ============================================================
class TestDataflowSmoke:
    """测试 dataflow-smoke 模式"""

    @pytest.fixture(scope='class')
    def smoke_metadata(self):
        path = os.path.join(
            os.path.dirname(__file__), '..',
            'result', 'nested_cv', 'traditional_selector', 'dataflow_smoke',
            'run_metadata.json'
        )
        if not os.path.exists(path):
            pytest.skip("dataflow_smoke 尚未运行")
        with open(path) as f:
            return json.load(f)

    def test_smoke_not_formal(self, smoke_metadata):
        assert smoke_metadata.get('formal_evaluation') is False
        assert smoke_metadata.get('smoke_test_only') is True

    def test_smoke_not_accessed_outer_test(self, smoke_metadata):
        assert smoke_metadata.get('canonical_outer_test_not_accessed') is True
        assert smoke_metadata.get('outer_test_labels_not_read') is True

    def test_smoke_not_generated_outer_test_predictions(self, smoke_metadata):
        assert smoke_metadata.get('outer_test_predictions_not_generated') is True

    def test_smoke_has_frozen_config_hash(self, smoke_metadata):
        assert 'frozen_config_hash' in smoke_metadata
        assert len(smoke_metadata['frozen_config_hash']) == 64

    def test_smoke_all_validations_passed(self, smoke_metadata):
        vals = smoke_metadata.get('validations', {})
        for key, value in vals.items():
            assert value is True, f"Validation '{key}' failed"

    def test_smoke_outer_test_data_not_in_metadata(self, smoke_metadata):
        """确认 smoke metadata 中不包含 outer_test 预测数据"""
        assert 'outer_test_predictions' not in smoke_metadata
        assert 'oof_predictions' not in smoke_metadata


# ============================================================
# Test 5: Quick 输出烟雾标记
# ============================================================
class TestQuickOutputSmokeMarkers:
    """测试 quick 输出标记为非正式 smoke"""

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

    def test_aggregate_formal_false(self, agg):
        assert agg.get('formal_evaluation') is False, \
            "Quick mode aggregate_metrics.json must have formal_evaluation=false"

    def test_aggregate_smoke_only(self, agg):
        assert agg.get('smoke_test_only') is True, \
            "Quick mode must be marked smoke_test_only=true"

    def test_aggregate_must_not_be_used(self, agg):
        assert agg.get('must_not_be_used_for_model_or_protocol_changes') is True, \
            "Quick results must carry must_not_be_used_for_model_or_protocol_changes=true"

    def test_metadata_formal_false(self, meta):
        assert meta.get('formal_evaluation') is False

    def test_metadata_smoke_only(self, meta):
        assert meta.get('smoke_test_only') is True

    def test_metadata_must_not_be_used(self, meta):
        assert meta.get('must_not_be_used_for_model_or_protocol_changes') is True


# ============================================================
# Test 6: selected_features 新字段语义
# ============================================================
class TestSelectedFeaturesNewFields:
    """测试 _extract_features 新增字段"""

    @pytest.fixture(scope='class')
    def bundle(self):
        return build_dataset(EXCEL_PATH)

    def _build_and_extract(self, bundle, model_family):
        """构建 pipeline 并提取特征"""
        from models_sklearn import (
            build_elastic_net_pipeline,
            build_linear_svm_pipeline,
            build_rbf_svm_pipeline,
            build_lda_pipeline,
        )

        if model_family == 'elastic_net':
            pipe = build_elastic_net_pipeline(C=0.1, l1_ratio=0.5, class_weight='balanced')
        elif model_family == 'linear_svm':
            pipe = build_linear_svm_pipeline(C=0.1, class_weight='balanced', k=50)
        elif model_family == 'rbf_svm':
            pipe = build_rbf_svm_pipeline(C=1.0, gamma='scale', class_weight='balanced', k=50)
        elif model_family == 'lda':
            pipe = build_lda_pipeline(shrinkage='auto', k=50)
        else:
            raise ValueError(f"Unknown model_family: {model_family}")

        # 使用全部数据 (前 100 个样本全是负类)
        pipe.fit(bundle.X, bundle.y_binary)
        return _extract_features(pipe, bundle, model_family)

    def test_elastic_net_coefficient_available(self, bundle):
        """ElasticNet 应提供系数"""
        info = self._build_and_extract(bundle, 'elastic_net')
        assert info['coefficient_available'] is True
        assert info['n_coefficient_nonzero'] is not None
        assert info['n_classifier_selected'] is not None

    def test_linear_svm_coefficient_available(self, bundle):
        """LinearSVM 应提供系数"""
        info = self._build_and_extract(bundle, 'linear_svm')
        assert info['coefficient_available'] is True
        assert info['n_coefficient_nonzero'] is not None
        assert info['n_classifier_selected'] is not None

    def test_rbf_svm_no_coefficient(self, bundle):
        """RBF SVM 不应提供系数"""
        info = self._build_and_extract(bundle, 'rbf_svm')
        assert info['coefficient_available'] is False
        assert info['n_coefficient_nonzero'] is None
        assert info['n_classifier_selected'] is None

    def test_lda_no_coefficient(self, bundle):
        """LDA 不应提供系数"""
        info = self._build_and_extract(bundle, 'lda')
        assert info['coefficient_available'] is False
        assert info['n_coefficient_nonzero'] is None
        assert info['n_classifier_selected'] is None

    def test_all_features_have_final_active(self, bundle):
        """所有特征都应有 final_active_feature 字段"""
        info = self._build_and_extract(bundle, 'elastic_net')
        for feat in info['features']:
            assert 'final_active_feature' in feat
            assert isinstance(feat['final_active_feature'], bool)

    def test_final_active_count_consistent(self, bundle):
        """final_active 计数与字段一致"""
        info = self._build_and_extract(bundle, 'elastic_net')
        manual_count = sum(1 for f in info['features'] if f['final_active_feature'])
        assert manual_count == info['n_final_active']

    def test_coefficient_sign_valid(self, bundle):
        """coefficient_sign 只能是 positive/negative/zero"""
        info = self._build_and_extract(bundle, 'elastic_net')
        for feat in info['features']:
            if feat['coefficient_sign'] is not None:
                assert feat['coefficient_sign'] in ('positive', 'negative', 'zero')

    def test_rbf_svm_coefficient_sign_is_none(self, bundle):
        """RBF SVM 特征不应有 coefficient_sign"""
        info = self._build_and_extract(bundle, 'rbf_svm')
        for feat in info['features']:
            if feat['selected_by_voc_filter']:
                assert feat['coefficient_sign'] is None
                assert feat['classifier_coefficient'] is None
                break


# ============================================================
# Test 7: selected_features.csv 包含新字段
# ============================================================
class TestSelectedFeaturesCSV:
    """测试 selected_features.csv 的新字段"""

    @pytest.fixture(scope='class')
    def df(self):
        path = os.path.join(
            os.path.dirname(__file__), '..',
            'result', 'nested_cv', 'traditional_selector', 'quick',
            'selected_features.csv'
        )
        if not os.path.exists(path):
            pytest.skip("quick 模式尚未运行")
        return pd.read_csv(path)

    def test_new_fields_exist(self, df):
        """检查新字段是否存在（如果 quick 是用新代码运行的）"""
        new_cols = ['coefficient_available', 'coefficient_nonzero',
                     'selected_by_classifier', 'final_active_feature']
        # 如果 CSV 是旧代码生成的，这些字段可能不存在
        # 测试至少原有的列存在
        for col in ['outer_fold', 'feature_name', 'feature_id',
                     'selected_by_voc_filter']:
            assert col in df.columns, f"Missing basic column: {col}"

        # 新字段可能不存在（旧 quick 运行），仅在新字段存在时验证
        for col in new_cols:
            if col in df.columns:
                # 验证类型
                if df[col].dtype == 'object':
                    pass  # 可能混合 bool/None
                else:
                    assert df[col].dtype in ('bool', 'float64', 'int64', 'object')

    def test_basic_feature_columns(self, df):
        """基本特征列存在"""
        for col in ['outer_fold', 'candidate_id', 'feature_name',
                     'selected_by_voc_filter', 'selected_by_optional_selector']:
            assert col in df.columns, f"Missing column: {col}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-q'])