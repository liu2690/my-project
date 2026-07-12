#!/usr/bin/env python3
"""
test_data_pipeline.py — 测试 data_pipeline.py 的数据读取和构建接口
"""

import os
import sys
import pytest
import numpy as np
import tempfile

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import (
    load_raw_excel,
    build_dataset,
    normalize_feature_name,
    is_unknown_feature,
    compute_dataset_fingerprint,
    EXCEL_PATH,
)
from dataset_schema import VOCDatasetBundle


class TestNormalizeFeatureName:
    """测试特征名规范化"""

    def test_normalize_strips_whitespace(self):
        assert normalize_feature_name("  foo  ") == "foo"
        assert normalize_feature_name("bar") == "bar"
        assert normalize_feature_name("\t baz \n") == "baz"

    def test_normalize_non_string(self):
        assert normalize_feature_name(123) == "123"
        assert normalize_feature_name(3.14) == "3.14"


class TestIsUnknownFeature:
    """测试 Unknown 特征判断"""

    def test_exact_unknown(self):
        assert is_unknown_feature("Unknown") is True
        assert is_unknown_feature("  Unknown  ") is True

    def test_non_unknown(self):
        assert is_unknown_feature("Unknown compound candidate") is False
        assert is_unknown_feature("unknown") is False  # 大小写
        assert is_unknown_feature("UNKNOWN") is False
        assert is_unknown_feature("") is False


class TestLoadRawExcel:
    """测试原始 Excel 读取"""

    def test_load_returns_correct_keys(self):
        raw = load_raw_excel(EXCEL_PATH)
        expected_keys = {
            'classes', 'data_raw', 'voc_ids', 'voc_names',
            'n_samples', 'n_features_raw', 'sample_ids', 'excel_path',
        }
        assert set(raw.keys()) == expected_keys

    def test_sample_count(self):
        raw = load_raw_excel(EXCEL_PATH)
        assert raw['n_samples'] == 159
        assert raw['data_raw'].shape[0] == 159

    def test_raw_feature_count(self):
        raw = load_raw_excel(EXCEL_PATH)
        assert raw['n_features_raw'] == 1734
        assert raw['data_raw'].shape[1] == 1734

    def test_classes_are_valid(self):
        raw = load_raw_excel(EXCEL_PATH)
        valid_classes = set(np.unique(raw['classes']))
        assert valid_classes == {1, 2, 3}

    def test_sample_ids_format(self):
        raw = load_raw_excel(EXCEL_PATH)
        assert raw['sample_ids'][0] == 'sample_row_0003'
        assert raw['sample_ids'][-1] == 'sample_row_0161'

    def test_no_nan_in_data(self):
        raw = load_raw_excel(EXCEL_PATH)
        assert not np.any(np.isnan(raw['data_raw']))

    def test_data_dtype(self):
        raw = load_raw_excel(EXCEL_PATH)
        assert raw['data_raw'].dtype == np.float64


class TestBuildDataset:
    """测试 build_dataset()"""

    def test_returns_voc_dataset_bundle(self):
        bundle = build_dataset(EXCEL_PATH)
        assert isinstance(bundle, VOCDatasetBundle)

    def test_n_samples_159(self):
        bundle = build_dataset(EXCEL_PATH)
        assert bundle.n_samples == 159

    def test_n_features_988(self):
        bundle = build_dataset(EXCEL_PATH)
        assert bundle.n_features == 988  # 1734 - 746 Unknown

    def test_unknown_features_removed(self):
        bundle = build_dataset(EXCEL_PATH)
        for name in bundle.feature_names:
            assert not is_unknown_feature(name)

    def test_label_mapping(self):
        bundle = build_dataset(EXCEL_PATH)
        # y_original = 1,2 → y_binary = 0
        assert np.all(bundle.y_binary[bundle.y_original == 1] == 0)
        assert np.all(bundle.y_binary[bundle.y_original == 2] == 0)
        assert np.all(bundle.y_binary[bundle.y_original == 3] == 1)

    def test_class_counts(self):
        bundle = build_dataset(EXCEL_PATH)
        bc = bundle.class_counts_binary
        assert bc[0] == 106  # 53 + 53
        assert bc[1] == 53
        oc = bundle.class_counts_original
        assert oc[1] == 53
        assert oc[2] == 53
        assert oc[3] == 53

    def test_fingerprint_is_string(self):
        bundle = build_dataset(EXCEL_PATH)
        assert isinstance(bundle.dataset_fingerprint, str)
        assert len(bundle.dataset_fingerprint) == 64  # SHA-256 hex

    def test_fingerprint_deterministic(self):
        bundle1 = build_dataset(EXCEL_PATH)
        bundle2 = build_dataset(EXCEL_PATH)
        assert bundle1.dataset_fingerprint == bundle2.dataset_fingerprint

    def test_consistency_check(self):
        bundle = build_dataset(EXCEL_PATH)
        assert bundle.check_consistency() is True

    def test_sample_ids_length(self):
        bundle = build_dataset(EXCEL_PATH)
        assert len(bundle.sample_ids) == 159

    def test_metadata_has_sample_id(self):
        bundle = build_dataset(EXCEL_PATH)
        assert 'sample_id' in bundle.metadata.columns

    def test_source_path_exists(self):
        bundle = build_dataset(EXCEL_PATH)
        assert os.path.exists(bundle.source_path)


class TestComputeFingerprint:
    """测试 fingerprint 计算"""

    def test_fingerprint_changes_with_X(self):
        bundle = build_dataset(EXCEL_PATH)
        fp_original = bundle.dataset_fingerprint

        X_modified = bundle.X.copy()
        X_modified[0, 0] += 1.0  # 修改一个值

        fp_modified = compute_dataset_fingerprint(
            X=X_modified,
            y_binary=bundle.y_binary,
            y_original=bundle.y_original,
            sample_ids=bundle.sample_ids,
            feature_names=bundle.feature_names,
            feature_ids=bundle.feature_ids,
            source_path=bundle.source_path,
            label_mapping={1: 0, 2: 0, 3: 1},
            n_unknown_removed=746,
        )
        assert fp_original != fp_modified

    def test_fingerprint_changes_with_label_mapping(self):
        bundle = build_dataset(EXCEL_PATH)
        fp_original = bundle.dataset_fingerprint

        fp_modified = compute_dataset_fingerprint(
            X=bundle.X,
            y_binary=bundle.y_binary,
            y_original=bundle.y_original,
            sample_ids=bundle.sample_ids,
            feature_names=bundle.feature_names,
            feature_ids=bundle.feature_ids,
            source_path=bundle.source_path,
            label_mapping={1: 0, 2: 1, 3: 1},  # different mapping
            n_unknown_removed=746,
        )
        assert fp_original != fp_modified


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-q'])