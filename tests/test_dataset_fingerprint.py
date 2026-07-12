#!/usr/bin/env python3
"""
test_dataset_fingerprint.py — 测试 dataset fingerprint 的不可变性和完整性
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import build_dataset, compute_dataset_fingerprint, EXCEL_PATH
from dataset_schema import VOCDatasetBundle


class TestFingerprintImmutability:
    """测试 fingerprint 不可变性"""

    def test_fingerprint_stable_across_calls(self):
        """多次调用 build_dataset 应返回相同 fingerprint"""
        bundle1 = build_dataset(EXCEL_PATH)
        bundle2 = build_dataset(EXCEL_PATH)
        assert bundle1.dataset_fingerprint == bundle2.dataset_fingerprint

    def test_fingerprint_detects_X_change(self):
        """X 改变后 fingerprint 应不同"""
        bundle = build_dataset(EXCEL_PATH)
        X_mod = bundle.X.copy()
        X_mod[10, 50] = X_mod[10, 50] + 1e-6

        fp_mod = compute_dataset_fingerprint(
            X=X_mod, y_binary=bundle.y_binary,
            y_original=bundle.y_original,
            sample_ids=bundle.sample_ids,
            feature_names=bundle.feature_names,
            feature_ids=bundle.feature_ids,
            source_path=bundle.source_path,
            label_mapping={1: 0, 2: 0, 3: 1},
            n_unknown_removed=746,
        )
        assert bundle.dataset_fingerprint != fp_mod

    def test_fingerprint_detects_y_binary_change(self):
        """y_binary 改变后 fingerprint 应不同"""
        bundle = build_dataset(EXCEL_PATH)
        y_mod = bundle.y_binary.copy()
        y_mod[0] = 1 - y_mod[0]

        fp_mod = compute_dataset_fingerprint(
            X=bundle.X, y_binary=y_mod,
            y_original=bundle.y_original,
            sample_ids=bundle.sample_ids,
            feature_names=bundle.feature_names,
            feature_ids=bundle.feature_ids,
            source_path=bundle.source_path,
            label_mapping={1: 0, 2: 0, 3: 1},
            n_unknown_removed=746,
        )
        assert bundle.dataset_fingerprint != fp_mod

    def test_fingerprint_detects_sample_id_change(self):
        """sample_ids 改变后 fingerprint 应不同"""
        bundle = build_dataset(EXCEL_PATH)
        sids_mod = bundle.sample_ids.copy()
        sids_mod[0] = 'sample_row_9999'

        fp_mod = compute_dataset_fingerprint(
            X=bundle.X, y_binary=bundle.y_binary,
            y_original=bundle.y_original,
            sample_ids=sids_mod,
            feature_names=bundle.feature_names,
            feature_ids=bundle.feature_ids,
            source_path=bundle.source_path,
            label_mapping={1: 0, 2: 0, 3: 1},
            n_unknown_removed=746,
        )
        assert bundle.dataset_fingerprint != fp_mod

    def test_fingerprint_detects_label_mapping_change(self):
        """label_mapping 改变后 fingerprint 应不同"""
        bundle = build_dataset(EXCEL_PATH)
        fp_mod = compute_dataset_fingerprint(
            X=bundle.X, y_binary=bundle.y_binary,
            y_original=bundle.y_original,
            sample_ids=bundle.sample_ids,
            feature_names=bundle.feature_names,
            feature_ids=bundle.feature_ids,
            source_path=bundle.source_path,
            label_mapping={1: 1, 2: 0, 3: 1},  # 不同映射
            n_unknown_removed=746,
        )
        assert bundle.dataset_fingerprint != fp_mod

    def test_fingerprint_detects_n_unknown_removed_change(self):
        """n_unknown_removed 改变后 fingerprint 应不同"""
        bundle = build_dataset(EXCEL_PATH)
        fp_mod = compute_dataset_fingerprint(
            X=bundle.X, y_binary=bundle.y_binary,
            y_original=bundle.y_original,
            sample_ids=bundle.sample_ids,
            feature_names=bundle.feature_names,
            feature_ids=bundle.feature_ids,
            source_path=bundle.source_path,
            label_mapping={1: 0, 2: 0, 3: 1},
            n_unknown_removed=999,  # 不同的数量
        )
        assert bundle.dataset_fingerprint != fp_mod


class TestFingerprintFormat:
    """测试 fingerprint 格式"""

    def test_length_is_64(self):
        bundle = build_dataset(EXCEL_PATH)
        assert len(bundle.dataset_fingerprint) == 64  # SHA-256

    def test_is_hex_string(self):
        bundle = build_dataset(EXCEL_PATH)
        assert all(c in '0123456789abcdef' for c in bundle.dataset_fingerprint)

    def test_not_empty(self):
        bundle = build_dataset(EXCEL_PATH)
        assert bundle.dataset_fingerprint != ''
        assert bundle.dataset_fingerprint is not None


class TestFingerprintConsistency:
    """测试 fingerprint 与 manifest 一致性"""

    def test_fingerprint_in_outer_manifest(self):
        import json
        bundle = build_dataset(EXCEL_PATH)
        manifest_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'splits', 'canonical_outer_folds.json'
        )
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        assert manifest['dataset_fingerprint'] == bundle.dataset_fingerprint

    def test_fingerprint_in_legacy_manifest(self):
        import json
        bundle = build_dataset(EXCEL_PATH)
        manifest_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'splits', 'legacy_holdout_manifest.json'
        )
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        assert manifest['dataset_fingerprint'] == bundle.dataset_fingerprint


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-q'])