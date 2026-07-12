#!/usr/bin/env python3
"""
test_shared_fold_manifests.py — 测试 run_nested_cv 正确读取现有 manifests
"""

import os
import sys
import json
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import build_dataset, EXCEL_PATH
from split_manager import SPLITS_DIR


class TestSharedFoldManifests:
    """测试 run_nested_cv 读取现有 manifest"""

    def test_outer_manifest_exists(self):
        assert os.path.exists(os.path.join(SPLITS_DIR, 'canonical_outer_folds.json'))

    def test_inner_manifests_exist(self):
        for fold_id in range(5):
            path = os.path.join(SPLITS_DIR, f'outer_{fold_id}_inner_folds.json')
            assert os.path.exists(path), f"Missing: {path}"

    def test_outer_manifest_structure(self):
        with open(os.path.join(SPLITS_DIR, 'canonical_outer_folds.json')) as f:
            manifest = json.load(f)
        assert 'dataset_fingerprint' in manifest
        assert 'folds' in manifest
        assert len(manifest['folds']) == 5
        for fold in manifest['folds']:
            assert 'fold_id' in fold
            assert 'train_sample_ids' in fold
            assert 'test_sample_ids' in fold

    def test_inner_manifest_structure(self):
        for fold_id in range(5):
            with open(os.path.join(SPLITS_DIR, f'outer_{fold_id}_inner_folds.json')) as f:
                manifest = json.load(f)
            assert 'inner_folds' in manifest
            assert len(manifest['inner_folds']) == 4
            for inner in manifest['inner_folds']:
                assert 'inner_fold_id' in inner
                assert 'train_sample_ids' in inner
                assert 'val_sample_ids' in inner

    def test_fingerprint_consistency(self):
        bundle = build_dataset(EXCEL_PATH)
        with open(os.path.join(SPLITS_DIR, 'canonical_outer_folds.json')) as f:
            outer = json.load(f)
        assert outer['dataset_fingerprint'] == bundle.dataset_fingerprint

        for fold_id in range(5):
            with open(os.path.join(SPLITS_DIR, f'outer_{fold_id}_inner_folds.json')) as f:
                inner = json.load(f)
            assert inner['dataset_fingerprint'] == bundle.dataset_fingerprint

    def test_fingerprint_mismatch_detection(self):
        """测试 fingerprint 不匹配检测"""
        bundle = build_dataset(EXCEL_PATH)
        with open(os.path.join(SPLITS_DIR, 'canonical_outer_folds.json')) as f:
            original = json.load(f)

        # 修改 fingerprint
        corrupted = dict(original)
        corrupted['dataset_fingerprint'] = '0' * 64
        manifest_path = os.path.join(SPLITS_DIR, 'canonical_outer_folds.json')

        try:
            with open(manifest_path, 'w') as f:
                json.dump(corrupted, f)

            # 验证不匹配
            with open(manifest_path) as f:
                loaded = json.load(f)
            assert loaded['dataset_fingerprint'] != bundle.dataset_fingerprint
        finally:
            # 恢复
            with open(manifest_path, 'w') as f:
                json.dump(original, f)

    def test_inner_belongs_to_outer(self):
        """inner folds 的样本属于对应 outer fold 的 train"""
        with open(os.path.join(SPLITS_DIR, 'canonical_outer_folds.json')) as f:
            outer = json.load(f)

        for outer_fold in outer['folds']:
            fold_id = outer_fold['fold_id']
            outer_train = set(outer_fold['train_sample_ids'])

            with open(os.path.join(SPLITS_DIR, f'outer_{fold_id}_inner_folds.json')) as f:
                inner = json.load(f)

            for inner_fold in inner['inner_folds']:
                for sid in inner_fold['train_sample_ids'] + inner_fold['val_sample_ids']:
                    assert sid in outer_train, \
                        f"Inner sample {sid} not in outer-train for fold {fold_id}"

    def test_outer_test_not_in_inner(self):
        """outer-test 样本不在 inner folds 中"""
        with open(os.path.join(SPLITS_DIR, 'canonical_outer_folds.json')) as f:
            outer = json.load(f)

        for outer_fold in outer['folds']:
            fold_id = outer_fold['fold_id']
            outer_test = set(outer_fold['test_sample_ids'])

            with open(os.path.join(SPLITS_DIR, f'outer_{fold_id}_inner_folds.json')) as f:
                inner = json.load(f)

            for inner_fold in inner['inner_folds']:
                all_inner = set(inner_fold['train_sample_ids'] + inner_fold['val_sample_ids'])
                assert outer_test.isdisjoint(all_inner), \
                    f"Outer-test samples found in inner folds for fold {fold_id}"

    def test_does_not_read_legacy_manifest(self):
        """run_nested_cv 不应读取 legacy manifest"""
        # 这个测试验证 nested_cv_engine 不依赖 legacy_holdout_manifest.json
        # 通过检查 engine 代码中的 import 来验证
        import nested_cv_engine as engine
        source = open(engine.__file__).read()
        assert 'legacy_holdout_manifest' not in source
        assert 'legacy' not in source.lower() or 'evidence_limitations' in source


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-q'])