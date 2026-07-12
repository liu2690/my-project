#!/usr/bin/env python3
"""
test_split_manager.py — 测试 split_manager.py 的 fold 生成和验证
"""

import os
import sys
import json
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import build_dataset, EXCEL_PATH
from split_manager import (
    generate_canonical_outer_folds,
    generate_canonical_inner_folds,
    generate_legacy_holdout,
    validate_outer_folds,
    validate_legacy_holdout,
    SPLITS_DIR,
    OUTER_N_SPLITS,
    INNER_N_SPLITS,
)


@pytest.fixture(scope='module')
def bundle():
    return build_dataset(EXCEL_PATH)


@pytest.fixture(scope='module')
def outer_manifest(bundle):
    return generate_canonical_outer_folds(bundle, force_regenerate=True)


class TestCanonicalOuterFolds:
    """测试 canonical outer folds"""

    def test_manifest_schema_version(self, outer_manifest):
        assert outer_manifest['schema_version'] == '1.0.0'

    def test_manifest_protocol_name(self, outer_manifest):
        assert outer_manifest['protocol_name'] == 'canonical_outer_folds_v1'

    def test_n_splits_5(self, outer_manifest):
        assert outer_manifest['n_splits'] == 5

    def test_folds_count(self, outer_manifest):
        assert len(outer_manifest['folds']) == 5

    def test_all_folds_present(self, outer_manifest):
        fold_ids = [f['fold_id'] for f in outer_manifest['folds']]
        assert fold_ids == [0, 1, 2, 3, 4]

    def test_train_test_sizes(self, outer_manifest):
        """每个 fold 的 train/test 大小合理"""
        for fold in outer_manifest['folds']:
            assert fold['n_train'] + fold['n_test'] == 159
            assert fold['n_test'] in [31, 32]  # 5-fold, 159/5 ≈ 31.8

    def test_train_test_no_overlap(self, bundle, outer_manifest):
        """每个 fold 内 train/test 无交集"""
        for fold in outer_manifest['folds']:
            train_set = set(fold['train_sample_ids'])
            test_set = set(fold['test_sample_ids'])
            assert train_set.isdisjoint(test_set)

    def test_test_folds_cover_all_samples(self, bundle, outer_manifest):
        """5 个 test fold 合并覆盖全部样本"""
        all_test_ids = set()
        for fold in outer_manifest['folds']:
            all_test_ids.update(fold['test_sample_ids'])
        all_sample_ids = set(bundle.sample_ids.tolist())
        assert all_test_ids == all_sample_ids

    def test_test_folds_mutually_exclusive(self, outer_manifest):
        """不同 fold 的 test 集互不重叠"""
        folds = outer_manifest['folds']
        for i in range(len(folds)):
            for j in range(i + 1, len(folds)):
                set_i = set(folds[i]['test_sample_ids'])
                set_j = set(folds[j]['test_sample_ids'])
                assert set_i.isdisjoint(set_j)

    def test_stratification_binary(self, outer_manifest):
        """每个 fold 的 test 集包含两类样本 (binary)"""
        for fold in outer_manifest['folds']:
            tc = fold['test_counts']
            assert tc['y_binary']['0'] > 0
            assert tc['y_binary']['1'] > 0

    def test_stratification_original(self, outer_manifest):
        """每个 fold 的 test 集包含原始三类样本"""
        for fold in outer_manifest['folds']:
            tc = fold['test_counts']
            assert tc['y_original']['1'] > 0
            assert tc['y_original']['2'] > 0
            assert tc['y_original']['3'] > 0

    def test_validate_outer_folds_passes(self, bundle, outer_manifest):
        assert validate_outer_folds(outer_manifest, bundle) is True

    def test_manifest_hash_present(self, outer_manifest):
        assert 'manifest_hash' in outer_manifest
        assert len(outer_manifest['manifest_hash']) == 64

    def test_dataset_fingerprint_matches(self, bundle, outer_manifest):
        assert outer_manifest['dataset_fingerprint'] == bundle.dataset_fingerprint

    def test_manifest_file_exists(self):
        assert os.path.exists(
            os.path.join(SPLITS_DIR, 'canonical_outer_folds.json')
        )


class TestCanonicalInnerFolds:
    """测试 canonical inner folds"""

    @pytest.fixture(scope='module')
    def inner_manifests(self, bundle, outer_manifest):
        return generate_canonical_inner_folds(bundle, outer_manifest, force_regenerate=True)

    def test_inner_manifests_count(self, inner_manifests):
        assert len(inner_manifests) == 5

    def test_each_inner_has_4_folds(self, inner_manifests):
        for im in inner_manifests:
            assert im['n_inner_splits'] == 4
            assert len(im['inner_folds']) == 4

    def test_inner_train_val_no_overlap(self, inner_manifests):
        for im in inner_manifests:
            for inner_fold in im['inner_folds']:
                train_set = set(inner_fold['train_sample_ids'])
                val_set = set(inner_fold['val_sample_ids'])
                assert train_set.isdisjoint(val_set)

    def test_inner_val_covers_outer_train(self, inner_manifests):
        """4 个 inner val fold 合并覆盖 outer train 全部样本"""
        for im in inner_manifests:
            all_val_ids = set()
            for inner_fold in im['inner_folds']:
                all_val_ids.update(inner_fold['val_sample_ids'])
            outer_train_ids = set(im['outer_train_sample_ids'])
            assert all_val_ids == outer_train_ids

    def test_inner_val_mutually_exclusive(self, inner_manifests):
        for im in inner_manifests:
            inner_folds = im['inner_folds']
            for i in range(len(inner_folds)):
                for j in range(i + 1, len(inner_folds)):
                    set_i = set(inner_folds[i]['val_sample_ids'])
                    set_j = set(inner_folds[j]['val_sample_ids'])
                    assert set_i.isdisjoint(set_j)

    def test_inner_random_state_varies(self, inner_manifests):
        """每个 outer fold 的 inner random_state 不同"""
        states = [im['random_state'] for im in inner_manifests]
        assert len(states) == len(set(states))

    def test_inner_manifest_files_exist(self):
        for fold_id in range(5):
            path = os.path.join(SPLITS_DIR, f'outer_{fold_id}_inner_folds.json')
            assert os.path.exists(path), f"Missing: {path}"


class TestLegacyHoldout:
    """测试 legacy holdout"""

    @pytest.fixture(scope='module')
    def legacy_manifest(self, bundle):
        return generate_legacy_holdout(bundle, force_regenerate=True)

    def test_status_is_reconstructed(self, legacy_manifest):
        assert legacy_manifest['status'] == 'legacy_v1_reconstructed_unverified'

    def test_train_test_no_overlap(self, legacy_manifest):
        train_set = set(legacy_manifest['train_sample_ids'])
        test_set = set(legacy_manifest['test_sample_ids'])
        assert train_set.isdisjoint(test_set)

    def test_covers_all_samples(self, bundle, legacy_manifest):
        train_set = set(legacy_manifest['train_sample_ids'])
        test_set = set(legacy_manifest['test_sample_ids'])
        all_sample_ids = set(bundle.sample_ids.tolist())
        assert train_set | test_set == all_sample_ids

    def test_approx_80_20_split(self, legacy_manifest):
        """大致 8:2 划分"""
        ratio = legacy_manifest['n_train'] / legacy_manifest['n_total']
        assert 0.78 <= ratio <= 0.82  # 允许一些浮动

    def test_test_has_both_classes(self, legacy_manifest):
        tc = legacy_manifest['test_counts']
        assert tc['y_binary']['0'] > 0
        assert tc['y_binary']['1'] > 0

    def test_test_has_original_classes(self, legacy_manifest):
        tc = legacy_manifest['test_counts']
        assert tc['y_original']['1'] > 0
        assert tc['y_original']['2'] > 0
        assert tc['y_original']['3'] > 0

    def test_validate_legacy_passes(self, bundle, legacy_manifest):
        assert validate_legacy_holdout(legacy_manifest, bundle) is True

    def test_manifest_file_exists(self):
        assert os.path.exists(
            os.path.join(SPLITS_DIR, 'legacy_holdout_manifest.json')
        )


class TestFingerprintMismatchDetection:
    """测试 fingerprint 不匹配检测"""

    def test_outer_folds_rejects_mismatch(self, bundle, monkeypatch):
        """当 fingerprint 不匹配时，应抛出 ValueError"""
        # 修改已保存的 manifest 中的 fingerprint
        manifest_path = os.path.join(SPLITS_DIR, 'canonical_outer_folds.json')
        with open(manifest_path, 'r') as f:
            original = json.load(f)

        # 临时写入错误 fingerprint
        corrupted = dict(original)
        corrupted['dataset_fingerprint'] = '0' * 64
        with open(manifest_path, 'w') as f:
            json.dump(corrupted, f)

        try:
            with pytest.raises(ValueError, match='fingerprint'):
                generate_canonical_outer_folds(bundle)
        finally:
            # 恢复
            with open(manifest_path, 'w') as f:
                json.dump(original, f)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-q'])