#!/usr/bin/env python3
"""
split_manager.py — 阶段一：canonical folds 与 legacy holdout 管理

生成并管理不可变的外层/内层交叉验证 fold manifests 和 legacy 8:2 holdout。
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn import __version__ as sklearn_version

from data_pipeline import build_dataset
from data_pipeline import EXCEL_PATH as _EXCEL_PATH
from dataset_schema import VOCDatasetBundle

# ============================================================
# 配置
# ============================================================
SPLITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'splits')
MANIFEST_SCHEMA_VERSION = "1.0.0"

OUTER_N_SPLITS = 5
OUTER_RANDOM_STATE = 42
INNER_N_SPLITS = 4
INNER_RANDOM_STATE_BASE = 43  # + outer_fold_index

LEGACY_SEED = 42
LEGACY_SPLIT_LENGTH = [0.8, 0.2]


# ============================================================
# 工具函数
# ============================================================
def _compute_manifest_hash(data: dict) -> str:
    """计算 manifest 的 SHA-256 哈希 (排除 hash 字段自身和 created_at 时间戳)"""
    data_copy = {k: v for k, v in data.items() if k not in ('manifest_hash', 'created_at')}
    serialized = json.dumps(data_copy, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def _count_labels(y_binary: np.ndarray, y_original: np.ndarray, indices: np.ndarray) -> dict:
    """统计指定索引的标签分布"""
    yb = y_binary[indices]
    yo = y_original[indices]
    return {
        'y_binary': {'0': int((yb == 0).sum()), '1': int((yb == 1).sum())},
        'y_original': {
            '1': int((yo == 1).sum()),
            '2': int((yo == 2).sum()),
            '3': int((yo == 3).sum()),
        },
    }


def _indices_to_sample_ids(sample_ids: np.ndarray, indices: np.ndarray) -> List[str]:
    return sample_ids[indices].tolist()


def _indices_to_excel_rows(indices: np.ndarray) -> List[int]:
    """将数组索引转换为原始 Excel 行号 (1-indexed)"""
    # 原始 Excel 行号 = 索引 + DATA_START_ROW + 1 (因为 DATA_START_ROW=2 是 0-indexed)
    return [int(i) + 3 for i in indices]  # +3 = DATA_START_ROW(2) + 1


# ============================================================
# Canonical Outer Folds
# ============================================================
def generate_canonical_outer_folds(
    bundle: VOCDatasetBundle,
    force_regenerate: bool = False,
) -> dict:
    """生成或验证 canonical outer fold manifest。

    Args:
        bundle: VOCDatasetBundle
        force_regenerate: 是否强制重新生成

    Returns:
        outer manifest dict
    """
    manifest_path = os.path.join(SPLITS_DIR, 'canonical_outer_folds.json')

    if os.path.exists(manifest_path) and not force_regenerate:
        # 验证已有 manifest
        with open(manifest_path, 'r') as f:
            existing = json.load(f)
        if existing.get('dataset_fingerprint') != bundle.dataset_fingerprint:
            raise ValueError(
                f"Dataset fingerprint mismatch. "
                f"Existing: {existing.get('dataset_fingerprint')[:16]}..., "
                f"Current: {bundle.dataset_fingerprint[:16]}... "
                f"Use --force-regenerate to overwrite."
            )
        print(f"[split_manager] 复用已有 canonical outer folds: {manifest_path}")
        return existing

    os.makedirs(SPLITS_DIR, exist_ok=True)

    outer_cv = StratifiedKFold(
        n_splits=OUTER_N_SPLITS,
        shuffle=True,
        random_state=OUTER_RANDOM_STATE,
    )

    n = bundle.n_samples
    fold_indices = np.arange(n)
    folds = []

    for fold_id, (train_idx, test_idx) in enumerate(
        outer_cv.split(np.zeros(n), bundle.y_binary)
    ):
        train_ids = _indices_to_sample_ids(bundle.sample_ids, train_idx)
        test_ids = _indices_to_sample_ids(bundle.sample_ids, test_idx)
        train_excel_rows = _indices_to_excel_rows(train_idx)
        test_excel_rows = _indices_to_excel_rows(test_idx)

        folds.append({
            'fold_id': fold_id,
            'train_sample_ids': train_ids,
            'test_sample_ids': test_ids,
            'train_excel_rows': train_excel_rows,
            'test_excel_rows': test_excel_rows,
            'train_counts': _count_labels(bundle.y_binary, bundle.y_original, train_idx),
            'test_counts': _count_labels(bundle.y_binary, bundle.y_original, test_idx),
            'n_train': int(len(train_idx)),
            'n_test': int(len(test_idx)),
        })

    manifest = {
        'schema_version': MANIFEST_SCHEMA_VERSION,
        'protocol_name': 'canonical_outer_folds_v1',
        'dataset_fingerprint': bundle.dataset_fingerprint,
        'task_name': '1+2_vs_3',
        'label_mapping': {'1': 0, '2': 0, '3': 1},
        'split_algorithm': 'StratifiedKFold',
        'n_splits': OUTER_N_SPLITS,
        'shuffle': True,
        'random_state': OUTER_RANDOM_STATE,
        'stratify_on': 'y_binary',
        'sklearn_version': sklearn_version,
        'group_metadata_available': False,
        'independence_assumption': 'sample-level independence is unverified',
        'evidence_limit': 'subject leakage and batch confounding cannot currently be excluded',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'n_total_samples': n,
        'folds': folds,
    }
    manifest['manifest_hash'] = _compute_manifest_hash(manifest)

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[split_manager] 已生成 canonical outer folds: {manifest_path}")
    return manifest


# ============================================================
# Canonical Inner Folds
# ============================================================
def generate_canonical_inner_folds(
    bundle: VOCDatasetBundle,
    outer_manifest: dict,
    force_regenerate: bool = False,
) -> List[dict]:
    """为每个 outer fold 生成 inner fold manifest。

    Args:
        bundle: VOCDatasetBundle
        outer_manifest: outer manifest dict
        force_regenerate: 是否强制重新生成

    Returns:
        list of inner manifest dicts
    """
    inner_manifests = []
    n = bundle.n_samples

    for outer_fold in outer_manifest['folds']:
        fold_id = outer_fold['fold_id']
        manifest_path = os.path.join(
            SPLITS_DIR, f'outer_{fold_id}_inner_folds.json'
        )

        if os.path.exists(manifest_path) and not force_regenerate:
            with open(manifest_path, 'r') as f:
                existing = json.load(f)
            if existing.get('dataset_fingerprint') != bundle.dataset_fingerprint:
                raise ValueError(
                    f"Dataset fingerprint mismatch for inner fold {fold_id}. "
                    f"Use --force-regenerate."
                )
            print(f"[split_manager] 复用已有 inner folds: {manifest_path}")
            inner_manifests.append(existing)
            continue

        # 获取 outer train 样本的索引
        train_ids_set = set(outer_fold['train_sample_ids'])
        train_mask = np.array([sid in train_ids_set for sid in bundle.sample_ids])
        outer_train_indices = np.where(train_mask)[0]

        X_train = bundle.X[outer_train_indices]
        y_train_binary = bundle.y_binary[outer_train_indices]
        y_train_original = bundle.y_original[outer_train_indices]
        train_sample_ids = bundle.sample_ids[outer_train_indices]

        inner_cv = StratifiedKFold(
            n_splits=INNER_N_SPLITS,
            shuffle=True,
            random_state=INNER_RANDOM_STATE_BASE + fold_id,
        )

        inner_folds = []
        for inner_fold_id, (inner_train_idx, inner_val_idx) in enumerate(
            inner_cv.split(np.zeros(len(outer_train_indices)), y_train_binary)
        ):
            inner_folds.append({
                'inner_fold_id': inner_fold_id,
                'train_sample_ids': _indices_to_sample_ids(train_sample_ids, inner_train_idx),
                'val_sample_ids': _indices_to_sample_ids(train_sample_ids, inner_val_idx),
                'train_excel_rows': _indices_to_excel_rows(outer_train_indices[inner_train_idx]),
                'val_excel_rows': _indices_to_excel_rows(outer_train_indices[inner_val_idx]),
                'train_counts': _count_labels(y_train_binary, y_train_original, inner_train_idx),
                'val_counts': _count_labels(y_train_binary, y_train_original, inner_val_idx),
                'n_train': int(len(inner_train_idx)),
                'n_val': int(len(inner_val_idx)),
            })

        inner_manifest = {
            'schema_version': MANIFEST_SCHEMA_VERSION,
            'protocol_name': f'canonical_inner_folds_v1_outer_{fold_id}',
            'outer_fold_id': fold_id,
            'outer_train_sample_ids': outer_fold['train_sample_ids'],
            'dataset_fingerprint': bundle.dataset_fingerprint,
            'task_name': '1+2_vs_3',
            'inner_algorithm': 'StratifiedKFold',
            'n_inner_splits': INNER_N_SPLITS,
            'shuffle': True,
            'random_state': INNER_RANDOM_STATE_BASE + fold_id,
            'stratify_on': 'y_binary',
            'sklearn_version': sklearn_version,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'n_outer_train': int(len(outer_train_indices)),
            'inner_folds': inner_folds,
        }
        inner_manifest['manifest_hash'] = _compute_manifest_hash(inner_manifest)

        with open(manifest_path, 'w') as f:
            json.dump(inner_manifest, f, indent=2, ensure_ascii=False)

        print(f"[split_manager] 已生成 inner folds (outer_{fold_id}): {manifest_path}")
        inner_manifests.append(inner_manifest)

    return inner_manifests


# ============================================================
# Legacy 8:2 Holdout
# ============================================================
def _attempt_legacy_holdout(bundle: VOCDatasetBundle) -> dict:
    """尝试重建旧的 8:2 holdout 划分。

    基于旧代码逻辑:
    - torch.manual_seed(42)
    - 对每个二分类类别分别调用 random_split(dataset, [0.8, 0.2])
    - 合并各类别的 train 和 test 部分

    Returns:
        legacy manifest dict
    """
    # 设置与旧代码相同的 seed
    torch.manual_seed(LEGACY_SEED)

    X_tensor = torch.from_numpy(bundle.X)
    y_tensor = torch.from_numpy(bundle.y_binary)

    # 对每个二分类类别分别 random_split
    train_indices = []
    test_indices = []

    for class_label in [0, 1]:
        class_mask = y_tensor == class_label
        class_indices = torch.where(class_mask)[0]

        # 旧代码: random_split(TensorDataset(x[y==i], y[y==i]), split_length)
        # 等价于对 indices 做 random_split
        dataset = torch.utils.data.TensorDataset(
            X_tensor[class_mask], y_tensor[class_mask]
        )
        train_ds, test_ds = torch.utils.data.random_split(
            dataset, LEGACY_SPLIT_LENGTH
        )

        # 获取原始索引
        train_class_indices = class_indices[train_ds.indices].numpy()
        test_class_indices = class_indices[test_ds.indices].numpy()

        train_indices.append(train_class_indices)
        test_indices.append(test_class_indices)

    train_indices = np.concatenate(train_indices)
    test_indices = np.concatenate(test_indices)

    train_ids = _indices_to_sample_ids(bundle.sample_ids, train_indices)
    test_ids = _indices_to_sample_ids(bundle.sample_ids, test_indices)
    train_excel_rows = _indices_to_excel_rows(train_indices)
    test_excel_rows = _indices_to_excel_rows(test_indices)

    return {
        'train_sample_ids': train_ids,
        'test_sample_ids': test_ids,
        'train_excel_rows': train_excel_rows,
        'test_excel_rows': test_excel_rows,
        'train_counts': _count_labels(bundle.y_binary, bundle.y_original, train_indices),
        'test_counts': _count_labels(bundle.y_binary, bundle.y_original, test_indices),
        'n_train': int(len(train_indices)),
        'n_test': int(len(test_indices)),
        'train_indices': train_indices.tolist(),
        'test_indices': test_indices.tolist(),
    }


def generate_legacy_holdout(
    bundle: VOCDatasetBundle,
    force_regenerate: bool = False,
) -> dict:
    """生成或验证 legacy 8:2 holdout manifest。

    Args:
        bundle: VOCDatasetBundle
        force_regenerate: 是否强制重新生成

    Returns:
        legacy manifest dict
    """
    manifest_path = os.path.join(SPLITS_DIR, 'legacy_holdout_manifest.json')

    if os.path.exists(manifest_path) and not force_regenerate:
        with open(manifest_path, 'r') as f:
            existing = json.load(f)
        if existing.get('dataset_fingerprint') != bundle.dataset_fingerprint:
            raise ValueError(
                "Dataset fingerprint mismatch for legacy holdout. "
                "Use --force-regenerate."
            )
        print(f"[split_manager] 复用已有 legacy holdout: {manifest_path}")
        return existing

    os.makedirs(SPLITS_DIR, exist_ok=True)

    # 尝试重建
    legacy_data = _attempt_legacy_holdout(bundle)

    # 状态判定: 没有独立证据 (历史 sample IDs 或 checkpoint 索引) 来逐样本验证
    legacy_status = 'legacy_v1_reconstructed_unverified'

    manifest = {
        'schema_version': MANIFEST_SCHEMA_VERSION,
        'protocol_name': 'legacy_holdout_v1',
        'dataset_fingerprint': bundle.dataset_fingerprint,
        'task_name': '1+2_vs_3',
        'label_mapping': {'1': 0, '2': 0, '3': 1},
        'split_algorithm': 'PyTorch random_split per class',
        'split_length': LEGACY_SPLIT_LENGTH,
        'seed': LEGACY_SEED,
        'torch_version': torch.__version__,
        'status': legacy_status,
        'status_evidence': (
            'Reconstructed from old code logic: torch.manual_seed(42), '
            'random_split per binary class with [0.8, 0.2]. '
            'No independent evidence (historical sample IDs, checkpoint indices, '
            'or per-sample predictions) exists to verify exact match. '
            'Only class counts can be compared.'
        ),
        'expected_class_counts': {
            'train_total': '~128 (varies with random_split fractional allocation)',
            'test_total': '~31',
            'test_negative': '~21',
            'test_positive': '~10',
        },
        'created_at': datetime.now(timezone.utc).isoformat(),
        'n_total': bundle.n_samples,
        'n_train': legacy_data['n_train'],
        'n_test': legacy_data['n_test'],
        'train_sample_ids': legacy_data['train_sample_ids'],
        'test_sample_ids': legacy_data['test_sample_ids'],
        'train_excel_rows': legacy_data['train_excel_rows'],
        'test_excel_rows': legacy_data['test_excel_rows'],
        'train_counts': legacy_data['train_counts'],
        'test_counts': legacy_data['test_counts'],
    }
    manifest['manifest_hash'] = _compute_manifest_hash(manifest)

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[split_manager] 已生成 legacy holdout: {manifest_path}")
    print(f"  状态: {legacy_status}")
    print(f"  Train: {legacy_data['n_train']}, Test: {legacy_data['n_test']}")
    return manifest


# ============================================================
# 验证函数
# ============================================================
def validate_outer_folds(outer_manifest: dict, bundle: VOCDatasetBundle) -> bool:
    """验证 outer folds 的正确性"""
    folds = outer_manifest['folds']
    all_test_ids = set()
    all_train_ids = set()

    for fold in folds:
        test_set = set(fold['test_sample_ids'])
        train_set = set(fold['train_sample_ids'])

        # Train/test 无交集
        if test_set & train_set:
            raise ValueError(f"Fold {fold['fold_id']}: train/test overlap")

        # 记录
        all_test_ids.update(test_set)

    # 5 个 outer test folds 合并后覆盖全部样本
    all_sample_ids = set(bundle.sample_ids.tolist())
    if all_test_ids != all_sample_ids:
        missing = all_sample_ids - all_test_ids
        extra = all_test_ids - all_sample_ids
        raise ValueError(
            f"Outer test folds don't cover all samples. "
            f"Missing: {len(missing)}, Extra: {len(extra)}"
        )

    # 检查 test folds 之间互不重叠
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            set_i = set(folds[i]['test_sample_ids'])
            set_j = set(folds[j]['test_sample_ids'])
            if set_i & set_j:
                raise ValueError(f"Test fold {i} and {j} overlap")

    return True


def validate_legacy_holdout(legacy_manifest: dict, bundle: VOCDatasetBundle) -> bool:
    """验证 legacy holdout 的正确性"""
    train_set = set(legacy_manifest['train_sample_ids'])
    test_set = set(legacy_manifest['test_sample_ids'])

    # Train/test 无交集
    if train_set & test_set:
        raise ValueError("Legacy holdout: train/test overlap")

    # 覆盖全部样本
    all_sample_ids = set(bundle.sample_ids.tolist())
    if train_set | test_set != all_sample_ids:
        missing = all_sample_ids - (train_set | test_set)
        extra = (train_set | test_set) - all_sample_ids
        raise ValueError(
            f"Legacy holdout doesn't cover all samples. "
            f"Missing: {len(missing)}, Extra: {len(extra)}"
        )

    return True


# ============================================================
# 主入口
# ============================================================
def generate_all_splits(
    bundle: VOCDatasetBundle,
    force_regenerate: bool = False,
) -> Dict[str, Any]:
    """生成所有 splits 并返回 manifests。

    Args:
        bundle: VOCDatasetBundle
        force_regenerate: 是否强制重新生成

    Returns:
        dict with keys: outer_manifest, inner_manifests, legacy_manifest
    """
    print("\n" + "=" * 60)
    print(" 生成 canonical splits")
    print("=" * 60)

    outer_manifest = generate_canonical_outer_folds(bundle, force_regenerate)
    validate_outer_folds(outer_manifest, bundle)
    print("  [✓] Outer folds validated")

    inner_manifests = generate_canonical_inner_folds(bundle, outer_manifest, force_regenerate)
    for im in inner_manifests:
        print(f"  [✓] Inner folds for outer_{im['outer_fold_id']} validated")

    legacy_manifest = generate_legacy_holdout(bundle, force_regenerate)
    validate_legacy_holdout(legacy_manifest, bundle)
    print("  [✓] Legacy holdout validated")

    return {
        'outer_manifest': outer_manifest,
        'inner_manifests': inner_manifests,
        'legacy_manifest': legacy_manifest,
    }


# ============================================================
# CLI 入口
# ============================================================
if __name__ == '__main__':
    force = '--force-regenerate' in sys.argv

    print("Loading dataset...")
    bundle = build_dataset(_EXCEL_PATH)
    print(f"  Dataset: {bundle.n_samples} samples, {bundle.n_features} features")
    print(f"  Fingerprint: {bundle.dataset_fingerprint[:32]}...")

    manifests = generate_all_splits(bundle, force_regenerate=force)

    # 打印摘要
    om = manifests['outer_manifest']
    print("\n" + "=" * 60)
    print(" Canonical Outer Folds 摘要")
    print("=" * 60)
    for fold in om['folds']:
        tc = fold['train_counts']
        vc = fold['test_counts']
        print(f"  Fold {fold['fold_id']}: "
              f"train={fold['n_train']} (neg={tc['y_binary']['0']}, pos={tc['y_binary']['1']}), "
              f"test={fold['n_test']} (neg={vc['y_binary']['0']}, pos={vc['y_binary']['1']})")
        print(f"    y_original train: 1={tc['y_original']['1']}, 2={tc['y_original']['2']}, 3={tc['y_original']['3']}")
        print(f"    y_original test:  1={vc['y_original']['1']}, 2={vc['y_original']['2']}, 3={vc['y_original']['3']}")

    lm = manifests['legacy_manifest']
    print("\n" + "=" * 60)
    print(" Legacy Holdout 摘要")
    print("=" * 60)
    print(f"  Status: {lm['status']}")
    print(f"  Train: {lm['n_train']}, Test: {lm['n_test']}")
    tc = lm['train_counts']
    vc = lm['test_counts']
    print(f"  Train: neg={tc['y_binary']['0']}, pos={tc['y_binary']['1']}")
    print(f"  Test:  neg={vc['y_binary']['0']}, pos={vc['y_binary']['1']}")
    print(f"  y_original test: 1={vc['y_original']['1']}, 2={vc['y_original']['2']}, 3={vc['y_original']['3']}")