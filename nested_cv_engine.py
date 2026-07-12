#!/usr/bin/env python3
"""
nested_cv_engine.py — 阶段二：Nested CV 引擎

实现完整的 Nested CV 流程：
- 每个 outer fold 内，对每个候选独立生成 inner OOF score
- 对每个候选独立选择阈值
- 按 inner OOF tuned F1 排名选择最佳候选
- 在完整 outer-train 上重新 fit 最佳候选
- 在 outer-test 上预测
- 支持 dataflow-smoke 模式（不读取 outer-test 标签）
- 支持 frozen config 验证
"""

import hashlib
import os
import json
import time
import warnings
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from data_pipeline import build_dataset, EXCEL_PATH
from dataset_schema import VOCDatasetBundle
from metrics import compute_binary_metrics, compute_foldwise_mean_std
from thresholding import select_threshold, ThresholdSelectionResult
from candidate_registry import (
    generate_candidates,
    build_candidate_pipeline,
    get_candidate_counts,
)
from models_sklearn import (
    get_pipeline_score,
    get_pipeline_score_type,
    get_default_threshold,
    SCORE_FUNCTIONS,
)


# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR = os.path.join(PROJECT_ROOT, 'splits')
RESULT_DIR = os.path.join(PROJECT_ROOT, 'result', 'nested_cv', 'traditional_selector')
CONFIGS_DIR = os.path.join(PROJECT_ROOT, 'configs')
FROZEN_CONFIG_PATH = os.path.join(CONFIGS_DIR, 'traditional_selector_frozen.yaml')


# ============================================================
# Frozen Config 操作
# ============================================================
def _compute_frozen_config_hash() -> str:
    """计算 frozen config 文件的 SHA-256 哈希"""
    if not os.path.exists(FROZEN_CONFIG_PATH):
        raise FileNotFoundError(f"Frozen config not found: {FROZEN_CONFIG_PATH}")
    with open(FROZEN_CONFIG_PATH, 'r') as f:
        content = f.read()
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def _validate_frozen_config(
    bundle: VOCDatasetBundle,
    outer_manifest: dict,
    inner_manifests: List[dict],
) -> Optional[str]:
    """验证 frozen config 与当前数据集/manifests 的一致性。

    Returns:
        None 表示一致，否则返回错误信息字符串
    """
    if not os.path.exists(FROZEN_CONFIG_PATH):
        return "Frozen config file not found. Run --freeze-config to create it."

    import yaml
    with open(FROZEN_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # 验证 dataset fingerprint
    frozen_fp = config.get('dataset', {}).get('fingerprint')
    if frozen_fp and frozen_fp != bundle.dataset_fingerprint:
        return (f"Dataset fingerprint mismatch. "
                f"Frozen: {frozen_fp[:16]}..., Current: {bundle.dataset_fingerprint[:16]}...")

    # 验证 outer manifest hash
    frozen_outer_hash = config.get('manifests', {}).get('outer', {}).get('hash')
    if frozen_outer_hash and frozen_outer_hash != outer_manifest.get('manifest_hash'):
        return (f"Outer manifest hash mismatch. "
                f"Frozen: {frozen_outer_hash[:16]}..., Current: {outer_manifest.get('manifest_hash', 'N/A')[:16]}...")

    return None


def _detect_frozen_config_mismatch(
    bundle: VOCDatasetBundle,
    outer_manifest: dict,
) -> dict:
    """检测 frozen config 与实际配置的差异。

    Returns:
        dict with keys: frozen_config_hash, match, mismatches
    """
    result = {
        'frozen_config_hash': _compute_frozen_config_hash(),
        'match': True,
        'mismatches': [],
    }

    import yaml
    with open(FROZEN_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    frozen_fp = config.get('dataset', {}).get('fingerprint')
    if frozen_fp and frozen_fp != bundle.dataset_fingerprint:
        result['match'] = False
        result['mismatches'].append({
            'field': 'dataset.fingerprint',
            'frozen': frozen_fp,
            'current': bundle.dataset_fingerprint,
        })

    frozen_outer_hash = config.get('manifests', {}).get('outer', {}).get('hash')
    if frozen_outer_hash and frozen_outer_hash != outer_manifest.get('manifest_hash'):
        result['match'] = False
        result['mismatches'].append({
            'field': 'manifests.outer.hash',
            'frozen': frozen_outer_hash,
            'current': outer_manifest.get('manifest_hash'),
        })

    return result


def _load_manifest(manifest_name: str) -> dict:
    """加载 manifest 文件"""
    path = os.path.join(SPLITS_DIR, manifest_name)
    with open(path, 'r') as f:
        return json.load(f)


# ============================================================
# 内层候选评估
# ============================================================
def evaluate_candidate_inner(
    candidate: dict,
    bundle: VOCDatasetBundle,
    outer_fold: dict,
    inner_manifest: dict,
) -> Optional[dict]:
    """为单个候选在 inner folds 上生成 OOF scores 并选择阈值。

    Args:
        candidate: 候选配置 dict
        bundle: VOCDatasetBundle
        outer_fold: outer fold manifest entry
        inner_manifest: inner fold manifest

    Returns:
        dict with inner evaluation results, or None if failed
    """
    candidate_id = candidate['candidate_id']
    model_family = candidate['model_family']
    score_type = candidate['score_type']
    default_threshold = candidate['default_threshold']
    start_time = time.time()

    # 获取 outer-train 样本索引
    train_ids_set = set(outer_fold['train_sample_ids'])
    train_mask = np.array([sid in train_ids_set for sid in bundle.sample_ids])
    outer_train_indices = np.where(train_mask)[0]
    n_outer_train = len(outer_train_indices)

    # OOF score 数组
    oof_scores = np.full(n_outer_train, np.nan, dtype=np.float64)
    oof_y_true = bundle.y_binary[outer_train_indices].copy()
    oof_y_original = bundle.y_original[outer_train_indices].copy()
    oof_inner_fold = np.full(n_outer_train, -1, dtype=int)

    # 创建 sample_id → outer_train 位置映射
    sid_to_outer_pos = {
        sid: i for i, sid in enumerate(bundle.sample_ids[outer_train_indices])
    }

    convergence_warnings = 0
    inner_fold_results = []

    for inner_fold in inner_manifest['inner_folds']:
        inner_train_ids = set(inner_fold['train_sample_ids'])
        inner_val_ids = set(inner_fold['val_sample_ids'])

        # 获取 inner-train 和 inner-val 在 outer_train 中的位置
        inner_train_outer_pos = [
            sid_to_outer_pos[sid] for sid in inner_fold['train_sample_ids']
            if sid in sid_to_outer_pos
        ]
        inner_val_outer_pos = [
            sid_to_outer_pos[sid] for sid in inner_fold['val_sample_ids']
            if sid in sid_to_outer_pos
        ]

        if not inner_train_outer_pos or not inner_val_outer_pos:
            continue

        # 获取实际数据
        X_inner_train = bundle.X[outer_train_indices[inner_train_outer_pos]]
        y_inner_train = bundle.y_binary[outer_train_indices[inner_train_outer_pos]]
        X_inner_val = bundle.X[outer_train_indices[inner_val_outer_pos]]

        # 构建并 fit Pipeline
        try:
            pipe = candidate['build_pipeline']()

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                pipe.fit(X_inner_train, y_inner_train)

                # 检查收敛警告
                for warning in w:
                    if issubclass(warning.category, UserWarning):
                        if 'Convergence' in str(warning.message):
                            convergence_warnings += 1

            # 生成 score
            val_scores = get_pipeline_score(pipe, X_inner_val, model_family)

            # 写入 OOF 数组
            for pos, score in zip(inner_val_outer_pos, val_scores):
                oof_scores[pos] = score
                oof_inner_fold[pos] = inner_fold['inner_fold_id']

            inner_fold_results.append({
                'inner_fold_id': inner_fold['inner_fold_id'],
                'n_train': len(inner_train_outer_pos),
                'n_val': len(inner_val_outer_pos),
                'status': 'ok',
            })

        except Exception as e:
            return {
                'candidate_id': candidate_id,
                'model_family': model_family,
                'status': 'invalid',
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'runtime': time.time() - start_time,
                'convergence_warnings': convergence_warnings,
            }

    # 验证 OOF 完整性
    if np.any(np.isnan(oof_scores)):
        nan_count = int(np.isnan(oof_scores).sum())
        return {
            'candidate_id': candidate_id,
            'model_family': model_family,
            'status': 'invalid',
            'error_type': 'IncompleteOOF',
            'error_message': f'{nan_count} OOF scores missing (NaN)',
            'runtime': time.time() - start_time,
            'convergence_warnings': convergence_warnings,
        }

    # 选择阈值
    try:
        thresh_result = select_threshold(oof_y_true, oof_scores, score_type)
    except Exception as e:
        return {
            'candidate_id': candidate_id,
            'model_family': model_family,
            'status': 'invalid',
            'error_type': 'ThresholdFailed',
            'error_message': str(e),
            'runtime': time.time() - start_time,
            'convergence_warnings': convergence_warnings,
        }

    # 使用调优阈值计算预测
    from thresholding import _apply_threshold
    y_pred_tuned = _apply_threshold(oof_scores, thresh_result.threshold, score_type)
    y_pred_default = _apply_threshold(oof_scores, default_threshold, score_type)

    # 计算 inner OOF metrics
    tuned_metrics = compute_binary_metrics(oof_y_true, y_pred_tuned, oof_scores)
    default_metrics = compute_binary_metrics(oof_y_true, y_pred_default, oof_scores)

    return {
        'candidate_id': candidate_id,
        'model_family': model_family,
        'status': 'valid',
        'params': candidate['params'],
        'score_type': score_type,
        'inner_tuned_threshold': thresh_result.threshold,
        'inner_tuned_f1': tuned_metrics['f1'],
        'inner_tuned_mcc': tuned_metrics['mcc'],
        'inner_tuned_balanced_accuracy': tuned_metrics['balanced_accuracy'],
        'inner_tuned_precision': tuned_metrics['precision'],
        'inner_tuned_recall': tuned_metrics['recall'],
        'inner_tuned_specificity': tuned_metrics['specificity'],
        'inner_tuned_accuracy': tuned_metrics['accuracy'],
        'inner_pr_auc': tuned_metrics['pr_auc'],
        'inner_roc_auc': tuned_metrics['roc_auc'],
        'inner_default_f1': default_metrics['f1'],
        'inner_default_mcc': default_metrics['mcc'],
        'inner_default_balanced_accuracy': default_metrics['balanced_accuracy'],
        'inner_predicted_positive_count': tuned_metrics['predicted_positive_count'],
        'inner_predicted_positive_rate': tuned_metrics['predicted_positive_rate'],
        'convergence_warnings': convergence_warnings,
        'runtime': time.time() - start_time,
        'oof_scores': oof_scores,
        'oof_y_true': oof_y_true,
        'threshold_result': thresh_result,
        'outer_train_indices': outer_train_indices,
    }


def rank_candidates(candidate_results: List[dict]) -> List[dict]:
    """按 inner OOF tuned F1 → MCC → Balanced Accuracy → PR-AUC → candidate_id 排名。

    Args:
        candidate_results: 有效候选结果列表

    Returns:
        排序后的候选结果列表
    """
    def sort_key(cr):
        f1 = cr.get('inner_tuned_f1') or -1.0
        mcc = cr.get('inner_tuned_mcc') or -1.0
        ba = cr.get('inner_tuned_balanced_accuracy') or 0.0
        pr_auc = cr.get('inner_pr_auc') or -1.0
        cid = cr['candidate_id']
        return (-f1, -mcc, -ba, -pr_auc, cid)

    return sorted(candidate_results, key=sort_key)


# ============================================================
# Outer Fold 最终拟合
# ============================================================
def fit_and_evaluate_outer(
    best_candidate: dict,
    bundle: VOCDatasetBundle,
    outer_fold: dict,
    inner_result: dict,
) -> dict:
    """在完整 outer-train 上重新 fit 最佳候选，在 outer-test 上预测。

    Args:
        best_candidate: 候选配置
        bundle: VOCDatasetBundle
        outer_fold: outer fold manifest
        inner_result: 内层评估结果

    Returns:
        outer fold 结果 dict
    """
    model_family = best_candidate['model_family']
    score_type = best_candidate['score_type']
    threshold = inner_result['inner_tuned_threshold']
    default_threshold = best_candidate['default_threshold']

    # 获取 outer-train 和 outer-test 索引
    train_ids_set = set(outer_fold['train_sample_ids'])
    test_ids_set = set(outer_fold['test_sample_ids'])

    train_mask = np.array([sid in train_ids_set for sid in bundle.sample_ids])
    test_mask = np.array([sid in test_ids_set for sid in bundle.sample_ids])

    X_train = bundle.X[train_mask]
    y_train = bundle.y_binary[train_mask]
    X_test = bundle.X[test_mask]
    y_test = bundle.y_binary[test_mask]
    y_test_original = bundle.y_original[test_mask]
    test_sample_ids = bundle.sample_ids[test_mask]

    # 构建并 fit Pipeline
    pipe = None
    fit_warnings = 0
    try:
        pipe = best_candidate['build_pipeline']()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipe.fit(X_train, y_train)

            for warning in w:
                if issubclass(warning.category, UserWarning):
                    if 'Convergence' in str(warning.message):
                        fit_warnings += 1
    except Exception as e:
        return {
            'status': 'outer_fit_failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
        }

    # 提取特征信息
    feature_info = _extract_features(pipe, bundle, best_candidate['model_family'])

    # 预测
    test_scores = get_pipeline_score(pipe, X_test, model_family)

    from thresholding import _apply_threshold
    y_pred_tuned = _apply_threshold(test_scores, threshold, score_type)
    y_pred_default = _apply_threshold(test_scores, default_threshold, score_type)

    # 计算指标
    tuned_metrics = compute_binary_metrics(y_test, y_pred_tuned, test_scores)
    default_metrics = compute_binary_metrics(y_test, y_pred_default, test_scores)

    # 子组错误
    subgroup_errors = _compute_subgroup_errors(
        y_test, y_test_original, y_pred_tuned, y_pred_default
    )

    # 获取 Excel 行号
    excel_rows = _get_excel_rows(test_sample_ids)

    return {
        'outer_fold_id': outer_fold['fold_id'],
        'selected_candidate_id': best_candidate['candidate_id'],
        'selected_model_family': model_family,
        'selected_threshold': threshold,
        'score_type': score_type,
        'default_threshold': default_threshold,
        'tuned_metrics': tuned_metrics,
        'default_metrics': default_metrics,
        'test_scores': test_scores,
        'y_test': y_test,
        'y_test_original': y_test_original,
        'test_sample_ids': test_sample_ids,
        'test_excel_rows': excel_rows,
        'y_pred_tuned': y_pred_tuned,
        'y_pred_default': y_pred_default,
        'subgroup_errors': subgroup_errors,
        'feature_info': feature_info,
        'fit_warnings': fit_warnings,
        'n_train': len(y_train),
        'n_test': len(y_test),
        'train_counts': {
            'neg': int((y_train == 0).sum()),
            'pos': int((y_train == 1).sum()),
        },
        'test_counts': {
            'neg': int((y_test == 0).sum()),
            'pos': int((y_test == 1).sum()),
        },
    }


def _extract_features(pipe: Pipeline, bundle: VOCDatasetBundle, model_family: str) -> dict:
    """提取 fold-wise 保留特征和系数。

    字段语义:
    - selected_by_voc_filter: 通过 VOCAbundanceIQRFilter 保留
    - selected_by_optional_selector: 通过 SelectKBest 保留 (无 SelectKBest 时为 None)
    - coefficient_available: 模型是否提供系数 (ElasticNet/LinearSVM=True, RBF/LDA=False)
    - coefficient_nonzero: 系数绝对值 > 1e-10
    - selected_by_classifier: 对于 ElasticNet/LinearSVM, 系数非零; 对于 RBF/LDA, None/not_applicable
    - final_active_feature: 同时满足所有适用选择步骤
    """
    COEF_TOLERANCE = 1e-10
    features = []

    voc_filter = pipe.named_steps.get('voc_filter')
    selector = pipe.named_steps.get('selector')
    classifier = pipe.named_steps.get('classifier')

    voc_support = None
    if voc_filter is not None and hasattr(voc_filter, 'support_'):
        voc_support = voc_filter.support_

    selector_support = None
    if selector is not None and hasattr(selector, 'get_support'):
        selector_support = selector.get_support(indices=False)

    # 确定模型是否提供系数
    coef_available = model_family in ['elastic_net', 'linear_svm']
    coef = None
    if coef_available:
        if hasattr(classifier, 'coef_'):
            coef = classifier.coef_.flatten()

    # 构建特征列表
    n_features = bundle.n_features
    for i in range(n_features):
        feat = {
            'feature_name': str(bundle.feature_names[i]),
            'feature_id': str(bundle.feature_ids[i]) if bundle.feature_ids is not None else '',
            'selected_by_voc_filter': bool(voc_support[i]) if voc_support is not None else None,
            'selected_by_optional_selector': None,
            'classifier_coefficient': None,
            'coefficient_sign': None,
            'coefficient_available': coef_available,
            'coefficient_nonzero': None,
            'selected_by_classifier': None,
            'final_active_feature': False,
        }
        if voc_support is not None and voc_support[i]:
            voc_idx = int(voc_support[:i].sum())
            if selector_support is not None and voc_idx < len(selector_support):
                feat['selected_by_optional_selector'] = bool(selector_support[voc_idx])
            if coef is not None and voc_idx < len(coef):
                feat['classifier_coefficient'] = float(coef[voc_idx])
                feat['coefficient_sign'] = 'positive' if coef[voc_idx] > 0 else ('negative' if coef[voc_idx] < 0 else 'zero')
                feat['coefficient_nonzero'] = abs(coef[voc_idx]) > COEF_TOLERANCE
                feat['selected_by_classifier'] = feat['coefficient_nonzero']
            else:
                # 无系数可用 (RBF SVM, LDA)
                feat['coefficient_available'] = coef_available

        # 对 RBF SVM 和 LDA: 明确标记为 not_applicable
        if not coef_available:
            feat['coefficient_nonzero'] = None
            feat['selected_by_classifier'] = None

        # 计算 final_active_feature
        # 同时满足: voc_filter 通过 + (selector 通过或无 selector) + (classifier 通过或无系数)
        voc_ok = feat['selected_by_voc_filter'] is True
        sel_ok = (feat['selected_by_optional_selector'] is None
                  or feat['selected_by_optional_selector'] is True)
        cls_ok = (feat['selected_by_classifier'] is None
                  or feat['selected_by_classifier'] is True)
        feat['final_active_feature'] = voc_ok and sel_ok and cls_ok

        features.append(feat)

    # 统计
    voc_filtered_count = int(voc_support.sum()) if voc_support is not None else n_features
    selector_count = None
    if selector_support is not None:
        if selector is not None:
            selector_count = int(selector_support.sum())

    # 统计 final_active 特征数
    final_active_count = sum(1 for f in features if f['final_active_feature'])
    classifier_selected_count = sum(1 for f in features if f['selected_by_classifier'] is True)
    coef_nonzero_count = sum(1 for f in features if f['coefficient_nonzero'] is True)

    return {
        'features': features,
        'n_before_voc_filter': n_features,
        'n_after_voc_filter': voc_filtered_count,
        'n_after_selector': selector_count,
        'n_final_active': final_active_count,
        'n_classifier_selected': classifier_selected_count if coef_available else None,
        'n_coefficient_nonzero': coef_nonzero_count if coef_available else None,
        'coefficient_available': coef_available,
    }


def _compute_subgroup_errors(
    y_test: np.ndarray,
    y_test_original: np.ndarray,
    y_pred_tuned: np.ndarray,
    y_pred_default: np.ndarray,
) -> dict:
    """计算子组错误"""
    result = {}

    for orig_label in [1, 2, 3]:
        mask = y_test_original == orig_label
        n = int(mask.sum())

        if n == 0:
            result[f'class_{orig_label}'] = {
                'n': 0, 'tuned_pred_pos_count': 0, 'tuned_pred_pos_rate': None,
                'default_pred_pos_count': 0, 'default_pred_pos_rate': None,
            }
            if orig_label == 3:
                result[f'class_{orig_label}']['tuned_recall'] = None
                result[f'class_{orig_label}']['default_recall'] = None
        else:
            tuned_pos = int((y_pred_tuned[mask] == 1).sum())
            default_pos = int((y_pred_default[mask] == 1).sum())
            entry = {
                'n': n,
                'tuned_pred_pos_count': tuned_pos,
                'tuned_pred_pos_rate': float(tuned_pos / n),
                'default_pred_pos_count': default_pos,
                'default_pred_pos_rate': float(default_pos / n),
            }
            if orig_label == 3:
                entry['tuned_recall'] = float(tuned_pos / n)
                entry['default_recall'] = float(default_pos / n)
            result[f'class_{orig_label}'] = entry

    return result


def _get_excel_rows(sample_ids: np.ndarray) -> List[int]:
    """从 sample_id 提取 Excel 行号"""
    rows = []
    for sid in sample_ids:
        # sample_row_XXXX → XXXX
        parts = sid.split('_')
        if len(parts) >= 3:
            rows.append(int(parts[-1]))
        else:
            rows.append(-1)
    return rows


# ============================================================
# 基线评估
# ============================================================
def evaluate_baselines(
    bundle: VOCDatasetBundle,
    outer_fold: dict,
) -> List[dict]:
    """评估基线模型 (Dummy + AllPositive)"""
    train_ids_set = set(outer_fold['train_sample_ids'])
    test_ids_set = set(outer_fold['test_sample_ids'])

    train_mask = np.array([sid in train_ids_set for sid in bundle.sample_ids])
    test_mask = np.array([sid in test_ids_set for sid in bundle.sample_ids])

    X_train = bundle.X[train_mask]
    y_train = bundle.y_binary[train_mask]
    X_test = bundle.X[test_mask]
    y_test = bundle.y_binary[test_mask]

    results = []

    # Dummy most_frequent
    dummy_mf = DummyClassifier(strategy='most_frequent')
    dummy_mf.fit(X_train, y_train)
    y_pred_mf = dummy_mf.predict(X_test)
    metrics_mf = compute_binary_metrics(y_test, y_pred_mf, None)
    results.append({
        'baseline_name': 'DummyMostFrequent',
        'metrics': metrics_mf,
    })

    # Dummy stratified
    dummy_st = DummyClassifier(
        strategy='stratified',
        random_state=1000 + outer_fold['fold_id'],
    )
    dummy_st.fit(X_train, y_train)
    y_pred_st = dummy_st.predict(X_test)
    metrics_st = compute_binary_metrics(y_test, y_pred_st, None)
    results.append({
        'baseline_name': 'DummyStratified',
        'metrics': metrics_st,
    })

    # AllPositive
    y_pred_ap = np.ones(len(y_test), dtype=int)
    metrics_ap = compute_binary_metrics(y_test, y_pred_ap, None)
    results.append({
        'baseline_name': 'AllPositive',
        'metrics': metrics_ap,
    })

    return results


# ============================================================
# Dataflow Smoke 模式
# ============================================================
def run_dataflow_smoke(n_jobs: int = 1) -> dict:
    """运行 dataflow smoke 测试，验证 pipeline 数据流但不读取 outer-test 标签。

    此模式:
    - 加载数据和 manifests
    - 验证 manifest 加载
    - 构建并验证 pipeline 构建
    - 在 inner fold 上训练并生成 OOF scores
    - 执行 threshold selection
    - 验证 outer-train refit 接口
    - 验证输出 schema
    - 绝不对 canonical outer-test 生成 score 或读取标签计算指标

    使用 outer fold 0 的 inner folds 进行验证，不接触 outer-test。
    """
    start_time = time.time()
    start_datetime = datetime.now(timezone.utc).isoformat()

    bundle = build_dataset(EXCEL_PATH)
    print(f"Dataset loaded: {bundle.n_samples} samples, {bundle.n_features} features")

    outer_manifest = _load_manifest('canonical_outer_folds.json')
    if outer_manifest['dataset_fingerprint'] != bundle.dataset_fingerprint:
        raise ValueError("Dataset fingerprint mismatch with outer manifest")

    inner_manifest = _load_manifest('outer_0_inner_folds.json')
    if inner_manifest['dataset_fingerprint'] != bundle.dataset_fingerprint:
        raise ValueError("Inner fold 0 fingerprint mismatch")

    outer_fold = outer_manifest['folds'][0]

    # 验证 inner 样本属于 outer-train
    train_ids_set = set(outer_fold['train_sample_ids'])
    for inner_fold in inner_manifest['inner_folds']:
        for sid in inner_fold['train_sample_ids'] + inner_fold['val_sample_ids']:
            if sid not in train_ids_set:
                raise ValueError(f"Inner sample {sid} not in outer-train for fold 0")

    # 验证 outer-test 不在 inner folds
    test_ids_set = set(outer_fold['test_sample_ids'])
    for inner_fold in inner_manifest['inner_folds']:
        all_inner_ids = set(inner_fold['train_sample_ids'] + inner_fold['val_sample_ids'])
        overlap = all_inner_ids & test_ids_set
        if overlap:
            raise ValueError(f"Outer-test samples in inner folds: {overlap}")

    # 生成候选
    candidates = generate_candidates('quick')
    print(f"\nDataflow Smoke: {len(candidates)} candidates")

    # 验证 pipeline 构建
    print("\n--- Pipeline construction validation ---")
    for candidate in candidates:
        try:
            pipe = candidate['build_pipeline']()
            print(f"  OK {candidate['candidate_id']}: {len(pipe.steps)} steps")
        except Exception as e:
            print(f"  FAIL {candidate['candidate_id']}: {e}")
            raise

    # 验证 inner fold fit + OOF generation
    print("\n--- Inner fold fit + OOF validation ---")
    train_mask = np.array([sid in train_ids_set for sid in bundle.sample_ids])
    outer_train_indices = np.where(train_mask)[0]
    n_outer_train = len(outer_train_indices)

    sid_to_outer_pos = {
        sid: i for i, sid in enumerate(bundle.sample_ids[outer_train_indices])
    }

    for candidate in candidates[:3]:
        cid = candidate['candidate_id']
        print(f"  {cid} ...", end=' ')

        oof_scores = np.full(n_outer_train, np.nan, dtype=np.float64)
        oof_y_true = bundle.y_binary[outer_train_indices].copy()

        for inner_fold in inner_manifest['inner_folds']:
            inner_train_outer_pos = [
                sid_to_outer_pos[sid] for sid in inner_fold['train_sample_ids']
                if sid in sid_to_outer_pos
            ]
            inner_val_outer_pos = [
                sid_to_outer_pos[sid] for sid in inner_fold['val_sample_ids']
                if sid in sid_to_outer_pos
            ]

            if not inner_train_outer_pos or not inner_val_outer_pos:
                continue

            X_inner_train = bundle.X[outer_train_indices[inner_train_outer_pos]]
            y_inner_train = bundle.y_binary[outer_train_indices[inner_train_outer_pos]]
            X_inner_val = bundle.X[outer_train_indices[inner_val_outer_pos]]

            pipe = candidate['build_pipeline']()
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                pipe.fit(X_inner_train, y_inner_train)

            val_scores = get_pipeline_score(pipe, X_inner_val, candidate['model_family'])
            for pos, score in zip(inner_val_outer_pos, val_scores):
                oof_scores[pos] = score

        if np.any(np.isnan(oof_scores)):
            nan_count = int(np.isnan(oof_scores).sum())
            print(f"FAIL ({nan_count} NaN OOF)")
            raise ValueError(f"Incomplete OOF for {cid}: {nan_count} NaN")
        print("OK OOF complete")

        score_type = candidate['score_type']
        thresh_result = select_threshold(oof_y_true, oof_scores, score_type)
        print(f"    threshold={thresh_result.threshold:.6f}, "
              f"tuned_f1={thresh_result.f1:.4f}")

    # 验证 outer-train refit 接口
    print("\n--- Outer-train refit interface validation ---")
    best_candidate = candidates[0]
    X_outer_train = bundle.X[outer_train_indices]
    y_outer_train = bundle.y_binary[outer_train_indices]

    pipe = best_candidate['build_pipeline']()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        pipe.fit(X_outer_train, y_outer_train)
    print(f"  OK {best_candidate['candidate_id']}: fit on outer-train (n={len(y_outer_train)})")

    # 验证特征提取
    feature_info = _extract_features(pipe, bundle, best_candidate['model_family'])
    print(f"  OK Feature extraction: {feature_info['n_after_voc_filter']} after VOC filter, "
          f"{feature_info.get('n_final_active', '?')} final active")

    # 验证输出 schema
    print("\n--- Output schema validation ---")
    output_dir = os.path.join(RESULT_DIR, 'dataflow_smoke')
    os.makedirs(output_dir, exist_ok=True)

    test_files = [
        'oof_predictions.csv', 'outer_fold_metrics.csv', 'aggregate_metrics.json',
        'inner_candidate_results.csv', 'selected_configs.json', 'thresholds.csv',
        'selected_features.csv', 'subgroup_errors.csv', 'run_metadata.json',
    ]
    for fname in test_files:
        test_path = os.path.join(output_dir, fname)
        with open(test_path, 'w') as f:
            f.write('')
        print(f"  OK {fname} writable")

    # 写入 smoke metadata
    total_runtime = time.time() - start_time
    frozen_cfg_hash = _compute_frozen_config_hash()
    mismatch_info = _detect_frozen_config_mismatch(bundle, outer_manifest)

    smoke_metadata = {
        'mode': 'dataflow_smoke',
        'smoke_test_only': True,
        'formal_evaluation': False,
        'canonical_outer_test_not_accessed': True,
        'outer_test_labels_not_read': True,
        'outer_test_predictions_not_generated': True,
        'start_time': start_datetime,
        'end_time': datetime.now(timezone.utc).isoformat(),
        'runtime': total_runtime,
        'frozen_config_hash': frozen_cfg_hash,
        'frozen_config_match': mismatch_info['match'],
        'frozen_config_mismatches': mismatch_info['mismatches'],
        'dataset_fingerprint': bundle.dataset_fingerprint,
        'outer_manifest_hash': outer_manifest['manifest_hash'],
        'validations': {
            'manifest_loading': True,
            'pipeline_construction': True,
            'inner_fold_fit': True,
            'inner_validation_score': True,
            'candidate_specific_oof': True,
            'threshold_selection': True,
            'outer_train_refit_interface': True,
            'output_schema': True,
        },
    }

    with open(os.path.join(output_dir, 'run_metadata.json'), 'w') as f:
        json.dump(smoke_metadata, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nDataflow smoke passed. Runtime: {total_runtime:.1f}s")
    print(f"Output: {output_dir}")

    return smoke_metadata


# ============================================================
# 主 Nested CV 流程
# ============================================================
def run_nested_cv(
    mode: str = 'full',
    n_jobs: int = 1,
    outer_folds: Optional[List[int]] = None,
) -> dict:
    """运行完整 Nested CV 流程。

    Args:
        mode: 'full' 或 'quick'
        n_jobs: 并行任务数 (预留)
        outer_folds: 要运行的 outer fold ID 列表，None 表示全部

    Returns:
        dict: 完整结果
    """
    start_time = time.time()
    start_datetime = datetime.now(timezone.utc).isoformat()

    # 加载数据
    bundle = build_dataset(EXCEL_PATH)
    print(f"Dataset loaded: {bundle.n_samples} samples, {bundle.n_features} features")
    print(f"Fingerprint: {bundle.dataset_fingerprint[:32]}...")

    # 加载 manifests
    outer_manifest = _load_manifest('canonical_outer_folds.json')
    if outer_manifest['dataset_fingerprint'] != bundle.dataset_fingerprint:
        raise ValueError("Dataset fingerprint mismatch with outer manifest")

    # Full 模式：验证 frozen config
    frozen_config_hash = None
    if mode == 'full':
        frozen_config_hash = _compute_frozen_config_hash()
        mismatch_err = _validate_frozen_config(bundle, outer_manifest, [])
        if mismatch_err:
            raise RuntimeError(
                f"Frozen config validation failed: {mismatch_err}\n"
                f"To create/update frozen config, run: python run_nested_cv.py --freeze-config\n"
                f"To create a new protocol version, use --freeze-config --protocol-version v2.0.0"
            )
        print(f"Frozen config validated: {frozen_config_hash[:16]}...")

    # 确定运行哪些 outer folds
    if outer_folds is None:
        outer_folds = list(range(len(outer_manifest['folds'])))
    folds_to_run = [f for f in outer_manifest['folds'] if f['fold_id'] in outer_folds]

    # 生成候选
    candidates = generate_candidates(mode)
    print(f"\nMode: {mode}")
    print(f"Candidates: {len(candidates)}")
    print(f"Outer folds to run: {[f['fold_id'] for f in folds_to_run]}")

    # 输出目录
    output_dir = os.path.join(RESULT_DIR, mode)
    os.makedirs(output_dir, exist_ok=True)

    # 存储结果
    all_outer_results = []
    all_inner_results = []
    all_baseline_results = []
    all_oof_predictions = []
    all_selected_features = []
    all_subgroup_errors = []
    all_thresholds = []

    for outer_fold in folds_to_run:
        fold_id = outer_fold['fold_id']
        print(f"\n{'='*60}")
        print(f" Outer Fold {fold_id}")
        print(f"{'='*60}")

        # 加载 inner manifest
        inner_manifest = _load_manifest(f'outer_{fold_id}_inner_folds.json')
        if inner_manifest['dataset_fingerprint'] != bundle.dataset_fingerprint:
            raise ValueError(f"Inner fold {fold_id} fingerprint mismatch")

        # 验证 inner 样本属于 outer-train
        train_ids_set = set(outer_fold['train_sample_ids'])
        for inner_fold in inner_manifest['inner_folds']:
            for sid in inner_fold['train_sample_ids'] + inner_fold['val_sample_ids']:
                if sid not in train_ids_set:
                    raise ValueError(f"Inner sample {sid} not in outer-train for fold {fold_id}")

        # 验证 outer-test 不在 inner folds
        test_ids_set = set(outer_fold['test_sample_ids'])
        for inner_fold in inner_manifest['inner_folds']:
            all_inner_ids = set(inner_fold['train_sample_ids'] + inner_fold['val_sample_ids'])
            overlap = all_inner_ids & test_ids_set
            if overlap:
                raise ValueError(f"Outer-test samples in inner folds: {overlap}")

        # 评估每个候选
        fold_inner_results = []
        valid_count = 0
        invalid_count = 0

        for i, candidate in enumerate(candidates):
            cid = candidate['candidate_id']
            print(f"  [{i+1}/{len(candidates)}] {cid} ...", end=' ')

            result = evaluate_candidate_inner(
                candidate, bundle, outer_fold, inner_manifest
            )

            if result['status'] == 'valid':
                valid_count += 1
                print(f"✓ F1={result['inner_tuned_f1']:.4f}")
            else:
                invalid_count += 1
                print(f"✗ {result['error_type']}")

            # 保存到 inner results
            inner_save = {
                'outer_fold': fold_id,
                'candidate_id': cid,
                'model_family': candidate['model_family'],
                'params_json': json.dumps(candidate['params']),
                'selector_k': candidate['params'].get('k'),
                'status': result['status'],
                'inner_tuned_threshold': result.get('inner_tuned_threshold'),
                'inner_tuned_f1': result.get('inner_tuned_f1'),
                'inner_tuned_mcc': result.get('inner_tuned_mcc'),
                'inner_tuned_balanced_accuracy': result.get('inner_tuned_balanced_accuracy'),
                'inner_pr_auc': result.get('inner_pr_auc'),
                'inner_roc_auc': result.get('inner_roc_auc'),
                'inner_default_f1': result.get('inner_default_f1'),
                'inner_default_mcc': result.get('inner_default_mcc'),
                'predicted_positive_count': result.get('inner_predicted_positive_count'),
                'warning_count': result.get('convergence_warnings', 0),
                'error_type': result.get('error_type'),
                'error_message': result.get('error_message'),
                'runtime': result.get('runtime'),
            }
            all_inner_results.append(inner_save)

            if result['status'] == 'valid':
                fold_inner_results.append(result)

        print(f"  Valid: {valid_count}, Invalid: {invalid_count}")

        if len(fold_inner_results) == 0:
            raise RuntimeError(f"All candidates failed for outer fold {fold_id}")

        # 排名
        ranked = rank_candidates(fold_inner_results)
        best = ranked[0]
        best_candidate = next(
            c for c in candidates if c['candidate_id'] == best['candidate_id']
        )

        print(f"  Best: {best['candidate_id']} (F1={best['inner_tuned_f1']:.4f})")

        # Outer 最终拟合
        outer_result = fit_and_evaluate_outer(
            best_candidate, bundle, outer_fold, best
        )

        if outer_result.get('status') == 'outer_fit_failed':
            print(f"  ✗ Outer fit failed: {outer_result['error']}")
            continue

        # 保存 outer 结果
        all_outer_results.append(outer_result)

        # 保存 OOF predictions
        for i in range(len(outer_result['test_sample_ids'])):
            all_oof_predictions.append({
                'sample_id': outer_result['test_sample_ids'][i],
                'excel_row': outer_result['test_excel_rows'][i],
                'y_original': int(outer_result['y_test_original'][i]),
                'y_binary': int(outer_result['y_test'][i]),
                'outer_fold': fold_id,
                'selected_candidate_id': outer_result['selected_candidate_id'],
                'selected_model_family': outer_result['selected_model_family'],
                'selected_threshold': outer_result['selected_threshold'],
                'score_type': outer_result['score_type'],
                'score': float(outer_result['test_scores'][i]),
                'prediction_tuned': int(outer_result['y_pred_tuned'][i]),
                'prediction_default': int(outer_result['y_pred_default'][i]),
                'default_threshold': outer_result['default_threshold'],
                'dataset_fingerprint': bundle.dataset_fingerprint,
                'outer_manifest_hash': outer_manifest['manifest_hash'],
            })

        # 保存 selected features
        for feat in outer_result['feature_info']['features']:
            if feat['selected_by_voc_filter']:
                all_selected_features.append({
                    'outer_fold': fold_id,
                    'candidate_id': outer_result['selected_candidate_id'],
                    'feature_name': feat['feature_name'],
                    'feature_id': feat['feature_id'],
                    'selected_by_voc_filter': feat['selected_by_voc_filter'],
                    'selected_by_optional_selector': feat['selected_by_optional_selector'],
                    'classifier_coefficient': feat['classifier_coefficient'],
                    'coefficient_sign': feat['coefficient_sign'],
                    'coefficient_available': feat['coefficient_available'],
                    'coefficient_nonzero': feat['coefficient_nonzero'],
                    'selected_by_classifier': feat['selected_by_classifier'],
                    'final_active_feature': feat['final_active_feature'],
                })

        # 保存 subgroup errors
        for orig_label in [1, 2, 3]:
            entry = outer_result['subgroup_errors'][f'class_{orig_label}']
            all_subgroup_errors.append({
                'outer_fold': fold_id,
                'original_class': orig_label,
                'n': entry['n'],
                'tuned_pred_pos_count': entry['tuned_pred_pos_count'],
                'tuned_pred_pos_rate': entry['tuned_pred_pos_rate'],
                'default_pred_pos_count': entry['default_pred_pos_count'],
                'default_pred_pos_rate': entry['default_pred_pos_rate'],
                'tuned_recall': entry.get('tuned_recall'),
                'default_recall': entry.get('default_recall'),
            })

        # 保存阈值信息
        all_thresholds.append({
            'outer_fold': fold_id,
            'threshold': outer_result['selected_threshold'],
            'default_threshold': outer_result['default_threshold'],
            'score_type': outer_result['score_type'],
            'inner_predicted_positive_rate': best.get('inner_predicted_positive_rate'),
            'outer_predicted_positive_rate': outer_result['tuned_metrics']['predicted_positive_rate'],
        })

        # 基线
        baselines = evaluate_baselines(bundle, outer_fold)
        for bl in baselines:
            bl['outer_fold'] = fold_id
        all_baseline_results.extend(baselines)

        print(f"  Tuned:   F1={outer_result['tuned_metrics']['f1']:.4f}, "
              f"MCC={outer_result['tuned_metrics']['mcc']:.4f}, "
              f"BAcc={outer_result['tuned_metrics']['balanced_accuracy']:.4f}")
        print(f"  Default: F1={outer_result['default_metrics']['f1']:.4f}, "
              f"MCC={outer_result['default_metrics']['mcc']:.4f}, "
              f"BAcc={outer_result['default_metrics']['balanced_accuracy']:.4f}")

    # ============================================================
    # 汇总
    # ============================================================
    total_runtime = time.time() - start_time

    # Pooled OOF metrics
    all_y_true = [r['y_test'] for r in all_outer_results]
    all_y_pred_tuned = [r['y_pred_tuned'] for r in all_outer_results]
    all_y_pred_default = [r['y_pred_default'] for r in all_outer_results]
    all_scores = [r['test_scores'] for r in all_outer_results]

    pooled_tuned = compute_binary_metrics(
        np.concatenate(all_y_true),
        np.concatenate(all_y_pred_tuned),
        np.concatenate(all_scores),
    )
    pooled_default = compute_binary_metrics(
        np.concatenate(all_y_true),
        np.concatenate(all_y_pred_default),
        np.concatenate(all_scores),
    )

    # Fold-wise mean/std
    fold_tuned_metrics = [r['tuned_metrics'] for r in all_outer_results]
    fold_default_metrics = [r['default_metrics'] for r in all_outer_results]
    foldwise_mean_tuned, foldwise_std_tuned = compute_foldwise_mean_std(fold_tuned_metrics)
    foldwise_mean_default, foldwise_std_default = compute_foldwise_mean_std(fold_default_metrics)

    # Baseline pooled
    baseline_pooled = {}
    for bl_name in ['DummyMostFrequent', 'DummyStratified', 'AllPositive']:
        bl_metrics = [b['metrics'] for b in all_baseline_results if b['baseline_name'] == bl_name]
        if bl_metrics:
            baseline_pooled[bl_name] = compute_foldwise_mean_std(bl_metrics)[0]

    # Model family selection frequency
    family_counts = defaultdict(int)
    for r in all_outer_results:
        family_counts[r['selected_model_family']] += 1

    # Threshold stats
    thresholds = [r['selected_threshold'] for r in all_outer_results]
    threshold_stats = {
        'min': float(np.min(thresholds)),
        'max': float(np.max(thresholds)),
        'mean': float(np.mean(thresholds)),
        'median': float(np.median(thresholds)),
        'std': float(np.std(thresholds, ddof=1)) if len(thresholds) > 1 else 0.0,
    }

    # 汇总结果
    aggregate = {
        'pooled_oof_metrics': {
            'tuned': pooled_tuned,
            'default': pooled_default,
        },
        'foldwise_mean_metrics': {
            'tuned': foldwise_mean_tuned,
            'default': foldwise_mean_default,
        },
        'foldwise_std_metrics': {
            'tuned': foldwise_std_tuned,
            'default': foldwise_std_default,
        },
        'baseline_pooled_oof_metrics': baseline_pooled,
        'model_family_selection_frequency': dict(family_counts),
        'threshold_stats': threshold_stats,
        'evidence_limitations': {
            'subject_independence_unverified': True,
            'batch_confounding_cannot_be_excluded': True,
            'internal_validation_only': True,
        },
        'is_quick_run': mode == 'quick',
        'formal_evaluation': mode == 'full',
        'smoke_test_only': mode == 'quick',
        'n_outer_folds': len(all_outer_results),
        'n_candidates': len(candidates),
        'n_invalid_candidates': sum(
            1 for r in all_inner_results if r['status'] == 'invalid'
        ),
        'runtime': total_runtime,
    }
    if mode == 'quick':
        aggregate['canonical_outer_results_partially_exposed'] = True
        aggregate['exposed_outer_folds'] = outer_folds
        aggregate['must_not_be_used_for_model_or_protocol_changes'] = True
        aggregate['frozen_config_hash'] = frozen_config_hash
    if frozen_config_hash:
        aggregate['frozen_config_hash'] = frozen_config_hash

    # ============================================================
    # 保存文件
    # ============================================================
    _save_outputs(
        output_dir, mode, bundle, outer_manifest,
        all_oof_predictions, all_outer_results, aggregate,
        all_inner_results, all_selected_features,
        all_subgroup_errors, all_thresholds,
        start_datetime, total_runtime, len(candidates),
        sum(1 for r in all_inner_results if r['status'] == 'invalid'),
    )

    print(f"\n{'='*60}")
    print(f" Nested CV Complete")
    print(f"{'='*60}")
    print(f"Pooled OOF Tuned F1: {pooled_tuned['f1']:.4f}")
    print(f"Pooled OOF Default F1: {pooled_default['f1']:.4f}")
    print(f"Runtime: {total_runtime:.1f}s")
    print(f"Output: {output_dir}")

    return aggregate


def _save_outputs(
    output_dir, mode, bundle, outer_manifest,
    oof_predictions, outer_results, aggregate,
    inner_results, selected_features,
    subgroup_errors, thresholds,
    start_datetime, total_runtime, n_candidates, n_invalid,
):
    """保存所有输出文件"""
    import json as _json

    # 1. oof_predictions.csv
    df_pred = pd.DataFrame(oof_predictions)
    df_pred.to_csv(os.path.join(output_dir, 'oof_predictions.csv'), index=False)
    print(f"  Saved: oof_predictions.csv ({len(df_pred)} rows)")

    # 2. outer_fold_metrics.csv
    outer_metrics_rows = []
    for r in outer_results:
        row = {
            'outer_fold': r['outer_fold_id'],
            'selected_candidate_id': r['selected_candidate_id'],
            'selected_model_family': r['selected_model_family'],
            'selected_threshold': r['selected_threshold'],
            'score_type': r['score_type'],
            'default_threshold': r['default_threshold'],
            'n_train': r['n_train'],
            'n_test': r['n_test'],
            'train_neg': r['train_counts']['neg'],
            'train_pos': r['train_counts']['pos'],
            'test_neg': r['test_counts']['neg'],
            'test_pos': r['test_counts']['pos'],
            'n_before_voc_filter': r['feature_info']['n_before_voc_filter'],
            'n_after_voc_filter': r['feature_info']['n_after_voc_filter'],
            'n_after_selector': r['feature_info']['n_after_selector'],
            'fit_warnings': r['fit_warnings'],
            'tuned_f1': r['tuned_metrics']['f1'],
            'tuned_mcc': r['tuned_metrics']['mcc'],
            'tuned_balanced_accuracy': r['tuned_metrics']['balanced_accuracy'],
            'tuned_roc_auc': r['tuned_metrics']['roc_auc'],
            'tuned_pr_auc': r['tuned_metrics']['pr_auc'],
            'default_f1': r['default_metrics']['f1'],
            'default_mcc': r['default_metrics']['mcc'],
            'default_balanced_accuracy': r['default_metrics']['balanced_accuracy'],
        }
        outer_metrics_rows.append(row)
    df_outer = pd.DataFrame(outer_metrics_rows)
    df_outer.to_csv(os.path.join(output_dir, 'outer_fold_metrics.csv'), index=False)
    print(f"  Saved: outer_fold_metrics.csv")

    # 3. aggregate_metrics.json
    with open(os.path.join(output_dir, 'aggregate_metrics.json'), 'w') as f:
        _json.dump(aggregate, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: aggregate_metrics.json")

    # 4. inner_candidate_results.csv
    df_inner = pd.DataFrame(inner_results)
    df_inner.to_csv(os.path.join(output_dir, 'inner_candidate_results.csv'), index=False)
    print(f"  Saved: inner_candidate_results.csv ({len(df_inner)} rows)")

    # 5. selected_configs.json
    selected_configs = []
    for r in outer_results:
        selected_configs.append({
            'outer_fold': r['outer_fold_id'],
            'selected_candidate_id': r['selected_candidate_id'],
            'selected_model_family': r['selected_model_family'],
            'threshold': r['selected_threshold'],
            'score_type': r['score_type'],
            'tuned_f1': r['tuned_metrics']['f1'],
            'tuned_mcc': r['tuned_metrics']['mcc'],
        })
    with open(os.path.join(output_dir, 'selected_configs.json'), 'w') as f:
        _json.dump(selected_configs, f, indent=2, ensure_ascii=False)
    print(f"  Saved: selected_configs.json")

    # 6. thresholds.csv
    df_thresh = pd.DataFrame(thresholds)
    df_thresh.to_csv(os.path.join(output_dir, 'thresholds.csv'), index=False)
    print(f"  Saved: thresholds.csv")

    # 7. selected_features.csv
    df_feat = pd.DataFrame(selected_features)
    df_feat.to_csv(os.path.join(output_dir, 'selected_features.csv'), index=False)
    print(f"  Saved: selected_features.csv ({len(df_feat)} rows)")

    # 8. subgroup_errors.csv
    df_sub = pd.DataFrame(subgroup_errors)
    df_sub.to_csv(os.path.join(output_dir, 'subgroup_errors.csv'), index=False)
    print(f"  Saved: subgroup_errors.csv")

    # 9. run_metadata.json
    import sys, platform
    import sklearn as _sklearn
    metadata = {
        'git_commit': 'bbe61de',
        'git_dirty': True,
        'dataset_fingerprint': bundle.dataset_fingerprint,
        'outer_manifest_hash': outer_manifest['manifest_hash'],
        'python_version': sys.version,
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'sklearn_version': _sklearn.__version__,
        'command': ' '.join(sys.argv),
        'mode': mode,
        'formal_evaluation': mode == 'full',
        'smoke_test_only': mode == 'quick',
        'start_time': start_datetime,
        'end_time': datetime.now(timezone.utc).isoformat(),
        'runtime': total_runtime,
        'candidate_count': n_candidates,
        'invalid_candidate_count': n_invalid,
        'is_quick_run': mode == 'quick',
    }
    # 计算 frozen config hash
    try:
        frozen_cfg_hash = _compute_frozen_config_hash()
        metadata['frozen_config_hash'] = frozen_cfg_hash
        metadata['frozen_config_match'] = _detect_frozen_config_mismatch(
            bundle, outer_manifest
        )['match']
    except FileNotFoundError:
        metadata['frozen_config_hash'] = None
        metadata['frozen_config_match'] = None

    if mode == 'quick':
        metadata['canonical_outer_results_partially_exposed'] = True
        metadata['exposed_outer_folds'] = [0, 1]
        metadata['must_not_be_used_for_model_or_protocol_changes'] = True
    with open(os.path.join(output_dir, 'run_metadata.json'), 'w') as f:
        _json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Saved: run_metadata.json")