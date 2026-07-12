#!/usr/bin/env python3
"""
metrics.py — 阶段二：统一二分类指标计算

所有后续 sklearn 和 MultiView 指标计算的唯一实现。
不在其他脚本中复制指标公式。
"""

import warnings
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)


def compute_binary_metrics(
    y_true,
    y_pred,
    scores=None,
    positive_label=1,
):
    """计算完整二分类指标集。

    Args:
        y_true: 真实标签，shape (n_samples,)
        y_pred: 预测标签，shape (n_samples,)
        scores: 连续 score (probability 或 decision)，shape (n_samples,)
                用于计算 ROC-AUC 和 PR-AUC
        positive_label: 正类标签 (默认 1)

    Returns:
        dict with keys:
            f1, precision, recall, sensitivity, specificity,
            accuracy, balanced_accuracy, mcc,
            roc_auc, pr_auc,
            tn, fp, fn, tp,
            predicted_positive_count, predicted_positive_rate,
            roc_auc_reason, pr_auc_reason (None if computed, str if skipped)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # 混淆矩阵 (固定 labels=[0,1])
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    except Exception:
        tn, fp, fn, tp = 0, 0, 0, 0

    n = len(y_true)

    # F1, Precision, Recall (zero_division=0)
    f1 = safe_metric(lambda: float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)))
    precision = safe_metric(lambda: float(precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)))
    recall = safe_metric(lambda: float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)))
    sensitivity = recall  # sensitivity == recall

    # Specificity
    specificity = safe_metric(lambda: _compute_specificity(tn, fp))

    # Accuracy
    accuracy = safe_metric(lambda: float(accuracy_score(y_true, y_pred)))

    # Balanced Accuracy
    balanced_accuracy = safe_metric(lambda: float(balanced_accuracy_score(y_true, y_pred)))

    # MCC
    mcc = safe_metric(lambda: float(matthews_corrcoef(y_true, y_pred)))

    # ROC-AUC and PR-AUC (require scores)
    roc_auc = None
    pr_auc = None
    roc_auc_reason = None
    pr_auc_reason = None

    if scores is not None:
        scores = np.asarray(scores, dtype=np.float64)
        # Check for NaN in scores
        if np.any(np.isnan(scores)):
            roc_auc_reason = "NaN_in_scores"
            pr_auc_reason = "NaN_in_scores"
        else:
            unique_y = np.unique(y_true)
            if len(unique_y) < 2:
                roc_auc_reason = "single_class_in_y_true"
                pr_auc_reason = "single_class_in_y_true"
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        roc_auc = float(roc_auc_score(y_true, scores))
                except Exception as e:
                    roc_auc_reason = f"computation_failed: {str(e)[:80]}"

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        pr_auc = float(average_precision_score(y_true, scores))
                except Exception as e:
                    pr_auc_reason = f"computation_failed: {str(e)[:80]}"

    # predicted positive count / rate
    predicted_positive_count = int((y_pred == positive_label).sum())
    predicted_positive_rate = float(predicted_positive_count / n) if n > 0 else 0.0

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'mcc': mcc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'roc_auc_reason': roc_auc_reason,
        'pr_auc_reason': pr_auc_reason,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'predicted_positive_count': predicted_positive_count,
        'predicted_positive_rate': predicted_positive_rate,
    }


def _compute_specificity(tn, fp):
    """计算 specificity = TN / (TN + FP)"""
    denom = tn + fp
    if denom == 0:
        return 0.0
    return float(tn) / float(denom)


def safe_metric(func):
    """安全执行指标计算，返回 None 而非崩溃"""
    try:
        return func()
    except Exception:
        return None


def compute_pooled_oof_metrics(all_y_true, all_y_pred, all_scores=None):
    """从 pooled OOF 预测计算 pooled 指标。

    Args:
        all_y_true: 所有 fold 的 y_true 拼接
        all_y_pred: 所有 fold 的 y_pred 拼接
        all_scores: 所有 fold 的 scores 拼接 (可选)

    Returns:
        dict: pooled metrics
    """
    return compute_binary_metrics(
        np.concatenate(all_y_true) if isinstance(all_y_true, list) else all_y_true,
        np.concatenate(all_y_pred) if isinstance(all_y_pred, list) else all_y_pred,
        np.concatenate(all_scores) if all_scores is not None and isinstance(all_scores, list) else all_scores,
    )


def compute_foldwise_mean_std(metrics_list):
    """从 fold-wise metrics dict 列表计算 mean 和 std。

    Args:
        metrics_list: list of dict (每个 fold 一个 metrics dict)

    Returns:
        tuple: (mean_dict, std_dict)
    """
    numeric_keys = [
        'f1', 'precision', 'recall', 'sensitivity', 'specificity',
        'accuracy', 'balanced_accuracy', 'mcc',
        'roc_auc', 'pr_auc',
        'tn', 'fp', 'fn', 'tp',
        'predicted_positive_count', 'predicted_positive_rate',
    ]

    mean_dict = {}
    std_dict = {}

    for key in numeric_keys:
        values = []
        for m in metrics_list:
            v = m.get(key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                values.append(float(v))
        if values:
            mean_dict[key] = float(np.mean(values))
            std_dict[key] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        else:
            mean_dict[key] = None
            std_dict[key] = None

    return mean_dict, std_dict