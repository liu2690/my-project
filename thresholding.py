#!/usr/bin/env python3
"""
thresholding.py — 阶段二：候选特定的阈值联合选择

每个候选根据自己在 outer-train 上的 inner OOF score 独立选择阈值。
阈值选择不接触 outer-test 数据。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from metrics import compute_binary_metrics


@dataclass
class ThresholdSelectionResult:
    """阈值选择结果"""
    threshold: float
    score_type: str  # 'probability' or 'decision'
    default_threshold: float  # 0.5 for proba, 0.0 for decision

    # 调优阈值指标
    f1: float
    mcc: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float
    accuracy: float

    # 默认阈值指标
    default_f1: float
    default_mcc: float
    default_balanced_accuracy: float
    default_precision: float
    default_recall: float
    default_specificity: float
    default_accuracy: float

    # PR-AUC / ROC-AUC (基于连续 score，不随阈值变化)
    pr_auc: Optional[float]
    roc_auc: Optional[float]

    predicted_positive_count: int
    predicted_positive_rate: float

    # 样本数
    n_samples: int
    n_positive: int
    n_negative: int

    def to_dict(self) -> dict:
        return {
            'threshold': self.threshold,
            'score_type': self.score_type,
            'default_threshold': self.default_threshold,
            'f1': self.f1,
            'mcc': self.mcc,
            'balanced_accuracy': self.balanced_accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'specificity': self.specificity,
            'accuracy': self.accuracy,
            'default_f1': self.default_f1,
            'default_mcc': self.default_mcc,
            'default_balanced_accuracy': self.default_balanced_accuracy,
            'default_precision': self.default_precision,
            'default_recall': self.default_recall,
            'default_specificity': self.default_specificity,
            'default_accuracy': self.default_accuracy,
            'pr_auc': self.pr_auc,
            'roc_auc': self.roc_auc,
            'predicted_positive_count': self.predicted_positive_count,
            'predicted_positive_rate': self.predicted_positive_rate,
            'n_samples': self.n_samples,
            'n_positive': self.n_positive,
            'n_negative': self.n_negative,
        }


def select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    score_type: str = 'probability',
) -> ThresholdSelectionResult:
    """为单个候选选择最优阈值。

    输入只能是当前候选在 outer-train 上的 inner OOF:
        - y_true: 真实标签
        - scores: OOF prediction scores

    不得传入 outer-test 数据。

    Args:
        y_true: 真实标签 (n_samples,)
        scores: 连续 score (n_samples,)
        score_type: 'probability' (默认阈值 0.5) 或 'decision' (默认阈值 0.0)

    Returns:
        ThresholdSelectionResult
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)

    n = len(y_true)

    if n == 0:
        raise ValueError("Empty y_true/scores")

    # 检查 NaN
    if np.any(np.isnan(scores)):
        raise ValueError("NaN in scores")

    unique_y = np.unique(y_true)
    if len(unique_y) < 2:
        # 单一类别：所有候选阈值等效
        default_threshold = 0.5 if score_type == 'probability' else 0.0
        y_pred = _apply_threshold(scores, default_threshold, score_type)
        metrics = compute_binary_metrics(y_true, y_pred, scores)
        return ThresholdSelectionResult(
            threshold=default_threshold,
            score_type=score_type,
            default_threshold=default_threshold,
            f1=metrics['f1'] or 0.0,
            mcc=metrics['mcc'] or 0.0,
            balanced_accuracy=metrics['balanced_accuracy'] or 0.0,
            precision=metrics['precision'] or 0.0,
            recall=metrics['recall'] or 0.0,
            specificity=metrics['specificity'] or 0.0,
            accuracy=metrics['accuracy'] or 0.0,
            default_f1=metrics['f1'] or 0.0,
            default_mcc=metrics['mcc'] or 0.0,
            default_balanced_accuracy=metrics['balanced_accuracy'] or 0.0,
            default_precision=metrics['precision'] or 0.0,
            default_recall=metrics['recall'] or 0.0,
            default_specificity=metrics['specificity'] or 0.0,
            default_accuracy=metrics['accuracy'] or 0.0,
            pr_auc=metrics['pr_auc'],
            roc_auc=metrics['roc_auc'],
            predicted_positive_count=metrics['predicted_positive_count'],
            predicted_positive_rate=metrics['predicted_positive_rate'],
            n_samples=n,
            n_positive=int((y_true == 1).sum()),
            n_negative=int((y_true == 0).sum()),
        )

    default_threshold = 0.5 if score_type == 'probability' else 0.0

    # 构建候选阈值列表
    # 从排序后的 score 边界生成候选阈值
    sorted_scores = np.sort(scores)
    candidate_thresholds = _generate_threshold_candidates(sorted_scores, score_type)

    best_result = None
    best_f1 = -1.0
    best_mcc = -1.0
    best_balanced_acc = -1.0
    best_distance = float('inf')

    for threshold in candidate_thresholds:
        y_pred = _apply_threshold(scores, threshold, score_type)
        metrics = compute_binary_metrics(y_true, y_pred, scores)

        f1 = metrics['f1'] or 0.0
        mcc = metrics['mcc'] or 0.0
        balanced_acc = metrics['balanced_accuracy'] or 0.0

        better = False

        if f1 > best_f1:
            better = True
        elif f1 == best_f1 and mcc > best_mcc:
            better = True
        elif f1 == best_f1 and mcc == best_mcc and balanced_acc > best_balanced_acc:
            better = True
        elif f1 == best_f1 and mcc == best_mcc and balanced_acc == best_balanced_acc:
            dist = abs(threshold - default_threshold)
            if dist < best_distance:
                better = True

        if better:
            best_f1 = f1
            best_mcc = mcc
            best_balanced_acc = balanced_acc
            best_distance = abs(threshold - default_threshold)
            best_result = ThresholdSelectionResult(
                threshold=threshold,
                score_type=score_type,
                default_threshold=default_threshold,
                f1=f1,
                mcc=mcc,
                balanced_accuracy=balanced_acc,
                precision=metrics['precision'] or 0.0,
                recall=metrics['recall'] or 0.0,
                specificity=metrics['specificity'] or 0.0,
                accuracy=metrics['accuracy'] or 0.0,
                default_f1=0.0,  # 稍后填充
                default_mcc=0.0,
                default_balanced_accuracy=0.0,
                default_precision=0.0,
                default_recall=0.0,
                default_specificity=0.0,
                default_accuracy=0.0,
                pr_auc=metrics['pr_auc'],
                roc_auc=metrics['roc_auc'],
                predicted_positive_count=metrics['predicted_positive_count'],
                predicted_positive_rate=metrics['predicted_positive_rate'],
                n_samples=n,
                n_positive=int((y_true == 1).sum()),
                n_negative=int((y_true == 0).sum()),
            )

    if best_result is None:
        raise RuntimeError("Threshold selection failed to find any valid threshold")

    # 填充默认阈值指标
    y_pred_default = _apply_threshold(scores, default_threshold, score_type)
    default_metrics = compute_binary_metrics(y_true, y_pred_default, scores)
    best_result.default_f1 = default_metrics['f1'] or 0.0
    best_result.default_mcc = default_metrics['mcc'] or 0.0
    best_result.default_balanced_accuracy = default_metrics['balanced_accuracy'] or 0.0
    best_result.default_precision = default_metrics['precision'] or 0.0
    best_result.default_recall = default_metrics['recall'] or 0.0
    best_result.default_specificity = default_metrics['specificity'] or 0.0
    best_result.default_accuracy = default_metrics['accuracy'] or 0.0

    return best_result


def _apply_threshold(scores: np.ndarray, threshold: float, score_type: str) -> np.ndarray:
    """根据阈值生成二值预测。

    Args:
        scores: 连续 score
        threshold: 阈值
        score_type: 'probability' or 'decision'

    Returns:
        y_pred (0/1)
    """
    if score_type == 'probability':
        return (scores > threshold).astype(int)
    elif score_type == 'decision':
        return (scores > threshold).astype(int)
    else:
        raise ValueError(f"Unknown score_type: {score_type}")


def _generate_threshold_candidates(
    sorted_scores: np.ndarray,
    score_type: str,
) -> List[float]:
    """从排序 score 边界生成候选阈值列表。

    覆盖:
    - 全预测负类 (max_score + epsilon)
    - 全预测正类 (min_score - epsilon)
    - 每个有意义的预测变化位置 (相邻 score 中点)

    Args:
        sorted_scores: 排序后的 score
        score_type: 'probability' or 'decision'

    Returns:
        候选阈值列表
    """
    n = len(sorted_scores)
    if n == 0:
        return [0.5] if score_type == 'probability' else [0.0]

    candidates = []

    # 全预测正类: 阈值低于最小 score
    min_score = sorted_scores[0]
    epsilon = 1e-9
    all_positive_threshold = float(min_score - max(abs(min_score) * epsilon, epsilon))
    candidates.append(all_positive_threshold)

    # 全预测负类: 阈值高于最大 score
    max_score = sorted_scores[-1]
    all_negative_threshold = float(max_score + max(abs(max_score) * epsilon, epsilon))
    candidates.append(all_negative_threshold)

    # 默认阈值
    default = 0.5 if score_type == 'probability' else 0.0
    candidates.append(default)

    # 每个相邻 score 之间的中点
    for i in range(n - 1):
        low = float(sorted_scores[i])
        high = float(sorted_scores[i + 1])
        if high > low:
            mid = (low + high) / 2.0
            candidates.append(mid)

    # 去重并排序
    candidates = sorted(set(round(c, 12) for c in candidates))
    return candidates