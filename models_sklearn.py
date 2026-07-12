#!/usr/bin/env python3
"""
models_sklearn.py — 阶段二：传统模型定义

定义所有候选模型族和 Pipeline 构建。
不包含候选注册逻辑（见 candidate_registry.py）。
"""

import warnings
from typing import Optional, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from voc_preprocessing import VOCAbundanceIQRFilter


# ============================================================
# Pipeline 基础组件
# ============================================================
def make_base_pipeline(classifier) -> Pipeline:
    """构建基础 Pipeline: Imputer → VOCFilter → Scaler → [optional SelectKBest] → Classifier

    SelectKBest 需要在外部添加，因为不同模型族可能有不同配置。
    """
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('voc_filter', VOCAbundanceIQRFilter(
            abundance_percentile=40.0,
            iqr_percentile=25.0,
            apply_log1p=True,
        )),
        ('scaler', StandardScaler()),
    ]
    # 临时占位，classifier 将在调用方添加
    return Pipeline(steps, memory=None)


def add_selector_to_pipeline(
    base_pipe: Pipeline,
    classifier,
    k: Optional[int] = None,
) -> Pipeline:
    """在 Scaler 后添加 SelectKBest (可选) 和 classifier。

    Args:
        base_pipe: 基础 Pipeline (imputer → voc_filter → scaler)
        classifier: sklearn 分类器
        k: SelectKBest 的 k 值，None 或 'all' 表示不使用 SelectKBest

    Returns:
        完整 Pipeline
    """
    steps = list(base_pipe.steps)
    if k is not None and k != 'all':
        steps.append(('selector', SelectKBest(score_func=f_classif, k=k)))
    steps.append(('classifier', classifier))
    return Pipeline(steps, memory=None)


# ============================================================
# Elastic Net Logistic Regression
# ============================================================
def build_elastic_net_pipeline(
    C: float = 1.0,
    l1_ratio: float = 0.5,
    class_weight: Optional[str] = None,
) -> Pipeline:
    """Elastic Net Logistic Regression Pipeline。

    不使用额外 SelectKBest，依靠 Elastic Net 自身 L1/L2 正则。
    """
    clf = LogisticRegression(
        solver='saga',
        penalty='elasticnet',
        C=C,
        l1_ratio=l1_ratio,
        class_weight=class_weight if class_weight != 'none' else None,
        max_iter=50000,
        tol=1e-4,
        random_state=42,
    )
    base = make_base_pipeline(None)
    steps = list(base.steps) + [('classifier', clf)]
    return Pipeline(steps, memory=None)


def get_elastic_net_score_type() -> str:
    return 'probability'


def get_elastic_net_score(pipeline: Pipeline, X) -> "np.ndarray":
    """获取 predict_proba[:, 1]"""
    return pipeline.predict_proba(X)[:, 1]


# ============================================================
# Linear SVM
# ============================================================
def build_linear_svm_pipeline(
    C: float = 1.0,
    class_weight: Optional[str] = None,
    k=None,
) -> Pipeline:
    """Linear SVM Pipeline。

    Args:
        C: 正则化参数
        class_weight: None 或 'balanced'
        k: SelectKBest k 值，None 或 'all' 表示不使用
    """
    clf = LinearSVC(
        C=C,
        class_weight=class_weight if class_weight != 'none' else None,
        max_iter=50000,
        dual=True,
        random_state=42,
    )
    base = make_base_pipeline(None)
    return add_selector_to_pipeline(base, clf, k)


def get_linear_svm_score_type() -> str:
    return 'decision'


def get_linear_svm_score(pipeline: Pipeline, X) -> "np.ndarray":
    """获取 decision_function"""
    return pipeline.decision_function(X)


# ============================================================
# RBF SVM
# ============================================================
def build_rbf_svm_pipeline(
    C: float = 1.0,
    gamma: str = 'scale',
    class_weight: Optional[str] = None,
    k=None,
) -> Pipeline:
    """RBF SVM Pipeline。

    不使用 probability=True。
    """
    clf = SVC(
        kernel='rbf',
        C=C,
        gamma=gamma,
        class_weight=class_weight if class_weight != 'none' else None,
        probability=False,
        random_state=42,
    )
    base = make_base_pipeline(None)
    return add_selector_to_pipeline(base, clf, k)


def get_rbf_svm_score_type() -> str:
    return 'decision'


def get_rbf_svm_score(pipeline: Pipeline, X) -> "np.ndarray":
    """获取 decision_function"""
    return pipeline.decision_function(X)


# ============================================================
# Shrinkage LDA
# ============================================================
def build_lda_pipeline(
    shrinkage=None,
    k=None,
) -> Pipeline:
    """Shrinkage LDA Pipeline。

    Args:
        shrinkage: 'auto', 0.1, 0.5, 0.9, or None
        k: SelectKBest k 值
    """
    clf = LinearDiscriminantAnalysis(
        solver='lsqr',
        shrinkage=shrinkage,
    )
    base = make_base_pipeline(None)
    return add_selector_to_pipeline(base, clf, k)


def get_lda_score_type() -> str:
    return 'probability'


def get_lda_score(pipeline: Pipeline, X) -> "np.ndarray":
    """获取 predict_proba[:, 1]"""
    return pipeline.predict_proba(X)[:, 1]


# ============================================================
# Score 获取统一接口
# ============================================================
SCORE_FUNCTIONS = {
    'elastic_net': {
        'type': 'probability',
        'func': get_elastic_net_score,
        'default_threshold': 0.5,
    },
    'linear_svm': {
        'type': 'decision',
        'func': get_linear_svm_score,
        'default_threshold': 0.0,
    },
    'rbf_svm': {
        'type': 'decision',
        'func': get_rbf_svm_score,
        'default_threshold': 0.0,
    },
    'lda': {
        'type': 'probability',
        'func': get_lda_score,
        'default_threshold': 0.5,
    },
}


def get_score_info(model_family: str) -> Dict[str, Any]:
    """获取模型族的 score 信息。

    Returns:
        dict with keys: type, func, default_threshold
    """
    if model_family not in SCORE_FUNCTIONS:
        raise ValueError(f"Unknown model family: {model_family}")
    return SCORE_FUNCTIONS[model_family]


def get_pipeline_score(pipeline: Pipeline, X, model_family: str) -> "np.ndarray":
    """统一获取 pipeline 的连续 score。"""
    return SCORE_FUNCTIONS[model_family]['func'](pipeline, X)


def get_pipeline_score_type(model_family: str) -> str:
    return SCORE_FUNCTIONS[model_family]['type']


def get_default_threshold(model_family: str) -> float:
    return SCORE_FUNCTIONS[model_family]['default_threshold']