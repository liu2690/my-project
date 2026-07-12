#!/usr/bin/env python3
"""
candidate_registry.py — 阶段二：候选注册模块

生成所有候选配置，每个候选拥有稳定、可读且唯一的 candidate_id。
参数顺序固定，不依赖 Python dict 的偶然输出形式。
"""

from typing import List, Dict, Any, Optional
from copy import deepcopy
from sklearn.pipeline import Pipeline

from models_sklearn import (
    build_elastic_net_pipeline,
    build_linear_svm_pipeline,
    build_rbf_svm_pipeline,
    build_lda_pipeline,
    get_score_info,
)


# ============================================================
# Candidate ID 生成
# ============================================================
def make_candidate_id(
    model_family: str,
    params: Dict[str, Any],
) -> str:
    """生成稳定的 candidate_id。

    Args:
        model_family: 'elastic_net', 'linear_svm', 'rbf_svm', 'lda'
        params: 参数字典

    Returns:
        candidate_id 字符串
    """
    parts = [model_family]

    # 固定参数顺序
    if model_family == 'elastic_net':
        parts.append(f"C={params['C']}")
        parts.append(f"l1={params['l1_ratio']}")
        parts.append(f"cw={params.get('class_weight', 'none')}")
    elif model_family == 'linear_svm':
        parts.append(f"C={params['C']}")
        parts.append(f"cw={params.get('class_weight', 'none')}")
        parts.append(f"k={params.get('k', 'all')}")
    elif model_family == 'rbf_svm':
        parts.append(f"C={params['C']}")
        parts.append(f"gamma={params['gamma']}")
        parts.append(f"cw={params.get('class_weight', 'none')}")
        parts.append(f"k={params.get('k', 'all')}")
    elif model_family == 'lda':
        parts.append(f"shrinkage={params['shrinkage']}")
        parts.append(f"k={params.get('k', 'all')}")
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    return '__'.join(parts)


def parse_candidate_id(candidate_id: str) -> Dict[str, Any]:
    """从 candidate_id 解析参数。

    Returns:
        dict with keys: model_family, params
    """
    parts = candidate_id.split('__')
    model_family = parts[0]
    params = {}

    for part in parts[1:]:
        key, val = part.split('=', 1)
        if key == 'k':
            if val == 'all':
                params['k'] = 'all'
            else:
                params['k'] = int(val)
        elif key == 'C':
            params['C'] = float(val)
        elif key == 'l1':
            params['l1_ratio'] = float(val)
        elif key == 'cw':
            params['class_weight'] = val if val != 'none' else None
        elif key == 'gamma':
            params['gamma'] = val
        elif key == 'shrinkage':
            if val == 'auto':
                params['shrinkage'] = 'auto'
            else:
                params['shrinkage'] = float(val)

    return {'model_family': model_family, 'params': params}


# ============================================================
# Pipeline 构建
# ============================================================
def build_candidate_pipeline(
    model_family: str,
    params: Dict[str, Any],
) -> Pipeline:
    """根据模型族和参数构建 sklearn Pipeline。

    Args:
        model_family: 模型族名称
        params: 参数字典

    Returns:
        sklearn Pipeline
    """
    if model_family == 'elastic_net':
        return build_elastic_net_pipeline(
            C=params.get('C', 1.0),
            l1_ratio=params.get('l1_ratio', 0.5),
            class_weight=params.get('class_weight'),
        )
    elif model_family == 'linear_svm':
        return build_linear_svm_pipeline(
            C=params.get('C', 1.0),
            class_weight=params.get('class_weight'),
            k=params.get('k'),
        )
    elif model_family == 'rbf_svm':
        return build_rbf_svm_pipeline(
            C=params.get('C', 1.0),
            gamma=params.get('gamma', 'scale'),
            class_weight=params.get('class_weight'),
            k=params.get('k'),
        )
    elif model_family == 'lda':
        return build_lda_pipeline(
            shrinkage=params.get('shrinkage'),
            k=params.get('k'),
        )
    else:
        raise ValueError(f"Unknown model family: {model_family}")


# ============================================================
# Candidate 注册
# ============================================================
def generate_candidates(mode: str = 'full') -> List[Dict[str, Any]]:
    """生成所有候选配置。

    Args:
        mode: 'full' 或 'quick'

    Returns:
        list of candidate dicts:
            {
                'candidate_id': str,
                'model_family': str,
                'params': dict,
                'build_pipeline': callable,
                'score_type': str,
                'default_threshold': float,
            }
    """
    candidates = []

    if mode == 'quick':
        candidates.extend(_elastic_net_quick())
        candidates.extend(_linear_svm_quick())
        candidates.extend(_rbf_svm_quick())
        candidates.extend(_lda_quick())
    elif mode == 'full':
        candidates.extend(_elastic_net_full())
        candidates.extend(_linear_svm_full())
        candidates.extend(_rbf_svm_full())
        candidates.extend(_lda_full())
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 验证 candidate_id 唯一性
    ids = [c['candidate_id'] for c in candidates]
    if len(ids) != len(set(ids)):
        from collections import Counter
        duplicates = [i for i, count in Counter(ids).items() if count > 1]
        raise ValueError(f"Duplicate candidate_ids: {duplicates}")

    return candidates


def _elastic_net_full() -> List[Dict]:
    candidates = []
    for C in [0.01, 0.1, 1.0, 10.0]:
        for l1_ratio in [0.25, 0.5, 0.75]:
            for cw in [None, 'balanced']:
                cw_str = 'none' if cw is None else cw
                params = {'C': C, 'l1_ratio': l1_ratio, 'class_weight': cw}
                cid = make_candidate_id('elastic_net', {**params, 'class_weight': cw_str})
                score_info = get_score_info('elastic_net')
                candidates.append({
                    'candidate_id': cid,
                    'model_family': 'elastic_net',
                    'params': params,
                    'build_pipeline': lambda p=params: build_candidate_pipeline('elastic_net', p),
                    'score_type': score_info['type'],
                    'default_threshold': score_info['default_threshold'],
                })
    return candidates


def _elastic_net_quick() -> List[Dict]:
    candidates = []
    for C in [0.1, 1.0]:
        for l1_ratio in [0.5]:
            for cw in [None, 'balanced']:
                cw_str = 'none' if cw is None else cw
                params = {'C': C, 'l1_ratio': l1_ratio, 'class_weight': cw}
                cid = make_candidate_id('elastic_net', {**params, 'class_weight': cw_str})
                score_info = get_score_info('elastic_net')
                candidates.append({
                    'candidate_id': cid,
                    'model_family': 'elastic_net',
                    'params': params,
                    'build_pipeline': lambda p=params: build_candidate_pipeline('elastic_net', p),
                    'score_type': score_info['type'],
                    'default_threshold': score_info['default_threshold'],
                })
    return candidates


def _linear_svm_full() -> List[Dict]:
    candidates = []
    for C in [0.01, 0.1, 1.0, 10.0]:
        for cw in [None, 'balanced']:
            for k in [20, 50, 100, 'all']:
                cw_str = 'none' if cw is None else cw
                params = {'C': C, 'class_weight': cw, 'k': k}
                cid = make_candidate_id('linear_svm', {**params, 'class_weight': cw_str})
                score_info = get_score_info('linear_svm')
                candidates.append({
                    'candidate_id': cid,
                    'model_family': 'linear_svm',
                    'params': params,
                    'build_pipeline': lambda p=params: build_candidate_pipeline('linear_svm', p),
                    'score_type': score_info['type'],
                    'default_threshold': score_info['default_threshold'],
                })
    return candidates


def _linear_svm_quick() -> List[Dict]:
    candidates = []
    for C in [0.1, 1.0]:
        for cw in [None, 'balanced']:
            for k in [50, 'all']:
                cw_str = 'none' if cw is None else cw
                params = {'C': C, 'class_weight': cw, 'k': k}
                cid = make_candidate_id('linear_svm', {**params, 'class_weight': cw_str})
                score_info = get_score_info('linear_svm')
                candidates.append({
                    'candidate_id': cid,
                    'model_family': 'linear_svm',
                    'params': params,
                    'build_pipeline': lambda p=params: build_candidate_pipeline('linear_svm', p),
                    'score_type': score_info['type'],
                    'default_threshold': score_info['default_threshold'],
                })
    return candidates


def _rbf_svm_full() -> List[Dict]:
    candidates = []
    for C in [0.1, 1.0, 10.0]:
        for gamma in ['scale', 0.01, 0.1]:
            for cw in [None, 'balanced']:
                for k in [20, 50, 100, 'all']:
                    cw_str = 'none' if cw is None else cw
                    params = {'C': C, 'gamma': gamma, 'class_weight': cw, 'k': k}
                    cid = make_candidate_id('rbf_svm', {**params, 'class_weight': cw_str})
                    score_info = get_score_info('rbf_svm')
                    candidates.append({
                        'candidate_id': cid,
                        'model_family': 'rbf_svm',
                        'params': params,
                        'build_pipeline': lambda p=params: build_candidate_pipeline('rbf_svm', p),
                        'score_type': score_info['type'],
                        'default_threshold': score_info['default_threshold'],
                    })
    return candidates


def _rbf_svm_quick() -> List[Dict]:
    candidates = []
    for C in [1.0]:
        for gamma in ['scale']:
            for cw in [None]:
                for k in [50]:
                    cw_str = 'none'
                    params = {'C': C, 'gamma': gamma, 'class_weight': cw, 'k': k}
                    cid = make_candidate_id('rbf_svm', {**params, 'class_weight': cw_str})
                    score_info = get_score_info('rbf_svm')
                    candidates.append({
                        'candidate_id': cid,
                        'model_family': 'rbf_svm',
                        'params': params,
                        'build_pipeline': lambda p=params: build_candidate_pipeline('rbf_svm', p),
                        'score_type': score_info['type'],
                        'default_threshold': score_info['default_threshold'],
                    })
    return candidates


def _lda_full() -> List[Dict]:
    candidates = []
    for shrinkage in ['auto', 0.1, 0.5, 0.9]:
        for k in [20, 50, 100, 'all']:
            params = {'shrinkage': shrinkage, 'k': k}
            cid = make_candidate_id('lda', params)
            score_info = get_score_info('lda')
            candidates.append({
                'candidate_id': cid,
                'model_family': 'lda',
                'params': params,
                'build_pipeline': lambda p=params: build_candidate_pipeline('lda', p),
                'score_type': score_info['type'],
                'default_threshold': score_info['default_threshold'],
            })
    return candidates


def _lda_quick() -> List[Dict]:
    candidates = []
    for shrinkage in ['auto']:
        for k in [50, 'all']:
            params = {'shrinkage': shrinkage, 'k': k}
            cid = make_candidate_id('lda', params)
            score_info = get_score_info('lda')
            candidates.append({
                'candidate_id': cid,
                'model_family': 'lda',
                'params': params,
                'build_pipeline': lambda p=params: build_candidate_pipeline('lda', p),
                'score_type': score_info['type'],
                'default_threshold': score_info['default_threshold'],
            })
    return candidates


def get_candidate_counts() -> Dict[str, int]:
    """返回各模式的候选数量统计"""
    full = generate_candidates('full')
    quick = generate_candidates('quick')
    counts = {}
    for mode, candidates in [('full', full), ('quick', quick)]:
        counts[mode] = {}
        for c in candidates:
            fam = c['model_family']
            counts[mode][fam] = counts[mode].get(fam, 0) + 1
        counts[mode]['total'] = len(candidates)
    return counts