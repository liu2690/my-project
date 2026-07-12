#!/usr/bin/env python3
"""
test_candidate_registry.py — 测试 candidate_registry.py
"""

import os
import sys
import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from candidate_registry import (
    generate_candidates,
    make_candidate_id,
    parse_candidate_id,
    build_candidate_pipeline,
    get_candidate_counts,
)
from models_sklearn import (
    get_pipeline_score_type,
    get_default_threshold,
)


class TestCandidateId:
    def test_id_unique(self):
        full = generate_candidates('full')
        ids = [c['candidate_id'] for c in full]
        assert len(ids) == len(set(ids))

    def test_id_quick_unique(self):
        quick = generate_candidates('quick')
        ids = [c['candidate_id'] for c in quick]
        assert len(ids) == len(set(ids))

    def test_id_format_elastic_net(self):
        cid = make_candidate_id('elastic_net', {'C': 0.1, 'l1_ratio': 0.5, 'class_weight': 'none'})
        assert cid == 'elastic_net__C=0.1__l1=0.5__cw=none'

    def test_id_format_linear_svm(self):
        cid = make_candidate_id('linear_svm', {'C': 1.0, 'class_weight': 'none', 'k': 50})
        assert cid == 'linear_svm__C=1.0__cw=none__k=50'

    def test_id_format_rbf_svm(self):
        cid = make_candidate_id('rbf_svm', {'C': 1.0, 'gamma': 'scale', 'class_weight': 'none', 'k': 'all'})
        assert cid == 'rbf_svm__C=1.0__gamma=scale__cw=none__k=all'

    def test_id_format_lda(self):
        cid = make_candidate_id('lda', {'shrinkage': 'auto', 'k': 'all'})
        assert cid == 'lda__shrinkage=auto__k=all'


class TestCandidateCounts:
    def test_full_counts(self):
        counts = get_candidate_counts()
        fc = counts['full']
        assert fc['elastic_net'] == 24  # 4*3*2
        assert fc['linear_svm'] == 32  # 4*2*4
        assert fc['rbf_svm'] == 72  # 3*3*2*4
        assert fc['lda'] == 16  # 4*4
        assert fc['total'] == 144

    def test_quick_counts(self):
        counts = get_candidate_counts()
        qc = counts['quick']
        assert qc['elastic_net'] == 4  # 2*1*2
        assert qc['linear_svm'] == 8  # 2*2*2
        assert qc['rbf_svm'] == 1  # 1*1*1*1
        assert qc['lda'] == 2  # 1*2
        assert qc['total'] == 15


class TestPipelineBuild:
    def test_elastic_net_no_selector(self):
        candidates = generate_candidates('full')
        en_cands = [c for c in candidates if c['model_family'] == 'elastic_net']
        pipe = en_cands[0]['build_pipeline']()
        assert 'selector' not in pipe.named_steps
        assert 'classifier' in pipe.named_steps

    def test_linear_svm_has_selector(self):
        candidates = generate_candidates('full')
        svm_cands = [c for c in candidates if c['model_family'] == 'linear_svm' and c['params'].get('k') != 'all']
        if svm_cands:
            pipe = svm_cands[0]['build_pipeline']()
            assert 'selector' in pipe.named_steps

    def test_linear_svm_k_all_no_selector(self):
        candidates = generate_candidates('full')
        svm_cands = [c for c in candidates if c['model_family'] == 'linear_svm' and c['params'].get('k') == 'all']
        if svm_cands:
            pipe = svm_cands[0]['build_pipeline']()
            assert 'selector' not in pipe.named_steps

    def test_all_pipelines_cloneable(self):
        candidates = generate_candidates('quick')
        for c in candidates:
            pipe = c['build_pipeline']()
            cloned = deepcopy(pipe)
            assert cloned is not pipe

    def test_score_type_recognized(self):
        candidates = generate_candidates('quick')
        for c in candidates:
            st = c['score_type']
            assert st in ['probability', 'decision']

    def test_pipeline_order(self):
        """Pipeline 步骤顺序正确"""
        candidates = generate_candidates('quick')
        for c in candidates:
            pipe = c['build_pipeline']()
            step_names = list(pipe.named_steps.keys())
            assert 'imputer' in step_names
            assert 'voc_filter' in step_names
            assert 'scaler' in step_names
            assert 'classifier' in step_names
            imputer_idx = step_names.index('imputer')
            filter_idx = step_names.index('voc_filter')
            scaler_idx = step_names.index('scaler')
            clf_idx = step_names.index('classifier')
            assert imputer_idx < filter_idx < scaler_idx < clf_idx


class TestParseCandidateId:
    def test_roundtrip(self):
        candidates = generate_candidates('quick')
        for c in candidates:
            parsed = parse_candidate_id(c['candidate_id'])
            assert parsed['model_family'] == c['model_family']


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-q'])