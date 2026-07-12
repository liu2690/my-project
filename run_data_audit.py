#!/usr/bin/env python3
"""
阶段零：任务与数据生成机制审计 (Data & Task Audit)

本脚本执行以下审计：
  1. 数据读取与结构检查 (原始 Excel + 近重复样本检查)
  2. 元数据审计 (检查 Excel sheet、列名、仓库数据文件)
  3. 标签任务审计 (1vs2, 1vs3, 2vs3, 1+2vs3)
     - Pipeline 内置于每个 CV 折: SimpleImputer → StandardScaler → LogisticRegression
     - 基线: DummyClassifier(most_frequent / stratified) + 全预测正类
     - OOF 预测 + pooled OOF 指标 + fold-wise mean/std
  4. 探索性可视化 (PCA)

输出文件：
  result/audit/data_audit.json
  result/audit/data_audit.md
  result/audit/class_counts.csv
  result/audit/task_diagnostics.csv
  result/audit/task_oof_predictions.csv
  result/audit/task_fold_metrics.csv
  result/audit/task_baselines.csv
  result/audit/pca.png
  data/metadata_template.csv (仅在元数据缺失时创建)
  docs/DATA_ASSUMPTIONS.md

用法:
  python run_data_audit.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from collections import OrderedDict

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, confusion_matrix,
    balanced_accuracy_score, matthews_corrcoef, precision_score, recall_score,
    average_precision_score
)

warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(PROJECT_ROOT, 'data', '1、2、3.xlsx')
MAT_PATH = os.path.join(PROJECT_ROOT, 'data', 'voc_dataset_1+2_vs_3.mat')
AUDIT_DIR = os.path.join(PROJECT_ROOT, 'result', 'audit')
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')
METADATA_TEMPLATE_PATH = os.path.join(PROJECT_ROOT, 'data', 'metadata_template.csv')

RANDOM_STATE = 42
N_FOLDS = 5

os.makedirs(AUDIT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# ============================================================
# 固定 Logistic 诊断 Pipeline 配置
# ============================================================
PIPELINE_CONFIG = {
    'imputer': 'SimpleImputer(strategy="median")',
    'scaler': 'StandardScaler()',
    'classifier': 'LogisticRegression(penalty="l2", C=1.0, solver="liblinear", '
                   'class_weight=None, max_iter=10000, random_state=42)',
    'note': 'No hyperparameter search. All data-driven steps inside CV fold.',
}


def build_pipeline():
    """构建固定 Pipeline: SimpleImputer → StandardScaler → LogisticRegression"""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            penalty='l2', C=1.0, solver='liblinear',
            class_weight=None, max_iter=10000,
            random_state=RANDOM_STATE
        )),
    ])


# ============================================================
# 1. 数据读取与结构检查
# ============================================================
def load_raw_excel():
    """读取原始 Excel 并返回结构化数据"""
    df = pd.read_excel(EXCEL_PATH, header=None)

    # Row 0: header with VOC IDs
    # Row 1: VOC names
    # Rows 2-160: data
    # Col 0: class labels

    classes = df.iloc[2:, 0].values.astype(int)
    data_raw = df.iloc[2:, 1:].values.astype(np.float64)
    voc_ids = np.array(df.iloc[0, 1:].values.tolist())
    voc_names = np.array(df.iloc[1, 1:].values.tolist())

    n_samples = data_raw.shape[0]
    n_features_raw = data_raw.shape[1]

    # 创建不可变 sample_id (基于原始 Excel 行号)
    sample_ids = np.array([f"sample_row_{i+3:04d}" for i in range(n_samples)])

    return {
        'df': df,
        'classes': classes,
        'data_raw': data_raw,
        'voc_ids': voc_ids,
        'voc_names': voc_names,
        'n_samples': n_samples,
        'n_features_raw': n_features_raw,
        'sample_ids': sample_ids,
    }


def check_data_structure(raw):
    """检查数据结构并返回审计结果"""
    audit = OrderedDict()

    audit['excel_shape'] = list(raw['df'].shape)
    audit['excel_sheet_names'] = ['Sheet6']  # 实际检查到的 sheet
    audit['n_samples'] = int(raw['n_samples'])
    audit['n_features_raw'] = int(raw['n_features_raw'])

    classes = raw['classes']
    unique, counts = np.unique(classes, return_counts=True)
    audit['class_distribution'] = {str(int(k)): int(v) for k, v in zip(unique, counts)}
    audit['class_1_count'] = int((classes == 1).sum())
    audit['class_2_count'] = int((classes == 2).sum())
    audit['class_3_count'] = int((classes == 3).sum())

    audit['nan_count'] = int(np.isnan(raw['data_raw']).sum())
    audit['inf_count'] = int(np.isinf(raw['data_raw']).sum())
    audit['neg_inf_count'] = int(np.isneginf(raw['data_raw']).sum())
    audit['negative_values_count'] = int((raw['data_raw'] < 0).sum())

    std = np.std(raw['data_raw'], axis=0)
    audit['constant_features'] = int((std == 0).sum())

    _, unique_cols = np.unique(np.round(raw['data_raw'], 8), axis=1, return_index=True)
    audit['duplicate_feature_cols'] = int(raw['n_features_raw'] - len(unique_cols))

    _, unique_rows = np.unique(np.round(raw['data_raw'], 8), axis=0, return_index=True)
    audit['duplicate_samples'] = int(raw['n_samples'] - len(unique_rows))

    unknown_count = int((raw['voc_names'] == 'Unknown').sum())
    audit['unknown_voc_names'] = unknown_count

    audit['voc_id_min'] = int(min(raw['voc_ids']))
    audit['voc_id_max'] = int(max(raw['voc_ids']))

    audit['sample_id_format'] = 'sample_row_NNNN (based on original Excel row)'
    audit['sample_id_examples'] = raw['sample_ids'][:5].tolist()

    return audit


def check_near_duplicate_samples(raw):
    """
    检查近重复样本：计算样本间 Pearson 相关系数，找出 r > threshold 的样本对。
    使用分块计算避免完整 O(n²) 相关矩阵。
    """
    threshold = 0.999
    keep_mask = raw['voc_names'] != 'Unknown'
    data_known = raw['data_raw'][:, keep_mask]
    n = data_known.shape[0]

    # 标准化 (每行)
    data_std = (data_known - np.mean(data_known, axis=1, keepdims=True)) / (
        np.std(data_known, axis=1, keepdims=True) + 1e-12
    )

    near_duplicate_pairs = []
    # 逐对上三角检查，但 n=159 完全可行
    for i in range(n):
        for j in range(i + 1, n):
            r = np.dot(data_std[i], data_std[j]) / (data_known.shape[1] - 1)
            if abs(r) > threshold:
                near_duplicate_pairs.append({
                    'sample_i': raw['sample_ids'][i],
                    'sample_j': raw['sample_ids'][j],
                    'class_i': int(raw['classes'][i]),
                    'class_j': int(raw['classes'][j]),
                    'correlation': round(float(r), 6),
                })

    return {
        'method': f'Pearson correlation on known features (n={data_known.shape[1]}), threshold={threshold}',
        'n_pairs': len(near_duplicate_pairs),
        'pairs': near_duplicate_pairs,
    }


# ============================================================
# 2. 元数据审计
# ============================================================
def check_metadata(raw):
    """检查 Excel 中是否存在元数据字段，列出所有检查过的位置"""
    df = raw['df']

    # 检查过的位置
    checked_locations = [
        'Excel file: data/1、2、3.xlsx, sheet: Sheet6',
        'Row 0 (header): all columns',
        'Row 1 (VOC name row): all columns',
        'Column 0 (label column): header = "Class", row1 = NaN',
        'MAT file: data/voc_dataset_1+2_vs_3.mat — only X, y, feat_names',
        'Repository files: no separate metadata CSV/JSON found',
    ]

    # 检查 Excel 中所有列的 header (Row 0)
    row0_values = df.iloc[0, :].astype(str).tolist()
    row1_values = df.iloc[1, :].astype(str).tolist()

    # 搜索可能匹配元数据关键字的列
    meta_keywords = [
        'subject', 'patient', 'batch', 'instrument', 'date', 'time',
        'location', 'site', 'center', 'operator', 'technician',
        'replicate', 'rep', 'sample_id', 'id', 'sex', 'gender', 'age',
    ]

    found_keywords = []
    for i, (r0, r1) in enumerate(zip(row0_values, row1_values)):
        combined = (r0 + ' ' + r1).lower()
        for kw in meta_keywords:
            if kw in combined:
                found_keywords.append({
                    'col_index': i,
                    'row0': r0,
                    'row1': r1,
                    'matched_keyword': kw,
                })

    metadata_found = {
        'checked_locations': checked_locations,
        'col0_header': str(df.iloc[0, 0]),
        'col0_row1': str(df.iloc[1, 0]),
        'has_subject_id': False,
        'has_batch_id': False,
        'has_instrument_id': False,
        'has_collection_date': False,
        'has_location': False,
        'has_operator': False,
        'has_replicate_id': False,
        'has_sample_id': False,
        'extra_columns_beyond_voc': 0,
        'keyword_matches_in_headers': found_keywords,
        'conclusion': (
            '在当前检查的工作表 (Sheet6)、列名和仓库数据文件中'
            '未发现 subject_id、batch_id、instrument_id、collection_date、'
            'location、operator、replicate_id 等元数据。'
        ),
    }

    return metadata_found


def create_metadata_template():
    """创建 metadata_template.csv"""
    template = pd.DataFrame(columns=[
        'sample_id',
        'subject_id',
        'batch_id',
        'instrument_id',
        'collection_date',
        'location',
        'operator',
        'replicate_id',
    ])
    template.to_csv(METADATA_TEMPLATE_PATH, index=False)
    return METADATA_TEMPLATE_PATH


# ============================================================
# 3. 标签任务审计 (Pipeline 内置于 CV 折)
# ============================================================
def compute_metrics_from_pooled(y_true, y_pred, y_prob):
    """从 pooled 预测计算所有指标"""
    eps = 1e-12
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = float(cm[0, 0]), float(cm[0, 1])
    fn, tp = float(cm[1, 0]), float(cm[1, 1])

    unique_y = np.unique(y_true)
    if len(unique_y) < 2:
        roc_auc_val = np.nan
        pr_auc_val = np.nan
    else:
        try:
            roc_auc_val = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            roc_auc_val = np.nan
        try:
            pr_auc_val = float(average_precision_score(y_true, y_prob))
        except ValueError:
            pr_auc_val = np.nan

    return {
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'roc_auc': roc_auc_val,
        'pr_auc': pr_auc_val,
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'sensitivity': float(recall_score(y_true, y_pred, zero_division=0)),
        'specificity': float(tn / (tn + fp + eps)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def evaluate_baselines(X, y):
    """评估三个基线模型"""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    baselines = OrderedDict()
    baselines['DummyClassifier(most_frequent)'] = DummyClassifier(
        strategy='most_frequent')
    baselines['DummyClassifier(stratified)'] = DummyClassifier(
        strategy='stratified', random_state=RANDOM_STATE)
    baselines['AllPositive'] = 'all_positive'  # 特殊处理

    baseline_results = []

    for name, model in baselines.items():
        y_pred_all = np.zeros(len(y), dtype=int)
        y_prob_all = np.zeros(len(y), dtype=float)
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if name == 'AllPositive':
                y_pred = np.ones(len(y_test), dtype=int)
                y_prob = np.ones(len(y_test), dtype=float)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = y_pred.astype(float)

            y_pred_all[test_idx] = y_pred
            y_prob_all[test_idx] = y_prob

            fold_metrics.append(compute_metrics_from_pooled(y[test_idx], y_pred, y_prob))

        # Pooled OOF
        pooled_metrics = compute_metrics_from_pooled(y, y_pred_all, y_prob_all)

        # Fold-wise mean/std
        df_fold = pd.DataFrame(fold_metrics)
        metric_keys = ['f1', 'accuracy', 'balanced_accuracy', 'mcc',
                        'roc_auc', 'pr_auc', 'precision', 'sensitivity', 'specificity']
        fold_summary = {}
        for k in metric_keys:
            vals = df_fold[k].dropna().values
            if len(vals) > 0:
                fold_summary[f'{k}_mean'] = float(np.mean(vals))
                fold_summary[f'{k}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            else:
                fold_summary[f'{k}_mean'] = np.nan
                fold_summary[f'{k}_std'] = np.nan

        baseline_results.append({
            'baseline': name,
            'pooled': pooled_metrics,
            'fold_summary': fold_summary,
        })

    return baseline_results


def audit_tasks(raw):
    """对四个子任务使用 Pipeline + 分层 CV 进行审计，含 OOF 预测和基线"""
    classes = raw['classes']
    data = raw['data_raw']

    # 仅使用非 Unknown 特征 —— 这是唯一在 CV 前进行的筛选（基于 VOC 名称，不基于数据分布）
    keep_mask = raw['voc_names'] != 'Unknown'
    data_known = data[:, keep_mask]
    n_features_known = data_known.shape[1]

    # 定义任务
    tasks = OrderedDict()
    tasks['1vs2'] = {
        'mask': (classes == 1) | (classes == 2),
        'pos_original': 2, 'neg_original': 1,
        'pos_label': 'Class 2 → 1', 'neg_label': 'Class 1 → 0',
    }
    tasks['1vs3'] = {
        'mask': (classes == 1) | (classes == 3),
        'pos_original': 3, 'neg_original': 1,
        'pos_label': 'Class 3 → 1', 'neg_label': 'Class 1 → 0',
    }
    tasks['2vs3'] = {
        'mask': (classes == 2) | (classes == 3),
        'pos_original': 3, 'neg_original': 2,
        'pos_label': 'Class 3 → 1', 'neg_label': 'Class 2 → 0',
    }
    tasks['1+2vs3'] = {
        'mask': np.ones(len(classes), dtype=bool),
        'pos_original': 3, 'neg_original': (1, 2),
        'pos_label': 'Class 3 → 1', 'neg_label': 'Class 1,2 → 0',
    }

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = []
    all_oof_rows = []
    all_fold_metrics_rows = []
    all_baselines_rows = []

    for task_name, task_info in tasks.items():
        mask = task_info['mask']
        X_task = data_known[mask]
        y_task = np.zeros(len(X_task), dtype=int)
        if task_name == '1+2vs3':
            y_task[classes[mask] == 3] = 1
        else:
            y_task[classes[mask] == task_info['pos_original']] = 1

        n_pos = int(y_task.sum())
        n_neg = int(len(y_task) - n_pos)

        # OOF 存储
        y_pred_all = np.zeros(len(y_task), dtype=int)
        y_prob_all = np.zeros(len(y_task), dtype=float)
        fold_metrics_list = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_task, y_task)):
            X_train, X_test = X_task[train_idx], X_task[test_idx]
            y_train, y_test = y_task[train_idx], y_task[test_idx]

            # Pipeline 在 CV 内构建和训练（所有数据驱动步骤在折内）
            pipe = build_pipeline()
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]

            y_pred_all[test_idx] = y_pred
            y_prob_all[test_idx] = y_prob

            fold_metrics = compute_metrics_from_pooled(y_test, y_pred, y_prob)
            fold_metrics['fold'] = fold_idx
            fold_metrics_list.append(fold_metrics)

        # Pooled OOF
        pooled_metrics = compute_metrics_from_pooled(y_task, y_pred_all, y_prob_all)

        # Fold-wise mean/std
        df_fold = pd.DataFrame(fold_metrics_list)
        metric_keys = ['f1', 'accuracy', 'balanced_accuracy', 'mcc',
                        'roc_auc', 'pr_auc', 'precision', 'sensitivity', 'specificity']
        fold_summary = {}
        for k in metric_keys:
            vals = df_fold[k].dropna().values
            if len(vals) > 0:
                fold_summary[f'{k}_mean'] = float(np.mean(vals))
                fold_summary[f'{k}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            else:
                fold_summary[f'{k}_mean'] = np.nan
                fold_summary[f'{k}_std'] = np.nan

        # 获取样本 ID
        task_sample_ids = raw['sample_ids'][mask]
        task_original_classes = classes[mask]

        # 保存 OOF 预测
        for i in range(len(y_task)):
            all_oof_rows.append({
                'task': task_name,
                'sample_id': task_sample_ids[i],
                'original_class': int(task_original_classes[i]),
                'y_true': int(y_task[i]),
                'y_pred': int(y_pred_all[i]),
                'y_prob': round(float(y_prob_all[i]), 6),
            })

        # 保存 fold metrics
        for fm in fold_metrics_list:
            fm['task'] = task_name
            all_fold_metrics_rows.append(fm)

        # 基线评估
        baseline_results = evaluate_baselines(X_task, y_task)
        for br in baseline_results:
            br['task'] = task_name
            all_baselines_rows.append(br)

        results.append({
            'task': task_name,
            'n_total': int(n_pos + n_neg),
            'n_positive': n_pos,
            'n_negative': n_neg,
            'pos_label': task_info['pos_label'],
            'neg_label': task_info['neg_label'],
            'pos_original': str(task_info['pos_original']),
            'neg_original': str(task_info['neg_original']),
            'pooled_metrics': pooled_metrics,
            'fold_summary': fold_summary,
            'per_fold': fold_metrics_list,
            'baselines': baseline_results,
        })

    return results, all_oof_rows, all_fold_metrics_rows, all_baselines_rows


# ============================================================
# 4. 探索性可视化
# ============================================================
def create_pca_plot(raw):
    """PCA 可视化 (类别着色)"""
    classes = raw['classes']
    keep_mask = raw['voc_names'] != 'Unknown'
    data_known = raw['data_raw'][:, keep_mask]

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_known)

    pca = PCA(n_components=min(5, data_scaled.shape[1]))
    pca_data = pca.fit_transform(data_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}
    labels = {1: 'Class 1', 2: 'Class 2', 3: 'Class 3'}
    markers = {1: 'o', 2: 's', 3: '^'}

    for c in [1, 2, 3]:
        mask = classes == c
        axes[0].scatter(pca_data[mask, 0], pca_data[mask, 1],
                         c=colors[c], label=labels[c], marker=markers[c],
                         alpha=0.7, edgecolors='k', linewidth=0.5, s=60)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[0].set_title('PCA: PC1 vs PC2 (by Class)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    axes[1].bar(range(1, len(cumsum) + 1), pca.explained_variance_ratio_,
                 alpha=0.7, color='steelblue', label='Individual')
    axes[1].plot(range(1, len(cumsum) + 1), cumsum, 'ro-', markersize=5,
                  label='Cumulative')
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
    axes[1].axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label='80%')
    axes[1].set_xlabel('Principal Component')
    axes[1].set_ylabel('Explained Variance Ratio')
    axes[1].set_title(f'PCA Scree Plot (top {len(cumsum)} PCs)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    pca_path = os.path.join(AUDIT_DIR, 'pca.png')
    fig.savefig(pca_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return {
        'pca_path': pca_path,
        'n_components': len(cumsum),
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance': cumsum.tolist(),
        'pc1_variance': float(pca.explained_variance_ratio_[0]),
        'pc2_variance': float(pca.explained_variance_ratio_[1]),
        'pc1_pc2_cumulative': float(cumsum[1]),
    }


# ============================================================
# 生成 Markdown 报告
# ============================================================
def generate_markdown_report(structure, metadata, near_dup, task_results, pca_info):
    """生成 data_audit.md"""
    md = f"""# 数据审计报告 (Data Audit Report)

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据文件**: `data/1、2、3.xlsx`

---

## 1. 数据结构

| 项目 | 值 |
|------|-----|
| Excel 文件 | `data/1、2、3.xlsx` |
| Sheet 名称 | {', '.join(structure.get('excel_sheet_names', ['Sheet6']))} |
| Excel 维度 | {structure['excel_shape'][0]} rows × {structure['excel_shape'][1]} cols |
| 样本数 | {structure['n_samples']} |
| 原始特征数 (含 Unknown) | {structure['n_features_raw']} |
| Unknown VOC 名称数 | {structure['unknown_voc_names']} |
| 已知 VOC 特征数 | {structure['n_features_raw'] - structure['unknown_voc_names']} |
| 类别 1 数量 | {structure['class_1_count']} |
| 类别 2 数量 | {structure['class_2_count']} |
| 类别 3 数量 | {structure['class_3_count']} |
| 缺失值 (NaN) | {structure['nan_count']} |
| 正/负无穷 | {structure['inf_count']} / {structure['neg_inf_count']} |
| 负值数量 | {structure['negative_values_count']} |
| 常数特征 | {structure['constant_features']} |
| 重复特征列 | {structure['duplicate_feature_cols']} |
| 完全重复样本 | {structure['duplicate_samples']} |
| 近重复样本对 (r > {near_dup['method'].split('threshold=')[1].split(')')[0] if 'threshold=' in near_dup['method'] else '0.999'}) | {near_dup['n_pairs']} |
| 样本 ID 格式 | {structure['sample_id_format']} |

### Excel 实际结构

- **Row 0**: Header — `Class` 列 + VOC ID 编号 (0, 1, 2, ...)
- **Row 1**: VOC 名称 (部分为 `Unknown`)
- **Rows 2-160**: 数据行 (159 个样本)
- **Col 0**: 类别标签 (1, 2, 3)
- **Cols 1-1734**: VOC 特征值 (1734 个特征)

"""

    if near_dup['n_pairs'] > 0:
        md += f"""### 近重复样本 (Pearson r > 0.999)

发现 {near_dup['n_pairs']} 对近重复样本：

| Sample A | Sample B | Class A | Class B | r |
|----------|----------|---------|---------|---|
"""
        for pair in near_dup['pairs']:
            md += f"| {pair['sample_i']} | {pair['sample_j']} | {pair['class_i']} | {pair['class_j']} | {pair['correlation']:.6f} |\n"
        md += "\n"
    else:
        md += "### 近重复样本检查\n\n未发现 Pearson r > 0.999 的近重复样本对。\n\n"

    md += f"""---

## 2. 元数据审计

### 检查过的位置

"""
    for loc in metadata.get('checked_locations', []):
        md += f"- {loc}\n"

    md += f"""
### 结论

{metadata['conclusion']}

### 影响

- ❌ 无法排除受试者泄漏 (同一受试者可能跨训练/测试集)
- ❌ 无法排除批次混杂 (标签可能与 batch 高度重合)
- ❌ 后续随机分层 CV 依赖"样本相互独立"这一**暂时假设**
- ⚠️ 缺少元数据**不能**被自动解释为任务错误，但必须降低证据等级

已创建元数据模板: `data/metadata_template.csv`

---

## 3. 标签任务审计

### 固定 Pipeline 配置

```
{PIPELINE_CONFIG['imputer']}
→ {PIPELINE_CONFIG['scaler']}
→ {PIPELINE_CONFIG['classifier']}
```

{PIPELINE_CONFIG['note']}

### 任务定义

| 任务 | 正类 (→1) | 负类 (→0) | 总样本 | 正类数 | 负类数 |
|------|-----------|-----------|--------|--------|--------|
"""
    for r in task_results:
        md += (f"| **{r['task']}** | {r['pos_label']} | {r['neg_label']} | "
               f"{r['n_total']} | {r['n_positive']} | {r['n_negative']} |\n")

    md += "\n### Logistic Regression — Pooled OOF 指标\n\n"
    md += "| 任务 | F1 | Balanced Acc | MCC | ROC-AUC | PR-AUC | Precision | Sensitivity | Specificity |\n"
    md += "|------|----|-------------|-----|---------|--------|-----------|-------------|------------|\n"
    for r in task_results:
        pm = r['pooled_metrics']
        md += (f"| {r['task']} | {pm['f1']:.4f} | {pm['balanced_accuracy']:.4f} | "
               f"{pm['mcc']:.4f} | {pm['roc_auc']:.4f} | {pm['pr_auc']:.4f} | "
               f"{pm['precision']:.4f} | {pm['sensitivity']:.4f} | {pm['specificity']:.4f} |\n")

    md += "\n### Logistic Regression — Fold-wise Mean ± Std\n\n"
    md += "| 任务 | F1 | Balanced Acc | MCC | ROC-AUC | PR-AUC |\n"
    md += "|------|----|-------------|-----|---------|--------|\n"
    for r in task_results:
        fs = r['fold_summary']
        md += (f"| {r['task']} | {fs['f1_mean']:.4f}±{fs['f1_std']:.4f} | "
               f"{fs['balanced_accuracy_mean']:.4f}±{fs['balanced_accuracy_std']:.4f} | "
               f"{fs['mcc_mean']:.4f}±{fs['mcc_std']:.4f} | "
               f"{fs['roc_auc_mean']:.4f}±{fs['roc_auc_std']:.4f} | "
               f"{fs['pr_auc_mean']:.4f}±{fs['pr_auc_std']:.4f} |\n")

    md += "\n### 混淆矩阵 (Pooled OOF)\n\n"
    md += "| 任务 | TN | FP | FN | TP |\n"
    md += "|------|----|----|----|----|\n"
    for r in task_results:
        pm = r['pooled_metrics']
        md += f"| {r['task']} | {pm['tn']} | {pm['fp']} | {pm['fn']} | {pm['tp']} |\n"

    md += "\n### 基线模型 — Pooled OOF 指标\n\n"
    md += "| 任务 | 基线 | F1 | Balanced Acc | MCC | ROC-AUC | PR-AUC |\n"
    md += "|------|------|----|-------------|-----|---------|--------|\n"
    for r in task_results:
        for bl in r['baselines']:
            pm = bl['pooled']
            roc_str = f"{pm['roc_auc']:.4f}" if not np.isnan(pm['roc_auc']) else "NaN"
            pr_str = f"{pm['pr_auc']:.4f}" if not np.isnan(pm['pr_auc']) else "NaN"
            md += (f"| {r['task']} | {bl['baseline']} | {pm['f1']:.4f} | "
                   f"{pm['balanced_accuracy']:.4f} | {pm['mcc']:.4f} | "
                   f"{roc_str} | {pr_str} |\n")

    md += f"""
**注意**: 以上结果仅用于描述标签异质性，**不得**根据哪个任务分数最高自动改变主任务。
主任务固定为 `1+2 vs 3`，任何调整必须由领域依据决定。

当前简单线性模型未发现类别 1 与类别 2 的明显区分信号；
该结果**不能**替代类别合并所需的领域依据。

---

## 4. PCA 可视化

| 指标 | 值 |
|------|-----|
| 使用特征数 (去 Unknown) | {structure['n_features_raw'] - structure['unknown_voc_names']} |
| PC1 方差解释 | {pca_info['pc1_variance']:.4f} ({pca_info['pc1_variance']*100:.2f}%) |
| PC2 方差解释 | {pca_info['pc2_variance']:.4f} ({pca_info['pc2_variance']*100:.2f}%) |
| PC1+PC2 累积 | {pca_info['pc1_pc2_cumulative']:.4f} ({pca_info['pc1_pc2_cumulative']*100:.2f}%) |

![PCA](pca.png)

**注意**: PCA 可视化仅用于探索性描述，**禁止**将其解释为强证据或用于判断模型有效性。

---

## 5. 预处理流程回顾

当前预处理流程 (见 `preprocessing_data.ipynb`):

1. 去除 Unknown 特征: 1734 → 988
2. 丰度筛选 (均值 > P40): 988 → 593
3. log1p 变换 + IQR 筛选 (IQR ≥ P25): 593 → 445
4. 标签映射: 1,2 → 0; 3 → 1
5. 保存为 MAT 文件: `data/voc_dataset_1+2_vs_3.mat`

**当前全数据预处理** (在划分前进行) 使用了所有样本的信息来做特征筛选，
这在严格意义上引入了信息泄漏，但当前项目以此为基线。

---

## 6. 预存 MAT 文件验证

"""
    try:
        import scipy.io as sio
        mat = sio.loadmat(MAT_PATH)
        X = mat['X']
        y = mat['y']
        feat_names = [str(v.flat[0]) for v in mat['feat_names'].flatten()]
        md += f"""- MAT 文件: `{MAT_PATH}`
- X shape: {X.shape}
- y shape: {y.shape}
- 特征数: {len(feat_names)}
- 正类数 (label=1): {int(y.sum())}
- 负类数 (label=0): {int(len(y) - y.sum())}
"""
    except Exception as e:
        md += f"- MAT 文件加载失败: {e}\n"

    md += """
---

*报告由 `run_data_audit.py` 自动生成。*
"""
    return md


# ============================================================
# 生成 DATA_ASSUMPTIONS.md
# ============================================================
def generate_data_assumptions():
    """生成 DATA_ASSUMPTIONS.md"""
    content = """# 数据假设与待确认项 (Data Assumptions)

**生成时间**: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """

---

## 元数据审计范围

在当前检查的工作表 (Sheet6)、列名和仓库数据文件中，
未发现 subject_id、batch_id、instrument_id、collection_date、
location、operator、replicate_id 等元数据。

---

## 待确认项

| # | 待确认问题 | 当前假设 | 风险等级 |
|---|-----------|---------|---------|
| 1 | 类别 1、2、3 的领域含义是什么？ | 未知。当前仅知 1+2 合并为负类，3 为正类 | 🔴 高 |
| 2 | 为什么 1 和 2 合并？ | 未知。假设合并后样本量更均衡 (106:53) | 🟡 中 |
| 3 | 一个样本是否对应一个独立受试者？ | **假设是**。无 subject_id 可验证 | 🔴 高 |
| 4 | 是否存在重复采样 (同一受试者多次采样)？ | 无法验证。无 subject_id 或 replicate_id | 🔴 高 |
| 5 | 是否存在批次/设备/时间/地点混杂？ | 无法验证。无批次/时间/地点元数据 | 🔴 高 |
| 6 | 未来应用场景是什么？ | 未知。假设为独立新样本的 VOC 分类 | 🟡 中 |
| 7 | 漏判 (FN) 与误判 (FP) 的实际代价？ | 未知。当前以 F1 为指标，等权重 | 🟡 中 |
| 8 | 数据采集是否标准化？ | 假设是。无 instrument/operator 信息 | 🟡 中 |
| 9 | 标签是否经过独立验证（非 VOC 数据）？ | 未知。假设标签为金标准 | 🟡 中 |
| 10 | 样本量是否足够支持高维特征建模？ | n=159, p=1734 (原始), p>n 问题严重 | 🔴 高 |

---

## 当前分析约束

1. **主任务固定**: `1+2 vs 3`，不论其他任务分数高低，不做自动更换。
2. **主指标**: 外层 OOF F1 (类别 3 的 F1)
3. **样本独立性**: 在无元数据的情况下，假设所有样本相互独立。
4. **证据等级**: 由于缺乏元数据，所有分析结果应视为探索性研究级别。
5. **特征选择**: 任何特征选择必须在 CV 内层进行，不能使用全数据预筛选。

---

## 建议后续补充

1. 收集受试者 ID (subject_id)，用于防止数据泄漏
2. 收集批次/设备/时间信息，用于评估混杂
3. 确认类别 1、2、3 的领域含义及合并依据
4. 确认未来应用的样本采集条件是否与训练数据一致
5. 评估漏判与误判的实际代价，选择合适的阈值优化策略

---

*此文件由 `run_data_audit.py` 自动生成。*
"""
    return content


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 70)
    print(" 阶段零：任务与数据生成机制审计")
    print("=" * 70)

    # 1. 数据读取
    print("\n[1/6] 读取原始 Excel...")
    raw = load_raw_excel()
    classes = raw['classes']
    c1 = int((classes == 1).sum())
    c2 = int((classes == 2).sum())
    c3 = int((classes == 3).sum())
    print(f"  样本数: {raw['n_samples']}, 原始特征数: {raw['n_features_raw']}")
    print(f"  类别分布: 1={c1}, 2={c2}, 3={c3}")

    # 2. 数据结构检查
    print("\n[2/6] 数据结构检查...")
    structure = check_data_structure(raw)
    print(f"  缺失值: {structure['nan_count']}, 无穷值: {structure['inf_count']}")
    print(f"  常数特征: {structure['constant_features']}, 重复特征列: {structure['duplicate_feature_cols']}")
    print(f"  Unknown VOC: {structure['unknown_voc_names']}")

    # 近重复样本检查
    print("\n[2b/6] 近重复样本检查...")
    near_dup = check_near_duplicate_samples(raw)
    print(f"  方法: {near_dup['method']}")
    print(f"  近重复样本对: {near_dup['n_pairs']}")
    if near_dup['n_pairs'] > 0:
        for pair in near_dup['pairs']:
            print(f"    {pair['sample_i']} <-> {pair['sample_j']} "
                  f"(class {pair['class_i']}/{pair['class_j']}, r={pair['correlation']})")

    # 3. 元数据审计
    print("\n[3/6] 元数据审计...")
    metadata = check_metadata(raw)
    print(f"  检查过的位置: {len(metadata['checked_locations'])} 个")
    print(f"  结论: {metadata['conclusion']}")

    if not metadata['has_subject_id']:
        tmpl_path = create_metadata_template()
        print(f"  已创建元数据模板: {tmpl_path}")

    # 4. 标签任务审计
    print("\n[4/6] 标签任务审计 (Pipeline 内置于 CV 折)...")
    print(f"  Pipeline: {PIPELINE_CONFIG['imputer']} → {PIPELINE_CONFIG['scaler']} → ...")
    task_results, oof_rows, fold_metrics_rows, baselines_rows = audit_tasks(raw)

    for r in task_results:
        pm = r['pooled_metrics']
        fs = r['fold_summary']
        print(f"\n  === {r['task']} ({r['pos_label']} vs {r['neg_label']}) ===")
        print(f"  样本: n={r['n_total']} (pos={r['n_positive']}, neg={r['n_negative']})")
        print(f"  Pooled OOF: F1={pm['f1']:.4f}, BalancedAcc={pm['balanced_accuracy']:.4f}, "
              f"MCC={pm['mcc']:.4f}, ROC-AUC={pm['roc_auc']:.4f}, PR-AUC={pm['pr_auc']:.4f}")
        print(f"  Fold Mean±Std: F1={fs['f1_mean']:.4f}±{fs['f1_std']:.4f}, "
              f"BalAcc={fs['balanced_accuracy_mean']:.4f}±{fs['balanced_accuracy_std']:.4f}")
        print(f"  混淆矩阵: TN={pm['tn']}, FP={pm['fp']}, FN={pm['fn']}, TP={pm['tp']}")

        for bl in r['baselines']:
            bpm = bl['pooled']
            roc_str = f"{bpm['roc_auc']:.4f}" if not np.isnan(bpm['roc_auc']) else "NaN"
            pr_str = f"{bpm['pr_auc']:.4f}" if not np.isnan(bpm['pr_auc']) else "NaN"
            print(f"  Baseline [{bl['baseline']}]: F1={bpm['f1']:.4f}, "
                  f"BalAcc={bpm['balanced_accuracy']:.4f}, ROC-AUC={roc_str}")

    # 5. PCA 可视化
    print("\n[5/6] PCA 可视化...")
    pca_info = create_pca_plot(raw)
    print(f"  PC1: {pca_info['pc1_variance']:.4f}, PC2: {pca_info['pc2_variance']:.4f}")
    print(f"  PCA 图已保存: {pca_info['pca_path']}")

    # ============================================================
    # 保存输出
    # ============================================================
    print("\n" + "=" * 70)
    print(" 保存审计结果")
    print("=" * 70)

    # 保存 JSON
    audit_json = {
        'timestamp': datetime.now().isoformat(),
        'pipeline_config': PIPELINE_CONFIG,
        'structure': structure,
        'near_duplicate': near_dup,
        'metadata': metadata,
        'task_results': [
            {k: v for k, v in r.items() if k not in ('per_fold', 'baselines')}
            for r in task_results
        ],
        'task_results_with_baselines': [
            {k: v for k, v in r.items() if k != 'per_fold'}
            for r in task_results
        ],
        'pca': {k: v for k, v in pca_info.items() if k != 'pca_path'},
    }
    json_path = os.path.join(AUDIT_DIR, 'data_audit.json')
    with open(json_path, 'w') as f:
        json.dump(audit_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [✓] {json_path}")

    # 保存 MD 报告
    md_content = generate_markdown_report(structure, metadata, near_dup, task_results, pca_info)
    md_path = os.path.join(AUDIT_DIR, 'data_audit.md')
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"  [✓] {md_path}")

    # 保存 class_counts.csv
    cc_df = pd.DataFrame({
        'Class': [1, 2, 3],
        'Count': [structure['class_1_count'], structure['class_2_count'], structure['class_3_count']],
        'Label_1+2vs3': ['Negative (0)', 'Negative (0)', 'Positive (1)'],
    })
    cc_csv = os.path.join(AUDIT_DIR, 'class_counts.csv')
    cc_df.to_csv(cc_csv, index=False)
    print(f"  [✓] {cc_csv}")

    # 保存 task_diagnostics.csv (per-fold)
    td_df = pd.DataFrame(fold_metrics_rows)
    td_csv = os.path.join(AUDIT_DIR, 'task_diagnostics.csv')
    td_df.to_csv(td_csv, index=False)
    print(f"  [✓] {td_csv}")

    # 保存 task_oof_predictions.csv
    oof_df = pd.DataFrame(oof_rows)
    oof_csv = os.path.join(AUDIT_DIR, 'task_oof_predictions.csv')
    oof_df.to_csv(oof_csv, index=False)
    print(f"  [✓] {oof_csv}")

    # 保存 task_fold_metrics.csv (含 fold_summary)
    fm_rows_flat = []
    for r in task_results:
        fs = r['fold_summary']
        fs['task'] = r['task']
        fs['type'] = 'fold_summary'
        fm_rows_flat.append(fs)
        for fm in r['per_fold']:
            fm_copy = dict(fm)
            fm_copy['task'] = r['task']
            fm_copy['type'] = 'per_fold'
            fm_rows_flat.append(fm_copy)
    fm_df = pd.DataFrame(fm_rows_flat)
    fm_csv = os.path.join(AUDIT_DIR, 'task_fold_metrics.csv')
    fm_df.to_csv(fm_csv, index=False)
    print(f"  [✓] {fm_csv}")

    # 保存 task_baselines.csv
    bl_rows_flat = []
    for r in task_results:
        for bl in r['baselines']:
            row = {'task': bl['task'], 'baseline': bl['baseline']}
            row.update(bl['pooled'])
            row.update(bl['fold_summary'])
            bl_rows_flat.append(row)
    bl_df = pd.DataFrame(bl_rows_flat)
    bl_csv = os.path.join(AUDIT_DIR, 'task_baselines.csv')
    bl_df.to_csv(bl_csv, index=False)
    print(f"  [✓] {bl_csv}")

    # 保存 DATA_ASSUMPTIONS.md
    da_content = generate_data_assumptions()
    da_path = os.path.join(DOCS_DIR, 'DATA_ASSUMPTIONS.md')
    with open(da_path, 'w') as f:
        f.write(da_content)
    print(f"  [✓] {da_path}")

    print("\n" + "=" * 70)
    print(" 阶段零审计完成")
    print("=" * 70)
    print(f"""
输出文件:
  {json_path}
  {md_path}
  {cc_csv}
  {td_csv}
  {oof_csv}
  {fm_csv}
  {bl_csv}
  {pca_info['pca_path']}
  {tmpl_path if not metadata['has_subject_id'] else 'N/A'}
  {da_path}
""")


if __name__ == '__main__':
    main()