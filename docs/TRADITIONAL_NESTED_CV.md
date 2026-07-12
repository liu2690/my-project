# 传统模型 Nested CV 文档

## 概述

本模块实现传统机器学习模型的 Nested Cross-Validation，采用**候选独立内层 OOF 分数生成 + 候选独立阈值选择**策略。每个候选超参数配置在每一折外层独立运行内层 CV，生成 OOF 预测分数，然后独立选择最优阈值。

## 架构

```
run_nested_cv.py          -- CLI 入口（--quick / --full / --n-jobs）
nested_cv_engine.py        -- Nested CV 核心引擎
candidate_registry.py      -- 候选生成与 pipeline 构建
models_sklearn.py          -- 模型族 pipeline 工厂
thresholding.py            -- 阈值选择模块
metrics.py                 -- 统一指标计算
```

## Pipeline 结构

所有候选共享统一的预处理前置：

```
SimpleImputer(median) → VOCAbundanceIQRFilter → StandardScaler → [SelectKBest(f_classif)] → classifier
```

- `SimpleImputer(strategy='median')`：缺失值填充
- `VOCAbundanceIQRFilter`：VOC 丰度 + IQR 过滤器（Phase 1 实现）
- `StandardScaler`：标准化
- `SelectKBest(f_classif, k)`：可选特征选择（仅 LinearSVM / RBF SVM / LDA）
- classifier：最终分类器

## 模型族

### 1. Elastic Net Logistic Regression

| 参数 | 候选值 |
|------|--------|
| C | 0.01, 0.1, 1.0, 10.0 |
| l1_ratio | 0.25, 0.5, 0.75 |
| class_weight | None, balanced |

- 无 SelectKBest
- solver: saga, penalty: elasticnet, max_iter: 50000
- 分数类型: probability
- 默认阈值: 0.5

### 2. Linear SVM

| 参数 | 候选值 |
|------|--------|
| C | 0.01, 0.1, 1.0, 10.0 |
| class_weight | None, balanced |
| k | 20, 50, 100, all |

- 可选 SelectKBest
- dual=True, max_iter=50000
- 分数类型: decision_function
- 默认阈值: 0.0

### 3. RBF SVM

| 参数 | 候选值 |
|------|--------|
| C | 0.1, 1.0, 10.0 |
| gamma | scale, 0.01, 0.1 |
| class_weight | None, balanced |
| k | 20, 50, 100, all |

- 可选 SelectKBest
- kernel='rbf', probability=False（使用 decision_function 加速）
- 分数类型: decision_function
- 默认阈值: 0.0

### 4. Shrinkage LDA

| 参数 | 候选值 |
|------|--------|
| shrinkage | auto, 0.1, 0.5, 0.9 |
| k | 20, 50, 100, all |

- 可选 SelectKBest
- solver='lsqr'
- 分数类型: probability
- 默认阈值: 0.5

## 候选数量

| 模式 | elastic_net | linear_svm | rbf_svm | lda | 总计 |
|------|-------------|------------|---------|-----|------|
| quick | 4 | 8 | 1 | 2 | 15 |
| full | 24 | 32 | 72 | 16 | 144 |

Quick 模式: 仅运行 outer fold 0 和 1。

## 候选 ID 格式

```
{model_family}__{param1}={value1}__{param2}={value2}__...
```

示例：
- `elastic_net__C=0.1__l1=0.5__cw=none`
- `linear_svm__C=1.0__cw=balanced__k=50`
- `rbf_svm__C=1.0__gamma=scale__cw=none__k=50`
- `lda__shrinkage=auto__k=all`

## 阈值选择策略

每个候选独立运行内层 4-fold CV，生成 OOF 预测分数后，独立选择最优阈值。

**选优顺序**（严格级联）：
1. 最大化 F1 分数
2. 平局时最大化 MCC
3. 仍平局时最大化 Balanced Accuracy
4. 仍平局时选择最接近默认阈值的候选

**候选阈值生成**：以排序后的唯一 OOF 分数之间的中点作为候选阈值，确保每个阈值边界都被测试。

## 外层评估

每个外层 fold：
1. 在内层 train 上运行所有候选的 inner CV
2. 按 inner tuned F1 排名，选择最优候选
3. 在完整 outer-train 上重新拟合最优候选
4. 在 outer-test 上预测并计算指标

## 输出文件

所有输出位于 `result/nested_cv/traditional_selector/{mode}/`：

| 文件 | 内容 |
|------|------|
| `oof_predictions.csv` | 每行一个样本，含 y_true, y_pred_tuned, y_pred_default, score, outer_fold, selected_candidate_id |
| `outer_fold_metrics.csv` | 每折外层的 tuned/default 指标 |
| `aggregate_metrics.json` | 汇总指标（pooled OOF, foldwise mean/std, baseline, 证据局限性） |
| `inner_candidate_results.csv` | 所有候选在所有 fold 的内部评估结果 |
| `selected_configs.json` | 每折选择的候选配置及阈值 |
| `thresholds.csv` | 阈值详情 |
| `selected_features.csv` | 选择的特征（VOC filter support + selector support + coefficients） |
| `subgroup_errors.csv` | 按原始类别（1/2/3）的亚组错误分析 |
| `run_metadata.json` | 运行元数据（git commit, fingerprint, 时间戳, 版本） |

## Baseline 评估

三种基线模型（使用相同 outer fold split）：
- **DummyMostFrequent**：始终预测多数类
- **DummyStratified**：按训练集类别比例随机预测
- **AllPositive**：始终预测正类

## 聚合指标

- `pooled_oof_metrics`：将所有 outer fold 的 OOF 预测拼接后计算全局指标
- `foldwise_mean_metrics`：各 fold 指标的均值
- `foldwise_std_metrics`：各 fold 指标的标准差
- 同时报告 tuned（使用候选独立阈值）和 default（使用默认阈值）两套指标

## 证据局限性

所有结果自动附带以下声明：
- **subject_independence_unverified**: 受试者独立性未经验证
- **batch_confounding_cannot_be_excluded**: 批次混杂无法排除
- **internal_validation_only**: 仅内部验证，无外部验证集

## 运行方式

```bash
# Quick 模式（15 候选，2 折外层）
python run_nested_cv.py --quick

# Full 模式（144 候选，5 折外层）
python run_nested_cv.py --full

# 指定并行数
python run_nested_cv.py --full --n-jobs 8

# 查看帮助
python run_nested_cv.py --help
```

## 收敛警告处理

- 记录每个候选的收敛警告数量
- 不因警告而排除候选（标记为 valid 但记录 warning_count）
- 无效候选（如训练失败）标记为 invalid，跳过外层评估

## 设计原则

1. **候选独立阈值选择**：每个候选在内层 CV 中独立生成 OOF 分数，独立选择阈值。不先选择模型再调阈值。
2. **完整数据流隔离**：内层仅使用 outer-train，外层仅使用 outer-test，绝无信息泄露。
3. **可复现性**：固定随机种子（random_state=42），记录 git commit 和 dataset fingerprint。
4. **统一指标**：所有指标计算统一通过 `metrics.py`，避免分散实现。