# 私有数据放置说明

原始数据不随仓库公开。请通过数据申请或外部授权渠道获得下列文件，并放到本目录：

```text
data/voc_dataset_table1_1+2_vs_3_bpd.mat
```

MAT 文件必须包含以下字段：

| 字段 | 形状 | 类型 | 说明 |
|---|---:|---|---|
| `X` | `(159, 1692)` | `float32` | 样本 × VOC 特征矩阵 |
| `y` | `(159, 1)` | `int32` | 二分类标签（0/1） |
| `original_class` | `(159, 1)` | `int32` | 原始类别 |
| `feature_ids` | `(1, 1692)` | `int32` | VOC 特征编号 |
| `feature_names` | `(1, 1692)` | MATLAB cell/object | VOC 名称 |
| `feat_names` | `(1, 1692)` | MATLAB cell/object | 兼容字段，VOC 名称 |

四个 `run.py --check` 会检查文件是否存在，以及 `X`、`y` 的形状和标签是否满足实验要求。数据文件受根目录 `.gitignore` 保护，但提交前仍应运行：

```bash
git ls-files "*.mat" "*.xlsx"
```

该命令必须没有输出。
