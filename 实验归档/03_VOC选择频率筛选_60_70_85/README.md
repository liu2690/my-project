# VOC 选择频率阈值筛选实验（60%、70%、85%）

## 1. 实验目的

研究提高 VOC 特征选择频率阈值后，稳定 VOC 数量及分类性能的变化。

## 2. 特征选择方法

基于原始 BPD-MLP 实验得到的 10 个冠军模型，统计每个 VOC 被选中的次数。

选择频率计算公式：

F_j = Selected_Count_j / 10

其中，F_j 表示第 j 个 VOC 的选择频率。

## 3. 筛选条件及结果

| 筛选阈值 | 稳定 VOC 数量 |
|---|---:|
| >60% | 264 |
| >70% | 94 |
| >85% | 16 |

由于只有 10 个冠军模型，选择频率以 10% 为间隔。

## 4. MLP 内部验证结果

| VOC 数量 | Accuracy | Sensitivity | Specificity | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| 264 | 84.38% | 81.82% | 85.71% | 0.7826 | 0.8377 |
| 94 | 71.88% | 72.73% | 71.43% | 0.6400 | 0.7381 |
| 16 | 65.63% | 0% | 100% | 0 | 0.5022 |

## 5. 实验分析

提高 VOC 选择频率阈值能够减少特征数量，但特征数量减少后，模型分类性能也出现下降。

264 个 VOC 的内部验证效果最好；94 个 VOC 保留了一定的分类能力；16 个 VOC 筛选过于严格，正类识别能力丧失。

## 6. 主要文件

- run.py：从选择频率表重建三组 mask，并使用固定划分完成 MLP 内部验证
- feature_selection_frequency.csv：全部 VOC 选择频率
- stable_features_gt60%.csv：大于 60% 的 VOC 名单
- stable_features_gt70%.csv：大于 70% 的 VOC 名单
- stable_features_gt85%.csv：大于 85% 的 VOC 名单
- voc_threshold_comparison_strict.csv：不同阈值下的特征数量
- validation_summary.csv：三组 MLP 内部验证结果

本实验为开发集内部探索性验证，不作为独立测试集的最终泛化性能。

## 7. 复现

从仓库根目录运行：

```bash
python "实验归档/03_VOC选择频率筛选_60_70_85/run.py" --check
python "实验归档/03_VOC选择频率筛选_60_70_85/run.py"
```

脚本直接使用本目录公开的 `feature_selection_frequency.csv`，不依赖未公开的 `.npy` mask 或 `.npz` 划分文件；固定外层种子为 42，内部验证种子为 1042。新结果写入 `result/reproduced/03_voc_thresholds/`。
