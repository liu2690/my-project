# 删除棉球物质后的 BPD-MLP 实验

## 1. 实验目的

使用删除棉球物质后的表一数据进行 BPD 二分类实验，采用原始 MLP 分类器进行 VOC 特征选择和分类。

## 2. 数据集

- 样本数：159
- VOC 特征数：1692
- 开发集 A：127
- 独立测试集 B：32
- 分类任务：1+2 vs 3

## 3. 实验参数

- K-means：3 类
- 随机种子：42
- 开发集随机划分：10 次
- 每次候选模型：10 个
- 候选模型总数：100
- Epoch：30
- 稳定 VOC 筛选阈值：Selection Frequency > 60%
- 最终稳定 VOC：264 个
- 分类阈值：0.05
- Bootstrap：1000 次

## 4. 独立测试集结果

| 指标 | 结果 |
|---|---:|
| Accuracy | 78.12% |
| Sensitivity | 72.73% |
| Specificity | 80.95% |
| F1 | 69.57% |
| AUC | 0.7684 |

## 5. 主要文件

- `run.py`：可复现入口；`--check` 仅检查数据和代码，默认执行正式 Notebook 的单元 0、1
- `testing_MLP.ipynb`：完整实验 Notebook
- `modules.py`：MLP 模型结构
- `utils.py`：实验辅助函数
- `stable_features.csv`：264 个稳定 VOC 名单
- `feature_selection_frequency.csv`：1692 个 VOC 的选择频率
- `final_model.pt`：最终训练模型
- `fixed_outer_split.npz`：固定数据集划分
- `final_test_metrics.csv`：独立测试指标
- `final_test_roc.png`：ROC 曲线
- `run_config.json`：实验配置

## 6. 实验结论

在独立测试集 B 上，模型 Accuracy 达到 78.12%，AUC 为 0.7684，筛选出 264 个稳定 VOC。

## 7. 复现

从仓库根目录运行：

```bash
python "实验归档/01_删除棉球物质_BPD_MLP_ACC78.12/run.py" --check
python "实验归档/01_删除棉球物质_BPD_MLP_ACC78.12/run.py"
```

输入数据文件名和字段见根目录 `data/README.md`，新结果写入 `result/reproduced/01_mlp/`。
