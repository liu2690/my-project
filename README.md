# BPD-MLP 84.38% Reproduction

本项目用于复现基于264个稳定VOC特征的BPD-MLP内部验证实验。

## 1. Experimental Results

| Metric | Value |
|---|---:|
| Validation Samples | 32 |
| Stable VOC Features | 264 |
| Accuracy | 84.375% |
| Sensitivity | 81.818% |
| Specificity | 85.714% |
| F1 | 0.782609 |
| AUC | 0.837662 |

Confusion Matrix:

[[18, 3],
 [2, 9]]

注意：84.375%为内部验证集Accuracy，不是独立测试集Accuracy。

## 2. Dataset

原始MAT数据不上传GitHub，需要单独获取。

文件名：

voc_dataset_table1_1+2_vs_3_bpd.mat

收到文件后放置于：

data/voc_dataset_table1_1+2_vs_3_bpd.mat

数据规格：

- Samples: 159
- Features: 1692
- Class 0: 106
- Class 1: 53

## 3. Environment

- Python 3.10.20
- PyTorch 2.7.0
- NumPy 2.2.6
- Pandas 2.3.3
- SciPy 1.15.3
- Scikit-learn 1.7.2
- Matplotlib 3.10.9

安装依赖：

pip install -r requirements.txt

PyTorch CUDA版本请根据计算设备单独安装。

## 4. Reproduce Existing Model

在项目根目录执行：

python reproduce_84_38.py

该脚本不会重新训练模型。

执行内容：

1. 读取原始MAT数据。
2. 加载固定数据划分。
3. 加载264 VOC Mask。
4. 加载K-means Mask。
5. 初始化MultiView模型。
6. 加载已有模型权重。
7. 预测32个内部验证样本。
8. 比较原始预测结果。

## 5. Expected Results

Expected Accuracy:

0.84375

Expected Confusion Matrix:

[[18, 3],
 [2, 9]]

32个验证样本的预测类别和预测概率应与归档记录一致。

## 6. Directory Structure

- reproduce_84_38.py: 独立模型验证入口
- requirements.txt: Python依赖
- data/: 原始MAT数据存放位置
- 实验归档/06_84.38%_VOC面板实验归档/: 完整实验材料

## 7. Full Experiment Archive

完整Notebook、模型定义、训练配置、模型权重、固定数据划分、VOC统计、PCA、U检验及ROC结果均保存在06实验归档目录中。

Notebook部分单元格仍引用原项目result目录。独立验证请优先使用reproduce_84_38.py。

## 8. Reproducibility

本项目已使用保存的模型权重和固定验证集完成直接预测验证。

以下内容均与原始预测记录一致：

- Sample indices
- Ground-truth labels
- Predicted classes
- Predicted probabilities

本次验证不涉及重新训练。
