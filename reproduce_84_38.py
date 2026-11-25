
import os
import json
import numpy as np
import pandas as pd
import torch

from scipy.io import loadmat
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)

from pathlib import Path

# ============================================================
# 1. 路径设置
# ============================================================

ROOT = Path(__file__).resolve().parent

ARCHIVE = (
    ROOT
    / "实验归档"
    / "06_84.38%_VOC面板实验归档"
)

MAT_PATH = (
    ROOT
    / "data"
    / "voc_dataset_table1_1+2_vs_3_bpd.mat"
)

MODEL_PATH = ARCHIVE / "model" / "model_checkpoint.pt"

SPLIT_PATH = ARCHIVE / "data_split" / "data_split.npz"

MASK_PATH = ARCHIVE / "data_split" / "gt60_mask.npy"

KMEANS_PATH = ARCHIVE / "data_split" / "kmeans_mask.npy"

ORIGINAL_PRED_PATH = (
    ARCHIVE
    / "results"
    / "validation_predictions.csv"
)

OUT_DIR = ARCHIVE / "results" / "reproduction_check"

OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DTYPE = torch.float32

print("=" * 65)
print("BPD-MLP 84.38% 独立模型复现检查")
print("=" * 65)

print("运行设备：", DEVICE)

# ============================================================
# 2. 读取原始数据
# ============================================================

if not MAT_PATH.exists():
    raise FileNotFoundError(
        f"找不到MAT数据：{MAT_PATH}"
    )

mat = loadmat(MAT_PATH)

X = np.asarray(mat["X"], dtype=np.float32)

y = np.asarray(
    mat["y"].reshape(-1),
    dtype=np.int64
)

assert X.shape == (159, 1692)
assert y.shape == (159,)

assert np.sum(y == 0) == 106
assert np.sum(y == 1) == 53

print("\n数据检查通过")
print("X:", X.shape)
print("y:", y.shape)

# ============================================================
# 3. 读取固定划分
# ============================================================

split = np.load(SPLIT_PATH)

train_indices = split["train_indices"]
val_indices = split["val_indices"]

development_indices = split["development_indices"]
test_indices = split["test_indices"]

assert len(train_indices) == 95
assert len(val_indices) == 32

assert len(development_indices) == 127
assert len(test_indices) == 32

assert len(np.intersect1d(
    train_indices, val_indices
)) == 0

assert np.array_equal(
    np.sort(np.concatenate([
        train_indices,
        val_indices
    ])),
    np.sort(development_indices)
)

print("\n固定划分检查通过")
print("训练集：", len(train_indices))
print("验证集：", len(val_indices))
print("独立测试集：", len(test_indices))

# ============================================================
# 4. 读取Mask
# ============================================================

stable_mask = np.load(MASK_PATH)

kmeans_mask = np.load(KMEANS_PATH)

assert stable_mask.shape == (1692,)
assert int(stable_mask.sum()) == 264

assert kmeans_mask.shape == (3, 1692)

print("\nMask检查通过")
print("稳定VOC：", int(stable_mask.sum()))
print("K-means Mask：", kmeans_mask.shape)

# ============================================================
# 5. 初始化原始MultiView模型
# ============================================================

import sys

sys.path.insert(
    0,
    str(ARCHIVE / "code")
)

from modules import MultiView

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

config = checkpoint["model_config"]

model = MultiView(
    k_mean_mask=torch.as_tensor(
        kmeans_mask,
        dtype=DTYPE
    ),
    num_blocks=config["num_blocks"],
    head_dim=config["head_dim"],
    num_class=config["num_class"],
    temperature=config["temperature"]
)

missing, unexpected = model.load_state_dict(
    checkpoint["state_dict"],
    strict=True
)

assert len(missing) == 0
assert len(unexpected) == 0

model = model.to(
    DEVICE,
    dtype=DTYPE
)

model.eval()

print("\n模型权重加载成功")
print("Seed：", checkpoint["seed"])
print("VOC数量：", checkpoint["VOC_Count"])

# ============================================================
# 6. 使用固定验证集直接预测
# 不训练，不更新任何参数
# ============================================================

X_val = torch.as_tensor(
    X[val_indices],
    dtype=DTYPE,
    device=DEVICE
)

y_val = y[val_indices]

mask_tensor = torch.as_tensor(
    stable_mask,
    dtype=DTYPE,
    device=DEVICE
)

with torch.no_grad():

    logits = model.mlp_classifier(
        X_val * mask_tensor
    )

    probabilities = torch.softmax(
        logits.float(),
        dim=1
    )[:, 1].cpu().numpy()

predictions = (
    probabilities >= 0.5
).astype(int)

# ============================================================
# 7. 计算指标
# ============================================================

cm = confusion_matrix(
    y_val,
    predictions,
    labels=[0, 1]
)

accuracy = accuracy_score(
    y_val,
    predictions
)

sensitivity = (
    cm[1, 1] / max(cm[1].sum(), 1)
)

specificity = (
    cm[0, 0] / max(cm[0].sum(), 1)
)

f1 = f1_score(
    y_val,
    predictions,
    zero_division=0
)

auc_value = roc_auc_score(
    y_val,
    probabilities
)

# ============================================================
# 8. 与原始预测逐样本比较
# ============================================================

original = pd.read_csv(
    ORIGINAL_PRED_PATH
)

current = pd.DataFrame({
    "Sample_Index": val_indices,
    "Target": y_val,
    "Probability_Class1": probabilities,
    "Prediction": predictions
})

original = original.sort_values(
    "Sample_Index"
).reset_index(drop=True)

current = current.sort_values(
    "Sample_Index"
).reset_index(drop=True)

same_indices = np.array_equal(
    original["Sample_Index"].to_numpy(),
    current["Sample_Index"].to_numpy()
)

same_targets = np.array_equal(
    original["Target"].to_numpy(),
    current["Target"].to_numpy()
)

same_predictions = np.array_equal(
    original["Prediction"].to_numpy(),
    current["Prediction"].to_numpy()
)

same_probabilities = np.allclose(
    original["Probability_Class1"].to_numpy(),
    current["Probability_Class1"].to_numpy(),
    rtol=1e-5,
    atol=1e-7
)

# ============================================================
# 9. 保存复现结果
# ============================================================

current.to_csv(
    OUT_DIR / "reproduced_predictions.csv",
    index=False
)

metrics = {
    "Validation_Samples": len(y_val),
    "VOC_Count": int(stable_mask.sum()),
    "Accuracy": float(accuracy),
    "Sensitivity": float(sensitivity),
    "Specificity": float(specificity),
    "F1": float(f1),
    "AUC": float(auc_value),
    "Same_Indices": bool(same_indices),
    "Same_Targets": bool(same_targets),
    "Same_Predictions": bool(same_predictions),
    "Same_Probabilities": bool(same_probabilities)
}

with open(
    OUT_DIR / "reproduction_metrics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        ensure_ascii=False,
        indent=4
    )

# ============================================================
# 10. 输出
# ============================================================

print("\n" + "=" * 65)
print("独立验证集预测结果")
print("=" * 65)

print("Accuracy：", accuracy)
print("Sensitivity：", sensitivity)
print("Specificity：", specificity)
print("F1：", f1)
print("AUC：", auc_value)

print("\n混淆矩阵：")
print(cm)

print("\n原始预测一致性检查")
print("样本索引一致：", same_indices)
print("真实标签一致：", same_targets)
print("预测类别一致：", same_predictions)
print("预测概率一致：", same_probabilities)

print("\n结果保存位置：")
print(OUT_DIR)

print("=" * 65)
