
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 84.38%实验：264个稳定VOC选择频率与重要性图
# ============================================================

ROOT = os.path.dirname(os.path.abspath(__file__))

SOURCE = os.path.join(
    ROOT,
    "result",
    "table1_10x10",
    "feature_selection_frequency.csv"
)

OUT_DIR = os.path.join(
    ROOT,
    "实验归档",
    "06_84.38%_VOC面板实验归档",
    "statistics"
)

os.makedirs(OUT_DIR, exist_ok=True)

# 1. 读取频率统计
df = pd.read_csv(SOURCE)

required = [
    "VOC_Index",
    "VOC_Name",
    "Selection_Frequency",
    "Soft_Mean",
    "Soft_Std",
    "Soft_SEM"
]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(f"缺少字段：{missing}")

# 2. 筛选264个稳定VOC
stable = df[
    df["Selection_Frequency"] > 0.60
].copy()

stable = stable.sort_values(
    ["Selection_Frequency", "Soft_Mean"],
    ascending=[False, False]
).reset_index(drop=True)

if len(stable) != 264:
    raise ValueError(
        f"稳定VOC数量应为264，实际为{len(stable)}"
    )

print("稳定VOC数量：", len(stable))

# 3. 准备数据
x = np.arange(len(stable))

frequency = stable[
    "Selection_Frequency"
].to_numpy()

soft_mean = stable[
    "Soft_Mean"
].to_numpy()

soft_sem = stable[
    "Soft_SEM"
].to_numpy()

soft_std = stable[
    "Soft_Std"
].to_numpy()

# 4. 绘图
fig, axes = plt.subplots(
    2, 1,
    figsize=(18, 10),
    sharex=True
)

# 上图：选择频率
axes[0].bar(
    x,
    frequency,
    width=0.8,
    color="lightseagreen",
    edgecolor="white",
    linewidth=0.3
)

axes[0].axhline(
    0.60,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label="Selection Threshold = 0.60"
)

axes[0].set_ylabel("Selection Frequency")
axes[0].set_title(
    "Selection Frequency of 264 Stable VOC Features"
)

axes[0].set_ylim(0, 1.08)
axes[0].legend(loc="upper right")
axes[0].grid(axis="y", alpha=0.2)

# 下图：Soft Importance
axes[1].bar(
    x,
    soft_mean,
    width=0.8,
    color="lightsalmon",
    edgecolor="white",
    linewidth=0.3,
    label="Mean Soft Importance"
)

# SEM误差线
axes[1].errorbar(
    x,
    soft_mean,
    yerr=soft_sem,
    fmt="none",
    ecolor="gray",
    elinewidth=0.7,
    capsize=2,
    label="± SEM"
)

# 标准差范围
axes[1].fill_between(
    x,
    np.maximum(0, soft_mean - soft_std),
    soft_mean + soft_std,
    color="gray",
    alpha=0.15,
    label="± Standard Deviation"
)

axes[1].set_xlabel("VOC Index (Ranked)")
axes[1].set_ylabel("Soft Importance")
axes[1].set_title(
    "Soft Importance of 264 Stable VOC Features"
)

axes[1].legend(loc="upper right")
axes[1].grid(axis="y", alpha=0.2)

# 5. 保存高清图片
plt.tight_layout()

output_path = os.path.join(
    OUT_DIR,
    "stable_feature_importance_264.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# 6. 同时保存排序后的数据
stable.to_csv(
    os.path.join(
        OUT_DIR,
        "stable_feature_importance_264.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("=" * 60)
print("264个稳定VOC特征重要性图生成完成")
print("VOC数量：", len(stable))
print("保存位置：", output_path)
print("=" * 60)
