
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径
# ============================================================

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)

PCA_SCORE_PATH = os.path.join(
    os.path.dirname(__file__),
    "PCA分析结果",
    "PCA样本得分.csv"
)

OUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "PCA三分类结果"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 2. 读取之前计算好的 PCA 得分
# 不重新计算 PCA
# ============================================================

df = pd.read_csv(PCA_SCORE_PATH)

required_columns = [
    "Original_Class",
    "PC1",
    "PC2"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"缺少字段：{col}")

df["Original_Class"] = pd.to_numeric(
    df["Original_Class"]
)

print("=" * 60)
print("三分类 PCA 数据检查")
print("=" * 60)

print("样本数量：", len(df))

print("\n原始类别分布：")
print(
    df["Original_Class"]
    .value_counts()
    .sort_index()
)

assert len(df) == 159

assert set(
    df["Original_Class"].unique()
) == {1, 2, 3}

assert (
    df["Original_Class"]
    .value_counts()
    .sort_index()
    .to_list()
    == [53, 53, 53]
)


# ============================================================
# 3. 绘制三分类 PCA 得分图
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 8)
)

class_labels = {
    1: "Class 1",
    2: "Class 2",
    3: "Class 3"
}

colors = {
    1: "#2878B5",
    2: "#E07B39",
    3: "#269B70"
}

markers = {
    1: "o",
    2: "s",
    3: "^"
}

for class_value in [1, 2, 3]:

    subset = df[
        df["Original_Class"] == class_value
    ]

    ax.scatter(
        subset["PC1"],
        subset["PC2"],
        s=55,
        alpha=0.78,
        color=colors[class_value],
        marker=markers[class_value],
        edgecolors="white",
        linewidths=0.5,
        label=(
            f"{class_labels[class_value]} "
            f"(n={len(subset)})"
        )
    )


# ============================================================
# 4. 图形格式
# ============================================================

ax.set_xlabel(
    "PC1 (21.26%)",
    fontsize=13
)

ax.set_ylabel(
    "PC2 (10.41%)",
    fontsize=13
)

ax.set_title(
    "Three-Class PCA Score Plot "
    "of 264 Stable VOC Features",
    fontsize=15
)

ax.legend(
    fontsize=11,
    frameon=True
)

ax.grid(
    alpha=0.2
)

fig.tight_layout()


# ============================================================
# 5. 保存图片
# ============================================================

output_path = os.path.join(
    OUT_DIR,
    "PCA三分类得分散点图.png"
)

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 6. 保存三分类样本得分
# ============================================================

df.to_csv(
    os.path.join(
        OUT_DIR,
        "PCA三分类样本得分.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 7. 输出各类别 PCA 中心
# ============================================================

centers = (
    df.groupby("Original_Class")[["PC1", "PC2"]]
    .agg(["mean", "std"])
)

centers.to_csv(
    os.path.join(
        OUT_DIR,
        "PCA三分类中心统计.csv"
    ),
    encoding="utf-8-sig"
)

print("\n===== PCA三分类分析完成 =====")

print("各类别 PCA 中心：")

print(
    df.groupby("Original_Class")[["PC1", "PC2"]]
    .mean()
)

print("\n图片保存位置：")
print(output_path)

print("\n分析结果目录：")
print(OUT_DIR)
