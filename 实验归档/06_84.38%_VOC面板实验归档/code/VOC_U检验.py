
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.stats import mannwhitneyu


# ============================================================
# 1. 路径设置
# ============================================================

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)

MAT_PATH = os.path.join(
    ROOT,
    "data",
    "voc_dataset_table1_1+2_vs_3_bpd.mat"
)

FEATURE_PATH = os.path.join(
    ROOT,
    "result",
    "table1_10x10",
    "stable_features.csv"
)

OUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "VOC_U检验结果"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 2. 读取 MAT 数据和稳定 VOC 名单
# ============================================================

mat = loadmat(MAT_PATH)

X = np.asarray(
    mat["X"],
    dtype=np.float64
)

y = np.asarray(
    mat["y"]
).reshape(-1).astype(int)

features = pd.read_csv(FEATURE_PATH)


# ============================================================
# 3. 解码 MAT 中的 VOC 名称
# ============================================================

def decode_name(x):

    while isinstance(x, np.ndarray):
        x = x.item()

    return str(x)


feature_names = [
    decode_name(x)
    for x in mat["feature_names"].reshape(-1)
]


# ============================================================
# 4. 数据和 VOC 索引严格验证
#
# 重要：
# VOC_Index 是 X 矩阵中的零起始列位置
# 不能使用 feature_ids 进行二次映射
# ============================================================

print("=" * 65)
print("一、数据及特征索引检查")
print("=" * 65)

assert X.shape == (159, 1692), (
    f"X形状错误：{X.shape}"
)

assert len(y) == 159

assert len(features) == 264, (
    f"稳定VOC数量错误：{len(features)}"
)

assert np.isin(y, [0, 1]).all()

assert np.isfinite(X).all(), (
    "X中存在NaN或无穷值，请先检查原始数据"
)

# 直接使用 VOC_Index 定位 X 的列
columns = features[
    "VOC_Index"
].to_numpy(dtype=int)

# 检查索引范围
assert np.all(columns >= 0)
assert np.all(columns < X.shape[1])

# 检查是否存在重复索引
assert len(np.unique(columns)) == 264

# 检查名称与 MAT 特征名称是否完全匹配
errors = []

for _, row in features.iterrows():

    col = int(row["VOC_Index"])

    csv_name = str(row["VOC_Name"])
    mat_name = feature_names[col]

    if csv_name != mat_name:

        errors.append({
            "VOC_Index": col,
            "CSV_Name": csv_name,
            "MAT_Name": mat_name
        })

if errors:

    print("VOC名称匹配失败：")

    for item in errors[:10]:
        print(item)

    raise ValueError(
        f"发现{len(errors)}个VOC名称不匹配，停止分析"
    )

# 提取264个稳定VOC
X_selected = X[:, columns]

assert X_selected.shape == (159, 264)

print("原始样本数量：", len(y))
print("原始特征数量：", X.shape[1])
print("稳定VOC数量：", X_selected.shape[1])
print("VOC索引匹配错误：", len(errors))
print("VOC名称与MAT列位置全部匹配！")


# ============================================================
# 5. 类别检查
# ============================================================

group0 = y == 0
group1 = y == 1

n0 = int(group0.sum())
n1 = int(group1.sum())

assert n0 == 106, (
    f"Class 0数量异常：{n0}"
)

assert n1 == 53, (
    f"Class 1数量异常：{n1}"
)

print("\n===== 样本分组 =====")

print("Class 0：", n0)
print("Class 1：", n1)

print("Class 0 + Class 1：", n0 + n1)


# ============================================================
# 6. Mann-Whitney U 检验
#
# 双侧检验
# Class 0 vs Class 1
# ============================================================

records = []

for j, (_, row) in enumerate(
    features.iterrows()
):

    values0 = X_selected[group0, j]
    values1 = X_selected[group1, j]

    result = mannwhitneyu(
        values0,
        values1,
        alternative="two-sided",
        method="asymptotic"
    )

    u_stat = float(result.statistic)

    p_value = float(result.pvalue)

    # --------------------------------------------------------
    # Rank-biserial correlation
    #
    # scipy 返回的是以 Class 0 为第一组的 U 统计量
    #
    # 正值：Class 1 整体取值倾向高于 Class 0
    # 负值：Class 1 整体取值倾向低于 Class 0
    # --------------------------------------------------------

    effect = (
        1.0
        -
        2.0 * u_stat / (n0 * n1)
    )

    records.append({

        "VOC_Index":
            int(row["VOC_Index"]),

        "VOC_Name":
            row["VOC_Name"],

        "Selection_Frequency":
            row["Selection_Frequency"],

        "Soft_Mean":
            row["Soft_Mean"],

        "U_Statistic":
            u_stat,

        "P_Value":
            p_value,

        "Median_Class0":
            float(np.median(values0)),

        "Median_Class1":
            float(np.median(values1)),

        "Rank_Biserial":
            effect
    })


result_df = pd.DataFrame(records)


# ============================================================
# 7. Benjamini-Hochberg FDR 多重检验校正
# ============================================================

p = result_df[
    "P_Value"
].to_numpy()

m = len(p)

# 按P值升序排列
order = np.argsort(p)

ranked_p = p[order]

# BH 校正
adjusted = (
    ranked_p
    * m
    / np.arange(1, m + 1)
)

# 从后向前保证单调性
adjusted = np.minimum.accumulate(
    adjusted[::-1]
)[::-1]

adjusted = np.clip(
    adjusted,
    0,
    1
)

# 恢复原始顺序
q_values = np.empty(m)

q_values[order] = adjusted

result_df["FDR"] = q_values

# 显著性判断
result_df["Significant_P05"] = (
    result_df["P_Value"] < 0.05
)

result_df["Significant_FDR05"] = (
    result_df["FDR"] < 0.05
)


# ============================================================
# 8. 排序
# ============================================================

result_df = result_df.sort_values(
    ["FDR", "P_Value"],
    ascending=[True, True]
).reset_index(drop=True)


# ============================================================
# 9. 保存完整 U 检验结果
# ============================================================

result_path = os.path.join(
    OUT_DIR,
    "VOC_U检验结果.csv"
)

result_df.to_csv(
    result_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. 保存显著 VOC 名单
# ============================================================

significant = result_df[
    result_df["Significant_FDR05"]
].copy()

significant_path = os.path.join(
    OUT_DIR,
    "VOC显著差异名单_FDR05.csv"
)

significant.to_csv(
    significant_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. 输出统计结果
# ============================================================

print("\n" + "=" * 65)
print("二、Mann-Whitney U 检验结果")
print("=" * 65)

print("总检验VOC数量：", len(result_df))

print(
    "原始P < 0.05：",
    int(result_df["Significant_P05"].sum())
)

print(
    "FDR < 0.05：",
    len(significant)
)

print(
    "FDR < 0.01：",
    int((result_df["FDR"] < 0.01).sum())
)

print("\n显著差异VOC前20名：")

print(
    significant[
        [
            "VOC_Index",
            "VOC_Name",
            "Median_Class0",
            "Median_Class1",
            "P_Value",
            "FDR",
            "Rank_Biserial"
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# 12. 绘制 U 检验散点图
#
# 横轴：Rank-biserial correlation
# 纵轴：-log10(FDR)
# ============================================================

plt.figure(figsize=(11, 7))

x_plot = result_df[
    "Rank_Biserial"
].to_numpy()

q = result_df[
    "FDR"
].to_numpy()

y_plot = -np.log10(
    np.maximum(q, 1e-300)
)

significant_mask = q < 0.05

# 未显著 VOC
plt.scatter(
    x_plot[~significant_mask],
    y_plot[~significant_mask],
    alpha=0.65,
    s=28,
    label="FDR >= 0.05"
)

# 显著 VOC
plt.scatter(
    x_plot[significant_mask],
    y_plot[significant_mask],
    alpha=0.85,
    s=38,
    label="FDR < 0.05"
)

# FDR阈值
plt.axhline(
    -np.log10(0.05),
    linestyle="--",
    linewidth=1.2,
    label="FDR = 0.05"
)

# 效应量零点
plt.axvline(
    0,
    color="gray",
    linewidth=0.8
)

plt.xlabel(
    "Rank-biserial correlation",
    fontsize=12
)

plt.ylabel(
    "-log10(FDR)",
    fontsize=12
)

plt.title(
    "Mann-Whitney U Test of 264 Stable VOC Features",
    fontsize=14
)

plt.legend(
    frameon=True
)

plt.grid(
    alpha=0.2
)

plt.tight_layout()

figure_path = os.path.join(
    OUT_DIR,
    "VOC_U检验结果图.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. 保存统计配置
# ============================================================

config = {

    "Sample_Count": len(y),

    "Class_0_Count": n0,

    "Class_1_Count": n1,

    "Stable_VOC_Count": len(features),

    "Test_Method": "Mann-Whitney U",

    "Alternative": "two-sided",

    "Multiple_Test_Correction": "Benjamini-Hochberg",

    "Significance_Level": 0.05,

    "Effect_Size": "Rank-biserial correlation",

    "Effect_Positive_Meaning":
        "Class 1 tends to have higher values than Class 0",

    "Analysis_Type":
        "Exploratory analysis using all 159 samples",

    "Independent_Test_Set_Evaluation": False
}

import json

with open(
    os.path.join(OUT_DIR, "analysis_config.json"),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        ensure_ascii=False,
        indent=4
    )


# ============================================================
# 14. 最终报告
# ============================================================

print("\n" + "=" * 65)

print("Mann-Whitney U 检验全部完成")

print("=" * 65)

print("检验VOC数量：", len(result_df))

print("FDR显著VOC数量：", len(significant))

print("\n结果保存位置：")

print(OUT_DIR)

print("\n生成文件：")

print("1. VOC_U检验结果.csv")

print("2. VOC显著差异名单_FDR05.csv")

print("3. VOC_U检验结果图.png")

print("4. analysis_config.json")

print("=" * 65)

print(
    "\n注意：本分析使用全部159个样本，"
    "属于探索性差异分析，"
    "不代表独立测试集验证结果。"
)
