
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.io import loadmat
from scipy.stats import mannwhitneyu


# ============================================================
# 1. 路径设置
# ============================================================

ARCHIVE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

MAT_PATH = (
    PROJECT_ROOT
    / "data"
    / "voc_dataset_table1_1+2_vs_3_bpd.mat"
)

FEATURE_PATH = (
    ARCHIVE
    / "statistics"
    / "stable_features.csv"
)

OUT_DIR = (
    ARCHIVE
    / "statistics"
    / "U_test"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# 新增：多VOC小提琴图参数
TOP_N = 25
N_COLS = 5
RANDOM_SEED = 42

# 绘图颜色
PALETTE = ["#C7D0E3", "#9ADDD2"]

# 字体
plt.rcParams["font.sans-serif"] = [
    "DejaVu Sans",
    "SimHei",
    "Microsoft YaHei"
]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 2. 读取 MAT 数据和稳定 VOC 名单
# ============================================================

if not MAT_PATH.exists():
    raise FileNotFoundError(f"找不到MAT数据：{MAT_PATH}")

if not FEATURE_PATH.exists():
    raise FileNotFoundError(f"找不到VOC名单：{FEATURE_PATH}")

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
        if x.size == 1:
            x = x.item()
        else:
            x = x.reshape(-1)[0]

    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")

    return str(x)


feature_names = [
    decode_name(x)
    for x in mat["feature_names"].reshape(-1)
]


# ============================================================
# 4. 数据和 VOC 索引严格验证
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

assert np.isfinite(X).all()

columns = features[
    "VOC_Index"
].to_numpy(dtype=int)

assert np.all(columns >= 0)
assert np.all(columns < X.shape[1])
assert len(np.unique(columns)) == 264

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

    for item in errors[:10]:
        print(item)

    raise ValueError(
        f"发现{len(errors)}个VOC名称不匹配"
    )

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

assert n0 == 106
assert n1 == 53

print("\n===== 样本分组 =====")
print("Class 0：", n0)
print("Class 1：", n1)


# ============================================================
# 6. Mann-Whitney U 检验
# ============================================================

print("\n开始进行264个VOC的U检验...")

records = []

for j, (_, row) in enumerate(features.iterrows()):

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

    effect = (
        1.0
        - 2.0 * u_stat / (n0 * n1)
    )

    records.append({

        "VOC_Index": int(row["VOC_Index"]),
        "VOC_Name": row["VOC_Name"],
        "Selection_Frequency": row["Selection_Frequency"],
        "Soft_Mean": row["Soft_Mean"],
        "U_Statistic": u_stat,
        "P_Value": p_value,
        "Median_Class0": float(np.median(values0)),
        "Median_Class1": float(np.median(values1)),
        "Rank_Biserial": effect

    })


result_df = pd.DataFrame(records)


# ============================================================
# 7. Benjamini-Hochberg FDR 校正
# ============================================================

p = result_df["P_Value"].to_numpy()

m = len(p)

order = np.argsort(p)

ranked_p = p[order]

adjusted = (
    ranked_p
    * m
    / np.arange(1, m + 1)
)

adjusted = np.minimum.accumulate(
    adjusted[::-1]
)[::-1]

adjusted = np.clip(
    adjusted,
    0,
    1
)

q_values = np.empty(m)

q_values[order] = adjusted

result_df["FDR"] = q_values

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
# 9. 保存完整U检验结果
# ============================================================

result_df.to_csv(
    OUT_DIR / "VOC_U检验结果.csv",
    index=False,
    encoding="utf-8-sig"
)

significant = result_df[
    result_df["Significant_FDR05"]
].copy()

significant.to_csv(
    OUT_DIR / "VOC显著差异名单_FDR05.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. 输出统计结果
# ============================================================

print("\n" + "=" * 65)
print("二、Mann-Whitney U 检验结果")
print("=" * 65)

print("总检验VOC数量：", len(result_df))
print("原始P < 0.05：",
      int(result_df["Significant_P05"].sum()))

print("FDR < 0.05：", len(significant))
print("FDR < 0.01：",
      int((result_df["FDR"] < 0.01).sum()))

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
# 11. 原有U检验效应量散点图
# ============================================================

plt.figure(figsize=(11, 7))

x_plot = result_df["Rank_Biserial"].to_numpy()
q = result_df["FDR"].to_numpy()

y_plot = -np.log10(
    np.maximum(q, 1e-300)
)

significant_mask = q < 0.05

plt.scatter(
    x_plot[~significant_mask],
    y_plot[~significant_mask],
    alpha=0.65,
    s=28,
    label="FDR >= 0.05"
)

plt.scatter(
    x_plot[significant_mask],
    y_plot[significant_mask],
    alpha=0.85,
    s=38,
    label="FDR < 0.05"
)

plt.axhline(
    -np.log10(0.05),
    linestyle="--",
    linewidth=1.2,
    label="FDR = 0.05"
)

plt.axvline(
    0,
    color="gray",
    linewidth=0.8
)

plt.xlabel("Rank-biserial correlation", fontsize=12)
plt.ylabel("-log10(FDR)", fontsize=12)

plt.title(
    "Mann-Whitney U Test of 264 Stable VOC Features",
    fontsize=14
)

plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    OUT_DIR / "VOC_U检验结果图.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. 新增：师兄风格的VOC小提琴图
# Violin + Boxplot + Individual Samples
# ============================================================

print("\n开始生成多VOC小提琴图...")

# 按FDR排序，取前25个
plot_df = result_df.head(TOP_N).copy()

n_plots = len(plot_df)

n_rows = int(np.ceil(n_plots / N_COLS))

fig, axes = plt.subplots(
    n_rows,
    N_COLS,
    figsize=(20, 3.5 * n_rows),
    squeeze=False
)

axes = axes.flatten()

rng = np.random.default_rng(RANDOM_SEED)


def p_to_star(q_value):

    if q_value < 0.001:
        return "***"

    elif q_value < 0.01:
        return "**"

    elif q_value < 0.05:
        return "*"

    else:
        return "n.s."


for i, (_, row) in enumerate(plot_df.iterrows()):

    ax = axes[i]

    voc_idx = int(row["VOC_Index"])
    voc_name = str(row["VOC_Name"])

    # 使用原始MAT数据，不额外改变统计数据
    values0 = X[:, voc_idx][group0]
    values1 = X[:, voc_idx][group1]

    data_groups = [
        values0,
        values1
    ]

    # --------------------------------------------------------
    # 1. 小提琴图
    # --------------------------------------------------------

    violin = ax.violinplot(
        data_groups,
        positions=[0, 1],
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for body, color in zip(
        violin["bodies"],
        PALETTE
    ):

        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.80)
        body.set_linewidth(0.8)

    # --------------------------------------------------------
    # 2. 箱线图
    # --------------------------------------------------------

    bp = ax.boxplot(
        data_groups,
        positions=[0, 1],
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        medianprops={
            "color": "black",
            "linewidth": 1.2
        },
        whiskerprops={
            "color": "#555555",
            "linewidth": 0.9
        },
        capprops={
            "color": "#555555",
            "linewidth": 0.9
        }
    )

    for box in bp["boxes"]:

        box.set_facecolor("white")
        box.set_edgecolor("#555555")
        box.set_linewidth(0.9)

    # --------------------------------------------------------
    # 3. 单个样本散点
    # --------------------------------------------------------

    for position, values, color in zip(
        [0, 1],
        data_groups,
        PALETTE
    ):

        jitter = rng.uniform(
            -0.11,
            0.11,
            size=len(values)
        )

        ax.scatter(
            position + jitter,
            values,
            s=9,
            color=color,
            alpha=0.72,
            edgecolors="none",
            zorder=3
        )

    # --------------------------------------------------------
    # 4. 显著性横线和星号
    # --------------------------------------------------------

    q_value = float(row["FDR"])

    y_min = min(
        np.min(values0),
        np.min(values1)
    )

    y_max = max(
        np.max(values0),
        np.max(values1)
    )

    data_range = y_max - y_min

    if data_range == 0:
        data_range = max(abs(y_max), 1.0)

    bracket_y = y_max + data_range * 0.13
    bracket_h = data_range * 0.025

    ax.plot(
        [0, 0, 1, 1],
        [
            bracket_y,
            bracket_y + bracket_h,
            bracket_y + bracket_h,
            bracket_y
        ],
        color="#444444",
        linewidth=0.85
    )

    ax.text(
        0.5,
        bracket_y + bracket_h,
        p_to_star(q_value),
        ha="center",
        va="bottom",
        fontsize=9
    )

    # --------------------------------------------------------
    # 5. 标题及坐标轴
    # --------------------------------------------------------

    title = f"#{voc_idx}  {voc_name}"

    if len(title) > 30:
        title = title[:27] + "..."

    ax.set_title(
        title,
        fontsize=9,
        pad=8
    )

    ax.set_xticks([0, 1])

    ax.set_xticklabels(
        [
            f"Class 0\n(n={n0})",
            f"Class 1\n(n={n1})"
        ],
        fontsize=8
    )

    if i % N_COLS == 0:
        ax.set_ylabel(
            "Relative intensity (MAT scale)",
            fontsize=8
        )
    else:
        ax.set_ylabel("")

    ax.tick_params(
        axis="y",
        labelsize=7
    )

    ax.grid(
        axis="y",
        alpha=0.16,
        linewidth=0.5
    )

    ax.set_axisbelow(True)

    # 给显著性横线留出空间
    ax.set_ylim(
        bottom=min(0, y_min - data_range * 0.08),
        top=bracket_y + data_range * 0.16
    )

    # 边框美化
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# 删除多余空白子图
for j in range(n_plots, len(axes)):
    fig.delaxes(axes[j])


fig.suptitle(
    "Mann-Whitney U Test of Top 25 Stable VOC Features",
    fontsize=17,
    y=1.002
)

fig.tight_layout()

violin_path = (
    OUT_DIR
    / "VOC_U检验_Top25_小提琴箱线散点图.png"
)

fig.savefig(
    violin_path,
    dpi=400,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)


# 保存本次绘图所使用的VOC名单
plot_df.to_csv(
    OUT_DIR / "VOC_U检验_Top25绘图名单.csv",
    index=False,
    encoding="utf-8-sig"
)

print("多VOC小提琴图生成完成")
print("展示VOC数量：", n_plots)
print("图片保存位置：", violin_path)


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

    "Plot_Type": "Violin + Boxplot + Individual Samples",

    "Plot_VOC_Count": n_plots,

    "Plot_Ranking": "FDR ascending",

    "Plot_Significance": "FDR adjusted",

    "Analysis_Type":
        "Exploratory analysis using all 159 samples",

    "Independent_Test_Set_Evaluation": False
}

with open(
    OUT_DIR / "analysis_config.json",
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
print("原始P < 0.05：",
      int(result_df["Significant_P05"].sum()))
print("FDR显著VOC数量：", len(significant))

print("\n结果保存位置：")
print(OUT_DIR)

print("\n新增图片：")
print("VOC_U检验_Top25_小提琴箱线散点图.png")

print("=" * 65)

print(
    "\n注意：本分析使用全部159个样本，"
    "属于探索性差异分析，"
    "不代表独立测试集验证结果。"
)
