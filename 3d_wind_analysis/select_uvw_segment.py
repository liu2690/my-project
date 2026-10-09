import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径
# ============================================================

THIS_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    THIS_DIR,
    "data",
    "UVW_unstable_benchmark.nc"
)

RESULT_DIR = os.path.join(
    THIS_DIR,
    "result",
    "selected_segment"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# 2. 参数
# ============================================================

FS = 20.0

WINDOW_SIZE = 5000

# 为避免扫描180万个窗口导致计算太慢，
# 每隔1000点检查一个窗口
SCAN_STEP = 1000

CHANNELS = [
    "u_110_5m",
    "v_110_5m",
    "w_110_5m",
    "u_179_5m",
    "v_179_5m",
    "w_179_5m",
]


# ============================================================
# 3. 读取数据
# ============================================================

print("\n" + "=" * 80)
print("读取 UVW 数据")
print("=" * 80)

df = pd.read_csv(
    DATA_PATH
)

df["time20hz"] = pd.to_datetime(
    df["time20hz"]
)

X = df[
    CHANNELS
].to_numpy(
    dtype=float
)


print(
    "完整数据 shape =",
    X.shape
)

print(
    "采样频率 =",
    FS,
    "Hz"
)

print(
    "窗口长度 =",
    WINDOW_SIZE,
    "points"
)

print(
    "窗口持续时间 =",
    WINDOW_SIZE / FS,
    "s"
)


# ============================================================
# 4. 全局统计
# ============================================================

global_mean = np.mean(
    X,
    axis=0
)

global_std = np.std(
    X,
    axis=0
)


lower_6sigma = (
    global_mean
    -
    6.0 * global_std
)

upper_6sigma = (
    global_mean
    +
    6.0 * global_std
)


print("\n全局均值：")

for col, value in zip(
    CHANNELS,
    global_mean
):
    print(
        f"{col:15s}",
        f"{value:10.4f}"
    )


print("\n全局标准差：")

for col, value in zip(
    CHANNELS,
    global_std
):
    print(
        f"{col:15s}",
        f"{value:10.4f}"
    )


# ============================================================
# 5. 扫描候选窗口
# ============================================================

print("\n" + "=" * 80)
print("开始扫描 5000 点连续窗口")
print("=" * 80)


records = []


for start in range(
    0,
    len(X) - WINDOW_SIZE + 1,
    SCAN_STEP
):

    end = (
        start
        +
        WINDOW_SIZE
    )

    segment = X[
        start:end
    ]


    # ----------------------------------------
    # 极端值数量
    # ----------------------------------------

    extreme_mask = (
        (segment < lower_6sigma)
        |
        (segment > upper_6sigma)
    )

    extreme_count = int(
        np.sum(
            extreme_mask
        )
    )


    # ----------------------------------------
    # 每个通道窗口标准差
    # ----------------------------------------

    seg_std = np.std(
        segment,
        axis=0
    )


    # 归一化波动强度
    normalized_std = (
        seg_std
        /
        global_std
    )


    variability_score = float(
        np.mean(
            normalized_std
        )
    )


    # ----------------------------------------
    # 同分量两个测点的相关系数
    # ----------------------------------------

    corr_u = np.corrcoef(
        segment[:, 0],
        segment[:, 3]
    )[0, 1]

    corr_v = np.corrcoef(
        segment[:, 1],
        segment[:, 4]
    )[0, 1]

    corr_w = np.corrcoef(
        segment[:, 2],
        segment[:, 5]
    )[0, 1]


    mean_abs_corr = float(
        np.mean(
            np.abs(
                [
                    corr_u,
                    corr_v,
                    corr_w
                ]
            )
        )
    )


    # ----------------------------------------
    # 综合评分
    #
    # 希望：
    # 1. 无极端点
    # 2. 具有一定波动
    # 3. 两测点存在空间关联
    # ----------------------------------------

    score = (
        variability_score
        +
        0.5 * mean_abs_corr
        -
        0.1 * extreme_count
    )


    records.append(
        {
            "start_index":
                start,

            "end_index":
                end,

            "start_time":
                df["time20hz"].iloc[start],

            "end_time":
                df["time20hz"].iloc[end - 1],

            "extreme_count":
                extreme_count,

            "variability_score":
                variability_score,

            "corr_u":
                corr_u,

            "corr_v":
                corr_v,

            "corr_w":
                corr_w,

            "mean_abs_corr":
                mean_abs_corr,

            "score":
                score,
        }
    )


scan_df = pd.DataFrame(
    records
)


# ============================================================
# 6. 优先选择没有极端值的窗口
# ============================================================

clean_df = scan_df[
    scan_df[
        "extreme_count"
    ] == 0
].copy()


if len(clean_df) > 0:

    candidate_df = (
        clean_df
        .sort_values(
            "score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

else:

    print(
        "警告：没有找到完全无极端值窗口，"
        "改为从全部窗口中选择。"
    )

    candidate_df = (
        scan_df
        .sort_values(
            "score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 7. 保存完整扫描结果
# ============================================================

scan_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "uvw_window_scan_all.csv"
    ),
    index=False
)


candidate_df.head(
    20
).to_csv(
    os.path.join(
        RESULT_DIR,
        "uvw_top20_candidates.csv"
    ),
    index=False
)


# ============================================================
# 8. 打印前10候选
# ============================================================

print("\n" + "=" * 100)
print("前10个候选区间")
print("=" * 100)


print(
    candidate_df[
        [
            "start_index",
            "end_index",
            "start_time",
            "end_time",
            "extreme_count",
            "variability_score",
            "corr_u",
            "corr_v",
            "corr_w",
            "score",
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 9. 选评分最高的候选
# ============================================================

best = candidate_df.iloc[
    0
]


BEST_START = int(
    best[
        "start_index"
    ]
)

BEST_END = int(
    best[
        "end_index"
    ]
)


print("\n" + "=" * 80)
print("自动选择的候选区间")
print("=" * 80)

print(
    "start index =",
    BEST_START
)

print(
    "end index =",
    BEST_END
)

print(
    "start time =",
    best[
        "start_time"
    ]
)

print(
    "end time =",
    best[
        "end_time"
    ]
)

print(
    "extreme count =",
    best[
        "extreme_count"
    ]
)

print(
    "corr u =",
    best[
        "corr_u"
    ]
)

print(
    "corr v =",
    best[
        "corr_v"
    ]
)

print(
    "corr w =",
    best[
        "corr_w"
    ]
)


# ============================================================
# 10. 截取原始矩阵
# ============================================================

selected = X[
    BEST_START:BEST_END
]


# 转成：
# 6 x 5000
Q_raw = selected.T


print(
    "\nQ_raw shape =",
    Q_raw.shape
)


# ============================================================
# 11. 构造脉动风速
# ============================================================

channel_mean = np.mean(
    Q_raw,
    axis=1,
    keepdims=True
)


Q_fluctuation = (
    Q_raw
    -
    channel_mean
)


print(
    "Q_fluctuation shape =",
    Q_fluctuation.shape
)


print(
    "\n脉动风速各通道均值："
)

print(
    np.mean(
        Q_fluctuation,
        axis=1
    )
)


# ============================================================
# 12. 重构检查
# ============================================================

reconstruction_error = np.max(
    np.abs(
        Q_raw
        -
        (
            channel_mean
            +
            Q_fluctuation
        )
    )
)


print(
    "\n去均值最大重构误差 =",
    reconstruction_error
)


# ============================================================
# 13. 保存数据
# ============================================================

np.save(
    os.path.join(
        RESULT_DIR,
        "uvw_selected_raw.npy"
    ),
    Q_raw
)


np.save(
    os.path.join(
        RESULT_DIR,
        "uvw_selected_fluctuation.npy"
    ),
    Q_fluctuation
)


np.save(
    os.path.join(
        RESULT_DIR,
        "uvw_selected_mean.npy"
    ),
    channel_mean
)


with open(
    os.path.join(
        RESULT_DIR,
        "selected_time.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"start_index = {BEST_START}\n"
    )

    f.write(
        f"end_index = {BEST_END}\n"
    )

    f.write(
        f"start_time = {best['start_time']}\n"
    )

    f.write(
        f"end_time = {best['end_time']}\n"
    )

    f.write(
        f"FS = {FS} Hz\n"
    )

    f.write(
        f"N = {WINDOW_SIZE}\n"
    )


# ============================================================
# 14. 绘制选中区间原始数据
# ============================================================

time_s = (
    np.arange(
        WINDOW_SIZE
    )
    /
    FS
)


fig, ax = plt.subplots(
    figsize=(13, 6)
)


for i, col in enumerate(
    CHANNELS
):

    ax.plot(
        time_s,
        Q_raw[i],
        linewidth=0.8,
        label=col
    )


ax.set_xlabel(
    "Time (s)"
)

ax.set_ylabel(
    "Velocity (m/s)"
)

ax.set_title(
    "Selected UVW raw velocity"
)

ax.legend(
    ncol=2
)

ax.grid(
    True,
    alpha=0.3
)


fig.tight_layout()


fig.savefig(
    os.path.join(
        RESULT_DIR,
        "uvw_selected_raw.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# 15. 绘制选中区间脉动数据
# ============================================================

fig, ax = plt.subplots(
    figsize=(13, 6)
)


for i, col in enumerate(
    CHANNELS
):

    ax.plot(
        time_s,
        Q_fluctuation[i],
        linewidth=0.8,
        label=col
    )


ax.set_xlabel(
    "Time (s)"
)

ax.set_ylabel(
    "Velocity fluctuation (m/s)"
)

ax.set_title(
    "Selected UVW velocity fluctuation"
)

ax.legend(
    ncol=2
)

ax.grid(
    True,
    alpha=0.3
)


fig.tight_layout()


fig.savefig(
    os.path.join(
        RESULT_DIR,
        "uvw_selected_fluctuation.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# 16. 分别绘制 u / v / w
# ============================================================

component_info = [
    (
        "u",
        0,
        3
    ),
    (
        "v",
        1,
        4
    ),
    (
        "w",
        2,
        5
    ),
]


for name, i1, i2 in component_info:

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        time_s,
        Q_fluctuation[i1],
        linewidth=0.8,
        label=CHANNELS[i1]
    )

    ax.plot(
        time_s,
        Q_fluctuation[i2],
        linewidth=0.8,
        label=CHANNELS[i2]
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        f"{name}' (m/s)"
    )

    ax.set_title(
        f"Selected {name} fluctuation"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            RESULT_DIR,
            f"selected_{name}_fluctuation.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# 17. 完成
# ============================================================

print("\n" + "=" * 80)
print("三分量风速实验段选择完成")
print("=" * 80)

print(
    "结果目录："
)

print(
    RESULT_DIR
)

print(
    "\n生成的数据："
)

print(
    "uvw_selected_raw.npy"
)

print(
    "uvw_selected_fluctuation.npy"
)

print(
    "uvw_selected_mean.npy"
)

print(
    "\n生成图片："
)

print(
    "uvw_selected_raw.png"
)

print(
    "uvw_selected_fluctuation.png"
)

print(
    "selected_u_fluctuation.png"
)

print(
    "selected_v_fluctuation.png"
)

print(
    "selected_w_fluctuation.png"
)

print("=" * 80)