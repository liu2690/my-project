import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径设置
# ============================================================

THIS_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    THIS_DIR,
    "data"
)

RESULT_DIR = os.path.join(
    THIS_DIR,
    "result"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


DATA_PATH = os.path.join(
    DATA_DIR,
    "UVW_unstable_benchmark.nc"
)


# ============================================================
# 2. 基本参数
# ============================================================

FS = 20.0

DT_EXPECTED = 1.0 / FS

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
print("读取真实三分量 UVW 风速数据")
print("=" * 80)


df = pd.read_csv(
    DATA_PATH
)


print(
    "完整数据 shape =",
    df.shape
)

print(
    "数据列："
)

for col in df.columns:
    print("   ", col)


# ============================================================
# 4. 时间转换
# ============================================================

df["time20hz"] = pd.to_datetime(
    df["time20hz"]
)


print("\n时间范围：")

print(
    "开始：",
    df["time20hz"].iloc[0]
)

print(
    "结束：",
    df["time20hz"].iloc[-1]
)


duration_seconds = (
    df["time20hz"].iloc[-1]
    -
    df["time20hz"].iloc[0]
).total_seconds()


print(
    "时间跨度 =",
    duration_seconds,
    "s"
)

print(
    "约 =",
    duration_seconds / 3600,
    "h"
)


# ============================================================
# 5. 缺失值检查
# ============================================================

print("\n" + "=" * 80)
print("缺失值检查")
print("=" * 80)


nan_count = df[
    CHANNELS
].isna().sum()


print(
    nan_count
)


# ============================================================
# 6. Inf 检查
# ============================================================

print("\n" + "=" * 80)
print("Inf 检查")
print("=" * 80)


for col in CHANNELS:

    values = df[col].to_numpy()

    inf_count = np.isinf(
        values
    ).sum()

    print(
        f"{col:15s}",
        "Inf =",
        inf_count
    )


# ============================================================
# 7. 基本统计
# ============================================================

print("\n" + "=" * 100)
print("各通道基本统计")
print("=" * 100)


stats = []


for col in CHANNELS:

    values = df[
        col
    ].to_numpy(
        dtype=float
    )


    mean_value = np.mean(
        values
    )

    std_value = np.std(
        values
    )

    min_value = np.min(
        values
    )

    max_value = np.max(
        values
    )


    stats.append(
        {
            "channel":
                col,

            "mean":
                mean_value,

            "std":
                std_value,

            "min":
                min_value,

            "max":
                max_value,
        }
    )


    print(
        f"{col:15s}",
        f"mean={mean_value:10.4f}",
        f"std={std_value:10.4f}",
        f"min={min_value:10.4f}",
        f"max={max_value:10.4f}"
    )


stats_df = pd.DataFrame(
    stats
)


stats_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "uvw_channel_statistics.csv"
    ),
    index=False
)


# ============================================================
# 8. 时间连续性检查
# ============================================================

print("\n" + "=" * 80)
print("时间连续性检查")
print("=" * 80)


time_diff = (
    df["time20hz"]
    .diff()
    .dt.total_seconds()
)


valid_diff = time_diff.iloc[
    1:
].to_numpy()


print(
    "期望时间间隔 =",
    DT_EXPECTED,
    "s"
)

print(
    "实际最小间隔 =",
    np.min(
        valid_diff
    )
)

print(
    "实际最大间隔 =",
    np.max(
        valid_diff
    )
)


bad_time = np.where(
    ~np.isclose(
        valid_diff,
        DT_EXPECTED,
        atol=1e-6
    )
)[0]


print(
    "非 0.05 s 间隔数量 =",
    len(
        bad_time
    )
)


if len(bad_time) == 0:

    print(
        "时间序列连续。"
    )

else:

    print(
        "发现时间间隔异常。"
    )


# ============================================================
# 9. 重复时间戳检查
# ============================================================

duplicate_time = (
    df["time20hz"]
    .duplicated()
    .sum()
)


print(
    "重复时间戳数量 =",
    duplicate_time
)


# ============================================================
# 10. 简单异常值检查
#
# 使用全数据 mean ± 6σ
# 这里只用于发现极端异常，
# 不直接删除数据。
# ============================================================

print("\n" + "=" * 80)
print("极端异常值检查（mean ± 6 sigma）")
print("=" * 80)


outlier_records = []


for col in CHANNELS:

    values = df[
        col
    ].to_numpy(
        dtype=float
    )


    mean_value = np.mean(
        values
    )

    std_value = np.std(
        values
    )


    lower = (
        mean_value
        -
        6.0 * std_value
    )

    upper = (
        mean_value
        +
        6.0 * std_value
    )


    mask = (
        (values < lower)
        |
        (values > upper)
    )


    count = np.sum(
        mask
    )


    print(
        f"{col:15s}",
        "6sigma异常数量 =",
        count
    )


    outlier_records.append(
        {
            "channel":
                col,

            "lower_6sigma":
                lower,

            "upper_6sigma":
                upper,

            "count":
                int(count)
        }
    )


pd.DataFrame(
    outlier_records
).to_csv(
    os.path.join(
        RESULT_DIR,
        "uvw_outlier_summary.csv"
    ),
    index=False
)


# ============================================================
# 11. 绘制前30分钟时间序列
# ============================================================

PLOT_MINUTES = 30

N_PLOT = int(
    PLOT_MINUTES
    *
    60
    *
    FS
)


N_PLOT = min(
    N_PLOT,
    len(df)
)


time_plot = (
    np.arange(
        N_PLOT
    )
    /
    FS
    /
    60.0
)


# ============================================================
# 12. 测点 110 三分量
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 5)
)


for col in [
    "u_110_5m",
    "v_110_5m",
    "w_110_5m"
]:

    ax.plot(
        time_plot,
        df[col].iloc[
            :N_PLOT
        ].to_numpy(),
        linewidth=0.7,
        label=col
    )


ax.set_xlabel(
    "Time (min)"
)

ax.set_ylabel(
    "Velocity (m/s)"
)

ax.set_title(
    "UVW time series - point 110_5m"
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
        "uvw_110_first30min.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# 13. 测点179三分量
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 5)
)


for col in [
    "u_179_5m",
    "v_179_5m",
    "w_179_5m"
]:

    ax.plot(
        time_plot,
        df[col].iloc[
            :N_PLOT
        ].to_numpy(),
        linewidth=0.7,
        label=col
    )


ax.set_xlabel(
    "Time (min)"
)

ax.set_ylabel(
    "Velocity (m/s)"
)

ax.set_title(
    "UVW time series - point 179_5m"
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
        "uvw_179_first30min.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# 14. 两测点 u 分量比较
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 5)
)


for col in [
    "u_110_5m",
    "u_179_5m"
]:

    ax.plot(
        time_plot,
        df[col].iloc[
            :N_PLOT
        ].to_numpy(),
        linewidth=0.7,
        label=col
    )


ax.set_xlabel(
    "Time (min)"
)

ax.set_ylabel(
    "u velocity (m/s)"
)

ax.set_title(
    "Comparison of u component"
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
        "u_two_points_first30min.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# 15. 两测点 v 分量比较
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 5)
)


for col in [
    "v_110_5m",
    "v_179_5m"
]:

    ax.plot(
        time_plot,
        df[col].iloc[
            :N_PLOT
        ].to_numpy(),
        linewidth=0.7,
        label=col
    )


ax.set_xlabel(
    "Time (min)"
)

ax.set_ylabel(
    "v velocity (m/s)"
)

ax.set_title(
    "Comparison of v component"
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
        "v_two_points_first30min.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# 16. 两测点 w 分量比较
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 5)
)


for col in [
    "w_110_5m",
    "w_179_5m"
]:

    ax.plot(
        time_plot,
        df[col].iloc[
            :N_PLOT
        ].to_numpy(),
        linewidth=0.7,
        label=col
    )


ax.set_xlabel(
    "Time (min)"
)

ax.set_ylabel(
    "w velocity (m/s)"
)

ax.set_title(
    "Comparison of w component"
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
        "w_two_points_first30min.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# 17. 相关系数
# ============================================================

print("\n" + "=" * 80)
print("两测点同分量相关系数")
print("=" * 80)


pairs = [
    (
        "u_110_5m",
        "u_179_5m"
    ),
    (
        "v_110_5m",
        "v_179_5m"
    ),
    (
        "w_110_5m",
        "w_179_5m"
    )
]


for col1, col2 in pairs:

    corr = np.corrcoef(
        df[col1].to_numpy(),
        df[col2].to_numpy()
    )[0, 1]


    print(
        f"{col1} <-> {col2}:",
        corr
    )


# ============================================================
# 18. 完成
# ============================================================

print("\n" + "=" * 80)
print("UVW 数据检查完成")
print("=" * 80)

print(
    "结果目录："
)

print(
    RESULT_DIR
)

print(
    "\n主要输出："
)

print(
    "uvw_channel_statistics.csv"
)

print(
    "uvw_outlier_summary.csv"
)

print(
    "uvw_110_first30min.png"
)

print(
    "uvw_179_first30min.png"
)

print(
    "u_two_points_first30min.png"
)

print(
    "v_two_points_first30min.png"
)

print(
    "w_two_points_first30min.png"
)

print("=" * 80)