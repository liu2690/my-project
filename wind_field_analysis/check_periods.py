import os
import numpy as np
import matplotlib.pyplot as plt


ROOT = "/home/liuxy/0418"
FS = 50

PERIODS = [
    "13-14",
    "15-16",
    "16-17"
]


# ============================================================
# 需要比较的展向测点
#
# -20 m ：低速区
# -15 m ：高速区
# 用于检查空间突变是否在多个时段持续存在
# ============================================================

SPANWISE_POINTS = [
    -20,
    -15
]


# ============================================================
# 需要比较的垂向测点
#
# 7.15 m  ：高速区
# 10.24 m ：低速区
# ============================================================

VERTICAL_POINTS = [
    7.15,
    10.24
]


# ============================================================
# 输出目录
# ============================================================

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "selected_data"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 构造展向文件路径
# ============================================================

def spanwise_file(position, period):

    if position < 0:
        pos_text = f"{position:g}"
    else:
        pos_text = f"{position:g}"

    filename = (
        f"20240418_sp{pos_text}m_"
        f"fux5m{period}.dat"
    )

    return os.path.join(
        ROOT,
        "展向",
        filename
    )


# ============================================================
# 构造垂向文件路径
# ============================================================

def vertical_file(position, period):

    filename = (
        f"20240418_fux{position:g}m_"
        f"{period}.dat"
    )

    return os.path.join(
        ROOT,
        "垂向",
        filename
    )


# ============================================================
# 统计单个文件
# ============================================================

def load_and_print(path, label):

    if not os.path.isfile(path):

        print(
            f"{label:<25} 文件不存在"
        )

        return None

    signal = np.loadtxt(path)

    print(
        f"{label:<25}"
        f"mean={np.mean(signal):8.3f}  "
        f"std={np.std(signal):7.3f}  "
        f"min={np.min(signal):8.3f}  "
        f"max={np.max(signal):8.3f}"
    )

    return signal


# ============================================================
# 1. 展向多个时段比较
# ============================================================

print("\n")
print("=" * 90)
print("展向不同时间段比较")
print("=" * 90)


spanwise_signals = {}


for position in SPANWISE_POINTS:

    print(
        f"\n展向 y = {position} m"
    )

    for period in PERIODS:

        path = spanwise_file(
            position,
            period
        )

        signal = load_and_print(
            path,
            f"{period}"
        )

        if signal is not None:

            spanwise_signals[
                (position, period)
            ] = signal


# ============================================================
# 2. 垂向多个时段比较
# ============================================================

print("\n")
print("=" * 90)
print("垂向不同时间段比较")
print("=" * 90)


vertical_signals = {}


for position in VERTICAL_POINTS:

    print(
        f"\n垂向 z = {position} m"
    )

    for period in PERIODS:

        path = vertical_file(
            position,
            period
        )

        signal = load_and_print(
            path,
            f"{period}"
        )

        if signal is not None:

            vertical_signals[
                (position, period)
            ] = signal


# ============================================================
# 3. 画平均风速对比柱状图
# ============================================================

labels = []
means = []


for position in SPANWISE_POINTS:

    for period in PERIODS:

        key = (
            position,
            period
        )

        if key in spanwise_signals:

            labels.append(
                f"y={position:g}\n{period}"
            )

            means.append(
                np.mean(
                    spanwise_signals[key]
                )
            )


plt.figure(
    figsize=(10, 5)
)

plt.bar(
    labels,
    means
)

plt.ylabel(
    "Mean velocity (m/s)"
)

plt.title(
    "Spanwise mean velocity across periods"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

spanwise_save = os.path.join(
    OUTPUT_DIR,
    "spanwise_period_comparison.png"
)

plt.savefig(
    spanwise_save,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 4. 垂向平均风速对比
# ============================================================

labels = []
means = []


for position in VERTICAL_POINTS:

    for period in PERIODS:

        key = (
            position,
            period
        )

        if key in vertical_signals:

            labels.append(
                f"z={position:g}\n{period}"
            )

            means.append(
                np.mean(
                    vertical_signals[key]
                )
            )


plt.figure(
    figsize=(10, 5)
)

plt.bar(
    labels,
    means
)

plt.ylabel(
    "Mean velocity (m/s)"
)

plt.title(
    "Vertical mean velocity across periods"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

vertical_save = os.path.join(
    OUTPUT_DIR,
    "vertical_period_comparison.png"
)

plt.savefig(
    vertical_save,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 5. 前60秒时间序列对比
# ============================================================

N_PLOT = FS * 60

time = (
    np.arange(N_PLOT)
    / FS
)


plt.figure(
    figsize=(14, 7)
)


for key, signal in spanwise_signals.items():

    position, period = key

    plt.plot(
        time,
        signal[:N_PLOT],
        linewidth=0.8,
        label=(
            f"y={position:g} m "
            f"{period}"
        )
    )


plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Velocity (m/s)"
)

plt.title(
    "Spanwise comparison across periods"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "spanwise_period_time_series.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


plt.figure(
    figsize=(14, 7)
)


for key, signal in vertical_signals.items():

    position, period = key

    plt.plot(
        time,
        signal[:N_PLOT],
        linewidth=0.8,
        label=(
            f"z={position:g} m "
            f"{period}"
        )
    )


plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Velocity (m/s)"
)

plt.title(
    "Vertical comparison across periods"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "vertical_period_time_series.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\n")
print("=" * 90)
print("检查完成")
print("=" * 90)

print(
    "结果目录：",
    OUTPUT_DIR
)