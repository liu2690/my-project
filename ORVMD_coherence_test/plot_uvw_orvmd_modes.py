import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径设置
# ============================================================

THIS_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ROOT = os.path.dirname(
    THIS_DIR
)

ORVMD_DIR = os.path.join(
    ROOT,
    "ORVMD"
)

DATA_DIR = os.path.join(
    ROOT,
    "3d_wind_analysis",
    "result",
    "selected_segment"
)

RESULT_DIR = os.path.join(
    THIS_DIR,
    "result",
    "uvw_orvmd_mode_analysis"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


if ORVMD_DIR not in sys.path:
    sys.path.insert(
        0,
        ORVMD_DIR
    )


from orvmd import orvmd


# ============================================================
# 2. 实验参数
# ============================================================

FS = 20.0

K = 4

ALPHA = 1000.0

LAMBDA_ORTH = 0.03

TOL = 0.005

MAX_STEPS = 500


CHANNEL_NAMES = [
    "u_110_5m",
    "v_110_5m",
    "w_110_5m",
    "u_179_5m",
    "v_179_5m",
    "w_179_5m",
]


# ============================================================
# 3. 工具函数
# ============================================================

def safe_corrcoef(x, y):
    """
    安全计算 Pearson 相关系数
    """

    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    x_std = np.std(x)
    y_std = np.std(y)

    if x_std < 1e-14 or y_std < 1e-14:
        return 0.0

    return float(
        np.corrcoef(
            x,
            y
        )[0, 1]
    )


def normalized_rmse(x, y):
    """
    相对于原始信号 RMS 的归一化 RMSE
    """

    x = np.asarray(x)
    y = np.asarray(y)

    rmse = np.sqrt(
        np.mean(
            (x - y) ** 2
        )
    )

    denom = np.sqrt(
        np.mean(
            x ** 2
        )
    )

    if denom < 1e-14:
        return np.nan

    return float(
        rmse / denom
    )


def build_mode_fields(phi, c):
    """
    根据 phi 和 c 构造每个模态的完整时空场

    phi:
        S x K

    c:
        T x K

    return:
        K x S x T
    """

    S = phi.shape[0]
    T = c.shape[0]

    fields = np.zeros(
        (
            K,
            S,
            T
        ),
        dtype=float
    )

    for k in range(K):

        fields[k] = (
            phi[:, k][:, None]
            *
            c[:, k][None, :]
        )

    return fields


# ============================================================
# 4. 读取真实 UVW 脉动风场
# ============================================================

Q_PATH = os.path.join(
    DATA_DIR,
    "uvw_selected_fluctuation.npy"
)


if not os.path.exists(Q_PATH):

    raise FileNotFoundError(
        f"找不到输入文件：\n{Q_PATH}"
    )


Q = np.load(
    Q_PATH
)


print("=" * 80)
print("ORVMD 模态分析")
print("=" * 80)

print(
    "Q shape =",
    Q.shape
)


if Q.shape[0] != 6:

    raise ValueError(
        "当前程序预期 Q shape = (6, T)"
    )


S, T = Q.shape

time = (
    np.arange(T)
    /
    FS
)


print(
    "空间/通道数 =",
    S
)

print(
    "时间点数 =",
    T
)

print(
    "采样频率 =",
    FS,
    "Hz"
)

print(
    "持续时间 =",
    T / FS,
    "s"
)

print(
    "lambda_orth =",
    LAMBDA_ORTH
)


# ============================================================
# 5. 运行 ORVMD
# ============================================================

print()
print("=" * 80)
print("运行 ORVMD")
print("=" * 80)


mode, info, _ = orvmd(

    Q=Q,

    K=K,

    Alpha=ALPHA,

    Weight=1,

    lambda_orth=LAMBDA_ORTH,

    Tolerance=TOL,

    MaximumSteps=MAX_STEPS,

    InitFreqType=1,

    InitFreqMaximum=0.5,

    Device="cpu",

    FPPrecision="double",

    nDC=0,

    Display="off",

    Restart=None
)


# ============================================================
# 6. 提取结果
# ============================================================

phi = np.asarray(
    mode["phi"]
)

c = np.asarray(
    mode["c"]
)

omega = np.asarray(
    mode["omega"]
).reshape(-1)


print(
    "phi shape =",
    phi.shape
)

print(
    "c shape =",
    c.shape
)

print(
    "omega shape =",
    omega.shape
)


# ------------------------------------------------------------
# 兼容不同 ORVMD 输出
# ------------------------------------------------------------

if "mode" in mode:

    mode_fields = np.asarray(
        mode["mode"]
    )

else:

    mode_fields = build_mode_fields(
        phi,
        c
    )


# 如果维度顺序不正确，自动修正
if mode_fields.shape == (
    S,
    K,
    T
):

    mode_fields = np.transpose(
        mode_fields,
        (
            1,
            0,
            2
        )
    )


if mode_fields.shape != (
    K,
    S,
    T
):

    print(
        "mode 中的模态场尺寸不匹配，"
        "重新根据 phi 和 c 构造"
    )

    mode_fields = build_mode_fields(
        phi,
        c
    )


print(
    "mode fields shape =",
    mode_fields.shape
)


# ============================================================
# 7. 重构检查
# ============================================================

Q_hat = np.sum(
    mode_fields,
    axis=0
)


reconstruction_error = (
    np.linalg.norm(
        Q - Q_hat
    )
    /
    np.linalg.norm(
        Q
    )
)


print()
print(
    "reconstruction error =",
    reconstruction_error
)


# ============================================================
# 8. 中心频率
# ============================================================

# ORVMD 内部 omega 通常是归一化频率
# 与你前面正式实验保持一致：
frequency_hz = (
    omega
    *
    FS
)


print(
    "center frequency Hz =",
    frequency_hz
)


# ============================================================
# 9. 每个模态总能量
# ============================================================

mode_energy = np.zeros(
    K
)


for k in range(K):

    mode_energy[k] = np.sum(
        mode_fields[k] ** 2
    )


energy_fraction = (
    mode_energy
    /
    np.sum(
        mode_energy
    )
)


print()
print("=" * 80)
print("各模态能量")
print("=" * 80)


for k in range(K):

    print(
        f"Mode {k + 1}: "
        f"energy = {mode_energy[k]:.6f}, "
        f"fraction = {energy_fraction[k] * 100:.2f} %, "
        f"frequency = {frequency_hz[k]:.6f} Hz"
    )


dominant_mode = int(
    np.argmax(
        energy_fraction
    )
)


print()
print(
    "全局主导模态 =",
    dominant_mode + 1
)


# ============================================================
# 10. 每个通道和各模态相关系数
# ============================================================

corr_matrix = np.zeros(
    (
        S,
        K
    )
)


for s in range(S):

    for k in range(K):

        corr_matrix[s, k] = safe_corrcoef(

            Q[s],

            mode_fields[k, s]

        )


abs_corr_matrix = np.abs(
    corr_matrix
)


best_mode_per_channel = np.argmax(
    abs_corr_matrix,
    axis=1
)


print()
print("=" * 80)
print("各通道最佳相关模态")
print("=" * 80)


records = []


for s in range(S):

    k = int(
        best_mode_per_channel[s]
    )

    corr = corr_matrix[
        s,
        k
    ]

    nrmse = normalized_rmse(
        Q[s],
        mode_fields[k, s]
    )

    print(
        f"{CHANNEL_NAMES[s]:12s} "
        f"-> Mode {k + 1}, "
        f"corr = {corr:.6f}, "
        f"|corr| = {abs(corr):.6f}, "
        f"frequency = {frequency_hz[k]:.6f} Hz"
    )

    records.append(
        {
            "channel":
                CHANNEL_NAMES[s],

            "best_mode":
                k + 1,

            "correlation":
                corr,

            "absolute_correlation":
                abs(corr),

            "center_frequency_Hz":
                frequency_hz[k],

            "normalized_rmse":
                nrmse
        }
    )


best_mode_df = pd.DataFrame(
    records
)


best_mode_df.to_csv(

    os.path.join(
        RESULT_DIR,
        "best_mode_per_channel.csv"
    ),

    index=False

)


# ============================================================
# 11. 保存模态总体信息
# ============================================================

mode_summary_df = pd.DataFrame(
    {
        "mode":
            np.arange(
                1,
                K + 1
            ),

        "center_frequency_Hz":
            frequency_hz,

        "energy":
            mode_energy,

        "energy_fraction":
            energy_fraction,

        "energy_percent":
            energy_fraction * 100
    }
)


mode_summary_df.to_csv(

    os.path.join(
        RESULT_DIR,
        "mode_summary.csv"
    ),

    index=False

)


# ============================================================
# 12. 图1：全部四阶模态 —— 每个通道分别画
# ============================================================

print()
print("=" * 80)
print("绘制各通道全部模态")
print("=" * 80)


for s in range(S):

    fig, axes = plt.subplots(
        K + 1,
        1,
        figsize=(14, 10),
        sharex=True
    )


    # 原始信号
    axes[0].plot(
        time,
        Q[s],
        linewidth=0.8
    )

    axes[0].set_ylabel(
        "Original"
    )

    axes[0].set_title(
        f"{CHANNEL_NAMES[s]}: original fluctuation and ORVMD modes "
        f"(lambda={LAMBDA_ORTH})"
    )

    axes[0].grid(
        alpha=0.3
    )


    # 各模态
    for k in range(K):

        axes[k + 1].plot(

            time,

            mode_fields[
                k,
                s
            ],

            linewidth=0.8
        )

        axes[k + 1].set_ylabel(
            f"Mode {k + 1}"
        )

        axes[k + 1].grid(
            alpha=0.3
        )

        axes[k + 1].text(

            0.99,
            0.88,

            (
                f"f = {frequency_hz[k]:.4f} Hz\n"
                f"corr = {corr_matrix[s, k]:.3f}"
            ),

            transform=
                axes[k + 1].transAxes,

            ha="right",

            va="top",

            fontsize=9,

            bbox=dict(
                boxstyle="round",
                alpha=0.15
            )
        )


    axes[-1].set_xlabel(
        "Time (s)"
    )


    plt.tight_layout()


    output_path = os.path.join(

        RESULT_DIR,

        f"{CHANNEL_NAMES[s]}_all_modes.png"

    )


    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


# ============================================================
# 13. 图2：每个通道原始信号 vs 最佳模态
# ============================================================

fig, axes = plt.subplots(

    3,
    2,

    figsize=(16, 11),

    sharex=True

)


axes = axes.flatten()


for s in range(S):

    k = int(
        best_mode_per_channel[s]
    )


    axes[s].plot(

        time,

        Q[s],

        linewidth=1.0,

        label="Original fluctuation"

    )


    axes[s].plot(

        time,

        mode_fields[
            k,
            s
        ],

        linewidth=1.0,

        label=f"Best Mode {k + 1}"

    )


    axes[s].set_title(

        (
            f"{CHANNEL_NAMES[s]}  |  "
            f"Mode {k + 1}  |  "
            f"corr={corr_matrix[s, k]:.3f}"
        )

    )


    axes[s].set_ylabel(
        "Velocity (m/s)"
    )


    axes[s].grid(
        alpha=0.3
    )


    axes[s].legend(
        fontsize=8
    )


for ax in axes[-2:]:

    ax.set_xlabel(
        "Time (s)"
    )


plt.suptitle(

    (
        "Original UVW fluctuation vs "
        "best-correlated ORVMD mode "
        f"(lambda={LAMBDA_ORTH})"
    ),

    fontsize=16

)


plt.tight_layout()


plt.savefig(

    os.path.join(
        RESULT_DIR,
        "best_mode_vs_original_all_channels.png"
    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 14. 图3：全局主导模态
# ============================================================

fig, axes = plt.subplots(

    3,
    2,

    figsize=(16, 11),

    sharex=True

)


axes = axes.flatten()


for s in range(S):

    axes[s].plot(

        time,

        Q[s],

        linewidth=1.0,

        label="Original fluctuation"

    )


    axes[s].plot(

        time,

        mode_fields[
            dominant_mode,
            s
        ],

        linewidth=1.0,

        label=f"Dominant Mode {dominant_mode + 1}"

    )


    corr = safe_corrcoef(

        Q[s],

        mode_fields[
            dominant_mode,
            s
        ]

    )


    axes[s].set_title(

        (
            f"{CHANNEL_NAMES[s]}  |  "
            f"corr={corr:.3f}"
        )

    )


    axes[s].set_ylabel(
        "Velocity (m/s)"
    )


    axes[s].grid(
        alpha=0.3
    )


    axes[s].legend(
        fontsize=8
    )


for ax in axes[-2:]:

    ax.set_xlabel(
        "Time (s)"
    )


plt.suptitle(

    (
        f"Dominant ORVMD Mode {dominant_mode + 1} "
        f"(energy fraction = "
        f"{energy_fraction[dominant_mode] * 100:.2f}%)"
    ),

    fontsize=16

)


plt.tight_layout()


plt.savefig(

    os.path.join(
        RESULT_DIR,
        "dominant_mode_vs_original.png"
    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 15. 图4：四个模态能量比例
# ============================================================

plt.figure(
    figsize=(9, 6)
)


x = np.arange(
    1,
    K + 1
)


plt.bar(

    x,

    energy_fraction * 100

)


plt.xlabel(
    "Mode"
)


plt.ylabel(
    "Energy fraction (%)"
)


plt.title(
    "ORVMD modal energy distribution"
)


plt.xticks(
    x
)


plt.grid(
    axis="y",
    alpha=0.3
)


for k in range(K):

    plt.text(

        k + 1,

        energy_fraction[k] * 100,

        f"{energy_fraction[k] * 100:.2f}%",

        ha="center",

        va="bottom"

    )


plt.tight_layout()


plt.savefig(

    os.path.join(
        RESULT_DIR,
        "orvmd_mode_energy_distribution.png"
    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 16. 图5：通道 × 模态相关系数矩阵
# ============================================================

plt.figure(
    figsize=(9, 6)
)


im = plt.imshow(

    abs_corr_matrix,

    aspect="auto",

    vmin=0,

    vmax=1

)


plt.colorbar(
    im,
    label="Absolute correlation"
)


plt.xticks(

    np.arange(K),

    [
        f"Mode {k + 1}"
        for k in range(K)
    ]

)


plt.yticks(

    np.arange(S),

    CHANNEL_NAMES

)


plt.xlabel(
    "ORVMD mode"
)


plt.ylabel(
    "UVW channel"
)


plt.title(
    "Correlation between original UVW signals and ORVMD modes"
)


# 每个格子标数字
for s in range(S):

    for k in range(K):

        plt.text(

            k,
            s,

            f"{abs_corr_matrix[s, k]:.2f}",

            ha="center",

            va="center"

        )


plt.tight_layout()


plt.savefig(

    os.path.join(
        RESULT_DIR,
        "channel_mode_correlation_matrix.png"
    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 17. 保存数组
# ============================================================

np.save(

    os.path.join(
        RESULT_DIR,
        "orvmd_mode_fields.npy"
    ),

    mode_fields

)


np.save(

    os.path.join(
        RESULT_DIR,
        "orvmd_reconstruction.npy"
    ),

    Q_hat

)


# ============================================================
# 18. 输出总结
# ============================================================

print()
print("#" * 80)
print("ORVMD 模态分析完成")
print("#" * 80)

print(
    "结果目录："
)

print(
    RESULT_DIR
)

print()

print(
    "全局主导模态：",
    f"Mode {dominant_mode + 1}"
)

print(
    "能量占比：",
    f"{energy_fraction[dominant_mode] * 100:.2f}%"
)

print(
    "中心频率：",
    f"{frequency_hz[dominant_mode]:.6f} Hz"
)

print()

print(
    "主要结果图片："
)

print(
    "dominant_mode_vs_original.png"
)

print(
    "best_mode_vs_original_all_channels.png"
)

print(
    "orvmd_mode_energy_distribution.png"
)

print(
    "channel_mode_correlation_matrix.png"
)

print()

print(
    "每个通道全部模态："
)

for name in CHANNEL_NAMES:

    print(
        f"{name}_all_modes.png"
    )

print("#" * 80)