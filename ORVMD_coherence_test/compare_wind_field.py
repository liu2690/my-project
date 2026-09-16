import os
import sys
import time
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 项目路径
# ============================================================

THIS_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ROOT = os.path.dirname(
    THIS_DIR
)

OVMD_DIR = os.path.join(
    ROOT,
    "OVMD"
)

ORVMD_DIR = os.path.join(
    ROOT,
    "ORVMD"
)

DATA_DIR = os.path.join(
    ROOT,
    "wind_field_analysis",
    "selected_data"
)

RESULT_DIR = os.path.join(
    THIS_DIR,
    "result",
    "spanwise_wind_comparison"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# 2. 导入 OVMD
#
# 为避免 OVMD / ORVMD 文件名冲突，
# OVMD 使用 importlib 直接加载
# ============================================================

ovmd_file = os.path.join(
    OVMD_DIR,
    "ovmd.py"
)

ovmd_spec = importlib.util.spec_from_file_location(
    "ovmd_algorithm",
    ovmd_file
)

ovmd_module = importlib.util.module_from_spec(
    ovmd_spec
)

ovmd_spec.loader.exec_module(
    ovmd_module
)

OVMD = ovmd_module.OVMD


# ============================================================
# 3. 导入 ORVMD
#
# ORVMD/orvmd.py 内部使用：
# from utils import *
#
# 因此需要先把 ORVMD 目录加入 sys.path
# ============================================================

if ORVMD_DIR not in sys.path:

    sys.path.insert(
        0,
        ORVMD_DIR
    )

from orvmd import orvmd


# ============================================================
# 4. 实验参数
# ============================================================

FS = 50.0

# 第一轮调试只使用前5000点
# 5000 / 50 = 100 s
N_SAMPLES = 5000

# 模态数
K = 4


# ------------------------------------------------------------
# 带宽惩罚参数
#
# OVMD:
# denominator ~ 1 + alpha * (f-w)^2
#
# ORVMD:
# denominator ~ 1 + 2*Alpha * (f-w)^2
#
# 因此为了让带宽惩罚量级尽可能接近：
#
# Alpha_ORVMD ≈ alpha_OVMD / 2
# ------------------------------------------------------------

OVMD_ALPHA = 2000.0

ORVMD_ALPHA = (
    OVMD_ALPHA / 2.0
)


# ------------------------------------------------------------
# OVMD参数
# ------------------------------------------------------------

OVMD_TOL = 1e-5

OVMD_MAX_STEPS = 300

OVMD_TAU = 0.0


# ------------------------------------------------------------
# ORVMD参数
# ------------------------------------------------------------

ORVMD_TOL = 5e-3

ORVMD_MAX_STEPS = 300

LAMBDA_ORTH = 0.05


# ============================================================
# 5. 读取数据
# ============================================================

data_path = os.path.join(
    DATA_DIR,
    "spanwise_fluctuation.npy"
)

position_path = os.path.join(
    DATA_DIR,
    "spanwise_positions.npy"
)


Q_full = np.load(
    data_path
)

positions = np.load(
    position_path
)


print("\n" + "=" * 80)
print("读取展向脉动风速")
print("=" * 80)

print(
    "原始数据 shape:",
    Q_full.shape
)

print(
    "空间位置:",
    positions
)


if Q_full.ndim != 2:

    raise ValueError(
        "输入风场必须是 S x T 二维矩阵。"
    )


if Q_full.shape[0] != len(positions):

    raise ValueError(
        "空间位置数量与数据空间维度不一致。"
    )


N_SAMPLES_REAL = min(
    N_SAMPLES,
    Q_full.shape[1]
)


Q = Q_full[
    :,
    :N_SAMPLES_REAL
].astype(
    np.float64
)


S, T = Q.shape


print(
    "本次实验 shape:",
    Q.shape
)

print(
    "测点数量 S:",
    S
)

print(
    "时间点数量 T:",
    T
)

print(
    "持续时间:",
    T / FS,
    "s"
)

print(
    "模态数 K:",
    K
)

print(
    "OVMD alpha:",
    OVMD_ALPHA
)

print(
    "ORVMD Alpha:",
    ORVMD_ALPHA
)

print(
    "ORVMD lambda_orth:",
    LAMBDA_ORTH
)


# ============================================================
# 6. 基础函数
# ============================================================

def relative_reconstruction_error(
    original,
    reconstructed
):
    """
    相对 Frobenius 重构误差

            ||Q-Qhat||_F
    E = -------------------
               ||Q||_F
    """

    numerator = np.linalg.norm(
        original - reconstructed
    )

    denominator = np.linalg.norm(
        original
    )

    return float(
        numerator
        /
        max(
            denominator,
            np.finfo(float).eps
        )
    )


def modal_coherence_matrix(
    mode_fields
):
    """
    模态场之间的归一化相干/相关矩阵。

    mode_fields shape:
        K x S x T

    C_ij =
        |<Qi,Qj>_F|
        ----------------
        ||Qi||_F ||Qj||_F

    理想情况下：
        对角线 = 1
        非对角线越接近0越好
    """

    mode_fields = np.asarray(
        mode_fields
    )

    K_local = mode_fields.shape[0]

    C = np.zeros(
        (
            K_local,
            K_local
        ),
        dtype=float
    )


    for i in range(K_local):

        Qi = mode_fields[i]

        norm_i = np.linalg.norm(
            Qi
        )

        for j in range(K_local):

            Qj = mode_fields[j]

            norm_j = np.linalg.norm(
                Qj
            )

            denominator = (
                norm_i
                *
                norm_j
            )

            if denominator <= 0:

                C[i, j] = 0.0

            else:

                inner = np.vdot(
                    Qi,
                    Qj
                )

                C[i, j] = (
                    np.abs(inner)
                    /
                    denominator
                )

    return C


def mean_off_diagonal(
    matrix
):
    """
    计算相干矩阵非对角元素平均值。

    越小表示模态之间越独立。
    """

    n = matrix.shape[0]

    mask = ~np.eye(
        n,
        dtype=bool
    )

    values = matrix[
        mask
    ]

    return float(
        np.mean(values)
    )


def max_off_diagonal(
    matrix
):
    """
    最大非对角相干度。
    """

    n = matrix.shape[0]

    mask = ~np.eye(
        n,
        dtype=bool
    )

    values = matrix[
        mask
    ]

    return float(
        np.max(values)
    )


def save_coherence_figure(
    matrix,
    title,
    path
):

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(
        matrix,
        vmin=0,
        vmax=1,
        aspect="equal"
    )

    cbar = fig.colorbar(
        image,
        ax=ax
    )

    cbar.set_label(
        "Normalized modal coherence"
    )

    ax.set_xlabel(
        "Mode index"
    )

    ax.set_ylabel(
        "Mode index"
    )

    ax.set_title(
        title
    )


    ticks = np.arange(
        matrix.shape[0]
    )

    ax.set_xticks(
        ticks
    )

    ax.set_yticks(
        ticks
    )

    ax.set_xticklabels(
        ticks + 1
    )

    ax.set_yticklabels(
        ticks + 1
    )


    for i in range(
        matrix.shape[0]
    ):

        for j in range(
            matrix.shape[1]
        ):

            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=8
            )


    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_frequency_figure(
    ovmd_frequency,
    orvmd_frequency,
    path
):

    mode_index = np.arange(
        1,
        K + 1
    )


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    ax.plot(
        mode_index,
        ovmd_frequency,
        marker="o",
        linewidth=1.5,
        label="OVMD"
    )

    ax.plot(
        mode_index,
        orvmd_frequency,
        marker="s",
        linewidth=1.5,
        label="ORVMD"
    )


    ax.set_xlabel(
        "Mode index"
    )

    ax.set_ylabel(
        "Center frequency (Hz)"
    )

    ax.set_title(
        "Center frequency comparison"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()


    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_mode_energy_figure(
    ovmd_energy,
    orvmd_energy,
    path
):

    mode_index = np.arange(
        1,
        K + 1
    )


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    width = 0.35


    ax.bar(
        mode_index - width / 2,
        ovmd_energy,
        width=width,
        label="OVMD"
    )

    ax.bar(
        mode_index + width / 2,
        orvmd_energy,
        width=width,
        label="ORVMD"
    )


    ax.set_xlabel(
        "Mode index"
    )

    ax.set_ylabel(
        "Normalized modal energy"
    )

    ax.set_title(
        "Modal energy comparison"
    )

    ax.set_xticks(
        mode_index
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    ax.legend()


    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 7. OVMD
#
# 每个空间测点分别进行一维OVMD
# ============================================================

print("\n")
print("#" * 80)
print("开始 OVMD 分解")
print("#" * 80)


ovmd_start_time = time.time()


# ------------------------------------------------------------
# 保存每个空间点各阶模态
#
# shape:
#     K x S x T
# ------------------------------------------------------------

ovmd_mode_fields = np.zeros(
    (
        K,
        S,
        T
    ),
    dtype=float
)


# 每个空间点每个模态的中心频率
ovmd_frequency_all = np.zeros(
    (
        S,
        K
    ),
    dtype=float
)


for s in range(S):

    signal = Q[
        s,
        :
    ]


    print(
        f"\nOVMD 测点 "
        f"{s + 1}/{S}, "
        f"position={positions[s]} m"
    )


    mode = OVMD(
        signal=signal,
        alpha=OVMD_ALPHA,
        K=K,
        Tolerance=OVMD_TOL,
        Tau=OVMD_TAU,
        MaximumSteps=OVMD_MAX_STEPS
    )


    # --------------------------------------------------------
    # mode["u"] shape:
    #
    # K x T
    # --------------------------------------------------------

    u = np.asarray(
        mode["u"]
    )


    if u.shape != (
        K,
        T
    ):

        raise RuntimeError(
            f"OVMD 输出维度异常："
            f"{u.shape}, "
            f"预期 {(K, T)}"
        )


    # --------------------------------------------------------
    # OVMD omega 为迭代历史：
    #
    # iteration x K
    #
    # 取最终一行作为最终中心频率
    # --------------------------------------------------------

    omega_history = np.asarray(
        mode["omega"]
    )


    final_omega = omega_history[
        -1,
        :
    ]


    # --------------------------------------------------------
    # OVMD原程序最终按能量排序，
    # 不一定按频率排序。
    #
    # 为了把不同空间点的第k阶模态对齐，
    # 这里再次按最终中心频率从低到高排列。
    # --------------------------------------------------------

    order = np.argsort(
        final_omega
    )


    u = u[
        order,
        :
    ]

    final_omega = final_omega[
        order
    ]


    ovmd_mode_fields[
        :,
        s,
        :
    ] = u


    # omega 是归一化频率 cycles/sample
    # 转换成 Hz
    ovmd_frequency_all[
        s,
        :
    ] = (
        final_omega
        *
        FS
    )


ovmd_runtime = (
    time.time()
    -
    ovmd_start_time
)


# ------------------------------------------------------------
# OVMD整个风场重构
# ------------------------------------------------------------

ovmd_reconstructed = np.sum(
    ovmd_mode_fields,
    axis=0
)


ovmd_error = (
    relative_reconstruction_error(
        Q,
        ovmd_reconstructed
    )
)


# 各空间点中心频率取平均
ovmd_frequency_mean = np.mean(
    ovmd_frequency_all,
    axis=0
)


# OVMD每个场模态能量
ovmd_energy = np.array([
    np.sum(
        ovmd_mode_fields[k] ** 2
    )
    for k in range(K)
])


ovmd_energy_normalized = (
    ovmd_energy
    /
    np.sum(
        ovmd_energy
    )
)


# OVMD模态相干矩阵
ovmd_coherence = (
    modal_coherence_matrix(
        ovmd_mode_fields
    )
)


ovmd_mean_coherence = (
    mean_off_diagonal(
        ovmd_coherence
    )
)


ovmd_max_coherence = (
    max_off_diagonal(
        ovmd_coherence
    )
)


print("\n")
print("=" * 80)
print("OVMD完成")
print("=" * 80)

print(
    "运行时间:",
    ovmd_runtime,
    "s"
)

print(
    "重构误差:",
    ovmd_error
)

print(
    "平均非对角模态相干度:",
    ovmd_mean_coherence
)

print(
    "最大非对角模态相干度:",
    ovmd_max_coherence
)

print(
    "平均中心频率 Hz:",
    ovmd_frequency_mean
)


# ============================================================
# 8. ORVMD
#
# 整个 S x T 风场联合分解
# ============================================================

print("\n")
print("#" * 80)
print("开始 ORVMD 分解")
print("#" * 80)


orvmd_start_time = time.time()


orvmd_mode, orvmd_info, _ = orvmd(

    Q=Q,

    K=K,

    Alpha=ORVMD_ALPHA,

    Weight=1,

    lambda_orth=LAMBDA_ORTH,

    Tolerance=ORVMD_TOL,

    MaximumSteps=ORVMD_MAX_STEPS,

    InitFreqType=1,

    InitFreqMaximum=0.5,

    Device="cpu",

    FPPrecision="double",

    nDC=0,

    Display="iter",

    Restart=None
)


orvmd_runtime = (
    time.time()
    -
    orvmd_start_time
)


# ============================================================
# 9. ORVMD模态重构
# ============================================================

phi = np.asarray(
    orvmd_mode["phi"]
)

c = np.asarray(
    orvmd_mode["c"]
)

orvmd_omega = np.asarray(
    orvmd_mode["omega"]
)


print("\nORVMD phi shape:", phi.shape)
print("ORVMD c shape:", c.shape)
print("ORVMD omega shape:", orvmd_omega.shape)


if phi.shape != (
    S,
    K
):

    raise RuntimeError(
        f"ORVMD phi维度异常："
        f"{phi.shape}"
    )


if c.shape != (
    T,
    K
):

    raise RuntimeError(
        f"ORVMD c维度异常："
        f"{c.shape}"
    )


# ------------------------------------------------------------
# 第k个时空模态
#
# Q_k(x,t) = phi_k(x) * c_k(t)
#
# shape:
#     K x S x T
# ------------------------------------------------------------

orvmd_mode_fields = np.zeros(
    (
        K,
        S,
        T
    ),
    dtype=float
)


for k in range(K):

    orvmd_mode_fields[
        k
    ] = np.outer(
        phi[:, k],
        c[:, k]
    )


# ------------------------------------------------------------
# ORVMD重构
# ------------------------------------------------------------

orvmd_reconstructed = np.sum(
    orvmd_mode_fields,
    axis=0
)


orvmd_error = (
    relative_reconstruction_error(
        Q,
        orvmd_reconstructed
    )
)


# ------------------------------------------------------------
# ORVMD中心频率转 Hz
# ------------------------------------------------------------

orvmd_frequency_hz = (
    orvmd_omega
    *
    FS
)


# ------------------------------------------------------------
# ORVMD模态能量
# ------------------------------------------------------------

orvmd_energy = np.array([
    np.sum(
        orvmd_mode_fields[k] ** 2
    )
    for k in range(K)
])


orvmd_energy_normalized = (
    orvmd_energy
    /
    np.sum(
        orvmd_energy
    )
)


# ------------------------------------------------------------
# ORVMD模态相干矩阵
# ------------------------------------------------------------

orvmd_coherence = (
    modal_coherence_matrix(
        orvmd_mode_fields
    )
)


orvmd_mean_coherence = (
    mean_off_diagonal(
        orvmd_coherence
    )
)


orvmd_max_coherence = (
    max_off_diagonal(
        orvmd_coherence
    )
)


print("\n")
print("=" * 80)
print("ORVMD完成")
print("=" * 80)

print(
    "运行时间:",
    orvmd_runtime,
    "s"
)

print(
    "迭代次数:",
    orvmd_info[
        "Iteration"
    ][
        "steps"
    ]
)

print(
    "是否收敛:",
    orvmd_info[
        "Iteration"
    ][
        "converged"
    ]
)

print(
    "重构误差:",
    orvmd_error
)

print(
    "平均非对角模态相干度:",
    orvmd_mean_coherence
)

print(
    "最大非对角模态相干度:",
    orvmd_max_coherence
)

print(
    "中心频率 Hz:",
    orvmd_frequency_hz
)


# ============================================================
# 10. 保存总体评价
# ============================================================

summary = pd.DataFrame(
    {
        "Method": [
            "OVMD",
            "ORVMD"
        ],

        "Reconstruction_Error": [
            ovmd_error,
            orvmd_error
        ],

        "Mean_OffDiagonal_Coherence": [
            ovmd_mean_coherence,
            orvmd_mean_coherence
        ],

        "Max_OffDiagonal_Coherence": [
            ovmd_max_coherence,
            orvmd_max_coherence
        ],

        "Runtime_s": [
            ovmd_runtime,
            orvmd_runtime
        ]
    }
)


summary_path = os.path.join(
    RESULT_DIR,
    "comparison_summary.csv"
)


summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 11. 每个模态结果
# ============================================================

mode_table = pd.DataFrame(
    {
        "Mode": np.arange(
            1,
            K + 1
        ),

        "OVMD_Frequency_Hz":
            ovmd_frequency_mean,

        "ORVMD_Frequency_Hz":
            orvmd_frequency_hz,

        "OVMD_Energy_Ratio":
            ovmd_energy_normalized,

        "ORVMD_Energy_Ratio":
            orvmd_energy_normalized
    }
)


mode_table_path = os.path.join(
    RESULT_DIR,
    "mode_comparison.csv"
)


mode_table.to_csv(
    mode_table_path,
    index=False
)


# ============================================================
# 12. 保存相干矩阵
# ============================================================

np.savetxt(
    os.path.join(
        RESULT_DIR,
        "ovmd_coherence_matrix.csv"
    ),
    ovmd_coherence,
    delimiter=","
)


np.savetxt(
    os.path.join(
        RESULT_DIR,
        "orvmd_coherence_matrix.csv"
    ),
    orvmd_coherence,
    delimiter=","
)


save_coherence_figure(
    ovmd_coherence,
    "OVMD modal coherence",
    os.path.join(
        RESULT_DIR,
        "ovmd_coherence_matrix.png"
    )
)


save_coherence_figure(
    orvmd_coherence,
    "ORVMD modal coherence",
    os.path.join(
        RESULT_DIR,
        "orvmd_coherence_matrix.png"
    )
)


# ============================================================
# 13. 中心频率比较图
# ============================================================

save_frequency_figure(
    ovmd_frequency_mean,
    orvmd_frequency_hz,
    os.path.join(
        RESULT_DIR,
        "center_frequency_comparison.png"
    )
)


# ============================================================
# 14. 模态能量比较图
# ============================================================

save_mode_energy_figure(
    ovmd_energy_normalized,
    orvmd_energy_normalized,
    os.path.join(
        RESULT_DIR,
        "mode_energy_comparison.png"
    )
)


# ============================================================
# 15. 保存分解结果
# ============================================================

np.save(
    os.path.join(
        RESULT_DIR,
        "ovmd_mode_fields.npy"
    ),
    ovmd_mode_fields
)


np.save(
    os.path.join(
        RESULT_DIR,
        "orvmd_mode_fields.npy"
    ),
    orvmd_mode_fields
)


np.save(
    os.path.join(
        RESULT_DIR,
        "ovmd_reconstructed.npy"
    ),
    ovmd_reconstructed
)


np.save(
    os.path.join(
        RESULT_DIR,
        "orvmd_reconstructed.npy"
    ),
    orvmd_reconstructed
)


np.save(
    os.path.join(
        RESULT_DIR,
        "input_Q.npy"
    ),
    Q
)


# ============================================================
# 16. 打印最终结果
# ============================================================

print("\n")
print("#" * 80)
print("OVMD / ORVMD 风场分解对比完成")
print("#" * 80)


print("\n总体评价：")
print(
    summary.to_string(
        index=False
    )
)


print("\n各阶模态：")
print(
    mode_table.to_string(
        index=False
    )
)


print(
    "\n结果保存目录："
)

print(
    RESULT_DIR
)


print("\n主要结果文件：")

print(
    "comparison_summary.csv"
)

print(
    "mode_comparison.csv"
)

print(
    "ovmd_coherence_matrix.png"
)

print(
    "orvmd_coherence_matrix.png"
)

print(
    "center_frequency_comparison.png"
)

print(
    "mode_energy_comparison.png"
)

print("#" * 80)