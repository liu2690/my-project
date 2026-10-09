import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径
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

DATA_PATH = os.path.join(
    ROOT,
    "3d_wind_analysis",
    "result",
    "selected_segment",
    "uvw_selected_fluctuation.npy"
)

RESULT_DIR = os.path.join(
    THIS_DIR,
    "result",
    "uvw_rvmd_orvmd_compare"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# 2. 导入 ORVMD
#
# 当前实验中：
# lambda_orth = 0     -> RVMD baseline
# lambda_orth = 0.03  -> ORVMD
# ============================================================

if ORVMD_DIR not in sys.path:
    sys.path.insert(
        0,
        ORVMD_DIR
    )

from orvmd import orvmd


# ============================================================
# 3. 实验参数
# ============================================================

FS = 20.0

K = 4

ALPHA = 1000.0

TOL = 5e-3

MAX_STEPS = 500

RVMD_LAMBDA = 0.0

ORVMD_LAMBDA = 0.03


CHANNEL_NAMES = [
    "u_110_5m",
    "v_110_5m",
    "w_110_5m",
    "u_179_5m",
    "v_179_5m",
    "w_179_5m"
]


# ============================================================
# 4. 读取数据
# ============================================================

Q = np.load(
    DATA_PATH
).astype(
    np.float64
)


if Q.shape != (6, 5000):

    raise ValueError(
        f"输入 Q shape = {Q.shape}，"
        "预期为 (6, 5000)"
    )


S, T = Q.shape

time = np.arange(
    T
) / FS


print("\n" + "=" * 80)
print("真实三分量 UVW 风场：RVMD vs ORVMD 正式对比")
print("=" * 80)

print("Q shape =", Q.shape)
print("FS =", FS, "Hz")
print("持续时间 =", T / FS, "s")
print("K =", K)
print("Alpha =", ALPHA)
print("Tolerance =", TOL)
print("MaximumSteps =", MAX_STEPS)

print("\nRVMD baseline:")
print("lambda_orth =", RVMD_LAMBDA)

print("\nORVMD:")
print("lambda_orth =", ORVMD_LAMBDA)


# ============================================================
# 5. 工具函数
# ============================================================

def build_mode_fields(phi, c):
    """
    phi : (S, K)
    c   : (T, K)

    返回：
    fields : (K, S, T)
    """

    K_local = phi.shape[1]

    fields = np.zeros(
        (
            K_local,
            phi.shape[0],
            c.shape[0]
        ),
        dtype=float
    )

    for k in range(K_local):

        fields[k] = np.outer(
            phi[:, k],
            c[:, k]
        )

    return fields


def normalized_reconstruction_error(
    Q_true,
    Q_hat
):

    denominator = np.linalg.norm(
        Q_true
    )

    if denominator == 0:
        return np.nan

    return float(
        np.linalg.norm(
            Q_true - Q_hat
        )
        /
        denominator
    )


def coherence_matrix(
    fields
):
    """
    基于完整时空模态的归一化内积计算模态相干度。
    """

    K_local = fields.shape[0]

    matrix = np.zeros(
        (
            K_local,
            K_local
        ),
        dtype=float
    )

    for i in range(K_local):

        fi = fields[i].reshape(-1)

        norm_i = np.linalg.norm(fi)

        for j in range(K_local):

            fj = fields[j].reshape(-1)

            norm_j = np.linalg.norm(fj)

            denominator = norm_i * norm_j

            if denominator <= 0:

                matrix[i, j] = 0.0

            else:

                matrix[i, j] = (
                    np.abs(
                        np.vdot(
                            fi,
                            fj
                        )
                    )
                    /
                    denominator
                )

    return matrix


def offdiag_statistics(
    matrix
):

    mask = ~np.eye(
        matrix.shape[0],
        dtype=bool
    )

    values = matrix[mask]

    return (
        float(np.mean(values)),
        float(np.max(values))
    )


def mode_energy(
    fields
):
    """
    每个时空模态的平方能量。
    """

    return np.sum(
        fields ** 2,
        axis=(1, 2)
    )


def run_model(
    name,
    lambda_orth
):

    print("\n")
    print("#" * 80)
    print(name)
    print("lambda_orth =", lambda_orth)
    print("#" * 80)

    mode, info, _ = orvmd(

        Q=Q,

        K=K,

        Alpha=ALPHA,

        Weight=1,

        lambda_orth=lambda_orth,

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


    # --------------------------------------------------------
    # 提取结果
    # --------------------------------------------------------

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


    if phi.shape != (S, K):

        raise RuntimeError(
            f"{name}: phi shape 异常：{phi.shape}"
        )


    if c.shape != (T, K):

        raise RuntimeError(
            f"{name}: c shape 异常：{c.shape}"
        )


    # --------------------------------------------------------
    # 构造四个时空模态
    # --------------------------------------------------------

    fields = build_mode_fields(
        phi,
        c
    )


    # --------------------------------------------------------
    # 重构
    # --------------------------------------------------------

    reconstruction = np.sum(
        fields,
        axis=0
    )


    rec_error = normalized_reconstruction_error(
        Q,
        reconstruction
    )


    # --------------------------------------------------------
    # 相干度
    # --------------------------------------------------------

    coh_matrix = coherence_matrix(
        fields
    )


    mean_coh, max_coh = offdiag_statistics(
        coh_matrix
    )


    # --------------------------------------------------------
    # 能量
    # --------------------------------------------------------

    energy = mode_energy(
        fields
    )


    energy_fraction = (
        energy
        /
        np.sum(energy)
    )


    # --------------------------------------------------------
    # 收敛信息
    # --------------------------------------------------------

    iteration_info = info[
        "Iteration"
    ]


    converged = iteration_info[
        "converged"
    ]


    steps = iteration_info[
        "steps"
    ]


    difference_history = np.asarray(
        iteration_info[
            "difference"
        ]
    )


    final_difference = float(
        difference_history[-1]
    )


    # --------------------------------------------------------
    # 中心频率
    # --------------------------------------------------------

    frequency_hz = (
        omega
        *
        FS
    )


    print(
        "converged =",
        converged
    )

    print(
        "steps =",
        steps
    )

    print(
        "final difference =",
        final_difference
    )

    print(
        "reconstruction error =",
        rec_error
    )

    print(
        "mean coherence =",
        mean_coh
    )

    print(
        "max coherence =",
        max_coh
    )

    print(
        "frequency Hz =",
        frequency_hz
    )

    print(
        "mode energy =",
        energy
    )

    print(
        "energy fraction =",
        energy_fraction
    )


    return {

        "name":
            name,

        "lambda":
            lambda_orth,

        "phi":
            phi,

        "c":
            c,

        "omega":
            omega,

        "frequency_hz":
            frequency_hz,

        "fields":
            fields,

        "reconstruction":
            reconstruction,

        "reconstruction_error":
            rec_error,

        "coherence_matrix":
            coh_matrix,

        "mean_coherence":
            mean_coh,

        "max_coherence":
            max_coh,

        "energy":
            energy,

        "energy_fraction":
            energy_fraction,

        "converged":
            converged,

        "steps":
            steps,

        "final_difference":
            final_difference,

        "difference_history":
            difference_history
    }


# ============================================================
# 6. 正式运行 RVMD / ORVMD
# ============================================================

rvmd_result = run_model(
    "RVMD baseline",
    RVMD_LAMBDA
)


orvmd_result = run_model(
    "ORVMD",
    ORVMD_LAMBDA
)


# ============================================================
# 7. 汇总指标
# ============================================================

mean_improvement = (
    (
        rvmd_result[
            "mean_coherence"
        ]
        -
        orvmd_result[
            "mean_coherence"
        ]
    )
    /
    rvmd_result[
        "mean_coherence"
    ]
    *
    100.0
)


max_improvement = (
    (
        rvmd_result[
            "max_coherence"
        ]
        -
        orvmd_result[
            "max_coherence"
        ]
    )
    /
    rvmd_result[
        "max_coherence"
    ]
    *
    100.0
)


reconstruction_change = (
    (
        orvmd_result[
            "reconstruction_error"
        ]
        -
        rvmd_result[
            "reconstruction_error"
        ]
    )
    /
    rvmd_result[
        "reconstruction_error"
    ]
    *
    100.0
)


summary = pd.DataFrame(
    [
        {
            "method":
                "RVMD",

            "lambda_orth":
                RVMD_LAMBDA,

            "reconstruction_error":
                rvmd_result[
                    "reconstruction_error"
                ],

            "mean_offdiag_coherence":
                rvmd_result[
                    "mean_coherence"
                ],

            "max_offdiag_coherence":
                rvmd_result[
                    "max_coherence"
                ],

            "converged":
                rvmd_result[
                    "converged"
                ],

            "steps":
                rvmd_result[
                    "steps"
                ],

            "final_difference":
                rvmd_result[
                    "final_difference"
                ],

            "f1_Hz":
                rvmd_result[
                    "frequency_hz"
                ][0],

            "f2_Hz":
                rvmd_result[
                    "frequency_hz"
                ][1],

            "f3_Hz":
                rvmd_result[
                    "frequency_hz"
                ][2],

            "f4_Hz":
                rvmd_result[
                    "frequency_hz"
                ][3]
        },

        {
            "method":
                "ORVMD",

            "lambda_orth":
                ORVMD_LAMBDA,

            "reconstruction_error":
                orvmd_result[
                    "reconstruction_error"
                ],

            "mean_offdiag_coherence":
                orvmd_result[
                    "mean_coherence"
                ],

            "max_offdiag_coherence":
                orvmd_result[
                    "max_coherence"
                ],

            "converged":
                orvmd_result[
                    "converged"
                ],

            "steps":
                orvmd_result[
                    "steps"
                ],

            "final_difference":
                orvmd_result[
                    "final_difference"
                ],

            "f1_Hz":
                orvmd_result[
                    "frequency_hz"
                ][0],

            "f2_Hz":
                orvmd_result[
                    "frequency_hz"
                ][1],

            "f3_Hz":
                orvmd_result[
                    "frequency_hz"
                ][2],

            "f4_Hz":
                orvmd_result[
                    "frequency_hz"
                ][3]
        }
    ]
)


summary_path = os.path.join(
    RESULT_DIR,
    "uvw_rvmd_orvmd_summary.csv"
)


summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 8. 模态能量表
# ============================================================

energy_rows = []

for k in range(K):

    energy_rows.append(
        {
            "mode":
                k + 1,

            "rvmd_energy":
                rvmd_result[
                    "energy"
                ][k],

            "orvmd_energy":
                orvmd_result[
                    "energy"
                ][k],

            "rvmd_energy_fraction":
                rvmd_result[
                    "energy_fraction"
                ][k],

            "orvmd_energy_fraction":
                orvmd_result[
                    "energy_fraction"
                ][k]
        }
    )


energy_df = pd.DataFrame(
    energy_rows
)


energy_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "uvw_mode_energy_comparison.csv"
    ),
    index=False
)


# ============================================================
# 9. 中心频率表
# ============================================================

frequency_df = pd.DataFrame(
    {
        "mode":
            np.arange(
                1,
                K + 1
            ),

        "rvmd_frequency_Hz":
            rvmd_result[
                "frequency_hz"
            ],

        "orvmd_frequency_Hz":
            orvmd_result[
                "frequency_hz"
            ]
    }
)


frequency_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "uvw_center_frequency_comparison.csv"
    ),
    index=False
)


# ============================================================
# 10. 保存相干矩阵 CSV
# ============================================================

np.savetxt(
    os.path.join(
        RESULT_DIR,
        "rvmd_coherence_matrix.csv"
    ),
    rvmd_result[
        "coherence_matrix"
    ],
    delimiter=","
)


np.savetxt(
    os.path.join(
        RESULT_DIR,
        "orvmd_coherence_matrix.csv"
    ),
    orvmd_result[
        "coherence_matrix"
    ],
    delimiter=","
)


# ============================================================
# 11. 图1：平均/最大相干度对比
# ============================================================

labels = [
    "Mean coherence",
    "Max coherence"
]


rvmd_values = [
    rvmd_result[
        "mean_coherence"
    ],
    rvmd_result[
        "max_coherence"
    ]
]


orvmd_values = [
    orvmd_result[
        "mean_coherence"
    ],
    orvmd_result[
        "max_coherence"
    ]
]


x = np.arange(
    len(labels)
)


width = 0.35


fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.bar(
    x - width / 2,
    rvmd_values,
    width,
    label="RVMD"
)


ax.bar(
    x + width / 2,
    orvmd_values,
    width,
    label="ORVMD"
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    labels
)

ax.set_ylabel(
    "Coherence"
)

ax.set_title(
    "UVW field: RVMD vs ORVMD coherence"
)

ax.legend()

ax.grid(
    True,
    axis="y",
    alpha=0.3
)


fig.tight_layout()


fig.savefig(
    os.path.join(
        RESULT_DIR,
        "uvw_coherence_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 12. 图2：中心频率对比
# ============================================================

mode_axis = np.arange(
    1,
    K + 1
)


fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.plot(
    mode_axis,
    rvmd_result[
        "frequency_hz"
    ],
    marker="o",
    label="RVMD"
)


ax.plot(
    mode_axis,
    orvmd_result[
        "frequency_hz"
    ],
    marker="s",
    label="ORVMD"
)


ax.set_xticks(
    mode_axis
)

ax.set_xlabel(
    "Mode"
)

ax.set_ylabel(
    "Center frequency (Hz)"
)

ax.set_title(
    "UVW field: center frequency comparison"
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
        "uvw_center_frequency_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 13. 图3：模态能量占比
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.plot(
    mode_axis,
    rvmd_result[
        "energy_fraction"
    ],
    marker="o",
    label="RVMD"
)


ax.plot(
    mode_axis,
    orvmd_result[
        "energy_fraction"
    ],
    marker="s",
    label="ORVMD"
)


ax.set_xticks(
    mode_axis
)

ax.set_xlabel(
    "Mode"
)

ax.set_ylabel(
    "Energy fraction"
)

ax.set_title(
    "UVW field: modal energy comparison"
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
        "uvw_mode_energy_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 14. 图4：RVMD 相干矩阵
# ============================================================

fig, ax = plt.subplots(
    figsize=(6, 5)
)


im = ax.imshow(
    rvmd_result[
        "coherence_matrix"
    ],
    vmin=0,
    vmax=1,
    aspect="auto"
)


ax.set_title(
    "RVMD modal coherence matrix"
)

ax.set_xlabel(
    "Mode"
)

ax.set_ylabel(
    "Mode"
)


ticks = np.arange(K)

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


fig.colorbar(
    im,
    ax=ax,
    label="Coherence"
)


fig.tight_layout()


fig.savefig(
    os.path.join(
        RESULT_DIR,
        "rvmd_coherence_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 15. 图5：ORVMD 相干矩阵
# ============================================================

fig, ax = plt.subplots(
    figsize=(6, 5)
)


im = ax.imshow(
    orvmd_result[
        "coherence_matrix"
    ],
    vmin=0,
    vmax=1,
    aspect="auto"
)


ax.set_title(
    "ORVMD modal coherence matrix"
)

ax.set_xlabel(
    "Mode"
)

ax.set_ylabel(
    "Mode"
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


fig.colorbar(
    im,
    ax=ax,
    label="Coherence"
)


fig.tight_layout()


fig.savefig(
    os.path.join(
        RESULT_DIR,
        "orvmd_coherence_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 16. 图6：空间模态 phi 对比
# ============================================================

for k in range(K):

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    channel_axis = np.arange(
        S
    )


    ax.plot(
        channel_axis,
        rvmd_result[
            "phi"
        ][:, k],
        marker="o",
        label="RVMD"
    )


    ax.plot(
        channel_axis,
        orvmd_result[
            "phi"
        ][:, k],
        marker="s",
        label="ORVMD"
    )


    ax.set_xticks(
        channel_axis
    )

    ax.set_xticklabels(
        CHANNEL_NAMES,
        rotation=30
    )

    ax.set_ylabel(
        "Spatial mode coefficient"
    )

    ax.set_title(
        f"Mode {k + 1}: spatial mode comparison"
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
            f"mode_{k + 1}_spatial_comparison.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


# ============================================================
# 17. 图7：时间系数 c 对比
# ============================================================

for k in range(K):

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )


    ax.plot(
        time,
        rvmd_result[
            "c"
        ][:, k],
        linewidth=1.0,
        label="RVMD"
    )


    ax.plot(
        time,
        orvmd_result[
            "c"
        ][:, k],
        linewidth=1.0,
        label="ORVMD"
    )


    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Temporal coefficient"
    )

    ax.set_title(
        (
            f"Mode {k + 1}: temporal coefficient "
            f"(RVMD vs ORVMD)"
        )
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
            f"mode_{k + 1}_temporal_comparison.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


# ============================================================
# 18. 图8：收敛过程
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)


rvmd_diff = rvmd_result[
    "difference_history"
]

orvmd_diff = orvmd_result[
    "difference_history"
]


ax.plot(
    np.arange(
        1,
        len(rvmd_diff) + 1
    ),
    rvmd_diff,
    label="RVMD"
)


ax.plot(
    np.arange(
        1,
        len(orvmd_diff) + 1
    ),
    orvmd_diff,
    label="ORVMD"
)


ax.axhline(
    TOL,
    linestyle="--",
    label="Tolerance"
)


ax.set_yscale(
    "log"
)

ax.set_xlabel(
    "Iteration"
)

ax.set_ylabel(
    "Difference"
)

ax.set_title(
    "UVW field: convergence comparison"
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
        "uvw_convergence_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 19. 图9：每个输入通道的原信号与重构结果
# ============================================================

for s in range(S):

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )


    ax.plot(
        time,
        Q[s],
        linewidth=1.0,
        label="Original"
    )


    ax.plot(
        time,
        rvmd_result[
            "reconstruction"
        ][s],
        linewidth=1.0,
        label="RVMD reconstruction"
    )


    ax.plot(
        time,
        orvmd_result[
            "reconstruction"
        ][s],
        linewidth=1.0,
        label="ORVMD reconstruction"
    )


    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Velocity fluctuation (m/s)"
    )

    ax.set_title(
        f"Reconstruction comparison: {CHANNEL_NAMES[s]}"
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
            f"reconstruction_{CHANNEL_NAMES[s]}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


# ============================================================
# 20. 保存重构数据
# ============================================================

np.save(
    os.path.join(
        RESULT_DIR,
        "rvmd_reconstruction.npy"
    ),
    rvmd_result[
        "reconstruction"
    ]
)


np.save(
    os.path.join(
        RESULT_DIR,
        "orvmd_reconstruction.npy"
    ),
    orvmd_result[
        "reconstruction"
    ]
)


# ============================================================
# 21. 打印最终结果
# ============================================================

print("\n")
print("=" * 80)
print("RVMD vs ORVMD 正式比较结果")
print("=" * 80)

print(
    summary.to_string(
        index=False
    )
)


print("\n" + "-" * 80)

print(
    "平均相干度降低 =",
    mean_improvement,
    "%"
)

print(
    "最大相干度降低 =",
    max_improvement,
    "%"
)

print(
    "重构误差变化 =",
    reconstruction_change,
    "%"
)


print("\n" + "=" * 80)
print("实验完成")
print("=" * 80)

print(
    "结果目录："
)

print(
    RESULT_DIR
)


print("\n主要结果文件：")

print(
    "uvw_rvmd_orvmd_summary.csv"
)

print(
    "uvw_mode_energy_comparison.csv"
)

print(
    "uvw_center_frequency_comparison.csv"
)

print(
    "uvw_coherence_comparison.png"
)

print(
    "rvmd_coherence_matrix.png"
)

print(
    "orvmd_coherence_matrix.png"
)

print(
    "uvw_center_frequency_comparison.png"
)

print(
    "uvw_mode_energy_comparison.png"
)

print(
    "uvw_convergence_comparison.png"
)

print(
    "mode_1_spatial_comparison.png ~ mode_4_spatial_comparison.png"
)

print(
    "mode_1_temporal_comparison.png ~ mode_4_temporal_comparison.png"
)

print(
    "reconstruction_*.png"
)

print("=" * 80)