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
    "uvw_lambda_test"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# 2. 导入 ORVMD
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

ORVMD_ALPHA = 1000.0

ORVMD_TOL = 5e-3

ORVMD_MAX_STEPS = 500


# ------------------------------------------------------------
# lambda 粗扫描
# ------------------------------------------------------------

LAMBDA_LIST = [
    0.0,
    0.001,
    0.005,
    0.01,
    0.02,
    0.05
]


# ============================================================
# 4. 读取三分量脉动风速
# ============================================================

Q = np.load(
    DATA_PATH
).astype(
    np.float64
)


print("\n" + "=" * 80)
print("ORVMD 三分量 UVW 风场 lambda_orth 粗扫描")
print("=" * 80)


print(
    "Q shape =",
    Q.shape
)


if Q.ndim != 2:

    raise ValueError(
        "UVW 输入数据必须为二维 S x T 矩阵"
    )


if Q.shape != (
    6,
    5000
):

    raise ValueError(
        "当前输入数据 shape 与预期 (6, 5000) 不一致："
        f"{Q.shape}"
    )


S, T = Q.shape


print(
    "空间/通道数 S =",
    S
)

print(
    "时间点数 T =",
    T
)

print(
    "采样频率 FS =",
    FS,
    "Hz"
)

print(
    "持续时间 =",
    T / FS,
    "s"
)

print(
    "K =",
    K
)

print(
    "Alpha =",
    ORVMD_ALPHA
)

print(
    "Tolerance =",
    ORVMD_TOL
)

print(
    "MaximumSteps =",
    ORVMD_MAX_STEPS
)

print(
    "Lambda list =",
    LAMBDA_LIST
)


# ============================================================
# 5. 工具函数
# ============================================================

def relative_reconstruction_error(
    original,
    reconstructed
):
    """
    相对 Frobenius 重构误差：

             ||Q - Q_hat||_F
        E = ------------------
                 ||Q||_F

    越小越好。
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


def build_mode_fields(
    phi,
    c
):
    """
    根据 ORVMD 输出构造每一阶完整时空模态。

    phi:
        S x K

    c:
        T x K

    第 k 阶模态：

        Q_k = phi_k c_k^T

    返回：
        K x S x T
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


    for k in range(
        K_local
    ):

        fields[k] = np.outer(
            phi[:, k],
            c[:, k]
        )


    return fields


def coherence_matrix(
    fields
):
    """
    计算时空模态之间的归一化内积：

                    | <Qi,Qj>_F |
        C_ij = -------------------------
                ||Qi||_F * ||Qj||_F

    对角线为 1。

    非对角元素越小，
    表示不同模态之间越独立。
    """

    K_local = fields.shape[0]

    C = np.zeros(
        (
            K_local,
            K_local
        ),
        dtype=float
    )


    for i in range(
        K_local
    ):

        Fi = fields[i]

        norm_i = np.linalg.norm(
            Fi
        )


        for j in range(
            K_local
        ):

            Fj = fields[j]

            norm_j = np.linalg.norm(
                Fj
            )


            denominator = (
                norm_i
                *
                norm_j
            )


            if denominator <= 0:

                C[i, j] = 0.0

            else:

                C[i, j] = (
                    np.abs(
                        np.vdot(
                            Fi,
                            Fj
                        )
                    )
                    /
                    denominator
                )


    return C


def offdiag_mean(
    C
):
    """
    平均非对角相干度。

    越小越好。
    """

    mask = ~np.eye(
        C.shape[0],
        dtype=bool
    )

    return float(
        np.mean(
            C[mask]
        )
    )


def offdiag_max(
    C
):
    """
    最大非对角相干度。

    越小越好。
    """

    mask = ~np.eye(
        C.shape[0],
        dtype=bool
    )

    return float(
        np.max(
            C[mask]
        )
    )


# ============================================================
# 6. lambda 粗扫描
# ============================================================

records = []

coherence_dict = {}

difference_history_dict = {}


for lam in LAMBDA_LIST:

    print("\n")
    print("#" * 80)

    print(
        f"lambda_orth = {lam}"
    )

    print("#" * 80)


    # ========================================================
    # ORVMD
    #
    # 和已经成功的垂向实验完全保持一致
    # ========================================================

    mode, info, _ = orvmd(

        Q=Q,

        K=K,

        Alpha=ORVMD_ALPHA,

        Weight=1,

        lambda_orth=lam,

        Tolerance=ORVMD_TOL,

        MaximumSteps=ORVMD_MAX_STEPS,

        InitFreqType=1,

        InitFreqMaximum=0.5,

        Device="cpu",

        FPPrecision="double",

        nDC=0,

        Display="off",

        Restart=None
    )


    # ========================================================
    # 7. 提取 ORVMD 输出
    # ========================================================

    phi = np.asarray(
        mode["phi"]
    )

    c = np.asarray(
        mode["c"]
    )

    omega = np.asarray(
        mode["omega"]
    )


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


    # --------------------------------------------------------
    # 输出维度检查
    # --------------------------------------------------------

    if phi.shape != (
        S,
        K
    ):

        raise RuntimeError(
            "ORVMD phi shape 异常："
            f"{phi.shape}，"
            f"预期 {(S, K)}"
        )


    if c.shape != (
        T,
        K
    ):

        raise RuntimeError(
            "ORVMD c shape 异常："
            f"{c.shape}，"
            f"预期 {(T, K)}"
        )


    # ========================================================
    # 8. 构造 K 个时空模态
    # ========================================================

    fields = build_mode_fields(
        phi,
        c
    )


    # shape:
    # K x 6 x 5000

    print(
        "mode fields shape =",
        fields.shape
    )


    # ========================================================
    # 9. 重构
    # ========================================================

    reconstructed = np.sum(
        fields,
        axis=0
    )


    rec_error = (
        relative_reconstruction_error(
            Q,
            reconstructed
        )
    )


    # ========================================================
    # 10. 模态相干矩阵
    # ========================================================

    C = coherence_matrix(
        fields
    )


    mean_coh = offdiag_mean(
        C
    )


    max_coh = offdiag_max(
        C
    )


    coherence_dict[
        lam
    ] = C


    # ========================================================
    # 11. 收敛信息
    # ========================================================

    iteration_info = info[
        "Iteration"
    ]


    converged = iteration_info[
        "converged"
    ]


    steps = iteration_info[
        "steps"
    ]


    diff_history = np.asarray(
        iteration_info[
            "difference"
        ]
    )


    final_difference = float(
        diff_history[-1]
    )


    difference_history_dict[
        lam
    ] = diff_history


    # ========================================================
    # 12. 中心频率
    #
    # 与垂向实验保持完全一致：
    #
    # f_Hz = omega * FS
    # ========================================================

    omega_hz = (
        omega
        *
        FS
    )


    omega_hz = np.asarray(
        omega_hz
    ).reshape(
        -1
    )


    # ========================================================
    # 13. 输出
    # ========================================================

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
        omega_hz
    )


    # ========================================================
    # 14. 保存当前结果
    # ========================================================

    records.append(
        {
            "lambda_orth":
                lam,

            "reconstruction_error":
                rec_error,

            "mean_offdiag_coherence":
                mean_coh,

            "max_offdiag_coherence":
                max_coh,

            "converged":
                converged,

            "steps":
                steps,

            "final_difference":
                final_difference,

            "f1_Hz":
                omega_hz[0],

            "f2_Hz":
                omega_hz[1],

            "f3_Hz":
                omega_hz[2],

            "f4_Hz":
                omega_hz[3]
        }
    )


# ============================================================
# 15. 汇总表
# ============================================================

df = pd.DataFrame(
    records
)


CSV_PATH = os.path.join(
    RESULT_DIR,
    "uvw_lambda_scan_summary.csv"
)


df.to_csv(
    CSV_PATH,
    index=False
)


print("\n")
print("=" * 80)
print("三分量 UVW lambda 粗扫描结果")
print("=" * 80)


print(
    df.to_string(
        index=False
    )
)


# ============================================================
# 16. 计算相对于 RVMD baseline 的改善率
#
# lambda = 0:
#     RVMD baseline
# ============================================================

baseline = df[
    df[
        "lambda_orth"
    ] == 0.0
].iloc[0]


baseline_mean = float(
    baseline[
        "mean_offdiag_coherence"
    ]
)


baseline_max = float(
    baseline[
        "max_offdiag_coherence"
    ]
)


baseline_rec = float(
    baseline[
        "reconstruction_error"
    ]
)


df[
    "mean_coherence_improvement_percent"
] = (
    (
        baseline_mean
        -
        df[
            "mean_offdiag_coherence"
        ]
    )
    /
    baseline_mean
    *
    100.0
)


df[
    "max_coherence_improvement_percent"
] = (
    (
        baseline_max
        -
        df[
            "max_offdiag_coherence"
        ]
    )
    /
    baseline_max
    *
    100.0
)


df[
    "reconstruction_change_percent"
] = (
    (
        df[
            "reconstruction_error"
        ]
        -
        baseline_rec
    )
    /
    baseline_rec
    *
    100.0
)


# 重新保存加入改善率后的表格

df.to_csv(
    CSV_PATH,
    index=False
)


print("\n")
print("=" * 80)
print("相对于 RVMD (lambda=0) 的变化")
print("=" * 80)


print(
    df[
        [
            "lambda_orth",
            "mean_coherence_improvement_percent",
            "max_coherence_improvement_percent",
            "reconstruction_change_percent",
            "converged",
            "steps"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 17. 图1：平均相干度
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.plot(
    df[
        "lambda_orth"
    ],
    df[
        "mean_offdiag_coherence"
    ],
    marker="o"
)


ax.axhline(
    baseline_mean,
    linestyle="--",
    label="RVMD baseline"
)


ax.set_xlabel(
    "lambda_orth"
)

ax.set_ylabel(
    "Mean off-diagonal coherence"
)

ax.set_title(
    "UVW field: mean modal coherence"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend()


fig.tight_layout()


fig.savefig(
    os.path.join(
        RESULT_DIR,
        "uvw_lambda_mean_coherence.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 18. 图2：最大相干度
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.plot(
    df[
        "lambda_orth"
    ],
    df[
        "max_offdiag_coherence"
    ],
    marker="o"
)


ax.axhline(
    baseline_max,
    linestyle="--",
    label="RVMD baseline"
)


ax.set_xlabel(
    "lambda_orth"
)

ax.set_ylabel(
    "Maximum off-diagonal coherence"
)

ax.set_title(
    "UVW field: maximum modal coherence"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend()


fig.tight_layout()


fig.savefig(
    os.path.join(
        RESULT_DIR,
        "uvw_lambda_max_coherence.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 19. 图3：重构误差
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.plot(
    df[
        "lambda_orth"
    ],
    df[
        "reconstruction_error"
    ],
    marker="o"
)


ax.axhline(
    baseline_rec,
    linestyle="--",
    label="RVMD baseline"
)


ax.set_xlabel(
    "lambda_orth"
)

ax.set_ylabel(
    "Reconstruction error"
)

ax.set_title(
    "UVW field: reconstruction error"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend()


fig.tight_layout()


fig.savefig(
    os.path.join(
        RESULT_DIR,
        "uvw_lambda_reconstruction_error.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 20. 图4：最终 difference
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.plot(
    df[
        "lambda_orth"
    ],
    df[
        "final_difference"
    ],
    marker="o"
)


ax.axhline(
    ORVMD_TOL,
    linestyle="--",
    label="Tolerance"
)


ax.set_xlabel(
    "lambda_orth"
)

ax.set_ylabel(
    "Final difference"
)

ax.set_title(
    "UVW field: convergence indicator"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend()


fig.tight_layout()


fig.savefig(
    os.path.join(
        RESULT_DIR,
        "uvw_lambda_convergence.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 21. 图5：重构误差 - 相干度 trade-off
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)


ax.plot(
    df[
        "reconstruction_error"
    ],
    df[
        "mean_offdiag_coherence"
    ],
    marker="o"
)


for _, row in df.iterrows():

    ax.annotate(
        f"{row['lambda_orth']}",
        (
            row[
                "reconstruction_error"
            ],
            row[
                "mean_offdiag_coherence"
            ]
        )
    )


ax.set_xlabel(
    "Reconstruction error"
)

ax.set_ylabel(
    "Mean off-diagonal coherence"
)

ax.set_title(
    "UVW field: reconstruction-coherence trade-off"
)

ax.grid(
    True,
    alpha=0.3
)


fig.tight_layout()


fig.savefig(
    os.path.join(
        RESULT_DIR,
        "uvw_tradeoff_error_coherence.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 22. 图6：相干度改善率
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.plot(
    df[
        "lambda_orth"
    ],
    df[
        "mean_coherence_improvement_percent"
    ],
    marker="o",
    label="Mean coherence"
)


ax.plot(
    df[
        "lambda_orth"
    ],
    df[
        "max_coherence_improvement_percent"
    ],
    marker="s",
    label="Max coherence"
)


ax.axhline(
    0,
    linestyle="--"
)


ax.set_xlabel(
    "lambda_orth"
)

ax.set_ylabel(
    "Improvement relative to RVMD (%)"
)

ax.set_title(
    "UVW field: orthogonality improvement"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend()


fig.tight_layout()


fig.savefig(
    os.path.join(
        RESULT_DIR,
        "uvw_coherence_improvement_percent.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 23. 图7：完整收敛曲线
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)


for lam in LAMBDA_LIST:

    history = difference_history_dict[
        lam
    ]


    iteration_axis = np.arange(
        1,
        len(
            history
        ) + 1
    )


    ax.plot(
        iteration_axis,
        history,
        linewidth=1.0,
        label=f"lambda={lam}"
    )


ax.axhline(
    ORVMD_TOL,
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
    "UVW field: convergence histories"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend(
    fontsize=8
)


fig.tight_layout()


fig.savefig(
    os.path.join(
        RESULT_DIR,
        "uvw_convergence_histories.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 24. 保存各 lambda 相干矩阵
# ============================================================

for lam, C in coherence_dict.items():

    safe_name = str(
        lam
    ).replace(
        ".",
        "_"
    )


    np.savetxt(
        os.path.join(
            RESULT_DIR,
            f"uvw_coherence_lambda_{safe_name}.csv"
        ),
        C,
        delimiter=","
    )


# ============================================================
# 25. 自动筛选粗扫描候选
#
# 必须首先满足：
#     converged = True
#
# 然后在已收敛正 lambda 中
# 找平均相干度最低的一个。
#
# 这里只作为后续细扫描的候选，
# 不是最终最优参数。
# ============================================================

valid = df[
    (
        df[
            "lambda_orth"
        ] > 0
    )
    &
    (
        df[
            "converged"
        ] == True
    )
].copy()


if len(
    valid
) > 0:

    candidate = valid.sort_values(
        [
            "mean_offdiag_coherence",
            "max_offdiag_coherence"
        ]
    ).iloc[0]


    candidate_lambda = float(
        candidate[
            "lambda_orth"
        ]
    )


    print("\n")
    print("=" * 80)

    print(
        "粗扫描候选 lambda =",
        candidate_lambda
    )

    print(
        "注意：这里只用于确定细扫描范围，"
        "不是最终最优 lambda。"
    )

    print("=" * 80)


# ============================================================
# 26. 完成
# ============================================================

print("\n")
print("#" * 80)
print("三分量 UVW lambda_orth 粗粒度扫描完成")
print("#" * 80)


print(
    "结果目录："
)

print(
    RESULT_DIR
)


print(
    "\n结果表："
)

print(
    CSV_PATH
)


print(
    "\n主要图片："
)

print(
    "uvw_lambda_mean_coherence.png"
)

print(
    "uvw_lambda_max_coherence.png"
)

print(
    "uvw_lambda_reconstruction_error.png"
)

print(
    "uvw_lambda_convergence.png"
)

print(
    "uvw_tradeoff_error_coherence.png"
)

print(
    "uvw_coherence_improvement_percent.png"
)

print(
    "uvw_convergence_histories.png"
)

print("#" * 80)