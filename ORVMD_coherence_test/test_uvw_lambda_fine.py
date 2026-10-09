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
    "uvw_lambda_fine"
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
# 3. 参数
# ============================================================

FS = 20.0

K = 4

ORVMD_ALPHA = 1000.0

ORVMD_TOL = 5e-3

ORVMD_MAX_STEPS = 500


# ============================================================
# 4. UVW 细粒度 lambda
# ============================================================

LAMBDA_LIST = [
    0.0,
    0.0200,
    0.0225,
    0.0250,
    0.0275,
    0.0300,
    0.0325,
    0.0350,
    0.0375,
    0.0400,
    0.0425,
    0.0450,
    0.0475,
    0.0500
]


# ============================================================
# 5. 读取数据
# ============================================================

Q = np.load(
    DATA_PATH
).astype(
    np.float64
)


if Q.shape != (
    6,
    5000
):

    raise ValueError(
        f"UVW 输入 shape 异常：{Q.shape}，"
        "预期为 (6, 5000)"
    )


S, T = Q.shape


print("\n" + "=" * 80)
print("ORVMD 三分量 UVW lambda_orth 细粒度扫描")
print("=" * 80)

print(
    "Q shape =",
    Q.shape
)

print(
    "FS =",
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
# 6. 工具函数
# ============================================================

def reconstruction_error(
    original,
    reconstructed
):

    return float(
        np.linalg.norm(
            original - reconstructed
        )
        /
        max(
            np.linalg.norm(original),
            np.finfo(float).eps
        )
    )


def build_mode_fields(
    phi,
    c
):

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
# 7. 扫描
# ============================================================

records = []

difference_history_dict = {}

coherence_dict = {}


for lam in LAMBDA_LIST:

    print("\n")
    print("#" * 80)
    print(
        f"lambda_orth = {lam}"
    )
    print("#" * 80)


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
    # 提取结果
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


    if phi.shape != (
        S,
        K
    ):

        raise RuntimeError(
            f"phi shape 异常：{phi.shape}"
        )


    if c.shape != (
        T,
        K
    ):

        raise RuntimeError(
            f"c shape 异常：{c.shape}"
        )


    # ========================================================
    # 构造时空模态
    # ========================================================

    fields = build_mode_fields(
        phi,
        c
    )


    reconstructed = np.sum(
        fields,
        axis=0
    )


    rec_err = reconstruction_error(
        Q,
        reconstructed
    )


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
    # 收敛信息
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


    final_diff = float(
        diff_history[-1]
    )


    difference_history_dict[
        lam
    ] = diff_history


    # ========================================================
    # 中心频率
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


    records.append(
        {
            "lambda_orth":
                lam,

            "reconstruction_error":
                rec_err,

            "mean_offdiag_coherence":
                mean_coh,

            "max_offdiag_coherence":
                max_coh,

            "converged":
                converged,

            "steps":
                steps,

            "final_difference":
                final_diff,

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
        final_diff
    )

    print(
        "reconstruction error =",
        rec_err
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


# ============================================================
# 8. DataFrame
# ============================================================

df = pd.DataFrame(
    records
)


# ============================================================
# 9. lambda=0 作为 RVMD baseline
# ============================================================

baseline_row = df[
    df[
        "lambda_orth"
    ] == 0.0
].iloc[0]


baseline_mean = float(
    baseline_row[
        "mean_offdiag_coherence"
    ]
)


baseline_max = float(
    baseline_row[
        "max_offdiag_coherence"
    ]
)


baseline_rec = float(
    baseline_row[
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


# ============================================================
# 10. 自动选择候选 lambda
#
# 条件：
# 1. lambda > 0
# 2. 必须收敛
# 3. 平均相干度低于 RVMD baseline
# 4. 在这些结果中平均相干度最低
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
    &
    (
        df[
            "mean_offdiag_coherence"
        ]
        <
        baseline_mean
    )
].copy()


if len(
    valid
) > 0:

    best_row = valid.sort_values(
        [
            "mean_offdiag_coherence",
            "max_offdiag_coherence"
        ]
    ).iloc[0]


    best_lambda = float(
        best_row[
            "lambda_orth"
        ]
    )


    best_found = True

else:

    best_lambda = None

    best_found = False


# ============================================================
# 11. 保存 CSV
# ============================================================

CSV_PATH = os.path.join(
    RESULT_DIR,
    "uvw_lambda_fine_summary.csv"
)


df.to_csv(
    CSV_PATH,
    index=False
)


print("\n")
print("=" * 80)
print("UVW 三分量细粒度扫描结果")
print("=" * 80)


print(
    df.to_string(
        index=False
    )
)


print("\n" + "=" * 80)

if best_found:

    print(
        "自动筛选候选 lambda =",
        best_lambda
    )

else:

    print(
        "当前范围内没有找到满足条件的正 lambda"
    )

print("=" * 80)


# ============================================================
# 12. 图1：平均相干度
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
    "UVW field: fine lambda scan"
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
        "uvw_fine_lambda_mean_coherence.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 13. 图2：最大相干度
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
        "uvw_fine_lambda_max_coherence.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 14. 图3：相干度改善率
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
# 15. 图4：重构误差
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
        "uvw_fine_lambda_reconstruction_error.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 16. 图5：误差 - 相干度 trade-off
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
        f"{row['lambda_orth']:.4f}",
        (
            row[
                "reconstruction_error"
            ],
            row[
                "mean_offdiag_coherence"
            ]
        ),
        fontsize=8
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
        "uvw_tradeoff_error_coherence_fine.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 17. 图6：完整收敛曲线
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)


for lam in LAMBDA_LIST:

    diff = difference_history_dict[
        lam
    ]


    iteration_axis = np.arange(
        1,
        len(diff) + 1
    )


    ax.plot(
        iteration_axis,
        diff,
        linewidth=1.0,
        label=f"lambda={lam:g}"
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
    fontsize=7
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
# 18. 图7：最后100次迭代
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)


for lam in LAMBDA_LIST:

    diff = difference_history_dict[
        lam
    ]


    start = max(
        0,
        len(diff) - 100
    )


    iteration_axis = np.arange(
        start + 1,
        len(diff) + 1
    )


    ax.plot(
        iteration_axis,
        diff[start:],
        linewidth=1.0,
        label=f"lambda={lam:g}"
    )


ax.axhline(
    ORVMD_TOL,
    linestyle="--",
    label="Tolerance"
)


ax.set_xlabel(
    "Iteration"
)

ax.set_ylabel(
    "Difference"
)

ax.set_title(
    "UVW field: last 100 iterations"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend(
    fontsize=7
)


fig.tight_layout()


fig.savefig(
    os.path.join(
        RESULT_DIR,
        "uvw_convergence_last100.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 19. 保存各 lambda 相干矩阵
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
# 20. 最终输出
# ============================================================

print("\n")
print("#" * 80)
print("UVW 三分量 lambda_orth 细粒度扫描完成")
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


if best_found:

    print(
        "\n当前候选 lambda：",
        best_lambda
    )


print(
    "\n主要图片："
)

print(
    "uvw_fine_lambda_mean_coherence.png"
)

print(
    "uvw_fine_lambda_max_coherence.png"
)

print(
    "uvw_coherence_improvement_percent.png"
)

print(
    "uvw_fine_lambda_reconstruction_error.png"
)

print(
    "uvw_tradeoff_error_coherence_fine.png"
)

print(
    "uvw_convergence_histories.png"
)

print(
    "uvw_convergence_last100.png"
)

print("#" * 80)