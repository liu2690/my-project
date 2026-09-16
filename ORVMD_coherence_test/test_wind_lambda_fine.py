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

DATA_DIR = os.path.join(
    ROOT,
    "wind_field_analysis",
    "selected_data"
)

RESULT_DIR = os.path.join(
    THIS_DIR,
    "result",
    "wind_lambda_fine"
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

FS = 50.0

N_SAMPLES = 5000

K = 4

ORVMD_ALPHA = 1000.0

ORVMD_TOL = 5e-3

ORVMD_MAX_STEPS = 500


# ============================================================
# 4. 细粒度 lambda
# ============================================================

LAMBDA_LIST = [
    0.0,
    0.0001,
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.003,
    0.005
]


# ============================================================
# 5. 读取数据
# ============================================================

data_path = os.path.join(
    DATA_DIR,
    "spanwise_fluctuation.npy"
)

Q_full = np.load(
    data_path
)

Q = Q_full[
    :,
    :N_SAMPLES
].astype(
    np.float64
)

S, T = Q.shape


print("\n" + "=" * 80)
print("ORVMD lambda_orth 细粒度扫描")
print("=" * 80)

print(
    "Q shape =",
    Q.shape
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

    for k in range(K_local):

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
        )
    )

    for i in range(K_local):

        Fi = fields[i]

        norm_i = np.linalg.norm(
            Fi
        )

        for j in range(K_local):

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

    print("\n" + "#" * 80)
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


    phi = np.asarray(
        mode["phi"]
    )

    c = np.asarray(
        mode["c"]
    )

    omega = np.asarray(
        mode["omega"]
    )


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


    omega_hz = (
        omega
        *
        FS
    )


    difference_history_dict[
        lam
    ] = diff_history

    coherence_dict[
        lam
    ] = C


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


# ============================================================
# 8. DataFrame
# ============================================================

df = pd.DataFrame(
    records
)


# ============================================================
# 9. 以 lambda=0 作为 RVMD baseline
# ============================================================

baseline_row = df[
    df["lambda_orth"] == 0.0
].iloc[0]

baseline_mean_coh = float(
    baseline_row[
        "mean_offdiag_coherence"
    ]
)

baseline_max_coh = float(
    baseline_row[
        "max_offdiag_coherence"
    ]
)


df[
    "mean_coherence_improvement_percent"
] = (
    (
        baseline_mean_coh
        -
        df["mean_offdiag_coherence"]
    )
    /
    baseline_mean_coh
    *
    100.0
)


df[
    "max_coherence_improvement_percent"
] = (
    (
        baseline_max_coh
        -
        df["max_offdiag_coherence"]
    )
    /
    baseline_max_coh
    *
    100.0
)


# ============================================================
# 10. 自动寻找最佳 lambda
#
# 条件：
# 1. 必须收敛
# 2. 相干度低于 lambda=0
# 3. 在这些候选中平均相干度最低
# ============================================================

valid = df[
    (df["converged"] == True)
    &
    (
        df["mean_offdiag_coherence"]
        <
        baseline_mean_coh
    )
].copy()


if len(valid) > 0:

    best_row = valid.sort_values(
        "mean_offdiag_coherence"
    ).iloc[0]

    best_lambda = float(
        best_row["lambda_orth"]
    )

    best_found = True

else:

    best_lambda = 0.0

    best_found = False


# ============================================================
# 11. 保存表格
# ============================================================

csv_path = os.path.join(
    RESULT_DIR,
    "lambda_fine_scan_summary.csv"
)

df.to_csv(
    csv_path,
    index=False
)


print("\n" + "=" * 80)
print("细粒度扫描结果")
print("=" * 80)

print(
    df.to_string(
        index=False
    )
)


print("\n" + "=" * 80)

if best_found:

    print(
        "自动筛选出的候选 lambda =",
        best_lambda
    )

else:

    print(
        "当前扫描范围内没有找到"
        "“收敛且平均相干度优于 lambda=0”"
        "的正 lambda。"
    )

print("=" * 80)


# ============================================================
# 12. 图1：lambda vs 平均相干度
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    df["lambda_orth"],
    df["mean_offdiag_coherence"],
    marker="o"
)

ax.axhline(
    baseline_mean_coh,
    linestyle="--",
    label="RVMD baseline (lambda=0)"
)

ax.set_xlabel(
    "lambda_orth"
)

ax.set_ylabel(
    "Mean off-diagonal coherence"
)

ax.set_title(
    "Fine scan: modal coherence vs lambda_orth"
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
        "fine_lambda_mean_coherence.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 13. 图2：重构误差
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    df["lambda_orth"],
    df["reconstruction_error"],
    marker="o"
)

ax.set_xlabel(
    "lambda_orth"
)

ax.set_ylabel(
    "Reconstruction error"
)

ax.set_title(
    "Fine scan: reconstruction error"
)

ax.grid(
    True,
    alpha=0.3
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        RESULT_DIR,
        "fine_lambda_reconstruction_error.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 14. 图3：相对 RVMD 改善率
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    df["lambda_orth"],
    df[
        "mean_coherence_improvement_percent"
    ],
    marker="o",
    label="Mean coherence"
)

ax.plot(
    df["lambda_orth"],
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
    "Improvement relative to lambda=0 (%)"
)

ax.set_title(
    "Orthogonality improvement relative to RVMD"
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
        "coherence_improvement_percent.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 15. 图4：全部收敛曲线
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)


for lam in LAMBDA_LIST:

    diff = difference_history_dict[
        lam
    ]

    steps_axis = np.arange(
        1,
        len(diff) + 1
    )

    ax.plot(
        steps_axis,
        diff,
        linewidth=1.2,
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
    "Convergence histories"
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
        "convergence_histories.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 16. 图5：只看后期收敛
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

    steps_axis = np.arange(
        start + 1,
        len(diff) + 1
    )

    ax.plot(
        steps_axis,
        diff[start:],
        linewidth=1.2,
        label=f"lambda={lam}"
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
    "Last 100 iterations"
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
        "convergence_last100.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 17. 保存相干矩阵
# ============================================================

for lam, C in coherence_dict.items():

    safe = str(
        lam
    ).replace(
        ".",
        "_"
    )

    np.savetxt(
        os.path.join(
            RESULT_DIR,
            f"coherence_lambda_{safe}.csv"
        ),
        C,
        delimiter=","
    )


# ============================================================
# 18. 最终输出
# ============================================================

print("\n" + "#" * 80)
print("lambda_orth 细粒度扫描完成")
print("#" * 80)

print(
    "结果目录：",
    RESULT_DIR
)

print(
    "结果CSV：",
    csv_path
)

print(
    "主要图片："
)

print(
    "fine_lambda_mean_coherence.png"
)

print(
    "fine_lambda_reconstruction_error.png"
)

print(
    "coherence_improvement_percent.png"
)

print(
    "convergence_histories.png"
)

print(
    "convergence_last100.png"
)

print("#" * 80)