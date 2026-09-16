import os
import sys
import importlib.util

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
    "wind_lambda_test"
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

FS = 50.0

N_SAMPLES = 5000

K = 4

ORVMD_ALPHA = 1000.0

ORVMD_TOL = 5e-3

ORVMD_MAX_STEPS = 500


# ------------------------------------------------------------
# lambda 扫描
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
# 4. 读取展向脉动风速
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
print("ORVMD lambda_orth 参数扫描")
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
# 5. 指标函数
# ============================================================

def relative_reconstruction_error(
    original,
    reconstructed
):

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
        ),
        dtype=float
    )

    for i in range(K_local):

        fi = fields[i]

        norm_i = np.linalg.norm(
            fi
        )

        for j in range(K_local):

            fj = fields[j]

            norm_j = np.linalg.norm(
                fj
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
                            fi,
                            fj
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
# 6. 扫描 lambda
# ============================================================

records = []

all_coherence = {}


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


    rec_error = (
        relative_reconstruction_error(
            Q,
            reconstructed
        )
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


    converged = info[
        "Iteration"
    ][
        "converged"
    ]


    steps = info[
        "Iteration"
    ][
        "steps"
    ]


    final_difference = info[
        "Iteration"
    ][
        "difference"
    ][
        -1
    ]


    omega_hz = (
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
        omega_hz
    )


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


    all_coherence[
        lam
    ] = C


# ============================================================
# 7. 保存 CSV
# ============================================================

df = pd.DataFrame(
    records
)

csv_path = os.path.join(
    RESULT_DIR,
    "lambda_scan_summary.csv"
)

df.to_csv(
    csv_path,
    index=False
)


print("\n")
print("=" * 80)
print("lambda 扫描结果")
print("=" * 80)

print(
    df.to_string(
        index=False
    )
)


# ============================================================
# 8. 图1：lambda - 重构误差
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
    "Reconstruction error vs lambda_orth"
)

ax.grid(
    True,
    alpha=0.3
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        RESULT_DIR,
        "lambda_reconstruction_error.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 9. 图2：lambda - 平均模态相干度
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    df["lambda_orth"],
    df["mean_offdiag_coherence"],
    marker="o"
)

ax.set_xlabel(
    "lambda_orth"
)

ax.set_ylabel(
    "Mean off-diagonal coherence"
)

ax.set_title(
    "Modal coherence vs lambda_orth"
)

ax.grid(
    True,
    alpha=0.3
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        RESULT_DIR,
        "lambda_mean_coherence.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 10. 图3：lambda - 最大相干度
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    df["lambda_orth"],
    df["max_offdiag_coherence"],
    marker="o"
)

ax.set_xlabel(
    "lambda_orth"
)

ax.set_ylabel(
    "Maximum off-diagonal coherence"
)

ax.set_title(
    "Maximum modal coherence vs lambda_orth"
)

ax.grid(
    True,
    alpha=0.3
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        RESULT_DIR,
        "lambda_max_coherence.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 11. 图4：误差-相干度折中
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

ax.plot(
    df["reconstruction_error"],
    df["mean_offdiag_coherence"],
    marker="o"
)


for _, row in df.iterrows():

    ax.annotate(
        f"{row['lambda_orth']}",
        (
            row["reconstruction_error"],
            row["mean_offdiag_coherence"]
        )
    )


ax.set_xlabel(
    "Reconstruction error"
)

ax.set_ylabel(
    "Mean off-diagonal coherence"
)

ax.set_title(
    "Reconstruction-coherence trade-off"
)

ax.grid(
    True,
    alpha=0.3
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        RESULT_DIR,
        "tradeoff_error_coherence.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 12. 图5：最终 difference
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    df["lambda_orth"],
    df["final_difference"],
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
    "Convergence indicator vs lambda_orth"
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
        "lambda_convergence.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 13. 保存各 lambda 相干矩阵
# ============================================================

for lam, C in all_coherence.items():

    safe_name = str(
        lam
    ).replace(
        ".",
        "_"
    )

    np.savetxt(
        os.path.join(
            RESULT_DIR,
            f"coherence_lambda_{safe_name}.csv"
        ),
        C,
        delimiter=","
    )


# ============================================================
# 14. 最终输出
# ============================================================

print("\n")
print("#" * 80)
print("lambda_orth 扫描完成")
print("#" * 80)

print(
    "结果目录：",
    RESULT_DIR
)

print(
    "CSV：",
    csv_path
)

print(
    "主要图片："
)

print(
    "lambda_reconstruction_error.png"
)

print(
    "lambda_mean_coherence.png"
)

print(
    "lambda_max_coherence.png"
)

print(
    "tradeoff_error_coherence.png"
)

print(
    "lambda_convergence.png"
)

print("#" * 80)