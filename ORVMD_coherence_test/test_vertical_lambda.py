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
    "vertical_lambda_test"
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

# 与展向实验保持一致
N_SAMPLES = 5000

K = 4

ORVMD_ALPHA = 1000.0

ORVMD_TOL = 5e-3

ORVMD_MAX_STEPS = 500


# ============================================================
# 4. lambda 粗扫描
# ============================================================

LAMBDA_LIST = [
    0.0,
    0.001,
    0.005,
    0.01,
    0.02,
    0.05
]


# ============================================================
# 5. 读取垂向脉动风场
# ============================================================

data_path = os.path.join(
    DATA_DIR,
    "vertical_fluctuation.npy"
)

position_path = os.path.join(
    DATA_DIR,
    "vertical_positions.npy"
)


Q_full = np.load(
    data_path
)

positions = np.load(
    position_path
)


print("\n" + "=" * 80)
print("ORVMD 垂向风场 lambda_orth 粗扫描")
print("=" * 80)

print(
    "原始 Q shape =",
    Q_full.shape
)

print(
    "垂向测点位置 ="
)

print(
    positions
)


if Q_full.ndim != 2:
    raise ValueError(
        "vertical_fluctuation.npy 必须是二维矩阵 S x T"
    )


if Q_full.shape[0] != len(positions):
    raise ValueError(
        "垂向空间测点数量与数据空间维度不一致"
    )


N_REAL = min(
    N_SAMPLES,
    Q_full.shape[1]
)


Q = Q_full[
    :,
    :N_REAL
].astype(
    np.float64
)


S, T = Q.shape


print(
    "本次实验 Q shape =",
    Q.shape
)

print(
    "空间测点数 S =",
    S
)

print(
    "时间点数 T =",
    T
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
# 6. 指标函数
# ============================================================

def relative_reconstruction_error(
    original,
    reconstructed
):
    """
    相对 Frobenius 重构误差：

        ||Q - Qhat||_F
    E = ----------------
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


def build_mode_fields(
    phi,
    c
):
    """
    根据 ORVMD 输出的空间模态 phi 和时间系数 c
    构造各阶完整时空模态。

    phi shape:
        S x K

    c shape:
        T x K

    return:
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

    for k in range(K_local):

        fields[k] = np.outer(
            phi[:, k],
            c[:, k]
        )

    return fields


def coherence_matrix(
    fields
):
    """
    模态之间归一化内积：

                  |<Qi,Qj>|
    C_ij = -------------------------
           ||Qi||_F * ||Qj||_F

    非对角元素越小，模态之间越独立。
    """

    K_local = fields.shape[0]

    C = np.zeros(
        (
            K_local,
            K_local
        ),
        dtype=float
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
# 7. lambda 粗扫描
# ============================================================

records = []

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


    # --------------------------------------------------------
    # 提取 ORVMD 结果
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 构造 K 个时空模态
    # --------------------------------------------------------

    fields = build_mode_fields(
        phi,
        c
    )


    # --------------------------------------------------------
    # 重构
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 模态相干矩阵
    # --------------------------------------------------------

    C = coherence_matrix(
        fields
    )


    mean_coh = offdiag_mean(
        C
    )


    max_coh = offdiag_max(
        C
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


    diff_history = np.asarray(
        iteration_info[
            "difference"
        ]
    )


    final_difference = float(
        diff_history[-1]
    )


    # --------------------------------------------------------
    # 中心频率
    # --------------------------------------------------------

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


    coherence_dict[
        lam
    ] = C


# ============================================================
# 8. 保存结果表
# ============================================================

df = pd.DataFrame(
    records
)


csv_path = os.path.join(
    RESULT_DIR,
    "vertical_lambda_scan_summary.csv"
)


df.to_csv(
    csv_path,
    index=False
)


print("\n")
print("=" * 80)
print("垂向 lambda 粗扫描结果")
print("=" * 80)

print(
    df.to_string(
        index=False
    )
)


# ============================================================
# 9. 图1：重构误差
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
    "Vertical field: reconstruction error vs lambda_orth"
)

ax.grid(
    True,
    alpha=0.3
)


fig.tight_layout()


fig.savefig(
    os.path.join(
        RESULT_DIR,
        "vertical_lambda_reconstruction_error.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# 10. 图2：平均模态相干度
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
    "Vertical field: mean modal coherence"
)

ax.grid(
    True,
    alpha=0.3
)


fig.tight_layout()


fig.savefig(
    os.path.join(
        RESULT_DIR,
        "vertical_lambda_mean_coherence.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# 11. 图3：最大模态相干度
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
    "Vertical field: maximum modal coherence"
)

ax.grid(
    True,
    alpha=0.3
)


fig.tight_layout()


fig.savefig(
    os.path.join(
        RESULT_DIR,
        "vertical_lambda_max_coherence.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# 12. 图4：重构误差 - 相干度 trade-off
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
    "Vertical field: reconstruction-coherence trade-off"
)

ax.grid(
    True,
    alpha=0.3
)


fig.tight_layout()


fig.savefig(
    os.path.join(
        RESULT_DIR,
        "vertical_tradeoff_error_coherence.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# 13. 图5：收敛指标
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
    "Vertical field: convergence indicator"
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
        "vertical_lambda_convergence.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# 14. 保存各 lambda 的相干矩阵
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
            f"vertical_coherence_lambda_{safe_name}.csv"
        ),
        C,
        delimiter=","
    )


# ============================================================
# 15. 最终输出
# ============================================================

print("\n")
print("#" * 80)
print("垂向 lambda_orth 粗粒度扫描完成")
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
    csv_path
)

print(
    "\n主要图片："
)

print(
    "vertical_tradeoff_error_coherence.png"
)

print(
    "vertical_lambda_convergence.png"
)

print(
    "vertical_lambda_mean_coherence.png"
)

print(
    "vertical_lambda_max_coherence.png"
)

print(
    "vertical_lambda_reconstruction_error.png"
)

print("#" * 80)