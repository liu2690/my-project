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

if ORVMD_DIR not in sys.path:
    sys.path.insert(
        0,
        ORVMD_DIR
    )

from orvmd import orvmd


DATA_PATH = os.path.join(
    ROOT,
    "3d_wind_analysis",
    "data",
    "UVW_unstable_benchmark.nc"
)

RESULT_DIR = os.path.join(
    THIS_DIR,
    "result",
    "uvw_window_lambda_scan"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# 2. 固定实验参数
# ============================================================

FS = 20.0

N_SAMPLES = 5000

K = 4

ALPHA = 1000.0

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
# 3. 五个时间窗
# ============================================================

WINDOW_START_TIMES = [

    "2023-07-24 18:53:20",

    "2023-07-24 20:17:30",

    "2023-07-24 21:14:10",

    "2023-07-24 21:24:10",

    "2023-07-24 22:32:30",

]


# ============================================================
# 4. lambda 二维扫描范围
# ============================================================

LAMBDA_LIST = [

    0.000,

    0.005,

    0.010,

    0.015,

    0.020,

    0.025,

    0.030,

    0.035,

    0.040,

    0.045,

    0.050,

]


# ============================================================
# 5. 构造时空模态
# ============================================================

def build_mode_fields(
    phi,
    c
):

    phi = np.asarray(
        phi,
        dtype=float
    )

    c = np.asarray(
        c,
        dtype=float
    )


    S, K_phi = phi.shape

    T, K_c = c.shape


    if K_phi != K_c:

        raise ValueError(
            f"phi/c 模态数不一致: "
            f"{phi.shape} vs {c.shape}"
        )


    fields = np.zeros(
        (
            K_phi,
            S,
            T
        ),
        dtype=float
    )


    for k in range(
        K_phi
    ):

        fields[k] = np.outer(
            phi[:, k],
            c[:, k]
        )


    return fields


# ============================================================
# 6. 重构误差
# ============================================================

def reconstruction_error(
    Q,
    reconstruction
):

    denominator = np.linalg.norm(
        Q
    )


    if denominator == 0:

        return np.nan


    return float(

        np.linalg.norm(
            Q - reconstruction
        )

        /

        denominator

    )


# ============================================================
# 7. 相干矩阵
# ============================================================

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


    flattened = [

        fields[k].reshape(-1)

        for k in range(
            K_local
        )

    ]


    norms = [

        np.linalg.norm(x)

        for x in flattened

    ]


    for i in range(
        K_local
    ):

        for j in range(
            K_local
        ):

            denominator = (
                norms[i]
                *
                norms[j]
            )


            if denominator <= 0:

                C[i, j] = 0.0

            else:

                C[i, j] = (

                    np.abs(

                        np.vdot(
                            flattened[i],
                            flattened[j]
                        )

                    )

                    /

                    denominator

                )


    return C


# ============================================================
# 8. 非对角相干统计
# ============================================================

def offdiag_statistics(
    C
):

    mask = ~np.eye(
        C.shape[0],
        dtype=bool
    )


    values = C[
        mask
    ]


    return (

        float(
            np.mean(values)
        ),

        float(
            np.max(values)
        )

    )


# ============================================================
# 9. 单次 ORVMD
# ============================================================

def run_model(
    Q,
    lambda_orth
):

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


    phi = np.asarray(
        mode["phi"],
        dtype=float
    )


    c = np.asarray(
        mode["c"],
        dtype=float
    )


    omega = np.asarray(
        mode["omega"],
        dtype=float
    ).reshape(-1)


    fields = build_mode_fields(
        phi,
        c
    )


    reconstruction = np.sum(
        fields,
        axis=0
    )


    rec_error = reconstruction_error(
        Q,
        reconstruction
    )


    C = coherence_matrix(
        fields
    )


    mean_coh, max_coh = (
        offdiag_statistics(
            C
        )
    )


    # --------------------------------------------------------
    # 正确读取收敛信息
    # --------------------------------------------------------

    iteration_info = info[
        "Iteration"
    ]


    steps = int(
        iteration_info[
            "steps"
        ]
    )


    difference_history = np.asarray(
        iteration_info[
            "difference"
        ],
        dtype=float
    ).reshape(-1)


    converged = bool(
        iteration_info[
            "converged"
        ]
    )


    if difference_history.size > 0:

        final_difference = float(
            difference_history[-1]
        )

    else:

        final_difference = np.nan


    frequency_hz = (
        omega
        *
        FS
    )


    return {

        "reconstruction_error":
            rec_error,

        "mean_coherence":
            mean_coh,

        "max_coherence":
            max_coh,

        "converged":
            converged,

        "steps":
            steps,

        "final_difference":
            final_difference,

        "frequency_hz":
            frequency_hz,

    }


# ============================================================
# 10. 读取完整数据
# ============================================================

print()

print(
    "=" * 90
)

print(
    "UVW Window × lambda_orth 二维扫描"
)

print(
    "=" * 90
)


df = pd.read_csv(
    DATA_PATH
)


df[
    "time20hz"
] = pd.to_datetime(
    df[
        "time20hz"
    ]
)


time_values = df[
    "time20hz"
].values


print(
    "完整数据 shape =",
    df.shape
)

print(
    "窗口数量 =",
    len(
        WINDOW_START_TIMES
    )
)

print(
    "lambda 数量 =",
    len(
        LAMBDA_LIST
    )
)

print(
    "总实验次数 =",
    len(
        WINDOW_START_TIMES
    )
    *
    len(
        LAMBDA_LIST
    )
)

print(
    "Lambda list =",
    LAMBDA_LIST
)


# ============================================================
# 11. 正式二维扫描
# ============================================================

rows = []


for window_index, start_text in enumerate(
    WINDOW_START_TIMES,
    start=1
):

    print()

    print(
        "=" * 90
    )

    print(
        f"Window {window_index}"
    )

    print(
        "=" * 90
    )


    start_time = pd.Timestamp(
        start_text
    )


    start_index = int(
        np.searchsorted(
            time_values,
            np.datetime64(
                start_time
            )
        )
    )


    end_index = (
        start_index
        +
        N_SAMPLES
    )


    segment = df.iloc[
        start_index:
        end_index
    ]


    actual_start = segment[
        "time20hz"
    ].iloc[0]


    actual_end = segment[
        "time20hz"
    ].iloc[-1]


    Q_raw = segment[
        CHANNEL_NAMES
    ].to_numpy(
        dtype=float
    ).T


    # --------------------------------------------------------
    # 脉动风速
    # --------------------------------------------------------

    Q = (

        Q_raw

        -

        np.mean(
            Q_raw,
            axis=1,
            keepdims=True
        )

    )


    print(
        "时间 =",
        actual_start,
        "->",
        actual_end
    )

    print(
        "Q shape =",
        Q.shape
    )


    # ========================================================
    # 先跑 lambda=0，作为该窗口 RVMD baseline
    # ========================================================

    baseline_result = run_model(
        Q,
        0.0
    )


    baseline_mean = (
        baseline_result[
            "mean_coherence"
        ]
    )


    baseline_max = (
        baseline_result[
            "max_coherence"
        ]
    )


    baseline_rec = (
        baseline_result[
            "reconstruction_error"
        ]
    )


    for lam in LAMBDA_LIST:

        print()

        print(
            f"Window {window_index}, "
            f"lambda = {lam}"
        )


        # 避免 lambda=0 再算一次
        if lam == 0.0:

            result = (
                baseline_result
            )

        else:

            result = run_model(
                Q,
                lam
            )


        mean_improvement = (

            (
                baseline_mean
                -
                result[
                    "mean_coherence"
                ]
            )

            /

            baseline_mean

            *
            100.0

        )


        max_improvement = (

            (
                baseline_max
                -
                result[
                    "max_coherence"
                ]
            )

            /

            baseline_max

            *
            100.0

        )


        reconstruction_change = (

            (
                result[
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


        frequency_hz = result[
            "frequency_hz"
        ]


        print(
            "converged =",
            result[
                "converged"
            ]
        )

        print(
            "steps =",
            result[
                "steps"
            ]
        )

        print(
            "rec error =",
            result[
                "reconstruction_error"
            ]
        )

        print(
            "mean coherence =",
            result[
                "mean_coherence"
            ]
        )

        print(
            "max coherence =",
            result[
                "max_coherence"
            ]
        )

        print(
            "mean improvement =",
            mean_improvement,
            "%"
        )

        print(
            "max improvement =",
            max_improvement,
            "%"
        )


        rows.append({

            "window":
                window_index,

            "start_time":
                actual_start,

            "end_time":
                actual_end,

            "lambda_orth":
                lam,

            "reconstruction_error":
                result[
                    "reconstruction_error"
                ],

            "mean_offdiag_coherence":
                result[
                    "mean_coherence"
                ],

            "max_offdiag_coherence":
                result[
                    "max_coherence"
                ],

            "mean_coherence_improvement_percent":
                mean_improvement,

            "max_coherence_improvement_percent":
                max_improvement,

            "reconstruction_change_percent":
                reconstruction_change,

            "converged":
                result[
                    "converged"
                ],

            "steps":
                result[
                    "steps"
                ],

            "final_difference":
                result[
                    "final_difference"
                ],

            "f1_Hz":
                frequency_hz[0],

            "f2_Hz":
                frequency_hz[1],

            "f3_Hz":
                frequency_hz[2],

            "f4_Hz":
                frequency_hz[3],

        })


# ============================================================
# 12. 保存完整扫描
# ============================================================

scan_df = pd.DataFrame(
    rows
)


SCAN_PATH = os.path.join(
    RESULT_DIR,
    "uvw_window_lambda_scan.csv"
)


scan_df.to_csv(
    SCAN_PATH,
    index=False
)


# ============================================================
# 13. 自动筛选每个窗口候选 lambda
#
# 条件：
# 1. lambda > 0
# 2. 必须收敛
# 3. 重构误差相对变化 < 1%
# 4. 平均相干度必须优于 RVMD
# 5. 满足以上条件后，取平均相干度最低
# ============================================================

best_rows = []


for window_index in sorted(
    scan_df[
        "window"
    ].unique()
):

    one = scan_df[
        scan_df[
            "window"
        ]
        ==
        window_index
    ].copy()


    baseline = one[
        one[
            "lambda_orth"
        ]
        ==
        0.0
    ].iloc[0]


    candidates = one[

        (
            one[
                "lambda_orth"
            ]
            >
            0
        )

        &

        (
            one[
                "converged"
            ]
            ==
            True
        )

        &

        (
            np.abs(
                one[
                    "reconstruction_change_percent"
                ]
            )
            <
            1.0
        )

        &

        (
            one[
                "mean_offdiag_coherence"
            ]
            <
            baseline[
                "mean_offdiag_coherence"
            ]
        )

    ].copy()


    if len(
        candidates
    ) > 0:

        best = candidates.sort_values(

            [
                "mean_offdiag_coherence",
                "max_offdiag_coherence"
            ]

        ).iloc[0]


        status = (
            "found"
        )


    else:

        best = baseline

        status = (
            "no_positive_lambda_improved_mean_coherence"
        )


    best_rows.append({

        "window":
            window_index,

        "status":
            status,

        "best_lambda":
            float(
                best[
                    "lambda_orth"
                ]
            ),

        "mean_offdiag_coherence":
            float(
                best[
                    "mean_offdiag_coherence"
                ]
            ),

        "max_offdiag_coherence":
            float(
                best[
                    "max_offdiag_coherence"
                ]
            ),

        "mean_improvement_percent":
            float(
                best[
                    "mean_coherence_improvement_percent"
                ]
            ),

        "max_improvement_percent":
            float(
                best[
                    "max_coherence_improvement_percent"
                ]
            ),

        "reconstruction_change_percent":
            float(
                best[
                    "reconstruction_change_percent"
                ]
            ),

        "converged":
            bool(
                best[
                    "converged"
                ]
            ),

    })


best_df = pd.DataFrame(
    best_rows
)


BEST_PATH = os.path.join(
    RESULT_DIR,
    "uvw_window_best_lambda.csv"
)


best_df.to_csv(
    BEST_PATH,
    index=False
)


print()

print(
    "=" * 90
)

print(
    "各窗口候选 lambda"
)

print(
    "=" * 90
)


print(
    best_df.to_string(
        index=False
    )
)


# ============================================================
# 14. 构造热力图矩阵
# ============================================================

window_ids = sorted(
    scan_df[
        "window"
    ].unique()
)


lambda_values = np.array(
    LAMBDA_LIST,
    dtype=float
)


def make_matrix(
    column
):

    matrix = np.full(
        (
            len(
                window_ids
            ),
            len(
                lambda_values
            )
        ),
        np.nan
    )


    for i, window_id in enumerate(
        window_ids
    ):

        for j, lam in enumerate(
            lambda_values
        ):

            row = scan_df[

                (
                    scan_df[
                        "window"
                    ]
                    ==
                    window_id
                )

                &

                (
                    np.isclose(
                        scan_df[
                            "lambda_orth"
                        ],
                        lam
                    )
                )

            ]


            if len(
                row
            ) == 1:

                matrix[
                    i,
                    j
                ] = row.iloc[0][
                    column
                ]


    return matrix


mean_matrix = make_matrix(
    "mean_offdiag_coherence"
)


max_matrix = make_matrix(
    "max_offdiag_coherence"
)


improvement_matrix = make_matrix(
    "mean_coherence_improvement_percent"
)


reconstruction_matrix = make_matrix(
    "reconstruction_change_percent"
)


# ============================================================
# 15. 通用 heatmap
# ============================================================

def plot_heatmap(
    matrix,
    title,
    colorbar_label,
    filename
):

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )


    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper"
    )


    ax.set_xticks(
        np.arange(
            len(
                lambda_values
            )
        )
    )


    ax.set_xticklabels(
        [
            f"{x:g}"
            for x in lambda_values
        ],
        rotation=45
    )


    ax.set_yticks(
        np.arange(
            len(
                window_ids
            )
        )
    )


    ax.set_yticklabels(
        [
            f"Window {x}"
            for x in window_ids
        ]
    )


    ax.set_xlabel(
        "lambda_orth"
    )

    ax.set_ylabel(
        "Time window"
    )

    ax.set_title(
        title
    )


    fig.colorbar(
        im,
        ax=ax,
        label=colorbar_label
    )


    fig.tight_layout()


    fig.savefig(
        os.path.join(
            RESULT_DIR,
            filename
        ),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


plot_heatmap(

    mean_matrix,

    "Mean modal coherence: Window x lambda",

    "Mean off-diagonal coherence",

    "window_lambda_mean_coherence_heatmap.png"

)


plot_heatmap(

    max_matrix,

    "Maximum modal coherence: Window x lambda",

    "Maximum off-diagonal coherence",

    "window_lambda_max_coherence_heatmap.png"

)


plot_heatmap(

    improvement_matrix,

    "Mean coherence improvement: Window x lambda",

    "Improvement relative to RVMD (%)",

    "window_lambda_improvement_heatmap.png"

)


plot_heatmap(

    reconstruction_matrix,

    "Reconstruction change: Window x lambda",

    "Reconstruction error change (%)",

    "window_lambda_reconstruction_heatmap.png"

)


# ============================================================
# 16. 每窗口最佳 lambda
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.bar(
    best_df[
        "window"
    ].astype(str),
    best_df[
        "best_lambda"
    ]
)


ax.set_xlabel(
    "Time window"
)

ax.set_ylabel(
    "Selected lambda_orth"
)

ax.set_title(
    "Selected lambda across UVW time windows"
)

ax.grid(
    axis="y",
    alpha=0.3
)


fig.tight_layout()


fig.savefig(
    os.path.join(
        RESULT_DIR,
        "window_best_lambda.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 17. 最终输出
# ============================================================

print()

print(
    "#" * 90
)

print(
    "UVW Window × lambda 扫描完成"
)

print(
    "#" * 90
)


print(
    "完整结果："
)

print(
    SCAN_PATH
)


print()

print(
    "每窗口候选 lambda："
)

print(
    BEST_PATH
)


print()

print(
    "主要图片："
)

print(
    "window_lambda_mean_coherence_heatmap.png"
)

print(
    "window_lambda_max_coherence_heatmap.png"
)

print(
    "window_lambda_improvement_heatmap.png"
)

print(
    "window_lambda_reconstruction_heatmap.png"
)

print(
    "window_best_lambda.png"
)

print(
    "#" * 90
)