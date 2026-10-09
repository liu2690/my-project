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
    "uvw_multiwindow"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# 2. 实验参数
# ============================================================

FS = 20.0

N_SAMPLES = 5000

WINDOW_DURATION = (
    N_SAMPLES / FS
)

K = 4

ALPHA = 1000.0

TOL = 0.005

MAX_STEPS = 500

RVMD_LAMBDA = 0.0

ORVMD_LAMBDA = 0.03


CHANNEL_NAMES = [
    "u_110_5m",
    "v_110_5m",
    "w_110_5m",
    "u_179_5m",
    "v_179_5m",
    "w_179_5m",
]


# ============================================================
# 3. 多时间窗
# ============================================================

WINDOW_START_TIMES = [

    "2023-07-24 18:53:20",

    "2023-07-24 20:17:30",

    "2023-07-24 21:14:10",

    "2023-07-24 21:24:10",

    "2023-07-24 22:32:30",

]


# ============================================================
# 4. 构造时空模态
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


    S, K_local = phi.shape

    T, K_c = c.shape


    if K_local != K_c:

        raise ValueError(
            "phi 和 c 的模态数不一致："
            f"{phi.shape} vs {c.shape}"
        )


    fields = np.zeros(
        (
            K_local,
            S,
            T
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


# ============================================================
# 5. 重构误差
# ============================================================

def normalized_reconstruction_error(
    Q,
    reconstruction
):

    numerator = np.linalg.norm(
        Q - reconstruction
    )

    denominator = np.linalg.norm(
        Q
    )


    if denominator == 0:

        return np.nan


    return float(
        numerator / denominator
    )


# ============================================================
# 6. 模态相干矩阵
# ============================================================

def coherence_matrix(
    mode_fields
):

    mode_fields = np.asarray(
        mode_fields,
        dtype=float
    )


    K_local = mode_fields.shape[0]


    coherence = np.zeros(
        (
            K_local,
            K_local
        ),
        dtype=float
    )


    flattened = [

        mode_fields[k].reshape(-1)

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

                coherence[i, j] = 0.0

            else:

                coherence[i, j] = (

                    np.abs(

                        np.vdot(
                            flattened[i],
                            flattened[j]
                        )

                    )

                    /
                    denominator

                )


    return coherence


# ============================================================
# 7. 非对角相干度
# ============================================================

def offdiag_statistics(
    coherence
):

    mask = ~np.eye(
        coherence.shape[0],
        dtype=bool
    )


    values = coherence[
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
# 8. 单次运行
# ============================================================

def run_model(
    Q,
    lambda_orth,
    method_name
):

    print()
    print("#" * 80)

    print(
        method_name
    )

    print(
        "lambda_orth =",
        lambda_orth
    )

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
    # 提取模态
    # --------------------------------------------------------

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


    print(
        "phi shape =",
        phi.shape
    )

    print(
        "c shape =",
        c.shape
    )


    # --------------------------------------------------------
    # 构造时空模态
    # --------------------------------------------------------

    fields = build_mode_fields(
        phi,
        c
    )


    print(
        "mode fields shape =",
        fields.shape
    )


    # --------------------------------------------------------
    # 重构
    # --------------------------------------------------------

    reconstruction = np.sum(
        fields,
        axis=0
    )


    rec_error = (
        normalized_reconstruction_error(
            Q,
            reconstruction
        )
    )


    # --------------------------------------------------------
    # 相干矩阵
    # --------------------------------------------------------

    coherence = coherence_matrix(
        fields
    )


    mean_coherence, max_coherence = (
        offdiag_statistics(
            coherence
        )
    )


    # ========================================================
    # 关键修正：
    # 真实收敛信息在 info["Iteration"] 中
    # ========================================================

    iteration_info = info[
        "Iteration"
    ]


    steps = int(
        iteration_info[
            "steps"
        ]
    )


    history = np.asarray(
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


    if history.size > 0:

        final_difference = float(
            history[-1]
        )

    else:

        final_difference = np.nan


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
        mean_coherence
    )

    print(
        "max coherence =",
        max_coherence
    )

    print(
        "frequency Hz =",
        frequency_hz
    )


    return {

        "phi":
            phi,

        "c":
            c,

        "fields":
            fields,

        "reconstruction":
            reconstruction,

        "reconstruction_error":
            rec_error,

        "coherence_matrix":
            coherence,

        "mean_coherence":
            mean_coherence,

        "max_coherence":
            max_coherence,

        "steps":
            steps,

        "final_difference":
            final_difference,

        "converged":
            converged,

        "frequency_hz":
            frequency_hz,

        "history":
            history,

    }


# ============================================================
# 9. 读取完整数据
# ============================================================

print()

print(
    "=" * 90
)

print(
    "真实 UVW 风场：RVMD vs ORVMD 多时间窗验证"
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


print(
    "完整数据 shape =",
    df.shape
)

print(
    "数据开始时间 =",
    df["time20hz"].iloc[0]
)

print(
    "数据结束时间 =",
    df["time20hz"].iloc[-1]
)


print()

print(
    "FS =",
    FS,
    "Hz"
)

print(
    "每窗口点数 =",
    N_SAMPLES
)

print(
    "每窗口持续时间 =",
    WINDOW_DURATION,
    "s"
)

print(
    "K =",
    K
)

print(
    "Alpha =",
    ALPHA
)

print(
    "Tolerance =",
    TOL
)

print(
    "MaximumSteps =",
    MAX_STEPS
)

print(
    "RVMD lambda =",
    RVMD_LAMBDA
)

print(
    "ORVMD lambda =",
    ORVMD_LAMBDA
)


# ============================================================
# 10. 循环所有窗口
# ============================================================

summary_rows = []

improvement_rows = []


time_values = df[
    "time20hz"
].values


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


    if end_index > len(df):

        raise ValueError(
            f"Window {window_index} 超出数据范围"
        )


    segment = df.iloc[
        start_index:
        end_index
    ].copy()


    actual_start = segment[
        "time20hz"
    ].iloc[0]


    actual_end = segment[
        "time20hz"
    ].iloc[-1]


    print(
        "start index =",
        start_index
    )

    print(
        "end index =",
        end_index
    )

    print(
        "start time =",
        actual_start
    )

    print(
        "end time =",
        actual_end
    )


    # --------------------------------------------------------
    # 6 × 5000
    # --------------------------------------------------------

    Q_raw = segment[
        CHANNEL_NAMES
    ].to_numpy(
        dtype=float
    ).T


    print(
        "Q_raw shape =",
        Q_raw.shape
    )


    # --------------------------------------------------------
    # 脉动风速
    # --------------------------------------------------------

    channel_mean = np.mean(
        Q_raw,
        axis=1,
        keepdims=True
    )


    Q = (
        Q_raw
        -
        channel_mean
    )


    print(
        "Q fluctuation shape =",
        Q.shape
    )

    print(
        "脉动风速各通道均值 ="
    )

    print(
        np.mean(
            Q,
            axis=1
        )
    )


    # --------------------------------------------------------
    # RVMD
    # --------------------------------------------------------

    rvmd_result = run_model(
        Q,
        RVMD_LAMBDA,
        "RVMD baseline"
    )


    # --------------------------------------------------------
    # ORVMD
    # --------------------------------------------------------

    orvmd_result = run_model(
        Q,
        ORVMD_LAMBDA,
        "ORVMD"
    )


    # --------------------------------------------------------
    # 改善率
    # --------------------------------------------------------

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


    print()

    print(
        "-" * 80
    )

    print(
        f"Window {window_index} 改善结果"
    )

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

    print(
        "-" * 80
    )


    # --------------------------------------------------------
    # 保存两种方法结果
    # --------------------------------------------------------

    for method_name, lam, result in [

        (
            "RVMD",
            RVMD_LAMBDA,
            rvmd_result
        ),

        (
            "ORVMD",
            ORVMD_LAMBDA,
            orvmd_result
        ),

    ]:

        freq = result[
            "frequency_hz"
        ]


        summary_rows.append({

            "window":
                window_index,

            "start_time":
                actual_start,

            "end_time":
                actual_end,

            "method":
                method_name,

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
                freq[0],

            "f2_Hz":
                freq[1],

            "f3_Hz":
                freq[2],

            "f4_Hz":
                freq[3],

        })


    improvement_rows.append({

        "window":
            window_index,

        "start_time":
            actual_start,

        "end_time":
            actual_end,

        "mean_coherence_improvement_percent":
            mean_improvement,

        "max_coherence_improvement_percent":
            max_improvement,

        "reconstruction_change_percent":
            reconstruction_change,

        "rvmd_converged":
            rvmd_result[
                "converged"
            ],

        "orvmd_converged":
            orvmd_result[
                "converged"
            ],

        "rvmd_steps":
            rvmd_result[
                "steps"
            ],

        "orvmd_steps":
            orvmd_result[
                "steps"
            ],

        "rvmd_final_difference":
            rvmd_result[
                "final_difference"
            ],

        "orvmd_final_difference":
            orvmd_result[
                "final_difference"
            ],

    })


# ============================================================
# 11. DataFrame
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)


improvement_df = pd.DataFrame(
    improvement_rows
)


summary_path = os.path.join(
    RESULT_DIR,
    "uvw_multiwindow_summary.csv"
)


improvement_path = os.path.join(
    RESULT_DIR,
    "uvw_multiwindow_improvement.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


improvement_df.to_csv(
    improvement_path,
    index=False
)


# ============================================================
# 12. 打印结果
# ============================================================

print()

print(
    "=" * 90
)

print(
    "全部时间窗结果"
)

print(
    "=" * 90
)


print(
    summary_df.to_string(
        index=False
    )
)


print()

print(
    "=" * 90
)

print(
    "各窗口 ORVMD 相对 RVMD 改善"
)

print(
    "=" * 90
)


print(
    improvement_df.to_string(
        index=False
    )
)


# ============================================================
# 13. 只统计“双方均收敛”的窗口
# ============================================================

valid_improvement_df = improvement_df[

    (
        improvement_df[
            "rvmd_converged"
        ]
        ==
        True
    )

    &

    (
        improvement_df[
            "orvmd_converged"
        ]
        ==
        True
    )

].copy()


print()

print(
    "=" * 90
)

print(
    "双方均收敛的有效窗口"
)

print(
    "=" * 90
)


if len(
    valid_improvement_df
) > 0:

    print(
        valid_improvement_df.to_string(
            index=False
        )
    )

else:

    print(
        "没有双方同时收敛的窗口。"
    )


# ============================================================
# 14. 总体统计
# ============================================================

if len(
    valid_improvement_df
) > 0:

    mean_imp_mean = (
        valid_improvement_df[
            "mean_coherence_improvement_percent"
        ].mean()
    )

    mean_imp_std = (
        valid_improvement_df[
            "mean_coherence_improvement_percent"
        ].std()
    )


    max_imp_mean = (
        valid_improvement_df[
            "max_coherence_improvement_percent"
        ].mean()
    )

    max_imp_std = (
        valid_improvement_df[
            "max_coherence_improvement_percent"
        ].std()
    )


    rec_change_mean = (
        valid_improvement_df[
            "reconstruction_change_percent"
        ].mean()
    )

    rec_change_std = (
        valid_improvement_df[
            "reconstruction_change_percent"
        ].std()
    )


else:

    mean_imp_mean = np.nan
    mean_imp_std = np.nan

    max_imp_mean = np.nan
    max_imp_std = np.nan

    rec_change_mean = np.nan
    rec_change_std = np.nan


overall_df = pd.DataFrame({

    "metric": [

        "mean_coherence_improvement_percent",

        "max_coherence_improvement_percent",

        "reconstruction_change_percent",

    ],

    "mean": [

        mean_imp_mean,

        max_imp_mean,

        rec_change_mean,

    ],

    "std": [

        mean_imp_std,

        max_imp_std,

        rec_change_std,

    ]

})


overall_path = os.path.join(
    RESULT_DIR,
    "uvw_multiwindow_overall_statistics.csv"
)


overall_df.to_csv(
    overall_path,
    index=False
)


# ============================================================
# 15. 绘图准备
# ============================================================

rvmd_df = (

    summary_df[
        summary_df[
            "method"
        ] == "RVMD"
    ]

    .sort_values(
        "window"
    )

)


orvmd_df = (

    summary_df[
        summary_df[
            "method"
        ] == "ORVMD"
    ]

    .sort_values(
        "window"
    )

)


windows = rvmd_df[
    "window"
].to_numpy()


x = np.arange(
    len(windows)
)


bar_width = 0.36


# ============================================================
# 16. 平均相干度图
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.bar(
    x - bar_width / 2,
    rvmd_df[
        "mean_offdiag_coherence"
    ],
    width=bar_width,
    label="RVMD"
)


plt.bar(
    x + bar_width / 2,
    orvmd_df[
        "mean_offdiag_coherence"
    ],
    width=bar_width,
    label="ORVMD"
)


plt.xticks(
    x,
    [
        f"Window {i}"
        for i in windows
    ]
)


plt.xlabel(
    "Time window"
)

plt.ylabel(
    "Mean off-diagonal coherence"
)

plt.title(
    "UVW field: mean coherence across time windows"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULT_DIR,
        "multiwindow_mean_coherence.png"
    ),
    dpi=300
)


plt.close()


# ============================================================
# 17. 最大相干度
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.bar(
    x - bar_width / 2,
    rvmd_df[
        "max_offdiag_coherence"
    ],
    width=bar_width,
    label="RVMD"
)


plt.bar(
    x + bar_width / 2,
    orvmd_df[
        "max_offdiag_coherence"
    ],
    width=bar_width,
    label="ORVMD"
)


plt.xticks(
    x,
    [
        f"Window {i}"
        for i in windows
    ]
)


plt.xlabel(
    "Time window"
)

plt.ylabel(
    "Maximum off-diagonal coherence"
)

plt.title(
    "UVW field: maximum coherence across time windows"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULT_DIR,
        "multiwindow_max_coherence.png"
    ),
    dpi=300
)


plt.close()


# ============================================================
# 18. 重构误差
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    windows,
    rvmd_df[
        "reconstruction_error"
    ],
    marker="o",
    label="RVMD"
)


plt.plot(
    windows,
    orvmd_df[
        "reconstruction_error"
    ],
    marker="s",
    label="ORVMD"
)


plt.xticks(
    windows
)

plt.xlabel(
    "Time window"
)

plt.ylabel(
    "Normalized reconstruction error"
)

plt.title(
    "UVW field: reconstruction error across windows"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULT_DIR,
        "multiwindow_reconstruction_error.png"
    ),
    dpi=300
)


plt.close()


# ============================================================
# 19. 改善率
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    improvement_df[
        "window"
    ],
    improvement_df[
        "mean_coherence_improvement_percent"
    ],
    marker="o",
    label="Mean coherence"
)


plt.plot(
    improvement_df[
        "window"
    ],
    improvement_df[
        "max_coherence_improvement_percent"
    ],
    marker="s",
    label="Max coherence"
)


plt.axhline(
    y=0,
    linestyle="--"
)


plt.xlabel(
    "Time window"
)

plt.ylabel(
    "Improvement relative to RVMD (%)"
)

plt.title(
    "UVW field: ORVMD coherence improvement"
)

plt.xticks(
    improvement_df[
        "window"
    ]
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULT_DIR,
        "multiwindow_coherence_improvement.png"
    ),
    dpi=300
)


plt.close()


# ============================================================
# 20. 最终输出
# ============================================================

print()

print(
    "#" * 90
)

print(
    "修正后的 UVW 多时间窗验证完成"
)

print(
    "#" * 90
)


print(
    "结果目录："
)

print(
    RESULT_DIR
)


print()

print(
    "有效窗口数量 =",
    len(
        valid_improvement_df
    ),
    "/",
    len(
        improvement_df
    )
)


print()

print(
    "双方均收敛窗口的平均相干度改善："
)

print(
    mean_imp_mean,
    "±",
    mean_imp_std,
    "%"
)


print()

print(
    "双方均收敛窗口的最大相干度改善："
)

print(
    max_imp_mean,
    "±",
    max_imp_std,
    "%"
)


print()

print(
    "双方均收敛窗口的重构误差变化："
)

print(
    rec_change_mean,
    "±",
    rec_change_std,
    "%"
)


print(
    "#" * 90
)