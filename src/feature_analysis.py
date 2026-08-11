"""
VOC 特征选择结果分析

输入:
    selection_masks.npy
    VOC feature names

输出:
    feature_selection_stats.csv
    selected_voc_panel.csv
    feature_importance.png
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def analyze_feature_selection(
        mask_file,
        feature_names,
        save_dir,
        threshold=0.5,
        top_n=30
):
    """
    分析多次重复实验中的 VOC 特征选择稳定性。

    Parameters
    ----------
    mask_file:
        selection_masks.npy 路径

    feature_names:
        与 mask 列一一对应的 VOC 名称

    save_dir:
        结果保存目录

    threshold:
        最终 panel 的选择频率阈值

    top_n:
        特征重要性图显示前多少个 VOC
    """

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    # -------------------------------------------------
    # 读取 mask
    # shape = [repeats, features]
    # -------------------------------------------------

    masks = np.load(
        mask_file
    )

    if masks.ndim != 2:
        raise ValueError(
            f"selection_masks 应为二维数组，"
            f"当前 shape={masks.shape}"
        )

    if masks.shape[1] != len(feature_names):
        raise ValueError(
            "mask 特征数量与 VOC 名称数量不一致："
            f"{masks.shape[1]} vs {len(feature_names)}"
        )

    print(
        "selection masks shape:",
        masks.shape
    )

    # -------------------------------------------------
    # 每个 VOC 的选择频率
    # -------------------------------------------------

    selection_freq = masks.mean(
        axis=0
    )

    stats_df = pd.DataFrame(
        {
            "VOC_Index":
                np.arange(
                    len(selection_freq)
                ),

            "VOC_Name":
                feature_names,

            "Selection_Frequency":
                selection_freq,

            "Selected_Count":
                masks.sum(axis=0)
        }
    )

    stats_df = (
        stats_df
        .sort_values(
            "Selection_Frequency",
            ascending=False
        )
        .reset_index(drop=True)
    )

    stats_path = os.path.join(
        save_dir,
        "feature_selection_stats.csv"
    )

    stats_df.to_csv(
        stats_path,
        index=False
    )

    print(
        "saved:",
        stats_path
    )

    # -------------------------------------------------
    # 稳定 VOC panel
    # -------------------------------------------------

    panel_df = (
        stats_df[
            stats_df["Selection_Frequency"]
            > threshold
        ]
        .copy()
        .reset_index(drop=True)
    )

    panel_path = os.path.join(
        save_dir,
        "selected_voc_panel.csv"
    )

    panel_df.to_csv(
        panel_path,
        index=False
    )

    print(
        f"selected VOCs "
        f"(frequency > {threshold}): "
        f"{len(panel_df)}"
    )

    print(
        "saved:",
        panel_path
    )

    # -------------------------------------------------
    # Top VOC 图
    # -------------------------------------------------

    plot_df = stats_df.head(
        min(
            top_n,
            len(stats_df)
        )
    )

    plt.figure(
        figsize=(14, 7)
    )

    plt.bar(
        np.arange(
            len(plot_df)
        ),
        plot_df[
            "Selection_Frequency"
        ].values
    )

    plt.axhline(
        y=threshold,
        linestyle="--",
        label=f"Threshold = {threshold}"
    )

    plt.xticks(
        np.arange(
            len(plot_df)
        ),
        plot_df[
            "VOC_Name"
        ].values,
        rotation=90,
        fontsize=8
    )

    plt.ylabel(
        "Selection Frequency"
    )

    plt.xlabel(
        "VOC"
    )

    plt.title(
        f"Top {len(plot_df)} VOC Selection Frequency "
        f"(n={masks.shape[0]} repeats)"
    )

    plt.ylim(
        0,
        1.05
    )

    plt.legend()

    plt.tight_layout()

    figure_path = os.path.join(
        save_dir,
        "feature_importance.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "saved:",
        figure_path
    )

    return stats_df, panel_df