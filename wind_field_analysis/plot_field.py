import os
import numpy as np
import matplotlib.pyplot as plt


def plot_field(
    data,
    positions,
    ylabel,
    title,
    save,
    fs=None,
    plot_step=10
):
    """
    绘制三类图：

    1. 原始风速场
    2. 脉动风速场
    3. 平均风速空间分布

    参数
    ----------
    data : np.ndarray
        二维数组，shape = (空间测点数, 时间采样点数)

    positions : np.ndarray
        空间测点真实位置，例如：
        展向 [-30, -25, ..., 30]
        垂向 [0.03, 0.05, ..., 30]

    ylabel : str
        y轴名称

    title : str
        图标题

    save : str
        原始风速场保存路径，例如：
        selected_data/spanwise_selected.png

    fs : float or None
        采样频率。
        如果 fs=50，横轴显示 Time (s)
        如果 fs=None，横轴显示 Time sample

    plot_step : int
        仅用于绘图的时间降采样间隔。
        plot_step=10 表示每10个采样点画1个。
        不会修改原始 data。
    """

    # ========================================================
    # 1. 数据检查
    # ========================================================

    data = np.asarray(
        data,
        dtype=float
    )

    positions = np.asarray(
        positions,
        dtype=float
    )

    if data.ndim != 2:
        raise ValueError(
            f"data 必须是二维数组，"
            f"当前 shape = {data.shape}"
        )

    if len(positions) != data.shape[0]:
        raise ValueError(
            "positions 数量与空间测点数量不一致："
            f"{len(positions)} != {data.shape[0]}"
        )

    if plot_step < 1:
        raise ValueError(
            "plot_step 必须 >= 1"
        )

    # ========================================================
    # 2. 只对画图数据进行时间降采样
    # ========================================================

    data_plot = data[
        :,
        ::plot_step
    ]

    # ========================================================
    # 3. 构建时间轴
    # ========================================================

    if fs is None:

        x = np.arange(
            data_plot.shape[1]
        ) * plot_step

        xlabel = "Time sample"

    else:

        x = (
            np.arange(
                data_plot.shape[1]
            )
            * plot_step
            / fs
        )

        xlabel = "Time (s)"

    # ========================================================
    # 4. 生成输出文件名
    # ========================================================

    save_dir = os.path.dirname(
        os.path.abspath(save)
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    base_name = os.path.splitext(
        os.path.basename(save)
    )[0]

    raw_path = os.path.join(
        save_dir,
        base_name + "_raw.png"
    )

    fluctuation_path = os.path.join(
        save_dir,
        base_name + "_fluctuation.png"
    )

    mean_path = os.path.join(
        save_dir,
        base_name + "_mean_profile.png"
    )

    # ========================================================
    # 5. 原始风速场
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    mesh = ax.pcolormesh(
        x,
        positions,
        data_plot,
        shading="auto"
    )

    cbar = fig.colorbar(
        mesh,
        ax=ax
    )

    cbar.set_label(
        "Velocity (m/s)"
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title + " - Raw velocity"
    )

    fig.tight_layout()

    fig.savefig(
        raw_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close(fig)

    # ========================================================
    # 6. 计算脉动风速
    #
    # u'(x,t) = u(x,t) - mean(u(x,t))
    #
    # 每个测点分别减去自己的时间平均值
    # ========================================================

    mean_velocity = np.mean(
        data,
        axis=1,
        keepdims=True
    )

    fluctuation = (
        data
        - mean_velocity
    )

    fluctuation_plot = fluctuation[
        :,
        ::plot_step
    ]

    # ========================================================
    # 7. 脉动风速场
    #
    # 为了正确表现正负脉动，
    # colorbar 以0为中心，正负范围对称
    # ========================================================

    fluctuation_limit = np.nanpercentile(
        np.abs(fluctuation_plot),
        99
    )

    if fluctuation_limit == 0:
        fluctuation_limit = 1.0

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    mesh = ax.pcolormesh(
        x,
        positions,
        fluctuation_plot,
        shading="auto",
        cmap="RdBu_r",
        vmin=-fluctuation_limit,
        vmax=fluctuation_limit
    )

    cbar = fig.colorbar(
        mesh,
        ax=ax
    )

    cbar.set_label(
        "Velocity fluctuation (m/s)"
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title + " - Velocity fluctuation"
    )

    fig.tight_layout()

    fig.savefig(
        fluctuation_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close(fig)

    # ========================================================
    # 8. 平均风速空间分布
    # ========================================================

    mean_profile = np.mean(
        data,
        axis=1
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.plot(
        mean_profile,
        positions,
        marker="o",
        linewidth=1.5
    )

    ax.set_xlabel(
        "Mean velocity (m/s)"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title + " - Mean velocity profile"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        mean_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close(fig)

    # ========================================================
    # 9. 打印结果
    # ========================================================

    print("\n" + "=" * 70)
    print("绘图完成")
    print("=" * 70)

    print(
        "原始数据 shape：",
        data.shape
    )

    print(
        "绘图降采样后 shape：",
        data_plot.shape
    )

    print(
        "plot_step：",
        plot_step
    )

    print(
        "\n保存文件："
    )

    print(
        raw_path
    )

    print(
        fluctuation_path
    )

    print(
        mean_path
    )

    print("=" * 70)