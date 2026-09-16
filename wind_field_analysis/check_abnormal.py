import os
import numpy as np
import matplotlib.pyplot as plt


ROOT = "/home/liuxy/0418"
FS = 50

# ------------------------------------------------------------
# 需要重点检查的文件
# ------------------------------------------------------------

files_to_check = [
    # 展向异常区与正常区
    "/home/liuxy/0418/展向/20240418_sp-30m_fux5m15-16.dat",
    "/home/liuxy/0418/展向/20240418_sp-25m_fux5m15-16.dat",
    "/home/liuxy/0418/展向/20240418_sp-20m_fux5m15-16.dat",
    "/home/liuxy/0418/展向/20240418_sp-15m_fux5m15-16.dat",
    "/home/liuxy/0418/展向/20240418_sp-10m_fux5m15-16.dat",

    # 垂向突变附近
    "/home/liuxy/0418/垂向/20240418_fux7.15m_15-16.dat",
    "/home/liuxy/0418/垂向/20240418_fux10.24m_15-16.dat",
    "/home/liuxy/0418/垂向/20240418_fux14.65m_15-16.dat",
]


data_list = []


print("=" * 80)
print("异常测点详细检查")
print("=" * 80)


for path in files_to_check:

    if not os.path.isfile(path):
        print("\n文件不存在：", path)
        continue

    signal = np.loadtxt(path)

    print("\n文件：")
    print(os.path.basename(path))

    print(
        "shape =",
        signal.shape
    )

    print(
        "mean =",
        np.mean(signal)
    )

    print(
        "std =",
        np.std(signal)
    )

    print(
        "min =",
        np.min(signal)
    )

    print(
        "max =",
        np.max(signal)
    )

    print(
        "前10个值 =",
        signal[:10]
    )

    data_list.append(
        (
            os.path.basename(path),
            signal
        )
    )


# ------------------------------------------------------------
# 只看前 60 秒
# ------------------------------------------------------------

n_plot = FS * 60

time = np.arange(
    n_plot
) / FS


plt.figure(
    figsize=(14, 8)
)


for name, signal in data_list:

    plt.plot(
        time,
        signal[:n_plot],
        linewidth=0.8,
        label=name
    )


plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Velocity (m/s)"
)

plt.title(
    "Comparison of suspicious measurement points"
)

plt.legend(
    fontsize=7
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

save_path = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "selected_data",
    "abnormal_points_comparison.png"
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\n图已保存：")
print(save_path)