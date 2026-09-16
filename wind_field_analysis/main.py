import os
import numpy as np

from read_wind import read_files
from plot_field import plot_field


# ============================================================
# 1. 基本配置
# ============================================================

ROOT = "/home/liuxy/0418"

# 展向选择的完整时段
SPANWISE_TIME = "15-16"

# 垂向时段
VERTICAL_TIME = "15-16"

# 截取的采样点范围
START_IDX = 0
END_IDX = 20000

# 如果知道真实采样频率，例如 50 Hz，可以改成：
# FS = 50
#
# 如果暂时不知道采样频率，就保持 None
FS = 50

# 输出目录
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "selected_data"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. 检查是否存在完全重复测点
# ============================================================

def check_duplicates(U, files, name):

    print("\n" + "=" * 70)
    print(name, "重复数据检查")
    print("=" * 70)

    found = False

    for i in range(len(files)):

        for j in range(i + 1, len(files)):

            if np.array_equal(
                U[i],
                U[j]
            ):

                print(
                    "完全相同：",
                    os.path.basename(files[i]),
                    "<=>",
                    os.path.basename(files[j])
                )

                found = True

    if not found:
        print("没有发现完全重复的数据")


# ============================================================
# 3. 打印当前读取文件
# ============================================================

def print_files(files, positions, name):

    print("\n" + "=" * 70)
    print(name, "文件列表")
    print("=" * 70)

    for i, (file, position) in enumerate(
        zip(files, positions)
    ):

        print(
            f"{i:02d}  "
            f"position = {position:8.3f} m  "
            f"{os.path.basename(file)}"
        )


# ============================================================
# 4. 检查数据有效性
# ============================================================

def check_data(U, name):

    print("\n" + "=" * 70)
    print(name, "数据检查")
    print("=" * 70)

    print(
        "shape =",
        U.shape
    )

    print(
        "NaN 数量 =",
        np.isnan(U).sum()
    )

    print(
        "Inf 数量 =",
        np.isinf(U).sum()
    )

    print(
        "min =",
        np.nanmin(U)
    )

    print(
        "max =",
        np.nanmax(U)
    )

    print(
        "mean =",
        np.nanmean(U)
    )


# ============================================================
# 5. 截取固定时间窗口
# ============================================================

def crop_data(
    U,
    start_idx,
    end_idx,
    name
):

    if U.ndim != 2:

        raise ValueError(
            f"{name} 数据必须是二维数组，"
            f"当前 shape = {U.shape}"
        )

    n_samples = U.shape[1]

    if start_idx < 0:

        raise ValueError(
            "START_IDX 不能小于 0"
        )

    if start_idx >= n_samples:

        raise ValueError(
            f"{name} START_IDX={start_idx} "
            f"已经超过总采样点数 {n_samples}"
        )

    real_end_idx = min(
        end_idx,
        n_samples
    )

    U_part = U[
        :,
        start_idx:real_end_idx
    ]

    print("\n" + "=" * 70)
    print(name, "数据截取")
    print("=" * 70)

    print(
        "原始 shape =",
        U.shape
    )

    print(
        "截取范围 =",
        start_idx,
        ":",
        real_end_idx
    )

    print(
        "截取后 shape =",
        U_part.shape
    )

    if FS is not None:

        duration_seconds = (
            U_part.shape[1] / FS
        )

        print(
            f"持续时间 = "
            f"{duration_seconds:.2f} s"
        )

        print(
            f"持续时间 = "
            f"{duration_seconds / 60:.2f} min"
        )

    return U_part


# ============================================================
# 6. 展向数据
# ============================================================

print("\n")
print("#" * 70)
print("开始处理展向数据")
print("#" * 70)


spanwise_folder = os.path.join(
    ROOT,
    "展向"
)


Uy, files_y, y_positions = read_files(
    spanwise_folder,
    include=[
        "sp",
        "fux5m",
        SPANWISE_TIME
    ],
    exclude=[
        "fuy",
        "fuz",
        "fte"
    ]
)


print_files(
    files_y,
    y_positions,
    "展向"
)


check_data(
    Uy,
    "展向"
)


check_duplicates(
    Uy,
    files_y,
    "展向"
)


# ============================================================
# 7. 检查展向测点是否完整
# ============================================================

EXPECTED_Y = np.array([
    -30.0,
    -25.0,
    -20.0,
    -15.0,
    -10.0,
    -5.0,
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0
])


print("\n" + "=" * 70)
print("展向测点完整性检查")
print("=" * 70)

print(
    "实际测点：",
    y_positions
)

print(
    "预期测点：",
    EXPECTED_Y
)


if np.array_equal(
    y_positions,
    EXPECTED_Y
):

    print(
        "展向测点完整：12 个测点全部存在。"
    )

else:

    missing_positions = np.setdiff1d(
        EXPECTED_Y,
        y_positions
    )

    extra_positions = np.setdiff1d(
        y_positions,
        EXPECTED_Y
    )

    print(
        "警告：展向测点不完整！"
    )

    print(
        "缺失测点：",
        missing_positions
    )

    if len(extra_positions) > 0:

        print(
            "额外测点：",
            extra_positions
        )

    raise RuntimeError(
        "当前展向时段存在缺失测点，"
        "不建议继续用于正式 OVMD / ORVMD 实验。"
    )


# ============================================================
# 8. 截取展向数据
# ============================================================

Uy_selected = crop_data(
    Uy,
    START_IDX,
    END_IDX,
    "展向"
)


# ============================================================
# 9. 保存展向数据
# ============================================================

spanwise_data_path = os.path.join(
    OUTPUT_DIR,
    "spanwise_selected.npy"
)

spanwise_position_path = os.path.join(
    OUTPUT_DIR,
    "spanwise_positions.npy"
)


np.save(
    spanwise_data_path,
    Uy_selected
)

np.save(
    spanwise_position_path,
    y_positions
)

# ============================================================
# 计算并保存展向脉动风速
# u'(y,t) = u(y,t) - mean_t[u(y,t)]
# ============================================================

Uy_fluctuation = (
    Uy_selected
    - np.mean(
        Uy_selected,
        axis=1,
        keepdims=True
    )
)

spanwise_fluctuation_path = os.path.join(
    OUTPUT_DIR,
    "spanwise_fluctuation.npy"
)

np.save(
    spanwise_fluctuation_path,
    Uy_fluctuation
)

print(
    "展向脉动风速已保存：",
    spanwise_fluctuation_path
)


print("\n展向数据已保存：")

print(
    spanwise_data_path
)

print(
    spanwise_position_path
)


# ============================================================
# 10. 绘制展向风场
# ============================================================

spanwise_figure_path = os.path.join(
    OUTPUT_DIR,
    "spanwise_selected.png"
)


plot_field(
    Uy_selected,
    y_positions,
    "Spanwise position y (m)",
    (
        f"Spanwise velocity field z=5m "
        f"({SPANWISE_TIME})"
    ),
    spanwise_figure_path,
    fs=FS,
    plot_step=10
)


# ============================================================
# 11. 垂向数据
# ============================================================

print("\n")
print("#" * 70)
print("开始处理垂向数据")
print("#" * 70)


vertical_folder = os.path.join(
    ROOT,
    "垂向"
)


Uz, files_z, z_positions = read_files(
    vertical_folder,
    include=[
        "fux",
        VERTICAL_TIME
    ],
    exclude=[
        "fuy",
        "fuz",
        "fte"
    ]
)


print_files(
    files_z,
    z_positions,
    "垂向"
)


check_data(
    Uz,
    "垂向"
)


check_duplicates(
    Uz,
    files_z,
    "垂向"
)


# ============================================================
# 12. 检查垂向测点
# ============================================================

EXPECTED_Z = np.array([
    0.03,
    0.05,
    0.10,
    0.16,
    0.28,
    0.50,
    0.90,
    1.71,
    2.50,
    3.49,
    5.00,
    7.15,
    10.24,
    14.65,
    20.96,
    30.00
])


print("\n" + "=" * 70)
print("垂向测点检查")
print("=" * 70)

print(
    "实际测点：",
    z_positions
)

print(
    "预期测点：",
    EXPECTED_Z
)


if np.allclose(
    z_positions,
    EXPECTED_Z,
    rtol=0,
    atol=1e-6
):

    print(
        "垂向测点完整：16 个测点全部存在。"
    )

else:

    print(
        "警告：垂向测点与预期不完全一致。"
    )

    print(
        "请根据上面的文件列表确认是否存在缺失。"
    )


# ============================================================
# 13. 截取垂向数据
# ============================================================

Uz_selected = crop_data(
    Uz,
    START_IDX,
    END_IDX,
    "垂向"
)


# ============================================================
# 14. 保存垂向数据
# ============================================================

vertical_data_path = os.path.join(
    OUTPUT_DIR,
    "vertical_selected.npy"
)

vertical_position_path = os.path.join(
    OUTPUT_DIR,
    "vertical_positions.npy"
)


np.save(
    vertical_data_path,
    Uz_selected
)
# ============================================================
# 计算并保存垂向脉动风速
# u'(z,t) = u(z,t) - mean_t[u(z,t)]
# ============================================================

Uz_fluctuation = (
    Uz_selected
    - np.mean(
        Uz_selected,
        axis=1,
        keepdims=True
    )
)

vertical_fluctuation_path = os.path.join(
    OUTPUT_DIR,
    "vertical_fluctuation.npy"
)

np.save(
    vertical_fluctuation_path,
    Uz_fluctuation
)

print(
    "垂向脉动风速已保存：",
    vertical_fluctuation_path
)

np.save(
    vertical_position_path,
    z_positions
)


print("\n垂向数据已保存：")

print(
    vertical_data_path
)

print(
    vertical_position_path
)


# ============================================================
# 15. 绘制垂向风场
# ============================================================

vertical_figure_path = os.path.join(
    OUTPUT_DIR,
    "vertical_selected.png"
)


plot_field(
    Uz_selected,
    z_positions,
    "Height z (m)",
    (
        f"Vertical velocity field "
        f"({VERTICAL_TIME})"
    ),
    vertical_figure_path,
    fs=FS,
    plot_step=10
)


# ============================================================
# 16. 最终汇总
# ============================================================

print("\n")
print("=" * 70)
print("处理完成")
print("=" * 70)

print(
    "展向最终数据 shape：",
    Uy_selected.shape
)

print(
    "展向位置：",
    y_positions
)

print(
    "\n垂向最终数据 shape：",
    Uz_selected.shape
)

print(
    "垂向位置：",
    z_positions
)

print(
    "\n结果保存目录：",
    OUTPUT_DIR
)

print("=" * 70)