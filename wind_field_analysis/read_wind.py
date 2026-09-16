import numpy as np
import os
import re


def read_files(folder, include=[], exclude=[]):

    files = []

    for f in os.listdir(folder):

        if not f.endswith(".dat"):
            continue

        if ":Zone.Identifier" in f:
            continue

        flag = True

        for key in include:
            if key not in f:
                flag = False

        for key in exclude:
            if key in f:
                flag = False

        if flag:
            files.append(f)

    # ==========================================
    # 提取真正的空间坐标
    # ==========================================
    def extract_position(filename):

        # 流向
        if "streamwise" in filename:
            match = re.search(
                r"streamwise(-?\d+(?:\.\d+)?)m",
                filename
            )

        # 展向
        elif "_sp" in filename:
            match = re.search(
                r"_sp(-?\d+(?:\.\d+)?)m",
                filename
            )

        # 垂向
        elif "_fux" in filename:
            match = re.search(
                r"_fux(-?\d+(?:\.\d+)?)m",
                filename
            )

        else:
            match = None

        if match is None:
            raise ValueError(
                f"无法从文件名提取空间位置: {filename}"
            )

        return float(match.group(1))

    # 按空间位置排序
    files.sort(key=extract_position)

    print("\n匹配数量:", len(files))
    print("排序后的文件：")

    data = []
    positions = []

    for f in files:

        position = extract_position(f)

        print(
            f"\n位置 = {position} m",
            f"文件 = {f}"
        )

        signal = np.loadtxt(
            os.path.join(folder, f)
        )

        print(
            "shape =", signal.shape,
            "NaN =", np.isnan(signal).sum(),
            "Inf =", np.isinf(signal).sum(),
            "min =", np.nanmin(signal),
            "max =", np.nanmax(signal),
            "mean =", np.nanmean(signal)
        )

        data.append(signal)
        positions.append(position)

    return (
        np.array(data),
        files,
        np.array(positions)
    )