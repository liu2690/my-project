import numpy as np
import matplotlib.pyplot as plt
import os


np.random.seed(42)


# =========================
# 参数
# =========================

S = 10        # 空间测点数量
T = 4000      # 时间长度

fs = 20

t = np.arange(T)/fs



# =========================
# 三个基础风速成分
# =========================


# 平均风速变化
f1 = (
    np.sin(
        2*np.pi*0.05*t
    )
)


# 阵风
f2 = (
    np.sin(
        2*np.pi*0.5*t
    )
)


# 高频湍流
f3 = (
    np.sin(
        2*np.pi*3*t
    )
)



# =========================
# 构造空间-时间风场
# =========================


Q = np.zeros(
    (S,T)
)


for i in range(S):

    # 不同高度权重

    A = 5 + 0.2*i

    B = 1 + 0.1*i

    C = 0.3 + 0.02*i


    noise = (
        0.1
        *
        np.random.randn(T)
    )


    Q[i] = (
        A
        +
        A*0.2*f1
        +
        B*f2
        +
        C*f3
        +
        noise
    )



# =========================
# 保存
# =========================


os.makedirs(
    "data",
    exist_ok=True
)


np.save(
    "data/wind_field.npy",
    Q
)


print(
    "wind field shape:",
    Q.shape
)



# =========================
# 绘制部分测点
# =========================


plt.figure(
    figsize=(12,5)
)


for i in range(3):

    plt.plot(
        t,
        Q[i],
        label=f"point {i+1}"
    )


plt.xlabel(
    "Time(s)"
)

plt.ylabel(
    "Wind speed"
)


plt.legend()


plt.title(
    "Synthetic wind field"
)


plt.tight_layout()


plt.savefig(
    "data/wind_field.png",
    dpi=300
)


plt.show()
