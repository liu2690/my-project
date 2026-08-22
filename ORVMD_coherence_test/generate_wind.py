import numpy as np
import matplotlib.pyplot as plt
import os


np.random.seed(42)


# =====================
# 参数
# =====================

fs = 20

T = 200

N = fs*T


t = np.arange(N)/fs



# =====================
# 风速组成
# =====================


# 平均风速变化
mode1 = (
    5
    +
    2*np.sin(
        2*np.pi*0.05*t
    )
)



# 阵风
mode2 = (
    1.5*np.sin(
        2*np.pi*0.5*t
    )
)



# 高频湍流
mode3 = (
    0.5*np.sin(
        2*np.pi*3*t
    )
    +
    0.2*np.random.randn(N)
)



# 总风速

wind = (
    mode1
    +
    mode2
    +
    mode3
)



os.makedirs(
    "data",
    exist_ok=True
)


np.save(
    "data/wind_speed.npy",
    wind
)



plt.figure(
    figsize=(12,4)
)


plt.plot(
    t,
    wind
)


plt.xlabel(
    "Time(s)"
)

plt.ylabel(
    "Wind speed(m/s)"
)


plt.title(
    "Synthetic wind speed"
)


plt.tight_layout()


plt.savefig(
    "data/wind_speed.png",
    dpi=300
)


plt.show()



print(
    "wind data saved:",
    wind.shape
)