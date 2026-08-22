import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ============================
# 添加项目路径
# ============================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(ROOT)

# 加入 ORVMD 路径
sys.path.append(
    os.path.join(
        ROOT,
        "ORVMD"
    )
)

# 加入 OVMD 路径
sys.path.append(
    os.path.join(
        ROOT,
        "OVMD"
    )
)


from OVMD.ovmd import OVMD
from ORVMD.orvmd import orvmd



# ============================
# 模态相干度
# ============================

def coherence_matrix(modes):

    """
    Frequency-domain modal coherence

    modes:
        K x T

    return:
        K x K coherence matrix
    """

    K = modes.shape[0]


    # FFT
    spectrum = np.fft.fft(
        modes,
        axis=1
    )


    C = np.zeros(
        (K,K)
    )


    for i in range(K):

        for j in range(K):

            if i == j:

                C[i,j] = 1.0

            else:

                value = np.abs(
                    spectrum[i]
                    *
                    np.conj(
                        spectrum[j]
                    )
                )


                value /= (
                    np.linalg.norm(
                        spectrum[i]
                    )
                    *
                    np.linalg.norm(
                        spectrum[j]
                    )
                    +
                    1e-12
                )


                # 对频率求平均

                C[i,j] = np.max(
                    value
                )


    return C



def average_coherence(C):

    values=[]

    K=C.shape[0]


    for i in range(K):

        for j in range(K):

            if i != j:

                values.append(
                    C[i,j]
                )


    return np.mean(values)



# ============================
# 绘图
# ============================


def plot_heatmap(
        C,
        title,
        filename
):

    plt.figure(
        figsize=(5,4)
    )

    plt.imshow(
        C,
        cmap="jet",
        vmin=0,
        vmax=1
    )

    plt.colorbar()

    plt.title(title)

    plt.xlabel(
        "Mode"
    )

    plt.ylabel(
        "Mode"
    )


    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300
    )

    plt.close()



def plot_modes(
        modes,
        title,
        filename
):

    K=modes.shape[0]


    plt.figure(
        figsize=(10,6)
    )


    for k in range(K):

        plt.subplot(
            K,
            1,
            k+1
        )

        plt.plot(
            modes[k]
        )

        plt.ylabel(
            f"Mode {k+1}"
        )


    plt.suptitle(
        title
    )

    plt.tight_layout()


    plt.savefig(
        filename,
        dpi=300
    )

    plt.close()



# ============================
# 读取风场
# ============================


Q = np.load(
    "data/wind_field.npy"
)


print(
    "Wind field:",
    Q.shape
)


os.makedirs(
    "result",
    exist_ok=True
)



# ============================
# 1. OVMD
# ============================


print("\nRunning OVMD...")


# 多测点平均风速

wind_mean = Q.mean(
    axis=0
)


ovmd_result = OVMD(

    signal=wind_mean,

    alpha=2000,

    K=4,

    Tolerance=1e-5,

    Tau=0,

    MaximumSteps=1000

)


ovmd_modes = np.asarray(
    ovmd_result["u"]
)



if ovmd_modes.shape[0] != 3:

    ovmd_modes = ovmd_modes.T



print(
    "OVMD modes:",
    ovmd_modes.shape
)



# ============================
# 2. ORVMD
# ============================


print("\nRunning ORVMD...")


orvmd_result, info, restart = orvmd(

    Q=Q,

    K=4,

    Alpha=2000,

    lambda_orth=5.0,

    Tolerance=5e-3,

    MaximumSteps=500,

    Display="off"

)



phi = orvmd_result["phi"]

c = orvmd_result["c"]


orvmd_modes=[]


for k in range(3):

    mode_k = (
        phi[:,k:k+1]
        *
        c[:,k].reshape(1,-1)
    )

    orvmd_modes.append(
        mode_k.flatten()
    )


orvmd_modes=np.array(
    orvmd_modes
)



print(
    "ORVMD modes:",
    orvmd_modes.shape
)



# ============================
# 3. coherence
# ============================


C_ovmd = coherence_matrix(
    ovmd_modes
)


C_orvmd = coherence_matrix(
    orvmd_modes
)



mc_ovmd = average_coherence(
    C_ovmd
)


mc_orvmd = average_coherence(
    C_orvmd
)



print("\n====================")

print(
    "OVMD average coherence:",
    mc_ovmd
)

print(
    "ORVMD average coherence:",
    mc_orvmd
)

print("====================")



# ============================
# 4. 保存
# ============================


pd.DataFrame(
    C_ovmd
).to_csv(
    "result/ovmd_coherence.csv",
    index=False
)


pd.DataFrame(
    C_orvmd
).to_csv(
    "result/orvmd_coherence.csv",
    index=False
)



pd.DataFrame(
    {
        "method":
        [
            "OVMD",
            "ORVMD"
        ],

        "average_coherence":
        [
            mc_ovmd,
            mc_orvmd
        ]
    }

).to_csv(
    "result/coherence_compare.csv",
    index=False
)



plot_modes(
    ovmd_modes,
    "OVMD modes",
    "result/ovmd_modes.png"
)


plot_modes(
    orvmd_modes,
    "ORVMD modes",
    "result/orvmd_modes.png"
)



plot_heatmap(
    C_ovmd,
    "OVMD modal coherence",
    "result/ovmd_heatmap.png"
)


plot_heatmap(
    C_orvmd,
    "ORVMD modal coherence",
    "result/orvmd_heatmap.png"
)



print(
    "\nExperiment finished."
)