import numpy as np
import sys
import os
import matplotlib.pyplot as plt


# ==========================
# import
# ==========================

sys.path.append(
    os.path.abspath("../RVMD")
)


from rvmd import rvmd
from orvmd import orvmd



# ==========================
# generate clean data
# ==========================


S = 200
T = 800


x = np.linspace(
    0,
    2*np.pi,
    S
)


t = np.linspace(
    0,
    20,
    T
)



# spatial modes

phi1 = np.sin(x)

phi2 = np.cos(2*x)



# temporal modes

c1 = np.sin(
    2*np.pi*1.0*t
)


c2 = (
    0.8
    *
    np.sin(
        2*np.pi*1.2*t
    )
)



Q_clean = (

    np.outer(
        phi1,
        c1
    )

    +

    np.outer(
        phi2,
        c2
    )

)



# ==========================
# add noise
# ==========================


def add_noise(
        Q,
        snr_db
):

    signal_power = np.mean(
        Q**2
    )


    noise_power = (
        signal_power
        /
        (10**(snr_db/10))
    )


    noise = np.sqrt(
        noise_power
    ) * np.random.randn(
        *Q.shape
    )


    return Q + noise



# ==========================
# metrics
# ==========================


def correlation_matrix(c):

    K = c.shape[1]

    C = np.zeros(
        (K,K)
    )


    for i in range(K):

        for j in range(K):

            C[i,j] = (

                np.abs(
                    np.dot(
                        c[:,i],
                        c[:,j]
                    )
                )

                /

                (
                    np.linalg.norm(c[:,i])
                    *
                    np.linalg.norm(c[:,j])
                    +
                    1e-12
                )

            )


    return C



def orthogonality_index(c):

    C = correlation_matrix(c)

    K = C.shape[0]


    total = 0
    count = 0


    for i in range(K):

        for j in range(K):

            if i != j:

                total += abs(
                    C[i,j]
                )

                count += 1


    return total/count



def reconstruction_error(
        Q,
        mode
):

    Qr = (

        mode["phi"]

        @

        mode["c"].T

    )


    return (

        np.linalg.norm(
            Q-Qr
        )

        /

        np.linalg.norm(Q)

    )



# ==========================
# noise test
# ==========================


SNR_list = [

    30,

    20,

    10,

    5

]


rvmd_orth = []

orvmd_orth = []


rvmd_error = []

orvmd_error = []



for snr in SNR_list:


    print("================")

    print(
        "SNR = ",
        snr
    )


    Q = add_noise(
        Q_clean,
        snr
    )



    # RVMD

    mode_rvmd,_,_ = rvmd(

        Q,

        K=4,

        Alpha=1000,

        MaximumSteps=500

    )


    # ORVMD

    mode_orvmd,_,_ = orvmd(

        Q,

        K=4,

        Alpha=1000,

        lambda_orth=0.05,

        MaximumSteps=500

    )



    r1 = orthogonality_index(
        mode_rvmd["c"]
    )


    r2 = orthogonality_index(
        mode_orvmd["c"]
    )


    e1 = reconstruction_error(
        Q,
        mode_rvmd
    )


    e2 = reconstruction_error(
        Q,
        mode_orvmd
    )



    rvmd_orth.append(r1)

    orvmd_orth.append(r2)


    rvmd_error.append(e1)

    orvmd_error.append(e2)



    print(
        "RVMD orth:",
        r1
    )


    print(
        "ORVMD orth:",
        r2
    )


    print(
        "RVMD error:",
        e1
    )


    print(
        "ORVMD error:",
        e2
    )



# ==========================
# plots
# ==========================


plt.figure(
    figsize=(6,4)
)


plt.plot(
    SNR_list,
    rvmd_orth,
    marker="o",
    label="RVMD"
)


plt.plot(
    SNR_list,
    orvmd_orth,
    marker="o",
    label="ORVMD"
)


plt.xlabel(
    "SNR(dB)"
)


plt.ylabel(
    "Orthogonality index"
)


plt.legend()

plt.grid()

plt.show()



plt.figure(
    figsize=(6,4)
)


plt.plot(
    SNR_list,
    rvmd_error,
    marker="o",
    label="RVMD"
)


plt.plot(
    SNR_list,
    orvmd_error,
    marker="o",
    label="ORVMD"
)


plt.xlabel(
    "SNR(dB)"
)


plt.ylabel(
    "Reconstruction error"
)


plt.legend()

plt.grid()

plt.savefig(
    "snr_orthogonality.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

plt.savefig(
    "snr_error.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()