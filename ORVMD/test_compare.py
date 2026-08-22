import numpy as np
import sys
import os
import matplotlib.pyplot as plt


# ==========================================
# import RVMD and ORVMD
# ==========================================

sys.path.append(
    os.path.abspath("../RVMD")
)


from rvmd import rvmd
from orvmd import orvmd



# ==========================================
# Generate space-time data
# ==========================================

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



# --------------------------
# spatial modes
# --------------------------

phi1 = np.sin(x)

phi2 = np.cos(2*x)



# --------------------------
# temporal dynamics
# close frequencies
# --------------------------

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



# --------------------------
# construct Q(x,t)
# --------------------------

Q = (

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



print("Data shape:")
print(Q.shape)



# ==========================================
# Run RVMD
# ==========================================


mode_rvmd, info_rvmd, _ = rvmd(

    Q,

    K=4,

    Alpha=1000,

    MaximumSteps=500

)



# ==========================================
# Run ORVMD
# ==========================================


mode_orvmd, info_orvmd, _ = orvmd(

    Q,

    K=4,

    Alpha=1000,

    lambda_orth=0.05,

    MaximumSteps=500

)



# ==========================================
# Correlation matrix
# ==========================================


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



# ==========================================
# Reconstruction error
# ==========================================


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



# ==========================================
# Orthogonality index
# ==========================================


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


    return total / count



# ==========================================
# Print results
# ==========================================


print("==============================")

print("RVMD energy")

print(
    mode_rvmd["energy"]
)


print("==============================")

print("ORVMD energy")

print(
    mode_orvmd["energy"]
)



print("==============================")

print("RVMD correlation")

print(
    correlation_matrix(
        mode_rvmd["c"]
    )
)



print("==============================")

print("ORVMD correlation")

print(
    correlation_matrix(
        mode_orvmd["c"]
    )
)



print("==============================")

print(
    "RVMD reconstruction error:"
)

print(
    reconstruction_error(
        Q,
        mode_rvmd
    )
)


print(
    "ORVMD reconstruction error:"
)

print(
    reconstruction_error(
        Q,
        mode_orvmd
    )
)



print("==============================")

print(
    "RVMD orthogonality index:"
)

print(
    orthogonality_index(
        mode_rvmd["c"]
    )
)



print(
    "ORVMD orthogonality index:"
)

print(
    orthogonality_index(
        mode_orvmd["c"]
    )
)



# ==========================================
# Spectrum plot
# ==========================================


def plot_spectrum(
        mode,
        title
):


    c = mode["c"]


    K = c.shape[1]


    T = c.shape[0]


    dt = 20/T


    freq = np.fft.rfftfreq(
        T,
        d=dt
    )


    plt.figure(
        figsize=(8,4)
    )


    for k in range(K):


        spectrum = np.abs(

            np.fft.rfft(
                c[:,k]
            )

        )


        plt.plot(

            freq,

            spectrum,

            label=f"mode {k+1}"

        )


    plt.xlabel(
        "Frequency"
    )


    plt.ylabel(
        "|C(f)|"
    )


    plt.title(
        title
    )


    plt.legend()


    plt.grid()


    plt.tight_layout()


    plt.show()



# ==========================================
# plot
# ==========================================


plot_spectrum(

    mode_rvmd,

    "RVMD spectrum"

)



plot_spectrum(

    mode_orvmd,

    "ORVMD spectrum"

)