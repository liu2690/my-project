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

from orvmd import orvmd



# ==========================
# generate test data
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



# temporal signals

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



# ==========================
# functions
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


    value = 0

    count = 0


    for i in range(K):

        for j in range(K):

            if i != j:

                value += abs(
                    C[i,j]
                )

                count += 1


    return value/count



def reconstruction_error(Q,mode):

    Qr = (

        mode["phi"]

        @

        mode["c"].T

    )


    return (

        np.linalg.norm(Q-Qr)

        /

        np.linalg.norm(Q)

    )



# ==========================
# lambda scan
# ==========================


lambda_list = [

    0,

    0.01,

    0.05,

    0.1,

    0.2,

    0.5,

    1.0

]


orth_values = []

error_values = []



for lam in lambda_list:


    print("====================")

    print(
        "lambda_orth = ",
        lam
    )


    mode,info,_ = orvmd(

        Q,

        K=4,

        Alpha=1000,

        lambda_orth=lam,

        MaximumSteps=500

    )


    orth = orthogonality_index(
        mode["c"]
    )


    error = reconstruction_error(
        Q,
        mode
    )


    orth_values.append(
        orth
    )


    error_values.append(
        error
    )


    print(
        "orthogonality:",
        orth
    )


    print(
        "error:",
        error
    )



# ==========================
# plot
# ==========================


plt.figure(
    figsize=(6,4)
)


plt.plot(

    lambda_list,

    orth_values,

    marker="o"

)


plt.xlabel(
    "lambda_orth"
)


plt.ylabel(
    "Orthogonality index"
)


plt.grid()


plt.show()



plt.figure(
    figsize=(6,4)
)


plt.plot(

    lambda_list,

    error_values,

    marker="o"

)


plt.xlabel(
    "lambda_orth"
)


plt.ylabel(
    "Reconstruction error"
)


plt.grid()


plt.show()