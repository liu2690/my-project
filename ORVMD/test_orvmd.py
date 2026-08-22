import numpy as np

from orvmd import orvmd


# ============================
# generate test data
# ============================

S = 100
T = 500


t = np.linspace(
    0,
    10,
    T
)


# two transient oscillations

signal1 = (
    np.sin(2*np.pi*1*t)
    *
    np.exp(
        -(t-5)**2
    )
)


signal2 = (
    0.5
    *
    np.sin(2*np.pi*3*t)
)



time_signal = signal1 + signal2


# create space-time matrix

Q = np.zeros(
    (S,T)
)


for i in range(S):

    Q[i,:] = (
        (1+i/S)
        *
        time_signal
    )



# ============================
# run ORVMD
# ============================


mode, info, restart = orvmd(

    Q,

    K=5,

    Alpha=1000,

    lambda_orth=0.05,

    MaximumSteps=500,

    Display="iter"

)



print("===================")

print("phi shape:")
print(
    mode["phi"].shape
)


print("c shape:")
print(
    mode["c"].shape
)


print("omega:")
print(
    mode["omega"]
)


print("energy:")
print(
    mode["energy"]
)


print("===================")

print(
    "iterations:",
    info["Iteration"]["steps"]
)