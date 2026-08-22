import numpy as np

from signal_generator import signalGenerator
from ovmd_decomposition import OVMDDecomposition

from VMD.vmd_decomposition import VMDDecomposition
from orthogonality import orthogonalityIMF


# =====================================
# Create Synthetic Signal
# =====================================


fs = 100          # Sampling frequency

t_end = 20        # Total time

SNR = 10          # Signal to noise ratio


newNoise = 0



# MATLAB:
# InSignal=[1,2,5,6,7,8,9]

InSignal = [
    1,
    2,
    5,
    6,
    7,
    8,
    9
]


Signal, time = signalGenerator(
    InSignal,
    SNR=SNR,
    tEnd=t_end,
    fs=fs,
    loadNoise=newNoise
)



# =====================================
# VMD and OVMD parameters
# =====================================


twoF = 100

tau = 0.0

tol = 1e-5

N = 1000

K = 10

DC = 0

init = 0

# =====================================
# VMD and OVMD decomposition
# =====================================


VMDwoNoise, VMDwNoise = VMDDecomposition(

    Signal["woNoise"],
    Signal["wNoise"],
    time,
    twoF,
    tau,
    K,
    DC,
    init,
    tol,
    N
)



OVMDwoNoise, OVMDwNoise = OVMDDecomposition(

    Signal["woNoise"],
    Signal["wNoise"],
    time,
    twoF,
    tau,
    K,
    tol,
    N
)



Signal["VMDwoNoise"] = VMDwoNoise
Signal["VMDwNoise"] = VMDwNoise

Signal["OVMDwoNoise"] = OVMDwoNoise
Signal["OVMDwNoise"] = OVMDwNoise



print("======================")

print("VMD and OVMD finished")


print(
    "VMD clean error:",
    VMDwoNoise["ErrL2"]
)

print(
    "VMD noise error:",
    VMDwNoise["ErrL2"]
)


print(
    "OVMD clean error:",
    OVMDwoNoise["ErrL2"]
)

print(
    "OVMD noise error:",
    OVMDwNoise["ErrL2"]
)


print(
    "OVMD iterations:",
    OVMDwoNoise["iter"]
)


print(
    "OVMD energy:",
    OVMDwoNoise["energy"]
)

ortho_result = orthogonalityIMF(
    Signal,
    K,
    twoF
)

print("Orthogonality calculation finished")

print("======================")