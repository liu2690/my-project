import numpy as np
import matplotlib.pyplot as plt
import time

import sys
import os


current_dir = os.path.dirname(
    os.path.abspath(__file__)
)

result_path = current_dir

rvmd_path = os.path.join(
    current_dir,
    "../RVMD"
)


sys.path.append(
    os.path.abspath(rvmd_path)
)


from rvmd import rvmd


# ============================
# signal generation
# ============================

np.random.seed(0)

T = 1000

fs = 1 / T

t = np.arange(1, T+1) / T



# center frequencies

f_1 = 2
f_2 = 24
f_3 = 288



# components

v_1 = np.cos(
    2*np.pi*f_1*t
)


v_2 = (
    1/4
    *
    np.cos(2*np.pi*f_2*t)
    *
    (10*t**2)
)


# MATLAB:
# [ones(1,ceil(T/2)),zeros(1,floor(T/2))]

window = np.concatenate(
    [
        np.ones(int(np.ceil(T/2))),
        np.zeros(int(np.floor(T/2)))
    ]
)


v_2 = v_2 * window



v_3 = (
    1/16
    *
    np.cos(
        2*np.pi*f_3*t
    )
)



noiseAmp = 1/24



# composite signal

f = (
    v_1
    +
    v_2
    +
    v_3
    +
    noiseAmp*np.random.randn(
        T
    )
)



# ============================
# plot signal
# ============================


plt.figure(figsize=(8,8))


plt.subplot(5,1,1)
plt.plot(t,f,'k')
plt.title("Original data")


plt.subplot(5,1,2)
plt.plot(t,f,'k')


plt.subplot(5,1,3)
plt.plot(t,v_1,'k')
plt.title(
    f"component 1 frequency={f_1*fs}"
)


plt.subplot(5,1,4)
plt.plot(t,v_2,'k')
plt.title(
    f"component 2 frequency={f_2*fs}"
)


plt.subplot(5,1,5)
plt.plot(t,v_3,'k')
plt.title(
    f"component 3 frequency={f_3*fs}"
)


plt.xlabel("time")

plt.tight_layout()

plt.savefig(
    os.path.join(
        result_path,
        "original_signal.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.close()



# ============================
# RVMD parameters
# ============================


K = 3

alpha = 1000

tol = 1e-5

N = 1000

init = 1

initFreqMax = 0.3



# ============================
# RVMD computation
# ============================

start = time.time()

mode, info, restart = rvmd(
    f.reshape(1,-1),
    K,
    alpha,
    Tolerance=tol,
    MaximumSteps=N,
    InitFreqType=init,
    InitFreqMaximum=initFreqMax,
    Device="cpu",
    FPPrecision="double",
    Display="iter"
)

end = time.time()

print(
    "Runtime:",
    end-start,
    "seconds"
)

# ============================
# reconstruction
# ============================


# MATLAB:
# f_rec = mode.phi * mode.c.'

f_rec = (
    mode["phi"]
    @
    mode["c"].T
)



f_rec = f_rec.flatten()



plt.figure(figsize=(8,8))


plt.subplot(K+2,1,1)

plt.plot(
    f_rec,
    'b'
)

plt.title(
    "RVMD reconstructed data"
)


for k in range(K):

    plt.subplot(
        K+2,
        1,
        k+3
    )

    plt.plot(
        t,
        mode["c"][:,k],
        'b'
    )

    plt.title(
        f"mode {k+1}, omega={mode['omega'][k]}"
    )


plt.tight_layout()

plt.savefig(
    os.path.join(
        result_path,
        "rvmd_reconstruction.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.close()



# ============================
# convergence curve
# ============================


plt.figure()

plt.plot(
    info["Iteration"]["omega"].T
)

plt.xlabel(
    "iteration step n"
)

plt.ylabel(
    "omega"
)

plt.savefig(
    os.path.join(
        result_path,
        "convergence.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.close()



# ============================
# spectrum
# ============================


f_spec = np.abs(
    np.fft.fft(f)
)**2


f_spec_rec = np.abs(
    np.fft.fft(f_rec)
)**2



nk = int(np.ceil(T/2))


freq = (
    np.arange(nk)
    /
    T
)



plt.figure()

plt.plot(
    freq,
    f_spec[:nk],
    label="original"
)


plt.plot(
    freq,
    f_spec_rec[:nk],
    '--',
    label="reconstructed"
)



for k in range(K):

    c_spec = (
        np.abs(
            np.fft.fft(
                mode["c"][:,k]
            )
        )**2
    )


    plt.plot(
        freq,
        c_spec[:nk],
        label=f"mode {k+1}"
    )



plt.xscale("log")
plt.yscale("log")

plt.legend()

plt.savefig(
    os.path.join(
        result_path,
        "spectrum.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("RVMD frequencies:")
print(mode["omega"])

print("Energy:")
print(mode["energy"])