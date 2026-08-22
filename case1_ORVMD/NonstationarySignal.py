import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os


# =====================================================
# path
# =====================================================

current_dir = os.path.dirname(
    os.path.abspath(__file__)
)

result_path = current_dir


sys.path.append(
    os.path.abspath("../RVMD")
)

sys.path.append(
    os.path.abspath("../ORVMD")
)


from rvmd import rvmd
from orvmd import orvmd



# =====================================================
# signal generation
# =====================================================

np.random.seed(0)


T = 1000


fs = 1 / T


t = np.arange(
    1,
    T+1
) / T



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

    np.cos(
        2*np.pi*f_2*t
    )

    *

    (10*t**2)

)



window = np.concatenate(

    [

        np.ones(
            int(np.ceil(T/2))
        ),

        np.zeros(
            int(np.floor(T/2))
        )

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



f = (

    v_1

    +

    v_2

    +

    v_3

    +

    noiseAmp*np.random.randn(T)

)



# =====================================================
# original signal
# =====================================================


plt.figure(
    figsize=(8,8)
)


plt.subplot(5,1,1)

plt.plot(
    t,
    f,
    'k'
)

plt.title(
    "Original signal"
)


plt.subplot(5,1,2)

plt.plot(
    t,
    f,
    'k'
)



plt.subplot(5,1,3)

plt.plot(
    t,
    v_1,
    'k'
)

plt.title(
    f"component 1 frequency={f_1*fs}"
)



plt.subplot(5,1,4)

plt.plot(
    t,
    v_2,
    'k'
)

plt.title(
    f"component 2 frequency={f_2*fs}"
)



plt.subplot(5,1,5)

plt.plot(
    t,
    v_3,
    'k'
)

plt.title(
    f"component 3 frequency={f_3*fs}"
)


plt.xlabel(
    "time"
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        result_path,
        "original_signal.png"
    ),

    dpi=300

)


plt.close()



# =====================================================
# parameters
# =====================================================


K = 3

alpha = 1000

tol = 1e-5

N = 1000

init = 1

initFreqMax = 0.3


lambda_orth = 0.05



# =====================================================
# RVMD
# =====================================================


print("======================")

print("Running RVMD")


start = time.time()



mode_rvmd, info_rvmd, _ = rvmd(

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



print(
    "RVMD runtime:",
    time.time()-start
)



# =====================================================
# ORVMD
# =====================================================


print("======================")

print("Running ORVMD")


start = time.time()



mode_orvmd, info_orvmd, _ = orvmd(

    f.reshape(1,-1),

    K,

    alpha,

    lambda_orth=lambda_orth,

    Tolerance=tol,

    MaximumSteps=N,

    InitFreqType=init,

    InitFreqMaximum=initFreqMax,

    Device="cpu",

    FPPrecision="double",

    Display="iter"

)



print(
    "ORVMD runtime:",
    time.time()-start
)

# =====================================================
# reconstruction
# =====================================================


f_rec_rvmd = (

    mode_rvmd["phi"]

    @

    mode_rvmd["c"].T

).flatten()



f_rec_orvmd = (

    mode_orvmd["phi"]

    @

    mode_orvmd["c"].T

).flatten()



# =====================================================
# reconstruction error
# =====================================================


def reconstruction_error(
        original,
        reconstructed
):

    return (

        np.linalg.norm(
            original-reconstructed
        )

        /

        np.linalg.norm(
            original
        )

    )



rvmd_error = reconstruction_error(
    f,
    f_rec_rvmd
)


orvmd_error = reconstruction_error(
    f,
    f_rec_orvmd
)


print("======================")

print(
    "RVMD error:",
    rvmd_error
)


print(
    "ORVMD error:",
    orvmd_error
)



# =====================================================
# reconstruction plot
# =====================================================


for name, rec, mode in [

    (
        "rvmd",
        f_rec_rvmd,
        mode_rvmd
    ),

    (
        "orvmd",
        f_rec_orvmd,
        mode_orvmd
    )

]:


    plt.figure(
        figsize=(8,8)
    )


    plt.subplot(
        K+2,
        1,
        1
    )


    plt.plot(
        t,
        rec,
        'b'
    )


    plt.title(
        name.upper()+" reconstruction"
    )



    for k in range(K):


        plt.subplot(
            K+2,
            1,
            k+2
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

            name+"_reconstruction.png"

        ),

        dpi=300

    )


    plt.close()



# =====================================================
# frequency spectrum comparison
# =====================================================


nk = int(
    np.ceil(T/2)
)


freq = (

    np.arange(nk)

    /

    T

)



plt.figure(
    figsize=(8,5)
)



plt.plot(

    freq,

    np.abs(
        np.fft.fft(f)
    )[:nk]**2,

    label="original"

)



for k in range(K):


    rvmd_spec = (

        np.abs(

            np.fft.fft(
                mode_rvmd["c"][:,k]
            )

        )**2

    )


    plt.plot(

        freq,

        rvmd_spec[:nk],

        label=f"RVMD mode {k+1}"

    )



for k in range(K):


    orvmd_spec = (

        np.abs(

            np.fft.fft(
                mode_orvmd["c"][:,k]
            )

        )**2

    )


    plt.plot(

        freq,

        orvmd_spec[:nk],

        "--",

        label=f"ORVMD mode {k+1}"

    )



plt.xscale(
    "log"
)

plt.yscale(
    "log"
)


plt.xlabel(
    "frequency"
)


plt.ylabel(
    "power"
)


plt.legend()


plt.tight_layout()


plt.savefig(

    os.path.join(

        result_path,

        "comparison_spectrum.png"

    ),

    dpi=300

)


plt.close()



# =====================================================
# convergence curve
# =====================================================


plt.figure()


plt.plot(

    info_rvmd["Iteration"]["omega"].T,

    label="RVMD"

)


plt.plot(

    info_orvmd["Iteration"]["omega"].T,

    "--",

    label="ORVMD"

)



plt.xlabel(
    "iteration"
)


plt.ylabel(
    "omega"
)


plt.legend()


plt.tight_layout()


plt.savefig(

    os.path.join(

        result_path,

        "convergence.png"

    ),

    dpi=300

)


plt.close()



# =====================================================
# correlation matrix
# =====================================================


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



C_rvmd = correlation_matrix(
    mode_rvmd["c"]
)


C_orvmd = correlation_matrix(
    mode_orvmd["c"]
)



print("======================")

print("RVMD correlation matrix")

print(C_rvmd)



print("======================")

print("ORVMD correlation matrix")

print(C_orvmd)



# =====================================================
# correlation heatmap
# =====================================================


plt.figure(
    figsize=(10,4)
)



plt.subplot(
    1,
    2,
    1
)


plt.imshow(

    C_rvmd,

    cmap="coolwarm",

    vmin=0,

    vmax=1

)


plt.colorbar()


plt.title(
    "RVMD correlation"
)



plt.subplot(
    1,
    2,
    2
)


plt.imshow(

    C_orvmd,

    cmap="coolwarm",

    vmin=0,

    vmax=1

)


plt.colorbar()


plt.title(
    "ORVMD correlation"
)



plt.tight_layout()


plt.savefig(

    os.path.join(

        result_path,

        "correlation_comparison.png"

    ),

    dpi=300

)


plt.close()

# =====================================================
# mode comparison
# =====================================================


for k in range(K):


    plt.figure(
        figsize=(8,4)
    )


    plt.plot(

        t,

        mode_rvmd["c"][:,k],

        label="RVMD",

        linewidth=1.2

    )


    plt.plot(

        t,

        mode_orvmd["c"][:,k],

        "--",

        label="ORVMD",

        linewidth=1.2

    )


    plt.xlabel(
        "time"
    )


    plt.ylabel(
        "c(t)"
    )


    plt.title(
        f"Mode {k+1} comparison"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            result_path,

            f"mode_{k+1}_comparison.png"

        ),

        dpi=300

    )


    plt.close()



# =====================================================
# all modes comparison
# =====================================================


plt.figure(
    figsize=(10,8)
)



for k in range(K):


    plt.subplot(
        K,
        1,
        k+1
    )


    plt.plot(

        t,

        mode_rvmd["c"][:,k],

        label="RVMD"

    )


    plt.plot(

        t,

        mode_orvmd["c"][:,k],

        "--",

        label="ORVMD"

    )


    plt.ylabel(
        f"mode {k+1}"
    )


    if k == 0:

        plt.legend()



plt.xlabel(
    "time"
)


plt.tight_layout()


plt.savefig(

    os.path.join(

        result_path,

        "mode_comparison_all.png"

    ),

    dpi=300

)


plt.close()



# =====================================================
# energy comparison
# =====================================================


plt.figure(
    figsize=(6,4)
)


index = np.arange(K)



plt.bar(

    index-0.2,

    mode_rvmd["energy"],

    width=0.4,

    label="RVMD"

)



plt.bar(

    index+0.2,

    mode_orvmd["energy"],

    width=0.4,

    label="ORVMD"

)



plt.xlabel(
    "mode"
)


plt.ylabel(
    "Energy"
)


plt.legend()


plt.tight_layout()


plt.savefig(

    os.path.join(

        result_path,

        "energy_comparison.png"

    ),

    dpi=300

)


plt.close()



# =====================================================
# orthogonality index
# =====================================================


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



rvmd_orth = orthogonality_index(
    mode_rvmd["c"]
)


orvmd_orth = orthogonality_index(
    mode_orvmd["c"]
)



print("======================")

print(
    "RVMD orthogonality index:",
    rvmd_orth
)


print(
    "ORVMD orthogonality index:",
    orvmd_orth
)



# =====================================================
# final output
# =====================================================


print("======================")

print(
    "lambda_orth:",
    lambda_orth
)


print(
    "K:",
    K
)


print(
    "Alpha:",
    alpha
)



print("======================")

print(
    "RVMD frequencies:"
)

print(
    mode_rvmd["omega"]
)



print(
    "ORVMD frequencies:"
)

print(
    mode_orvmd["omega"]
)



print("======================")

print(
    "RVMD energy:"
)

print(
    mode_rvmd["energy"]
)



print(
    "ORVMD energy:"
)

print(
    mode_orvmd["energy"]
)

# =====================================================
# summary table
# =====================================================

print("======================")
print("Summary")

print(
    "RVMD error:",
    rvmd_error
)

print(
    "ORVMD error:",
    orvmd_error
)


print(
    "RVMD orthogonality:",
    rvmd_orth
)


print(
    "ORVMD orthogonality:",
    orvmd_orth
)

print("======================")

print(
    "Finished"
)