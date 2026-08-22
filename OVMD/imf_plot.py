import numpy as np
import matplotlib.pyplot as plt


def IMFPlot(
        u,
        uhat,
        omega,
        ErrL2,
        K,
        X,
        time,
        alpha,
        type_name,
        subscript,
        notes,
        Legend=None
):
    """
    Plot OVMD decomposition results

    Equivalent to MATLAB IMFPlot.m
    """


    fs = 1 / (
        time[1]
        -
        time[0]
    )


    evenSize = (
        len(X)//2
    )*2



    maxModePlot = 12



    # ==============================
    # IMF time-domain plot
    # ==============================


    nFig = min(
        int(np.ceil(K/2))+1,
        int(np.ceil(maxModePlot/2))+1
    )


    fig, axes = plt.subplots(
        nFig,
        2,
        figsize=(10,3*nFig)
    )


    axes = np.array(
        axes
    ).reshape(-1)



    for i in range(
        min(K,maxModePlot)
    ):

        ax = axes[i]

        ax.plot(
            time[:evenSize],
            u[i,:],
            linewidth=0.8
        )


        freq = (
            omega[-1,i]
            *
            fs
        )


        ax.set_title(
            f"Mode {i+1}, fc={freq:.2f}"
        )


        ax.grid()



    # reconstruction error

    ax = axes[-1]


    error = (
        np.sum(u,axis=0)
        -
        X[:evenSize]
    )


    ax.plot(
        time[:evenSize],
        error
    )


    ax.set_title(
        f"Reconstruction error L2={ErrL2:.4f}"
    )


    ax.grid()

    import os

    os.makedirs("result", exist_ok=True)

    plt.tight_layout()



    plt.savefig(
        f"result/{type_name}_Decomposition_Alpha_{alpha}_{notes}.png",
        dpi=300
    )


    plt.close()



    # ==============================
    # Spectrum plot
    # ==============================


    N = len(time)


    freq = (
        fs
        *
        np.arange(
            0,
            N//2
        )
        /
        N
    )


    X_fft = np.abs(
        np.fft.fft(X)
    )


    fig, ax = plt.subplots(
        figsize=(8,4)
    )


    ax.loglog(
        freq,
        2*X_fft[:N//2]/len(freq),
        label="X"
    )



    for i in range(
        min(K,maxModePlot)
    ):

        mode_fft = np.abs(
            uhat[:,i]
        )


        ax.loglog(
            freq[:len(mode_fft)//2],
            2*mode_fft[:len(mode_fft)//2]/len(freq),
            label=f"u{i+1}"
        )


    ax.set_xlabel("Frequency")

    ax.set_ylabel("Amplitude")

    ax.grid()

    ax.legend()



    plt.tight_layout()


    plt.savefig(
        f"result/{type_name}_Spectrum_Alpha_{alpha}_{notes}.png",
        dpi=300
    )


    plt.close()