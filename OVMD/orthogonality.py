import numpy as np
import matplotlib.pyplot as plt
import os



def orthogonalityIMF(
        Signal,
        K,
        invBand
):
    """
    Python version of orthogonalityIMF.m

    Compare VMD and OVMD orthogonality

    Parameters
    ----------
    Signal : dict
        contains VMD and OVMD results

    K : int
        number of modes

    invBand : float
        alpha parameter
    """



    os.makedirs(
        "result",
        exist_ok=True
    )


    # ==================================
    # Combine VMD and OVMD matrices
    # ==================================


    OCoef = np.zeros(
        (K+1,K+1)
    )

    OCoefn = np.zeros(
        (K+1,K+1)
    )


    CProd = np.zeros(
        (K+1,K+1)
    )

    CProdn = np.zeros(
        (K+1,K+1)
    )


    # MATLAB:
    # triu(OVMD)+tril(VMD,-1)

    OCoef[:K,:K] = (

        np.triu(
            Signal["OVMDwoNoise"]["OCoef"]
        )

        +

        np.tril(
            Signal["VMDwoNoise"]["OCoef"],
            -1
        )

    )


    OCoefn[:K,:K] = (

        np.triu(
            Signal["OVMDwNoise"]["OCoef"]
        )

        +

        np.tril(
            Signal["VMDwNoise"]["OCoef"],
            -1
        )

    )



    CProd[:K,:K] = (

        np.triu(
            Signal["OVMDwoNoise"]["CProd"]
        )

        +

        np.tril(
            Signal["VMDwoNoise"]["CProd"],
            -1
        )

    )


    CProdn[:K,:K] = (

        np.triu(
            Signal["OVMDwNoise"]["CProdn"]
        )

        +

        np.tril(
            Signal["VMDwNoise"]["CProdn"],
            -1
        )

    )



    # diagonal cross product

    CProdDiag = np.zeros(
        (K,2)
    )

    CProdnDiag = np.zeros(
        (K,2)
    )


    CProdDiag[:,0] = np.diag(
        Signal["VMDwoNoise"]["CProd"]
    )

    CProdDiag[:,1] = np.diag(
        Signal["OVMDwoNoise"]["CProd"]
    )


    CProdnDiag[:,0] = np.diag(
        Signal["VMDwNoise"]["CProdn"]
    )

    CProdnDiag[:,1] = np.diag(
        Signal["OVMDwNoise"]["CProdn"]
    )



    # ==================================
    # First correlation plot
    # ==================================


    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10,4)
    )


    axes[0].imshow(
        np.abs(OCoef),
        origin="lower",
        aspect="equal"
    )

    axes[0].set_title(
        "Correlation coefficient"
    )


    axes[0].set_xlabel(
        "mode"
    )


    axes[0].set_ylabel(
        "mode"
    )



    axes[1].imshow(
        np.abs(CProdn),
        origin="lower",
        aspect="equal"
    )


    axes[1].set_title(
        "Cross Product"
    )


    axes[1].set_xlabel(
        "mode"
    )


    plt.tight_layout()


    plt.savefig(
        f"result/Ortho_Correlation_LocalizedEvents_Alpha_{invBand}.png",
        dpi=300
    )


    plt.close()

        # ==================================
    # Plot first 10 modes only
    # ==================================


    K_plot = min(
        K,
        10
    )


    OCoef10 = OCoef[
        :K_plot+1,
        :K_plot+1
    ]


    CProdn10 = CProdn[
        :K_plot+1,
        :K_plot+1
    ]



    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10,4)
    )



    # -------------------------------
    # without noise
    # -------------------------------

    im1 = axes[0].imshow(
        np.abs(OCoef10),
        origin="lower",
        aspect="equal"
    )


    axes[0].set_title(
        "OVMD/VMD Correlation (first 10)"
    )


    axes[0].set_xlabel(
        "mode"
    )


    axes[0].set_ylabel(
        "mode"
    )



    plt.colorbar(
        im1,
        ax=axes[0]
    )



    # -------------------------------
    # with noise
    # -------------------------------


    im2 = axes[1].imshow(
        np.abs(CProdn10),
        origin="lower",
        aspect="equal"
    )


    axes[1].set_title(
        "Cross Product (first 10)"
    )


    axes[1].set_xlabel(
        "mode"
    )


    axes[1].set_ylabel(
        "mode"
    )


    plt.colorbar(
        im2,
        ax=axes[1]
    )



    plt.tight_layout()



    plt.savefig(
        f"result/Ortho_Correlation_First10_LocalizedEvents_Alpha_{invBand}.png",
        dpi=300
    )


    plt.close()



    return {
        "OCoef": OCoef,
        "OCoefn": OCoefn,
        "CProd": CProd,
        "CProdn": CProdn,
        "CProdDiag": CProdDiag,
        "CProdnDiag": CProdnDiag
    }