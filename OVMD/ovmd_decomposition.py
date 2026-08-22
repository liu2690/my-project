import time
import numpy as np

from ovmd import OVMD
from imf_plot import IMFPlot


def OVMDDecomposition(
        X,
        Xn,
        time_array,
        twoF,
        tau,
        K,
        tol,
        N
):
    """
    Wrapper function for OVMD

    Parameters
    ----------
    X : ndarray
        clean signal

    Xn : ndarray
        noisy signal

    time_array : ndarray
        time vector

    twoF : float
        alpha parameter

    tau : float
        dual ascent step

    K : int
        number of modes

    tol : float
        convergence tolerance

    N : int
        maximum iteration


    Returns
    -------
    OVMDOut
        clean signal decomposition

    OVMDOutn
        noisy signal decomposition
    """


    # sampling frequency
    fs = 1 / (
        time_array[1]
        -
        time_array[0]
    )


    # MATLAB:
    # evenSize=floor(size(X,2)/2)*2

    evenSize = (
        len(X)//2
    )*2



    # =================================
    # clean signal OVMD
    # =================================

    start = time.time()


    mode = OVMD(
        X[:evenSize],
        twoF,
        K,
        Tolerance=tol,
        MaximumSteps=N,
        Tau=tau
    )


    timewoNoise = (
        time.time()
        -
        start
    )



    # =================================
    # noisy signal OVMD
    # =================================


    start = time.time()


    moden = OVMD(
        Xn[:evenSize],
        twoF,
        K,
        Tolerance=tol,
        MaximumSteps=N,
        Tau=tau
    )


    timewNoise = (
        time.time()
        -
        start
    )



    u = mode["u"]

    un = moden["u"]



    # =================================
    # reconstruction error
    # =================================


    ErrL2 = (
        np.linalg.norm(
            np.sum(u, axis=0)
            -
            X[:evenSize]
        )
        /
        np.linalg.norm(
            X[:evenSize]
        )
    )


    ErrL2n = (
        np.linalg.norm(
            np.sum(un, axis=0)
            -
            Xn[:evenSize]
        )
        /
        np.linalg.norm(
            Xn[:evenSize]
        )
    )



    # =================================
    # output dictionary
    # =================================


    OVMDOut = {

        "u": mode["u"],

        "uhat": mode["uhat"],

        "omega": mode["omega"],

        "ErrL2": ErrL2,

        "iter":
            mode["omega"].shape[0],

        "time":
            timewoNoise,

        "energy":
            np.sum(
                mode["u"]**2,
                axis=1
            ),

        "alpha":
            twoF
    }



    OVMDOutn = {

        "u": moden["u"],

        "uhat": moden["uhat"],

        "omega": moden["omega"],

        "ErrL2": ErrL2n,

        "iter":
            moden["omega"].shape[0],

        "time":
            timewNoise,

        "energy":
            np.sum(
                moden["u"]**2,
                axis=1
            ),

        "alpha":
            twoF
    }



    # =================================
    # Orthogonality calculation
    # =================================

    OCoef = np.ones(
        (K,K)
    )

    OCoefn = np.ones(
        (K,K)
    )


    CrossProduct = np.zeros(
        (K,K)
    )

    CrossProductn = np.zeros(
        (K,K)
    )



    u_center = u.copy()

    un_center = un.copy()


    # subtract mean

    for i in range(K):

        u_center[i,:] -= np.mean(
            u_center[i,:]
        )

        un_center[i,:] -= np.mean(
            un_center[i,:]
        )



    for i in range(K):

        for j in range(i,K):


            OCoef[i,j] = (
                np.sum(
                    u_center[i,:]
                    *
                    u_center[j,:]
                )
                /
                np.sqrt(
                    np.sum(
                        u_center[i,:]**2
                    )
                    *
                    np.sum(
                        u_center[j,:]**2
                    )
                )
            )


            OCoef[j,i] = OCoef[i,j]



            OCoefn[i,j] = (
                np.sum(
                    un_center[i,:]
                    *
                    un_center[j,:]
                )
                /
                np.sqrt(
                    np.sum(
                        un_center[i,:]**2
                    )
                    *
                    np.sum(
                        un_center[j,:]**2
                    )
                )
            )


            OCoefn[j,i] = OCoefn[i,j]



            n = min(i,j)


            CrossProduct[i,j] = (
                np.sum(
                    u_center[i,:]
                    *
                    u_center[j,:]
                )
                /
                np.sum(
                    u_center[n,:]**2
                )
            )


            CrossProduct[j,i] = CrossProduct[i,j]



            CrossProductn[i,j] = (
                np.sum(
                    un_center[i,:]
                    *
                    un_center[j,:]
                )
                /
                np.sum(
                    un_center[n,:]**2
                )
            )


            CrossProductn[j,i] = CrossProductn[i,j]



    OVMDOut["OCoef"] = OCoef
    OVMDOutn["OCoef"] = OCoefn

    OVMDOut["CProd"] = CrossProduct
    OVMDOutn["CProdn"] = CrossProductn


    IMFPlot(
        mode["u"],
        mode["uhat"],
        mode["omega"],
        ErrL2,
        K,
        X,
        time_array,
        twoF,
        "OVMD",
        "",
        "noNoise",
        "OVMD-Without Noise"
    )


    IMFPlot(
        moden["u"],
        moden["uhat"],
        moden["omega"],
        ErrL2n,
        K,
        Xn,
        time_array,
        twoF,
        "OVMD",
        "n",
        "wNoise",
        "OVMD-With Noise"
    )


    return OVMDOut, OVMDOutn