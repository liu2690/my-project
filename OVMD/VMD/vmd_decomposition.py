import time
import numpy as np

from .vmd import VMD
from imf_plot import IMFPlot

def VMDDecomposition(
        X,
        Xn,
        time_array,
        twoF,
        tau,
        K,
        DC,
        init,
        tol,
        N
):

    """
    Wrapper for VMD

    Equivalent to MATLAB VMDDecomposition.m
    """



    evenSize = (
        len(X)//2
    )*2



    # ============================
    # VMD without noise
    # ============================


    start = time.time()


    u, uhat, omega = VMD(
        X[:evenSize],
        twoF,
        tau,
        K,
        DC,
        init,
        tol,
        N
    )


    timewoNoise = (
        time.time()
        -
        start
    )



    # ============================
    # VMD with noise
    # ============================


    start = time.time()


    un, unhat, omegan = VMD(
        Xn[:evenSize],
        twoF,
        tau,
        K,
        DC,
        init,
        tol,
        N
    )


    timewNoise = (
        time.time()
        -
        start
    )



    # ============================
    # Reconstruction error
    # ============================


    ErrL2 = (

        np.linalg.norm(
            np.sum(u,axis=0)
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
            np.sum(un,axis=0)
            -
            Xn[:evenSize]
        )

        /

        np.linalg.norm(
            Xn[:evenSize]
        )

    )

    # ============================
    # Output
    # ============================


    VMDOut = {

        "u":u,

        "uhat":uhat,

        "omega":omega,

        "ErrL2":ErrL2,

        "iter":
            omega.shape[0],

        "time":
            timewoNoise,

        "energy":
            np.sum(
                u**2,
                axis=1
            )
    }



    VMDOutn = {

        "u":un,

        "uhat":unhat,

        "omega":omegan,

        "ErrL2":ErrL2n,

        "iter":
            omegan.shape[0],

        "time":
            timewNoise,

        "energy":
            np.sum(
                un**2,
                axis=1
            )
    }

    # ============================
    # Orthogonality
    # ============================


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

    VMDOut["OCoef"] = OCoef
    VMDOutn["OCoef"] = OCoefn


    VMDOut["CProd"] = CrossProduct
    VMDOutn["CProdn"] = CrossProductn



    return VMDOut, VMDOutn