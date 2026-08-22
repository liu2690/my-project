import numpy as np
from utils import *


def rvmd(
        Q,
        K,
        Alpha,
        Weight=1,
        Tolerance=5e-3,
        MaximumSteps=500,
        InitFreqType=1,
        InitFreqMaximum=0.5,
        Device="cpu",
        FPPrecision="single",
        nDC=0,
        Display="off",
        Restart=None
):
    """
    Reduced-order Variational Mode Decomposition (RVMD)

    Parameters
    ----------
    Q : ndarray
        S x T data matrix

    
    K : int
        number of modes

    Alpha : float
        bandwidth penalty

    Returns
    -------
    mode : dict
        mode["phi"]
        mode["c"]
        mode["omega"]

    info : dict

    restart : dict
    """

    # ==========================
    # Restart mode
    # ==========================

    if Restart is not None:

        state = Restart

        Q = state["Q"]
        K = state["K"]
        Alpha = state["alpha"]

        Weight = state["weight"]

        S = state["S"]
        T = state["T"]

        nDC = state["nDC"]

        phi = state["phi_n"]

        coefficientSpectrum = state["c_spec_n"]

        completedSteps = state["Iteration"]["steps"]

        frequencyHistoryStored = (
            state["Iteration"]["omega"]
        )

        differenceStored = (
            state["Iteration"]["difference"]
        )

        isRealInput = state["isRealInput"]

    else:

        validate_problem(Q, K, Alpha)

        completedSteps = 0

        isRealInput = np.isrealobj(Q)



    # ==========================
    # precision
    # ==========================

    if FPPrecision == "single":

        dtype = np.float32

    else:

        dtype = np.float64



    Q = np.asarray(Q, dtype=dtype)

    Alpha = dtype(Alpha)



    # ==========================
    # dimension
    # ==========================

    S, T = Q.shape



    # ==========================
    # weight
    # ==========================

    weight = normalize_weight(
        Weight,
        S,
        dtype
    )



    # ==========================
    # Extended signal
    #
    # MATLAB:
    #
    # extendedLength = 2*T
    #
    # QExtended(:,1:half)=...
    #
    # ==========================


    extendedLength = 2*T

    half = int(np.ceil(T/2))


    QExtended = np.zeros(
        (S, extendedLength),
        dtype=dtype
    )


    # MATLAB:
    #
    # Q(:,half:-1:1)

    QExtended[:,0:half] = (
        Q[:,half-1::-1]
    )


    # MATLAB:
    #
    # Q(:,1:T)

    QExtended[:,half:half+T] = Q


    # MATLAB:
    #
    # Q(:,T:-1:half+1)

    QExtended[:,half+T:] = (
        Q[:,T-1:half-1:-1]
    )



    # ==========================
    # FFT
    # ==========================

    QSpectrum = np.fft.fft(
        QExtended,
        axis=1
    )


    del QExtended



    if isRealInput:


        spectrumLength = T+1


        QSpectrum = (
            QSpectrum[:,:spectrumLength]
        )


        frequency = (
            np.arange(T+1)
            /
            extendedLength
        ).astype(dtype)



        frequencyWeight = np.ones(
            spectrumLength,
            dtype=dtype
        )


        frequencyWeight[0] = 0.5
        frequencyWeight[-1] = 0.5



    else:


        spectrumLength = extendedLength


        QSpectrum = (
            np.fft.fftshift(
                QSpectrum,
                axes=1
            )
        )


        frequency = (
            np.arange(-T,T)
            /
            extendedLength
        ).astype(dtype)


        frequencyWeight = np.ones(
            spectrumLength,
            dtype=dtype
        )



    absoluteFrequency = np.abs(
        frequency
    )



    # ==========================
    # initialize modes
    # ==========================

    if Restart is None:


        phi = initial_spatial_modes(
            S,
            K,
            weight,
            dtype,
            isRealInput
        )


        if np.linalg.norm(Q)==0:

            coefficientSpectrum = np.zeros(
                (spectrumLength,K),
                dtype=np.complex64 if dtype==np.float32 else np.complex128
            )

        else:

            complex_dtype = (
                 np.complex64
                 if dtype == np.float32
                 else np.complex128
            )   
                        
            coefficientSpectrum = (
                np.ones(
                    (spectrumLength,K),
                    dtype=complex_dtype
                )
                *
                np.finfo(dtype).eps
            )


    else:

        coefficientSpectrum = np.asarray(
            coefficientSpectrum,
            dtype=dtype
        )



    # ==========================
    # frequency initialization
    # ==========================

    frequencyHistory = np.zeros(
        (
            K,
            MaximumSteps+1
        ),
        dtype=dtype
    )


    differenceHistory = np.zeros(
        MaximumSteps,
        dtype=dtype
    )



    if Restart is None:


        frequencyHistory[:,0] = (
            initialize_frequencies(
                K,
                nDC,
                InitFreqType,
                InitFreqMaximum,
                dtype
            )
        )



    else:


        frequencyHistory[
            :,
            :completedSteps+1
        ] = frequencyHistoryStored


    # ==========================
    # residual initialization
    # ==========================


    residual = (
        QSpectrum
        -
        phi @ coefficientSpectrum.T
    )


    dataSpectrumNorm = np.linalg.norm(
        QSpectrum
    )


    scaleFloor = (
        np.finfo(dtype).eps
        *
        max(dataSpectrumNorm,1)
    )


    iteration = completedSteps + 1


    if completedSteps == 0:

        difference = np.inf

    else:

        difference = differenceHistory[
            completedSteps-1
        ]



    # ==========================
    # Main RVMD iteration
    # ==========================


    while (
        iteration <= MaximumSteps
        and
        difference > Tolerance
    ):


        differenceAccumulator = 0.0



        for k in range(K):


            # --------------------------
            # remove old mode
            #
            # MATLAB:
            #
            # oldMode =
            # phi(:,k)*coefficientSpectrum(:,k).'
            # --------------------------

            oldMode = (
                np.outer(
                    phi[:,k],
                    coefficientSpectrum[:,k]
                )
            )


            residual = residual + oldMode



            # --------------------------
            # projection
            #
            # MATLAB:
            #
            # projection =
            # residual *
            # (conj(c).*frequencyWeight)
            #
            # --------------------------


            projection = (
                residual
                @
                (
                    np.conj(
                        coefficientSpectrum[:,k]
                    )
                    *
                    frequencyWeight
                )
            )


            if isRealInput:

                projection = np.real(
                    projection
                )



            # --------------------------
            # normalize spatial mode
            #
            # MATLAB:
            #
            # sqrt(sum(abs(projection).^2 .* weight))
            #
            # --------------------------


            projectionNorm = np.sqrt(
                np.sum(
                    np.abs(projection)**2
                    *
                    weight
                )
            )



            if (
                np.isfinite(projectionNorm)
                and
                projectionNorm > 0
            ):


                phi[:,k] = (
                    projection
                    /
                    projectionNorm
                )


            else:


                phi[:,k] = normalized_fallback(
                    phi[:,k],
                    k,
                    weight,
                    isRealInput
                )



            # --------------------------
            # phase normalization
            # --------------------------

            phi[:,k] = canonicalize_phase(
                phi[:,k],
                isRealInput
            )



            # --------------------------
            # update temporal coefficient
            #
            # MATLAB:
            #
            # denominator =
            # 1+2*Alpha*(absoluteFrequency-frequency)^2
            #
            # --------------------------


            denominator = (
                1
                +
                2
                *
                Alpha
                *
                (
                    absoluteFrequency
                    -
                    frequencyHistory[k,iteration-1]
                )**2
            )



            coefficientSpectrum[:,k] = (
                (
                    residual.T
                    @
                    (
                        np.conj(phi[:,k])
                        *
                        weight
                    )
                )
                /
                denominator
            )



            # --------------------------
            # frequency update
            # --------------------------


            coefficientEnergy = np.sum(
                frequencyWeight
                *
                np.abs(
                    coefficientSpectrum[:,k]
                )**2
            )



            if k < nDC:


                frequencyHistory[
                    k,
                    iteration
                ] = 0



            elif (
                np.isfinite(coefficientEnergy)
                and
                coefficientEnergy > 0
            ):


                frequencyHistory[
                    k,
                    iteration
                ] = (

                    np.sum(
                        frequencyWeight
                        *
                        absoluteFrequency
                        *
                        np.abs(
                            coefficientSpectrum[:,k]
                        )**2
                    )

                    /
                    coefficientEnergy

                )


            else:


                frequencyHistory[
                    k,
                    iteration
                ] = (
                    frequencyHistory[
                        k,
                        iteration-1
                    ]
                )



            # --------------------------
            # update residual
            # --------------------------


            newMode = (
                np.outer(
                    phi[:,k],
                    coefficientSpectrum[:,k]
                )
            )


            residual = residual - newMode



            # --------------------------
            # convergence
            # --------------------------


            oldNorm = np.linalg.norm(
                oldMode,
                'fro'
            )


            changeNorm = np.linalg.norm(
                newMode-oldMode,
                'fro'
            )


            relativeChange = (
                changeNorm
                /
                max(oldNorm,scaleFloor)
            )


            differenceAccumulator += (
                relativeChange
            )



        difference = float(
            differenceAccumulator
        )


        differenceHistory[
            iteration-1
        ] = difference



        if Display == "iter":

            print(
                f"iteration step: {iteration}"
                f" difference: {difference:.8g}"
            )



        iteration += 1



    completedSteps = iteration-1

    # ==========================
    # reconstruct coefficient
    # ==========================


    if isRealInput:


        fullSpectrum = np.zeros(
            (
                extendedLength,
                K
            ),
            dtype=coefficientSpectrum.dtype
        )


        fullSpectrum[
            :spectrumLength,
            :
        ] = coefficientSpectrum



        # MATLAB:
        #
        # conj(coefficientSpectrum(
        # spectrumLength-1:-1:2,:))


        fullSpectrum[
            spectrumLength:,
            :
        ] = np.conj(
            coefficientSpectrum[
                spectrumLength-2:0:-1,
                :
            ]
        )


        coefficient = np.real(
            np.fft.ifft(
                fullSpectrum,
                axis=0
            )
        )


        phi = np.real(phi)



    else:


        coefficient = np.fft.ifft(
            np.fft.fftshift(
                coefficientSpectrum,
                axes=0
            ),
            axis=0
        )



    # MATLAB:
    #
    # coefficient =
    # coefficient((half+1):(half+T),:)


    coefficient = coefficient[
        half:half+T,
        :
    ]



    # ==========================
    # sort frequencies
    # ==========================


    finalFrequency = (
        frequencyHistory[
            :,
            completedSteps
        ]
    )


    order = np.argsort(
        finalFrequency
    )


    frequencySorted = (
        finalFrequency[order]
    )



    # ==========================
    # output mode
    # ==========================


    mode = {

        "phi":
            phi[:,order],

        "c":
            coefficient[:,order],

        "omega":
            frequencySorted,

        "energy":
            np.sum(
                np.abs(
                    coefficient[:,order]
                )**2,
                axis=0
            )
    }



    # ==========================
    # information
    # ==========================


    info = {

        "S":S,

        "T":T,

        "K":K,

        "alpha":Alpha,

        "weight":weight,

        "Tolerance":Tolerance,

        "MaximumSteps":MaximumSteps,

        "InitFreqType":InitFreqType,

        "InitFreqMaximum":InitFreqMaximum,

        "Device":Device,

        "FPPrecision":FPPrecision,

        "nDC":nDC,

        "isRealInput":isRealInput,


        "Iteration":
        {

            "steps":
                completedSteps,


            "omega":
                frequencyHistory[
                    :,
                    :completedSteps+1
                ],


            "difference":
                differenceHistory[
                    :completedSteps
                ],


            "converged":
                (
                    completedSteps>0
                    and
                    differenceHistory[
                        completedSteps-1
                    ]
                    <=
                    Tolerance
                )
        }

    }



    # ==========================
    # restart state
    # ==========================


    restart = {


        "version":3,


        "Q":Q,


        "S":S,


        "T":T,


        "K":K,


        "alpha":Alpha,


        "weight":weight,


        "Tolerance":Tolerance,


        "MaximumSteps":MaximumSteps,


        "InitFreqType":InitFreqType,


        "InitFreqMaximum":InitFreqMaximum,


        "Device":Device,


        "FPPrecision":FPPrecision,


        "nDC":nDC,


        "isRealInput":isRealInput,


        "Display":Display,


        "Iteration":
            info["Iteration"],


        "c_spec_n":
            coefficientSpectrum,


        "phi_n":
            phi,


        "residual_n":
            residual

    }



    return mode, info, restart