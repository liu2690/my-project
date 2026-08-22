import numpy as np


def OVMD(signal,
         alpha,
         K,
         Tolerance=1e-5,
         Tau=0.0,
         MaximumSteps=1000):
    """
    Orthogonalized Variational Mode Decomposition (OVMD)

    Parameters
    ----------
    signal : array_like
        1D input signal

    alpha : float
        bandwidth constraint parameter

    K : int
        number of modes

    Tolerance : float
        convergence tolerance

    Tau : float
        dual ascent step size

    MaximumSteps : int
        maximum iteration number

    Returns
    -------
    mode : dict
        mode['u']      decomposed modes
        mode['uhat']  spectra
        mode['omega'] center frequencies
    """

    # -----------------------------
    # preprocessing
    # -----------------------------

    signal = np.asarray(signal).flatten()

    mode = {}

    mode["T"] = len(signal)
    mode["K"] = K
    mode["alpha"] = alpha
    mode["Tolerance"] = Tolerance
    mode["MaximumSteps"] = MaximumSteps
    mode["Tau"] = Tau


    tol = Tolerance
    N = MaximumSteps
    tau = Tau


    # length of original signal
    save_T = len(signal)


    # -----------------------------
    # Mirror extension
    # -----------------------------

    T = save_T

    T2 = int(np.floor(T / 2))


    f_mirror = np.zeros(T + 2*T2)


    f_mirror[:T2] = signal[T2-1::-1]

    f_mirror[T2:T+T2] = signal

    f_mirror[T+T2:] = signal[T-1:T2-1:-1]


    f = f_mirror



    # -----------------------------
    # Time domain
    # -----------------------------

    T = len(f)

    t = np.arange(1, T+1) / T



    # -----------------------------
    # Frequency domain
    # -----------------------------

    freqs = t - 0.5 - 1/T



    # bandwidth parameter
    Alpha = alpha * np.ones(K)

    mode["alpha"] = alpha



    # -----------------------------
    # Fourier transform
    # -----------------------------

    f_hat = np.fft.fftshift(
        np.fft.fft(f)
    )


    f_hat_plus = f_hat.copy()


    # MATLAB:
    # f_hat_plus(1:T/2)=0

    f_hat_plus[:T//2] = 0



    # -----------------------------
    # initialize variables
    # -----------------------------


    u_hat_plus = (
        np.ones(
            (2, len(freqs), K),
            dtype=complex
        )
        * np.finfo(float).eps
    )


    max_uhat = (
        np.ones(K)
        * np.finfo(float).eps
    )


    norm_uhat = (
        np.ones(K)
        * np.finfo(float).eps
    )



    # frequency evolution

    omega_plus = np.zeros(
        (N, K)
    )


    omegaFilter = np.zeros(
        (K, len(freqs))
    )


    # initial frequency
    omega_plus[0, :] = 0



    # dual variables

    lambda_hat = np.zeros(
        (2, len(freqs)),
        dtype=complex
    )



    # other initialization

    uDiff = 1

    n = 1

    m = 0


    sum_uk = 0



    # -----------------------------
    # initialize norm and filters
    # -----------------------------

    for k in range(K):

        norm_uhat[k] = np.linalg.norm(
            u_hat_plus[1,:,k]
        )


        max_uhat[k] = np.max(
            np.abs(
                u_hat_plus[1,:,k]
            )
        )


        omegaFilter[k,:] = (
            1 /
            (
                1
                +
                Alpha[k]
                *
                (
                    np.abs(freqs)
                    -
                    omega_plus[0,k]
                )**2
            )
        )
    # ==========================================
    # Main iteration loop
    # ==========================================

    while uDiff > tol and m < N:


        # update each mode

        for k in range(K):

            receiveCont = 0
            sendCont = 0

            Osubs = 0
            Oadd = 0



            # ----------------------------------
            # orthogonalization terms
            # ----------------------------------

            projkk = (
                np.abs(
                    u_hat_plus[1,:,k]
                    *
                    np.conj(
                        u_hat_plus[1,:,k]
                    )
                )
                /
                (
                    norm_uhat[k]
                    *
                    norm_uhat[k]
                )
            )


            projkk = np.sqrt(projkk)



            for i in range(K):

                if i != k:


                    projik = (
                        np.abs(
                            u_hat_plus[1,:,i]
                            *
                            np.conj(
                                u_hat_plus[1,:,k]
                            )
                        )
                        /
                        (
                            norm_uhat[i]
                            *
                            norm_uhat[k]
                        )
                    )


                    projik = np.sqrt(projik)



                    if max_uhat[k] > max_uhat[i]:


                        Oadd = (
                            Oadd
                            +
                            projik
                            *
                            u_hat_plus[1,:,i]
                        )


                        Osubs = (
                            Osubs
                            +
                            projik**2
                            *
                            (
                                np.abs(
                                    u_hat_plus[1,:,i]
                                )**2
                            )
                        )


                        receiveCont = (
                            receiveCont
                            +
                            (
                                projik
                                *
                                projkk
                            )
                            *
                            u_hat_plus[1,:,i]
                        )


                    else:


                        sendCont = (
                            sendCont
                            +
                            omegaFilter[i,:]
                            *
                            projik**2
                        )



            receiveCont = (
                omegaFilter[k,:]
                *
                receiveCont
            )



            Okk = (
                projkk
                *
                u_hat_plus[1,:,k]
            )



            Osubs = (
                omegaFilter[k,:]**2
                *
                (
                    np.abs(
                        Oadd + Okk
                    )**2
                    -
                    (
                        Osubs
                        +
                        projkk**2
                        *
                        np.abs(
                            u_hat_plus[1,:,k]
                        )**2
                    )
                )
            )



            # ----------------------------------
            # residual excluding mode k
            # ----------------------------------

            kminus = k-1

            if k == 0:
                kminus = K-1



            sum_uk = (
                u_hat_plus[1,:,kminus]
                +
                sum_uk
                -
                u_hat_plus[0,:,k]
            )



            # ----------------------------------
            # update spectrum
            # ----------------------------------

            u_hat_plus[1,:,k] = (

                f_hat_plus
                -
                sum_uk
                -
                lambda_hat[0,:]/2
                +
                receiveCont

            ) / (

                1
                +
                Alpha[k]
                *
                (
                    freqs
                    -
                    omega_plus[m,k]
                )**2
                +
                sendCont

            )



            norm_uhat[k] = np.linalg.norm(
                u_hat_plus[1,:,k]
            )


            max_uhat[k] = np.max(
                np.abs(
                    u_hat_plus[1,:,k]
                )
            )



            # ----------------------------------
            # update center frequency
            # ----------------------------------

            positive = slice(
                T//2,
                T
            )


            numerator = np.sum(
                freqs[positive]
                *
                (
                    np.abs(
                        u_hat_plus[1,positive,k]
                    )**2
                    +
                    Osubs[positive]
                )
            )


            denominator = np.sum(
                np.abs(
                    u_hat_plus[1,positive,k]
                )**2
                +
                Osubs[positive]
            )


            omega_plus[m+1,k] = (
                numerator
                /
                denominator
            )



            omegaFilter[k,:] = (
                1
                /
                (
                    1
                    +
                    Alpha[k]
                    *
                    (
                        np.abs(freqs)
                        -
                        omega_plus[m+1,k]
                    )**2
                )
            )
        # ======================================
        # convergence calculation
        # ======================================

        uDiff = np.finfo(float).eps


        for i in range(K):

            Alpha[i] = (
                alpha /
                max(
                    omega_plus[m+1,i],
                    np.finfo(float).eps
                )
            )


            omegaFilter[i,:] = (
                1 /
                (
                    1
                    +
                    Alpha[i]
                    *
                    (
                        np.abs(freqs)
                        -
                        omega_plus[m+1,i]
                    )**2
                )
            )


            uDiff += (
                np.linalg.norm(
                    u_hat_plus[1,:,i]
                    -
                    u_hat_plus[0,:,i]
                )**2
                /
                np.linalg.norm(
                    u_hat_plus[1,:,i]
                )**2
            )



        # ======================================
        # Dual ascent
        # ======================================

        lambda_hat[1,:] = (
            lambda_hat[0,:]
            +
            tau
            *
            (
                np.sum(
                    u_hat_plus[1,:,:],
                    axis=1
                )
                -
                f_hat_plus
            )
        )


        lambda_hat[0,:] = lambda_hat[1,:]



        # iteration counter

        m += 1


        # save previous spectrum

        u_hat_plus[0,:,:] = (
            u_hat_plus[1,:,:]
        )


        uDiff = abs(uDiff)



    # ======================================
    # Post processing
    # ======================================


    # remove unused iterations

    N = min(N, m)


    omega = omega_plus[:N,:]



    # ======================================
    # Signal reconstruction
    # ======================================


    u_hat = np.zeros(
        (T, K),
        dtype=complex
    )


    positive = slice(
        T//2,
        T
    )


    for k in range(K):

        u_hat[T//2:T,k] = (
            u_hat_plus[1,positive,k]
        )


        u_hat[1:T//2+1,k] = (
            np.conj(
                u_hat_plus[1,positive,k][::-1]
            )
        )


        u_hat[0,k] = np.conj(
            u_hat[-1,k]
        )



    # ======================================
    # Sort modes by energy
    # ======================================


    energy = np.sum(
        np.abs(u_hat)**2,
        axis=0
    )


    I = np.argsort(
        energy
    )[::-1]


    omegaF = omega[:,I]


    u_hatF = u_hat[:,I]



    # ======================================
    # inverse FFT
    # ======================================


    u = np.zeros(
        (K,T)
    )


    for k in range(K):

        u[k,:] = np.real(
            np.fft.ifft(
                np.fft.ifftshift(
                    u_hatF[:,k]
                )
            )
        )



    # remove mirror part

    start = T//4
    end = 3*T//4


    u = u[:,start:end]



    # ======================================
    # recompute spectra
    # ======================================


    u_hatF = np.zeros(
        (
            u.shape[1],
            K
        ),
        dtype=complex
    )


    for k in range(K):

        u_hatF[:,k] = (
            np.fft.fftshift(
                np.fft.fft(
                    u[k,:]
                )
            )
        )



    # ======================================
    # output
    # ======================================


    mode["omega"] = omegaF

    mode["uhat"] = u_hatF

    mode["u"] = u


    return mode