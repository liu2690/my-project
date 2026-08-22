import numpy as np


def VMD(
        signal,
        alpha,
        tau,
        K,
        DC,
        init,
        tol,
        N
):
    """
    Variational Mode Decomposition

    Python version of VMD_mod.m

    Parameters
    ----------
    signal : ndarray
        input 1D signal

    alpha : float
        bandwidth constraint

    tau : float
        dual ascent step

    K : int
        number of modes

    DC : int
        keep first mode at zero frequency

    init :
        0 -> all omega start at 0
        1 -> uniformly distributed
        2 -> random initialization

    tol : float
        convergence tolerance

    N : int
        maximum iteration


    Returns
    -------
    u : ndarray
        decomposed modes

    u_hat : ndarray
        mode spectrum

    omega : ndarray
        center frequencies
    """



    signal = np.asarray(signal).flatten()



    # --------------------------------
    # Period and sampling frequency
    # --------------------------------

    save_T = len(signal)

    fs = 1 / save_T



    # --------------------------------
    # Mirror extension
    # --------------------------------


    T = save_T


    T2 = T // 2


    f_mirror = np.zeros(
        2*T
    )


    f_mirror[:T2] = (
        signal[T2-1::-1]
    )


    f_mirror[T2:T+T2] = signal


    f_mirror[T+T2:] = (
        signal[T-1:T2-1:-1]
    )


    f = f_mirror



    # --------------------------------
    # Time domain
    # --------------------------------


    T = len(f)

    t = (
        np.arange(1,T+1)
        /
        T
    )



    # --------------------------------
    # Frequency domain
    # --------------------------------


    freqs = (
        t
        -
        0.5
        -
        1/T
    )



    # bandwidth parameter

    Alpha = (
        alpha
        *
        np.ones(K)
    )



    # --------------------------------
    # Fourier transform
    # --------------------------------


    f_hat = np.fft.fftshift(
        np.fft.fft(f)
    )


    f_hat_plus = f_hat.copy()


    f_hat_plus[:T//2] = 0



    # --------------------------------
    # Initialization
    # --------------------------------


    u_hat_plus = np.zeros(
        (
            2,
            len(freqs),
            K
        ),
        dtype=complex
    )


    omega_plus = np.zeros(
        (
            N,
            K
        )
    )


    # initialize omega

    if init == 1:

        for i in range(K):

            omega_plus[0,i] = (
                0.5/K*i
            )


    elif init == 2:

        omega_plus[0,:] = np.sort(
            np.exp(
                np.log(fs)
                +
                (
                    np.log(0.5)
                    -
                    np.log(fs)
                )
                *
                np.random.rand(K)
            )
        )


    else:

        omega_plus[0,:] = 0



    # DC mode

    if DC:

        omega_plus[0,0] = 0



    # dual variable

    lambda_hat = np.zeros(
        (
            2,
            len(freqs)
        ),
        dtype=complex
    )



    # other initialization


    uDiff = tol + np.finfo(float).eps


    n = 0

    m = 0


    sum_uk = 0

        # ======================================
    # Main loop
    # ======================================


    while (
        uDiff > tol
        and
        m < N
    ):


        # ----------------------------------
        # update first mode
        # ----------------------------------

        k = 0

        n = 0


        sum_uk = (
            u_hat_plus[0,:,K-1]
            +
            sum_uk
            -
            u_hat_plus[0,:,0]
        )



        u_hat_plus[1,:,k] = (

            f_hat_plus
            -
            sum_uk
            -
            lambda_hat[n,:]/2

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

        )



        # update omega of first mode

        if not DC:


            numerator = np.sum(

                freqs[T//2:]
                *
                np.abs(
                    u_hat_plus[
                        1,
                        T//2:,
                        k
                    ]
                )**2

            )


            denominator = np.sum(

                np.abs(
                    u_hat_plus[
                        1,
                        T//2:,
                        k
                    ]
                )**2

            )


            omega_plus[m+1,k] = (
                numerator
                /
                denominator
            )



        # ----------------------------------
        # update remaining modes
        # ----------------------------------


        for k in range(1,K):


            sum_uk = (

                u_hat_plus[
                    1,:,k-1
                ]

                +
                sum_uk

                -
                u_hat_plus[
                    n,:,k
                ]

            )



            u_hat_plus[1,:,k] = (

                f_hat_plus
                -
                sum_uk
                -
                lambda_hat[n,:]/2

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

            )



            numerator = np.sum(

                freqs[T//2:]

                *
                np.abs(
                    u_hat_plus[
                        1,
                        T//2:,
                        k
                    ]
                )**2

            )


            denominator = np.sum(

                np.abs(
                    u_hat_plus[
                        1,
                        T//2:,
                        k
                    ]
                )**2

            )


            omega_plus[m+1,k] = (

                numerator
                /
                denominator

            )



        # ----------------------------------
        # update Alpha
        # ----------------------------------


        for i in range(K):

            Alpha[i] = (

                alpha
                /
                max(
                    omega_plus[m+1,i],
                    np.finfo(float).eps
                )

            )



        # ----------------------------------
        # Dual ascent
        # ----------------------------------


        lambda_hat[1,:] = (

            lambda_hat[n,:]

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



        # ----------------------------------
        # iteration counter
        # ----------------------------------

        m += 1



        # ----------------------------------
        # convergence
        # ----------------------------------

        uDiff = np.finfo(float).eps


        for i in range(K):

            uDiff += (
                np.linalg.norm(
                    u_hat_plus[1,:,i]
                    -
                    u_hat_plus[0,:,i]
                )**2
                /
                (
                    np.linalg.norm(
                        u_hat_plus[0,:,i]
                    )**2
                    +
                    np.finfo(float).eps
                )
            )



        u_hat_plus[0,:,:] = (
            u_hat_plus[1,:,:]
        )


        uDiff = abs(uDiff)

            # ======================================
    # Postprocessing
    # ======================================


    # discard unused iterations

    N = min(
        N,
        m
    )


    omega = omega_plus[:N,:]



    # ======================================
    # reconstruct spectrum
    # ======================================


    u_hat = np.zeros(
        (
            T,
            K
        ),
        dtype=complex
    )



    # positive frequencies

    positive = slice(
        T//2,
        T
    )



    for k in range(K):

        pos = u_hat_plus[
            1,
            positive,
            k
        ]


        # positive frequency
        u_hat[T//2:T//2+len(pos), k] = pos


        # negative frequency
        neg = np.conj(pos[::-1])


        u_hat[1:1+len(neg), k] = neg


        u_hat[0,k] = np.conj(
            u_hat[-1,k]
        )


        u_hat[0,k] = (
            np.conj(
                u_hat[-1,k]
            )
        )



    # ======================================
    # sort modes by energy
    # ======================================


    energy = np.sum(
        np.abs(u_hat)**2,
        axis=0
    )


    I = np.argsort(
        energy
    )[::-1]



    omega = omega[:,I]


    u_hat = u_hat[:,I]



    # ======================================
    # inverse FFT
    # ======================================


    u = np.zeros(
        (
            K,
            T
        )
    )



    for k in range(K):

        u[k,:] = np.real(

            np.fft.ifft(

                np.fft.ifftshift(
                    u_hat[:,k]
                )

            )

        )



    # remove mirrored part

    start = T//4

    end = 3*T//4


    u = u[:,start:end]



    # ======================================
    # recompute spectrum
    # ======================================


    u_hat = np.zeros(
        (
            u.shape[1],
            K
        ),
        dtype=complex
    )



    for k in range(K):

        u_hat[:,k] = (

            np.fft.fftshift(

                np.fft.fft(
                    u[k,:]
                )

            )

        )



    return u, u_hat, omega