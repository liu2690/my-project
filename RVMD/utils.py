import numpy as np



# =====================================================
# weight functions
# =====================================================


def normalize_weight(input_weight, S, dtype):

    weight = expand_weight(
        input_weight,
        S
    )

    weight = (
        weight /
        np.mean(weight)
    )

    return weight.astype(dtype)



def restore_weight(input_weight, S, dtype):

    return expand_weight(
        input_weight,
        S
    ).astype(dtype)



def expand_weight(input_weight, S):


    weight = np.asarray(
        input_weight
    )


    if (
        np.any(~np.isfinite(weight))
        or
        np.any(weight <= 0)
    ):

        raise ValueError(
            "Weight must be positive finite"
        )


    if weight.size == 1:

        return np.ones(
            S
        ) * weight.item()


    elif weight.size == S:

        return weight.reshape(
            -1
        )


    else:

        raise ValueError(
            "Weight dimension error"
        )





# =====================================================
# initialize spatial modes
# =====================================================


def initial_spatial_modes(
        S,
        K,
        weight,
        dtype,
        isRealInput
):


    phi = np.zeros(
        (S,K),
        dtype=dtype
    )


    for k in range(K):

        index = k % S

        phi[index,k] = (
            1 /
            np.sqrt(weight[index])
        )


    if not isRealInput:

        phi = phi.astype(
            complex
        )


    return phi





# =====================================================
# initialize frequencies
# =====================================================


def initialize_frequencies(
        K,
        nDC,
        initFreqType,
        initFreqMaximum,
        dtype
):


    omega = np.zeros(
        K,
        dtype=dtype
    )


    freeModes = K-nDC


    if freeModes == 0:

        return omega



    if initFreqType == -1:


        omega[nDC:] = (

            np.random.rand(
                freeModes
            )
            *
            initFreqMaximum

        )


    elif initFreqType == 0:

        pass



    elif initFreqType == 1:


        omega[nDC:] = (

            np.arange(
                1,
                freeModes+1
            )
            /
            freeModes
            *
            initFreqMaximum

        )


    return omega.astype(dtype)





# =====================================================
# fallback mode
# =====================================================


def normalized_fallback(
        phi,
        k,
        weight,
        isRealInput
):


    phiNorm = np.sqrt(
        np.sum(
            np.abs(phi)**2
            *
            weight
        )
    )


    if (
        np.isfinite(phiNorm)
        and
        phiNorm > 0
    ):

        phi = phi / phiNorm


    else:


        phi = np.zeros_like(
            phi
        )


        index = k % len(phi)


        phi[index] = (
            1 /
            np.sqrt(weight[index])
        )


    if isRealInput:

        phi = np.real(phi)


    return phi





# =====================================================
# phase normalization
# =====================================================


def canonicalize_phase(
        phi,
        isRealInput
):


    pivot = np.argmax(
        np.abs(phi)
    )


    pivotValue = phi[pivot]



    if isRealInput:


        if np.real(pivotValue)<0:

            phi = -phi



    else:


        magnitude = np.abs(
            pivotValue
        )


        if (
            np.isfinite(magnitude)
            and
            magnitude>0
        ):


            phi = (
                phi
                *
                np.conj(pivotValue)
                /
                magnitude
            )


            phi[pivot] = complex(
                magnitude,
                0
            )


    return phi





# =====================================================
# validation
# =====================================================


def validate_problem(Q,K,Alpha):


    if (
        not isinstance(Q,np.ndarray)
        or
        Q.size==0
        or
        len(Q.shape)!=2
    ):

        raise ValueError(
            "Q must be matrix"
        )


    if not np.all(
        np.isfinite(Q)
    ):

        raise ValueError(
            "Q contains invalid values"
        )


    validate_positive_integer(
        K,
        "K"
    )


    if (
        Alpha <0
        or
        not np.isfinite(Alpha)
    ):

        raise ValueError(
            "Alpha invalid"
        )





def validate_positive_integer(
        value,
        name
):


    if (
        value < 1
        or
        int(value)!=value
    ):

        raise ValueError(
            f"{name} must be positive integer"
        )



    return int(value)





def validate_nonnegative_integer(
        value,
        name
):


    if (
        value <0
        or
        int(value)!=value
    ):

        raise ValueError(
            f"{name} invalid"
        )


    return int(value)





def validate_choice(
        value,
        choices,
        name
):


    value=value.lower()


    if value not in choices:

        raise ValueError(
            f"{name} unsupported"
        )


    return value