import numpy as np


def constructSignal(f, time):
    """
    Equivalent to MATLAB constructSignal()
    """

    fs = 1 / (time[1] - time[0])

    freq = np.zeros(len(time))

    offsetTime = np.zeros(len(time))


    for i, t in enumerate(time):

        fIns = (
            f["fstart"]
            +
            (f["fend"] - f["fstart"])
            /
            (f["tEnd"] - f["tStart"])
            *
            t
        )


        n = np.ceil(
            (
                t - f["tStart"]
            )
            /
            (
                f["tDelay"]
                +
                f["nPeriod"]/max(fIns,1e-12)
            )
        )


        if (
            t >= f["tStart"]
            and
            t <= f["tEnd"]
            and
            (
                t-f["tStart"]
            )
            <=
            (
                (n-1)*f["tDelay"]
                +
                n*f["nPeriod"]/max(fIns,1e-12)
            )
        ):

            freq[i] = fIns

            offsetTime[i] = (
                f["tStart"]
                +
                (n-1)
                *
                (
                    f["tDelay"]
                    +
                    f["nPeriod"]/max(fIns,1e-12)
                )
            )


    X = (
        f["A"]
        *
        np.exp(
            1j*
            (
                2*np.pi
                *
                freq
                *
                (time-offsetTime)
                +
                f["phase"]
            )
        )
    )


    signal = np.real(X)

    phase = np.imag(X)


    return signal, phase, freq



def signalGenerator(
        InSignal,
        tStart=0,
        tEnd=20,
        fs=100,
        SNR=10,
        loadNoise=0
):

    """
    Python version of signalGenerator.m

    Returns
    -------
    Signal : dict
    time : ndarray
    """



    time = np.arange(
        tStart,
        tEnd+1/fs,
        1/fs
    )



    # -----------------------
    # signal definitions
    # -----------------------

    f = []


    definitions = [

        (0.1,0.1,0.75,0,0,0,20),

        (1,1,0.65,0,0,0,20),

        (7,7,1,0,0,0,20),

        (7,7,0.5,np.pi/2,0,0,20),

        (20,20,1,0,0,0,20),

        (22,22,0.3,0,0,0,20),

        (40,40,0.35,0,0,0,20),

        (16,16,0.85,-np.pi/2,1,3.125,16),

        (4,4,1,-np.pi/2,1,3.125,4)

    ]



    for item in definitions:

        f.append({

            "fstart":item[0],

            "fend":item[1],

            "A":item[2],

            "phase":item[3],

            "tStart":item[4],

            "tDelay":item[5],

            "nPeriod":item[6],

            "tEnd":tEnd
        })



    Signal={}



    X=[]
    phase=[]
    freq=[]
    energy=[]


    # construct components

    for index in InSignal:

        component=f[index-1]


        x,p,fr=constructSignal(
            component,
            time
        )


        X.append(x)
        phase.append(p)
        freq.append(fr)

        energy.append(
            np.sum(x**2)
        )



    X=np.array(X)

    phase=np.array(phase)

    freq=np.array(freq)

    energy=np.array(energy)



    # sort energy

    I=np.argsort(
        energy
    )[::-1]


    X=X[I]

    phase=phase[I]

    freq=freq[I]

    energy=energy[I]



    FirstEnergy=energy[0]


    Signal["X"]=X

    Signal["phase"]=phase

    Signal["freq"]=freq

    Signal["energy"]=energy/FirstEnergy


    Signal["FirstEnergy"]=FirstEnergy


    Signal["woNoise"]=np.sum(
        X,
        axis=0
    )


    Signal["SNR"]=SNR



    # -----------------------
    # noise
    # -----------------------

    noise = (
        np.random.rand(len(time))-0.5
    )*2



    noise_amp = np.sqrt(
        np.sum(Signal["woNoise"]**2)
        /
        (
            np.sum(noise**2)
            *
            10**(SNR/10)
        )
    )


    noise = noise*noise_amp



    Signal["Noise"]=noise


    Signal["wNoise"] = (
        Signal["woNoise"]
        +
        noise
    )


    Signal["time"]=time



    # FFT

    Signal["FFT"]={}

    Signal["FFT"]["woNoise"]=(
        np.fft.fftshift(
            np.fft.fft(
                Signal["woNoise"]
            )
        )
    )

    Signal["FFT"]["wNoise"]=(
        np.fft.fftshift(
            np.fft.fft(
                Signal["wNoise"]
            )
        )
    )



    return Signal,time