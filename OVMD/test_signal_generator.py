from signal_generator import signalGenerator
import matplotlib.pyplot as plt


Signal,time = signalGenerator(
    [1,2,3,4,5,6,7,8,9]
)


plt.plot(
    time,
    Signal["woNoise"]
)

plt.savefig(
    "generated_signal.png",
    dpi=300
)

print(
    Signal.keys()
)